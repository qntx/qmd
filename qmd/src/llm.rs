//! LLM module for QMD - embedding and vector search support.
//!
//! This module provides local LLM inference for:
//! - Document embeddings using GGUF models
//! - Vector similarity search
//! - Query expansion
//! - Reranking
//! - Automatic model download from `HuggingFace`

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use regex::Regex;

use crate::config;

/// Default embedding model (embeddinggemma-300M)
pub const DEFAULT_EMBED_MODEL: &str = "embeddinggemma-300M-Q8_0.gguf";

/// Default rerank model
pub const DEFAULT_RERANK_MODEL: &str = "qwen3-reranker-0.6b-q8_0.gguf";

/// Default generation model for query expansion
pub const DEFAULT_GENERATE_MODEL: &str = "qmd-query-expansion-1.7B-q4_k_m.gguf";

/// HuggingFace model URI for default embedding model.
pub const DEFAULT_EMBED_MODEL_URI: &str =
    "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";

/// HuggingFace model URI for default rerank model.
pub const DEFAULT_RERANK_MODEL_URI: &str =
    "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";

/// HuggingFace model URI for default query expansion model.
pub const DEFAULT_GENERATE_MODEL_URI: &str =
    "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf";

/// Chunk size in tokens for document splitting
pub const CHUNK_SIZE_TOKENS: usize = 800;

/// Overlap between chunks in tokens (15%)
pub const CHUNK_OVERLAP_TOKENS: usize = 120;

/// Chunk size in characters for document splitting (fallback)
pub const CHUNK_SIZE_CHARS: usize = 3200;

/// Overlap between chunks in characters
pub const CHUNK_OVERLAP_CHARS: usize = 480;

/// Embedding engine for generating document vectors.
#[derive(Debug)]
pub struct EmbeddingEngine {
    /// The loaded LLM model
    model: Arc<LlamaModel>,
    /// Model dimensions (set after first embedding)
    dimensions: Option<usize>,
}

/// Result of an embedding operation.
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// The embedding vector
    pub embedding: Vec<f32>,
    /// Model name used
    pub model: String,
}

/// A document chunk with position information.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// The text content of the chunk
    pub text: String,
    /// Character position in original document
    pub pos: usize,
}

impl EmbeddingEngine {
    /// Create a new embedding engine with the specified model.
    ///
    /// # Arguments
    /// * `model_path` - Path to the GGUF model file
    ///
    /// # Errors
    /// Returns an error if the model cannot be loaded.
    pub fn new(model_path: &Path) -> Result<Self> {
        let backend = LlamaBackend::init()?;

        let model_params = LlamaModelParams::default();

        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .with_context(|| format!("Failed to load model from {}", model_path.display()))?;

        Ok(Self {
            model: Arc::new(model),
            dimensions: None,
        })
    }

    /// Load the default embedding model from the cache directory.
    ///
    /// # Errors
    /// Returns an error if the model is not found or cannot be loaded.
    pub fn load_default() -> Result<Self> {
        let model_path = get_model_path(DEFAULT_EMBED_MODEL)?;
        Self::new(&model_path)
    }

    /// Generate an embedding for the given text.
    ///
    /// # Arguments
    /// * `text` - The text to embed
    ///
    /// # Errors
    /// Returns an error if embedding generation fails.
    pub fn embed(&mut self, text: &str) -> Result<EmbeddingResult> {
        let formatted = format_doc_for_embedding(text, None);
        self.embed_raw(&formatted)
    }

    /// Generate an embedding for a document with title.
    ///
    /// # Arguments
    /// * `text` - The document text
    /// * `title` - Optional document title
    ///
    /// # Errors
    /// Returns an error if embedding generation fails.
    pub fn embed_document(&mut self, text: &str, title: Option<&str>) -> Result<EmbeddingResult> {
        let formatted = format_doc_for_embedding(text, title);
        self.embed_raw(&formatted)
    }

    /// Generate an embedding for a search query.
    ///
    /// # Arguments
    /// * `query` - The search query
    ///
    /// # Errors
    /// Returns an error if embedding generation fails.
    pub fn embed_query(&mut self, query: &str) -> Result<EmbeddingResult> {
        let formatted = format_query_for_embedding(query);
        self.embed_raw(&formatted)
    }

    /// Generate embeddings for multiple texts in batch.
    ///
    /// # Arguments
    /// * `texts` - The texts to embed
    ///
    /// # Errors
    /// Returns an error if any embedding generation fails.
    pub fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<EmbeddingResult>> {
        texts.iter().map(|text| self.embed(text)).collect()
    }

    /// Raw embedding generation.
    fn embed_raw(&mut self, text: &str) -> Result<EmbeddingResult> {
        let ctx_params = LlamaContextParams::default().with_embeddings(true);

        let mut ctx = self
            .model
            .new_context(&LlamaBackend::init()?, ctx_params)
            .context("Failed to create context")?;

        // Tokenize input
        let tokens = self
            .model
            .str_to_token(text, AddBos::Always)
            .context("Failed to tokenize text")?;

        if tokens.is_empty() {
            bail!("Empty token sequence");
        }

        // Create batch and add tokens
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch.add(*token, i as i32, &[0], is_last)?;
        }

        // Decode
        ctx.decode(&mut batch).context("Failed to decode batch")?;

        // Get embeddings
        let embeddings = ctx
            .embeddings_seq_ith(0)
            .context("Failed to get embeddings")?;

        // Update dimensions
        if self.dimensions.is_none() {
            self.dimensions = Some(embeddings.len());
        }

        Ok(EmbeddingResult {
            embedding: embeddings.to_vec(),
            model: DEFAULT_EMBED_MODEL.to_string(),
        })
    }

    /// Get the embedding dimensions.
    #[must_use]
    pub const fn dimensions(&self) -> Option<usize> {
        self.dimensions
    }
}

/// Format a document for embedding using nomic-style format.
#[must_use]
pub fn format_doc_for_embedding(text: &str, title: Option<&str>) -> String {
    let title_str = title.unwrap_or("none");
    format!("title: {title_str} | text: {text}")
}

/// Format a query for embedding using nomic-style format.
#[must_use]
pub fn format_query_for_embedding(query: &str) -> String {
    format!("task: search result | query: {query}")
}

/// Get the path to a model in the cache directory.
///
/// # Errors
/// Returns an error if the model is not found.
pub fn get_model_path(model_name: &str) -> Result<PathBuf> {
    let cache_dir = config::get_model_cache_dir();
    let model_path = cache_dir.join(model_name);

    if !model_path.exists() {
        bail!(
            "Model not found: {}. Run 'qmd models pull' to download models.",
            model_path.display()
        );
    }

    Ok(model_path)
}

/// Check if a model exists in the cache.
#[must_use]
pub fn model_exists(model_name: &str) -> bool {
    let cache_dir = config::get_model_cache_dir();
    cache_dir.join(model_name).exists()
}

/// List available models in the cache.
#[must_use]
pub fn list_cached_models() -> Vec<String> {
    let cache_dir = config::get_model_cache_dir();
    if !cache_dir.exists() {
        return Vec::new();
    }

    fs::read_dir(&cache_dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "gguf"))
                .filter_map(|e| e.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Chunk a document into smaller pieces for embedding.
///
/// # Arguments
/// * `content` - The document content
/// * `max_chars` - Maximum characters per chunk
/// * `overlap_chars` - Overlap between chunks
#[must_use]
pub fn chunk_document(content: &str, max_chars: usize, overlap_chars: usize) -> Vec<Chunk> {
    if content.len() <= max_chars {
        return vec![Chunk {
            text: content.to_string(),
            pos: 0,
        }];
    }

    let mut chunks = Vec::new();
    let mut char_pos = 0;

    while char_pos < content.len() {
        let end_pos = (char_pos + max_chars).min(content.len());

        // Find a good break point if not at the end
        let actual_end = if end_pos < content.len() {
            find_break_point(content, char_pos, end_pos)
        } else {
            end_pos
        };

        // Ensure progress
        let actual_end = if actual_end <= char_pos {
            (char_pos + max_chars).min(content.len())
        } else {
            actual_end
        };

        chunks.push(Chunk {
            text: content[char_pos..actual_end].to_string(),
            pos: char_pos,
        });

        if actual_end >= content.len() {
            break;
        }

        // Move forward with overlap
        char_pos = actual_end.saturating_sub(overlap_chars);
        if let Some(last) = chunks.last()
            && char_pos <= last.pos
        {
            char_pos = actual_end;
        }
    }

    chunks
}

/// Find a good break point in text (paragraph > sentence > line > word).
fn find_break_point(content: &str, start: usize, end: usize) -> usize {
    let slice = &content[start..end];
    let search_start = slice.len() * 7 / 10; // Last 30%
    let search_slice = &slice[search_start..];

    // Priority: paragraph > sentence > line > word
    if let Some(pos) = search_slice.rfind("\n\n") {
        return start + search_start + pos + 2;
    }

    // Sentence endings
    for pattern in &[". ", ".\n", "? ", "?\n", "! ", "!\n"] {
        if let Some(pos) = search_slice.rfind(pattern) {
            return start + search_start + pos + 2;
        }
    }

    // Line break
    if let Some(pos) = search_slice.rfind('\n') {
        return start + search_start + pos + 1;
    }

    // Word break
    if let Some(pos) = search_slice.rfind(' ') {
        return start + search_start + pos + 1;
    }

    end
}

/// Calculate cosine similarity between two vectors.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Query type for different search backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// Lexical (BM25) search.
    Lex,
    /// Vector (semantic) search.
    Vec,
    /// `HyDE` - Hypothetical Document Embedding.
    Hyde,
}

/// A single query with its target backend type.
#[derive(Debug, Clone)]
pub struct Queryable {
    /// Query type.
    pub query_type: QueryType,
    /// Query text.
    pub text: String,
}

impl Queryable {
    /// Create a new queryable.
    #[must_use]
    pub fn new(query_type: QueryType, text: impl Into<String>) -> Self {
        Self {
            query_type,
            text: text.into(),
        }
    }

    /// Create a lexical query.
    #[must_use]
    pub fn lex(text: impl Into<String>) -> Self {
        Self::new(QueryType::Lex, text)
    }

    /// Create a vector query.
    #[must_use]
    pub fn vec(text: impl Into<String>) -> Self {
        Self::new(QueryType::Vec, text)
    }

    /// Create a `HyDE` query.
    #[must_use]
    pub fn hyde(text: impl Into<String>) -> Self {
        Self::new(QueryType::Hyde, text)
    }
}

/// Document to be reranked.
#[derive(Debug, Clone)]
pub struct RerankDocument {
    /// File path or identifier.
    pub file: String,
    /// Document text content.
    pub text: String,
    /// Optional title.
    pub title: Option<String>,
}

/// Rerank result for a single document.
#[derive(Debug, Clone)]
pub struct RerankResult {
    /// File path or identifier.
    pub file: String,
    /// Relevance score (higher is better).
    pub score: f32,
    /// Original index in input list.
    pub index: usize,
}

/// Parsed `HuggingFace` model reference.
#[derive(Debug, Clone)]
struct HfRef {
    /// Repository (e.g., "ggml-org/embeddinggemma-300M-GGUF").
    repo: String,
    /// File name (e.g., "embeddinggemma-300M-Q8_0.gguf").
    file: String,
}

/// Parse a `HuggingFace` URI like "hf:user/repo/file.gguf".
fn parse_hf_uri(uri: &str) -> Option<HfRef> {
    if !uri.starts_with("hf:") {
        return None;
    }
    let without_prefix = &uri[3..];
    let parts: Vec<&str> = without_prefix.splitn(3, '/').collect();
    if parts.len() < 3 {
        return None;
    }
    Some(HfRef {
        repo: format!("{}/{}", parts[0], parts[1]),
        file: parts[2].to_string(),
    })
}

/// Model pull result.
#[derive(Debug, Clone)]
pub struct PullResult {
    /// Model URI or name.
    pub model: String,
    /// Local file path.
    pub path: PathBuf,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Whether the model was refreshed (re-downloaded).
    pub refreshed: bool,
}

/// Download a model from `HuggingFace`.
///
/// # Arguments
/// * `model_uri` - Model URI (e.g., "hf:user/repo/file.gguf" or local filename)
/// * `refresh` - Force re-download even if cached
///
/// # Errors
/// Returns error if download fails.
pub fn pull_model(model_uri: &str, refresh: bool) -> Result<PullResult> {
    let cache_dir = config::get_model_cache_dir();
    fs::create_dir_all(&cache_dir)?;

    // Parse HuggingFace URI
    let hf_ref = parse_hf_uri(model_uri);

    let filename = if let Some(ref hf) = hf_ref {
        hf.file.clone()
    } else {
        // Assume it's already a filename
        model_uri.to_string()
    };

    let local_path = cache_dir.join(&filename);
    let etag_path = cache_dir.join(format!("{filename}.etag"));

    // Check if we need to download
    let should_download = if refresh {
        true
    } else if !local_path.exists() {
        true
    } else if let Some(ref hf) = hf_ref {
        // Check ETag for updates
        let remote_etag = get_remote_etag(hf);
        let local_etag = fs::read_to_string(&etag_path).ok();
        remote_etag.is_some() && remote_etag != local_etag
    } else {
        false
    };

    if should_download {
        if let Some(ref hf) = hf_ref {
            download_from_hf(hf, &local_path, &etag_path)?;
        } else {
            bail!("Model not found and no HuggingFace URI provided: {model_uri}");
        }
    }

    let size_bytes = fs::metadata(&local_path).map_or(0, |m| m.len());

    Ok(PullResult {
        model: model_uri.to_string(),
        path: local_path,
        size_bytes,
        refreshed: should_download,
    })
}

/// Get remote `ETag` from `HuggingFace` for cache validation.
fn get_remote_etag(hf: &HfRef) -> Option<String> {
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        hf.repo, hf.file
    );

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .ok()?;

    let resp = client.head(&url).send().ok()?;
    if !resp.status().is_success() {
        return None;
    }

    resp.headers()
        .get("etag")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_matches('"').to_string())
}

/// Download a file from `HuggingFace` with progress bar.
fn download_from_hf(hf: &HfRef, local_path: &Path, etag_path: &Path) -> Result<()> {
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        hf.repo, hf.file
    );

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_hours(1))
        .build()?;

    let mut resp = client.get(&url).send()?;
    if !resp.status().is_success() {
        bail!("Failed to download {}: HTTP {}", url, resp.status());
    }

    let total_size = resp.content_length().unwrap_or(0);

    // Create progress bar
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .expect("valid template")
            .progress_chars("#>-"),
    );
    pb.set_message(format!("Downloading {}", hf.file));

    // Download with progress
    let mut file = File::create(local_path)?;
    let mut downloaded: u64 = 0;
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = resp.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read])?;
        downloaded += bytes_read as u64;
        pb.set_position(downloaded);
    }

    pb.finish_with_message(format!("Downloaded {}", hf.file));

    // Save ETag for cache validation
    if let Some(etag) = resp.headers().get("etag")
        && let Ok(etag_str) = etag.to_str()
    {
        let _ = fs::write(etag_path, etag_str.trim_matches('"'));
    }

    Ok(())
}

/// Pull multiple models.
///
/// # Errors
/// Returns error if any download fails.
pub fn pull_models(models: &[&str], refresh: bool) -> Result<Vec<PullResult>> {
    models.iter().map(|m| pull_model(m, refresh)).collect()
}

/// Resolve a model URI to a local path, downloading if needed.
///
/// # Errors
/// Returns error if model cannot be resolved.
pub fn resolve_model(model_uri: &str) -> Result<PathBuf> {
    let result = pull_model(model_uri, false)?;
    Ok(result.path)
}

/// Expand a search query into multiple variations.
///
/// This is a simple implementation that generates variations without LLM.
/// For full LLM-based expansion, use the generation model.
#[must_use]
pub fn expand_query_simple(query: &str) -> Vec<Queryable> {
    let mut queries = Vec::new();

    // Original query for all search types
    queries.push(Queryable::lex(query));
    queries.push(Queryable::vec(query));

    // Generate simple HyDE-style expansion
    let hyde_text = format!("Information about {query}");
    queries.push(Queryable::hyde(hyde_text));

    queries
}

/// Parse query expansion output from LLM.
///
/// Expected format:
/// ```text
/// lex: keyword search terms
/// vec: semantic query
/// hyde: hypothetical document
/// ```
#[must_use]
pub fn parse_query_expansion(output: &str, original_query: &str) -> Vec<Queryable> {
    let mut queries = Vec::new();
    let query_lower = original_query.to_lowercase();

    // Regex to match "type: content" lines
    let line_re = Regex::new(r"^(lex|vec|hyde):\s*(.+)$").ok();

    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(ref re) = line_re
            && let Some(caps) = re.captures(line)
        {
            let query_type = match &caps[1] {
                "lex" => QueryType::Lex,
                "vec" => QueryType::Vec,
                "hyde" => QueryType::Hyde,
                _ => continue,
            };
            let text = caps[2].trim();

            // Validate that expansion contains at least one term from original query
            let text_lower = text.to_lowercase();
            let has_query_term = query_lower
                .split_whitespace()
                .any(|term| term.len() >= 3 && text_lower.contains(term));

            if has_query_term || query_lower.len() < 3 {
                queries.push(Queryable::new(query_type, text));
            }
        }
    }

    // Fallback if no valid queries found
    if queries.is_empty() {
        return expand_query_simple(original_query);
    }

    queries
}

/// RRF result with merged score.
#[derive(Debug, Clone)]
pub struct RrfResult {
    /// File path.
    pub file: String,
    /// Display path.
    pub display_path: String,
    /// Document title.
    pub title: String,
    /// Document body.
    pub body: String,
    /// Merged RRF score.
    pub score: f64,
    /// Best rank across all lists (0-indexed).
    pub best_rank: usize,
}

/// Combine multiple ranked lists using Reciprocal Rank Fusion.
///
/// RRF score = sum(weight / (k + rank + 1)) across all lists where doc appears.
/// k=60 is standard, provides good balance between top and lower ranks.
///
/// # Arguments
/// * `result_lists` - Vector of ranked result lists (file, `display_path`, title, body)
/// * `weights` - Optional weights for each list (default 1.0)
/// * `k` - RRF parameter (default 60)
#[must_use]
pub fn reciprocal_rank_fusion(
    result_lists: &[Vec<(String, String, String, String)>],
    weights: Option<&[f64]>,
    k: usize,
) -> Vec<RrfResult> {
    use std::collections::HashMap;

    let mut scores: HashMap<String, (f64, String, String, String, usize)> = HashMap::new();

    for (list_idx, results) in result_lists.iter().enumerate() {
        let weight = weights
            .and_then(|w| w.get(list_idx))
            .copied()
            .unwrap_or(1.0);

        for (rank, (file, display_path, title, body)) in results.iter().enumerate() {
            let rrf_score = weight / (k + rank + 1) as f64;

            scores
                .entry(file.clone())
                .and_modify(|(score, _, _, _, best_rank)| {
                    *score += rrf_score;
                    *best_rank = (*best_rank).min(rank);
                })
                .or_insert((
                    rrf_score,
                    display_path.clone(),
                    title.clone(),
                    body.clone(),
                    rank,
                ));
        }
    }

    // Convert to results and add bonus for best rank
    let mut results: Vec<RrfResult> = scores
        .into_iter()
        .map(|(file, (score, display_path, title, body, best_rank))| {
            // Add bonus for top-ranked documents to prevent dilution
            let bonus = match best_rank {
                0 => 0.05,     // Ranked #1 somewhere
                1..=2 => 0.02, // Ranked top-3 somewhere
                _ => 0.0,
            };

            RrfResult {
                file,
                display_path,
                title,
                body,
                score: score + bonus,
                best_rank,
            }
        })
        .collect();

    // Sort by score descending
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

/// Snippet extraction result.
#[derive(Debug, Clone)]
pub struct SnippetResult {
    /// Extracted snippet text.
    pub snippet: String,
    /// Line number where snippet starts.
    pub line: usize,
}

/// Extract a relevant snippet from document body.
///
/// # Arguments
/// * `body` - Full document body
/// * `query` - Search query for context
/// * `max_chars` - Maximum snippet length
/// * `chunk_pos` - Optional character position hint (from vector search)
#[must_use]
pub fn extract_snippet(
    body: &str,
    query: &str,
    max_chars: usize,
    chunk_pos: Option<usize>,
) -> SnippetResult {
    if body.len() <= max_chars {
        return SnippetResult {
            snippet: body.to_string(),
            line: 1,
        };
    }

    // Get query terms for matching
    let terms: Vec<&str> = query.split_whitespace().filter(|t| t.len() >= 3).collect();

    let body_lower = body.to_lowercase();

    // Find best position based on term matches or chunk_pos
    let start_pos = if let Some(pos) = chunk_pos {
        pos.min(body.len().saturating_sub(max_chars))
    } else {
        // Find first occurrence of any query term
        let mut best_pos = 0;
        for term in &terms {
            if let Some(pos) = body_lower.find(&term.to_lowercase()) {
                best_pos = pos.saturating_sub(50); // Start 50 chars before match
                break;
            }
        }
        best_pos
    };

    // Extend to line boundaries
    let line_start = body[..start_pos].rfind('\n').map_or(0, |p| p + 1);

    let end_pos = (line_start + max_chars).min(body.len());
    let line_end = body[end_pos..]
        .find('\n')
        .map_or(body.len(), |p| end_pos + p);

    // Calculate line number
    let line = body[..line_start].matches('\n').count() + 1;

    let snippet = body[line_start..line_end].to_string();

    SnippetResult { snippet, line }
}

/// Index health information.
#[derive(Debug, Clone, Copy)]
pub struct IndexHealth {
    /// Number of documents needing embedding.
    pub needs_embedding: usize,
    /// Total number of documents.
    pub total_docs: usize,
    /// Days since last update (None if never updated).
    pub days_stale: Option<u64>,
}

impl IndexHealth {
    /// Check if index is healthy.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        let embedding_ok = self.needs_embedding == 0
            || (self.needs_embedding as f64 / self.total_docs.max(1) as f64) < 0.1;
        let freshness_ok = self.days_stale.is_none() || self.days_stale < Some(14);
        embedding_ok && freshness_ok
    }

    /// Get warning message if any issues.
    #[must_use]
    pub fn warning_message(&self) -> Option<String> {
        let mut messages = Vec::new();

        if self.needs_embedding > 0 {
            let pct =
                (self.needs_embedding as f64 / self.total_docs.max(1) as f64 * 100.0) as usize;
            if pct >= 10 {
                messages.push(format!(
                    "{} documents ({}%) need embeddings. Run 'qmd embed' for better results.",
                    self.needs_embedding, pct
                ));
            }
        }

        if let Some(days) = self.days_stale
            && days >= 14
        {
            messages.push(format!(
                "Index last updated {days} days ago. Run 'qmd update' to refresh."
            ));
        }

        if messages.is_empty() {
            None
        } else {
            Some(messages.join("\n"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_doc_for_embedding() {
        let result = format_doc_for_embedding("hello world", Some("Test Title"));
        assert_eq!(result, "title: Test Title | text: hello world");

        let result = format_doc_for_embedding("hello world", None);
        assert_eq!(result, "title: none | text: hello world");
    }

    #[test]
    fn test_format_query_for_embedding() {
        let result = format_query_for_embedding("test query");
        assert_eq!(result, "task: search result | query: test query");
    }

    #[test]
    fn test_chunk_document_small() {
        let content = "Small content";
        let chunks = chunk_document(content, 100, 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, content);
        assert_eq!(chunks[0].pos, 0);
    }

    #[test]
    fn test_chunk_document_large() {
        let content = "a".repeat(500);
        let chunks = chunk_document(&content, 100, 10);
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }
}
