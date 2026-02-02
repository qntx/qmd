//! LLM module for QMD - embedding and vector search support.
//!
//! This module provides local LLM inference for:
//! - Document embeddings using GGUF models
//! - Vector similarity search
//! - Query expansion (future)

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};

use crate::config;

/// Default embedding model (embeddinggemma-300M)
pub const DEFAULT_EMBED_MODEL: &str = "embeddinggemma-300M-Q8_0.gguf";

/// Chunk size in characters for document splitting
pub const CHUNK_SIZE_CHARS: usize = 4000;

/// Overlap between chunks in characters
pub const CHUNK_OVERLAP_CHARS: usize = 200;

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
    pub fn dimensions(&self) -> Option<usize> {
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

    std::fs::read_dir(&cache_dir)
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
        if let Some(last) = chunks.last() {
            if char_pos <= last.pos {
                char_pos = actual_end;
            }
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
