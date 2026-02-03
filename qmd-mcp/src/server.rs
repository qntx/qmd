//! MCP Server implementation for QMD.
//!
//! Uses `spawn_blocking` to run synchronous rusqlite operations in a
//! dedicated thread pool, following the Rust community best practice.

use rmcp::{
    ServerHandler,
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::{
        CallToolResult, Content, Implementation, InitializeResult, ProtocolVersion,
        ServerCapabilities,
    },
    schemars::JsonSchema,
    tool, tool_handler, tool_router,
};
use serde::{Deserialize, Serialize};

/// Type alias for ServerInfo (same as InitializeResult).
type ServerInfo = InitializeResult;

/// QMD MCP Server that provides search and document retrieval tools.
#[derive(Clone, Default, Debug)]
pub struct QmdMcpServer {
    /// Tool router for handling tool calls.
    tool_router: ToolRouter<Self>,
}

impl QmdMcpServer {
    /// Create a new QMD MCP server instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

/// Parameters for search tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct SearchParams {
    /// Search query - keywords or phrases to find.
    pub query: String,
    /// Maximum number of results (default: 10).
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Minimum relevance score 0-1 (default: 0).
    #[serde(default)]
    pub min_score: f64,
    /// Filter to a specific collection by name.
    pub collection: Option<String>,
}

/// Parameters for vsearch tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct VsearchParams {
    /// Natural language query - describe what you're looking for.
    pub query: String,
    /// Maximum number of results (default: 10).
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Minimum similarity score 0-1 (default: 0.3).
    #[serde(default = "default_vsearch_min_score")]
    pub min_score: f64,
    /// Filter to a specific collection by name.
    pub collection: Option<String>,
}

/// Parameters for query tool (hybrid search).
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct QueryParams {
    /// Natural language query - describe what you're looking for.
    pub query: String,
    /// Maximum number of results (default: 10).
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Filter to a specific collection by name.
    pub collection: Option<String>,
}

/// Parameters for get tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct GetParams {
    /// File path or docid from search results (e.g., 'notes/meeting.md', '#abc123').
    pub file: String,
    /// Start from this line number (1-indexed).
    pub from_line: Option<usize>,
    /// Maximum number of lines to return.
    pub max_lines: Option<usize>,
    /// Add line numbers to output (default: true).
    #[serde(default = "default_true")]
    pub line_numbers: bool,
}

/// Parameters for multi_get tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct MultiGetParams {
    /// Comma-separated list of file paths or docids (e.g., 'notes/a.md,notes/b.md' or '#abc123,#def456').
    pub files: String,
    /// Maximum lines per file (default: no limit).
    pub max_lines: Option<usize>,
    /// Skip files larger than this many bytes (default: 10KB).
    #[serde(default = "default_max_bytes")]
    pub max_bytes: usize,
}

/// Parameters for ls tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct LsParams {
    /// Collection name to list files from. If empty, lists all collections.
    pub collection: Option<String>,
    /// Path prefix to filter files (e.g., 'journals/2025').
    pub prefix: Option<String>,
}

/// Parameters for ask tool (RAG-based Q&A).
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct AskParams {
    /// Natural language question to answer based on indexed documents.
    pub question: String,
    /// Number of context documents to use (default: 5).
    #[serde(default = "default_context_limit")]
    pub limit: usize,
    /// Maximum tokens for the generated answer (default: 500).
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Filter to a specific collection by name.
    pub collection: Option<String>,
}

/// Parameters for rerank tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct RerankParams {
    /// Query to rank documents against.
    pub query: String,
    /// Comma-separated list of file paths or docids to rerank.
    pub files: String,
    /// Number of top results to return (default: 10).
    #[serde(default = "default_rerank_limit")]
    pub limit: usize,
}

/// Parameters for qsearch tool (advanced hybrid search).
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct QsearchParams {
    /// Search query - natural language question or keywords.
    pub query: String,
    /// Maximum number of results (default: 10).
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Filter to a specific collection by name.
    pub collection: Option<String>,
    /// Skip query expansion (default: false).
    #[serde(default)]
    pub no_expand: bool,
    /// Skip reranking (default: false).
    #[serde(default)]
    pub no_rerank: bool,
}

fn default_limit() -> usize {
    10
}
fn default_true() -> bool {
    true
}
fn default_vsearch_min_score() -> f64 {
    0.3
}
fn default_max_bytes() -> usize {
    10240
}
fn default_context_limit() -> usize {
    5
}
fn default_max_tokens() -> usize {
    500
}
fn default_rerank_limit() -> usize {
    10
}

/// Search result item for JSON output.
#[derive(Debug, Serialize)]
struct SearchResultItem {
    docid: String,
    file: String,
    title: String,
    score: f64,
    context: Option<String>,
}

/// Status result for JSON output.
#[derive(Debug, Serialize)]
struct StatusResult {
    total_documents: usize,
    needs_embedding: usize,
    has_vector_index: bool,
    collections: Vec<CollectionStatus>,
}

/// Collection status for JSON output.
#[derive(Debug, Serialize)]
struct CollectionStatus {
    name: String,
    path: String,
    documents: usize,
}

/// Convert qmd error to MCP error.
fn to_mcp_error(e: impl std::fmt::Display) -> rmcp::ErrorData {
    rmcp::ErrorData::internal_error(e.to_string(), None)
}

/// Add line numbers to text.
fn add_line_numbers(text: &str, start: usize) -> String {
    text.lines()
        .enumerate()
        .map(|(i, line)| format!("{}: {}", start + i, line))
        .collect::<Vec<_>>()
        .join("\n")
}

#[tool_router]
impl QmdMcpServer {
    /// Fast keyword-based full-text search using BM25.
    /// Best for finding documents with specific words or phrases.
    #[tool(name = "search")]
    async fn search(
        &self,
        params: Parameters<SearchParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let p = params.0;

        // Run synchronous database operation in blocking thread pool
        let result =
            tokio::task::spawn_blocking(move || -> Result<Vec<SearchResultItem>, qmd::QmdError> {
                let store = qmd::Store::new()?;
                let results = store.search_fts(&p.query, p.limit, p.collection.as_deref())?;

                Ok(results
                    .into_iter()
                    .filter(|r| r.score >= p.min_score)
                    .map(|r| SearchResultItem {
                        docid: format!("#{}", r.doc.docid),
                        file: r.doc.display_path,
                        title: r.doc.title,
                        score: (r.score * 100.0).round() / 100.0,
                        context: r.doc.context,
                    })
                    .collect())
            })
            .await
            .map_err(|e| to_mcp_error(e))?
            .map_err(to_mcp_error)?;

        let summary = if result.is_empty() {
            "No results found".to_string()
        } else {
            result
                .iter()
                .map(|r| {
                    format!(
                        "{} {}% {} - {}",
                        r.docid,
                        (r.score * 100.0) as i32,
                        r.file,
                        r.title
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        Ok(CallToolResult::success(vec![Content::text(summary)]))
    }

    /// Retrieve the full content of a document by its file path or docid (#abc123).
    #[tool(name = "get")]
    async fn get(&self, params: Parameters<GetParams>) -> Result<CallToolResult, rmcp::ErrorData> {
        let p = params.0;
        let file_for_err = p.file.clone();

        let result = tokio::task::spawn_blocking(
            move || -> Result<Option<(String, String, Option<String>)>, qmd::QmdError> {
                let store = qmd::Store::new()?;

                // Check if it's a docid
                let (collection, path) = if p.file.starts_with('#') {
                    match store.find_document_by_docid(&p.file)? {
                        Some(cp) => cp,
                        None => return Ok(None),
                    }
                } else {
                    // Parse collection/path format
                    let parts: Vec<&str> = p.file.splitn(2, '/').collect();
                    if parts.len() == 2 {
                        (parts[0].to_string(), parts[1].to_string())
                    } else {
                        return Ok(None);
                    }
                };

                match store.get_document(&collection, &path)? {
                    Some(doc) => {
                        let mut body = doc.body.unwrap_or_default();

                        // Apply line range
                        if let Some(from) = p.from_line {
                            let lines: Vec<&str> = body.lines().collect();
                            let start = from.saturating_sub(1);
                            let end = p.max_lines.map(|m| start + m).unwrap_or(lines.len());
                            body = lines
                                .get(start..end.min(lines.len()))
                                .map(|s| s.join("\n"))
                                .unwrap_or_default();
                        }

                        // Add line numbers
                        if p.line_numbers {
                            body = add_line_numbers(&body, p.from_line.unwrap_or(1));
                        }

                        Ok(Some((doc.title, body, doc.context)))
                    }
                    None => Ok(None),
                }
            },
        )
        .await
        .map_err(|e| to_mcp_error(e))?
        .map_err(to_mcp_error)?;

        match result {
            Some((title, body, context)) => {
                let mut text = format!("# {}\n\n", title);
                if let Some(ctx) = context {
                    text.push_str(&format!("<!-- Context: {} -->\n\n", ctx));
                }
                text.push_str(&body);
                Ok(CallToolResult::success(vec![Content::text(text)]))
            }
            None => Ok(CallToolResult::success(vec![Content::text(format!(
                "Document not found: {}",
                file_for_err
            ))])),
        }
    }

    /// Show the status of the QMD index: collections, document counts, and health information.
    #[tool(name = "status")]
    async fn status(&self) -> Result<CallToolResult, rmcp::ErrorData> {
        let result = tokio::task::spawn_blocking(|| -> Result<StatusResult, qmd::QmdError> {
            let store = qmd::Store::new()?;
            let status = store.get_status()?;

            Ok(StatusResult {
                total_documents: status.total_documents,
                needs_embedding: status.needs_embedding,
                has_vector_index: status.has_vector_index,
                collections: status
                    .collections
                    .into_iter()
                    .map(|c| CollectionStatus {
                        name: c.name,
                        path: c.pwd,
                        documents: c.active_count,
                    })
                    .collect(),
            })
        })
        .await
        .map_err(|e| to_mcp_error(e))?
        .map_err(to_mcp_error)?;

        let mut lines = vec![
            "QMD Index Status:".to_string(),
            format!("  Total documents: {}", result.total_documents),
            format!("  Needs embedding: {}", result.needs_embedding),
            format!(
                "  Vector index: {}",
                if result.has_vector_index { "yes" } else { "no" }
            ),
            format!("  Collections: {}", result.collections.len()),
        ];
        for col in &result.collections {
            lines.push(format!("    - {} ({} docs)", col.name, col.documents));
        }

        Ok(CallToolResult::success(vec![Content::text(
            lines.join("\n"),
        )]))
    }

    /// Semantic similarity search using vector embeddings.
    /// Finds conceptually related content even without exact keyword matches.
    /// Requires embeddings to be generated first (run 'qmd embed').
    #[tool(name = "vsearch")]
    async fn vsearch(
        &self,
        params: Parameters<VsearchParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let p = params.0;

        let result =
            tokio::task::spawn_blocking(move || -> Result<Vec<SearchResultItem>, String> {
                let store = qmd::Store::new().map_err(|e| e.to_string())?;

                // Load embedding engine
                let mut engine = qmd::EmbeddingEngine::load_default().map_err(|e| e.to_string())?;

                // Embed query
                let query_emb = engine.embed_query(&p.query).map_err(|e| e.to_string())?;

                // Vector search
                let results = store
                    .search_vec(&query_emb.embedding, p.limit, p.collection.as_deref())
                    .map_err(|e| e.to_string())?;

                Ok(results
                    .into_iter()
                    .filter(|r| r.score >= p.min_score)
                    .map(|r| SearchResultItem {
                        docid: format!("#{}", r.doc.docid),
                        file: r.doc.display_path,
                        title: r.doc.title,
                        score: (r.score * 100.0).round() / 100.0,
                        context: r.doc.context,
                    })
                    .collect())
            })
            .await
            .map_err(|e| to_mcp_error(e))?
            .map_err(|e| to_mcp_error(e))?;

        let summary = if result.is_empty() {
            "No results found (ensure embeddings are generated with 'qmd embed')".to_string()
        } else {
            result
                .iter()
                .map(|r| {
                    format!(
                        "{} {:.0}% {} - {}",
                        r.docid,
                        r.score * 100.0,
                        r.file,
                        r.title
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        Ok(CallToolResult::success(vec![Content::text(summary)]))
    }

    /// Hybrid search combining BM25 + vector search with RRF fusion.
    /// Best quality results but requires embeddings.
    #[tool(name = "query")]
    async fn query(
        &self,
        params: Parameters<QueryParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let p = params.0;

        let result =
            tokio::task::spawn_blocking(move || -> Result<Vec<SearchResultItem>, String> {
                let store = qmd::Store::new().map_err(|e| e.to_string())?;

                // FTS search
                let fts_results = store
                    .search_fts(&p.query, p.limit * 2, p.collection.as_deref())
                    .map_err(|e| e.to_string())?;

                let fts_tuples: Vec<(String, String, String, String)> = fts_results
                    .iter()
                    .map(|r| {
                        (
                            r.doc.display_path.clone(),
                            r.doc.display_path.clone(),
                            r.doc.title.clone(),
                            String::new(),
                        )
                    })
                    .collect();

                // Try vector search (may fail if no embeddings)
                let vec_tuples: Vec<(String, String, String, String)> =
                    match qmd::EmbeddingEngine::load_default() {
                        Ok(mut engine) => match engine.embed_query(&p.query) {
                            Ok(query_emb) => {
                                match store.search_vec(
                                    &query_emb.embedding,
                                    p.limit * 2,
                                    p.collection.as_deref(),
                                ) {
                                    Ok(vec_results) => vec_results
                                        .iter()
                                        .map(|r| {
                                            (
                                                r.doc.display_path.clone(),
                                                r.doc.display_path.clone(),
                                                r.doc.title.clone(),
                                                String::new(),
                                            )
                                        })
                                        .collect(),
                                    Err(_) => Vec::new(),
                                }
                            }
                            Err(_) => Vec::new(),
                        },
                        Err(_) => Vec::new(),
                    };

                // RRF fusion
                let rrf_results = qmd::hybrid_search_rrf(fts_tuples, vec_tuples, 60);

                // Convert to SearchResultItem
                let items: Vec<SearchResultItem> = rrf_results
                    .into_iter()
                    .take(p.limit)
                    .map(|r| SearchResultItem {
                        docid: String::new(), // Will be filled below
                        file: r.display_path,
                        title: r.title,
                        score: r.score,
                        context: None,
                    })
                    .collect();

                // Enrich with docids
                let enriched: Vec<SearchResultItem> = items
                    .into_iter()
                    .filter_map(|mut item| {
                        let parts: Vec<&str> = item.file.splitn(2, '/').collect();
                        if parts.len() == 2 {
                            if let Ok(Some(doc)) = store.get_document(parts[0], parts[1]) {
                                item.docid = format!("#{}", doc.docid);
                                item.context = doc.context;
                                return Some(item);
                            }
                        }
                        None
                    })
                    .collect();

                Ok(enriched)
            })
            .await
            .map_err(|e| to_mcp_error(e))?
            .map_err(|e| to_mcp_error(e))?;

        let summary = if result.is_empty() {
            "No results found".to_string()
        } else {
            result
                .iter()
                .map(|r| format!("{} {:.2} {} - {}", r.docid, r.score, r.file, r.title))
                .collect::<Vec<_>>()
                .join("\n")
        };

        Ok(CallToolResult::success(vec![Content::text(summary)]))
    }

    /// Retrieve multiple documents by comma-separated file paths or docids.
    #[tool(name = "multi_get")]
    async fn multi_get(
        &self,
        params: Parameters<MultiGetParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let p = params.0;

        let result = tokio::task::spawn_blocking(move || -> Result<String, qmd::QmdError> {
            let store = qmd::Store::new()?;
            let files: Vec<&str> = p.files.split(',').map(str::trim).collect();
            let mut output = Vec::new();

            for file in files {
                // Resolve collection/path
                let (collection, path) = if file.starts_with('#') {
                    match store.find_document_by_docid(file)? {
                        Some(cp) => cp,
                        None => {
                            output.push(format!("--- {} ---\nNot found\n", file));
                            continue;
                        }
                    }
                } else {
                    let parts: Vec<&str> = file.splitn(2, '/').collect();
                    if parts.len() == 2 {
                        (parts[0].to_string(), parts[1].to_string())
                    } else {
                        output.push(format!("--- {} ---\nInvalid path format\n", file));
                        continue;
                    }
                };

                match store.get_document(&collection, &path)? {
                    Some(doc) => {
                        let body = doc.body.unwrap_or_default();

                        // Check size limit
                        if body.len() > p.max_bytes {
                            output.push(format!(
                                "--- {} ---\nSkipped: file too large ({} bytes)\n",
                                doc.display_path,
                                body.len()
                            ));
                            continue;
                        }

                        // Apply max_lines
                        let content = if let Some(max) = p.max_lines {
                            body.lines().take(max).collect::<Vec<_>>().join("\n")
                        } else {
                            body
                        };

                        output.push(format!(
                            "--- {} (#{}) ---\n# {}\n\n{}",
                            doc.display_path, doc.docid, doc.title, content
                        ));
                    }
                    None => {
                        output.push(format!("--- {}/{} ---\nNot found\n", collection, path));
                    }
                }
            }

            Ok(output.join("\n\n"))
        })
        .await
        .map_err(|e| to_mcp_error(e))?
        .map_err(to_mcp_error)?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// List collections or files in a collection.
    #[tool(name = "ls")]
    async fn ls(&self, params: Parameters<LsParams>) -> Result<CallToolResult, rmcp::ErrorData> {
        let p = params.0;

        let result = tokio::task::spawn_blocking(move || -> Result<String, qmd::QmdError> {
            let store = qmd::Store::new()?;

            match p.collection {
                Some(coll) => {
                    // List files in collection
                    let files = store.list_files(&coll, p.prefix.as_deref())?;
                    if files.is_empty() {
                        return Ok(format!("No files found in collection '{}'", coll));
                    }

                    let mut lines = vec![format!("Files in '{}':", coll)];
                    for (path, title, _modified, size) in files {
                        lines.push(format!("  {} ({}) - {}", path, format_size(size), title));
                    }
                    Ok(lines.join("\n"))
                }
                None => {
                    // List all collections
                    let collections = store.list_collections()?;
                    if collections.is_empty() {
                        return Ok(
                            "No collections found. Use 'qmd collection add <path>' to add one."
                                .to_string(),
                        );
                    }

                    let mut lines = vec!["Collections:".to_string()];
                    for coll in collections {
                        lines.push(format!(
                            "  {} ({} docs) - {}",
                            coll.name, coll.active_count, coll.pwd
                        ));
                    }
                    Ok(lines.join("\n"))
                }
            }
        })
        .await
        .map_err(|e| to_mcp_error(e))?
        .map_err(to_mcp_error)?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Ask a question and get an AI-generated answer based on relevant documents (RAG).
    /// Searches for context documents and generates a response using the LLM.
    #[tool(name = "ask")]
    async fn ask(&self, params: Parameters<AskParams>) -> Result<CallToolResult, rmcp::ErrorData> {
        let p = params.0;

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let store = qmd::Store::new().map_err(|e| e.to_string())?;

            // Search for relevant documents using vector search if available, fallback to FTS
            let context_docs = if let Ok(mut engine) = qmd::EmbeddingEngine::load_default() {
                if let Ok(query_result) = engine.embed_query(&p.question) {
                    store
                        .search_vec(&query_result.embedding, p.limit, p.collection.as_deref())
                        .unwrap_or_default()
                } else {
                    store
                        .search_fts(&p.question, p.limit, p.collection.as_deref())
                        .unwrap_or_default()
                }
            } else {
                store
                    .search_fts(&p.question, p.limit, p.collection.as_deref())
                    .unwrap_or_default()
            };

            if context_docs.is_empty() {
                return Ok("No relevant documents found to answer this question.".to_string());
            }

            // Build context from retrieved documents
            let mut context = String::new();
            let mut sources = Vec::new();
            for (i, result) in context_docs.iter().enumerate() {
                let body = store
                    .get_document(&result.doc.collection_name, &result.doc.path)
                    .ok()
                    .flatten()
                    .and_then(|d| d.body)
                    .unwrap_or_default();
                // Truncate to ~1000 chars per doc
                let truncated: String = body.chars().take(1000).collect();
                context.push_str(&format!(
                    "\n--- Document {} ({}): ---\n{}\n",
                    i + 1,
                    result.doc.display_path,
                    truncated
                ));
                sources.push(result.doc.display_path.clone());
            }

            // Generate answer using LLM
            let gen_engine = qmd::GenerationEngine::load_default()
                .map_err(|e| format!("Could not load generation model: {e}"))?;

            let prompt = format!(
                "Based on the following documents, answer the question concisely.\n\n\
                 Documents:\n{context}\n\n\
                 Question: {}\n\n\
                 Answer:",
                p.question
            );

            let gen_result = gen_engine
                .generate(&prompt, p.max_tokens)
                .map_err(|e| e.to_string())?;

            // Format output with answer and sources
            let mut output = format!("**Answer:**\n{}\n\n**Sources:**\n", gen_result.text);
            for src in &sources {
                output.push_str(&format!("- {}\n", src));
            }

            Ok(output)
        })
        .await
        .map_err(|e| to_mcp_error(e))?
        .map_err(|e| to_mcp_error(e))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Rerank documents by relevance to a query using a cross-encoder model.
    /// Improves search result quality by re-scoring documents against the query.
    #[tool(name = "rerank")]
    async fn rerank(
        &self,
        params: Parameters<RerankParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let p = params.0;

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let store = qmd::Store::new().map_err(|e| e.to_string())?;

            // Parse file list
            let file_list: Vec<&str> = p
                .files
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect();

            if file_list.is_empty() {
                return Err("No files specified".to_string());
            }

            // Resolve files and get content
            let mut docs: Vec<qmd::RerankDocument> = Vec::new();
            for file in &file_list {
                let (collection, path) = if file.starts_with('#') {
                    match store
                        .find_document_by_docid(file)
                        .map_err(|e| e.to_string())?
                    {
                        Some(cp) => cp,
                        None => continue,
                    }
                } else if qmd::is_virtual_path(file) {
                    qmd::parse_virtual_path(file).unwrap_or_else(|| {
                        let parts: Vec<&str> = file.splitn(2, '/').collect();
                        if parts.len() == 2 {
                            (parts[0].to_string(), parts[1].to_string())
                        } else {
                            (String::new(), file.to_string())
                        }
                    })
                } else {
                    let parts: Vec<&str> = file.splitn(2, '/').collect();
                    if parts.len() == 2 {
                        (parts[0].to_string(), parts[1].to_string())
                    } else {
                        continue;
                    }
                };

                if let Ok(Some(doc)) = store.get_document(&collection, &path) {
                    docs.push(qmd::RerankDocument {
                        file: doc.filepath.clone(),
                        text: doc.body.unwrap_or_default(),
                        title: Some(doc.title),
                    });
                }
            }

            if docs.is_empty() {
                return Err("No valid documents found".to_string());
            }

            // Rerank using cross-encoder
            let mut engine =
                qmd::RerankEngine::load_default().map_err(|e| format!("Rerank model: {e}"))?;

            let rerank_result = engine.rerank(&p.query, &docs).map_err(|e| e.to_string())?;

            // Format output
            let mut lines = Vec::new();
            for (i, r) in rerank_result.results.iter().take(p.limit).enumerate() {
                lines.push(format!("{}. {:.4} {}", i + 1, r.score, r.file));
            }

            Ok(lines.join("\n"))
        })
        .await
        .map_err(|e| to_mcp_error(e))?
        .map_err(|e| to_mcp_error(e))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Advanced hybrid search with query expansion, RRF fusion, and optional reranking.
    /// Best quality results combining multiple search strategies.
    #[tool(name = "qsearch")]
    async fn qsearch(
        &self,
        params: Parameters<QsearchParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let p = params.0;

        let result =
            tokio::task::spawn_blocking(move || -> Result<Vec<SearchResultItem>, String> {
                let store = qmd::Store::new().map_err(|e| e.to_string())?;

                // Expand query if enabled
                let queries = if p.no_expand || !qmd::GenerationEngine::is_available() {
                    vec![qmd::Queryable::lex(&p.query), qmd::Queryable::vec(&p.query)]
                } else {
                    match qmd::GenerationEngine::load_default() {
                        Ok(engine) => match engine.expand_query(&p.query, true) {
                            Ok(q) => q,
                            Err(_) => qmd::expand_query_simple(&p.query),
                        },
                        Err(_) => qmd::expand_query_simple(&p.query),
                    }
                };

                // Collect results from different search strategies
                let mut fts_results: Vec<(String, String, String, String)> = Vec::new();
                let mut vec_results: Vec<(String, String, String, String)> = Vec::new();

                for q in &queries {
                    match q.query_type {
                        qmd::QueryType::Lex => {
                            if let Ok(results) =
                                store.search_fts(&q.text, p.limit * 2, p.collection.as_deref())
                            {
                                for r in results {
                                    fts_results.push((
                                        r.doc.filepath.clone(),
                                        r.doc.display_path.clone(),
                                        r.doc.title.clone(),
                                        r.doc.body.clone().unwrap_or_default(),
                                    ));
                                }
                            }
                        }
                        qmd::QueryType::Vec | qmd::QueryType::Hyde => {
                            if let Ok(mut engine) = qmd::EmbeddingEngine::load_default() {
                                if let Ok(query_result) = engine.embed_query(&q.text) {
                                    if let Ok(results) = store.search_vec(
                                        &query_result.embedding,
                                        p.limit * 2,
                                        p.collection.as_deref(),
                                    ) {
                                        for r in results {
                                            let body = store
                                                .get_document(&r.doc.collection_name, &r.doc.path)
                                                .ok()
                                                .flatten()
                                                .and_then(|d| d.body)
                                                .unwrap_or_default();
                                            vec_results.push((
                                                r.doc.filepath.clone(),
                                                r.doc.display_path.clone(),
                                                r.doc.title.clone(),
                                                body,
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // RRF fusion
                let mut rrf_results = qmd::hybrid_search_rrf(fts_results, vec_results, 60);

                // Optional reranking
                if !p.no_rerank && qmd::RerankEngine::is_available() && !rrf_results.is_empty() {
                    if let Ok(mut reranker) = qmd::RerankEngine::load_default() {
                        let docs: Vec<qmd::RerankDocument> = rrf_results
                            .iter()
                            .take(p.limit * 2)
                            .map(|r| qmd::RerankDocument {
                                file: r.file.clone(),
                                text: r.body.clone(),
                                title: Some(r.title.clone()),
                            })
                            .collect();

                        if let Ok(reranked) = reranker.rerank(&p.query, &docs) {
                            let mut reordered = Vec::new();
                            for rr in reranked.results {
                                if let Some(orig) = rrf_results.iter().find(|r| r.file == rr.file) {
                                    reordered.push(orig.clone());
                                }
                            }
                            rrf_results = reordered;
                        }
                    }
                }

                rrf_results.truncate(p.limit);

                // Convert to SearchResultItem and enrich with docids
                let items: Vec<SearchResultItem> = rrf_results
                    .into_iter()
                    .filter_map(|r| {
                        let parts: Vec<&str> = r
                            .file
                            .strip_prefix("qmd://")
                            .unwrap_or(&r.file)
                            .splitn(2, '/')
                            .collect();
                        if parts.len() == 2 {
                            if let Ok(Some(doc)) = store.get_document(parts[0], parts[1]) {
                                return Some(SearchResultItem {
                                    docid: format!("#{}", doc.docid),
                                    file: r.display_path,
                                    title: r.title,
                                    score: r.score,
                                    context: doc.context,
                                });
                            }
                        }
                        None
                    })
                    .collect();

                Ok(items)
            })
            .await
            .map_err(|e| to_mcp_error(e))?
            .map_err(|e| to_mcp_error(e))?;

        let summary = if result.is_empty() {
            "No results found".to_string()
        } else {
            result
                .iter()
                .map(|r| format!("{} {:.2} {} - {}", r.docid, r.score, r.file, r.title))
                .collect::<Vec<_>>()
                .join("\n")
        };

        Ok(CallToolResult::success(vec![Content::text(summary)]))
    }
}

/// Format file size in human-readable form.
fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

#[tool_handler]
impl ServerHandler for QmdMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::LATEST,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "qmd".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                title: None,
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "QMD - Quick Markdown Search. A local search engine for markdown knowledge bases. \
                 Tools: 'search' (BM25), 'vsearch' (semantic), 'query'/'qsearch' (hybrid), \
                 'ask' (RAG Q&A), 'rerank' (cross-encoder), \
                 'get'/'multi_get' (retrieve), 'ls' (browse), 'status' (health)."
                    .into(),
            ),
        }
    }
}
