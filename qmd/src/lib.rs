//! QMD - Query Markdown Documents
//!
//! A high-performance local search engine for markdown files with full-text search,
//! vector semantic search, and LLM-powered features.
//!
//! ## Features
//!
//! - **Full-text search** with BM25 ranking via SQLite FTS5
//! - **Vector semantic search** with local embeddings (GGUF models)
//! - **Hybrid search** with query expansion and RRF fusion
//! - **Reranking** with cross-encoder models
//! - **Collection management** for organizing document sets
//! - **Automatic model download** from HuggingFace
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use qmd::{Store, EmbeddingEngine, Result};
//!
//! fn main() -> Result<()> {
//!     // Open the document store
//!     let store = Store::new()?;
//!
//!     // Full-text search (BM25)
//!     let results = store.search_fts("rust programming", 10, None)?;
//!
//!     // Vector search (requires embedding model)
//!     let mut engine = EmbeddingEngine::load_default()?;
//!     let query_emb = engine.embed_query("how to use rust")?;
//!     let vec_results = store.search_vec(&query_emb.embedding, 10, None)?;
//!
//!     Ok(())
//! }
//! ```

pub mod collections;
pub mod config;
pub mod error;
pub mod formatter;
pub mod llm;
pub mod store;

// Re-export core types for convenient access
pub use error::{QmdError, Result};

// Store and search
pub use store::{
    CollectionInfo, DocumentResult, IndexStatus, SearchResult, SearchSource, Store,
    convert_git_bash_path, find_similar_files, is_absolute_path, is_docid, is_virtual_path,
    match_files_by_glob, normalize_filesystem_path, normalize_path_separators, parse_virtual_path,
    should_exclude,
};

// LLM and embeddings
pub use llm::{
    BatchRerankResult, CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, Chunk, Cursor, EmbeddingEngine,
    EmbeddingResult, GenerationEngine, GenerationResult, IndexHealth, Progress, PullResult,
    QueryType, Queryable, RerankDocument, RerankEngine, RerankResult, RrfResult, SnippetResult,
    TokenChunk, chunk_document, chunk_document_by_tokens, cosine_similarity, expand_query_simple,
    extract_snippet, format_doc_for_embedding, format_eta, format_query_for_embedding,
    hybrid_search_rrf, pull_model, pull_models, reciprocal_rank_fusion, render_progress_bar,
    resolve_model,
};

// Collections management
pub use collections::{
    add_collection, add_context, get_collection, list_all_contexts, list_collections,
    remove_collection, remove_context, rename_collection, set_global_context,
};

// Formatting utilities
pub use formatter::{
    OutputFormat, add_line_numbers, format_bytes, format_documents, format_ls_time,
    format_search_results, format_time_ago,
};
