//! QMD - Query Markdown Documents
//!
//! A full-text search tool for markdown files with collection management,
//! context annotations, vector search, and virtual path support.
//!
//! ## Features
//!
//! - Full-text search with BM25 ranking
//! - Vector semantic search with local embeddings
//! - Query expansion and RRF fusion
//! - Automatic model download from `HuggingFace`
//! - Fuzzy file matching
//! - Index health monitoring

pub mod cli;
pub mod collections;
pub mod config;
pub mod error;
pub mod formatter;
pub mod llm;
pub mod store;

pub use cli::{Cli, Commands};
pub use error::{QmdError, Result};
pub use llm::{
    IndexHealth, PullResult, QueryType, Queryable, RerankDocument, RerankResult, RrfResult,
    SnippetResult, expand_query_simple, extract_snippet, pull_model, pull_models,
    reciprocal_rank_fusion, resolve_model,
};
pub use store::{Store, find_similar_files, match_files_by_glob};
