//! On-device hybrid search for markdown: BM25, vectors and LLM reranking.
//!
//! Rust port of [tobi/qmd](https://github.com/tobi/qmd) (v2.8.3).
//!
//! # P1.3 surface
//!
//! Storage and configuration foundation, incremental indexing, and BM25
//! full-text search: [`Qmd::builder`] opens a SQLite index (WAL, busy
//! timeout, schema versioning, sqlite-vec), configuration comes from a
//! YAML file, an inline [`Config`], or the database mirror,
//! [`Qmd::update`] scans collections into
//! `documents`/`content`/`documents_fts`, and [`Qmd::search_lex`]
//! answers BM25-ranked keyword queries with [`extract_snippet`] excerpts.
//!
//! ```
//! # fn main() -> qmd::Result<()> {
//! let qmd = qmd::Qmd::builder("/tmp/example-rs.sqlite").build()?;
//! qmd.add_collection(
//!     "notes",
//!     &qmd::CollectionSpec {
//!         path: "/tmp/notes".to_string(),
//!         ..qmd::CollectionSpec::default()
//!     },
//! )?;
//! assert_eq!(qmd.collections()?.len(), 1);
//! qmd.close()?;
//! # std::fs::remove_file("/tmp/example-rs.sqlite").ok();
//! # std::fs::remove_file("/tmp/example-rs.sqlite-wal").ok();
//! # std::fs::remove_file("/tmp/example-rs.sqlite-shm").ok();
//! # Ok(())
//! # }
//! ```

#![allow(
    unused_crate_dependencies,
    reason = "tempfile is a dev-dependency used only by tests/"
)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod collection;
pub mod config;
mod env;
mod error;
pub mod paths;
mod qmd;
pub mod snippet;
mod store;

pub use collection::split_glob_mask;
pub use config::{Collection, Config, EmbedPooling, ModelsConfig};
pub use env::Environment;
pub use error::{ConfigError, DbError, Error, Result};
pub use qmd::{
    CollectionInfo, CollectionPath, CollectionRemoval, CollectionSettings, CollectionSpec,
    ContextEntry, LexOptions, Qmd, QmdBuilder, UpdateOptions, UpdateProgress, UpdateReport,
};
pub use snippet::{SnippetResult, add_line_numbers, extract_snippet};
pub use store::documents::extract_title;
pub use store::fts::{
    build_fts5_query, sanitize_fts5_term, validate_lex_query, validate_semantic_query,
};
pub use store::search::{SearchResult, SearchSource, get_docid};
