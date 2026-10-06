//! On-device hybrid search for markdown: BM25, vectors and LLM reranking.
//!
//! Rust port of [tobi/qmd](https://github.com/tobi/qmd) (v2.8.3).
//!
//! # P1.1 surface
//!
//! Storage and configuration foundation: [`Qmd::builder`] opens a SQLite
//! index (WAL, busy timeout, schema versioning, sqlite-vec), configuration
//! comes from a YAML file, an inline [`Config`], or the database mirror.
//! Collection and context management follow upstream CLI semantics.
//! Indexing (P1.2) and search (P1.3) are not implemented yet.
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

pub mod config;
mod env;
mod error;
pub mod paths;
mod qmd;
mod store;

pub use config::{Collection, Config, EmbedPooling, ModelsConfig};
pub use env::Environment;
pub use error::{ConfigError, DbError, Error, Result};
pub use qmd::{
    CollectionInfo, CollectionPath, CollectionRemoval, CollectionSettings, CollectionSpec,
    ContextEntry, Qmd, QmdBuilder,
};
