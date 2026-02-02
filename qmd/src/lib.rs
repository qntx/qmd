//! QMD - Query Markdown Documents
//!
//! A full-text search tool for markdown files with collection management,
//! context annotations, and virtual path support.

pub mod cli;
pub mod collections;
pub mod config;
pub mod error;
pub mod formatter;
pub mod store;

pub use cli::{Cli, Commands};
pub use error::{QmdError, Result};
pub use store::Store;
