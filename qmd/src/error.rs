//! Error types for the qmd application.

use thiserror::Error;

/// Main error type for qmd operations.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum QmdError {
    /// Database error from rusqlite.
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// YAML parsing error.
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// JSON parsing error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Collection not found.
    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    /// Document not found.
    #[error("Document not found: {0}")]
    DocumentNotFound(String),

    /// Invalid path.
    #[error("Invalid path: {0}")]
    InvalidPath(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// General error with message.
    #[error("{0}")]
    General(String),
}

/// Result type alias for qmd operations.
pub type Result<T> = std::result::Result<T, QmdError>;
