//! Error types returned by the public API.

use std::path::PathBuf;

/// Opaque wrapper over [`rusqlite::Error`].
///
/// Keeps the dependency's error type out of the public API so a `rusqlite`
/// upgrade is not a breaking change. The wrapped error is still reachable via
/// [`std::error::Error::source`].
#[derive(Debug)]
#[allow(
    clippy::error_impl_error,
    reason = "opaque wrapper named by convention for the crate's error family"
)]
pub struct DbError(rusqlite::Error);

impl DbError {
    pub(crate) const fn new(inner: rusqlite::Error) -> Self {
        Self(inner)
    }
}

impl From<rusqlite::Error> for DbError {
    fn from(inner: rusqlite::Error) -> Self {
        Self::new(inner)
    }
}

impl std::fmt::Display for DbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for DbError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

/// Opaque wrapper over a YAML parse/serialize failure.
#[derive(Debug)]
#[allow(
    clippy::error_impl_error,
    reason = "opaque wrapper named by convention for the crate's error family"
)]
pub struct ConfigError(serde_yaml_bw::Error);

impl From<serde_yaml_bw::Error> for ConfigError {
    fn from(inner: serde_yaml_bw::Error) -> Self {
        Self(inner)
    }
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

/// Errors returned by [`crate::Qmd`] and related functions.
#[derive(Debug, thiserror::Error)]
#[allow(
    clippy::error_impl_error,
    reason = "`qmd::Error` is the conventional exported error type name"
)]
#[non_exhaustive]
pub enum Error {
    /// Caller-supplied input is invalid.
    #[error("invalid input: {reason}")]
    InvalidInput {
        /// Human-readable description of what was invalid.
        reason: String,
    },

    /// A collection with this name does not exist.
    #[error("collection not found: {name}")]
    CollectionNotFound {
        /// The missing collection name.
        name: String,
    },

    /// A collection with this name already exists.
    #[error("collection '{name}' already exists")]
    CollectionExists {
        /// The conflicting collection name.
        name: String,
    },

    /// A different collection already covers the same path and pattern.
    #[error("a collection already exists for this path and pattern: '{existing}'")]
    DuplicateCollectionSource {
        /// Name of the existing collection covering the same source.
        existing: String,
    },

    /// The database file exists but was not created by this application
    /// (e.g. it is an upstream `qmd` index). Refusing to touch it protects
    /// the data from an accidental rebuild.
    #[error(
        "database at {} is not a qmd-rs index (foreign application_id); \
         refusing to modify it",
        path.display()
    )]
    ForeignIndex {
        /// Path of the offending database file.
        path: PathBuf,
    },

    /// Reading, writing or serializing the YAML configuration failed.
    #[error("config {}: {source}", path.display())]
    Config {
        /// Path of the config file involved.
        path: PathBuf,
        /// Underlying parse or serialize failure.
        source: ConfigError,
    },

    /// Filesystem I/O failed.
    #[error("I/O error on {}: {source}", path.display())]
    Io {
        /// Path involved.
        path: PathBuf,
        /// Underlying error.
        source: std::io::Error,
    },

    /// SQLite layer failure.
    #[error("database error: {0}")]
    Db(#[from] DbError),
}

/// Result alias for the public API.
pub type Result<T, E = Error> = std::result::Result<T, E>;
