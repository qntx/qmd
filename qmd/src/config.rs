//! Configuration constants and utilities.

/// Default glob pattern for markdown files.
pub const DEFAULT_GLOB: &str = "**/*.md";

/// Default maximum bytes for multi-get operations (10KB).
pub const DEFAULT_MULTI_GET_MAX_BYTES: usize = 10 * 1024;

/// Chunk size in tokens for embedding.
pub const CHUNK_SIZE_TOKENS: usize = 800;

/// Chunk overlap in tokens (15% of chunk size).
pub const CHUNK_OVERLAP_TOKENS: usize = CHUNK_SIZE_TOKENS * 15 / 100;

/// Chunk size in characters (approx 4 chars per token).
pub const CHUNK_SIZE_CHARS: usize = CHUNK_SIZE_TOKENS * 4;

/// Chunk overlap in characters.
pub const CHUNK_OVERLAP_CHARS: usize = CHUNK_OVERLAP_TOKENS * 4;

/// Directories to exclude from indexing.
pub const EXCLUDE_DIRS: &[&str] = &[
    "node_modules",
    ".git",
    ".cache",
    "vendor",
    "dist",
    "build",
    "target",
];

/// Get the default database path.
///
/// Returns `~/.cache/qmd/index.sqlite` on Unix-like systems.
pub fn get_default_db_path(index_name: &str) -> Option<std::path::PathBuf> {
    let cache_dir = dirs::cache_dir()?;
    let qmd_cache = cache_dir.join("qmd");
    std::fs::create_dir_all(&qmd_cache).ok()?;
    Some(qmd_cache.join(format!("{index_name}.sqlite")))
}

/// Get the default config directory.
///
/// Returns `~/.config/qmd` on Unix-like systems.
pub fn get_config_dir() -> Option<std::path::PathBuf> {
    if let Ok(dir) = std::env::var("QMD_CONFIG_DIR") {
        return Some(std::path::PathBuf::from(dir));
    }
    let config_dir = dirs::config_dir()?;
    Some(config_dir.join("qmd"))
}

/// Get the config file path for a given index name.
pub fn get_config_path(index_name: &str) -> Option<std::path::PathBuf> {
    let config_dir = get_config_dir()?;
    Some(config_dir.join(format!("{index_name}.yml")))
}

/// Get the model cache directory.
///
/// Returns `~/.cache/qmd/models` on Unix-like systems.
#[must_use]
pub fn get_model_cache_dir() -> std::path::PathBuf {
    let cache_dir = dirs::cache_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    let model_dir = cache_dir.join("qmd").join("models");
    let _ = std::fs::create_dir_all(&model_dir);
    model_dir
}
