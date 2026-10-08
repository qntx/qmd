//! Injectable process environment.
//!
//! The library never reads process environment variables directly. Callers
//! that want upstream-compatible behavior construct [`Environment`] with
//! [`Environment::from_process`]; embedders and tests supply values
//! explicitly.

use std::path::PathBuf;

/// Values upstream `qmd` reads from the environment.
///
/// Empty strings are treated as unset, matching the upstream
/// `process.env.X || default` pattern.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct Environment {
    /// `$HOME`.
    pub home: Option<PathBuf>,
    /// `%USERPROFILE%` (Windows home fallback).
    pub userprofile: Option<PathBuf>,
    /// `$XDG_CONFIG_HOME`.
    pub xdg_config_home: Option<PathBuf>,
    /// `$XDG_CACHE_HOME`.
    pub xdg_cache_home: Option<PathBuf>,
    /// `$QMD_CONFIG_DIR` — overrides [`crate::paths::config_dir`].
    pub qmd_config_dir: Option<PathBuf>,
    /// `$INDEX_PATH` — overrides the default index location.
    pub index_path: Option<PathBuf>,
    /// `$QMD_SQLITE_BUSY_TIMEOUT` — raw string, parsed like upstream
    /// (`Number(...)`): non-finite or negative values fall back to the
    /// 120 s default.
    pub qmd_sqlite_busy_timeout: Option<String>,
    /// `$PWD` — used to resolve relative `--index` names, mirroring
    /// `process.env.PWD ?? process.cwd()`.
    pub pwd: Option<PathBuf>,
    /// `$QMD_EMBED_MODEL` — embed model URI override (llm.ts:303).
    pub qmd_embed_model: Option<String>,
    /// `$QMD_GENERATE_MODEL` — generation model URI override (llm.ts:307).
    pub qmd_generate_model: Option<String>,
    /// `$QMD_RERANK_MODEL` — reranker model URI override (llm.ts:311).
    pub qmd_rerank_model: Option<String>,
}

impl Environment {
    /// Capture the relevant process environment.
    #[must_use]
    pub fn from_process() -> Self {
        fn var_os(key: &str) -> Option<std::ffi::OsString> {
            std::env::var_os(key).filter(|v| !v.is_empty())
        }
        Self {
            home: var_os("HOME").map(PathBuf::from),
            userprofile: var_os("USERPROFILE").map(PathBuf::from),
            xdg_config_home: var_os("XDG_CONFIG_HOME").map(PathBuf::from),
            xdg_cache_home: var_os("XDG_CACHE_HOME").map(PathBuf::from),
            qmd_config_dir: var_os("QMD_CONFIG_DIR").map(PathBuf::from),
            index_path: var_os("INDEX_PATH").map(PathBuf::from),
            qmd_sqlite_busy_timeout: var_os("QMD_SQLITE_BUSY_TIMEOUT")
                .and_then(|v| v.into_string().ok()),
            pwd: var_os("PWD").map(PathBuf::from),
            qmd_embed_model: var_os("QMD_EMBED_MODEL").map(|v| v.to_string_lossy().into_owned()),
            qmd_generate_model: var_os("QMD_GENERATE_MODEL")
                .map(|v| v.to_string_lossy().into_owned()),
            qmd_rerank_model: var_os("QMD_RERANK_MODEL").map(|v| v.to_string_lossy().into_owned()),
        }
    }
}
