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
    /// `$HF_ENDPOINT` — `HuggingFace` download endpoint override
    /// (node-llama-cpp `resolveHuggingFaceEndpoint`).
    pub hf_endpoint: Option<String>,
    /// `$MODEL_ENDPOINT` — model download endpoint override; wins over
    /// `HF_ENDPOINT` when both are set.
    pub model_endpoint: Option<String>,
    /// `$QMD_EMBED_CONTEXT_SIZE` — embedding context size (llm.ts:1346).
    pub qmd_embed_context_size: Option<String>,
    /// `$QMD_RERANK_CONTEXT_SIZE` — rerank context size (llm.ts:1341).
    pub qmd_rerank_context_size: Option<String>,
    /// `$QMD_EXPAND_CONTEXT_SIZE` — generation context size (llm.ts:660).
    pub qmd_expand_context_size: Option<String>,
    /// `$QMD_EMBED_PARALLELISM` — worker-count override (llm.ts:671).
    pub qmd_embed_parallelism: Option<String>,
    /// `$QMD_LLAMA_GPU` — GPU mode: `metal`/`vulkan`/`cuda`/`auto`/off
    /// (llm.ts:755).
    pub qmd_llama_gpu: Option<String>,
    /// `$QMD_FORCE_CPU` — force CPU inference when truthy (llm.ts:756).
    pub qmd_force_cpu: Option<String>,
    /// `$QMD_TRUST_LOCAL_CONFIG` — trust project-local config unattended
    /// when truthy; shared opt-in for the local-config gate
    /// (`isLocalConfigTrustOptedIn`, trust.ts:272-274).
    pub qmd_trust_local_config: Option<String>,
    /// `$QMD_TRUST_UPDATE_HOOKS` — trust update hooks unattended when
    /// truthy; also feeds the shared `isLocalConfigTrustOptedIn` check, so
    /// it opts the path/model gate in too (trust.ts:272-274, 294).
    pub qmd_trust_update_hooks: Option<String>,
    /// `$NO_COLOR` — presence disables colored output (qmd.ts:248).
    pub no_color: Option<String>,
    /// `$QMD_EDITOR_URI` — editor link template for clickable TTY
    /// output (qmd.ts:2436).
    pub qmd_editor_uri: Option<String>,
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
            hf_endpoint: var_os("HF_ENDPOINT").map(|v| v.to_string_lossy().into_owned()),
            model_endpoint: var_os("MODEL_ENDPOINT").map(|v| v.to_string_lossy().into_owned()),
            qmd_embed_context_size: var_os("QMD_EMBED_CONTEXT_SIZE")
                .map(|v| v.to_string_lossy().into_owned()),
            qmd_rerank_context_size: var_os("QMD_RERANK_CONTEXT_SIZE")
                .map(|v| v.to_string_lossy().into_owned()),
            qmd_expand_context_size: var_os("QMD_EXPAND_CONTEXT_SIZE")
                .map(|v| v.to_string_lossy().into_owned()),
            qmd_embed_parallelism: var_os("QMD_EMBED_PARALLELISM")
                .map(|v| v.to_string_lossy().into_owned()),
            qmd_llama_gpu: var_os("QMD_LLAMA_GPU").map(|v| v.to_string_lossy().into_owned()),
            qmd_force_cpu: var_os("QMD_FORCE_CPU").map(|v| v.to_string_lossy().into_owned()),
            qmd_trust_local_config: var_os("QMD_TRUST_LOCAL_CONFIG")
                .map(|v| v.to_string_lossy().into_owned()),
            qmd_trust_update_hooks: var_os("QMD_TRUST_UPDATE_HOOKS")
                .map(|v| v.to_string_lossy().into_owned()),
            no_color: var_os("NO_COLOR").map(|v| v.to_string_lossy().into_owned()),
            qmd_editor_uri: var_os("QMD_EDITOR_URI").map(|v| v.to_string_lossy().into_owned()),
        }
    }
}
