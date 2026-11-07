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
    /// `$MEME_CONFIG_DIR` — overrides [`crate::paths::config_dir`].
    pub meme_config_dir: Option<PathBuf>,
    /// `$INDEX_PATH` — overrides the default index location.
    pub index_path: Option<PathBuf>,
    /// `$MEME_SQLITE_BUSY_TIMEOUT` — raw string, parsed like upstream
    /// (`Number(...)`): non-finite or negative values fall back to the
    /// 120 s default.
    pub meme_sqlite_busy_timeout: Option<String>,
    /// `$PWD` — used to resolve relative `--index` names, mirroring
    /// `process.env.PWD ?? process.cwd()`.
    pub pwd: Option<PathBuf>,
    /// `$MEME_EMBED_MODEL` — embed model URI override (llm.ts:303).
    pub meme_embed_model: Option<String>,
    /// `$MEME_GENERATE_MODEL` — generation model URI override (llm.ts:307).
    pub meme_generate_model: Option<String>,
    /// `$MEME_RERANK_MODEL` — reranker model URI override (llm.ts:311).
    pub meme_rerank_model: Option<String>,
    /// `$HF_ENDPOINT` — `HuggingFace` download endpoint override
    /// (node-llama-cpp `resolveHuggingFaceEndpoint`).
    pub hf_endpoint: Option<String>,
    /// `$MODEL_ENDPOINT` — model download endpoint override; wins over
    /// `HF_ENDPOINT` when both are set.
    pub model_endpoint: Option<String>,
    /// `$MEME_EMBED_CONTEXT_SIZE` — embedding context size (llm.ts:1346).
    pub meme_embed_context_size: Option<String>,
    /// `$MEME_RERANK_CONTEXT_SIZE` — rerank context size (llm.ts:1341).
    pub meme_rerank_context_size: Option<String>,
    /// `$MEME_EXPAND_CONTEXT_SIZE` — generation context size (llm.ts:660).
    pub meme_expand_context_size: Option<String>,
    /// `$MEME_EMBED_PARALLELISM` — worker-count override (llm.ts:671).
    pub meme_embed_parallelism: Option<String>,
    /// `$MEME_LLAMA_GPU` — GPU mode: `metal`/`vulkan`/`cuda`/`auto`/off
    /// (llm.ts:755).
    pub meme_llama_gpu: Option<String>,
    /// `$MEME_FORCE_CPU` — force CPU inference when truthy (llm.ts:756).
    pub meme_force_cpu: Option<String>,
    /// `$MEME_TRUST_LOCAL_CONFIG` — trust project-local config unattended
    /// when truthy; shared opt-in for the local-config gate
    /// (`isLocalConfigTrustOptedIn`, trust.ts:272-274).
    pub meme_trust_local_config: Option<String>,
    /// `$MEME_TRUST_UPDATE_HOOKS` — trust update hooks unattended when
    /// truthy; also feeds the shared `isLocalConfigTrustOptedIn` check, so
    /// it opts the path/model gate in too (trust.ts:272-274, 294).
    pub meme_trust_update_hooks: Option<String>,
    /// `$NO_COLOR` — presence disables colored output (qmd.ts:248).
    pub no_color: Option<String>,
    /// `$MEME_EDITOR_URI` — editor link template for clickable TTY
    /// output (qmd.ts:2436).
    pub meme_editor_uri: Option<String>,
    /// `$MEME_DOCTOR_DEVICE_PROBE` — `0`/`false`/`off`/`no`/`skip` skips
    /// `meme doctor`'s native backend probe (qmd.ts:4131-4136).
    pub meme_doctor_device_probe: Option<String>,
    /// `$CI` — upstream disables real LLM operations inside the
    /// `LlamaCpp` wrapper; `doctor` only lists it as an override.
    pub ci: Option<String>,
    /// `$MEME_SKILLS_DIR` — where bundled skills are discovered from;
    /// `doctor` lists it as an override.
    pub meme_skills_dir: Option<String>,
    /// `$MEME_METAL_KEEP_RESIDENCY` — opts back into libggml-metal
    /// residency sets on darwin (qmd.ts:3881).
    pub meme_metal_keep_residency: Option<String>,
    /// `$GGML_METAL_NO_RESIDENCY` — launcher-set Metal residency
    /// mitigation on darwin (qmd.ts:3882).
    pub ggml_metal_no_residency: Option<String>,
    /// `$MEME_WRAPPER_CAPTURE` — test/debug hook for the shell wrapper
    /// (qmd.ts:3886); `doctor` lists it as an override.
    pub meme_wrapper_capture: Option<String>,
    /// `$WSL_DISTRO_NAME` — enables WSL path heuristics upstream;
    /// `doctor` lists it as an override.
    pub wsl_distro_name: Option<String>,
    /// `$WSL_INTEROP` — enables WSL path heuristics upstream;
    /// `doctor` lists it as an override.
    pub wsl_interop: Option<String>,
    /// `$LD_LIBRARY_PATH` — searched by `meme doctor`'s Linux CUDA
    /// diagnostic (qmd.ts:4089-4092).
    pub ld_library_path: Option<String>,
    /// `$CUDA_PATH` — searched by `meme doctor`'s Linux CUDA diagnostic
    /// (qmd.ts:4089-4096).
    pub cuda_path: Option<String>,
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
            meme_config_dir: var_os("MEME_CONFIG_DIR").map(PathBuf::from),
            index_path: var_os("INDEX_PATH").map(PathBuf::from),
            meme_sqlite_busy_timeout: var_os("MEME_SQLITE_BUSY_TIMEOUT")
                .and_then(|v| v.into_string().ok()),
            pwd: var_os("PWD").map(PathBuf::from),
            meme_embed_model: var_os("MEME_EMBED_MODEL").map(|v| v.to_string_lossy().into_owned()),
            meme_generate_model: var_os("MEME_GENERATE_MODEL")
                .map(|v| v.to_string_lossy().into_owned()),
            meme_rerank_model: var_os("MEME_RERANK_MODEL")
                .map(|v| v.to_string_lossy().into_owned()),
            hf_endpoint: var_os("HF_ENDPOINT").map(|v| v.to_string_lossy().into_owned()),
            model_endpoint: var_os("MODEL_ENDPOINT").map(|v| v.to_string_lossy().into_owned()),
            meme_embed_context_size: var_os("MEME_EMBED_CONTEXT_SIZE")
                .map(|v| v.to_string_lossy().into_owned()),
            meme_rerank_context_size: var_os("MEME_RERANK_CONTEXT_SIZE")
                .map(|v| v.to_string_lossy().into_owned()),
            meme_expand_context_size: var_os("MEME_EXPAND_CONTEXT_SIZE")
                .map(|v| v.to_string_lossy().into_owned()),
            meme_embed_parallelism: var_os("MEME_EMBED_PARALLELISM")
                .map(|v| v.to_string_lossy().into_owned()),
            meme_llama_gpu: var_os("MEME_LLAMA_GPU").map(|v| v.to_string_lossy().into_owned()),
            meme_force_cpu: var_os("MEME_FORCE_CPU").map(|v| v.to_string_lossy().into_owned()),
            meme_trust_local_config: var_os("MEME_TRUST_LOCAL_CONFIG")
                .map(|v| v.to_string_lossy().into_owned()),
            meme_trust_update_hooks: var_os("MEME_TRUST_UPDATE_HOOKS")
                .map(|v| v.to_string_lossy().into_owned()),
            no_color: var_os("NO_COLOR").map(|v| v.to_string_lossy().into_owned()),
            meme_editor_uri: var_os("MEME_EDITOR_URI").map(|v| v.to_string_lossy().into_owned()),
            meme_doctor_device_probe: var_os("MEME_DOCTOR_DEVICE_PROBE")
                .map(|v| v.to_string_lossy().into_owned()),
            ci: var_os("CI").map(|v| v.to_string_lossy().into_owned()),
            meme_skills_dir: var_os("MEME_SKILLS_DIR").map(|v| v.to_string_lossy().into_owned()),
            meme_metal_keep_residency: var_os("MEME_METAL_KEEP_RESIDENCY")
                .map(|v| v.to_string_lossy().into_owned()),
            ggml_metal_no_residency: var_os("GGML_METAL_NO_RESIDENCY")
                .map(|v| v.to_string_lossy().into_owned()),
            meme_wrapper_capture: var_os("MEME_WRAPPER_CAPTURE")
                .map(|v| v.to_string_lossy().into_owned()),
            wsl_distro_name: var_os("WSL_DISTRO_NAME").map(|v| v.to_string_lossy().into_owned()),
            wsl_interop: var_os("WSL_INTEROP").map(|v| v.to_string_lossy().into_owned()),
            ld_library_path: var_os("LD_LIBRARY_PATH").map(|v| v.to_string_lossy().into_owned()),
            cuda_path: var_os("CUDA_PATH").map(|v| v.to_string_lossy().into_owned()),
        }
    }
}
