//! llama.cpp inference backend (`llm::llamacpp`).
//!
//! One [`LlamaBackend`] behind a [`OnceLock`], and a per-model worker pool
//! per backend struct (T3/T4): [`LlamaContext`] is `!Send`, so each worker
//! thread owns its context for the whole pool lifetime. `LlamaModel` is
//! loaded once per backend struct and deliberately leaked to `'static` so
//! worker-held contexts never dangle; the mapping lives for the process.
//!
//! Semantics ported from upstream `src/llm.ts`:
//! - embed: `truncateToContextSize` (token limit `min(ctx, n_ctx_train)`,
//!   safe margin `limit - 4`), eval input tokenized with model special
//!   tokens added (BOS/EOS resolved per vocab rules), then `decode` in
//!   512-token ubatch chunks and `embeddings_seq_ith(0)` — upstream
//!   tail-only behavior for inputs past one ubatch.
//! - rerank: `tokenizer.chat_template.rerank` with `{query}`/`{document}`
//!   placeholders parsed for specials only in the template pieces,
//!   `pooling = Rank`, doc budget `ctx - 512 - queryTokens`, dedup of
//!   effective texts.
//! - generate: Qwen3 chat rendering (verified byte-identical to upstream's
//!   `QwenChatWrapper`), GBNF grammar, `top_k 20 / top_p 0.8 / temp 0.7`,
//!   max 600 generated tokens, fresh context per call.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::{LlamaBackendDevice, LlamaBackendDeviceType, LogOptions};

use crate::CancelToken;
use crate::env::Environment;
use crate::error::{Error, Result};
use crate::llm::{Embedder, Generator, InferenceError, Reranker};

/// Upstream `EMBED_CONTEXT_SIZE` default (`llm.ts:1346`).
pub const DEFAULT_EMBED_CONTEXT_SIZE: u32 = 2048;
/// Upstream `RERANK_CONTEXT_SIZE` default (`llm.ts:1341`).
pub const DEFAULT_RERANK_CONTEXT_SIZE: u32 = 4096;
/// Upstream `DEFAULT_EXPAND_CONTEXT_SIZE` (`llm.ts:660`).
pub const DEFAULT_EXPAND_CONTEXT_SIZE: u32 = 2048;
/// Upstream `RERANK_TEMPLATE_OVERHEAD` (`llm.ts:1737`).
pub const RERANK_TEMPLATE_OVERHEAD: u32 = 512;
/// Upstream `BASELINE_EMBED_CONTEXT_MB` (`llm.ts:698`).
pub const BASELINE_EMBED_CONTEXT_MB: u64 = 150;
/// Upstream `EMBED_POOL_RERANK_RESERVE_MB` (`llm.ts:703`).
pub const EMBED_POOL_RERANK_RESERVE_MB: u64 = 1000;
/// Upstream `RERANK_TARGET_DOCS_PER_CONTEXT` (`llm.ts:1738`).
pub const RERANK_TARGET_DOCS_PER_CONTEXT: usize = 10;

const UPSTREAM_UBATCH: u32 = 512;
const EXPANSION_MAX_TOKENS: usize = 600;
const EXPANSION_SYSTEM_PROMPT: &str = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something incorrectly. If you don't know the answer to a question, don't share false information.";

const EXPANSION_GRAMMAR: &str = r#"root ::= line+
line ::= type ": " content "\n"
type ::= "lex" | "vec" | "hyde"
content ::= [^\n]+
"#;

// ============================================================================
// Env parsing + parallelism (llm.ts:660-830, 1173-1205)
// ============================================================================

/// JS `parseInt(str, 10)`: sign + leading digit run, `None` on NaN.
fn js_parse_int(raw: &str) -> Option<i64> {
    let s = raw.trim_start();
    let (sign, digits) = s.strip_prefix('-').map_or_else(
        || (1i64, s.strip_prefix('+').unwrap_or(s)),
        |rest| (-1i64, rest),
    );
    let digits: String = digits.chars().take_while(char::is_ascii_digit).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse::<i64>().ok().map(|v| v * sign)
}

/// `QMD_*_CONTEXT_SIZE` parse: `parseInt` result `> 0`, else the default
/// (upstream `llm.ts:1341-1349`).
#[must_use]
pub fn context_size_env(raw: Option<&str>, default: u32) -> u32 {
    raw.and_then(js_parse_int)
        .and_then(|v| u32::try_from(v).ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

/// Upstream `resolveParallelismOverride` (`llm.ts:671-682`): strict
/// `Number()` parse, integer ≥ 1, capped at 8 — else automatic.
#[must_use]
pub fn resolve_parallelism_override(env_value: Option<&str>) -> Option<u32> {
    let normalized = env_value?.trim();
    if normalized.is_empty() {
        return None;
    }
    let parsed = normalized.parse::<f64>().ok()?;
    if parsed.fract() != 0.0 || parsed < 1.0 {
        return None;
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "checked integral and >= 1 above"
    )]
    Some((parsed as u32).min(8))
}

/// Upstream `LlamaGpuMode` (`llm.ts:662`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaGpuMode {
    /// Let llama.cpp pick (default device order).
    Auto,
    /// Restrict to Metal devices.
    Metal,
    /// Restrict to Vulkan devices.
    Vulkan,
    /// Restrict to CUDA devices.
    Cuda,
    /// `QMD_FORCE_CPU` / `QMD_LLAMA_GPU=off` — CPU only.
    Cpu,
}

const GPU_OFF_VALUES: &[&str] = &["false", "off", "none", "disable", "disabled", "0"];

/// Upstream `resolveLlamaGpuMode` (`llm.ts:755-769`).
#[must_use]
pub fn resolve_llama_gpu_mode(
    env_value: Option<&str>,
    force_cpu_value: Option<&str>,
) -> LlamaGpuMode {
    let force_cpu = force_cpu_value.unwrap_or("").trim().to_lowercase();
    if !force_cpu.is_empty() && !GPU_OFF_VALUES.contains(&force_cpu.as_str()) {
        return LlamaGpuMode::Cpu;
    }
    let normalized = env_value.unwrap_or("").trim().to_lowercase();
    if normalized.is_empty() {
        return LlamaGpuMode::Auto;
    }
    if GPU_OFF_VALUES.contains(&normalized.as_str()) {
        return LlamaGpuMode::Cpu;
    }
    match normalized.as_str() {
        "metal" => LlamaGpuMode::Metal,
        "vulkan" => LlamaGpuMode::Vulkan,
        "cuda" => LlamaGpuMode::Cuda,
        _ => LlamaGpuMode::Auto,
    }
}

/// Upstream `computeGpuContextPoolSize` (`llm.ts:712-722`): 25% of
/// `(free − reserve)` per context, clamped to `[1, cap]`.
#[must_use]
pub fn compute_gpu_context_pool_size(
    free_mb: u64,
    per_context_mb: u64,
    reserve_mb: u64,
    cap: u32,
) -> u32 {
    let usable = free_mb.saturating_sub(reserve_mb);
    let max_by_vram = (usable / 4) / per_context_mb.max(1);
    #[allow(
        clippy::cast_possible_truncation,
        reason = "pool size is small by construction"
    )]
    (max_by_vram.clamp(1, u64::from(cap)) as u32).max(1)
}

/// Upstream `estimateEmbedContextMB` (`llm.ts:734-753`): ≤350 MB weights
/// stay at the 150 MB baseline; larger models scale ~1.85× weights.
#[must_use]
pub fn estimate_embed_context_mb(model_bytes: u64, context_size: u32) -> u64 {
    let ctx_scale_num = u64::from(if context_size > 0 { context_size } else { 2048 });
    let scaled = |mb: u64| (mb * ctx_scale_num / 2048).max(1);
    let model_mb = model_bytes / (1024 * 1024);
    if model_mb <= 350 {
        scaled(BASELINE_EMBED_CONTEXT_MB)
    } else {
        scaled(model_mb * 185 / 100)
    }
}

/// Upstream `resolveSafeParallelism` (`llm.ts:685-696`): override wins;
/// Windows + CUDA serializes; otherwise `max(1, computed)`.
#[must_use]
pub fn resolve_safe_parallelism(
    windows_cuda: bool,
    computed: usize,
    override_: Option<u32>,
) -> usize {
    if let Some(n) = override_ {
        return usize::try_from(n).unwrap_or(1);
    }
    if windows_cuda {
        return 1;
    }
    computed.max(1)
}

fn cpu_math_cores() -> usize {
    std::thread::available_parallelism().map_or(4, std::num::NonZero::get)
}

/// Upstream `computeParallelism`'s CPU branch (`llm.ts:1187-1190`).
fn cpu_context_count() -> usize {
    (cpu_math_cores() / 4).clamp(1, 4)
}

/// Upstream `threadsPerContext` (`llm.ts:1198-1203`): `None` on GPU
/// (library decides), else `max(1, cores / parallelism)`.
fn threads_per_context(gpu: bool, parallelism: usize) -> Option<i32> {
    if gpu {
        return None;
    }
    let v = (cpu_math_cores() / parallelism.max(1)).max(1);
    Some(i32::try_from(v).unwrap_or(i32::MAX))
}

// ============================================================================
// Backend + devices
// ============================================================================

/// Upstream sets `GGML_METAL_NO_RESIDENCY=1` from its launcher before the
/// native binding loads, disabling Metal residency sets whose
/// static-destructor asserts at process exit while contexts/models are
/// still alive (ggml-org/llama.cpp#22593). We set it inside the `OnceLock`
/// init in [`backend`], so it lands before `llama_backend_init()` reads it
/// in `ggml_metal_device_init`. `QMD_METAL_KEEP_RESIDENCY=1` opts back in,
/// same as upstream.
///
/// This reads the process environment directly, deliberately bypassing
/// [`Environment`]: the residency decision is process-global (it lands
/// inside `llama_backend_init`, before any model context exists), matching
/// upstream's launcher-level toggle. Per-handle env values would race the
/// `OnceLock` — the first caller's value would win silently.
#[cfg(target_os = "macos")]
#[allow(
    unsafe_code,
    clippy::disallowed_methods,
    reason = "env write inside OnceLock init, before any llama.cpp call"
)]
fn disable_metal_residency_sets() {
    use std::ffi::OsStr;

    if std::env::var_os("QMD_METAL_KEEP_RESIDENCY").as_deref() == Some(OsStr::new("1")) {
        return;
    }
    // SAFETY: called only from `OnceLock::get_or_init` in `backend()`, which
    // runs once on one thread before the first llama.cpp call; nothing else
    // in this crate mutates the environment.
    unsafe { std::env::set_var("GGML_METAL_NO_RESIDENCY", "1") };
}

fn backend() -> Result<&'static LlamaBackend, InferenceError> {
    static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();
    let res = BACKEND.get_or_init(|| {
        #[cfg(target_os = "macos")]
        disable_metal_residency_sets();
        llama_cpp_2::send_logs_to_tracing(LogOptions::default());
        LlamaBackend::init().map_err(|e| e.to_string())
    });
    match res {
        Ok(b) => Ok(b),
        Err(e) => Err(InferenceError::Backend(e.clone())),
    }
}

const fn device_is_gpu(d: &LlamaBackendDevice) -> bool {
    matches!(
        d.device_type,
        LlamaBackendDeviceType::Gpu
            | LlamaBackendDeviceType::IntegratedGpu
            | LlamaBackendDeviceType::Accelerator
    )
}

/// Backend-name filter for explicit GPU modes (device names like
/// "Vulkan0" live in `name`; the `backend` field carries the backend kind).
fn backend_matches(d: &LlamaBackendDevice, mode: LlamaGpuMode) -> bool {
    match mode {
        LlamaGpuMode::Auto => device_is_gpu(d),
        LlamaGpuMode::Metal => d.backend == "MTL",
        LlamaGpuMode::Vulkan => d.backend.starts_with("Vulkan"),
        LlamaGpuMode::Cuda => d.backend.starts_with("CUDA"),
        LlamaGpuMode::Cpu => false,
    }
}

/// Device indices the GPU mode permits; `None` = let llama.cpp decide.
fn device_indices(mode: LlamaGpuMode) -> Option<Vec<usize>> {
    match mode {
        LlamaGpuMode::Cpu | LlamaGpuMode::Auto => return None,
        _ => {}
    }
    let idx: Vec<usize> = llama_cpp_2::list_llama_ggml_backend_devices()
        .iter()
        .filter(|d| backend_matches(d, mode))
        .map(|d| d.index)
        .collect();
    if idx.is_empty() { None } else { Some(idx) }
}

/// Free VRAM (MB) on the best GPU-class device for pool sizing, like
/// upstream `llama.getVramState().free`. `None` = CPU path.
fn gpu_free_mb(mode: LlamaGpuMode) -> Option<u64> {
    if matches!(mode, LlamaGpuMode::Cpu) {
        return None;
    }
    llama_cpp_2::list_llama_ggml_backend_devices()
        .iter()
        .filter(|d| backend_matches(d, mode) && device_is_gpu(d))
        .map(|d| u64::try_from(d.memory_free).unwrap_or(0) / (1024 * 1024))
        .max()
        .filter(|mb| *mb > 0)
}

fn model_params(mode: LlamaGpuMode) -> Result<LlamaModelParams, InferenceError> {
    let params = match mode {
        LlamaGpuMode::Cpu => LlamaModelParams::default().with_n_gpu_layers(0),
        _ => match device_indices(mode) {
            Some(idx) => LlamaModelParams::default()
                .with_devices(&idx)
                .map_err(|e| InferenceError::Backend(e.to_string()))?,
            None => LlamaModelParams::default(),
        },
    };
    Ok(params)
}

fn load_leaked_model(
    path: &Path,
    params: &LlamaModelParams,
) -> Result<&'static LlamaModel, InferenceError> {
    let model = LlamaModel::load_from_file(backend()?, path, params)
        .map_err(|e| InferenceError::Backend(e.to_string()))?;
    // The model must outlive every context created from it; workers hold
    // their contexts for the pool's lifetime, so the model is intentionally
    // leaked to 'static. The OS reclaims the mapping at process exit.
    Ok(Box::leak(Box::new(model)))
}

// ============================================================================
// Worker pool (T3/T4)
// ============================================================================

struct Job<R, O> {
    req: R,
    cancel: CancelToken,
    reply: mpsc::Sender<Result<O, InferenceError>>,
}

/// A fixed-size pool of worker threads; each owns its `W` for life.
/// Contexts stay on the thread that created them (`LlamaContext: !Send`).
struct Pool<R, O> {
    senders: Vec<mpsc::Sender<Job<R, O>>>,
}

impl<R, O> std::fmt::Debug for Pool<R, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool")
            .field("workers", &self.senders.len())
            .finish_non_exhaustive()
    }
}

impl<R: Send + 'static, O: Send + 'static> Pool<R, O> {
    /// Spawn `n` workers. Each runs `init()` for its owned state, then
    /// serves jobs via `handle(&mut worker, req, cancel)`.
    ///
    /// Upstream keeps whatever contexts it could create (reduced
    /// parallelism); only a zero-worker pool is an error here.
    fn spawn<Init, W, Handle>(n: usize, init: Init, handle: Handle) -> Result<Self, InferenceError>
    where
        Init: Fn() -> Result<W, String> + Send + Sync + 'static,
        // `W` is created inside the worker thread and never leaves it —
        // deliberately no `Send` bound: `LlamaContext` is `!Send` and that
        // is exactly why this pool exists.
        W: 'static,
        Handle: Fn(&mut W, R, &CancelToken) -> Result<O, InferenceError> + Send + Sync + 'static,
    {
        let init = Arc::new(init);
        let handle = Arc::new(handle);
        let (report_tx, report_rx) = mpsc::channel::<(usize, Result<(), String>)>();
        let mut txs = Vec::with_capacity(n);
        for i in 0..n {
            let (tx, rx) = mpsc::channel::<Job<R, O>>();
            txs.push(tx);
            let init = Arc::clone(&init);
            let handle = Arc::clone(&handle);
            let report_tx = report_tx.clone();
            std::thread::spawn(move || {
                worker_thread(i, init.as_ref(), handle.as_ref(), &rx, &report_tx);
            });
        }
        drop(report_tx);
        let mut senders = Vec::with_capacity(n);
        let mut last_err = None;
        for (i, r) in report_rx.iter().take(n) {
            match (r, txs.get(i)) {
                (Ok(()), Some(tx)) => senders.push(tx.clone()),
                (Ok(()), None) => {}
                (Err(e), _) => last_err = Some(e),
            }
        }
        if senders.is_empty() {
            return Err(InferenceError::Backend(format!(
                "failed to create any worker contexts: {}",
                last_err.unwrap_or_else(|| "no workers started".to_owned())
            )));
        }
        Ok(Self { senders })
    }

    const fn worker_count(&self) -> usize {
        self.senders.len()
    }

    /// Dispatch `reqs[i]` to worker `i % workers` and collect results in
    /// submission order (upstream's contiguous-chunk fan-out: callers chunk
    /// so `reqs.len() <= workers`).
    fn dispatch_each(&self, reqs: Vec<R>, cancel: &CancelToken) -> Vec<Result<O, InferenceError>> {
        let n_workers = self.senders.len();
        let mut rxs = Vec::with_capacity(reqs.len());
        for (i, req) in reqs.into_iter().enumerate() {
            let (tx, rx) = mpsc::channel();
            let sent = self.senders.get(i % n_workers).is_some_and(|s| {
                s.send(Job {
                    req,
                    cancel: cancel.clone(),
                    reply: tx,
                })
                .is_ok()
            });
            rxs.push(if sent { Some(rx) } else { None });
        }
        rxs.into_iter()
            .map(|rx| {
                rx.and_then(|rx| rx.recv().ok())
                    .unwrap_or_else(|| Err(InferenceError::Backend("worker exited".to_owned())))
            })
            .collect()
    }

    /// Dispatch a single request (to worker 0 — single-worker pools only).
    fn dispatch(&self, req: R, cancel: &CancelToken) -> Result<O, InferenceError> {
        self.dispatch_each(vec![req], cancel)
            .into_iter()
            .next()
            .unwrap_or_else(|| Err(InferenceError::Backend("worker exited".to_owned())))
    }
}

/// Worker thread body: build the worker state, report init status, then
/// serve jobs until the pool's senders drop.
fn worker_thread<W, R, O, Init, H>(
    i: usize,
    init: &Init,
    handle: &H,
    rx: &mpsc::Receiver<Job<R, O>>,
    report: &mpsc::Sender<(usize, Result<(), String>)>,
) where
    Init: Fn() -> Result<W, String>,
    H: Fn(&mut W, R, &CancelToken) -> Result<O, InferenceError>,
{
    let worker = init();
    let status = worker.as_ref().map(|_| ()).map_err(Clone::clone);
    drop(report.send((i, status)));
    let Ok(mut worker) = worker else { return };
    while let Ok(job) = rx.recv() {
        let out = handle(&mut worker, job.req, &job.cancel);
        drop(job.reply.send(out));
    }
}

// ============================================================================
// decode helpers
// ============================================================================

/// Decode `tokens` in `UPSTREAM_UBATCH`-sized calls, mirroring
/// node-llama-cpp splitting queued decodes at `batchSize = 512`
/// (non-causal models assert `n_ubatch >= n_tokens` per call). Positions
/// stay absolute across calls; `embeddings_seq_ith` reads only the last
/// call's sequence — upstream tail-only behavior for inputs over one
/// ubatch.
fn decode_all(
    ctx: &mut LlamaContext<'_>,
    tokens: &[LlamaToken],
    logits_last: bool,
) -> Result<(), InferenceError> {
    ctx.clear_kv_cache();
    let ubatch = usize::try_from(UPSTREAM_UBATCH).unwrap_or(usize::MAX);
    let mut pos = 0i32;
    for chunk in tokens.chunks(ubatch) {
        let mut batch = LlamaBatch::new(chunk.len(), 1);
        let last = chunk.len().saturating_sub(1);
        for (i, t) in chunk.iter().enumerate() {
            let offset = i32::try_from(i)
                .map_err(|_| InferenceError::Backend("position overflow".to_owned()))?;
            batch
                .add(*t, pos + offset, &[0], logits_last && i == last)
                .map_err(|e| InferenceError::Backend(e.to_string()))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| InferenceError::Backend(e.to_string()))?;
        pos += i32::try_from(chunk.len())
            .map_err(|_| InferenceError::Backend("position overflow".to_owned()))?;
    }
    Ok(())
}

fn seq_vector(ctx: &LlamaContext<'_>) -> Result<Vec<f32>, InferenceError> {
    ctx.embeddings_seq_ith(0)
        .map(<[f32]>::to_vec)
        .map_err(|e| InferenceError::Backend(e.to_string()))
}

/// Evaluate one text on an embedding/ranking context and read the
/// sequence vector (`embeddings_seq_ith(0)`).
fn eval_seq_vector(
    ctx: &mut LlamaContext<'_>,
    tokens: &[LlamaToken],
) -> Result<Vec<f32>, InferenceError> {
    decode_all(ctx, tokens, false)?;
    seq_vector(ctx)
}

fn token_ids(tokens: &[LlamaToken]) -> Result<Vec<u32>, InferenceError> {
    tokens
        .iter()
        .map(|t| {
            u32::try_from(t.0)
                .map_err(|_| InferenceError::Backend(format!("negative token id {}", t.0)))
        })
        .collect()
}

// ============================================================================
// LlamaEmbedder
// ============================================================================

/// How the embedder treats inputs that exceed one ubatch (T8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbeddingMode {
    /// Decode in 512-token ubatches and read the last decode's sequence
    /// vector — upstream's exact behavior (documents over 512 tokens embed
    /// their tail only).
    #[default]
    UpstreamTail,
    /// One decode call covering the whole input (`n_ubatch = n_ctx`):
    /// the full sequence's pooled vector. Vectors for >512-token inputs
    /// differ from upstream — the fingerprint carries an extra line.
    FullSequence,
}

impl From<crate::config::EmbedPooling> for EmbeddingMode {
    fn from(p: crate::config::EmbedPooling) -> Self {
        match p {
            crate::config::EmbedPooling::UpstreamTail => Self::UpstreamTail,
            crate::config::EmbedPooling::FullSequence => Self::FullSequence,
        }
    }
}

/// Embed `texts` on one worker context, checking `cancel` between items
/// (upstream aborts a cancelled batch mid-flight too).
fn embed_batch(
    ctx: &mut LlamaContext<'_>,
    texts: &[String],
    cancel: &CancelToken,
) -> Result<Vec<Vec<f32>>, InferenceError> {
    let vocab = ctx.model.vocab();
    let mut out = Vec::with_capacity(texts.len());
    for text in texts {
        if cancel.is_cancelled() {
            return Err(InferenceError::Cancelled);
        }
        // `add_special = true`: llama.cpp resolves the model's
        // BOS/EOS-per-vocab rules, matching upstream's eval-input
        // construction (P2.2 verification ④).
        let tokens = vocab.tokenize(text.as_bytes(), true, false);
        out.push(eval_seq_vector(ctx, &tokens)?);
    }
    Ok(out)
}

/// llama.cpp embedding backend — upstream `LlamaCpp.embed`/`embedBatch`.
pub struct LlamaEmbedder {
    uri: String,
    model: &'static LlamaModel,
    pool: Pool<Vec<String>, Vec<Vec<f32>>>,
    token_limit: usize,
    mode: EmbeddingMode,
}

impl std::fmt::Debug for LlamaEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaEmbedder")
            .field("uri", &self.uri)
            .field("token_limit", &self.token_limit)
            .field("mode", &self.mode)
            .field("pool", &self.pool)
            .finish_non_exhaustive()
    }
}

/// Pool sizing shared by embed (`reserve_mb` = 1000, `cap` = 8) and rerank
/// (`per_ctx_mb` = 1000, reserve 0, then `min(.., 4)` at the caller).
fn pool_size(
    gpu_mode: LlamaGpuMode,
    per_ctx_mb: u64,
    reserve_mb: u64,
    cap: u32,
    env: &Environment,
) -> usize {
    let override_ = resolve_parallelism_override(env.qmd_embed_parallelism.as_deref());
    let computed = gpu_free_mb(gpu_mode).map_or_else(cpu_context_count, |free| {
        usize::try_from(compute_gpu_context_pool_size(
            free, per_ctx_mb, reserve_mb, cap,
        ))
        .unwrap_or(usize::MAX)
    });
    let windows_cuda = cfg!(windows) && matches!(gpu_mode, LlamaGpuMode::Cuda);
    resolve_safe_parallelism(windows_cuda, computed, override_)
}

impl LlamaEmbedder {
    /// Load `path` (a GGUF file, e.g. from [`crate::llm::download`]) and
    /// build the worker pool from `env`.
    ///
    /// # Errors
    /// [`Error::Model`] when the file fails to load; [`Error::Inference`]
    /// when no embedding context can be created.
    pub fn new(
        path: &Path,
        model_uri: &str,
        env: &Environment,
        mode: EmbeddingMode,
    ) -> Result<Self> {
        let gpu_mode =
            resolve_llama_gpu_mode(env.qmd_llama_gpu.as_deref(), env.qmd_force_cpu.as_deref());
        let ctx_size = context_size_env(
            env.qmd_embed_context_size.as_deref(),
            DEFAULT_EMBED_CONTEXT_SIZE,
        );
        let model =
            load_leaked_model(path, &model_params(gpu_mode)?).map_err(|source| Error::Model {
                uri: model_uri.to_owned(),
                source,
            })?;
        // Upstream resolveEmbedTokenLimit: max(1, min(ctx, train_ctx)).
        let trained = model.n_ctx_train();
        let token_limit = usize::try_from(if trained > 0 {
            ctx_size.min(trained)
        } else {
            ctx_size
        })
        .unwrap_or(usize::MAX)
        .max(1);
        let gpu = gpu_free_mb(gpu_mode).is_some();
        let model_bytes = std::fs::metadata(path).map_or(0, |m| m.len());
        let per_ctx_mb = estimate_embed_context_mb(model_bytes, ctx_size);
        let n = pool_size(gpu_mode, per_ctx_mb, EMBED_POOL_RERANK_RESERVE_MB, 8, env);
        let threads = threads_per_context(gpu, n);
        let n_ubatch = match mode {
            EmbeddingMode::UpstreamTail => UPSTREAM_UBATCH,
            EmbeddingMode::FullSequence => ctx_size,
        };
        let pool = Pool::spawn(
            n,
            move || {
                let mut params = LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(ctx_size))
                    .with_n_batch(n_ubatch)
                    .with_n_ubatch(n_ubatch)
                    .with_embeddings(true);
                if let Some(t) = threads {
                    params = params.with_n_threads(t);
                }
                model
                    .new_context(backend().map_err(|e| e.to_string())?, params)
                    .map_err(|e| e.to_string())
            },
            |ctx: &mut LlamaContext<'_>, texts: Vec<String>, cancel: &CancelToken| {
                embed_batch(ctx, &texts, cancel)
            },
        )?;
        Ok(Self {
            uri: model_uri.to_owned(),
            model,
            pool,
            token_limit,
            mode,
        })
    }

    /// Number of worker contexts in the pool.
    #[must_use]
    pub const fn worker_count(&self) -> usize {
        self.pool.worker_count()
    }

    /// Upstream `truncateToContextSize`: tokenize without specials; if
    /// over limit, detokenize `tokens[..limit-4]` and re-embed that text.
    fn truncate_to_context_size(&self, text: &str) -> String {
        let vocab = self.model.vocab();
        let tokens = vocab.tokenize(text.as_bytes(), false, false);
        if tokens.len() <= self.token_limit {
            return text.to_owned();
        }
        let safe = self.token_limit.saturating_sub(4).max(1);
        let cut = tokens.get(..safe).unwrap_or(&tokens);
        // Upstream logs `⚠ Text truncated to fit embedding context (N tokens)`.
        tracing::warn!(
            limit = self.token_limit,
            tokens = tokens.len(),
            "text truncated to fit embedding context"
        );
        String::from_utf8_lossy(&vocab.detokenize(cut, true, false)).into_owned()
    }
}

impl Embedder for LlamaEmbedder {
    fn model_id(&self) -> &str {
        &self.uri
    }

    fn fingerprint_extra(&self) -> Option<&str> {
        match self.mode {
            // FullSequence vectors differ from upstream for >512-token
            // inputs, so the vector identity must differ too — the
            // canonical extra string is `EmbedPooling`'s (§8.3).
            EmbeddingMode::FullSequence => {
                crate::config::EmbedPooling::FullSequence.fingerprint_extra()
            }
            EmbeddingMode::UpstreamTail => None,
        }
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, InferenceError> {
        let toks = self.model.vocab().tokenize(text.as_bytes(), false, false);
        token_ids(&toks)
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, InferenceError> {
        let toks: Vec<LlamaToken> = tokens
            .iter()
            .map(|t| LlamaToken(i32::try_from(*t).unwrap_or(i32::MAX)))
            .collect();
        Ok(
            String::from_utf8_lossy(&self.model.vocab().detokenize(&toks, true, false))
                .into_owned(),
        )
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, InferenceError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let safe: Vec<String> = texts
            .iter()
            .map(|t| self.truncate_to_context_size(t))
            .collect();
        // Upstream splits the batch into contiguous chunks, one per
        // context (llm.ts embedBatch chunkSize = ceil(len / contexts)).
        let n_chunks = self.pool.worker_count().min(safe.len());
        let chunk_size = safe.len().div_ceil(n_chunks);
        let chunks: Vec<Vec<String>> = safe.chunks(chunk_size).map(<[String]>::to_vec).collect();
        let results = self.pool.dispatch_each(chunks, &CancelToken::new());
        let mut out = Vec::with_capacity(texts.len());
        for r in results {
            out.extend(r?);
        }
        Ok(out)
    }
}

/// Lazily-built [`LlamaEmbedder`] — upstream resolves and loads the
/// model inside the first inference call (`LlamaCpp::resolveModel`,
/// llm.ts:1128-1133), not when the session handle is produced.
///
/// With the `llama` feature [`crate::Qmd::embedder`] returns this cheap
/// handle immediately; the GGUF download, model load and worker-pool
/// spin-up happen on the first `tokenize`/`embed` call, and a failed
/// build leaves the slot empty so the next call retries (upstream
/// clears the load promise on rejection).
pub struct LazyLlamaEmbedder {
    model_uri: String,
    env: Environment,
    mode: EmbeddingMode,
    inner: Mutex<Option<Arc<LlamaEmbedder>>>,
}

impl LazyLlamaEmbedder {
    /// Wrap the embedder for `model_uri` (`hf:` or local path) without
    /// touching the network or the model file. `mode` is the
    /// configured pooling (`upstream_tail` default).
    #[must_use]
    pub const fn new(model_uri: String, env: Environment, mode: EmbeddingMode) -> Self {
        Self {
            model_uri,
            env,
            mode,
            inner: Mutex::new(None),
        }
    }

    #[allow(
        clippy::significant_drop_tightening,
        reason = "the init lock must stay held across check-and-fill"
    )]
    fn get(&self) -> Result<Arc<LlamaEmbedder>> {
        let mut slot = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(e) = slot.as_ref() {
            return Ok(Arc::clone(e));
        }
        let pulled = super::download::pull_models(
            std::slice::from_ref(&self.model_uri),
            &self.env,
            None,
            false,
        )?;
        let path = pulled
            .first()
            .map(|r| r.path.clone())
            .ok_or_else(|| Error::InvalidInput {
                reason: format!("model resolution returned no path for {}", self.model_uri),
            })?;
        let built = Arc::new(LlamaEmbedder::new(
            &path,
            &self.model_uri,
            &self.env,
            self.mode,
        )?);
        *slot = Some(Arc::clone(&built));
        Ok(built)
    }

    fn delegate<T>(
        &self,
        f: impl FnOnce(&LlamaEmbedder) -> Result<T, InferenceError>,
    ) -> Result<T, InferenceError> {
        self.get()
            .map_err(|e| InferenceError::Backend(e.to_string()))
            .and_then(|e| f(&e))
    }
}

impl std::fmt::Debug for LazyLlamaEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyLlamaEmbedder")
            .field("model_uri", &self.model_uri)
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

impl Embedder for LazyLlamaEmbedder {
    fn model_id(&self) -> &str {
        &self.model_uri
    }

    fn fingerprint_extra(&self) -> Option<&str> {
        match self.mode {
            EmbeddingMode::FullSequence => {
                crate::config::EmbedPooling::FullSequence.fingerprint_extra()
            }
            EmbeddingMode::UpstreamTail => None,
        }
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, InferenceError> {
        self.delegate(|e| e.tokenize(text))
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, InferenceError> {
        self.delegate(|e| e.detokenize(tokens))
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, InferenceError> {
        self.delegate(|e| e.embed(texts))
    }
}

// ============================================================================
// LlamaReranker
// ============================================================================

/// Tokenize a rerank evaluation input like upstream
/// `LlamaRankingContext._getEvaluationInput`: template literal pieces are
/// parsed for special tokens; `{query}`/`{document}` substitutions are not.
#[allow(
    clippy::literal_string_with_formatting_args,
    reason = "the braces are GGUF template placeholders, not format args"
)]
fn rerank_tokens(
    vocab: &llama_cpp_2::vocab::LlamaVocab<'_>,
    template: &str,
    query: &str,
    doc: &str,
) -> Vec<LlamaToken> {
    let mut out = Vec::new();
    let mut rest = template;
    loop {
        let q_pos = rest.find("{query}");
        let d_pos = rest.find("{document}");
        let next = match (q_pos, d_pos) {
            (None, None) => break,
            (Some(q), Some(d)) => q.min(d),
            (Some(q), None) => q,
            (None, Some(d)) => d,
        };
        out.extend(vocab.tokenize(rest.get(..next).unwrap_or("").as_bytes(), false, true));
        let after = rest.get(next..).unwrap_or("");
        let (placeholder, value) = if after.starts_with("{query}") {
            ("{query}".len(), query)
        } else {
            ("{document}".len(), doc)
        };
        out.extend(vocab.tokenize(value.as_bytes(), false, false));
        rest = after.get(placeholder..).unwrap_or("");
    }
    out.extend(vocab.tokenize(rest.as_bytes(), false, true));
    out
}

struct RerankJob {
    query: Arc<str>,
    texts: Vec<String>,
}

/// JS `tokens.slice(0, k)`: `k >= 0` keeps the first `k` items; `k < 0`
/// drops `|k|` items from the end (clamped at 0). Upstream's rerank
/// truncation depends on the negative-index arm when the query alone
/// already overflows the context budget.
fn slice_head(len: usize, k: i64) -> usize {
    if k >= 0 {
        usize::try_from(k).unwrap_or(usize::MAX).min(len)
    } else {
        len.saturating_sub(usize::try_from(-k).unwrap_or(usize::MAX))
    }
}

/// Map unique-text scores back onto the original document order.
fn scatter_scores(scores: &mut [f32], groups: &[Vec<usize>], unique_scores: &[f32]) {
    for (u, group) in groups.iter().enumerate() {
        let score = unique_scores.get(u).copied().unwrap_or(0.0);
        for &i in group {
            if let Some(s) = scores.get_mut(i) {
                *s = score;
            }
        }
    }
}

/// Score `job.texts` against `job.query` on one ranking context.
fn rerank_batch(
    ctx: &mut LlamaContext<'_>,
    template: &str,
    job: &RerankJob,
    cancel: &CancelToken,
) -> Result<Vec<f32>, InferenceError> {
    let vocab = ctx.model.vocab();
    let mut scores = Vec::with_capacity(job.texts.len());
    for text in &job.texts {
        if cancel.is_cancelled() {
            return Err(InferenceError::Cancelled);
        }
        let tokens = rerank_tokens(&vocab, template, &job.query, text);
        let emb = eval_seq_vector(ctx, &tokens)?;
        scores.push(emb.first().copied().unwrap_or(0.0));
    }
    Ok(scores)
}

/// llama.cpp reranker — upstream `LlamaRankingContext` + `rerank`.
pub struct LlamaReranker {
    uri: String,
    model: &'static LlamaModel,
    pool: Pool<RerankJob, Vec<f32>>,
    ctx_size: u32,
}

impl std::fmt::Debug for LlamaReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaReranker")
            .field("uri", &self.uri)
            .field("ctx_size", &self.ctx_size)
            .field("pool", &self.pool)
            .finish_non_exhaustive()
    }
}

impl LlamaReranker {
    /// Load `path` and the model's `tokenizer.chat_template.rerank`
    /// metadata; build the context pool (`min(pool, 4)` upstream).
    ///
    /// # Errors
    /// [`Error::Model`] on load/template failure; [`Error::Inference`]
    /// when no ranking context can be created.
    pub fn new(path: &Path, model_uri: &str, env: &Environment) -> Result<Self> {
        let gpu_mode =
            resolve_llama_gpu_mode(env.qmd_llama_gpu.as_deref(), env.qmd_force_cpu.as_deref());
        let ctx_size = context_size_env(
            env.qmd_rerank_context_size.as_deref(),
            DEFAULT_RERANK_CONTEXT_SIZE,
        );
        let model =
            load_leaked_model(path, &model_params(gpu_mode)?).map_err(|source| Error::Model {
                uri: model_uri.to_owned(),
                source,
            })?;
        let template: Arc<str> = model
            .meta_val_str("tokenizer.chat_template.rerank")
            .map_err(|e| Error::Model {
                uri: model_uri.to_owned(),
                source: InferenceError::Backend(format!("tokenizer.chat_template.rerank: {e}")),
            })?
            .into();
        let gpu = gpu_free_mb(gpu_mode).is_some();
        let n = pool_size(gpu_mode, 1000, 0, 8, env).min(4);
        let threads = threads_per_context(gpu, n);
        let worker_template = Arc::clone(&template);
        let pool = Pool::spawn(
            n,
            move || {
                let mut params = LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(ctx_size))
                    .with_n_batch(ctx_size)
                    .with_n_ubatch(ctx_size)
                    .with_embeddings(true)
                    .with_pooling_type(LlamaPoolingType::Rank);
                if let Some(t) = threads {
                    params = params.with_n_threads(t);
                }
                model
                    .new_context(backend().map_err(|e| e.to_string())?, params)
                    .map_err(|e| e.to_string())
            },
            move |ctx: &mut LlamaContext<'_>, job: RerankJob, cancel: &CancelToken| {
                rerank_batch(ctx, &worker_template, &job, cancel)
            },
        )?;
        Ok(Self {
            uri: model_uri.to_owned(),
            model,
            pool,
            ctx_size,
        })
    }
}

impl Reranker for LlamaReranker {
    fn model_id(&self) -> &str {
        &self.uri
    }

    fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>, InferenceError> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        // Budget = ctx - template overhead - query tokens (llm.ts:1754).
        let vocab = self.model.vocab();
        let query_tokens =
            i64::try_from(vocab.tokenize(query.as_bytes(), false, false).len()).unwrap_or(i64::MAX);
        let budget = i64::from(self.ctx_size)
            .saturating_sub(i64::from(RERANK_TEMPLATE_OVERHEAD))
            .saturating_sub(query_tokens);
        let effective: Vec<String> = docs
            .iter()
            .map(|doc| {
                let tokens = vocab.tokenize(doc.as_bytes(), false, false);
                if i64::try_from(tokens.len()).unwrap_or(i64::MAX) <= budget {
                    return (*doc).to_owned();
                }
                // `tokens.slice(0, maxDocTokens)` upstream — negative budget
                // counts back from the end via `slice_head`.
                let keep = slice_head(tokens.len(), budget);
                let cut = tokens.get(..keep).unwrap_or(&[]);
                String::from_utf8_lossy(&vocab.detokenize(cut, true, false)).into_owned()
            })
            .collect();

        // Deduplicate identical effective texts (upstream textToDocs);
        // first-seen order is preserved.
        let mut index_of: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        let mut unique: Vec<&str> = Vec::new();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (i, text) in effective.iter().enumerate() {
            if let Some(g) = index_of.get(text.as_str()).and_then(|&u| groups.get_mut(u)) {
                g.push(i);
            } else {
                index_of.insert(text.as_str(), unique.len());
                unique.push(text.as_str());
                groups.push(vec![i]);
            }
        }

        // Fan out contiguous chunks across contexts, sized to roughly
        // RERANK_TARGET_DOCS_PER_CONTEXT docs (llm.ts:1762-1780).
        let n_active = unique
            .len()
            .div_ceil(RERANK_TARGET_DOCS_PER_CONTEXT)
            .clamp(1, self.pool.worker_count());
        let chunk_size = unique.len().div_ceil(n_active);
        let query: Arc<str> = query.into();
        let jobs: Vec<RerankJob> = unique
            .chunks(chunk_size)
            .map(|c| RerankJob {
                query: Arc::clone(&query),
                texts: c.iter().map(|s| (*s).to_owned()).collect(),
            })
            .collect();
        let chunk_results = self.pool.dispatch_each(jobs, &CancelToken::new());
        let mut unique_scores = Vec::with_capacity(unique.len());
        for r in chunk_results {
            unique_scores.extend(r?);
        }
        let mut scores = vec![0.0f32; docs.len()];
        scatter_scores(&mut scores, &groups, &unique_scores);
        Ok(scores)
    }
}

// ============================================================================
// LlamaGenerator
// ============================================================================

struct GenJob {
    query: String,
    seed: u32,
}

/// Render the expansion prompt via the model's own chat template —
/// verified byte-identical to upstream's `QwenChatWrapper` output (P2.2
/// verification ③).
fn render_expansion_prompt(model: &LlamaModel, query: &str) -> Result<String, InferenceError> {
    let tmpl = model
        .chat_template(None)
        .map_err(|e| InferenceError::Backend(e.to_string()))?;
    model
        .apply_chat_template(
            &tmpl,
            &[
                LlamaChatMessage::new("system".to_owned(), EXPANSION_SYSTEM_PROMPT.to_owned())
                    .map_err(|e| InferenceError::Backend(e.to_string()))?,
                LlamaChatMessage::new(
                    "user".to_owned(),
                    format!("/no_think Expand this search query: {query}"),
                )
                .map_err(|e| InferenceError::Backend(e.to_string()))?,
            ],
            true,
        )
        .map_err(|e| InferenceError::Backend(e.to_string()))
}

/// One expansion call: fresh context, chat rendering, grammar-constrained
/// sampling until EOG or `EXPANSION_MAX_TOKENS`.
fn run_expansion(
    model: &LlamaModel,
    expand_ctx: u32,
    job: &GenJob,
) -> Result<String, InferenceError> {
    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(expand_ctx))
        .with_n_batch(expand_ctx);
    let mut ctx = model
        .new_context(backend()?, params)
        .map_err(|e| InferenceError::Backend(e.to_string()))?;
    let vocab = model.vocab();
    let rendered = render_expansion_prompt(model, &job.query)?;
    let prompt_tokens = vocab.tokenize(rendered.as_bytes(), false, true);
    decode_all(&mut ctx, &prompt_tokens, true)?;
    // Upstream's presencePenalty is a no-op (AddonSampler only builds the
    // penalties sampler when penalty != 1, and qmd never sets it), so the
    // chain is grammar → top_k → top_p → temp → dist.
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::grammar(model, EXPANSION_GRAMMAR, "root")
            .map_err(|e| InferenceError::Backend(e.to_string()))?,
        LlamaSampler::top_k(20),
        LlamaSampler::top_p(0.8, 1),
        LlamaSampler::temp(0.7),
        LlamaSampler::dist(job.seed),
    ]);
    let start = i32::try_from(prompt_tokens.len())
        .map_err(|_| InferenceError::Backend("position overflow".to_owned()))?;
    let end = start + i32::try_from(EXPANSION_MAX_TOKENS).unwrap_or(600);
    // Accumulate token ids and detokenize once at the end — per-token
    // `token_to_piece` splits multi-byte UTF-8 chars across tokens and
    // produces replacement characters (upstream detokenizes the stream).
    let mut out: Vec<LlamaToken> = Vec::new();
    for pos in start..end {
        let tok = sampler.sample(&ctx, -1);
        if vocab.is_eog(tok) {
            break;
        }
        out.push(tok);
        let mut batch = LlamaBatch::new(1, 1);
        batch
            .add(tok, pos, &[0], true)
            .map_err(|e| InferenceError::Backend(e.to_string()))?;
        ctx.decode(&mut batch)
            .map_err(|e| InferenceError::Backend(e.to_string()))?;
    }
    Ok(String::from_utf8_lossy(&vocab.detokenize(&out, true, false)).into_owned())
}

/// llama.cpp query-expansion backend — upstream `expandQueryWithLlm`.
///
/// Returns the raw model output (`lex:`/`vec:`/`hyde:` lines); parsing
/// lives in the core layer.
pub struct LlamaGenerator {
    uri: String,
    pool: Pool<GenJob, String>,
    seed: AtomicU32,
}

impl std::fmt::Debug for LlamaGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaGenerator")
            .field("uri", &self.uri)
            .field("pool", &self.pool)
            .finish_non_exhaustive()
    }
}

impl LlamaGenerator {
    /// Load `path`; one worker owns the model and builds a fresh context
    /// per call, like upstream's per-call `LlamaContext`.
    ///
    /// # Errors
    /// [`Error::Model`] on load failure; [`Error::Inference`] when the
    /// worker cannot start.
    pub fn new(path: &Path, model_uri: &str, env: &Environment) -> Result<Self> {
        let gpu_mode =
            resolve_llama_gpu_mode(env.qmd_llama_gpu.as_deref(), env.qmd_force_cpu.as_deref());
        let expand_ctx = context_size_env(
            env.qmd_expand_context_size.as_deref(),
            DEFAULT_EXPAND_CONTEXT_SIZE,
        );
        let model =
            load_leaked_model(path, &model_params(gpu_mode)?).map_err(|source| Error::Model {
                uri: model_uri.to_owned(),
                source,
            })?;
        let pool = Pool::spawn(
            1,
            move || Ok(model),
            move |model: &mut &'static LlamaModel, job: GenJob, _cancel: &CancelToken| {
                run_expansion(model, expand_ctx, &job)
            },
        )?;
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        Ok(Self {
            uri: model_uri.to_owned(),
            pool,
            seed: AtomicU32::new(seed),
        })
    }
}

impl Generator for LlamaGenerator {
    fn model_id(&self) -> &str {
        &self.uri
    }

    fn expand(&self, query: &str) -> Result<String, InferenceError> {
        let seed = self.seed.fetch_add(1, Ordering::Relaxed);
        self.pool.dispatch(
            GenJob {
                query: query.to_owned(),
                seed,
            },
            &CancelToken::new(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Upstream llm.test.ts: resolveParallelismOverride / QMD_EMBED_PARALLELISM.
    #[test]
    fn parallelism_override() {
        assert_eq!(resolve_parallelism_override(Some("4")), Some(4));
        assert_eq!(resolve_parallelism_override(Some("1")), Some(1));
        assert_eq!(resolve_parallelism_override(Some("8")), Some(8));
        // capped at 8
        assert_eq!(resolve_parallelism_override(Some("64")), Some(8));
        // non-integer / non-positive / junk -> automatic
        for v in ["0", "-3", "2.5", "abc", "4x", "", "NaN", "Infinity"] {
            assert_eq!(
                resolve_parallelism_override(Some(v)),
                None,
                "value {v:?} should fall back to automatic"
            );
        }
        assert_eq!(resolve_parallelism_override(None), None);
        // Number() trims whitespace
        assert_eq!(resolve_parallelism_override(Some("  3  ")), Some(3));
    }

    // Upstream llm.test.ts: context-size env parsing (parseInt + > 0).
    #[test]
    fn context_size_env_parsing() {
        let d = DEFAULT_EMBED_CONTEXT_SIZE;
        assert_eq!(context_size_env(Some("1024"), d), 1024);
        assert_eq!(context_size_env(Some(" 512 "), d), 512);
        // parseInt keeps the leading integer run
        assert_eq!(context_size_env(Some("300x"), d), 300);
        assert_eq!(context_size_env(Some("2048.9"), d), 2048);
        for v in ["0", "-512", "abc", "", "x300", "NaN"] {
            assert_eq!(context_size_env(Some(v), d), d, "value {v:?}");
        }
        assert_eq!(context_size_env(None, d), d);
    }

    // Upstream llm.test.ts: resolveLlamaGpuMode incl. QMD_FORCE_CPU.
    #[test]
    fn llama_gpu_mode_parsing() {
        use LlamaGpuMode::*;
        let cases = [
            (Some("metal"), None, Metal),
            (Some("Metal"), None, Metal),
            (Some("vulkan"), None, Vulkan),
            (Some("cuda"), None, Cuda),
            (Some("auto"), None, Auto),
            (Some(""), None, Auto),
            (None, None, Auto),
            (Some("off"), None, Cpu),
            (Some("false"), None, Cpu),
            (Some("none"), None, Cpu),
            (Some("disabled"), None, Cpu),
            (Some("0"), None, Cpu),
            (Some("nonsense"), None, Auto),
            // force_cpu truthy wins over any env value
            (Some("metal"), Some("1"), Cpu),
            (Some("cuda"), Some("yes"), Cpu),
            (None, Some("true"), Cpu),
            // force_cpu off-values defer to QMD_LLAMA_GPU
            (Some("metal"), Some("off"), Metal),
            (Some("metal"), Some("0"), Metal),
            (Some("cuda"), Some("false"), Cuda),
            // both empty -> auto
            (None, Some("0"), Auto),
            (None, Some("off"), Auto),
        ];
        for (env, force, want) in cases {
            assert_eq!(
                resolve_llama_gpu_mode(env, force),
                want,
                "env={env:?} force={force:?}"
            );
        }
    }

    // Upstream llm.test.ts: computeGpuContextPoolSize.
    #[test]
    fn gpu_pool_size() {
        // (10906-1000)*0.25/150 ≈ 16.5 -> floor 16 -> cap 8
        assert_eq!(compute_gpu_context_pool_size(10906, 150, 1000, 8), 8);
        // (5000-1000)*0.25/1000 = 1
        assert_eq!(compute_gpu_context_pool_size(5000, 1000, 1000, 8), 1);
        // free <= reserve -> clamped to 1
        assert_eq!(compute_gpu_context_pool_size(900, 150, 1000, 8), 1);
        // small contexts fan out to the cap
        assert_eq!(compute_gpu_context_pool_size(40_000, 50, 0, 8), 8);
        // cap honored
        assert_eq!(compute_gpu_context_pool_size(40_000, 50, 0, 4), 4);
    }

    // Upstream llm.test.ts: estimateEmbedContextMB.
    #[test]
    fn embed_context_mb_estimate() {
        // <=350MB model keeps the 150MB baseline at ctx 2048
        assert_eq!(estimate_embed_context_mb(250 * 1024 * 1024, 2048), 150);
        // ctx scaling applies to the baseline
        assert_eq!(estimate_embed_context_mb(250 * 1024 * 1024, 4096), 300);
        // >350MB: ~1.85x weights
        assert_eq!(estimate_embed_context_mb(700 * 1024 * 1024, 2048), 1295);
        // ctx=0 treated as 2048
        assert_eq!(estimate_embed_context_mb(250 * 1024 * 1024, 0), 150);
    }

    // Upstream llm.test.ts: resolveSafeParallelism precedence.
    #[test]
    fn safe_parallelism() {
        // explicit override wins over computed and platform clamps
        assert_eq!(resolve_safe_parallelism(true, 4, Some(3)), 3);
        // windows + cuda serializes without an override
        assert_eq!(resolve_safe_parallelism(true, 4, None), 1);
        // otherwise max(1, computed)
        assert_eq!(resolve_safe_parallelism(false, 4, None), 4);
        assert_eq!(resolve_safe_parallelism(false, 0, None), 1);
    }

    // JS slice(0, k) semantics used by rerank truncation.
    #[test]
    fn slice_head_js_semantics() {
        // positive index: first k
        assert_eq!(slice_head(10, 5), 5);
        assert_eq!(slice_head(10, 20), 10);
        assert_eq!(slice_head(10, 0), 0);
        // negative index: drop |k| from the end
        assert_eq!(slice_head(100, -50), 50);
        assert_eq!(slice_head(10, -20), 0);
    }

    // `QMD_EMBED_PARALLELISM` feeds pool_size via Environment; keep the
    // override behavior pinned at the unit level here.
    #[test]
    fn override_is_read_from_env() {
        let env = Environment {
            qmd_embed_parallelism: Some("2".to_owned()),
            ..Default::default()
        };
        // CPU path: computed is cpu_context_count() (>=1), override wins.
        assert_eq!(pool_size(LlamaGpuMode::Cpu, 1000, 0, 8, &env), 2);
    }
}

/// AC4 real-model parity tests against vendored E0 upstream data
/// (`tests/fixtures/e0/`, provenance in that directory's README).
/// Run with `cargo test -p qmd --features llama -- --ignored`.
#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    reason = "parity assertions index known fixture positions"
)]
mod e0_parity {
    use std::path::PathBuf;

    use serde_json::Value;

    use super::*;
    use crate::llm::download::model_cache_dir;

    const EVAL_DOCS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/eval-docs");

    fn model_file(name: &str) -> PathBuf {
        model_cache_dir(&Environment::default()).join(name)
    }

    fn fixture(name: &str) -> Value {
        let text = std::fs::read_to_string(format!(
            "{}/tests/fixtures/e0/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        serde_json::from_str(&text).unwrap()
    }

    fn f32_list(v: &Value) -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    fn u32_list(v: &Value) -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| u32::try_from(x.as_u64().unwrap()).unwrap())
            .collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            dot = f64::mul_add(f64::from(*x), f64::from(*y), dot);
            na = f64::mul_add(f64::from(*x), f64::from(*x), na);
            nb = f64::mul_add(f64::from(*y), f64::from(*y), nb);
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    /// AC4: every E0 embedding input reproduces upstream's vector.
    /// `token_ids` carry BOS/EOS (eval tokens); detokenizing strips them
    /// back to the formatted input text.
    #[test]
    #[ignore = "requires cached GGUF models"]
    fn embed_matches_e0() {
        let path = model_file("hf_ggml-org_embeddinggemma-300M-Q8_0.gguf");
        assert!(path.exists(), "model cache missing: {}", path.display());
        let env = Environment::default();
        let emb = LlamaEmbedder::new(
            &path,
            crate::llm::DEFAULT_EMBED_MODEL,
            &env,
            EmbeddingMode::UpstreamTail,
        )
        .unwrap();
        let items = fixture("embed.json")["items"].as_array().unwrap().clone();
        let mut worst_cos = 1.0f64;
        let mut worst_diff = 0.0f64;
        for item in &items {
            let ids = u32_list(&item["token_ids"]);
            let want = f32_list(&item["vector"]);
            let text = emb.detokenize(&ids).unwrap();
            // content tokens: tokenize must equal ids without BOS/EOS
            assert_eq!(
                emb.tokenize(&text).unwrap().as_slice(),
                &ids[1..ids.len() - 1],
                "content tokens diverge for {}",
                item["id"]
            );
            let got = emb.embed(&[&text]).unwrap();
            let got = &got[0];
            assert_eq!(got.len(), want.len(), "dim mismatch {}", item["id"]);
            let cos = cosine(got, &want);
            let mad = got
                .iter()
                .zip(want.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            worst_cos = worst_cos.min(cos);
            worst_diff = f64::from(mad).max(worst_diff);
            assert!(
                cos > 0.999,
                "{}: cosine {cos} vs upstream vector",
                item["id"]
            );
        }
        eprintln!("worst cosine {worst_cos}, worst |diff| {worst_diff}");
    }

    /// AC4: rerank template tokenization is byte-identical to upstream's
    /// `LlamaRankingContext` eval input, and scores stay within tolerance.
    #[test]
    #[ignore = "requires cached GGUF models"]
    fn rerank_matches_e0() {
        let path = model_file("hf_ggml-org_qwen3-reranker-0.6b-q8_0.gguf");
        assert!(path.exists(), "model cache missing: {}", path.display());
        let model = load_leaked_model(&path, &model_params(LlamaGpuMode::Auto).unwrap()).unwrap();
        let template = model
            .meta_val_str("tokenizer.chat_template.rerank")
            .unwrap();
        let vocab = model.vocab();
        let env = Environment::default();
        let reranker = LlamaReranker::new(&path, crate::llm::DEFAULT_RERANK_MODEL, &env).unwrap();
        let items = fixture("rerank.json")["items"].as_array().unwrap().clone();
        for item in &items {
            let query = item["query"].as_str().unwrap();
            let score_items = item["scores"].as_array().unwrap();
            let mut doc_texts = Vec::with_capacity(score_items.len());
            for s in score_items {
                let file = s["file"].as_str().unwrap();
                doc_texts.push(std::fs::read_to_string(format!("{EVAL_DOCS}/{file}")).unwrap());
            }
            // token-level parity for the template eval input
            for (i, s) in score_items.iter().enumerate() {
                let toks = rerank_tokens(&vocab, &template, query, &doc_texts[i]);
                assert_eq!(
                    token_ids(&toks).unwrap(),
                    u32_list(&s["token_ids"]),
                    "rerank tokens diverge: {} × {}",
                    query,
                    s["file"]
                );
            }
            // score parity
            let doc_refs: Vec<&str> = doc_texts.iter().map(String::as_str).collect();
            let got = reranker.rerank(query, &doc_refs).unwrap();
            for (i, s) in score_items.iter().enumerate() {
                let want = s["score"].as_f64().unwrap() as f32;
                assert!(
                    (got[i] - want).abs() < 0.05,
                    "score drift: {} × {} = {} vs {want}",
                    query,
                    s["file"],
                    got[i]
                );
            }
            // argmax parity: the top doc must match upstream's winner
            let argmax = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|x| x.0)
            };
            let want_scores: Vec<f32> = score_items
                .iter()
                .map(|s| s["score"].as_f64().unwrap() as f32)
                .collect();
            assert_eq!(
                argmax(&got),
                argmax(&want_scores),
                "argmax diverged for {query}"
            );
        }

        // Truncation path (review fix): a document past the context budget
        // is cut via `slice_head`, not passed through — it must score
        // instead of overflowing the context. The six eval docs
        // concatenated (~3.7k tokens) exceed the ~3.5k budget.
        let long_doc: String = [
            "api-design-principles.md",
            "distributed-systems-overview.md",
            "machine-learning-primer.md",
            "product-launch-retrospective.md",
            "remote-work-policy.md",
            "startup-fundraising-memo.md",
        ]
        .iter()
        .map(|f| std::fs::read_to_string(format!("{EVAL_DOCS}/{f}")).unwrap())
        .collect();
        let budget = i64::from(reranker.ctx_size)
            - i64::from(RERANK_TEMPLATE_OVERHEAD)
            - i64::try_from(vocab.tokenize(b"API versioning", false, false).len()).unwrap();
        let long_tokens = vocab.tokenize(long_doc.as_bytes(), false, false);
        assert!(
            i64::try_from(long_tokens.len()).unwrap() > budget,
            "fixture doc does not exceed budget"
        );
        let scores = reranker
            .rerank("API versioning", &[long_doc.as_str()])
            .unwrap();
        assert_eq!(scores.len(), 1);
        assert!(
            scores[0].is_finite() && (0.0..=1.0).contains(&scores[0]),
            "truncated doc score out of range: {}",
            scores[0]
        );
        // duplicated docs: dedup must scatter one score to both positions
        let dup = std::fs::read_to_string(format!("{EVAL_DOCS}/api-design-principles.md")).unwrap();
        let dup_scores = reranker
            .rerank("API versioning", &[dup.as_str(), dup.as_str()])
            .unwrap();
        assert_eq!(dup_scores[0], dup_scores[1], "dedup scatter diverged");
        // empty input: upstream returns [] rather than panicking
        assert!(reranker.rerank("q", &[]).unwrap().is_empty());
    }

    /// AC4: the rendered expansion prompt tokenizes identically to
    /// upstream's `QwenChatWrapper` output.
    #[test]
    #[ignore = "requires cached GGUF models"]
    fn expansion_prompt_matches_e0() {
        let path = model_file("hf_tobil_qmd-query-expansion-1.7B-q4_k_m.gguf");
        assert!(path.exists(), "model cache missing: {}", path.display());
        let model = load_leaked_model(&path, &model_params(LlamaGpuMode::Auto).unwrap()).unwrap();
        // byte-parity against the recorded rendered prompt — upstream's
        // capture concatenates the sampled output after `assistant\n`, so
        // our render must be an exact prefix of it.
        let gen_fixture = fixture("generate.json");
        let want_text = gen_fixture["rendered_prompt"]["text"].as_str().unwrap();
        let items = fixture("prompt_tokens.json")["items"]
            .as_array()
            .unwrap()
            .clone();
        for item in &items {
            let query = item["query"].as_str().unwrap();
            let rendered = render_expansion_prompt(model, query).unwrap();
            if query == "API versioning" {
                assert!(
                    want_text.starts_with(&rendered),
                    "rendered prompt drifted:\n{rendered:?}\nis not a prefix of E0:\n{want_text:?}"
                );
            }
            let toks = model.vocab().tokenize(rendered.as_bytes(), false, true);
            assert_eq!(
                token_ids(&toks).unwrap(),
                u32_list(&item["token_ids"]),
                "prompt tokens diverge for {query}"
            );
        }
    }

    /// AC4: `expand()` emits upstream's `lex:`/`vec:`/`hyde:` line grammar.
    #[test]
    #[ignore = "requires cached GGUF models"]
    fn expand_returns_structured_lines() {
        let path = model_file("hf_tobil_qmd-query-expansion-1.7B-q4_k_m.gguf");
        assert!(path.exists(), "model cache missing: {}", path.display());
        let env = Environment::default();
        let generator =
            LlamaGenerator::new(&path, crate::llm::DEFAULT_GENERATE_MODEL, &env).unwrap();
        let out = generator.expand("API versioning").unwrap();
        assert!(!out.trim().is_empty(), "empty expansion");
        let mut kinds = (false, false, false);
        for line in out.lines() {
            let hit = line.starts_with("lex: ")
                || line.starts_with("vec: ")
                || line.starts_with("hyde: ");
            assert!(hit, "off-grammar line: {line:?}");
            kinds.0 |= line.starts_with("lex: ");
            kinds.1 |= line.starts_with("vec: ");
            kinds.2 |= line.starts_with("hyde: ");
        }
        assert!(kinds.0 && kinds.1 && kinds.2, "missing a line kind: {out}");
    }
}
