//! Inference backend traits — the injectable seam between the core
//! indexing/search pipeline and concrete model runtimes (architecture §8.1).
//!
//! The core crates formats embedding inputs, parses generator output and
//! decides caching/fallbacks; a backend only runs the model. `Send + Sync`
//! is required because [`crate::Qmd`] shares one handle across threads.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// The inference capability a backend provides — [`crate::Error::NoBackend`]
/// names the missing one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Capability {
    /// Text embedding ([`Embedder`]).
    Embed,
    /// Document reranking ([`Reranker`]).
    Rerank,
    /// Query expansion ([`Generator`]).
    Generate,
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Embed => "embedding",
            Self::Rerank => "reranking",
            Self::Generate => "query expansion",
        };
        f.write_str(s)
    }
}

/// Failure reported by an inference backend.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InferenceError {
    /// The backend or model runtime failed.
    #[error("{0}")]
    Backend(String),
    /// The operation was cancelled via a [`CancelToken`].
    #[error("operation cancelled")]
    Cancelled,
}

/// Cooperative cancellation flag — the Rust counterpart of upstream
/// `AbortSignal`/`session.isValid` (T6). Cheap to clone; cancelling any
/// clone cancels the operation.
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    /// A fresh, uncancelled token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Signal cancellation. Checked at chunk/batch boundaries; in-flight
    /// work is not interrupted.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation has been signalled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
}

/// Text embedding backend — upstream `LlamaCpp.embed`/`tokenize`.
///
/// `embed` receives already-formatted inputs (see
/// [`crate::llm::format_query_for_embedding`] /
/// [`crate::llm::format_doc_for_embedding`]) and must return one vector
/// per input. Upstream's per-text `null` results only occur because the
/// batch call swallows per-item exceptions; in Rust the batch returns
/// `Err` as a whole and the core layer retries per text (T1).
pub trait Embedder: Send + Sync + fmt::Debug {
    /// Model URI; recorded in `content_vectors.model` and the embedding
    /// fingerprint.
    fn model_id(&self) -> &str;

    /// Extra fingerprint line for backend-specific behavior
    /// (architecture §8.3). `None` — the default — keeps the fingerprint
    /// identical to upstream.
    fn fingerprint_extra(&self) -> Option<&str> {
        None
    }

    /// Encode `text` into token ids (upstream `llm.tokenize`).
    ///
    /// # Errors
    /// [`InferenceError::Backend`] on tokenizer failure.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, InferenceError>;

    /// Decode token ids back to text (upstream `llm.detokenize`).
    ///
    /// # Errors
    /// [`InferenceError::Backend`] on tokenizer failure.
    fn detokenize(&self, tokens: &[u32]) -> Result<String, InferenceError>;

    /// Embed already-formatted texts; the batch succeeds or fails as a
    /// whole. Output length must equal `texts.len()`.
    ///
    /// # Errors
    /// [`InferenceError::Backend`] on model failure.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, InferenceError>;
}

/// Document reranking backend — upstream `LlamaRankingContext`.
pub trait Reranker: Send + Sync + fmt::Debug {
    /// Model URI.
    fn model_id(&self) -> &str;

    /// Score each document against `query`; one score per entry of
    /// `docs`, in `[0, 1]`.
    ///
    /// # Errors
    /// [`InferenceError::Backend`] on model failure.
    fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>, InferenceError>;
}

/// Query-expansion backend — the LLM call inside upstream
/// `expandQueryWithLlm`.
///
/// Returns the raw model output (`lex:`/`vec:`/`hyde:` lines); parsing,
/// validation and fallback live in the core layer (architecture §8.2).
pub trait Generator: Send + Sync + fmt::Debug {
    /// Model URI.
    fn model_id(&self) -> &str;

    /// Generate expansion lines for `query`.
    ///
    /// # Errors
    /// [`InferenceError::Backend`] on model failure.
    fn expand(&self, query: &str) -> Result<String, InferenceError>;
}
