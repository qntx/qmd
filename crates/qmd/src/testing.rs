//! Deterministic in-memory inference backends (T10, architecture §11.2).
//!
//! Available with the `testing` feature: they cover the indexing/search
//! control flow on model-less CI; numeric parity is asserted by the
//! real-model alignment tests instead.

use std::fmt;
use std::sync::Arc;

use crate::llm::{Embedder, Generator, InferenceError, Reranker};

type TokenizeFn = dyn Fn(&str) -> Vec<u32> + Send + Sync;
type DetokenizeFn = dyn Fn(&[u32]) -> String + Send + Sync;
type ExpandFn = dyn Fn(&str) -> String + Send + Sync;

/// Deterministic [`Embedder`] — vectors from character n-gram hashing.
///
/// `tokenize` defaults to one token per character; `detokenize` to
/// `"x" * tokens.len()` (matching the fake tokenizers upstream tests
/// inject). Both are replaceable for guardrail scenarios.
#[derive(Clone)]
pub struct FakeEmbedder {
    model_id: String,
    dimension: usize,
    fingerprint_extra: Option<String>,
    tokenizer: Arc<TokenizeFn>,
    detokenizer: Arc<DetokenizeFn>,
}

impl fmt::Debug for FakeEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FakeEmbedder")
            .field("model_id", &self.model_id)
            .field("dimension", &self.dimension)
            .finish_non_exhaustive()
    }
}

impl FakeEmbedder {
    /// A fake backend identifying as `model_id`.
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            dimension: 64,
            fingerprint_extra: None,
            tokenizer: Arc::new(|text| text.chars().map(u32::from).collect()),
            detokenizer: Arc::new(|tokens| "x".repeat(tokens.len())),
        }
    }

    /// Override the produced vector dimension.
    #[allow(
        clippy::missing_const_for_fn,
        reason = "Self contains Arc fields which cannot be mutated in const fn"
    )]
    #[must_use]
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    /// Override [`Embedder::fingerprint_extra`].
    #[must_use]
    pub fn with_fingerprint_extra(mut self, extra: impl Into<String>) -> Self {
        self.fingerprint_extra = Some(extra.into());
        self
    }

    /// Replace the tokenizer (upstream tests inject arbitrary tokenize
    /// shapes, e.g. one token per UTF-16 unit:
    /// `|s| vec![1; s.encode_utf16().count()]`).
    #[allow(
        clippy::missing_const_for_fn,
        reason = "Self contains Arc fields which cannot be mutated in const fn"
    )]
    #[must_use]
    pub fn with_tokenizer(mut self, f: impl Fn(&str) -> Vec<u32> + Send + Sync + 'static) -> Self {
        self.tokenizer = Arc::new(f);
        self
    }

    /// Replace the detokenizer.
    #[allow(
        clippy::missing_const_for_fn,
        reason = "Self contains Arc fields which cannot be mutated in const fn"
    )]
    #[must_use]
    pub fn with_detokenizer(
        mut self,
        f: impl Fn(&[u32]) -> String + Send + Sync + 'static,
    ) -> Self {
        self.detokenizer = Arc::new(f);
        self
    }
}

impl Embedder for FakeEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn fingerprint_extra(&self) -> Option<&str> {
        self.fingerprint_extra.as_deref()
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, InferenceError> {
        Ok((self.tokenizer)(text))
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, InferenceError> {
        Ok((self.detokenizer)(tokens))
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, InferenceError> {
        Ok(texts
            .iter()
            .map(|text| fake_vector(text, self.dimension))
            .collect())
    }
}

/// Deterministic embedding vector: accumulate hashed uni-/bigrams into
/// `dim` buckets, then L2-normalize (empty vector stays zero).
fn fake_vector(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    if dim == 0 {
        return v;
    }
    let cps: Vec<u32> = text.chars().map(u32::from).collect();
    for (i, &c) in cps.iter().enumerate() {
        add_hash(&mut v, u64::from(c));
        if let Some(&next) = cps.get(i + 1) {
            add_hash(&mut v, (u64::from(c) << 8) | u64::from(next));
        }
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// FNV-1a of `value` → bump one bucket up or down.
fn add_hash(v: &mut [f32], value: u64) {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in value.to_le_bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let idx = usize::try_from(h % u64::try_from(v.len()).unwrap_or(u64::MAX)).unwrap_or(0);
    let delta = if h & (1 << 63) == 0 { 1.0 } else { -1.0 };
    if let Some(slot) = v.get_mut(idx) {
        *slot += delta;
    }
}

/// Deterministic [`Reranker`] — query/document word-overlap scores.
#[derive(Debug, Clone)]
pub struct FakeReranker {
    model_id: String,
}

impl FakeReranker {
    /// A fake backend identifying as `model_id`.
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
        }
    }
}

impl Reranker for FakeReranker {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>, InferenceError> {
        let query_words: Vec<&str> = query.split_whitespace().collect();
        Ok(docs
            .iter()
            .map(|doc| {
                if query_words.is_empty() {
                    return 0.0;
                }
                let doc_words: std::collections::HashSet<&str> = doc.split_whitespace().collect();
                let hits = query_words
                    .iter()
                    .filter(|w| doc_words.contains(*w))
                    .count();
                hits as f32 / query_words.len() as f32
            })
            .collect())
    }
}

/// Deterministic [`Generator`] — a fixed template by default, or a
/// caller-supplied output function for expansion tests.
#[derive(Clone)]
pub struct FakeGenerator {
    model_id: String,
    output: Arc<ExpandFn>,
}

impl fmt::Debug for FakeGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FakeGenerator")
            .field("model_id", &self.model_id)
            .finish_non_exhaustive()
    }
}

impl FakeGenerator {
    /// A fake backend producing `lex:`/`vec:`/`hyde:` template lines.
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            output: Arc::new(|query| format!("lex: {query}\nvec: {query}\nhyde: {query}")),
        }
    }

    /// Always produce `output`, ignoring the query.
    #[must_use]
    pub fn fixed(model_id: impl Into<String>, output: impl Into<String>) -> Self {
        let output = output.into();
        Self {
            model_id: model_id.into(),
            output: Arc::new(move |_| output.clone()),
        }
    }

    /// Produce output from a function of the query.
    #[must_use]
    pub fn with_output(
        model_id: impl Into<String>,
        f: impl Fn(&str) -> String + Send + Sync + 'static,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            output: Arc::new(f),
        }
    }
}

impl Generator for FakeGenerator {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn expand(&self, query: &str) -> Result<String, InferenceError> {
        Ok((self.output)(query))
    }
}
