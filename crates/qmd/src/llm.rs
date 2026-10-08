//! Model identity and embedding input formatting — the pure part of
//! upstream `src/llm.ts` (90-119, 275-321) plus `getEmbeddingFingerprint`
//! (`src/store.ts:108-131`).
//!
//! These exist so `status`/`index_health` can compute the fingerprint an
//! embedding run *would* use, without loading a model. Actual inference
//! arrives in P2.

use sha2::{Digest, Sha256};

use crate::config::ModelsConfig;
use crate::env::Environment;

/// Upstream `DEFAULT_EMBED_MODEL` (llm.ts:282): embeddinggemma-300M.
pub const DEFAULT_EMBED_MODEL: &str =
    "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
/// Upstream `DEFAULT_RERANK_MODEL` (llm.ts:283).
pub const DEFAULT_RERANK_MODEL: &str =
    "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
/// Upstream `DEFAULT_GENERATE_MODEL` (llm.ts:285).
pub const DEFAULT_GENERATE_MODEL: &str =
    "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf";

/// Upstream `CHUNK_SIZE_TOKENS` (store.ts:114).
pub const CHUNK_SIZE_TOKENS: u32 = 900;
/// Upstream `CHUNK_OVERLAP_TOKENS` (store.ts:115): 15% of the chunk size.
pub const CHUNK_OVERLAP_TOKENS: u32 = CHUNK_SIZE_TOKENS * 15 / 100;

const FINGERPRINT_PROBE_QUERY: &str = "__qmd_embedding_query_probe__";
const FINGERPRINT_PROBE_TITLE: &str = "__qmd_embedding_title_probe__";
const FINGERPRINT_PROBE_DOC: &str = "__qmd_embedding_document_probe__";

/// Upstream `isQwen3EmbeddingModel` (llm.ts:90-92):
/// `/qwen.*embed/i` or `/embed.*qwen/i` on the model URI.
#[must_use]
pub fn is_qwen3_embedding_model(model_uri: &str) -> bool {
    let lower = model_uri.to_lowercase();
    let qwen = lower.find("qwen");
    let embed = lower.find("embed");
    match (qwen, embed) {
        (Some(q), Some(e)) => q != e,
        _ => false,
    }
}

/// Upstream `formatQueryForEmbedding` (llm.ts:99-105): nomic-style task
/// prefix, or the Qwen3-Embedding instruct format.
#[must_use]
pub fn format_query_for_embedding(query: &str, model_uri: Option<&str>) -> String {
    let uri = model_uri.unwrap_or(DEFAULT_EMBED_MODEL);
    if is_qwen3_embedding_model(uri) {
        return format!(
            "Instruct: Retrieve relevant documents for the given query\nQuery: {query}"
        );
    }
    format!("task: search result | query: {query}")
}

/// Upstream `formatDocForEmbedding` (llm.ts:112-119): nomic-style
/// title/text fields; Qwen3-Embedding encodes documents as raw text.
#[must_use]
pub fn format_doc_for_embedding(
    text: &str,
    title: Option<&str>,
    model_uri: Option<&str>,
) -> String {
    let uri = model_uri.unwrap_or(DEFAULT_EMBED_MODEL);
    if is_qwen3_embedding_model(uri) {
        return title.map_or_else(|| text.to_owned(), |t| format!("{t}\n{text}"));
    }
    format!("title: {} | text: {text}", title.unwrap_or("none"))
}

/// Upstream `resolveEmbedModel` (llm.ts:303-305): `config.embed` →
/// `QMD_EMBED_MODEL` → [`DEFAULT_EMBED_MODEL`]. Empty values are unset.
#[must_use]
pub fn resolve_embed_model(models: Option<&ModelsConfig>, env: &Environment) -> String {
    models
        .and_then(|m| m.embed.as_deref())
        .or(env.qmd_embed_model.as_deref())
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_EMBED_MODEL)
        .to_owned()
}

/// Upstream `resolveGenerateModel` (llm.ts:307-309).
#[must_use]
pub fn resolve_generate_model(models: Option<&ModelsConfig>, env: &Environment) -> String {
    models
        .and_then(|m| m.generate.as_deref())
        .or(env.qmd_generate_model.as_deref())
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_GENERATE_MODEL)
        .to_owned()
}

/// Upstream `resolveRerankModel` (llm.ts:311-313).
#[must_use]
pub fn resolve_rerank_model(models: Option<&ModelsConfig>, env: &Environment) -> String {
    models
        .and_then(|m| m.rerank.as_deref())
        .or(env.qmd_rerank_model.as_deref())
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_RERANK_MODEL)
        .to_owned()
}

/// Resolved model triple — upstream `resolveModels` (llm.ts:315-321).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ResolvedModels {
    /// Embedding model URI.
    pub embed: String,
    /// Generation (query expansion) model URI.
    pub generate: String,
    /// Reranker model URI.
    pub rerank: String,
}

/// Upstream `resolveModels` (llm.ts:315-321).
#[must_use]
pub fn resolve_models(models: Option<&ModelsConfig>, env: &Environment) -> ResolvedModels {
    ResolvedModels {
        embed: resolve_embed_model(models, env),
        generate: resolve_generate_model(models, env),
        rerank: resolve_rerank_model(models, env),
    }
}

/// Upstream `getEmbeddingFingerprint` (store.ts:123-132): first 6 hex
/// chars of the SHA-256 over the significant inputs.
///
/// The hashed inputs are the model URI, the formatted probe query/doc,
/// and the chunking parameters. Vectors recorded under a different
/// fingerprint count as needing re-embedding.
#[must_use]
pub fn embedding_fingerprint(model: &str) -> String {
    let significant = [
        format!("model:{model}"),
        format!(
            "query:{}",
            format_query_for_embedding(FINGERPRINT_PROBE_QUERY, Some(model))
        ),
        format!(
            "doc:{}",
            format_doc_for_embedding(
                FINGERPRINT_PROBE_DOC,
                Some(FINGERPRINT_PROBE_TITLE),
                Some(model)
            )
        ),
        format!("chunk_tokens:{CHUNK_SIZE_TOKENS}"),
        format!("chunk_overlap_tokens:{CHUNK_OVERLAP_TOKENS}"),
    ]
    .join("\n");
    let digest = Sha256::digest(significant.as_bytes());
    let mut s = String::with_capacity(6);
    for b in digest.iter().take(3) {
        s.push(char::from_digit(u32::from(b >> 4), 16).unwrap_or('0'));
        s.push(char::from_digit(u32::from(b & 0x0f), 16).unwrap_or('0'));
    }
    s
}
