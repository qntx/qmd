//! Search orchestration — upstream `expandQuery`/`rerank`
//! (store.ts:4550-4651), `hybridQuery` (5538-5837), `vectorSearchQuery`
//! (5868-5917) and `structuredSearch` (5960-6238).
//!
//! T14: the two pipelines share [`run_pipeline`] for everything after
//! list construction; the five upstream differences live at list-build
//! time (weights, per-collection lists, strong-signal bypass + dud
//! eviction, rerank query choice, hook timing).

use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use rusqlite::Connection;

use super::cache::{
    CacheKeyBody, cache_key, delete_expansion_cache_entry, get_cached_result, set_cached_result,
};
use super::metadata::get_metadata_by_filepath;
use super::rrf::{
    FileRrfTrace, RRF_K, RankedListMeta, RankedResult, RrfContribution, build_rrf_trace,
    get_hybrid_rrf_weights, reciprocal_rank_fusion,
};
use super::search::{SearchResult, SearchSource, search_fts, search_vec_multi};
use super::status::has_vectors_vec;
use crate::chunk::{
    CharChunkOptions, ChunkStrategy, chunk_document_with_strategy, is_js_ws, utf16_len,
};
use crate::error::{Error, Result};
use crate::llm::{Embedder, Generator, InferenceError, Reranker};
use crate::metadata::DocumentMetadata;
use crate::metadata::filter::MetadataFilter;
use crate::query::{
    ExpandedQuery, INTENT_WEIGHT_CHUNK, QueryKind, expansion_error_fallback,
    expansion_parse_fallback, extract_intent_terms, parse_expansion_output,
};

/// Optional progress hooks for search orchestration — upstream
/// `SearchHooks` (store.ts:5447-5462). All callbacks default to no-ops;
/// `()` implements the trait for callers that want none.
pub trait SearchHooks {
    /// BM25 probe found strong signal — expansion will be skipped.
    fn on_strong_signal(&self, _top_score: f64) {}
    /// Query expansion starting.
    fn on_expand_start(&self) {}
    /// Query expansion complete. An empty `expanded` means either a
    /// strong-signal skip or, for structured search, no expansion —
    /// `elapsed` distinguishes nothing, upstream fires the same shape.
    fn on_expand(&self, _original: &str, _expanded: &[ExpandedQuery], _elapsed: Duration) {}
    /// Embedding starting (vec/hyde queries).
    fn on_embed_start(&self, _count: usize) {}
    /// Embedding complete.
    fn on_embed_done(&self, _elapsed: Duration) {}
    /// Reranking is about to start.
    fn on_rerank_start(&self, _chunk_count: usize) {}
    /// Reranking finished.
    fn on_rerank_done(&self, _elapsed: Duration) {}
}

impl SearchHooks for () {}

/// Lazily-resolved inference backends. Each resolver hits the handle's
/// `Mutex` slot — cheap after the first call — so the pipeline only
/// pays for the capabilities it actually reaches (no embedder on
/// FTS-only stores, no reranker under `skip_rerank`).
pub(crate) struct Backends<'a> {
    /// `Qmd::embedder` resolver.
    pub embedder: &'a dyn Fn() -> Result<Arc<dyn Embedder>>,
    /// `Qmd::reranker` resolver.
    pub reranker: &'a dyn Fn() -> Result<Arc<dyn Reranker>>,
    /// `Qmd::generator` resolver.
    pub generator: &'a dyn Fn() -> Result<Arc<dyn Generator>>,
}

/// Shared inputs for one orchestrated search.
///
/// `conn` is the shared connection mutex — [`SearchCtx::db`] locks it
/// per statement so an [`Embedder`]/[`Reranker`]/[`Generator`] call
/// never holds it (parity-matrix deviation 43). Upstream's
/// single-threaded store interleaves concurrent requests at each
/// `await`, so holding the lock across inference would serialize what
/// upstream does not.
pub(crate) struct SearchCtx<'a> {
    /// Index connection mutex.
    pub conn: &'a Mutex<Connection>,
    /// Cache-write counter for the deterministic prune (T5).
    pub cache_writes: &'a AtomicUsize,
    /// Backend resolvers.
    pub backends: Backends<'a>,
    /// Progress hooks.
    pub hooks: &'a dyn SearchHooks,
}

impl SearchCtx<'_> {
    /// Lock the connection for a single call — never held across
    /// inference. Poison-tolerant, same as [`crate::qmd`]'s lock.
    fn db(&self) -> MutexGuard<'_, Connection> {
        crate::qmd::lock(self.conn)
    }
}

fn search_result_to_ranked(r: &SearchResult) -> RankedResult {
    RankedResult {
        file: r.filepath.clone(),
        display_path: r.display_path.clone(),
        title: r.title.clone(),
        body: r.body.clone(),
        score: r.score,
        docid: r.docid.clone(),
        context: r.context.clone(),
    }
}

/// Upstream `attachResultMetadata` (store.ts:5496-5504): attach canonical
/// metadata to final results with ONE batch query, after RRF/reranking
/// so metadata is never duplicated through the intermediate ranked
/// lists. Results without a metadata row get `{}`.
fn attach_result_metadata(conn: &Connection, results: &mut [HybridResult]) -> Result<()> {
    let filepaths: Vec<String> = results.iter().map(|r| r.filepath.clone()).collect();
    let by_filepath = get_metadata_by_filepath(conn, &filepaths)?;
    for r in results.iter_mut() {
        if let Some(m) = by_filepath.get(&r.filepath) {
            r.metadata = m.clone();
        }
    }
    Ok(())
}

// =============================================================================
// Query expansion — upstream `expandQuery` (store.ts:4550-4589) with the
// line-parsing half of `LlamaCpp.expandQuery` (llm.ts:1686-1724) per T1.
// =============================================================================

/// One cached expansion row → [`ExpandedQuery`]. `String(r.query)` —
/// JS stringifies non-strings (`undefined` when absent); rows with an
/// unrecognised `type` never match a downstream kind filter, so
/// dropping them is equivalent.
fn cached_expansion_row(r: &serde_json::Value) -> Option<ExpandedQuery> {
    let kind = match r.get("type").and_then(|t| t.as_str()) {
        Some("lex") => QueryKind::Lex,
        Some("vec") => QueryKind::Vec,
        Some("hyde") => QueryKind::Hyde,
        _ => return None,
    };
    Some(ExpandedQuery {
        kind,
        query: match r.get("query") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(v) => v.to_string(),
            None => "undefined".to_owned(),
        },
        line: None,
    })
}

/// Returns the expanded queries plus the cache-key model — the dud
/// eviction in [`hybrid_query`] must delete the same key.
pub(crate) fn expand_query(
    ctx: &SearchCtx<'_>,
    query: &str,
) -> Result<(Vec<ExpandedQuery>, String)> {
    let generator = (ctx.backends.generator)()?;
    let model = generator.model_id().to_owned();
    let key = cache_key(
        "expandQuery",
        &CacheKeyBody {
            query: Some(query),
            model: Some(&model),
            chunk: None,
        },
    );
    let cached = get_cached_result(&ctx.db(), &key)?;
    if let Some(cached) = cached {
        // Upstream JSON.parse + `Array.isArray` + rows[0].query check:
        // valid non-array JSON returns `[]`; anything else re-expands.
        // The legacy `{type, text}` migration is not ported (D2).
        match serde_json::from_str::<serde_json::Value>(&cached) {
            Ok(serde_json::Value::Array(rows)) => {
                let first_is_typed = rows
                    .first()
                    .and_then(|r| r.get("query"))
                    .is_some_and(serde_json::Value::is_string);
                if !rows.is_empty() && first_is_typed {
                    let parsed = rows
                        .iter()
                        .filter_map(cached_expansion_row)
                        .collect::<Vec<_>>();
                    return Ok((parsed, model));
                }
                // rows[0] without a string `query` falls through to
                // re-expansion, like upstream's exhausted if-chain
                // (the legacy `{type, text}` branch is not ported — D2).
            }
            Ok(_) => return Ok((Vec::new(), model)),
            Err(_) => {}
        }
    }

    // `Generator::expand` returns the raw model output; parsing,
    // filtering and both fallbacks live here (T1).
    let results = match generator.expand(query) {
        // Model resolution/load failures are never degraded — upstream
        // `ensureGenerateModel` throws before generation (llm.ts:1635-1636).
        Err(e @ InferenceError::ModelLoad(_)) => return Err(e.into()),
        Err(_) => expansion_error_fallback(query),
        Ok(raw) => {
            let parsed = parse_expansion_output(query, &raw);
            if parsed.is_empty() {
                expansion_parse_fallback(query)
            } else {
                parsed
            }
        }
    };
    // Filter out entries that duplicate the original query text
    // (upstream store.ts:4580-4582).
    let expanded: Vec<ExpandedQuery> = results.into_iter().filter(|r| r.query != query).collect();

    if !expanded.is_empty() {
        #[allow(
            clippy::expect_used,
            clippy::unwrap_in_result,
            reason = "serializing a flat array of {type, query} strings cannot fail"
        )]
        let body = serde_json::to_string(
            &expanded
                .iter()
                .map(|e| serde_json::json!({"type": e.kind.as_str(), "query": e.query}))
                .collect::<Vec<_>>(),
        )
        .expect("infallible: only &str fields");
        set_cached_result(&ctx.db(), &key, &body, ctx.cache_writes)?;
    }
    Ok((expanded, model))
}

// =============================================================================
// Rerank orchestration — upstream `rerank` (store.ts:4605-4651).
// =============================================================================

/// `docs` is `(file, chunk_text)` pairs; the returned list is sorted by
/// score descending (stable) with missing/NaN scores coerced to 0.
pub(crate) fn rerank(
    ctx: &SearchCtx<'_>,
    query: &str,
    intent: Option<&str>,
    docs: &[(String, String)],
) -> Result<Vec<(String, f64)>> {
    let rerank_query = intent.map_or_else(|| query.to_owned(), |i| format!("{i}\n\n{query}"));
    // `llm.rerankModelName ?? model` — the resolved URI (#764).
    let reranker = (ctx.backends.reranker)()?;
    let cache_model = reranker.model_id().to_owned();

    let mut cached_scores: IndexMap<String, f64> = IndexMap::new();
    let mut uncached_by_chunk: IndexMap<String, String> = IndexMap::new();
    for (file, text) in docs {
        let key = cache_key(
            "rerank",
            &CacheKeyBody {
                query: Some(&rerank_query),
                model: Some(&cache_model),
                chunk: Some(text),
            },
        );
        let cached = get_cached_result(&ctx.db(), &key)?;
        if let Some(cached) = cached {
            // JS `parseFloat` — a corrupt entry becomes NaN, which the
            // caller's `|| 0` turns into a 0 score.
            cached_scores.insert(text.clone(), cached.parse::<f64>().unwrap_or(f64::NAN));
        } else {
            uncached_by_chunk.insert(text.clone(), file.clone());
        }
    }

    if !uncached_by_chunk.is_empty() {
        let texts: Vec<&str> = uncached_by_chunk.keys().map(String::as_str).collect();
        let scores = reranker
            .rerank(&rerank_query, &texts)
            .map_err(Error::from)?;
        for (text, score) in uncached_by_chunk.keys().zip(scores.iter()) {
            let score = f64::from(*score);
            let key = cache_key(
                "rerank",
                &CacheKeyBody {
                    query: Some(&rerank_query),
                    model: Some(&cache_model),
                    chunk: Some(text),
                },
            );
            // JS `score.toString()` — shortest-roundtrip decimal.
            set_cached_result(&ctx.db(), &key, &score.to_string(), ctx.cache_writes)?;
            cached_scores.insert(text.clone(), score);
        }
    }

    let mut results: Vec<(String, f64)> = docs
        .iter()
        .map(|(file, text)| {
            // `cachedResults.get(doc.text) || 0` — NaN and ±0 are
            // falsy in JS and collapse to 0.
            let score = cached_scores
                .get(text)
                .copied()
                .filter(|s| !s.is_nan() && *s != 0.0)
                .unwrap_or(0.0);
            (file.clone(), score)
        })
        .collect();
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    Ok(results)
}

// =============================================================================
// Shared post-fusion pipeline (T14)
// =============================================================================

/// Options shared by the hybrid and structured paths after list
/// construction — upstream `HybridQueryOptions`/`StructuredSearchOptions`.
pub(crate) struct PipelineOpts<'a> {
    /// Query used for chunk selection and reranking — the original
    /// query for hybrid, `primaryQuery` for structured.
    pub query: &'a str,
    /// Domain intent hint.
    pub intent: Option<&'a str>,
    /// `skipRerank`.
    pub skip_rerank: bool,
    /// `explain`.
    pub explain: bool,
    /// `minScore`.
    pub min_score: f64,
    /// `limit`.
    pub limit: usize,
    /// `candidateLimit` — also the fallback `rrfRank` for files missing
    /// from the fused candidates (upstream `|| candidateLimit`).
    pub candidate_limit: usize,
    /// `chunkStrategy`.
    pub chunk_strategy: ChunkStrategy,
    /// RRF trace per file, built only when `explain`.
    pub trace: Option<IndexMap<String, FileRrfTrace>>,
}

/// The explain payload attached to a [`HybridResult`] — upstream
/// `HybridQueryExplain`.
#[derive(Debug)]
#[non_exhaustive]
pub struct Explain {
    /// Backend FTS scores contributing to the file.
    pub fts_scores: Vec<f64>,
    /// Backend vector scores contributing to the file.
    pub vector_scores: Vec<f64>,
    /// RRF trace.
    pub rrf: RrfTrace,
    /// Reranker score for the selected chunk (0 when skipped).
    pub rerank_score: f64,
    /// Final blended score.
    pub blended_score: f64,
}

/// Upstream `HybridQueryExplain.rrf`.
#[derive(Debug)]
#[non_exhaustive]
pub struct RrfTrace {
    /// 1-indexed fusion rank.
    pub rank: usize,
    /// `1 / rank` positional score.
    pub position_score: f64,
    /// Position-protection weight (1.0 when skipped, else
    /// 0.75/0.60/0.40).
    pub weight: f64,
    /// RRF score before the top-rank bonus.
    pub base_score: f64,
    /// Top-rank bonus.
    pub top_rank_bonus: f64,
    /// `base_score + top_rank_bonus`.
    pub total_score: f64,
    /// Per-list contributions.
    pub contributions: Vec<RrfContribution>,
}

/// Upstream `HybridQueryResult` (store.ts:5745-5758).
#[derive(Debug)]
#[non_exhaustive]
pub struct HybridResult {
    /// `qmd://collection/path` URI (upstream `file`).
    pub filepath: String,
    /// `collection/path` for display.
    pub display_path: String,
    /// Document title.
    pub title: String,
    /// Full body (for snippet extraction).
    pub body: String,
    /// Best chunk text — body fallback when no chunking is available.
    pub best_chunk: String,
    /// UTF-16 offset of `best_chunk` inside `body` (0 on fallback).
    pub best_chunk_pos: usize,
    /// Blended score (full precision).
    pub score: f64,
    /// Folder context.
    pub context: Option<String>,
    /// Short docid (first 6 hash chars).
    pub docid: String,
    /// Canonical metadata (`{}` when absent) — attached after ranking
    /// via one batch query (upstream `attachResultMetadata`).
    pub metadata: DocumentMetadata,
    /// Score trace when `explain` was requested.
    pub explain: Option<Box<Explain>>,
}

fn build_explain(
    trace: Option<&FileRrfTrace>,
    rank: usize,
    position_score: f64,
    weight: f64,
    rerank_score: f64,
    blended_score: f64,
) -> Explain {
    let contributions = trace.map_or_else(Vec::new, |t| t.contributions.clone());
    Explain {
        fts_scores: contributions
            .iter()
            .filter(|c| c.source == SearchSource::Fts)
            .map(|c| c.backend_score)
            .collect(),
        vector_scores: contributions
            .iter()
            .filter(|c| c.source == SearchSource::Vec)
            .map(|c| c.backend_score)
            .collect(),
        rrf: RrfTrace {
            rank,
            position_score,
            weight,
            base_score: trace.map_or(0.0, |t| t.base_score),
            top_rank_bonus: trace.map_or(0.0, |t| t.top_rank_bonus),
            total_score: trace.map_or(0.0, |t| t.total_score),
            contributions,
        },
        rerank_score,
        blended_score,
    }
}

/// Term lists for chunk scoring — JS `split(/\s+/)`, and `t.length > 2`
/// counts UTF-16 units upstream (store.ts:5684, 6086).
fn selection_terms(query: &str, intent: Option<&str>) -> (Vec<String>, Vec<String>) {
    let query_terms: Vec<String> = query
        .to_lowercase()
        .split(is_js_ws)
        .filter(|t| utf16_len(t) > 2)
        .map(ToOwned::to_owned)
        .collect();
    let intent_terms = intent.map_or_else(Vec::new, extract_intent_terms);
    (query_terms, intent_terms)
}

/// One chunk's keyword score — query terms count 1.0 each, intent
/// terms [`INTENT_WEIGHT_CHUNK`] (upstream `scoreChunk` inlined at
/// store.ts:5684-5687 and 6086-6089).
fn chunk_score(lower_chunk: &str, query_terms: &[String], intent_terms: &[String]) -> f64 {
    let query_hits = query_terms
        .iter()
        .filter(|t| lower_chunk.contains(t.as_str()))
        .count() as f64;
    let intent_hits = intent_terms
        .iter()
        .filter(|t| lower_chunk.contains(t.as_str()))
        .count() as f64;
    intent_hits.mul_add(INTENT_WEIGHT_CHUNK, query_hits)
}

/// Chunk each candidate with the char strategy and pick the chunk with
/// the most keyword overlap — upstream step 5 (`store.ts:5682-5707`)
/// shared by both pipelines.
fn select_best_chunks<'a>(
    candidates: &'a [RankedResult],
    query: &str,
    intent: Option<&str>,
    chunk_strategy: ChunkStrategy,
    options: &CharChunkOptions,
) -> IndexMap<&'a str, (Vec<crate::chunk::Chunk<'a>>, usize)> {
    let (query_terms, intent_terms) = selection_terms(query, intent);
    let mut map: IndexMap<&'a str, (Vec<crate::chunk::Chunk<'a>>, usize)> = IndexMap::new();
    for cand in candidates {
        let chunks =
            chunk_document_with_strategy(&cand.body, Some(&cand.file), chunk_strategy, options);
        if chunks.is_empty() {
            continue;
        }
        let mut best_idx = 0usize;
        let mut best_score = -1.0f64;
        for (i, chunk) in chunks.iter().enumerate() {
            let score = chunk_score(&chunk.text.to_lowercase(), &query_terms, &intent_terms);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        map.insert(cand.file.as_str(), (chunks, best_idx));
    }
    map
}

/// Batch-embed all `queries` in ONE `embed` call — upstream
/// hybrid/structured embed the query texts together and the
/// `onEmbedStart`/`onEmbedDone` hooks bracket the batch.
fn embed_queries(ctx: &SearchCtx<'_>, queries: &[&str]) -> Result<Vec<Vec<f32>>> {
    let embedder = (ctx.backends.embedder)()?;
    let texts: Vec<String> = queries
        .iter()
        .map(|q| crate::llm::format_query_for_embedding(q, Some(embedder.model_id())))
        .collect();
    ctx.hooks.on_embed_start(texts.len());
    let embed_start = Instant::now();
    let embeddings = embedder
        .embed(&texts.iter().map(String::as_str).collect::<Vec<_>>())
        .map_err(Error::from)?;
    ctx.hooks.on_embed_done(embed_start.elapsed());
    Ok(embeddings)
}

/// `chunks[bestIdx]?.text || cand.body` upstream — an empty chunk
/// text still falls back to the body, and a file with no chunks at
/// all reports `(body, 0)`.
fn selected_chunk(
    cand: &RankedResult,
    selection: Option<&(Vec<crate::chunk::Chunk<'_>>, usize)>,
) -> (String, usize) {
    selection
        .and_then(|(chunks, best)| chunks.get(*best))
        .map_or_else(
            || (cand.body.clone(), 0),
            |c| {
                if c.text.is_empty() {
                    (cand.body.clone(), c.pos)
                } else {
                    (c.text.to_owned(), c.pos)
                }
            },
        )
}

/// T14 — everything after `candidates` is identical between
/// `hybridQuery` and `structuredSearch` (upstream steps 5-8 /
/// 4-7).
fn run_pipeline(
    ctx: &SearchCtx<'_>,
    candidates: &[RankedResult],
    opts: &PipelineOpts<'_>,
) -> Result<Vec<HybridResult>> {
    let chunk_opts = CharChunkOptions::default();
    let doc_chunks = select_best_chunks(
        candidates,
        opts.query,
        opts.intent,
        opts.chunk_strategy,
        &chunk_opts,
    );
    let trace_of = |file: &str| opts.trace.as_ref().and_then(|t| t.get(file));

    if opts.skip_rerank {
        let mut seen = std::collections::HashSet::new();
        let results: Vec<HybridResult> = candidates
            .iter()
            .enumerate()
            .map(|(i, cand)| {
                let rrf_rank = i + 1;
                let rrf_score = 1.0 / rrf_rank as f64;
                let (best_chunk, best_chunk_pos) =
                    selected_chunk(cand, doc_chunks.get(cand.file.as_str()));
                HybridResult {
                    filepath: cand.file.clone(),
                    display_path: cand.display_path.clone(),
                    title: cand.title.clone(),
                    body: cand.body.clone(),
                    best_chunk,
                    best_chunk_pos,
                    score: rrf_score,
                    context: cand.context.clone(),
                    docid: cand.docid.clone(),
                    // Filled by `attach_result_metadata` after ranking —
                    // upstream never carries metadata through the lists.
                    metadata: DocumentMetadata::default(),
                    explain: opts.explain.then(|| {
                        Box::new(build_explain(
                            trace_of(&cand.file),
                            rrf_rank,
                            rrf_score,
                            1.0,
                            0.0,
                            rrf_score,
                        ))
                    }),
                }
            })
            .filter(|r| seen.insert(r.filepath.clone()))
            .filter(|r| r.score >= opts.min_score)
            .take(opts.limit)
            .collect();
        return Ok(results);
    }

    // Rerank on chunks only (upstream step 6 — `chunksToRerank` skips
    // candidates without chunks entirely, which also drops them from
    // the final list).
    let chunks_to_rerank: Vec<(String, String)> = candidates
        .iter()
        .filter_map(|cand| {
            doc_chunks
                .get(cand.file.as_str())
                .and_then(|(chunks, best)| chunks.get(*best))
                .map(|c| (cand.file.clone(), c.text.to_owned()))
        })
        .collect();
    ctx.hooks.on_rerank_start(chunks_to_rerank.len());
    let rerank_start = Instant::now();
    let reranked = rerank(ctx, opts.query, opts.intent, &chunks_to_rerank)?;
    ctx.hooks.on_rerank_done(rerank_start.elapsed());

    let rrf_rank_of = |file: &str| -> usize {
        candidates
            .iter()
            .position(|c| c.file == file)
            .map_or(opts.candidate_limit, |i| i + 1)
    };
    let mut blended: Vec<HybridResult> = reranked
        .into_iter()
        .filter_map(|(file, rerank_score)| {
            let cand = candidates.iter().find(|c| c.file == file)?;
            let rrf_rank = rrf_rank_of(&file);
            let rrf_weight = if rrf_rank <= 3 {
                0.75
            } else if rrf_rank <= 10 {
                0.60
            } else {
                0.40
            };
            let position_score = 1.0 / rrf_rank as f64;
            let blended_score =
                f64::mul_add(1.0 - rrf_weight, rerank_score, rrf_weight * position_score);
            let (best_chunk, best_chunk_pos) = selected_chunk(cand, doc_chunks.get(file.as_str()));
            Some(HybridResult {
                filepath: file.clone(),
                display_path: cand.display_path.clone(),
                title: cand.title.clone(),
                body: cand.body.clone(),
                best_chunk,
                best_chunk_pos,
                score: blended_score,
                context: cand.context.clone(),
                docid: cand.docid.clone(),
                metadata: DocumentMetadata::default(),
                explain: opts.explain.then(|| {
                    Box::new(build_explain(
                        trace_of(&file),
                        rrf_rank,
                        position_score,
                        rrf_weight,
                        rerank_score,
                        blended_score,
                    ))
                }),
            })
        })
        .collect();
    blended.sort_by(|a, b| b.score.total_cmp(&a.score));

    let mut seen = std::collections::HashSet::new();
    let results: Vec<HybridResult> = blended
        .into_iter()
        .filter(|r| seen.insert(r.filepath.clone()))
        .filter(|r| r.score >= opts.min_score)
        .take(opts.limit)
        .collect();
    Ok(results)
}

// =============================================================================
// Hybrid query — upstream `hybridQuery` (store.ts:5538-5837).
// =============================================================================

/// `hybridQuery` options — populated from `SearchOptions` in `qmd.rs`.
pub(crate) struct HybridQueryOpts<'a> {
    /// Collection scope (`Option` = all collections).
    pub collections: Option<Vec<String>>,
    /// Metadata filter applied to every retrieval call (upstream
    /// `options.filter`, store.ts:5466).
    pub filter: Option<&'a MetadataFilter>,
    /// See [`PipelineOpts`].
    pub intent: Option<&'a str>,
    /// See [`PipelineOpts`].
    pub skip_rerank: bool,
    /// See [`PipelineOpts`].
    pub explain: bool,
    /// See [`PipelineOpts`].
    pub min_score: f64,
    /// See [`PipelineOpts`].
    pub limit: usize,
    /// See [`PipelineOpts`].
    pub candidate_limit: usize,
    /// See [`PipelineOpts`].
    pub chunk_strategy: ChunkStrategy,
}

/// BM25 strong-signal bypass thresholds — upstream
/// `STRONG_SIGNAL_MIN_SCORE`/`STRONG_SIGNAL_MIN_GAP` (store.ts:426-427).
const STRONG_SIGNAL_MIN_SCORE: f64 = 0.85;
const STRONG_SIGNAL_MIN_GAP: f64 = 0.15;

/// Default `rerankCandidateLimit` — upstream `RERANK_CANDIDATE_LIMIT`
/// (store.ts:430).
pub(crate) const RERANK_CANDIDATE_LIMIT: usize = 40;

/// Per-list hit limit — upstream `searchFTS`/`searchVec` are called
/// with `20` inside `hybridQuery`/`structuredSearch`.
const SUBQUERY_LIMIT: usize = 20;

/// BM25 strong-signal check — upstream store.ts:5564-5570. An intent
/// disables the bypass (upstream comment 5561-5563).
fn has_strong_signal(top: f64, second: f64, hits: usize, intent: Option<&str>) -> bool {
    intent.is_none()
        && hits > 0
        && top >= STRONG_SIGNAL_MIN_SCORE
        && (top - second) >= STRONG_SIGNAL_MIN_GAP
}

/// `hybridQuery` steps 1-3 — probe/expand/retrieve into ranked
/// lists (upstream store.ts:5564-5671).
fn gather_hybrid_lists(
    ctx: &SearchCtx<'_>,
    query: &str,
    opts: &HybridQueryOpts<'_>,
    has_vectors: bool,
) -> Result<(Vec<Vec<RankedResult>>, Vec<RankedListMeta>)> {
    let mut ranked_lists: Vec<Vec<RankedResult>> = Vec::new();
    let mut ranked_meta: Vec<RankedListMeta> = Vec::new();

    // Step 1: BM25 probe — strong signal skips expensive LLM expansion.
    // Intent disables the bypass (upstream comment store.ts:5561-5563).
    let initial_fts = search_fts(
        &ctx.db(),
        query,
        SUBQUERY_LIMIT,
        opts.collections.clone(),
        opts.filter,
    )?;
    let top_score = initial_fts.first().map_or(0.0, |r| r.score);
    let second_score = initial_fts.get(1).map_or(0.0, |r| r.score);
    let strong_signal = has_strong_signal(top_score, second_score, initial_fts.len(), opts.intent);
    if strong_signal {
        ctx.hooks.on_strong_signal(top_score);
    }

    // Step 2: expand (or skip on strong signal).
    ctx.hooks.on_expand_start();
    let expand_start = Instant::now();
    let (expanded, expand_model) = if strong_signal {
        (Vec::new(), String::new())
    } else {
        expand_query(ctx, query)?
    };
    ctx.hooks
        .on_expand(query, &expanded, expand_start.elapsed());

    if !initial_fts.is_empty() {
        ranked_lists.push(initial_fts.iter().map(search_result_to_ranked).collect());
        ranked_meta.push(RankedListMeta {
            source: SearchSource::Fts,
            query_type: QueryKind::Original,
            query: query.to_owned(),
        });
    }

    // Step 3a: FTS for lex expansions.
    for q in &expanded {
        if q.kind == QueryKind::Lex {
            let hits = search_fts(
                &ctx.db(),
                &q.query,
                SUBQUERY_LIMIT,
                opts.collections.clone(),
                opts.filter,
            )?;
            if !hits.is_empty() {
                ranked_lists.push(hits.iter().map(search_result_to_ranked).collect());
                ranked_meta.push(RankedListMeta {
                    source: SearchSource::Fts,
                    query_type: QueryKind::Lex,
                    query: q.query.clone(),
                });
            }
        }
    }

    // Step 3b: batch-embed original + vec/hyde, then sqlite-vec lookups.
    if has_vectors {
        let mut vec_queries: Vec<(&str, QueryKind)> = vec![(query, QueryKind::Original)];
        vec_queries.extend(
            expanded
                .iter()
                .filter(|q| matches!(q.kind, QueryKind::Vec | QueryKind::Hyde))
                .map(|q| (q.query.as_str(), q.kind)),
        );
        let texts: Vec<&str> = vec_queries.iter().map(|(t, _)| *t).collect();
        let embeddings = embed_queries(ctx, &texts)?;

        for (i, (text, query_type)) in vec_queries.iter().enumerate() {
            // Upstream `if (!embedding) continue` (store.ts:5638-5639)
            // — a null/empty embed result skips that query.
            let Some(embedding) = embeddings.get(i).filter(|v| !v.is_empty()) else {
                continue;
            };
            let hits = super::search::search_vec_with_embedding(
                &ctx.db(),
                embedding,
                SUBQUERY_LIMIT,
                opts.collections.clone(),
                opts.filter,
            )?;
            if hits.is_empty() {
                continue;
            }
            ranked_lists.push(hits.iter().map(search_result_to_ranked).collect());
            ranked_meta.push(RankedListMeta {
                source: SearchSource::Vec,
                query_type: *query_type,
                query: (*text).to_owned(),
            });
        }
    }

    // Step 3c: dud eviction — drop a cached expansion whose runnable
    // sub-queries all came back empty (upstream 5660-5671).
    if !expanded.is_empty() {
        let runnable = expanded
            .iter()
            .filter(|q| q.kind == QueryKind::Lex || has_vectors)
            .count();
        let contributed = ranked_meta
            .iter()
            .any(|m| m.query_type != QueryKind::Original);
        if runnable > 0 && !contributed {
            delete_expansion_cache_entry(&ctx.db(), query, &expand_model)?;
        }
    }

    Ok((ranked_lists, ranked_meta))
}

pub(crate) fn hybrid_query(
    ctx: &SearchCtx<'_>,
    query: &str,
    opts: &HybridQueryOpts<'_>,
) -> Result<Vec<HybridResult>> {
    let has_vectors = has_vectors_vec(&ctx.db())?;
    let (ranked_lists, ranked_meta) = gather_hybrid_lists(ctx, query, opts, has_vectors)?;

    // Step 4: RRF fusion.
    let weights = get_hybrid_rrf_weights(&ranked_meta);
    let fused = reciprocal_rank_fusion(&ranked_lists, &weights, RRF_K);
    let trace = opts
        .explain
        .then(|| build_rrf_trace(&ranked_lists, &weights, &ranked_meta, RRF_K));
    let (candidates, _) = fused.split_at(fused.len().min(opts.candidate_limit));
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let mut results = run_pipeline(
        ctx,
        candidates,
        &PipelineOpts {
            query,
            intent: opts.intent,
            skip_rerank: opts.skip_rerank,
            explain: opts.explain,
            min_score: opts.min_score,
            limit: opts.limit,
            candidate_limit: opts.candidate_limit,
            chunk_strategy: opts.chunk_strategy,
            trace,
        },
    )?;
    attach_result_metadata(&ctx.db(), &mut results)?;
    Ok(results)
}

// =============================================================================
// Structured search — upstream `structuredSearch` (store.ts:5960-6238).
// =============================================================================

/// `structuredSearch` options.
pub(crate) struct StructuredSearchOpts<'a> {
    /// Collection scope — `None` = all collections (upstream
    /// `collections ?? [undefined]`); `Some` runs one sub-list per
    /// name. `Qmd::search` normalizes `Some(vec![])` to `None` before
    /// this point (upstream `[]` → `undefined`, index.ts:436-449);
    /// internally `Some(vec![])` still yields no lists, mirroring
    /// `structuredSearch`'s own `[]`.
    pub collections: Option<Vec<String>>,
    /// Metadata filter applied to every retrieval call (upstream
    /// `options.filter`, store.ts:5929).
    pub filter: Option<&'a MetadataFilter>,
    /// See [`PipelineOpts`].
    pub intent: Option<&'a str>,
    /// See [`PipelineOpts`].
    pub skip_rerank: bool,
    /// See [`PipelineOpts`].
    pub explain: bool,
    /// See [`PipelineOpts`].
    pub min_score: f64,
    /// See [`PipelineOpts`].
    pub limit: usize,
    /// See [`PipelineOpts`].
    pub candidate_limit: usize,
    /// See [`PipelineOpts`].
    pub chunk_strategy: ChunkStrategy,
}

/// Step 1 of `structuredSearch` — one FTS list per
/// (lex search × collection); trace meta records each search's query
/// text (upstream store.ts:6007-6027).
fn push_lex_lists(
    ctx: &SearchCtx<'_>,
    searches: &[ExpandedQuery],
    collection_list: &[Option<String>],
    filter: Option<&MetadataFilter>,
    ranked_lists: &mut Vec<Vec<RankedResult>>,
    ranked_meta: &mut Vec<RankedListMeta>,
) -> Result<()> {
    for search in searches.iter().filter(|s| s.kind == QueryKind::Lex) {
        for coll in collection_list {
            let scope = coll.clone().map(|c| vec![c]);
            let hits = search_fts(&ctx.db(), &search.query, SUBQUERY_LIMIT, scope, filter)?;
            if hits.is_empty() {
                continue;
            }
            ranked_lists.push(hits.iter().map(search_result_to_ranked).collect());
            ranked_meta.push(RankedListMeta {
                source: SearchSource::Fts,
                query_type: QueryKind::Lex,
                query: search.query.clone(),
            });
        }
    }
    Ok(())
}

/// Step 2 of `structuredSearch` — one vec list per
/// (search × collection) over pre-computed embeddings.
fn push_structured_vec_lists(
    ctx: &SearchCtx<'_>,
    vec_searches: &[&ExpandedQuery],
    collection_list: &[Option<String>],
    embeddings: &[Vec<f32>],
    filter: Option<&MetadataFilter>,
    ranked_lists: &mut Vec<Vec<RankedResult>>,
    ranked_meta: &mut Vec<RankedListMeta>,
) -> Result<()> {
    for (i, s) in vec_searches.iter().enumerate() {
        // Upstream `if (!embedding) continue` (store.ts:6044-6045).
        let Some(embedding) = embeddings.get(i).filter(|v| !v.is_empty()) else {
            continue;
        };
        for coll in collection_list {
            let hits = search_vec_multi(
                &ctx.db(),
                embedding,
                SUBQUERY_LIMIT,
                coll.as_deref(),
                filter,
            )?;
            if hits.is_empty() {
                continue;
            }
            ranked_lists.push(hits.iter().map(search_result_to_ranked).collect());
            ranked_meta.push(RankedListMeta {
                source: SearchSource::Vec,
                query_type: s.kind,
                query: s.query.clone(),
            });
        }
    }
    Ok(())
}

pub(crate) fn structured_search(
    ctx: &SearchCtx<'_>,
    searches: &[ExpandedQuery],
    opts: &StructuredSearchOpts<'_>,
) -> Result<Vec<HybridResult>> {
    if searches.is_empty() {
        return Ok(Vec::new());
    }

    // Validate queries before executing (store.ts:5978-5995) — the
    // `Line N` / `Structured search` location prefix is verbatim.
    for search in searches {
        // Upstream `search.line ? `Line ${n}` : 'Structured search'`
        // (store.ts:5980) — `0` is falsy, so only positive line numbers
        // are treated as source locations.
        let location = search
            .line
            .filter(|&n| n > 0)
            .map_or_else(|| "Structured search".to_owned(), |n| format!("Line {n}"));
        if search.query.contains('\r') || search.query.contains('\n') {
            return Err(Error::InvalidQuery {
                reason: format!(
                    "{location} ({}): queries must be single-line. Remove newline characters.",
                    search.kind.as_str()
                ),
            });
        }
        let error = match search.kind {
            // `Original` is ranked-list metadata, not a valid
            // structured sub-query kind.
            QueryKind::Original => {
                return Err(Error::InvalidQuery {
                    reason: format!(
                        "{location} (original): structured searches must be lex, vec or hyde."
                    ),
                });
            }
            QueryKind::Lex => super::fts::validate_lex_query(&search.query),
            QueryKind::Vec | QueryKind::Hyde => super::fts::validate_semantic_query(&search.query),
        };
        if let Some(error) = error {
            return Err(Error::InvalidQuery {
                reason: format!("{location} ({}): {error}", search.kind.as_str()),
            });
        }
    }

    let mut ranked_lists: Vec<Vec<RankedResult>> = Vec::new();
    let mut ranked_meta: Vec<RankedListMeta> = Vec::new();
    let has_vectors = has_vectors_vec(&ctx.db())?;

    // Upstream `collections ?? [undefined]` (store.ts:5999) — `None`
    // maps to a single unscoped sub-list. `Qmd::search` normalizes
    // `Some(vec![])` to `None` (upstream `[]` → `undefined`); a
    // direct internal `Some(vec![])` still yields no sub-lists, like
    // upstream `[]` reaching `structuredSearch`.
    let collection_list: Vec<Option<String>> = opts
        .collections
        .as_ref()
        .map_or_else(|| vec![None], |cs| cs.iter().cloned().map(Some).collect());

    // Step 1: FTS for all lex searches — one list per (search × collection).
    push_lex_lists(
        ctx,
        searches,
        &collection_list,
        opts.filter,
        &mut ranked_lists,
        &mut ranked_meta,
    )?;

    // Step 2: batch-embed vec/hyde, then sqlite-vec lookups — one list
    // per (search × collection).
    if has_vectors {
        let vec_searches: Vec<&ExpandedQuery> = searches
            .iter()
            .filter(|s| matches!(s.kind, QueryKind::Vec | QueryKind::Hyde))
            .collect();
        if !vec_searches.is_empty() {
            let texts: Vec<&str> = vec_searches.iter().map(|s| s.query.as_str()).collect();
            let embeddings = embed_queries(ctx, &texts)?;
            push_structured_vec_lists(
                ctx,
                &vec_searches,
                &collection_list,
                &embeddings,
                opts.filter,
                &mut ranked_lists,
                &mut ranked_meta,
            )?;
        }
    }

    if ranked_lists.is_empty() {
        return Ok(Vec::new());
    }

    // Step 3: first list gets 2x weight (upstream 6072).
    let weights: Vec<f64> = ranked_lists
        .iter()
        .enumerate()
        .map(|(i, _)| if i == 0 { 2.0 } else { 1.0 })
        .collect();
    let fused = reciprocal_rank_fusion(&ranked_lists, &weights, RRF_K);
    let trace = opts
        .explain
        .then(|| build_rrf_trace(&ranked_lists, &weights, &ranked_meta, RRF_K));
    let (candidates, _) = fused.split_at(fused.len().min(opts.candidate_limit));
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    ctx.hooks.on_expand("", &[], Duration::ZERO); // Signal no expansion.

    // Step 4: first lex query is the chunk-select/rerank query, else
    // first vec, else the first search (upstream 6083-6085). The `||`
    // chain skips empty strings too.
    let primary_query = [
        searches.iter().find(|s| s.kind == QueryKind::Lex),
        searches.iter().find(|s| s.kind == QueryKind::Vec),
        searches.first(),
    ]
    .into_iter()
    .flatten()
    .map(|s| s.query.as_str())
    .find(|q| !q.is_empty())
    .unwrap_or("");

    let mut results = run_pipeline(
        ctx,
        candidates,
        &PipelineOpts {
            query: primary_query,
            intent: opts.intent,
            skip_rerank: opts.skip_rerank,
            explain: opts.explain,
            min_score: opts.min_score,
            limit: opts.limit,
            candidate_limit: opts.candidate_limit,
            chunk_strategy: opts.chunk_strategy,
            trace,
        },
    )?;
    attach_result_metadata(&ctx.db(), &mut results)?;
    Ok(results)
}

// =============================================================================
// Vector query — upstream `vectorSearchQuery` (store.ts:5868-5917).
// =============================================================================

/// `vectorSearchQuery` options — upstream's `intent` parameter is
/// ignored there too (quirk Q9), so the Rust API takes none.
pub(crate) struct VectorQueryOpts<'a> {
    /// Collection scope (`Option` = all).
    pub collections: Option<Vec<String>>,
    /// Metadata filter applied to every `searchVec` call (upstream
    /// `options.filter`, store.ts:5841).
    pub filter: Option<&'a MetadataFilter>,
    /// `limit` (default 10 upstream).
    pub limit: usize,
    /// `minScore` (default 0.3 upstream).
    pub min_score: f64,
}

/// Upstream `VectorSearchResult` (store.ts:5906-5915).
#[derive(Debug)]
#[non_exhaustive]
pub struct VectorQueryResult {
    /// `qmd://collection/path` URI (upstream `file`).
    pub filepath: String,
    /// `collection/path` for display.
    pub display_path: String,
    /// Document title.
    pub title: String,
    /// Full body.
    pub body: String,
    /// `1 - cosine distance`.
    pub score: f64,
    /// Folder context.
    pub context: Option<String>,
    /// Short docid.
    pub docid: String,
    /// Canonical metadata (`{}` when absent) — carried through from the
    /// `searchVec` rows (upstream `r.metadata`).
    pub metadata: DocumentMetadata,
}

pub(crate) fn vector_query(
    ctx: &SearchCtx<'_>,
    query: &str,
    opts: &VectorQueryOpts<'_>,
) -> Result<Vec<VectorQueryResult>> {
    let has_vectors = has_vectors_vec(&ctx.db())?;
    if !has_vectors {
        return Ok(Vec::new());
    }

    // Expand — vec/hyde only (lex targets FTS, upstream 5884-5888).
    let expand_start = Instant::now();
    let (all_expanded, _) = expand_query(ctx, query)?;
    let vec_expanded: Vec<ExpandedQuery> = all_expanded
        .into_iter()
        .filter(|q| q.kind != QueryKind::Lex)
        .collect();
    ctx.hooks
        .on_expand(query, &vec_expanded, expand_start.elapsed());

    let embedder = (ctx.backends.embedder)()?;
    let query_texts: Vec<&str> = std::iter::once(query)
        .chain(vec_expanded.iter().map(|q| q.query.as_str()))
        .collect();
    // Sequential — upstream comment: "concurrent embed() hangs"
    // (store.ts:5890); sequential is our only mode anyway.
    let mut all_results: IndexMap<String, VectorQueryResult> = IndexMap::new();
    for q in &query_texts {
        // Embed unlocked — upstream `searchVec` order:
        // vectors exist (checked above), embed, then search.
        // `if (!embedding) continue` — store.ts:5893-5894.
        let Some(embedding) = super::search::embed_query(embedder.as_ref(), q)? else {
            continue;
        };
        let hits = super::search::search_vec_with_embedding(
            &ctx.db(),
            &embedding,
            opts.limit,
            opts.collections.clone(),
            opts.filter,
        )?;
        for r in &hits {
            match all_results.get_mut(&r.filepath) {
                Some(existing) if r.score > existing.score => {
                    *existing = VectorQueryResult {
                        filepath: r.filepath.clone(),
                        display_path: r.display_path.clone(),
                        title: r.title.clone(),
                        body: r.body.clone(),
                        score: r.score,
                        context: r.context.clone(),
                        docid: r.docid.clone(),
                        metadata: r.metadata.clone(),
                    };
                }
                Some(_) => {}
                None => {
                    all_results.insert(
                        r.filepath.clone(),
                        VectorQueryResult {
                            filepath: r.filepath.clone(),
                            display_path: r.display_path.clone(),
                            title: r.title.clone(),
                            body: r.body.clone(),
                            score: r.score,
                            context: r.context.clone(),
                            docid: r.docid.clone(),
                            metadata: r.metadata.clone(),
                        },
                    );
                }
            }
        }
    }

    let mut results: Vec<VectorQueryResult> = all_results.into_values().collect();
    results.sort_by(|a, b| b.score.total_cmp(&a.score));
    results.retain(|r| r.score >= opts.min_score);
    results.truncate(opts.limit);
    Ok(results)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::missing_assert_message,
    clippy::panic,
    clippy::shadow_unrelated,
    reason = "idiomatic test-code patterns"
)]
mod tests {
    //! Ports of `intent.test.ts` "intent keyword extraction logic"
    //! (300-363), "strong-signal bypass logic" (365-394) and "intent
    //! constants" (505-512) — upstream re-implements the scoring in
    //! the test; here the assertions run against the real
    //! [`selection_terms`]/[`chunk_score`]/[`has_strong_signal`].
    use super::*;

    // Same chunk texts as intent.test.ts:312-317.
    const CHUNKS: &[&str] = &[
        "Web performance: optimize page load times, reduce latency, improve rendering pipeline.",
        "Team performance: build trust, give feedback, set clear expectations for the group.",
        "Health performance: exercise regularly, sleep 8 hours, manage stress for endurance.",
    ];

    fn score(text: &str, query: &str, intent: Option<&str>) -> f64 {
        let (query_terms, intent_terms) = selection_terms(query, intent);
        chunk_score(&text.to_lowercase(), &query_terms, &intent_terms)
    }

    #[test]
    fn intent_keyword_extraction_logic() {
        // Without intent all chunks score equally on "performance".
        let scores: Vec<f64> = CHUNKS
            .iter()
            .map(|c| score(c, "performance", None))
            .collect();
        assert_eq!(scores[0], scores[1]);
        assert_eq!(scores[1], scores[2]);

        // Web intent → web chunk wins.
        let intent = "looking for notes about page load times and latency optimization";
        let scores: Vec<f64> = CHUNKS
            .iter()
            .map(|c| score(c, "performance", Some(intent)))
            .collect();
        assert!(scores[0] > scores[1]);
        assert!(scores[0] > scores[2]);

        // Health intent → health chunk wins.
        let intent = "looking for notes about exercise, sleep, and endurance";
        let scores: Vec<f64> = CHUNKS
            .iter()
            .map(|c| score(c, "performance", Some(intent)))
            .collect();
        assert!(scores[2] > scores[0]);
        assert!(scores[2] > scores[1]);

        // Intent terms weigh INTENT_WEIGHT_CHUNK against 1.0 per query
        // term — chunk 0: "performance" (query) + "latency" (intent).
        let intent = "looking for latency";
        let with_both = score(CHUNKS[0], "performance", Some(intent));
        let query_only = score(CHUNKS[0], "performance", None);
        assert_eq!(with_both, query_only + INTENT_WEIGHT_CHUNK);

        // Stop words filtered; "art"/"web"/"performance" survive —
        // chunk 0 has web+performance, chunks 1-2 performance only.
        let intent = "the art of web performance";
        let scores: Vec<f64> = CHUNKS
            .iter()
            .map(|c| score(c, "test", Some(intent)))
            .collect();
        assert_eq!(scores[0], INTENT_WEIGHT_CHUNK * 2.0);
        assert_eq!(scores[1], INTENT_WEIGHT_CHUNK);
        assert_eq!(scores[2], INTENT_WEIGHT_CHUNK);
    }

    #[test]
    fn selection_terms_use_js_whitespace() {
        // Upstream splits on /\s+/ (store.ts:5684, 6086): U+FEFF is a
        // separator, U+0085 (NEL) is not.
        assert_eq!(score("alpha bravo", "alpha\u{feff}bravo", None), 2.0);
        assert_eq!(score("alpha bravo", "alpha\u{0085}bravo", None), 0.0);
        assert_eq!(
            score("alpha bravo", "", Some("ctx\u{feff}alpha")),
            INTENT_WEIGHT_CHUNK
        );
    }

    #[test]
    fn strong_signal_bypass_logic() {
        // Exact cases from intent.test.ts:375-393 (upstream's local
        // helper has no hit-count check; ours also requires hits > 0).
        assert!(has_strong_signal(0.90, 0.70, 1, None));
        assert!(!has_strong_signal(
            0.90,
            0.70,
            1,
            Some("looking for health performance")
        ));
        assert!(!has_strong_signal(0.50, 0.45, 1, None));
        assert!(!has_strong_signal(0.50, 0.45, 1, Some("some intent")));
        // Gap below 0.15 is not strong even without intent.
        assert!(!has_strong_signal(0.90, 0.80, 1, None));
        // No hits is never strong.
        assert!(!has_strong_signal(0.90, 0.70, 0, None));
    }

    #[test]
    fn intent_constants() {
        assert_eq!(crate::query::INTENT_WEIGHT_SNIPPET, 0.3);
        assert_eq!(INTENT_WEIGHT_CHUNK, 0.5);
    }
}

// =============================================================================
// AC6: upstream eval.test.ts "Vector Search" + "Hybrid Search (RRF)"
// threshold groups on the vendored eval-docs corpus (real models).
// =============================================================================

#[cfg(all(test, feature = "llama"))]
#[allow(
    clippy::excessive_nesting,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_assert_message,
    clippy::cast_possible_truncation,
    clippy::tests_outside_test_module,
    clippy::too_many_lines,
    clippy::significant_drop_tightening,
    clippy::shadow_unrelated,
    reason = "idiomatic test-code patterns"
)]
mod eval {
    //! The upstream eval drives `searchVec` + a manual two-list
    //! `reciprocalRankFusion` (no expansion/rerank). `Qmd::search_vector`
    //! is the same vec path, `Qmd::search_lex` the same FTS path, and the
    //! fusion below reuses the crate's `reciprocal_rank_fusion` with the
    //! same default-weight call shape.

    use std::sync::Arc;

    use super::*;
    use crate::llm::DEFAULT_EMBED_MODEL;
    use crate::llm::download::model_cache_dir;
    use crate::llm::llamacpp::{EmbeddingMode, LlamaEmbedder};
    use crate::{CollectionSpec, EmbedOptions, LexOptions, UpdateOptions, VectorOptions};

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Level {
        Easy,
        Medium,
        Hard,
        Fusion,
    }

    struct EvalQuery {
        query: &'static str,
        doc: &'static str,
        level: Level,
    }

    const fn q(query: &'static str, doc: &'static str, level: Level) -> EvalQuery {
        EvalQuery { query, doc, level }
    }

    // Upstream evalQueries — same queries, docs and difficulty bands
    // (test/eval.test.ts:43-80).
    const EVAL_QUERIES: &[EvalQuery] = &[
        q("API versioning", "api-design", Level::Easy),
        q("Series A fundraising", "fundraising", Level::Easy),
        q("CAP theorem", "distributed-systems", Level::Easy),
        q(
            "overfitting machine learning",
            "machine-learning",
            Level::Easy,
        ),
        q("remote work VPN", "remote-work", Level::Easy),
        q(
            "Project Phoenix retrospective",
            "product-launch",
            Level::Easy,
        ),
        q(
            "how to structure REST endpoints",
            "api-design",
            Level::Medium,
        ),
        q("raising money for startup", "fundraising", Level::Medium),
        q(
            "consistency vs availability tradeoffs",
            "distributed-systems",
            Level::Medium,
        ),
        q(
            "how to prevent models from memorizing data",
            "machine-learning",
            Level::Medium,
        ),
        q("working from home guidelines", "remote-work", Level::Medium),
        q(
            "what went wrong with the launch",
            "product-launch",
            Level::Medium,
        ),
        q("nouns not verbs", "api-design", Level::Hard),
        q("Sequoia investor pitch", "fundraising", Level::Hard),
        q(
            "Raft algorithm leader election",
            "distributed-systems",
            Level::Hard,
        ),
        q("F1 score precision recall", "machine-learning", Level::Hard),
        q(
            "quarterly team gathering travel",
            "remote-work",
            Level::Hard,
        ),
        q("beta program 47 bugs", "product-launch", Level::Hard),
        q(
            "how much runway before running out of money",
            "fundraising",
            Level::Fusion,
        ),
        q(
            "datacenter replication sync strategy",
            "distributed-systems",
            Level::Fusion,
        ),
        q(
            "splitting data for training and testing",
            "machine-learning",
            Level::Fusion,
        ),
        q(
            "JSON response codes error messages",
            "api-design",
            Level::Fusion,
        ),
        q(
            "video calls camera async messaging",
            "remote-work",
            Level::Fusion,
        ),
        q(
            "CI/CD pipeline testing coverage",
            "product-launch",
            Level::Fusion,
        ),
    ];

    fn hit(filepath: &str, doc: &str) -> bool {
        filepath.to_lowercase().contains(doc)
    }

    fn hit_rate(results: &[Vec<String>], queries: &[&EvalQuery], k: usize) -> f64 {
        let hits = results
            .iter()
            .zip(queries)
            .filter(|(files, q)| files[..k.min(files.len())].iter().any(|f| hit(f, q.doc)))
            .count();
        hits as f64 / queries.len() as f64
    }

    /// AC6: `eval-docs` corpus, real embeddinggemma — Vector and
    /// Hybrid(RRF) threshold groups of `eval.test.ts` in one run.
    /// Run with `cargo test -p qmd --features llama -- --ignored eval`.
    #[test]
    #[ignore = "requires cached GGUF models"]
    fn vector_and_hybrid_hit_rates() {
        let model_file = model_cache_dir(&crate::Environment::default())
            .join("hf_ggml-org_embeddinggemma-300M-Q8_0.gguf");
        assert!(
            model_file.exists(),
            "model cache missing: {}",
            model_file.display()
        );

        let tmp = tempfile::tempdir().unwrap();
        let corpus = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/eval-docs");
        let db = tmp.path().join("i.sqlite");
        let embedder = LlamaEmbedder::new(
            &model_file,
            DEFAULT_EMBED_MODEL,
            &crate::Environment::default(),
            EmbeddingMode::UpstreamTail,
        )
        .unwrap();
        let qmd = crate::Qmd::builder(&db)
            .embedder(Arc::new(embedder))
            .build()
            .unwrap();
        qmd.add_collection(
            "eval",
            &CollectionSpec {
                path: corpus.to_owned(),
                ..CollectionSpec::default()
            },
        )
        .unwrap();
        qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();
        let report = qmd.embed(&EmbedOptions::default(), &mut |_| {}).unwrap();
        assert_eq!(report.errors, 0, "failures: {:?}", report.failures);

        let vec_opts = VectorOptions {
            limit: Some(20),
            ..VectorOptions::default()
        };
        let lex_opts = LexOptions {
            limit: Some(20),
            ..LexOptions::default()
        };

        // Per-query result lists: (vec files, fts files, fused files).
        let results: Vec<(Vec<String>, Vec<String>, Vec<String>)> = EVAL_QUERIES
            .iter()
            .map(|eq| {
                let vec_files: Vec<String> = qmd
                    .search_vector(eq.query, &vec_opts)
                    .unwrap()
                    .iter()
                    .map(|r| r.filepath.clone())
                    .collect();
                let fts_files: Vec<String> = qmd
                    .search_lex(eq.query, &lex_opts)
                    .unwrap()
                    .iter()
                    .map(|r| r.filepath.clone())
                    .collect();
                // Upstream eval hybrid: FTS list + vec list, default
                // weights (all 1.0), k=60.
                let mut lists: Vec<Vec<RankedResult>> = Vec::new();
                let fts = {
                    let conn = qmd.lock_conn();
                    search_fts(&conn, eq.query, 20, None, None).unwrap()
                };
                if !fts.is_empty() {
                    lists.push(fts.iter().map(search_result_to_ranked).collect());
                }
                if !vec_files.is_empty() {
                    let vec_hits = qmd.search_vector(eq.query, &vec_opts).unwrap();
                    lists.push(
                        vec_hits
                            .iter()
                            .map(|r| RankedResult {
                                file: r.filepath.clone(),
                                display_path: r.display_path.clone(),
                                title: r.title.clone(),
                                body: r.body.clone(),
                                score: r.score,
                                docid: r.docid.clone(),
                                context: r.context.clone(),
                            })
                            .collect(),
                    );
                }
                let fused = reciprocal_rank_fusion(&lists, &[], RRF_K);
                let fused_files: Vec<String> = fused.iter().map(|r| r.file.clone()).collect();
                (vec_files, fts_files, fused_files)
            })
            .collect();

        let by_level = |level: Level| -> Vec<&EvalQuery> {
            EVAL_QUERIES.iter().filter(|e| e.level == level).collect()
        };
        // Reorder-safe accessors by level.
        let band = |level: Level, idx: usize| -> (Vec<&EvalQuery>, Vec<Vec<String>>) {
            let qs = by_level(level);
            let lists: Vec<Vec<String>> = EVAL_QUERIES
                .iter()
                .enumerate()
                .filter(|(_, e)| e.level == level)
                .map(|(i, _)| match idx {
                    0 => results[i].0.clone(),
                    1 => results[i].1.clone(),
                    _ => results[i].2.clone(),
                })
                .collect();
            (qs, lists)
        };

        // ── Vector Search group ──
        for (level, k, threshold) in [
            (Level::Easy, 3, 0.6),
            (Level::Medium, 3, 0.4),
            (Level::Hard, 5, 0.3),
        ] {
            let (qs, lists) = band(level, 0);
            let rate = hit_rate(&lists, &qs, k);
            assert!(
                rate >= threshold,
                "vector {level:?} Hit@{k} {rate} < {threshold}"
            );
        }
        let all: Vec<&EvalQuery> = EVAL_QUERIES.iter().collect();
        let all_vec: Vec<Vec<String>> = results.iter().map(|r| r.0.clone()).collect();
        let vec_overall = hit_rate(&all_vec, &all, 3);
        assert!(
            vec_overall >= 0.5,
            "vector overall Hit@3 {vec_overall} < 0.5"
        );

        // ── Hybrid (RRF) group ──
        for (level, k, threshold) in [
            (Level::Easy, 3, 0.8),
            (Level::Medium, 3, 0.5),
            (Level::Hard, 5, 0.35),
        ] {
            let (qs, lists) = band(level, 2);
            let rate = hit_rate(&lists, &qs, k);
            assert!(
                rate >= threshold,
                "hybrid {level:?} Hit@{k} {rate} < {threshold}"
            );
        }

        // Fusion group: ≥0.5 Hit@3 and ≥ max(bm25, vec) rates.
        let (qs, fused_lists) = band(Level::Fusion, 2);
        let hybrid_rate = hit_rate(&fused_lists, &qs, 3);
        let (_, bm25_lists) = band(Level::Fusion, 1);
        let bm25_rate = hit_rate(&bm25_lists, &qs, 3);
        let (_, vec_lists) = band(Level::Fusion, 0);
        let vec_rate = hit_rate(&vec_lists, &qs, 3);
        assert!(
            hybrid_rate >= 0.5,
            "fusion Hit@3 {hybrid_rate} < 0.5 (bm25 {bm25_rate}, vec {vec_rate})"
        );
        assert!(
            hybrid_rate >= bm25_rate.max(vec_rate),
            "fusion {hybrid_rate} < max(bm25 {bm25_rate}, vec {vec_rate})"
        );

        // Overall (non-fusion) Hit@3 ≥ 0.6.
        let standard: Vec<Vec<String>> = EVAL_QUERIES
            .iter()
            .enumerate()
            .filter(|(_, e)| e.level != Level::Fusion)
            .map(|(i, _)| results[i].2.clone())
            .collect();
        let standard_qs: Vec<&EvalQuery> = EVAL_QUERIES
            .iter()
            .filter(|e| e.level != Level::Fusion)
            .collect();
        let hybrid_overall = hit_rate(&standard, &standard_qs, 3);
        assert!(
            hybrid_overall >= 0.6,
            "hybrid overall Hit@3 {hybrid_overall} < 0.6"
        );
    }
}
