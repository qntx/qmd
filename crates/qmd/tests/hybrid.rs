#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::shadow_unrelated,
    clippy::field_reassign_with_default,
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::unwrap_in_result,
    clippy::expect_used,
    clippy::missing_assert_message,
    clippy::panic,
    reason = "idiomatic test-code patterns"
)]

//! Hybrid / structured / vector-query orchestration with scripted
//! stub backends — ports the `hybridQuery`/`structuredSearch`/
//! `vectorSearchQuery` behaviour of `store.test.ts` (Caching, Query
//! expansion cache, RRF) and `sdk.test.ts` search groups, plus the
//! T14 structured-vs-hybrid differences, without model downloads.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use qmd::{
    Embedder, Environment, ExpandedQuery, Generator, HybridResult, InferenceError, Qmd, QueryKind,
    Reranker, SearchHooks, SearchInput, SearchOptions, UpdateOptions, VectorQueryOptions,
    VectorQueryResult,
};

fn write(dir: &Path, rel: &str, content: &str) {
    let path = dir.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}

fn env() -> Environment {
    Environment::default()
}

// ── scripted backends ──

type EmbedFn = dyn Fn(&[&str]) -> Vec<Vec<f32>> + Send + Sync;

/// Deterministic text→vector stub (n-gram hash into `dim` buckets —
/// near-identical texts produce near-identical vectors).
struct ScriptedEmbedder {
    model_id: String,
    calls: AtomicUsize,
    texts: Mutex<Vec<String>>,
    f: Box<EmbedFn>,
}

impl ScriptedEmbedder {
    fn hashed(model_id: &str, dim: usize) -> Self {
        Self {
            model_id: model_id.to_owned(),
            calls: AtomicUsize::new(0),
            texts: Mutex::new(Vec::new()),
            f: Box::new(move |texts| {
                texts
                    .iter()
                    .map(|t| hashed_vector(t, dim))
                    .collect::<Vec<_>>()
            }),
        }
    }

    /// Vectors from `f`, but empty vectors (the `null` analogue) for
    /// texts matching `empty_for`.
    fn with_empty_for(model_id: &str, dim: usize, empty_for: &'static str) -> Self {
        Self {
            model_id: model_id.to_owned(),
            calls: AtomicUsize::new(0),
            texts: Mutex::new(Vec::new()),
            f: Box::new(move |texts| {
                texts
                    .iter()
                    .map(|t| hashed_or_empty(t, dim, empty_for))
                    .collect::<Vec<_>>()
            }),
        }
    }
}

fn orthogonal_vector(t: &&str) -> Vec<f32> {
    if t.contains("zzz") {
        vec![0.0, 1.0, 0.0]
    } else {
        vec![1.0, 0.0, 0.0]
    }
}

fn hashed_or_empty(t: &str, dim: usize, empty_for: &str) -> Vec<f32> {
    if t.contains(empty_for) {
        Vec::new()
    } else {
        hashed_vector(t, dim)
    }
}

fn hashed_vector(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    for (i, b) in text.bytes().enumerate() {
        v[i % dim] += f32::from(b) / 255.0;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

impl std::fmt::Debug for ScriptedEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptedEmbedder").finish_non_exhaustive()
    }
}

impl Embedder for ScriptedEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, InferenceError> {
        Ok(text.chars().map(u32::from).collect())
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, InferenceError> {
        Ok("x".repeat(tokens.len()))
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, InferenceError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.texts
            .lock()
            .unwrap()
            .extend(texts.iter().map(|t| (*t).to_owned()));
        Ok((self.f)(texts))
    }
}

type RerankFn = dyn Fn(&str, &str) -> f32 + Send + Sync;

/// Reranker whose per-chunk scores come from a closure; records the
/// queries it saw and how many batches ran.
struct ScriptedReranker {
    model_id: String,
    calls: AtomicUsize,
    queries: Mutex<Vec<String>>,
    f: Box<RerankFn>,
}

impl ScriptedReranker {
    fn new(model_id: &str, f: impl Fn(&str, &str) -> f32 + Send + Sync + 'static) -> Self {
        Self {
            model_id: model_id.to_owned(),
            calls: AtomicUsize::new(0),
            queries: Mutex::new(Vec::new()),
            f: Box::new(f),
        }
    }

    /// Every chunk scores `s` (covers the upstream 0.5 fallback shape).
    fn constant(model_id: &str, s: f32) -> Self {
        Self::new(model_id, move |_, _| s)
    }
}

impl std::fmt::Debug for ScriptedReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptedReranker").finish_non_exhaustive()
    }
}

impl Reranker for ScriptedReranker {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>, InferenceError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.queries.lock().unwrap().push(query.to_owned());
        Ok(docs.iter().map(|d| (self.f)(query, d)).collect())
    }
}

type ExpandFn = dyn Fn(&str) -> Result<String, InferenceError> + Send + Sync;

struct ScriptedGenerator {
    model_id: String,
    calls: AtomicUsize,
    f: Box<ExpandFn>,
}

impl ScriptedGenerator {
    fn new(
        model_id: &str,
        f: impl Fn(&str) -> Result<String, InferenceError> + Send + Sync + 'static,
    ) -> Self {
        Self {
            model_id: model_id.to_owned(),
            calls: AtomicUsize::new(0),
            f: Box::new(f),
        }
    }

    fn output(model_id: &str, output: &str) -> Self {
        let out = output.to_owned();
        Self::new(model_id, move |_| Ok(out.clone()))
    }
}

impl std::fmt::Debug for ScriptedGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptedGenerator").finish_non_exhaustive()
    }
}

impl Generator for ScriptedGenerator {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn expand(&self, query: &str) -> Result<String, InferenceError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        (self.f)(query)
    }
}

// ── hooks recorder ──

#[derive(Default)]
struct RecHooks {
    events: Mutex<Vec<String>>,
}

impl RecHooks {
    fn events(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }
}

impl std::fmt::Debug for RecHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecHooks").finish_non_exhaustive()
    }
}

impl SearchHooks for RecHooks {
    fn on_strong_signal(&self, top_score: f64) {
        self.events
            .lock()
            .unwrap()
            .push(format!("strong:{top_score:.3}"));
    }

    fn on_expand_start(&self) {
        self.events.lock().unwrap().push("expand_start".to_owned());
    }

    fn on_expand(&self, original: &str, expanded: &[ExpandedQuery], _elapsed: std::time::Duration) {
        self.events.lock().unwrap().push(format!(
            "expand:{original}:{}",
            expanded
                .iter()
                .map(|q| q.kind.as_str().to_owned())
                .collect::<Vec<_>>()
                .join(",")
        ));
    }

    fn on_embed_start(&self, count: usize) {
        self.events
            .lock()
            .unwrap()
            .push(format!("embed_start:{count}"));
    }

    fn on_embed_done(&self, _elapsed: std::time::Duration) {
        self.events.lock().unwrap().push("embed_done".to_owned());
    }

    fn on_rerank_start(&self, chunk_count: usize) {
        self.events
            .lock()
            .unwrap()
            .push(format!("rerank_start:{chunk_count}"));
    }

    fn on_rerank_done(&self, _elapsed: std::time::Duration) {
        self.events.lock().unwrap().push("rerank_done".to_owned());
    }
}

// ── store helpers ──

fn add_docs_collection(qmd: &Qmd, dir: &Path, name: &str) {
    qmd.add_collection(
        name,
        &qmd::CollectionSpec {
            path: dir.to_string_lossy().into_owned(),
            ..qmd::CollectionSpec::default()
        },
    )
    .unwrap();
    qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();
}

/// `Qmd` over `tmp/docs` (one `docs` collection) with the given
/// backends injected. `embed_now` runs `Qmd::embed` when an embedder is
/// injected so `vectors_vec` exists.
struct Backends {
    embedder: Option<Arc<dyn Embedder>>,
    reranker: Option<Arc<dyn Reranker>>,
    generator: Option<Arc<dyn Generator>>,
}

fn pipeline(
    tmp: &Path,
    docs: &[(&str, &str)],
    backends: &Backends,
    embed_now: bool,
) -> (Qmd, PathBuf) {
    let docs_dir = tmp.join("docs");
    for (rel, body) in docs {
        write(&docs_dir, rel, body);
    }
    let db = tmp.join("i.sqlite");
    let mut b = Qmd::builder(&db).environment(env());
    if let Some(e) = &backends.embedder {
        b = b.embedder(Arc::clone(e));
    }
    if let Some(r) = &backends.reranker {
        b = b.reranker(Arc::clone(r));
    }
    if let Some(g) = &backends.generator {
        b = b.generator(Arc::clone(g));
    }
    let qmd = b.build().unwrap();
    add_docs_collection(&qmd, &docs_dir, "docs");
    if embed_now && backends.embedder.is_some() {
        qmd.embed(&qmd::EmbedOptions::default(), &mut |_| {})
            .unwrap();
    }
    (qmd, db)
}

fn default_backends() -> (
    Arc<ScriptedEmbedder>,
    Arc<ScriptedReranker>,
    Arc<ScriptedGenerator>,
) {
    (
        Arc::new(ScriptedEmbedder::hashed("m-embed", 8)),
        Arc::new(ScriptedReranker::constant("m-rerank", 0.5)),
        // Echo the query inside each line — upstream's `hasQueryTerm`
        // filter drops expansion lines without an original term.
        Arc::new(ScriptedGenerator::new("m-gen", |q| {
            Ok(format!("lex: {q} expanded\nvec: {q} meaning"))
        })),
    )
}

fn query_opts(input: SearchInput) -> SearchOptions {
    let mut opts = SearchOptions::default();
    opts.input = input;
    opts
}

// ── expand_query ──

#[test]
fn expand_query_generates_parses_and_caches() {
    let tmp = tempfile::tempdir().unwrap();
    let (_e, _r, g) = default_backends();
    let g_calls = Arc::clone(&g);
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha bravo\n")],
        &Backends {
            embedder: None,
            reranker: None,
            generator: Some(g),
        },
        false,
    );

    let out1 = qmd.expand_query("alpha bravo").unwrap();
    assert_eq!(out1.len(), 2);
    assert_eq!(out1[0].kind, QueryKind::Lex);
    assert_eq!(out1[0].query, "alpha bravo expanded");
    assert_eq!(out1[1].kind, QueryKind::Vec);
    assert_eq!(out1[1].query, "alpha bravo meaning");

    // Second call is a cache hit — no generator invocation.
    let out2 = qmd.expand_query("alpha bravo").unwrap();
    assert_eq!(out1, out2);
    assert_eq!(g_calls.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn expand_query_drops_entries_identical_to_original() {
    let tmp = tempfile::tempdir().unwrap();
    let g = Arc::new(ScriptedGenerator::new("m-gen", |q| {
        Ok(format!("lex: {q}\nvec: same approach\nhyde: {q}"))
    }));
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha\n")],
        &Backends {
            embedder: None,
            reranker: None,
            generator: Some(g),
        },
        false,
    );
    let out = qmd.expand_query("same text").unwrap();
    // `lex: same text` and `hyde: same text` are dropped upstream
    // (store.ts:4580-4582 `r.text !== query`); `vec: different` would
    // also die to `hasQueryTerm`, so the variant must reuse a term.
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].kind, QueryKind::Vec);
    assert_eq!(out[0].query, "same approach");
}

#[test]
fn expand_query_parse_fallback_on_garbage_output() {
    let tmp = tempfile::tempdir().unwrap();
    let g = Arc::new(ScriptedGenerator::output(
        "m-gen",
        "total garbage no prefixes",
    ));
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha\n")],
        &Backends {
            embedder: None,
            reranker: None,
            generator: Some(g),
        },
        false,
    );
    let out = qmd.expand_query("my query").unwrap();
    // Parse fallback is hyde/lex/vec (llm.ts:1710-1714), then the
    // store-level `text !== query` filter drops the lex/vec copies of
    // the original — only the hyde variant survives.
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].kind, QueryKind::Hyde);
    assert_eq!(out[0].query, "Information about my query");
}

#[test]
fn expand_query_error_fallback_on_backend_failure() {
    let tmp = tempfile::tempdir().unwrap();
    let g = Arc::new(ScriptedGenerator::new("m-gen", |_| {
        Err(InferenceError::Backend("boom".to_owned()))
    }));
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha\n")],
        &Backends {
            embedder: None,
            reranker: None,
            generator: Some(g),
        },
        false,
    );
    let out = qmd.expand_query("q").unwrap();
    // Error fallback is lex/vec copies of the query (llm.ts:1716-1721)
    // — both are `=== query` and get filtered out upstream.
    assert!(out.is_empty());
}

#[test]
fn expand_query_empty_result_writes_no_cache() {
    let tmp = tempfile::tempdir().unwrap();
    // Output that parses to exactly the original query — filtered to
    // empty, so upstream skips `setCachedResult`.
    let g = Arc::new(ScriptedGenerator::new("m-gen", |q| Ok(format!("lex: {q}"))));
    let (qmd, db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha\n")],
        &Backends {
            embedder: None,
            reranker: None,
            generator: Some(g),
        },
        false,
    );
    let out = qmd.expand_query("solo").unwrap();
    assert!(out.is_empty());
    let n: i64 = rusqlite::Connection::open(&db)
        .unwrap()
        .query_row("SELECT COUNT(*) FROM llm_cache", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 0);
}

// ── hybrid pipeline ──

#[test]
fn hybrid_search_blends_rrf_and_rerank_scores() {
    let tmp = tempfile::tempdir().unwrap();
    let (e, _r, g) = default_backends();
    // Score chunks by whether they contain the literal query word —
    // doc "alpha" matches, "unrelated" does not.
    let r = Arc::new(ScriptedReranker::new("m-rerank", |q, doc| {
        if doc.to_lowercase().contains(&q.to_lowercase()) {
            0.9
        } else {
            0.1
        }
    }));
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[
            ("a.md", "# Alpha\nalpha bravo charlie delta\n"),
            ("b.md", "# Beta\nunrelated filler words here\n"),
        ],
        &Backends {
            embedder: Some(e),
            reranker: Some(r),
            generator: Some(g),
        },
        true,
    );

    let mut opts = query_opts(SearchInput::Query("alpha".to_owned()));
    opts.explain = true;
    let results = qmd.search(&opts, &()).unwrap();
    assert!(!results.is_empty());
    let top = &results[0];
    // a.md contains "alpha" → rerank score 0.9; b.md does not → 0.1.
    assert!(top.filepath.ends_with("a.md"), "top: {}", top.filepath);
    // Single candidate → rrfRank 1 → blended = 0.75/1 + 0.25*score.
    let explain = top.explain.as_ref().unwrap();
    assert_eq!(explain.rrf.rank, 1);
    assert!((explain.rrf.weight - 0.75).abs() < 1e-9);
    assert!((explain.rerank_score - 0.9).abs() < 1e-5);
    let expected = 0.25f64.mul_add(explain.rerank_score, 0.75);
    assert!((top.score - expected).abs() < 1e-9);
    assert_eq!(top.score, explain.blended_score);
}

#[test]
fn hybrid_strong_signal_skips_expansion() {
    let tmp = tempfile::tempdir().unwrap();
    let (_e, _r, g) = default_backends();
    let g_calls = Arc::clone(&g);
    // One doc saturated with the query terms; the rest unrelated —
    // normalized bm25 clears the 0.85 / 0.15-gap thresholds.
    let strong = "zebra quoll wombat ".repeat(30);
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[
            ("a.md", &format!("# A\n{strong}")),
            ("b.md", "# B\nquiet library catalog index reference\n"),
            ("c.md", "# C\ncooking recipe garlic onion simmer broth\n"),
            ("d.md", "# D\nrailway timetable departure platform signal\n"),
            (
                "e.md",
                "# E\nwatercolor pigment brush canvas easel sketch\n",
            ),
            ("f.md", "# F\nalgebra matrix determinant eigenvalue proof\n"),
            ("g.md", "# G\nvineyard harvest barrel fermentation grape\n"),
        ],
        &Backends {
            embedder: None,
            reranker: Some(Arc::new(ScriptedReranker::constant("m-rerank", 0.5))),
            generator: Some(g),
        },
        false,
    );

    let hooks = RecHooks::default();
    let results = qmd
        .search(
            &query_opts(SearchInput::Query("zebra quoll wombat".to_owned())),
            &hooks,
        )
        .unwrap();
    assert!(!results.is_empty());
    let events = hooks.events();
    assert!(
        events.iter().any(|e| e.starts_with("strong:")),
        "expected onStrongSignal, got {events:?}"
    );
    // Expansion skipped entirely — generator never invoked.
    assert_eq!(g_calls.calls.load(Ordering::SeqCst), 0);
    assert!(events.iter().any(|e| e == "expand:zebra quoll wombat:"));
}

#[test]
fn hybrid_intent_disables_strong_signal_bypass() {
    let tmp = tempfile::tempdir().unwrap();
    let (_e, _r, g) = default_backends();
    let g_calls = Arc::clone(&g);
    let strong = "zebra quoll wombat ".repeat(30);
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[
            ("a.md", &format!("# A\n{strong}")),
            ("b.md", "# B\nquiet library catalog index reference\n"),
            ("c.md", "# C\ncooking recipe garlic onion simmer broth\n"),
            ("d.md", "# D\nrailway timetable departure platform signal\n"),
            (
                "e.md",
                "# E\nwatercolor pigment brush canvas easel sketch\n",
            ),
            ("f.md", "# F\nalgebra matrix determinant eigenvalue proof\n"),
            ("g.md", "# G\nvineyard harvest barrel fermentation grape\n"),
        ],
        &Backends {
            embedder: None,
            reranker: Some(Arc::new(ScriptedReranker::constant("m-rerank", 0.5))),
            generator: Some(g),
        },
        false,
    );

    let mut opts = query_opts(SearchInput::Query("zebra quoll wombat".to_owned()));
    opts.intent = Some("unrelated meta commentary".to_owned());
    let hooks = RecHooks::default();
    qmd.search(&opts, &hooks).unwrap();
    let events = hooks.events();
    assert!(!events.iter().any(|e| e.starts_with("strong:")));
    assert_eq!(g_calls.calls.load(Ordering::SeqCst), 1);
    // Expansion ran — the expanded list carries the scripted kinds.
    assert!(
        events
            .iter()
            .any(|e| e == "expand:zebra quoll wombat:lex,vec")
    );
}

#[test]
fn hybrid_expansion_cache_warms_second_search() {
    let tmp = tempfile::tempdir().unwrap();
    // `lex: alpha` contributes a hit, so the expansion is kept — only a
    // zero-contribution expansion is evicted.
    let g = Arc::new(ScriptedGenerator::output("m-gen", "lex: alpha"));
    let g_calls = Arc::clone(&g);
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha bravo charlie\n")],
        &Backends {
            embedder: None,
            reranker: Some(Arc::new(ScriptedReranker::constant("m-rerank", 0.5))),
            generator: Some(g),
        },
        false,
    );
    let mut opts = query_opts(SearchInput::Query("alpha bravo".to_owned()));
    opts.rerank = false;
    qmd.search(&opts, &()).unwrap();
    qmd.search(&opts, &()).unwrap();
    assert_eq!(g_calls.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn hybrid_dud_expansion_is_evicted_and_regenerated() {
    let tmp = tempfile::tempdir().unwrap();
    // Expansion parses (`alphaqqq` embeds the term "alpha") but its
    // FTS search matches nothing — a runnable dud (upstream #818
    // eviction); vec/hyde entries would not count on an FTS-only store.
    let g = Arc::new(ScriptedGenerator::output("m-gen", "lex: alphaqqq"));
    let g_calls = Arc::clone(&g);
    let (qmd, db) = pipeline(
        tmp.path(),
        &[("a.md", "# Alpha\nalpha bravo charlie\n")],
        &Backends {
            embedder: None,
            reranker: Some(Arc::new(ScriptedReranker::constant("m-rerank", 0.5))),
            generator: Some(g),
        },
        false,
    );
    let mut opts = query_opts(SearchInput::Query("alpha bravo".to_owned()));
    // Intent disables strong-signal so the expansion actually runs.
    opts.intent = Some("unrelated meta".to_owned());
    opts.rerank = false;
    qmd.search(&opts, &()).unwrap();

    // The dud entry must not survive to poison warm repeats.
    let cache_rows: i64 = rusqlite::Connection::open(&db)
        .unwrap()
        .query_row("SELECT COUNT(*) FROM llm_cache", [], |r| r.get(0))
        .unwrap();
    assert_eq!(cache_rows, 0);
    qmd.search(&opts, &()).unwrap();
    assert_eq!(g_calls.calls.load(Ordering::SeqCst), 2);
}

#[test]
fn hybrid_contributing_expansion_stays_cached() {
    let tmp = tempfile::tempdir().unwrap();
    let g = Arc::new(ScriptedGenerator::output("m-gen", "lex: charlie"));
    let g_calls = Arc::clone(&g);
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# Alpha\nalpha bravo charlie\n")],
        &Backends {
            embedder: None,
            reranker: Some(Arc::new(ScriptedReranker::constant("m-rerank", 0.5))),
            generator: Some(g),
        },
        false,
    );
    let mut opts = query_opts(SearchInput::Query("alpha bravo".to_owned()));
    opts.intent = Some("unrelated meta".to_owned());
    opts.rerank = false;
    qmd.search(&opts, &()).unwrap();
    qmd.search(&opts, &()).unwrap();
    // Second search reused the contributing expansion.
    assert_eq!(g_calls.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn skip_rerank_scores_are_positional_1_over_rank() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[
            ("a.md", "# A\nalpha bravo charlie\n"),
            ("b.md", "# B\nalpha bravo echo\n"),
            ("c.md", "# C\nalpha foxtrot golf\n"),
        ],
        &Backends {
            embedder: None,
            reranker: Some(Arc::new(ScriptedReranker::constant("m-rerank", 0.5))),
            generator: Some(Arc::new(ScriptedGenerator::output("m-gen", ""))),
        },
        false,
    );
    let mut opts = query_opts(SearchInput::Query("alpha".to_owned()));
    opts.rerank = false;
    opts.explain = true;
    let results = qmd.search(&opts, &()).unwrap();
    assert!(results.len() >= 2);
    for (i, r) in results.iter().enumerate() {
        let expected = 1.0 / (i + 1) as f64;
        assert!(
            (r.score - expected).abs() < 1e-9,
            "rank {}: {} vs {expected}",
            i + 1,
            r.score
        );
        let explain = r.explain.as_ref().unwrap();
        assert_eq!(explain.rrf.rank, i + 1);
        assert_eq!(explain.rrf.weight, 1.0);
        assert_eq!(explain.rerank_score, 0.0);
        assert_eq!(explain.blended_score, r.score);
    }
}

#[test]
fn constant_half_rerank_scores_blend_at_rank_one() {
    let tmp = tempfile::tempdir().unwrap();
    // A reranker producing uniform 0.5s is exactly the shape the real
    // backend emits when the rerank context is unavailable (T1/T8);
    // that lazy fallback itself is covered by the llamacpp unit test
    // `lazy_backends_report_model_load_on_missing_gguf`.
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha bravo charlie\n")],
        &Backends {
            embedder: None,
            reranker: Some(Arc::new(ScriptedReranker::constant("m-rerank", 0.5))),
            generator: Some(Arc::new(ScriptedGenerator::output("m-gen", ""))),
        },
        false,
    );
    let mut opts = query_opts(SearchInput::Query("alpha".to_owned()));
    opts.explain = true;
    let results = qmd.search(&opts, &()).unwrap();
    assert_eq!(results.len(), 1);
    let r = &results[0];
    let explain = r.explain.as_ref().unwrap();
    assert_eq!(explain.rerank_score, 0.5);
    // rank 1 → 0.75*1 + 0.25*0.5
    assert!((r.score - (0.75 + 0.125)).abs() < 1e-9);
}

#[test]
fn rerank_chunk_cache_skips_repeat_calls() {
    let tmp = tempfile::tempdir().unwrap();
    let r = Arc::new(ScriptedReranker::constant("m-rerank", 0.7));
    let r_calls = Arc::clone(&r);
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha bravo charlie\n")],
        &Backends {
            embedder: None,
            reranker: Some(r),
            generator: Some(Arc::new(ScriptedGenerator::output("m-gen", ""))),
        },
        false,
    );
    let opts = query_opts(SearchInput::Query("alpha".to_owned()));
    qmd.search(&opts, &()).unwrap();
    assert_eq!(r_calls.calls.load(Ordering::SeqCst), 1);
    qmd.search(&opts, &()).unwrap();
    // Chunk score cached — second search reranks nothing.
    assert_eq!(r_calls.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn min_score_and_limit_filter_results() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[
            ("a.md", "# A\nalpha bravo charlie\n"),
            ("b.md", "# B\nalpha bravo echo\n"),
            ("c.md", "# C\nalpha foxtrot golf\n"),
        ],
        &Backends {
            embedder: None,
            reranker: Some(Arc::new(ScriptedReranker::constant("m-rerank", 0.0))),
            generator: Some(Arc::new(ScriptedGenerator::output("m-gen", ""))),
        },
        false,
    );

    // Baseline: all three alpha docs fuse to ranks 1..3.
    let mut opts = query_opts(SearchInput::Query("alpha".to_owned()));
    opts.rerank = false;
    let all = qmd.search(&opts, &()).unwrap();
    assert_eq!(all.len(), 3, "expected 3 candidates without a limit");

    // Skip-rerank scores are 1/rrf_rank → 1, 1/2, 1/3.
    opts.limit = 2;
    let results = qmd.search(&opts, &()).unwrap();
    assert_eq!(results.len(), 2);

    // min_score 0.8 keeps only rank 1.
    opts.limit = 10;
    opts.min_score = 0.8;
    let results = qmd.search(&opts, &()).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].score >= 0.8);
}

#[test]
fn results_dedupe_by_file_across_lists() {
    let tmp = tempfile::tempdir().unwrap();
    // Expansion produces a lex query hitting the same docs as the
    // original FTS list — RRF merges, output dedupes by file.
    let g = Arc::new(ScriptedGenerator::output("m-gen", "lex: alpha bravo"));
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[
            ("a.md", "# A\nalpha bravo charlie\n"),
            ("b.md", "# B\nalpha bravo echo\n"),
        ],
        &Backends {
            embedder: None,
            reranker: Some(Arc::new(ScriptedReranker::constant("m-rerank", 0.5))),
            generator: Some(g),
        },
        false,
    );
    let mut opts = query_opts(SearchInput::Query("alpha".to_owned()));
    opts.rerank = false;
    opts.explain = true;
    // Intent disables the strong-signal bypass so expansion runs.
    opts.intent = Some("context".to_owned());
    let results = qmd.search(&opts, &()).unwrap();
    assert_eq!(results.len(), 2, "{results:?}");
    let files: Vec<&str> = results.iter().map(|r| r.filepath.as_str()).collect();
    let unique: std::collections::HashSet<&str> = files.iter().copied().collect();
    assert_eq!(unique.len(), files.len());
    // Each doc was hit by two lists — original FTS plus the lex
    // expansion.
    for r in &results {
        let lists: std::collections::BTreeSet<usize> = r
            .explain
            .as_ref()
            .unwrap()
            .rrf
            .contributions
            .iter()
            .map(|c| c.list_index)
            .collect();
        assert_eq!(lists.len(), 2, "contributions for {}", r.filepath);
    }
}

#[test]
fn explain_reports_fts_and_vector_contributions() {
    let tmp = tempfile::tempdir().unwrap();
    let (e, _r, g) = default_backends();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# Alpha\nalpha bravo charlie delta\n")],
        &Backends {
            embedder: Some(e),
            reranker: Some(Arc::new(ScriptedReranker::constant("m-rerank", 0.5))),
            generator: Some(g),
        },
        true,
    );
    let mut opts = query_opts(SearchInput::Query("alpha".to_owned()));
    opts.explain = true;
    let results = qmd.search(&opts, &()).unwrap();
    assert!(!results.is_empty());
    let ex = results[0].explain.as_ref().unwrap();
    assert!(!ex.fts_scores.is_empty());
    assert!(!ex.vector_scores.is_empty());
    // Contributions carry list provenance.
    assert!(
        ex.rrf
            .contributions
            .iter()
            .any(|c| { c.source == qmd::SearchSource::Fts && c.query_type == QueryKind::Original })
    );
    assert!(ex.rrf.contributions.iter().all(|c| c.rank >= 1));
    assert!((ex.rrf.total_score - (ex.rrf.base_score + ex.rrf.top_rank_bonus)).abs() < 1e-9);
}

#[test]
fn hooks_fire_in_pipeline_order() {
    let tmp = tempfile::tempdir().unwrap();
    let (e, _r, g) = default_backends();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# Alpha\nalpha bravo charlie delta\n")],
        &Backends {
            embedder: Some(e),
            reranker: Some(Arc::new(ScriptedReranker::constant("m-rerank", 0.5))),
            generator: Some(g),
        },
        true,
    );
    let hooks = RecHooks::default();
    qmd.search(&query_opts(SearchInput::Query("alpha".to_owned())), &hooks)
        .unwrap();
    let events = hooks.events();
    let pos = |tag: &str| events.iter().position(|e| e.starts_with(tag)).unwrap();
    assert!(pos("expand_start") < pos("expand:"), "{events:?}");
    assert!(pos("expand:") < pos("embed_start"), "{events:?}");
    assert!(pos("embed_start") < pos("embed_done"), "{events:?}");
    assert!(pos("embed_done") < pos("rerank_start"), "{events:?}");
    assert!(pos("rerank_start") < pos("rerank_done"), "{events:?}");
}

// ── vector_query ──

#[test]
fn vector_query_returns_empty_without_vector_table() {
    let tmp = tempfile::tempdir().unwrap();
    let (_e, _r, g) = default_backends();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha bravo\n")],
        &Backends {
            embedder: None,
            reranker: None,
            generator: Some(g),
        },
        false,
    );
    let results = qmd
        .vector_query("alpha", &VectorQueryOptions::default(), &())
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn vector_query_expands_dedupes_and_filters_lex() {
    let tmp = tempfile::tempdir().unwrap();
    let e = Arc::new(ScriptedEmbedder::hashed("m-embed", 8));
    let e_calls = Arc::clone(&e);
    // The lex line shares the query term "alpha" so it survives the
    // `hasQueryTerm` parse filter — only vector_query's own lex-drop
    // keeps it away from the embedder.
    let g = Arc::new(ScriptedGenerator::output(
        "m-gen",
        "lex: alpha lexical\nvec: alpha semantic variant",
    ));
    let g_calls = Arc::clone(&g);
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[
            ("a.md", "# Alpha\nalpha bravo charlie delta\n"),
            ("b.md", "# Beta\nunrelated filler words here\n"),
        ],
        &Backends {
            embedder: Some(e),
            reranker: None,
            generator: Some(g),
        },
        true,
    );

    let hooks = RecHooks::default();
    let results = qmd
        .vector_query("alpha", &VectorQueryOptions::default(), &hooks)
        .unwrap();
    // Hook received only the vec/hyde expansions.
    let ev = hooks.events();
    assert!(ev.iter().any(|x| x == "expand:alpha:vec"), "{ev:?}");
    assert_eq!(g_calls.calls.load(Ordering::SeqCst), 1);
    // The lex expansion never reached the embedding model — no embedded
    // text carries its distinctive word.
    let embedded = e_calls.texts.lock().unwrap();
    assert!(
        !embedded.iter().any(|t| t.contains("lexical")),
        "lex expansion leaked into embeddings: {embedded:?}"
    );
    drop(embedded);
    // Deduped by file, sorted by score, min_score default 0.3 applied.
    let files: std::collections::HashSet<&str> = results
        .iter()
        .map(|r: &VectorQueryResult| r.filepath.as_str())
        .collect();
    assert_eq!(files.len(), results.len());
    assert!(results.windows(2).all(|w| w[0].score >= w[1].score));
    assert!(results.iter().all(|r| r.score >= 0.3));
}

#[test]
fn vector_query_min_score_default_is_point_three() {
    assert_eq!(VectorQueryOptions::default().min_score, 0.3);
    let tmp = tempfile::tempdir().unwrap();
    // Orthogonal scripted vectors: "zzz" texts embed to e1, everything
    // else to e0 — so b.md lands at cosine 0.0 against the query.
    let e = Arc::new(ScriptedEmbedder {
        model_id: "m-embed".to_owned(),
        calls: AtomicUsize::new(0),
        texts: Mutex::new(Vec::new()),
        f: Box::new(|texts| texts.iter().map(orthogonal_vector).collect()),
    });
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[
            ("a.md", "# Alpha\nalpha bravo charlie delta\n"),
            ("b.md", "# Beta\ncompletely unrelated zzz qqq www\n"),
        ],
        &Backends {
            embedder: Some(e),
            reranker: None,
            generator: Some(Arc::new(ScriptedGenerator::output("m-gen", ""))),
        },
        true,
    );
    let results = qmd
        .vector_query("alpha", &VectorQueryOptions::default(), &())
        .unwrap();
    assert!(results.iter().all(|r| r.score >= 0.3));
    // Explicitly lower the bar — a sub-0.3 result must actually exist,
    // otherwise the default filter proves nothing.
    let mut opts = VectorQueryOptions::default();
    opts.min_score = 0.0;
    let all = qmd.vector_query("alpha", &opts, &()).unwrap();
    assert!(all.len() > results.len(), "{all:?} vs {results:?}");
    assert!(all.iter().any(|r| r.score < 0.3), "{all:?}");
}

// ── structured search (T14 diffs) ──

fn structured(input: Vec<ExpandedQuery>) -> SearchOptions {
    let mut opts = query_opts(SearchInput::Expanded(input));
    opts.rerank = false;
    opts
}

#[test]
fn structured_empty_searches_returns_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha\n")],
        &Backends {
            embedder: None,
            reranker: None,
            generator: None,
        },
        false,
    );
    let results = qmd.search(&structured(vec![]), &()).unwrap();
    assert!(results.is_empty());
}

#[test]
fn structured_validation_reports_line_and_kind() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha\n")],
        &Backends {
            embedder: None,
            reranker: None,
            generator: None,
        },
        false,
    );

    // Newline inside a query — "Line 3 (lex): queries must be single-line."
    let err = qmd
        .search(
            &structured(vec![
                ExpandedQuery::new(QueryKind::Lex, "foo\nbar").with_line(3),
            ]),
            &(),
        )
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Line 3 (lex):"), "{msg}");
    assert!(msg.contains("single-line"), "{msg}");

    // Unmatched quote — validate_lex_query surfaces through the same
    // location prefix.
    let err = qmd
        .search(
            &structured(vec![
                ExpandedQuery::new(QueryKind::Lex, "\"unfinished").with_line(2),
            ]),
            &(),
        )
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Line 2 (lex):"), "{msg}");
    assert!(msg.to_lowercase().contains("unmatched"), "{msg}");

    // Semantic validation rejects negation.
    let err = qmd
        .search(
            &structured(vec![ExpandedQuery::new(QueryKind::Vec, "-redis pooling")]),
            &(),
        )
        .unwrap_err();
    assert!(err.to_string().contains("Negation"), "{err}");
}

#[test]
fn structured_first_list_gets_double_weight_not_original() {
    // T14 diff ①: structured weights are [2,1,...] by position while
    // hybrid weights follow is-original. Two lex queries — `alpha`
    // hits both docs, `bravo` hits only a.md, so a.md tops the fusion
    // and MUST carry contributions from both lists.
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[
            ("a.md", "# A\nalpha bravo charlie\n"),
            ("b.md", "# B\nalpha echo foxtrot\n"),
        ],
        &Backends {
            embedder: None,
            reranker: None,
            generator: None,
        },
        false,
    );
    let mut opts = structured(vec![
        ExpandedQuery::new(QueryKind::Lex, "alpha"),
        ExpandedQuery::new(QueryKind::Lex, "bravo"),
    ]);
    opts.explain = true;
    let results = qmd.search(&opts, &()).unwrap();
    assert!(!results.is_empty());
    let top = &results[0];
    assert!(top.filepath.ends_with("a.md"), "top: {}", top.filepath);
    let ex = top.explain.as_ref().unwrap();
    // First list contributes with weight 2.0, second with 1.0 —
    // both mandatory: a.md appears in list 0 (rank 1) and list 1.
    let weight_of = |list: usize| {
        ex.rrf
            .contributions
            .iter()
            .find(|c| c.list_index == list)
            .map(|c| c.weight)
    };
    assert_eq!(weight_of(0), Some(2.0));
    assert_eq!(weight_of(1), Some(1.0));
}

#[test]
fn structured_uses_first_lex_for_rerank_not_original_query() {
    // T14 diff ④: the rerank query is the first lex (else first vec),
    // never an "original" — there is none in structured input.
    let tmp = tempfile::tempdir().unwrap();
    let r = Arc::new(ScriptedReranker::constant("m-rerank", 0.5));
    let r_queries = Arc::clone(&r);
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha bravo charlie\n")],
        &Backends {
            embedder: None,
            reranker: Some(r),
            generator: None,
        },
        false,
    );
    // lex queries must match so rerank runs: use real terms.
    let opts = query_opts(SearchInput::Expanded(vec![
        ExpandedQuery::new(QueryKind::Lex, "alpha"),
        ExpandedQuery::new(QueryKind::Lex, "bravo"),
    ]));
    qmd.search(&opts, &()).unwrap();
    let queries = r_queries.queries.lock().unwrap();
    assert_eq!(queries.len(), 1);
    assert_eq!(queries[0], "alpha");
}

#[test]
fn structured_fires_empty_expand_hook() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha bravo\n")],
        &Backends {
            embedder: None,
            reranker: Some(Arc::new(ScriptedReranker::constant("m-rerank", 0.5))),
            generator: None,
        },
        false,
    );
    let mut opts = query_opts(SearchInput::Expanded(vec![ExpandedQuery::new(
        QueryKind::Lex,
        "alpha",
    )]));
    opts.rerank = false;
    let hooks = RecHooks::default();
    qmd.search(&opts, &hooks).unwrap();
    // Upstream `hooks?.onExpand?.("", [], 0)` — signals no expansion.
    let ev = hooks.events();
    assert!(ev.iter().any(|e| e == "expand::"), "{ev:?}");
}

#[test]
fn structured_per_collection_lists_vs_hybrid_merged() {
    // T14 diff ②: structured creates one list per (query × collection);
    // hybrid merges collections into one list per query.
    let tmp = tempfile::tempdir().unwrap();
    let dir_a = tmp.path().join("ca");
    let dir_b = tmp.path().join("cb");
    write(&dir_a, "a.md", "# A\nalpha bravo\n");
    write(&dir_b, "b.md", "# B\nalpha charlie\n");
    let db = tmp.path().join("i.sqlite");
    let qmd = Qmd::builder(&db)
        .environment(env())
        .reranker(Arc::new(ScriptedReranker::constant("m-rerank", 0.5)))
        // The hybrid arm expands — keep the scripted generator so no
        // lazy llama.cpp backend (and no model download) is resolved.
        .generator(Arc::new(ScriptedGenerator::output("m-gen", "")))
        .build()
        .unwrap();
    add_docs_collection(&qmd, &dir_a, "ca");
    add_docs_collection(&qmd, &dir_b, "cb");

    // Hybrid: one FTS list covering both collections — a file in each
    // collection can share one contribution list.
    let mut opts = query_opts(SearchInput::Query("alpha".to_owned()));
    opts.rerank = false;
    opts.explain = true;
    opts.collections = Some(vec!["ca".to_owned(), "cb".to_owned()]);
    let hybrid_results = qmd.search(&opts, &()).unwrap();
    let hybrid_lists: std::collections::BTreeSet<usize> = hybrid_results
        .iter()
        .flat_map(|r| {
            r.explain
                .as_ref()
                .unwrap()
                .rrf
                .contributions
                .iter()
                .map(|c| c.list_index)
        })
        .collect();
    assert_eq!(
        hybrid_lists.len(),
        1,
        "hybrid merges into one list: {hybrid_lists:?}"
    );

    // Structured: same query × two collections → two lists.
    let mut opts = structured(vec![ExpandedQuery::new(QueryKind::Lex, "alpha")]);
    opts.explain = true;
    opts.collections = Some(vec!["ca".to_owned(), "cb".to_owned()]);
    let structured_results = qmd.search(&opts, &()).unwrap();
    let structured_lists: std::collections::BTreeSet<usize> = structured_results
        .iter()
        .flat_map(|r| {
            r.explain
                .as_ref()
                .unwrap()
                .rrf
                .contributions
                .iter()
                .map(|c| c.list_index)
        })
        .collect();
    assert_eq!(
        structured_lists.len(),
        2,
        "structured keeps per-collection lists: {structured_lists:?}"
    );
}

#[test]
fn hybrid_result_shape_fields() {
    let tmp = tempfile::tempdir().unwrap();
    let body = "# Title\nalpha bravo charlie delta echo foxtrot\n";
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", body)],
        &Backends {
            embedder: None,
            reranker: Some(Arc::new(ScriptedReranker::constant("m-rerank", 0.5))),
            generator: Some(Arc::new(ScriptedGenerator::output("m-gen", ""))),
        },
        false,
    );
    let mut opts = query_opts(SearchInput::Query("alpha".to_owned()));
    opts.rerank = false;
    let results: Vec<HybridResult> = qmd.search(&opts, &()).unwrap();
    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert!(r.filepath.starts_with("qmd://docs/"));
    assert_eq!(r.body, body);
    assert!(!r.best_chunk.is_empty());
    // chunkPos is a UTF-16 offset (upstream `pos` unit).
    assert!(r.best_chunk_pos <= qmd::utf16_len(&r.body));
    assert!(!r.docid.is_empty());
}

// ── JS-falsy normalization (A1/A2/A6) ──

/// The BM25-saturating corpus shared by the strong-signal tests.
fn strong_docs() -> Vec<(String, String)> {
    let strong = "zebra quoll wombat ".repeat(30);
    [
        ("a.md", format!("# A\n{strong}")),
        (
            "b.md",
            "# B\nquiet library catalog index reference\n".to_owned(),
        ),
        (
            "c.md",
            "# C\ncooking recipe garlic onion simmer broth\n".to_owned(),
        ),
        (
            "d.md",
            "# D\nrailway timetable departure platform signal\n".to_owned(),
        ),
        (
            "e.md",
            "# E\nwatercolor pigment brush canvas easel sketch\n".to_owned(),
        ),
        (
            "f.md",
            "# F\nalgebra matrix determinant eigenvalue proof\n".to_owned(),
        ),
        (
            "g.md",
            "# G\nvineyard harvest barrel fermentation grape\n".to_owned(),
        ),
    ]
    .into_iter()
    .map(|(r, b)| (r.to_owned(), b))
    .collect()
}

#[test]
fn empty_intent_behaves_like_no_intent() {
    // A1 — upstream treats "" as no intent (store.ts:5570 `!intent`,
    // 4607 `intent ? ... : query`): the strong-signal bypass stays
    // armed and the reranker sees the plain query, never "\n\n...".
    let tmp = tempfile::tempdir().unwrap();
    let (_e, _r, g) = default_backends();
    let g_calls = Arc::clone(&g);
    let r = Arc::new(ScriptedReranker::constant("m-rerank", 0.5));
    let r_queries = Arc::clone(&r);
    let owned = strong_docs();
    let docs: Vec<(&str, &str)> = owned
        .iter()
        .map(|(r, b)| (r.as_str(), b.as_str()))
        .collect();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &docs,
        &Backends {
            embedder: None,
            reranker: Some(r),
            generator: Some(g),
        },
        false,
    );

    let mut opts = query_opts(SearchInput::Query("zebra quoll wombat".to_owned()));
    opts.intent = Some(String::new());
    let hooks = RecHooks::default();
    let results = qmd.search(&opts, &hooks).unwrap();
    assert!(!results.is_empty());
    let events = hooks.events();
    assert!(
        events.iter().any(|e| e.starts_with("strong:")),
        "empty intent must not disarm strong signal: {events:?}"
    );
    assert_eq!(g_calls.calls.load(Ordering::SeqCst), 0);
    let queries = r_queries.queries.lock().unwrap();
    assert_eq!(queries.len(), 1);
    assert_eq!(queries[0], "zebra quoll wombat");
}

#[test]
fn structured_empty_collections_means_all() {
    // A2 — `Some(vec![])` normalizes to `None` (all collections), like
    // upstream `[]` → `undefined` (index.ts:436-449).
    let tmp = tempfile::tempdir().unwrap();
    let dir_a = tmp.path().join("ca");
    let dir_b = tmp.path().join("cb");
    write(&dir_a, "a.md", "# A\nalpha bravo\n");
    write(&dir_b, "b.md", "# B\nalpha charlie\n");
    let db = tmp.path().join("i.sqlite");
    let qmd = Qmd::builder(&db)
        .environment(env())
        .reranker(Arc::new(ScriptedReranker::constant("m-rerank", 0.5)))
        .build()
        .unwrap();
    add_docs_collection(&qmd, &dir_a, "ca");
    add_docs_collection(&qmd, &dir_b, "cb");

    let all = qmd
        .search(
            &structured(vec![ExpandedQuery::new(QueryKind::Lex, "alpha")]),
            &(),
        )
        .unwrap();
    assert_eq!(all.len(), 2);

    let mut opts = structured(vec![ExpandedQuery::new(QueryKind::Lex, "alpha")]);
    opts.collections = Some(vec![]);
    let empty = qmd.search(&opts, &()).unwrap();
    let files_of =
        |rs: &[HybridResult]| -> Vec<String> { rs.iter().map(|r| r.filepath.clone()).collect() };
    assert_eq!(files_of(&empty), files_of(&all));
}

#[test]
fn search_empty_query_string_is_invalid() {
    // A6 — "" is falsy upstream: `search({query: ""})` throws the
    // unified-API error (index.ts:432-434).
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha\n")],
        &Backends {
            embedder: None,
            reranker: None,
            generator: None,
        },
        false,
    );
    let err = qmd
        .search(&query_opts(SearchInput::Query(String::new())), &())
        .unwrap_err();
    assert!(
        matches!(err, qmd::Error::InvalidQuery { .. }),
        "expected InvalidQuery, got {err:?}"
    );
    assert!(
        err.to_string()
            .contains("requires either 'query' or 'queries'"),
        "{err}"
    );
    // `Expanded([])` is not an error — just no results.
    assert!(qmd.search(&structured(vec![]), &()).unwrap().is_empty());
}

// ── model-load failures propagate (A3/A4 integration) ──

#[test]
fn generator_model_load_error_propagates_out_of_search() {
    // Upstream `ensureGenerateModel` throws outside the expansion
    // try/catch (llm.ts:1635-1636) — a model-load failure is never
    // degraded to the heuristic expansion fallback.
    let tmp = tempfile::tempdir().unwrap();
    let g = Arc::new(ScriptedGenerator::new("m-gen", |_| {
        Err(InferenceError::ModelLoad("no such gguf".to_owned()))
    }));
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha bravo\n")],
        &Backends {
            embedder: None,
            reranker: None,
            generator: Some(g),
        },
        false,
    );
    // Intent disables the strong-signal bypass so expansion is reached.
    let mut opts = query_opts(SearchInput::Query("alpha".to_owned()));
    opts.intent = Some("ctx".to_owned());
    opts.rerank = false;
    let err = qmd.search(&opts, &()).unwrap_err();
    assert!(
        matches!(err, qmd::Error::Inference(InferenceError::ModelLoad(_))),
        "expected Inference(ModelLoad), got {err:?}"
    );
}

#[test]
fn generator_backend_error_still_falls_back() {
    // Ordinary backend failure → heuristic expansion (lex/vec copies of
    // the query, filtered to empty) → FTS-only results.
    let tmp = tempfile::tempdir().unwrap();
    let g = Arc::new(ScriptedGenerator::new("m-gen", |_| {
        Err(InferenceError::Backend("boom".to_owned()))
    }));
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha bravo\n")],
        &Backends {
            embedder: None,
            reranker: None,
            generator: Some(g),
        },
        false,
    );
    let mut opts = query_opts(SearchInput::Query("alpha".to_owned()));
    opts.intent = Some("ctx".to_owned());
    opts.rerank = false;
    let results = qmd.search(&opts, &()).unwrap();
    assert!(!results.is_empty());
}

// ── empty embeddings skip the vec leg (B2) ──

#[test]
fn empty_query_embedding_yields_fts_only_results() {
    // Upstream `if (!embedding) continue` (store.ts:5638-5639): the
    // doc corpus is embedded for real, but query texts embed to an
    // empty vector — the search still succeeds on FTS alone.
    let tmp = tempfile::tempdir().unwrap();
    // `format_query_for_embedding` prefixes query inputs with
    // "task: search result" for non-qwen models — doc inputs use
    // "title: ... | text: ..." instead.
    let e = Arc::new(ScriptedEmbedder::with_empty_for(
        "m-embed",
        8,
        "task: search result",
    ));
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha bravo charlie\n")],
        &Backends {
            embedder: Some(e),
            reranker: Some(Arc::new(ScriptedReranker::constant("m-rerank", 0.5))),
            generator: Some(Arc::new(ScriptedGenerator::output("m-gen", ""))),
        },
        true,
    );
    let mut opts = query_opts(SearchInput::Query("alpha".to_owned()));
    opts.explain = true;
    let results = qmd.search(&opts, &()).unwrap();
    assert_eq!(results.len(), 1);
    let ex = results[0].explain.as_ref().unwrap();
    assert!(!ex.fts_scores.is_empty());
    assert!(ex.vector_scores.is_empty());
}

// ── structured validation (C2/C5) ──

#[test]
fn structured_rejects_original_kind() {
    // C2 — `Original` is ranked-list metadata, not a structured input
    // kind; upstream's validator would reject it as InvalidQuery.
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha\n")],
        &Backends {
            embedder: None,
            reranker: None,
            generator: None,
        },
        false,
    );
    let err = qmd
        .search(
            &structured(vec![ExpandedQuery::new(QueryKind::Original, "alpha")]),
            &(),
        )
        .unwrap_err();
    assert!(matches!(err, qmd::Error::InvalidQuery { .. }), "{err:?}");
    let msg = err.to_string();
    assert!(msg.contains("(original):"), "{msg}");
    assert!(msg.contains("lex, vec or hyde"), "{msg}");
}

#[test]
fn structured_line_zero_is_not_a_source_line() {
    // C5 — upstream `search.line ? ...` (store.ts:5980): line 0 is
    // falsy, so the generic location label is used.
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha\n")],
        &Backends {
            embedder: None,
            reranker: None,
            generator: None,
        },
        false,
    );
    let err = qmd
        .search(
            &structured(vec![
                ExpandedQuery::new(QueryKind::Lex, "foo\nbar").with_line(0),
            ]),
            &(),
        )
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Structured search (lex):"), "{msg}");
    assert!(!msg.contains("Line 0"), "{msg}");
}

// ── rerank cache keyed by reranker model (T7 / #764) ──

#[test]
fn rerank_cache_key_includes_reranker_model_id() {
    // store.test.ts "rerank cache key includes the resolved rerank
    // model" (#764): same database, a different `model_id` must not
    // reuse another model's cached scores.
    let tmp = tempfile::tempdir().unwrap();
    let docs_dir = tmp.path().join("docs");
    write(&docs_dir, "a.md", "# A\nalpha bravo charlie\n");
    let db = tmp.path().join("i.sqlite");
    let mk_gen = || Arc::new(ScriptedGenerator::output("m-gen", ""));

    let ra = Arc::new(ScriptedReranker::constant("m-rerank-a", 0.11));
    let ra_dyn: Arc<dyn Reranker> = Arc::<ScriptedReranker>::clone(&ra);
    let qmd_a = Qmd::builder(&db)
        .environment(env())
        .reranker(ra_dyn)
        .generator(mk_gen())
        .build()
        .unwrap();
    add_docs_collection(&qmd_a, &docs_dir, "docs");
    let mut opts = query_opts(SearchInput::Query("alpha".to_owned()));
    opts.explain = true;
    let first = qmd_a.search(&opts, &()).unwrap();
    assert_eq!(ra.calls.load(Ordering::SeqCst), 1);
    assert!((first[0].explain.as_ref().unwrap().rerank_score - f64::from(0.11_f32)).abs() < 1e-5);
    drop(qmd_a);

    // Second handle, same DB, different reranker model → the cached
    // model-A score must not be reused.
    let rb = Arc::new(ScriptedReranker::constant("m-rerank-b", 0.99));
    let rb_dyn: Arc<dyn Reranker> = Arc::<ScriptedReranker>::clone(&rb);
    let qmd_b = Qmd::builder(&db)
        .environment(env())
        .reranker(rb_dyn)
        .generator(mk_gen())
        .build()
        .unwrap();
    let second = qmd_b.search(&opts, &()).unwrap();
    assert_eq!(rb.calls.load(Ordering::SeqCst), 1);
    assert!((second[0].explain.as_ref().unwrap().rerank_score - f64::from(0.99_f32)).abs() < 1e-5);

    // Same model again → cache hit, no third reranker call.
    let third = qmd_b.search(&opts, &()).unwrap();
    assert_eq!(rb.calls.load(Ordering::SeqCst), 1);
    assert!((third[0].explain.as_ref().unwrap().rerank_score - f64::from(0.99_f32)).abs() < 1e-5);
}

// ── structuredSearch group (T5, structured-search.test.ts:283-352) ──

#[test]
fn structured_no_matching_documents_returns_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha bravo\n")],
        &Backends {
            embedder: None,
            reranker: None,
            generator: None,
        },
        false,
    );
    let results = qmd
        .search(
            &structured(vec![ExpandedQuery::new(
                QueryKind::Lex,
                "nonexistent-term-xyz123",
            )]),
            &(),
        )
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn structured_accepts_lex_vec_hyde_together() {
    // Needs real vectors: embed the docs so `vectors_vec` exists.
    let tmp = tempfile::tempdir().unwrap();
    let e = Arc::new(ScriptedEmbedder::hashed("m-embed", 8));
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nalpha bravo charlie\n")],
        &Backends {
            embedder: Some(e),
            reranker: Some(Arc::new(ScriptedReranker::constant("m-rerank", 0.5))),
            generator: None,
        },
        true,
    );
    let results = qmd
        .search(
            &structured(vec![
                ExpandedQuery::new(QueryKind::Lex, "alpha"),
                ExpandedQuery::new(QueryKind::Vec, "alpha meaning"),
                ExpandedQuery::new(QueryKind::Hyde, "a note about alpha"),
            ]),
            &(),
        )
        .unwrap();
    assert!(!results.is_empty());
}

#[test]
fn structured_respects_limit_and_min_score() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[
            ("a.md", "# A\nalpha bravo charlie\n"),
            ("b.md", "# B\nalpha echo foxtrot\n"),
            ("c.md", "# C\nalpha golf hotel\n"),
        ],
        &Backends {
            embedder: None,
            reranker: None,
            generator: None,
        },
        false,
    );

    let results = qmd
        .search(
            &structured(vec![ExpandedQuery::new(QueryKind::Lex, "alpha")]),
            &(),
        )
        .unwrap();
    assert_eq!(results.len(), 3);

    let mut opts = structured(vec![ExpandedQuery::new(QueryKind::Lex, "alpha")]);
    opts.limit = 2;
    assert_eq!(qmd.search(&opts, &()).unwrap().len(), 2);

    // Skip-rerank scores are 1/rrf_rank → 1, 1/2, 1/3; 0.5 keeps two.
    opts.limit = 10;
    opts.min_score = 0.5;
    let results = qmd.search(&opts, &()).unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.score >= 0.5));
}

// ── sdk.test.ts "search (unified API)" (T9, sdk.test.ts:582-689) ──

#[test]
fn sdk_pre_expanded_queries_with_rerank_false() {
    // "search() with pre-expanded queries and rerank:false".
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[
            ("a.md", "# Auth\nauthentication JWT tokens login session\n"),
            ("b.md", "# Notes\nunrelated cooking notes\n"),
        ],
        &Backends {
            embedder: None,
            reranker: None,
            generator: None,
        },
        false,
    );
    let results = qmd
        .search(
            &structured(vec![
                ExpandedQuery::new(QueryKind::Lex, "authentication"),
                ExpandedQuery::new(QueryKind::Lex, "login"),
            ]),
            &(),
        )
        .unwrap();
    assert!(!results.is_empty());
}

#[test]
fn sdk_candidate_limit_reaches_structured_search() {
    // "forwards candidateLimit" — with candidate_limit 1 only the top
    // fused candidate survives, regardless of `limit`.
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[
            ("a.md", "# A\nauthentication notes\n"),
            ("b.md", "# B\nauthentication other\n"),
            ("c.md", "# C\nauthentication third\n"),
        ],
        &Backends {
            embedder: None,
            reranker: None,
            generator: None,
        },
        false,
    );
    let mut opts = structured(vec![
        ExpandedQuery::new(QueryKind::Lex, "authentication"),
        ExpandedQuery::new(QueryKind::Lex, "notes"),
    ]);
    opts.limit = 5;
    opts.candidate_limit = 1;
    let results = qmd.search(&opts, &()).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn sdk_query_with_rerank_false_returns_results() {
    // "search() with query and rerank:false returns results" — the
    // scripted generator stands in for LLM expansion; field shape
    // mirrors the upstream `toHaveProperty` list.
    let tmp = tempfile::tempdir().unwrap();
    let (_e, _r, g) = default_backends();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nauthentication JWT tokens\n")],
        &Backends {
            embedder: None,
            reranker: Some(Arc::new(ScriptedReranker::constant("m-rerank", 0.5))),
            generator: Some(g),
        },
        false,
    );
    let mut opts = query_opts(SearchInput::Query("authentication".to_owned()));
    opts.rerank = false;
    let results = qmd.search(&opts, &()).unwrap();
    assert!(!results.is_empty());
    let r = &results[0];
    assert!(r.filepath.starts_with("qmd://"));
    assert!(r.score.is_finite());
    assert!(!r.title.is_empty());
    assert!(!r.best_chunk.is_empty());
    assert!(!r.docid.is_empty());
}

#[test]
fn sdk_query_with_intent_and_rerank_false() {
    // "search() with intent and rerank:false returns results".
    let tmp = tempfile::tempdir().unwrap();
    let (_e, _r, g) = default_backends();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[("a.md", "# A\nmeeting notes quarterly\n")],
        &Backends {
            embedder: None,
            reranker: Some(Arc::new(ScriptedReranker::constant("m-rerank", 0.5))),
            generator: Some(g),
        },
        false,
    );
    let mut opts = query_opts(SearchInput::Query("meeting".to_owned()));
    opts.intent = Some("quarterly planning and roadmap".to_owned());
    opts.rerank = false;
    let results = qmd.search(&opts, &()).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn sdk_collection_and_collections_filter() {
    // "search() with collection filter" / "with collections filter" —
    // upstream's singular and plural params both land in
    // `SearchOptions::collections`.
    let tmp = tempfile::tempdir().unwrap();
    let dir_a = tmp.path().join("docs");
    let dir_b = tmp.path().join("notes");
    write(&dir_a, "a.md", "# A\nauthentication tokens\n");
    write(&dir_b, "b.md", "# B\nauthentication diary\n");
    let db = tmp.path().join("i.sqlite");
    let qmd = Qmd::builder(&db)
        .environment(env())
        .reranker(Arc::new(ScriptedReranker::constant("m-rerank", 0.5)))
        .generator(Arc::new(ScriptedGenerator::output("m-gen", "")))
        .build()
        .unwrap();
    add_docs_collection(&qmd, &dir_a, "docs");
    add_docs_collection(&qmd, &dir_b, "notes");

    let mut opts = query_opts(SearchInput::Query("authentication".to_owned()));
    opts.rerank = false;
    opts.intent = Some("ctx".to_owned()); // keep expansion deterministic
    opts.collections = Some(vec!["docs".to_owned()]);
    let results = qmd.search(&opts, &()).unwrap();
    assert!(!results.is_empty());
    assert!(
        results
            .iter()
            .all(|r| r.filepath.starts_with("qmd://docs/"))
    );
}

#[test]
fn sdk_search_respects_limit_and_empty_for_non_match() {
    // "search() with limit" + "returns empty for non-matching query".
    let tmp = tempfile::tempdir().unwrap();
    let (_e, _r, g) = default_backends();
    let (qmd, _db) = pipeline(
        tmp.path(),
        &[
            ("a.md", "# A\nmeeting alpha\n"),
            ("b.md", "# B\nmeeting bravo\n"),
        ],
        &Backends {
            embedder: None,
            reranker: Some(Arc::new(ScriptedReranker::constant("m-rerank", 0.5))),
            generator: Some(g),
        },
        false,
    );
    let mut opts = query_opts(SearchInput::Query("meeting".to_owned()));
    opts.rerank = false;
    opts.limit = 1;
    let results = qmd.search(&opts, &()).unwrap();
    assert_eq!(results.len(), 1);

    let mut opts = query_opts(SearchInput::Query("xyznonexistentterm123".to_owned()));
    opts.rerank = false;
    assert!(qmd.search(&opts, &()).unwrap().is_empty());
}
