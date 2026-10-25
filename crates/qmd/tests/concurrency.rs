#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::shadow_unrelated,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_assert_message,
    clippy::panic,
    reason = "idiomatic test-code patterns"
)]

//! Connection-lock release during inference (parity-matrix deviation
//! 43): one shared `Arc<Qmd>` must serve a second caller
//! while a first call is parked inside an embedder/reranker/generator.
//! No upstream test file — upstream's store never takes a lock.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use qmd::{
    Embedder, Environment, ExpandedQuery, Generator, InferenceError, LexOptions, Qmd, QueryKind,
    Reranker, SearchInput, SearchOptions, UpdateOptions, VectorOptions, VectorQueryOptions,
};

fn write(dir: &Path, rel: &str, content: &str) {
    let path = dir.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}

// ── blocking gate ──

/// Inference calls signal `entered`, then block until the main thread
/// sends on `release`. `armed_for` selects which backend kind parks
/// (0 = off, 1 = embedder, 2 = reranker, 3 = generator); it is cleared
/// on the first park so a pipeline's later inference calls run
/// through — each test wants A parked at one specific backend.
struct Gate {
    entered: Sender<()>,
    release: Mutex<Receiver<()>>,
    armed_for: AtomicUsize,
    calls: AtomicUsize,
}

impl Gate {
    fn new() -> (Self, Receiver<()>, Sender<()>) {
        let (entered_tx, entered_rx) = channel();
        let (release_tx, release_rx) = channel();
        (
            Self {
                entered: entered_tx,
                release: Mutex::new(release_rx),
                armed_for: AtomicUsize::new(0),
                calls: AtomicUsize::new(0),
            },
            entered_rx,
            release_tx,
        )
    }

    fn wait(&self, kind: usize) {
        self.calls.fetch_add(1, Ordering::SeqCst);
        // Only the selected backend kind disarms and parks — a pipeline
        // reaches generator→embedder→reranker in order, so an
        // unconditional swap would let the wrong call steal the park.
        if self
            .armed_for
            .compare_exchange(kind, 0, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            // A dropped receiver/sender means the test abandoned A —
            // the park unblocks (or skips the signal) rather than
            // panicking inside the inference call.
            let _entered_sent = self.entered.send(());
            let _released = self.release.lock().unwrap().recv();
        }
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

struct GatedEmbedder {
    gate: Arc<Gate>,
}

impl std::fmt::Debug for GatedEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatedEmbedder").finish_non_exhaustive()
    }
}

impl Embedder for GatedEmbedder {
    fn model_id(&self) -> &'static str {
        "m-gated-embed"
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, InferenceError> {
        Ok(text.chars().map(u32::from).collect())
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, InferenceError> {
        Ok("x".repeat(tokens.len()))
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, InferenceError> {
        self.gate.wait(1);
        Ok(texts.iter().map(|t| hashed_vector(t, 8)).collect())
    }
}

struct GatedReranker {
    gate: Arc<Gate>,
}

impl std::fmt::Debug for GatedReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatedReranker").finish_non_exhaustive()
    }
}

impl Reranker for GatedReranker {
    fn model_id(&self) -> &'static str {
        "m-gated-rerank"
    }

    fn rerank(&self, _query: &str, docs: &[&str]) -> Result<Vec<f32>, InferenceError> {
        self.gate.wait(2);
        Ok(docs.iter().map(|_| 0.5).collect())
    }
}

struct GatedGenerator {
    gate: Arc<Gate>,
}

impl std::fmt::Debug for GatedGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatedGenerator").finish_non_exhaustive()
    }
}

impl Generator for GatedGenerator {
    fn model_id(&self) -> &'static str {
        "m-gated-gen"
    }

    fn expand(&self, query: &str) -> Result<String, InferenceError> {
        self.gate.wait(3);
        Ok(format!("lex: {query} expanded\nvec: {query} meaning"))
    }
}

// ── harness ──

struct Built {
    qmd: Arc<Qmd>,
    _tmp: tempfile::TempDir,
}

fn build(
    embedder: Arc<dyn Embedder>,
    reranker: Arc<dyn Reranker>,
    generator: Arc<dyn Generator>,
    embed_now: bool,
) -> Built {
    let tmp = tempfile::tempdir().unwrap();
    let docs_dir = tmp.path().join("docs");
    write(
        &docs_dir,
        "a.md",
        "alpha bravo charlie\n\nmore words here\n",
    );
    write(&docs_dir, "b.md", "delta echo foxtrot\n\nother content\n");
    let qmd = Qmd::builder(tmp.path().join("i.sqlite"))
        .environment(Environment::default())
        .embedder(embedder)
        .reranker(reranker)
        .generator(generator)
        .build()
        .unwrap();
    qmd.add_collection(
        "docs",
        &qmd::CollectionSpec {
            path: docs_dir.to_string_lossy().into_owned(),
            ..qmd::CollectionSpec::default()
        },
    )
    .unwrap();
    qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();
    if embed_now {
        qmd.embed(&qmd::EmbedOptions::default(), &mut |_| {})
            .unwrap();
    }
    Built {
        qmd: Arc::new(qmd),
        _tmp: tmp,
    }
}

/// Park thread A inside a gated inference call, then prove thread B's
/// DB-touching calls complete while A holds nothing. On a regression
/// (lock held through inference) B times out; release A first so the
/// panic unwinds with no parked thread.
fn run_contended<A, B>(built: &Built, entered: &Receiver<()>, release: &Sender<()>, a: A, b: B)
where
    A: FnOnce(Arc<Qmd>) -> Result<(), String> + Send + 'static,
    B: FnOnce(Arc<Qmd>) -> Result<(), String> + Send + 'static,
{
    // armed_for is set by each test's closure before spawning A.
    let qmd_a = Arc::clone(&built.qmd);
    let th_a = std::thread::spawn(move || a(qmd_a));
    if entered.recv_timeout(Duration::from_secs(10)).is_err() {
        // A finished (or errored) without parking — surface its
        // result instead of a bare timeout.
        let res = th_a.join().expect("thread A panicked");
        panic!("A never reached inference; A returned: {res:?}");
    }

    let (done_tx, done_rx) = channel();
    let qmd_b = Arc::clone(&built.qmd);
    std::thread::spawn(move || done_tx.send(b(qmd_b)));
    match done_rx.recv_timeout(Duration::from_secs(5)) {
        Ok(res) => {
            release.send(()).expect("release A");
            let a_res = th_a.join().expect("thread A panicked");
            res.unwrap_or_else(|e| panic!("B failed while A was parked: {e}"));
            a_res.unwrap_or_else(|e| panic!("A failed after release: {e}"));
        }
        Err(e) => {
            release.send(()).expect("release A before panicking");
            let _a_res = th_a.join();
            panic!("B timed out while A was parked in inference ({e}) — conn lock was held");
        }
    }
}

/// The gate, its two channel halves, and the three gated backends
/// wired to it — returned together so each test picks its slot.
type GatedParts = (
    Arc<Gate>,
    Receiver<()>,
    Sender<()>,
    Arc<GatedEmbedder>,
    Arc<GatedReranker>,
    Arc<GatedGenerator>,
);

fn gated_parts() -> GatedParts {
    let (gate, entered, release) = Gate::new();
    let gate = Arc::new(gate);
    (
        Arc::clone(&gate),
        entered,
        release,
        Arc::new(GatedEmbedder {
            gate: Arc::clone(&gate),
        }),
        Arc::new(GatedReranker {
            gate: Arc::clone(&gate),
        }),
        Arc::new(GatedGenerator {
            gate: Arc::clone(&gate),
        }),
    )
}

// (a) `search_vector` parked in the embedder while B runs get /
// search_lex / status.
#[test]
fn search_vector_releases_conn_during_embed() {
    const ARMED: usize = 1;
    let (gate, entered, release, emb, rer, g) = gated_parts();
    let built = build(emb, rer, g, true);
    gate.armed_for.store(ARMED, Ordering::SeqCst);
    run_contended(
        &built,
        &entered,
        &release,
        |qmd| {
            qmd.search_vector("alpha", &VectorOptions::default())
                .map(|_| ())
                .map_err(|e| e.to_string())
        },
        |qmd| {
            qmd.get("a.md", qmd::GetOptions::default())
                .map_err(|e| e.to_string())?;
            qmd.search_lex("alpha", &LexOptions::default())
                .map_err(|e| e.to_string())?;
            qmd.status().map_err(|e| e.to_string())?;
            Ok(())
        },
    );
}

// (b) hybrid `search` parked in the reranker while B runs search_lex
// and a structured lex-only search.
#[test]
fn search_releases_conn_during_rerank() {
    const ARMED: usize = 2;
    let (gate, entered, release, emb, rer, g) = gated_parts();
    let built = build(emb, rer, g, true);
    gate.armed_for.store(ARMED, Ordering::SeqCst);
    run_contended(
        &built,
        &entered,
        &release,
        |qmd| {
            let mut opts = SearchOptions::default();
            opts.input = SearchInput::Query("alpha".to_owned());
            opts.rerank = true;
            qmd.search(&opts, &())
                .map(|_| ())
                .map_err(|e| e.to_string())
        },
        |qmd| {
            qmd.search_lex("alpha", &LexOptions::default())
                .map_err(|e| e.to_string())?;
            let mut opts = SearchOptions::default();
            opts.input = SearchInput::Expanded(vec![ExpandedQuery::new(QueryKind::Lex, "delta")]);
            opts.rerank = false;
            qmd.search(&opts, &())
                .map(|_| ())
                .map_err(|e| e.to_string())
        },
    );
}

// (c) `expand_query` parked in the generator while B runs get + status.
#[test]
fn expand_query_releases_conn_during_generate() {
    const ARMED: usize = 3;
    let (gate, entered, release, emb, rer, g) = gated_parts();
    let built = build(emb, rer, g, false);
    gate.armed_for.store(ARMED, Ordering::SeqCst);
    run_contended(
        &built,
        &entered,
        &release,
        |qmd| {
            qmd.expand_query("alpha")
                .map(|_| ())
                .map_err(|e| e.to_string())
        },
        |qmd| {
            qmd.get("a.md", qmd::GetOptions::default())
                .map_err(|e| e.to_string())?;
            qmd.status().map_err(|e| e.to_string())?;
            Ok(())
        },
    );
}

// (d) `vector_query` parked in the embedder while B runs search_lex.
#[test]
fn vector_query_releases_conn_during_embed() {
    const ARMED: usize = 1;
    let (gate, entered, release, emb, rer, g) = gated_parts();
    let built = build(emb, rer, g, true);
    gate.armed_for.store(ARMED, Ordering::SeqCst);
    run_contended(
        &built,
        &entered,
        &release,
        |qmd| {
            qmd.vector_query("alpha", &VectorQueryOptions::default(), &())
                .map(|_| ())
                .map_err(|e| e.to_string())
        },
        |qmd| {
            qmd.search_lex("alpha", &LexOptions::default())
                .map_err(|e| e.to_string())?;
            Ok(())
        },
    );
}

// `search_vector` on an index with no `vectors_vec` returns `[]` and
// never touches the embedder.
#[test]
fn search_vector_without_vectors_skips_embedder() {
    let (gate, _entered, _release, emb, rer, g) = gated_parts();
    let calls = Arc::clone(&gate);
    let built = build(emb, rer, g, false);
    let out = built
        .qmd
        .search_vector("alpha", &VectorOptions::default())
        .unwrap();
    assert!(out.is_empty());
    assert_eq!(calls.calls.load(Ordering::SeqCst), 0);
}
