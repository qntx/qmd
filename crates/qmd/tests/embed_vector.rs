#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::shadow_unrelated,
    clippy::field_reassign_with_default,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_assert_message,
    clippy::panic,
    reason = "idiomatic test-code patterns"
)]

//! Embedding pipeline and vector search — ports the vector-facing
//! coverage of upstream `store.test.ts` (Vector Table, Embedding
//! batching, Vector Search collection filter, Integration, Edge Cases)
//! and `sdk.test.ts` (`embed`/`searchVector`), driven by injectable
//! stub embedders so retry/circuit-breaker/timeout paths are
//! deterministic.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use qmd::{
    CancelToken, CollectionSpec, EmbedOptions, Embedder, Environment, Error, InferenceError, Qmd,
    SearchSource, UpdateOptions, VectorOptions,
};

fn write(dir: &Path, rel: &str, content: &str) {
    let path = dir.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}

fn raw_conn(path: &Path) -> rusqlite::Connection {
    rusqlite::Connection::open(path).expect("open raw connection")
}

fn env() -> Environment {
    Environment::default()
}

/// `texts` are already formatted (`title: … | text: …`).
type EmbedFn = dyn Fn(&[&str]) -> Result<Vec<Vec<f32>>, InferenceError> + Send + Sync;

/// Embedder whose `embed` batch is fully controlled by a closure —
/// upstream tests inject arbitrary `session.embed`/`embedBatch` shapes.
struct StubEmbedder {
    model_id: String,
    dimension: usize,
    embed_fn: Box<EmbedFn>,
    calls: AtomicUsize,
}

impl StubEmbedder {
    fn new(
        model_id: &str,
        dimension: usize,
        f: impl Fn(&[&str]) -> Result<Vec<Vec<f32>>, InferenceError> + Send + Sync + 'static,
    ) -> Self {
        Self {
            model_id: model_id.to_owned(),
            dimension,
            embed_fn: Box::new(f),
            calls: AtomicUsize::new(0),
        }
    }
}

impl std::fmt::Debug for StubEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StubEmbedder")
            .field("model_id", &self.model_id)
            .field("dimension", &self.dimension)
            .finish_non_exhaustive()
    }
}

/// A deterministic pseudo-vector: hash the text, spread over `dim`.
fn hashed_vector(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    for (i, b) in text.bytes().enumerate() {
        let slot = i % dim;
        v[slot] += f32::from(b) / 255.0;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Convenience: every input gets its hashed vector.
fn ok_embedder(model_id: &str, dimension: usize) -> StubEmbedder {
    StubEmbedder::new(model_id, dimension, move |texts| {
        Ok(texts.iter().map(|t| hashed_vector(t, dimension)).collect())
    })
}

impl Embedder for StubEmbedder {
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
        (self.embed_fn)(texts)
    }
}

/// A `Qmd` handle indexing `tmp/docs` with `docs` collection and the
/// given embedder injected.
fn indexed_qmd(
    tmp: &Path,
    docs: &[(&str, &str)],
    embedder: Arc<dyn Embedder>,
) -> (Qmd, std::path::PathBuf) {
    let docs_dir = tmp.join("docs");
    for (rel, body) in docs {
        write(&docs_dir, rel, body);
    }
    let db = tmp.join("i.sqlite");
    let qmd = Qmd::builder(&db)
        .environment(env())
        .embedder(embedder)
        .build()
        .unwrap();
    qmd.add_collection(
        "docs",
        &CollectionSpec {
            path: docs_dir.to_string_lossy().into_owned(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();
    qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();
    (qmd, db)
}

fn vec_row_count(db: &Path) -> i64 {
    raw_conn(db)
        .query_row("SELECT COUNT(*) FROM vectors_vec", [], |r| r.get(0))
        .unwrap()
}

fn cv_row_count(db: &Path) -> i64 {
    raw_conn(db)
        .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))
        .unwrap()
}

// ── Happy path (store.test.ts Vector Table / Integration) ──

#[test]
fn embed_creates_vec_table_and_marks_done() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, db) = indexed_qmd(
        tmp.path(),
        &[
            ("a.md", "# Alpha\nalpha body\n"),
            ("b.md", "# Beta\nbeta body\n"),
        ],
        Arc::new(ok_embedder("test-embed", 8)),
    );

    let mut events = Vec::new();
    let report = qmd
        .embed(&EmbedOptions::default(), &mut |p| {
            events.push((p.chunks_embedded, p.total_chunks));
        })
        .unwrap();
    assert_eq!(report.docs_processed, 2);
    assert_eq!(report.chunks_embedded, 2);
    assert_eq!(report.errors, 0);
    assert_eq!(cv_row_count(&db), 2);
    assert_eq!(vec_row_count(&db), 2);
    assert_eq!(qmd.status().unwrap().needs_embedding, 0);
    assert!(!events.is_empty());

    // Second run: nothing pending → early return with zeroes (upstream).
    let report = qmd.embed(&EmbedOptions::default(), &mut |_| {}).unwrap();
    assert_eq!(report.docs_processed, 0);
    assert_eq!(report.chunks_embedded, 0);
}

#[test]
fn embed_row_shape_matches_upstream() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, db) = indexed_qmd(
        tmp.path(),
        &[("a.md", "# Alpha\nalpha body\n")],
        Arc::new(ok_embedder("test-embed", 8)),
    );
    qmd.embed(&EmbedOptions::default(), &mut |_| {}).unwrap();

    let conn = raw_conn(&db);
    let (model, total, pos, fp_len, at): (String, i64, i64, i64, String) = conn
        .query_row(
            "SELECT model, total_chunks, pos, length(embed_fingerprint), embedded_at \
             FROM content_vectors",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )
        .unwrap();
    assert_eq!(model, "test-embed");
    assert_eq!(total, 1);
    assert_eq!(pos, 0);
    assert_eq!(fp_len, 6, "fingerprint is a 6-hex prefix");
    assert!(at.ends_with('Z'), "ISO-8601 timestamp: {at}");
}

// ── force / collection scope ──

#[test]
fn embed_force_rebuilds() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, db) = indexed_qmd(
        tmp.path(),
        &[("a.md", "# Alpha\nalpha body\n")],
        Arc::new(ok_embedder("test-embed", 8)),
    );
    qmd.embed(&EmbedOptions::default(), &mut |_| {}).unwrap();
    // New embedder identity: rows are stale under a different
    // fingerprint, so force clears and re-embeds.
    let qmd2 = Qmd::builder(&db)
        .environment(env())
        .embedder(Arc::new(ok_embedder("test-embed-v2", 8)))
        .build()
        .unwrap();
    let report = qmd2
        .embed(
            &EmbedOptions {
                force: true,
                ..EmbedOptions::default()
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(report.docs_processed, 1);
    let n: i64 = raw_conn(&db)
        .query_row(
            "SELECT COUNT(*) FROM content_vectors WHERE model = 'test-embed-v2'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 1);
    qmd.close().unwrap();
}

#[test]
fn embed_scoped_collection_only() {
    let tmp = tempfile::tempdir().unwrap();
    let docs_a = tmp.path().join("a");
    let docs_b = tmp.path().join("b");
    write(&docs_a, "a.md", "# A\naaa\n");
    write(&docs_b, "b.md", "# B\nbbb\n");
    let db = tmp.path().join("i.sqlite");
    let qmd = Qmd::builder(&db)
        .environment(env())
        .embedder(Arc::new(ok_embedder("test-embed", 8)))
        .build()
        .unwrap();
    qmd.add_collection(
        "ca",
        &CollectionSpec {
            path: docs_a.to_string_lossy().into_owned(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();
    qmd.add_collection(
        "cb",
        &CollectionSpec {
            path: docs_b.to_string_lossy().into_owned(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();
    qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();

    let report = qmd
        .embed(
            &EmbedOptions {
                collection: Some("ca".to_owned()),
                ..EmbedOptions::default()
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(report.docs_processed, 1);
    // The other collection is still pending.
    assert_eq!(qmd.status().unwrap().needs_embedding, 1);
}

#[test]
fn embed_force_collection_preserves_shared_hash() {
    // Same body indexed in two collections shares one content hash —
    // clearing one collection must keep the vectors (upstream
    // exclusive-hash rule, store.ts:4444-4481).
    let tmp = tempfile::tempdir().unwrap();
    let docs_a = tmp.path().join("a");
    let docs_b = tmp.path().join("b");
    write(&docs_a, "same.md", "# Shared\nidentical body\n");
    write(&docs_b, "also-same.md", "# Shared\nidentical body\n");
    let db = tmp.path().join("i.sqlite");
    let qmd = Qmd::builder(&db)
        .environment(env())
        .embedder(Arc::new(ok_embedder("test-embed", 8)))
        .build()
        .unwrap();
    for (name, dir) in [("ca", &docs_a), ("cb", &docs_b)] {
        qmd.add_collection(
            name,
            &CollectionSpec {
                path: dir.to_string_lossy().into_owned(),
                ..CollectionSpec::default()
            },
        )
        .unwrap();
    }
    qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();
    qmd.embed(&EmbedOptions::default(), &mut |_| {}).unwrap();
    assert_eq!(cv_row_count(&db), 1); // shared hash → one row

    let report = qmd
        .embed(
            &EmbedOptions {
                force: true,
                collection: Some("ca".to_owned()),
                ..EmbedOptions::default()
            },
            &mut |_| {},
        )
        .unwrap();
    // Nothing deleted (hash is shared), nothing pending afterwards.
    assert_eq!(report.docs_processed, 0);
    assert_eq!(cv_row_count(&db), 1);
}

// ── Dimension handling ──

#[test]
fn embed_dimension_mismatch_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, db) = indexed_qmd(
        tmp.path(),
        &[("a.md", "# Alpha\nalpha body\n")],
        Arc::new(ok_embedder("test-embed", 8)),
    );
    // Pre-create a wrong-dimension table, as a previous model left it.
    raw_conn(&db)
        .execute_batch(
            "CREATE VIRTUAL TABLE vectors_vec USING vec0(\
               hash_seq TEXT PRIMARY KEY, embedding float[4] distance_metric=cosine)",
        )
        .unwrap();
    let err = qmd
        .embed(&EmbedOptions::default(), &mut |_| {})
        .unwrap_err();
    assert!(
        matches!(
            err,
            Error::EmbeddingDimensionMismatch {
                existing: 4,
                current: 8
            }
        ),
        "{err}"
    );
    qmd.close().unwrap();
}

// ── Batch limits (store.test.ts:4194/4262/4630, sdk.test.ts:987/1104) ──

/// Embedder whose `embed` records each batch's size.
fn recording_embedder(dimension: usize) -> (StubEmbedder, Arc<Mutex<Vec<usize>>>) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&calls);
    let embedder = StubEmbedder::new("test-embed", dimension, move |texts| {
        sink.lock().unwrap().push(texts.len());
        Ok(texts.iter().map(|t| hashed_vector(t, dimension)).collect())
    });
    (embedder, calls)
}

#[test]
fn embed_flushes_batches_at_max_docs_per_batch() {
    let tmp = tempfile::tempdir().unwrap();
    let (embedder, calls) = recording_embedder(8);
    let (qmd, db) = indexed_qmd(
        tmp.path(),
        &[
            ("one.md", "# One\n\nAlpha"),
            ("two.md", "# Two\n\nBeta"),
            ("three.md", "# Three\n\nGamma"),
        ],
        Arc::new(embedder),
    );
    let report = qmd
        .embed(
            &EmbedOptions {
                max_docs_per_batch: 1,
                max_batch_bytes: 1024 * 1024,
                ..EmbedOptions::default()
            },
            &mut |_| {},
        )
        .unwrap();
    // calls[0] is the single-chunk dimension probe — upstream runs that
    // probe through `session.embed`, which `embedBatchCalls` does not
    // record, so compare `calls[1..]`.
    assert_eq!(calls.lock().unwrap()[1..], [1, 1, 1]);
    assert_eq!(report.docs_processed, 3);
    assert_eq!(report.chunks_embedded, 3);
    assert_eq!(vec_row_count(&db), 3);
}

#[test]
fn embed_flushes_batches_at_max_batch_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    let doc_one = format!("# One\n\n{}", "A".repeat(36));
    let doc_two = format!("# Two\n\n{}", "B".repeat(36));
    let doc_three = format!("# Three\n\n{}", "C".repeat(36));
    let batch_limit = doc_one.len() + doc_two.len() + 1;
    let (embedder, calls) = recording_embedder(8);
    let (qmd, _db) = indexed_qmd(
        tmp.path(),
        &[
            ("a-one.md", &doc_one),
            ("b-two.md", &doc_two),
            ("c-three.md", &doc_three),
        ],
        Arc::new(embedder),
    );
    let report = qmd
        .embed(
            &EmbedOptions {
                max_docs_per_batch: 64,
                max_batch_bytes: batch_limit,
                ..EmbedOptions::default()
            },
            &mut |_| {},
        )
        .unwrap();
    // Same probe offset as above.
    assert_eq!(calls.lock().unwrap()[1..], [2, 1]);
    assert_eq!(report.docs_processed, 3);
    assert_eq!(report.chunks_embedded, 3);
}

#[test]
fn embed_rejects_invalid_batch_limits() {
    let tmp = tempfile::tempdir().unwrap();
    let (embedder, calls) = recording_embedder(8);
    // No collections: upstream builds the store with `collections: {}`.
    let qmd = Qmd::builder(tmp.path().join("i.sqlite"))
        .environment(env())
        .embedder(Arc::new(embedder))
        .build()
        .unwrap();
    let err = qmd
        .embed(
            &EmbedOptions {
                max_docs_per_batch: 0,
                ..EmbedOptions::default()
            },
            &mut |_| {},
        )
        .unwrap_err();
    match err {
        Error::InvalidInput { ref reason } => {
            assert!(reason.contains("maxDocsPerBatch"), "{reason}");
        }
        e => panic!("expected InvalidInput, got {e:?}"),
    }
    let err = qmd
        .embed(
            &EmbedOptions {
                max_batch_bytes: 0,
                ..EmbedOptions::default()
            },
            &mut |_| {},
        )
        .unwrap_err();
    match err {
        Error::InvalidInput { ref reason } => {
            assert!(reason.contains("maxBatchBytes"), "{reason}");
        }
        e => panic!("expected InvalidInput, got {e:?}"),
    }
    // Upstream validates before touching the LLM.
    assert!(calls.lock().unwrap().is_empty());
}

// ── Retry / fallback / circuit breaker / cancellation ──

#[test]
fn embed_batch_failure_falls_back_to_single() {
    let tmp = tempfile::tempdir().unwrap();
    // Batch calls (>1 text) fail; singles succeed — upstream's
    // embedBatch-throws path (store.ts:2180-2190).
    let stub = StubEmbedder::new("test-embed", 8, |texts| {
        if texts.len() > 1 {
            Err(InferenceError::Backend("batch exploded".to_owned()))
        } else {
            Ok(texts.iter().map(|t| hashed_vector(t, 8)).collect())
        }
    });
    let (qmd, db) = indexed_qmd(
        tmp.path(),
        &[
            ("a.md", "# Alpha\nalpha body\n"),
            ("b.md", "# Beta\nbeta body\n"),
        ],
        Arc::new(stub),
    );
    let report = qmd.embed(&EmbedOptions::default(), &mut |_| {}).unwrap();
    assert_eq!(report.chunks_embedded, 2);
    assert_eq!(report.errors, 0);
    assert_eq!(vec_row_count(&db), 2);
}

#[test]
fn embed_retry_recovers_and_reports_attempts() {
    let tmp = tempfile::tempdir().unwrap();
    // The dimension probe (first single embed) succeeds; the batch call
    // fails; the per-chunk fallback fails for "boom" once then recovers.
    let fail_once = Arc::new(AtomicUsize::new(0));
    let flag = fail_once;
    let stub = StubEmbedder::new("test-embed", 8, move |texts| {
        if texts.len() > 1 {
            return Err(InferenceError::Backend("batch dead".to_owned()));
        }
        if texts[0].contains("boom") && flag.fetch_add(1, Ordering::SeqCst) == 0 {
            return Err(InferenceError::Backend("transient".to_owned()));
        }
        Ok(texts.iter().map(|t| hashed_vector(t, 8)).collect())
    });
    let (qmd, db) = indexed_qmd(
        tmp.path(),
        &[
            ("a.md", "# Alpha\nalpha body\n"),
            ("b.md", "# Beta\nboom marker\n"),
        ],
        Arc::new(stub),
    );
    let report = qmd.embed(&EmbedOptions::default(), &mut |_| {}).unwrap();
    // Transient failure retried inside the same run → all embedded,
    // zero outstanding errors.
    assert_eq!(report.errors, 0);
    assert_eq!(report.chunks_embedded, 2);
    assert_eq!(vec_row_count(&db), 2);
}

#[test]
fn embed_circuit_breaker_skips_later_slices() {
    let tmp = tempfile::tempdir().unwrap();
    // Probe + first-slice batch + per-chunk fallback all fail (calls
    // 2..=34); the doc-batch-end force retry succeeds. Upstream's
    // breaker (store.ts:2163-2170) marks slice 2's chunks without
    // attempting them — observable as exactly ONE batch call (slice 1)
    // for 40 docs — then `retryFailedChunks(true)` recovers all of them
    // (recordFailure overwrites the marker reason on retry).
    let calls = Arc::new(AtomicUsize::new(0));
    let n = calls;
    let batch_calls = Arc::new(AtomicUsize::new(0));
    let bc = Arc::clone(&batch_calls);
    let stub = StubEmbedder::new("test-embed", 8, move |texts| {
        let i = n.fetch_add(1, Ordering::SeqCst);
        if texts.len() > 1 {
            bc.fetch_add(1, Ordering::SeqCst);
        }
        // Call 0 = dimension probe (must succeed); calls 1-33 = the
        // slice-1 batch + 32 per-chunk fallbacks (all fail); calls ≥34
        // = batch-end force retries (succeed).
        if i == 0 || i >= 34 {
            return Ok(texts.iter().map(|t| hashed_vector(t, 8)).collect());
        }
        Err(InferenceError::Backend("down".to_owned()))
    });
    let docs: Vec<(String, String)> = (0..40)
        .map(|i| (format!("d{i}.md"), format!("# Doc {i}\nbody {i}\n")))
        .collect();
    let refs: Vec<(&str, &str)> = docs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let (qmd, db) = indexed_qmd(tmp.path(), &refs, Arc::new(stub));
    let report = qmd.embed(&EmbedOptions::default(), &mut |_| {}).unwrap();
    assert_eq!(report.docs_processed, 40);
    // The probe isn't counted (it's dimension probing, not storage);
    // all 40 chunks land via the force-retry pass.
    assert_eq!(report.chunks_embedded, 40);
    assert_eq!(report.errors, 0);
    // The breaker skipped slice 2's batch call entirely.
    assert_eq!(batch_calls.load(Ordering::SeqCst), 1);
    assert_eq!(vec_row_count(&db), 40);
    assert_eq!(cv_row_count(&db), 40);
}

#[test]
fn embed_permanent_failure_cleans_incomplete_doc() {
    let tmp = tempfile::tempdir().unwrap();
    // >900-token doc (1 token/char tokenizer, 900-token cap) splits
    // into multiple chunks; "FAILCHUNK" sits past char 1000 so only the
    // tail chunk fails. remove_incomplete drops the partial hash
    // entirely (store.ts:4524-4547).
    let long_body = format!("# Long\n{}FAILCHUNK{}\n", "x".repeat(1000), "y".repeat(300));
    let stub = StubEmbedder::new("test-embed", 8, |texts| {
        // Fail only inputs that carry the marker — batch calls that
        // include it fail wholesale (forcing the per-chunk fallback).
        if texts.iter().any(|t| t.contains("FAILCHUNK")) {
            return Err(InferenceError::Backend("bad chunk".to_owned()));
        }
        Ok(texts.iter().map(|t| hashed_vector(t, 8)).collect())
    });
    let (qmd, db) = indexed_qmd(
        tmp.path(),
        &[("good.md", "# Good\nfine body\n"), ("long.md", &long_body)],
        Arc::new(stub),
    );
    let report = qmd.embed(&EmbedOptions::default(), &mut |_| {}).unwrap();
    assert!(report.errors > 0);
    // Partial content_vectors/vectors_vec rows for the failed doc were
    // removed; the good doc's rows stay.
    let conn = raw_conn(&db);
    let orphaned: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM content_vectors cv \
             WHERE cv.hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(orphaned, 0);
    let partial: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM content_vectors cv JOIN documents d ON d.hash = cv.hash \
             WHERE d.path = 'long.md'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(partial, 0, "incomplete hash must be fully removed");
}

#[test]
fn embed_cancel_marks_remaining_failed() {
    let tmp = tempfile::tempdir().unwrap();
    let docs: Vec<(String, String)> = (0..40)
        .map(|i| (format!("d{i}.md"), format!("# Doc {i}\nbody {i}\n")))
        .collect();
    let refs: Vec<(&str, &str)> = docs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let token = CancelToken::new();
    let cancel = token.clone();
    let calls = Arc::new(AtomicUsize::new(0));
    let n = calls;
    let stub = StubEmbedder::new("test-embed", 8, move |texts| {
        // Cancel once the first slice has embedded.
        if n.fetch_add(1, Ordering::SeqCst) >= 1 {
            cancel.cancel();
        }
        Ok(texts.iter().map(|t| hashed_vector(t, 8)).collect())
    });
    let (qmd, _db) = indexed_qmd(tmp.path(), &refs, Arc::new(stub));
    let report = qmd
        .embed(
            &EmbedOptions {
                cancel: token,
                ..EmbedOptions::default()
            },
            &mut |_| {},
        )
        .unwrap();
    // First 32-chunk slice embedded; the remaining 8 were marked
    // "session expired" — partial report, not an error.
    assert_eq!(report.chunks_embedded, 32);
    assert_eq!(report.errors, 8);
    assert!(
        report
            .failures
            .iter()
            .all(|f| f.reason.contains("session expired"))
    );

    // Re-run resumes where it stopped (upstream "re-run continues").
    let report = qmd.embed(&EmbedOptions::default(), &mut |_| {}).unwrap();
    assert_eq!(report.docs_processed, 8);
}

#[test]
fn embed_timeout_returns_partial_report() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = indexed_qmd(
        tmp.path(),
        &[("a.md", "# Alpha\nalpha body\n")],
        Arc::new(ok_embedder("test-embed", 8)),
    );
    let report = qmd
        .embed(
            &EmbedOptions {
                max_duration: Some(std::time::Duration::from_nanos(1)),
                ..EmbedOptions::default()
            },
            &mut |_| {},
        )
        .unwrap();
    // Session invalid before the first batch: upstream skips remaining
    // batches and reports the *pending* count (store.ts:2237).
    assert_eq!(report.docs_processed, 1);
    assert_eq!(report.chunks_embedded, 0);
    assert_eq!(report.errors, 0);
}

// ── search_vector ──

#[test]
fn search_vector_empty_without_table() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = indexed_qmd(
        tmp.path(),
        &[("a.md", "# Alpha\nalpha body\n")],
        Arc::new(ok_embedder("test-embed", 8)),
    );
    let out = qmd
        .search_vector("anything", &VectorOptions::default())
        .unwrap();
    assert!(out.is_empty());
}

#[test]
fn search_vector_returns_scored_hits() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = indexed_qmd(
        tmp.path(),
        &[
            ("a.md", "# Alpha\nalpha body\n"),
            ("b.md", "# Beta\nbeta body\n"),
        ],
        Arc::new(ok_embedder("test-embed", 8)),
    );
    qmd.embed(&EmbedOptions::default(), &mut |_| {}).unwrap();
    let out = qmd
        .search_vector("alpha", &VectorOptions::default())
        .unwrap();
    assert_eq!(out.len(), 2);
    for r in &out {
        assert_eq!(r.source, SearchSource::Vec);
        assert_eq!(r.chunk_pos, Some(0));
        assert!(r.score > 0.0 && r.score <= 1.0, "score {r:?}");
        assert!(r.filepath.starts_with("qmd://docs/"));
        assert_eq!(r.collection_name, "docs");
    }
    // Sorted by score descending.
    assert!(out.windows(2).all(|w| w[0].score >= w[1].score));
}

#[test]
fn search_vector_collection_filter() {
    let tmp = tempfile::tempdir().unwrap();
    let docs_a = tmp.path().join("a");
    let docs_b = tmp.path().join("b");
    write(&docs_a, "a.md", "# A\nalpha\n");
    write(&docs_b, "b.md", "# B\nbeta\n");
    let db = tmp.path().join("i.sqlite");
    let qmd = Qmd::builder(&db)
        .environment(env())
        .embedder(Arc::new(ok_embedder("test-embed", 8)))
        .build()
        .unwrap();
    for (name, dir) in [("ca", &docs_a), ("cb", &docs_b)] {
        qmd.add_collection(
            name,
            &CollectionSpec {
                path: dir.to_string_lossy().into_owned(),
                ..CollectionSpec::default()
            },
        )
        .unwrap();
    }
    qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();
    qmd.embed(&EmbedOptions::default(), &mut |_| {}).unwrap();

    let out = qmd
        .search_vector(
            "x",
            &VectorOptions {
                collections: Some(vec!["ca".to_owned()]),
                ..VectorOptions::default()
            },
        )
        .unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].collection_name, "ca");

    // Multi-collection scope merges per-collection queries.
    let out = qmd
        .search_vector(
            "x",
            &VectorOptions {
                collections: Some(vec!["ca".to_owned(), "cb".to_owned()]),
                ..VectorOptions::default()
            },
        )
        .unwrap();
    assert_eq!(out.len(), 2);
    let mut names: Vec<_> = out.iter().map(|r| r.collection_name.as_str()).collect();
    names.sort_unstable();
    assert_eq!(names, ["ca", "cb"]);
}

// ── AC4: eval-docs embed → search_vector end-to-end (real model) ──
//
// Indexes the vendored eval-docs corpus with the real
// embeddinggemma-300M backend, then asserts vector search returns
// semantically sane results (the doc about X wins for query X).
#[cfg(feature = "llama")]
mod eval_e2e {
    use qmd::llm::download::model_cache_dir;
    use qmd::llm::llamacpp::{EmbeddingMode, LlamaEmbedder};
    use qmd::{CollectionSpec, EmbedOptions, Qmd, UpdateOptions, VectorOptions};

    /// `update → embed → search_vector` on the eval corpus with the
    /// real embeddinggemma backend — the AC4 end-to-end check.
    #[test]
    #[ignore = "requires cached GGUF models"]
    fn embed_then_vector_search_eval_docs() {
        let model_file = model_cache_dir(&qmd::Environment::default())
            .join("hf_ggml-org_embeddinggemma-300M-Q8_0.gguf");
        assert!(
            model_file.exists(),
            "model cache missing: {}",
            model_file.display()
        );

        let tmp = tempfile::tempdir().unwrap();
        let corpus = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/eval-docs");
        let db = tmp.path().join("i.sqlite");
        // Local path — bypasses `pull_models` etag refresh (a missing
        // remote deletes + re-downloads; this test never downloads).
        let embedder = LlamaEmbedder::new(
            &model_file,
            qmd::llm::DEFAULT_EMBED_MODEL,
            &qmd::Environment::default(),
            EmbeddingMode::UpstreamTail,
        )
        .unwrap();
        let qmd = Qmd::builder(&db)
            .embedder(std::sync::Arc::new(embedder))
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
        assert_eq!(report.chunks_embedded, 9); // 6 docs → 9 golden chunks
        assert_eq!(qmd.status().unwrap().needs_embedding, 0);
        assert!(qmd.status().unwrap().has_vector_index);

        // Semantic sanity: the machine-learning doc wins an ML query.
        let hits = qmd
            .search_vector(
                "gradient descent and neural network training",
                &VectorOptions::default(),
            )
            .unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].display_path, "eval/machine-learning-primer.md");
        // chunk_pos is the chunk's UTF-16 offset into the document.
        assert!(hits[0].chunk_pos.is_some());
        // cosine scores are modest for embeddinggemma — assert ranking
        // separation rather than an absolute threshold.
        assert!(hits[0].score > hits[1].score, "hits {hits:?}");

        // Collection filter narrows to an empty set cleanly.
        let none = qmd
            .search_vector(
                "anything",
                &VectorOptions {
                    collections: Some(vec!["nonexistent".to_owned()]),
                    ..VectorOptions::default()
                },
            )
            .unwrap();
        assert!(none.is_empty());
    }
}
