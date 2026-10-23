//! Doctor data probes (P4.3 stage A) — covers the store side of
//! upstream `showDoctor` (cli/qmd.ts:4210-4310) and
//! `maybeAdoptLegacyEmbeddingFingerprint` (store.ts:2581-2650): version
//! probes, active-doc/`vectors_vec` checks, fingerprint groups, vector
//! sample consistency, and legacy fingerprint adoption.

#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::shadow_unrelated,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_assert_message,
    clippy::panic,
    reason = "idiomatic test-code patterns"
)]

use std::path::Path;
use std::sync::Arc;

use qmd::{
    CollectionSpec, EmbedOptions, Embedder, Environment, InferenceError, Qmd, UpdateOptions,
    active_doc_count, check_embedding_vector_samples, current_embed_fingerprint,
    fingerprint_groups, maybe_adopt_legacy_embedding_fingerprint, resolved_embed_model,
    short_hash_seq, sqlite_version, vec_version, vectors_table_exists,
};

/// Deterministic embedder — `salt` perturbs the hashed vector so a
/// same-model backend change produces a sample mismatch.
#[derive(Debug)]
struct HashEmbedder {
    model_id: String,
    dim: usize,
    salt: usize,
}

impl Embedder for HashEmbedder {
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
        Ok(texts
            .iter()
            .map(|t| hashed_vector(t, self.dim, self.salt))
            .collect())
    }
}

fn hashed_vector(text: &str, dim: usize, salt: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    for (i, b) in text.bytes().enumerate() {
        let slot = (i + salt) % dim;
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

/// `Qmd` indexing `tmp/docs` under collection `docs`, with `embedder`
/// injected (upstream's `getLlm` stub).
fn indexed_qmd(
    tmp: &Path,
    docs: &[(&str, &str)],
    embedder: Arc<dyn Embedder>,
) -> (Qmd, std::path::PathBuf) {
    let docs_dir = tmp.join("docs");
    std::fs::create_dir_all(&docs_dir).unwrap();
    for (rel, body) in docs {
        let path = docs_dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }
    let db = tmp.join("i.sqlite");
    let qmd = Qmd::builder(&db)
        .environment(Environment::default())
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

fn embedder(salt: usize) -> Arc<HashEmbedder> {
    Arc::new(HashEmbedder {
        model_id: "m-test".to_owned(),
        dim: 64,
        salt,
    })
}

fn run_embed(qmd: &Qmd) {
    qmd.embed(&EmbedOptions::default(), &mut |_| {}).unwrap();
}

// ── Version / count / table probes (cli/qmd.ts:4218-4234) ──

#[test]
fn probes_report_versions_and_counts() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = indexed_qmd(
        tmp.path(),
        &[("a.md", "# Alpha\nalpha body\n")],
        embedder(0),
    );

    let sqlite = sqlite_version(&qmd).unwrap();
    assert!(sqlite.chars().next().is_some_and(|c| c.is_ascii_digit()));
    let vec = vec_version(&qmd).unwrap();
    assert!(!vec.is_empty());
    assert_eq!(active_doc_count(&qmd).unwrap(), 1);
    // `vectors_vec` is created lazily on first embed.
    assert!(!vectors_table_exists(&qmd).unwrap());
    assert!(fingerprint_groups(&qmd).unwrap().is_empty());
    assert_eq!(resolved_embed_model(&qmd).unwrap(), "m-test");
    assert!(!current_embed_fingerprint(&qmd).unwrap().is_empty());
}

// ── checkEmbeddingVectorSamples (cli/qmd.ts:4001-4066) ──

#[test]
fn samples_report_no_docs_on_empty_index() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = indexed_qmd(tmp.path(), &[], embedder(0));
    let report = check_embedding_vector_samples(&qmd, embedder(0).as_ref(), 3).unwrap();
    assert!(report.ok);
    assert_eq!(report.details, "no active documents indexed");
}

#[test]
fn samples_report_missing_vec_table() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = indexed_qmd(
        tmp.path(),
        &[("a.md", "# Alpha\nalpha body\n")],
        embedder(0),
    );
    let report = check_embedding_vector_samples(&qmd, embedder(0).as_ref(), 3).unwrap();
    assert!(!report.ok);
    assert_eq!(
        report.details,
        "no vector table to test; please run qmd embed again"
    );
}

#[test]
fn samples_reproduce_stored_vectors() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = indexed_qmd(
        tmp.path(),
        &[
            ("a.md", "# Alpha\nalpha body\n"),
            ("b.md", "# Beta\nbeta body\n"),
        ],
        embedder(0),
    );
    run_embed(&qmd);
    assert!(vectors_table_exists(&qmd).unwrap());

    let report = check_embedding_vector_samples(&qmd, embedder(0).as_ref(), 3).unwrap();
    assert!(report.ok, "{}", report.details);
    assert_eq!(report.details, "2 sampled chunks reproduce stored vectors");

    let groups = fingerprint_groups(&qmd).unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].model, "m-test");
    assert_eq!(
        groups[0].fingerprint,
        current_embed_fingerprint(&qmd).unwrap()
    );
    assert_eq!(groups[0].docs, 2);
    assert_eq!(groups[0].chunks, 2);
}

#[test]
fn samples_detect_vector_mismatch() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = indexed_qmd(
        tmp.path(),
        &[
            ("a.md", "# Alpha\nalpha body\n"),
            ("b.md", "# Beta\nbeta body\n"),
        ],
        embedder(0),
    );
    run_embed(&qmd);

    // Same model/fingerprint, different vectors — the stored rows are
    // still "current" but no longer reproduce.
    let report = check_embedding_vector_samples(&qmd, embedder(7).as_ref(), 3).unwrap();
    assert!(!report.ok);
    assert!(
        report
            .details
            .contains("2/2 sampled chunks differ from stored vectors"),
        "{}",
        report.details
    );
    assert!(
        report.details.contains("Rebuild with `qmd embed --force`"),
        "{}",
        report.details
    );
}

// ── maybeAdoptLegacyEmbeddingFingerprint (store.ts:2581-2650) ──

fn clear_fingerprints(db: &Path) -> usize {
    rusqlite::Connection::open(db)
        .unwrap()
        .execute("UPDATE content_vectors SET embed_fingerprint = ''", [])
        .unwrap()
}

#[test]
fn adoption_skips_when_no_legacy_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = indexed_qmd(
        tmp.path(),
        &[("a.md", "# Alpha\nalpha body\n")],
        embedder(0),
    );
    run_embed(&qmd);
    let outcome = maybe_adopt_legacy_embedding_fingerprint(&qmd, embedder(0).as_ref()).unwrap();
    assert!(!outcome.checked);
    assert_eq!(outcome.adopted, 0);
    assert_eq!(outcome.reason, "no legacy empty-fingerprint embeddings");
}

#[test]
fn adoption_stamps_matching_legacy_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, db) = indexed_qmd(
        tmp.path(),
        &[
            ("a.md", "# Alpha\nalpha body\n"),
            ("b.md", "# Beta\nbeta body\n"),
        ],
        embedder(0),
    );
    run_embed(&qmd);
    assert_eq!(clear_fingerprints(&db), 2);

    let outcome = maybe_adopt_legacy_embedding_fingerprint(&qmd, embedder(0).as_ref()).unwrap();
    assert!(outcome.checked);
    assert_eq!(outcome.adopted, 2);
    assert!(
        outcome
            .reason
            .contains("matched current fingerprint at distance"),
        "{}",
        outcome.reason
    );

    // Rows moved back to the current fingerprint.
    let groups = fingerprint_groups(&qmd).unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(
        groups[0].fingerprint,
        current_embed_fingerprint(&qmd).unwrap()
    );
}

#[test]
fn adoption_refuses_mismatched_vectors() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, db) = indexed_qmd(
        tmp.path(),
        &[("a.md", "# Alpha\nalpha body\n")],
        embedder(0),
    );
    run_embed(&qmd);
    clear_fingerprints(&db);

    let outcome = maybe_adopt_legacy_embedding_fingerprint(&qmd, embedder(7).as_ref()).unwrap();
    assert!(outcome.checked);
    assert_eq!(outcome.adopted, 0);
    assert!(
        outcome.reason.contains("differs from current fingerprint"),
        "{}",
        outcome.reason
    );

    // Legacy rows stay untouched.
    let groups = fingerprint_groups(&qmd).unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].fingerprint, "");
}

// ── shortHashSeq (cli/qmd.ts:3770-3774) ──

#[test]
fn short_hash_seq_matches_upstream() {
    // With a `_`: first 12 hash chars + "_" + everything after the last
    // "_". JS `slice(0, 12)` clamps at the end, so a hash shorter than
    // 12 chars keeps its own `_` before the appended seq — the port
    // preserves that quirk.
    assert_eq!(
        short_hash_seq("a1b2c3d4e5f6a7b8c9d0e1f2_3"),
        "a1b2c3d4e5f6_3"
    );
    assert_eq!(short_hash_seq("short_0"), "short_0_0");
    // No `_`: 18-unit cap adds "...".
    assert_eq!(
        short_hash_seq("0123456789abcdefghijklm"),
        "0123456789abcdefgh..."
    );
    assert_eq!(short_hash_seq("0123456789abcdefgh"), "0123456789abcdefgh");
}
