//! Metadata on search results and metadata-filtered retrieval through
//! the public `Qmd` API — ports of upstream `metadata-search.test.ts`
//! (FTS/vec filter composition, extraction gate, shared-hash re-filter,
//! structured metadata) and the SDK half of `metadata-surfaces.test.ts`
//! (the MCP/HTTP half is P5 scope).
//!
//! Upstream drives `searchFTS`/`searchVec`/`structuredSearch` internals;
//! here the same paths run through `Qmd::search_lex`/`search_vector`/
//! `search` so stored metadata arrives via the real extraction pipeline.

#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_assert_message,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::shadow_unrelated,
    reason = "idiomatic test-code patterns"
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use qmd::{
    CollectionSpec, Embedder, Environment, ExpandedQuery, HybridResult, InferenceError, LexOptions,
    MetadataValue, Qmd, QueryKind, SearchInput, SearchOptions, UpdateOptions, VectorOptions,
    parse_metadata_filter,
};

fn write(dir: &Path, rel: &str, content: &str) {
    let path = dir.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}

fn doc(yaml: &str) -> String {
    format!("---\nqmd:\n  metadata:\n{yaml}---\nqmdtesttoken body\n")
}

fn no_metadata_doc() -> String {
    "qmdtesttoken body\n".to_owned()
}

fn indexed_store(tmp: &Path, docs: &[(&str, String)]) -> (Qmd, PathBuf) {
    indexed_store_multi(tmp, &[("docs", docs)], None)
}

fn indexed_store_with(
    tmp: &Path,
    docs: &[(&str, String)],
    embedder: Option<Arc<dyn Embedder>>,
) -> (Qmd, PathBuf) {
    indexed_store_multi(tmp, &[("docs", docs)], embedder)
}

/// `collections` maps each name to its `(relative path, body)` files.
fn indexed_store_multi(
    tmp: &Path,
    collections: &[(&str, &[(&str, String)])],
    embedder: Option<Arc<dyn Embedder>>,
) -> (Qmd, PathBuf) {
    let db_path = tmp.join("i.sqlite");
    let mut builder = Qmd::builder(&db_path).environment(Environment::default());
    if let Some(e) = embedder {
        builder = builder.embedder(e);
    }
    let qmd = builder.build().unwrap();
    for (name, docs) in collections {
        let dir = tmp.join(name);
        for (rel, content) in *docs {
            write(&dir, rel, content);
        }
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
    (qmd, db_path)
}

fn raw_conn(path: &Path) -> rusqlite::Connection {
    rusqlite::Connection::open(path).expect("open raw connection")
}

fn filter(json: &str) -> qmd::MetadataFilter {
    parse_metadata_filter(&serde_json::from_str(json).expect("json literal parses"))
        .expect("filter parses")
}

const fn lex_opts(f: Option<qmd::MetadataFilter>) -> LexOptions {
    LexOptions {
        limit: Some(20),
        collections: None,
        filter: f,
    }
}

fn filepaths(results: &[HybridResult]) -> Vec<String> {
    results.iter().map(|r| r.filepath.clone()).collect()
}

#[test]
fn filtered_and_unfiltered_lex_results_carry_metadata() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = indexed_store(
        tmp.path(),
        &[
            ("a.md", doc("    status: published\n    priority: 5\n")),
            ("b.md", doc("    status: draft\n")),
            ("c.md", no_metadata_doc()),
        ],
    );

    // Unfiltered: all docs, each carrying its canonical metadata.
    let all = qmd.search_lex("qmdtesttoken", &lex_opts(None)).unwrap();
    assert_eq!(all.len(), 3);
    let a = all
        .iter()
        .find(|r| r.filepath == "qmd://docs/a.md")
        .unwrap();
    assert_eq!(
        a.metadata.get("status"),
        Some(&MetadataValue::String("published".to_owned()))
    );
    let c = all
        .iter()
        .find(|r| r.filepath == "qmd://docs/c.md")
        .unwrap();
    assert!(c.metadata.is_empty());

    // Filtered: only matching docs.
    let published = qmd
        .search_lex(
            "qmdtesttoken",
            &lex_opts(Some(filter(
                r#"{"key": "status", "operator": "eq", "value": "published"}"#,
            ))),
        )
        .unwrap();
    assert_eq!(
        published
            .iter()
            .map(|r| r.filepath.as_str())
            .collect::<Vec<_>>(),
        ["qmd://docs/a.md"]
    );
    assert_eq!(
        published[0].metadata.get("priority"),
        Some(&MetadataValue::Number(5.0))
    );

    // `exists: false` — c.md (extracted, no `status` key) only.
    let no_status = qmd
        .search_lex(
            "qmdtesttoken",
            &lex_opts(Some(filter(
                r#"{"key": "status", "operator": "exists", "value": false}"#,
            ))),
        )
        .unwrap();
    assert_eq!(
        no_status
            .iter()
            .map(|r| r.filepath.as_str())
            .collect::<Vec<_>>(),
        ["qmd://docs/c.md"]
    );
}

#[test]
fn filtered_lex_requires_current_error_free_extraction() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, db) = indexed_store(
        tmp.path(),
        &[
            ("good.md", doc("    status: published\n")),
            ("stale.md", doc("    status: published\n")),
            ("errored.md", doc("    status: published\n")),
            ("unextracted.md", doc("    status: published\n")),
        ],
    );
    let conn = raw_conn(&db);
    // Stale extraction_version.
    conn.execute(
        "UPDATE document_metadata SET extraction_version = 0 \
         WHERE document_id = (SELECT id FROM documents WHERE path = 'stale.md')",
        [],
    )
    .unwrap();
    // Extraction error.
    conn.execute(
        "UPDATE document_metadata SET extraction_error = 'boom' \
         WHERE document_id = (SELECT id FROM documents WHERE path = 'errored.md')",
        [],
    )
    .unwrap();
    // Row deleted entirely — "not yet extracted".
    conn.execute(
        "DELETE FROM document_metadata \
         WHERE document_id = (SELECT id FROM documents WHERE path = 'unextracted.md')",
        [],
    )
    .unwrap();
    drop(conn);

    let status_eq = Some(filter(
        r#"{"key": "status", "operator": "eq", "value": "published"}"#,
    ));
    let filtered = qmd
        .search_lex("qmdtesttoken", &lex_opts(status_eq))
        .unwrap();
    assert_eq!(
        filtered
            .iter()
            .map(|r| r.filepath.as_str())
            .collect::<Vec<_>>(),
        ["qmd://docs/good.md"]
    );

    // The gate also protects `exists: false`: stale, errored and
    // unextracted documents must not match an absence predicate
    // (upstream store.ts:4135-4141) — only the current, error-free
    // extraction row qualifies.
    let absent = qmd
        .search_lex(
            "qmdtesttoken",
            &lex_opts(Some(filter(
                r#"{"key": "nosuchkey", "operator": "exists", "value": false}"#,
            ))),
        )
        .unwrap();
    assert_eq!(
        absent
            .iter()
            .map(|r| r.filepath.as_str())
            .collect::<Vec<_>>(),
        ["qmd://docs/good.md"]
    );

    // Unfiltered search still returns everything.
    let all = qmd.search_lex("qmdtesttoken", &lex_opts(None)).unwrap();
    assert_eq!(all.len(), 4);
}

/// Upstream `metadata-search.test.ts` "composes multi-collection scope
/// with one metadata filter".
#[test]
fn filtered_lex_composes_multi_collection_scope() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = indexed_store_multi(
        tmp.path(),
        &[
            ("notes", &[("a.md", doc("    status: published\n"))][..]),
            (
                "docs",
                &[
                    ("b.md", doc("    status: published\n")),
                    ("c.md", doc("    status: draft\n")),
                ][..],
            ),
            ("other", &[("d.md", doc("    status: published\n"))][..]),
        ],
        None,
    );

    let mut opts = lex_opts(Some(filter(
        r#"{"key": "status", "operator": "eq", "value": "published"}"#,
    )));
    opts.collections = Some(vec!["notes".to_owned(), "docs".to_owned()]);
    let filtered = qmd.search_lex("qmdtesttoken", &opts).unwrap();
    let mut paths: Vec<String> = filtered.iter().map(|r| r.filepath.clone()).collect();
    paths.sort();
    assert_eq!(paths, ["qmd://docs/b.md", "qmd://notes/a.md"]);
}

/// Upstream `metadata-search.test.ts` "nested filters apply through FTS".
#[test]
fn nested_filter_applies_through_fts() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = indexed_store(
        tmp.path(),
        &[
            (
                "match.md",
                doc(
                    "    topics: [typescript, programming]\n    status: published\n    priority: 5\n",
                ),
            ),
            (
                "wrong-topic.md",
                doc("    topics: [cooking]\n    status: published\n    priority: 5\n"),
            ),
            (
                "low-priority.md",
                doc("    topics: [typescript, programming]\n    status: draft\n    priority: 1\n"),
            ),
        ],
    );

    let nested = filter(
        r#"{"operator": "and", "operands": [
            {"key": "topics", "operator": "all", "value": ["typescript", "programming"]},
            {"operator": "or", "operands": [
                {"key": "status", "operator": "eq", "value": "published"},
                {"key": "priority", "operator": "gte", "value": 3}
            ]}
        ]}"#,
    );
    let filtered = qmd
        .search_lex("qmdtesttoken", &lex_opts(Some(nested)))
        .unwrap();
    assert_eq!(
        filtered
            .iter()
            .map(|r| r.filepath.as_str())
            .collect::<Vec<_>>(),
        ["qmd://docs/match.md"]
    );
}

/// Upstream `metadata-search.test.ts` "selective filters may under-fill
/// but never return an ineligible row" — the `limit * 10` candidate
/// window (store.ts:4104) keeps a selective filter from returning zero
/// rows when the match exists past `limit` raw hits.
#[test]
fn selective_filter_under_fills_but_never_returns_ineligible() {
    let tmp = tempfile::tempdir().unwrap();
    let docs: Vec<(String, String)> = (0..20)
        .map(|i| {
            (
                format!("doc-{i}.md"),
                doc(if i == 0 {
                    "    status: published\n"
                } else {
                    "    status: draft\n"
                }),
            )
        })
        .collect();
    let doc_refs: Vec<(&str, String)> = docs.iter().map(|(p, b)| (p.as_str(), b.clone())).collect();
    let (qmd, _db) = indexed_store(tmp.path(), &doc_refs);

    let mut opts = lex_opts(Some(filter(
        r#"{"key": "status", "operator": "eq", "value": "published"}"#,
    )));
    opts.limit = Some(5);
    let filtered = qmd.search_lex("qmdtesttoken", &opts).unwrap();
    assert_eq!(
        filtered
            .iter()
            .map(|r| r.filepath.as_str())
            .collect::<Vec<_>>(),
        ["qmd://docs/doc-0.md"]
    );
}

#[test]
fn malformed_metadata_json_falls_back_to_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, db) = indexed_store(
        tmp.path(),
        &[
            ("a.md", doc("    status: published\n")),
            ("b.md", doc("    status: draft\n")),
        ],
    );
    let conn = raw_conn(&db);
    conn.execute(
        "UPDATE document_metadata SET metadata_json = 'not json' \
         WHERE document_id = (SELECT id FROM documents WHERE path = 'a.md')",
        [],
    )
    .unwrap();
    conn.execute(
        "UPDATE document_metadata SET metadata_json = '5' \
         WHERE document_id = (SELECT id FROM documents WHERE path = 'b.md')",
        [],
    )
    .unwrap();
    drop(conn);

    let all = qmd.search_lex("qmdtesttoken", &lex_opts(None)).unwrap();
    assert_eq!(all.len(), 2);
    assert!(all.iter().all(|r| r.metadata.is_empty()));
}

#[test]
fn structured_search_attaches_metadata_via_batch_query() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = indexed_store(
        tmp.path(),
        &[
            ("a.md", doc("    status: published\n")),
            ("b.md", doc("    status: draft\n")),
        ],
    );
    let mut opts = SearchOptions::default();
    opts.input = SearchInput::Expanded(vec![ExpandedQuery::new(
        QueryKind::Lex,
        "qmdtesttoken".to_owned(),
    )]);
    opts.rerank = false;
    let results = qmd.search(&opts, &()).unwrap();
    assert_eq!(filepaths(&results).len(), 2);
    let a = results
        .iter()
        .find(|r| r.filepath == "qmd://docs/a.md")
        .unwrap();
    assert_eq!(
        a.metadata.get("status"),
        Some(&MetadataValue::String("published".to_owned()))
    );
    // Upstream "unfiltered structured search attaches metadata to
    // every result" — a document without a metadata row gets `{}`.
    let b = results
        .iter()
        .find(|r| r.filepath == "qmd://docs/b.md")
        .unwrap();
    assert_eq!(
        b.metadata.get("status"),
        Some(&MetadataValue::String("draft".to_owned()))
    );

    // Same pipeline with a metadata filter — only matching docs reach
    // the fused list (filter applies at every retrieval call).
    let mut filtered_opts = SearchOptions::default();
    filtered_opts.input = SearchInput::Expanded(vec![ExpandedQuery::new(
        QueryKind::Lex,
        "qmdtesttoken".to_owned(),
    )]);
    filtered_opts.filter = Some(filter(
        r#"{"key": "status", "operator": "eq", "value": "draft"}"#,
    ));
    filtered_opts.rerank = false;
    let filtered = qmd.search(&filtered_opts, &()).unwrap();
    assert_eq!(filepaths(&filtered), ["qmd://docs/b.md"]);
}

// =============================================================================
// Vector path — stub embedder drives `qmd.embed` + `search_vector`.
// =============================================================================

#[derive(Debug)]
struct StubEmbedder {
    model_id: String,
    dimension: usize,
    calls: AtomicUsize,
}

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
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(texts
            .iter()
            .map(|t| hashed_vector(t, self.dimension))
            .collect())
    }
}

const fn vec_opts(f: Option<qmd::MetadataFilter>) -> VectorOptions {
    VectorOptions {
        limit: Some(20),
        collections: None,
        filter: f,
    }
}

#[test]
fn filtered_vector_search_and_metadata_on_results() {
    let tmp = tempfile::tempdir().unwrap();
    let embedder = Arc::new(StubEmbedder {
        model_id: "test-embed".to_owned(),
        dimension: 8,
        calls: AtomicUsize::new(0),
    });
    let (qmd, _db) = indexed_store_with(
        tmp.path(),
        &[
            ("a.md", doc("    status: published\n")),
            ("b.md", doc("    status: draft\n")),
            ("c.md", no_metadata_doc()),
        ],
        Some(embedder),
    );
    qmd.embed(&qmd::EmbedOptions::default(), &mut |_| {})
        .unwrap();

    let all = qmd.search_vector("qmdtesttoken", &vec_opts(None)).unwrap();
    assert_eq!(all.len(), 3);
    assert!(
        all.iter()
            .find(|r| r.filepath == "qmd://docs/a.md")
            .is_some_and(|r| r.metadata.get("status")
                == Some(&MetadataValue::String("published".to_owned())))
    );

    let published = qmd
        .search_vector(
            "qmdtesttoken",
            &vec_opts(Some(filter(
                r#"{"key": "status", "operator": "eq", "value": "published"}"#,
            ))),
        )
        .unwrap();
    assert_eq!(
        published
            .iter()
            .map(|r| r.filepath.as_str())
            .collect::<Vec<_>>(),
        ["qmd://docs/a.md"]
    );
}

/// Upstream `metadata-search.test.ts` "returns empty when no documents
/// are eligible" — the eligible-set query short-circuits before any
/// vector scan.
#[test]
fn vector_filter_with_no_eligible_documents_returns_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let embedder = Arc::new(StubEmbedder {
        model_id: "test-embed".to_owned(),
        dimension: 8,
        calls: AtomicUsize::new(0),
    });
    let (qmd, _db) = indexed_store_with(
        tmp.path(),
        &[("doc.md", doc("    status: draft\n"))],
        Some(embedder),
    );
    qmd.embed(&qmd::EmbedOptions::default(), &mut |_| {})
        .unwrap();

    let filtered = qmd
        .search_vector(
            "qmdtesttoken",
            &vec_opts(Some(filter(
                r#"{"key": "status", "operator": "eq", "value": "published"}"#,
            ))),
        )
        .unwrap();
    assert!(filtered.is_empty());
}

/// Upstream `metadata-search.test.ts` "composes collection scope with
/// the metadata filter" (searchVec).
#[test]
fn vector_filter_composes_collection_scope() {
    let tmp = tempfile::tempdir().unwrap();
    let embedder = Arc::new(StubEmbedder {
        model_id: "test-embed".to_owned(),
        dimension: 8,
        calls: AtomicUsize::new(0),
    });
    let (qmd, _db) = indexed_store_multi(
        tmp.path(),
        &[
            ("notes", &[("a.md", doc("    status: published\n"))][..]),
            ("docs", &[("b.md", doc("    status: published\n"))][..]),
        ],
        Some(embedder),
    );
    qmd.embed(&qmd::EmbedOptions::default(), &mut |_| {})
        .unwrap();

    let mut opts = vec_opts(Some(filter(
        r#"{"key": "status", "operator": "eq", "value": "published"}"#,
    )));
    opts.collections = Some(vec!["docs".to_owned()]);
    let filtered = qmd.search_vector("qmdtesttoken", &opts).unwrap();
    assert_eq!(
        filtered
            .iter()
            .map(|r| r.filepath.as_str())
            .collect::<Vec<_>>(),
        ["qmd://docs/b.md"]
    );
}

#[test]
fn vector_filter_reapplied_on_document_join_for_shared_hashes() {
    // Two byte-identical documents share one content hash but keep
    // separate document ids and metadata rows. Staling one row makes the
    // hash still eligible (via the good doc) while the stale doc must be
    // re-filtered at the document join (upstream store.ts:4339-4345).
    let tmp = tempfile::tempdir().unwrap();
    let body = doc("    status: published\n");
    let embedder = Arc::new(StubEmbedder {
        model_id: "test-embed".to_owned(),
        dimension: 8,
        calls: AtomicUsize::new(0),
    });
    let (qmd, db) = indexed_store_with(
        tmp.path(),
        &[("good.md", body.clone()), ("stale.md", body)],
        Some(embedder),
    );
    qmd.embed(&qmd::EmbedOptions::default(), &mut |_| {})
        .unwrap();

    let conn = raw_conn(&db);
    let hashes: Vec<String> = conn
        .prepare("SELECT DISTINCT hash FROM documents")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(hashes.len(), 1, "docs must share one content hash");
    conn.execute(
        "UPDATE document_metadata SET extraction_version = 0 \
         WHERE document_id = (SELECT id FROM documents WHERE path = 'stale.md')",
        [],
    )
    .unwrap();
    drop(conn);

    // Unfiltered: both documents surface off the shared hash.
    let all = qmd.search_vector("qmdtesttoken", &vec_opts(None)).unwrap();
    assert_eq!(all.len(), 2);

    // Filtered: the stale document is re-filtered out even though its
    // hash_seq was eligible through the good document.
    let filtered = qmd
        .search_vector(
            "qmdtesttoken",
            &vec_opts(Some(filter(
                r#"{"key": "status", "operator": "eq", "value": "published"}"#,
            ))),
        )
        .unwrap();
    assert_eq!(
        filtered
            .iter()
            .map(|r| r.filepath.as_str())
            .collect::<Vec<_>>(),
        ["qmd://docs/good.md"]
    );
}
