//! Integration tests for index status/health and maintenance (P1.4):
//! `Qmd::status`, `index_health`, and the `Maintenance` cleanup
//! operations with their upstream ordering.
//!
//! Ported upstream coverage: `test/store.test.ts` Index Status group,
//! `test/store.helpers.unit.test.ts` `countOrphaned*`/`cleanupOrphaned*`,
//! and `test/cleanup.test.ts` ordering/consistency cases.

#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::shadow_unrelated,
    clippy::field_reassign_with_default,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_assert_message,
    clippy::panic,
    reason = "idiomatic test-code patterns"
)]

use std::fs;
use std::path::Path;

use qmd::{CollectionSpec, Environment, LexOptions, Qmd, UpdateOptions};

fn write(dir: &Path, rel: &str, content: &str) {
    let path = dir.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
}

fn raw_conn(path: &Path) -> rusqlite::Connection {
    rusqlite::Connection::open(path).expect("open raw connection")
}

fn indexed_db(tmp: &Path) -> (Qmd, std::path::PathBuf) {
    let docs = tmp.join("docs");
    write(&docs, "a.md", "# Alpha\nalpha body\n");
    write(&docs, "b.md", "# Beta\nbeta body\n");
    let db = tmp.join("i.sqlite");
    let qmd = Qmd::builder(&db)
        .environment(Environment::default())
        .build()
        .unwrap();
    qmd.add_collection(
        "docs",
        &CollectionSpec {
            path: docs.to_string_lossy().into_owned(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();
    qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();
    (qmd, db)
}

// --- status -------------------------------------------------------------------

#[test]
fn status_fresh_index() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("i.sqlite");
    let qmd = Qmd::builder(&db)
        .environment(Environment::default())
        .build()
        .unwrap();
    let s = qmd.status().unwrap();
    assert_eq!(s.total_documents, 0);
    assert_eq!(s.needs_embedding, 0);
    assert!(!s.has_vector_index);
    assert_eq!(s.vector_count, 0);
    assert_eq!(s.orphaned_vectors, 0);
    assert_eq!(s.latest_modified, None);
    assert!(!s.reindex_required);
    assert!(s.collections.is_empty());
}

#[test]
fn status_after_update() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = indexed_db(tmp.path());
    let docs_dir = tmp.path().join("docs");
    let s = qmd.status().unwrap();
    assert_eq!(s.total_documents, 2);
    // Nothing embedded yet — every active hash needs vectors.
    assert_eq!(s.needs_embedding, 2);
    assert!(!s.has_vector_index);
    assert_eq!(s.orphaned_vectors, 0);
    assert!(s.latest_modified.is_some());
    assert_eq!(s.collections.len(), 1);
    let c = &s.collections[0];
    assert_eq!(c.name, "docs");
    assert_eq!(c.path.as_deref(), docs_dir.to_str());
    assert_eq!(c.documents, 2);
    assert!(!c.last_updated.is_empty());
}

#[test]
fn status_counts_orphaned_vectors() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, db) = indexed_db(tmp.path());
    raw_conn(&db)
        .execute(
            "INSERT INTO content_vectors \
             (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at) \
             VALUES ('deadbeef', 0, 0, 'm', 'fp', 1, 't')",
            [],
        )
        .unwrap();
    let s = qmd.status().unwrap();
    assert_eq!(s.vector_count, 1);
    assert_eq!(s.orphaned_vectors, 1);
    assert_eq!(s.total_documents, 2);
}

// --- index_health --------------------------------------------------------------

#[test]
fn index_health_reports_pending_embeddings() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _db) = indexed_db(tmp.path());
    let h = qmd.index_health().unwrap();
    assert_eq!(h.total_docs, 2);
    assert_eq!(h.needs_embedding, 2);
    assert_eq!(h.days_stale, Some(0));
}

#[test]
fn index_health_future_modified_at_floors_to_negative_days() {
    // Upstream `Math.floor((now - last) / 86400000)` gives -1 for a
    // sub-day future `modified_at`; `div_euclid` keeps that floor where
    // `whole_days()` would truncate to 0.
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, db) = indexed_db(tmp.path());
    let future = (time::OffsetDateTime::now_utc() + time::Duration::hours(6))
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap();
    raw_conn(&db)
        .execute(
            "UPDATE documents SET modified_at = ? WHERE active = 1",
            [&future],
        )
        .unwrap();

    assert_eq!(qmd.index_health().unwrap().days_stale, Some(-1));
}

#[test]
fn index_health_empty_index() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("i.sqlite");
    let qmd = Qmd::builder(&db)
        .environment(Environment::default())
        .build()
        .unwrap();
    let h = qmd.index_health().unwrap();
    assert_eq!(h.total_docs, 0);
    assert_eq!(h.needs_embedding, 0);
    assert_eq!(h.days_stale, None);
}

// --- maintenance -----------------------------------------------------------------

#[test]
fn cleanup_orphaned_content() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, db) = indexed_db(tmp.path());
    raw_conn(&db)
        .execute(
            "INSERT INTO content (hash, doc, created_at) \
             VALUES ('orphan', 'x', 't')",
            [],
        )
        .unwrap();
    assert_eq!(qmd.maintenance().cleanup_orphaned_content().unwrap(), 1);
    assert_eq!(qmd.maintenance().cleanup_orphaned_content().unwrap(), 0);
}

#[test]
fn cleanup_orphaned_vectors_without_vec_table_is_noop() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, db) = indexed_db(tmp.path());
    raw_conn(&db)
        .execute(
            "INSERT INTO content_vectors \
             (hash, seq, pos, model, embedded_at) \
             VALUES ('deadbeef', 0, 0, 'm', 't')",
            [],
        )
        .unwrap();
    // `vectors_vec` does not exist yet — upstream's probe fails and the
    // call reports 0 without touching `content_vectors`.
    assert_eq!(qmd.maintenance().cleanup_orphaned_vectors().unwrap(), 0);
    assert_eq!(qmd.status().unwrap().orphaned_vectors, 1);
}

#[test]
fn cleanup_orphaned_vectors_removes_both_tables() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, db) = indexed_db(tmp.path());
    let conn = raw_conn(&db);
    conn.execute_batch(
        "CREATE VIRTUAL TABLE vectors_vec USING vec0(\
           hash_seq TEXT PRIMARY KEY, embedding float[2] distance_metric=cosine);\
         INSERT INTO content_vectors \
           (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at) \
           VALUES ('deadbeef', 0, 0, 'm', 'fp', 1, 't');\
         INSERT INTO vectors_vec (hash_seq, embedding) \
           VALUES ('deadbeef_0', '[0.0, 1.0]');",
    )
    .unwrap();
    assert_eq!(qmd.maintenance().cleanup_orphaned_vectors().unwrap(), 1);
    let n: i64 = conn
        .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 0);
    let n: i64 = conn
        .query_row("SELECT COUNT(*) FROM vectors_vec", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 0);
}

#[test]
fn delete_inactive_documents() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, db) = indexed_db(tmp.path());
    // Delete the file and re-index: the row becomes an inactive tombstone.
    fs::remove_file(tmp.path().join("docs/b.md")).unwrap();
    qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();
    let inactive: i64 = raw_conn(&db)
        .query_row("SELECT COUNT(*) FROM documents WHERE active = 0", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(inactive, 1);
    assert_eq!(qmd.maintenance().delete_inactive_documents().unwrap(), 1);
    let inactive: i64 = raw_conn(&db)
        .query_row("SELECT COUNT(*) FROM documents WHERE active = 0", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(inactive, 0);
    // The active document is untouched.
    assert_eq!(qmd.status().unwrap().total_documents, 1);
}

#[test]
fn clear_llm_cache_and_embeddings() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, db) = indexed_db(tmp.path());
    let conn = raw_conn(&db);
    conn.execute(
        "INSERT INTO llm_cache (hash, result, created_at) VALUES ('h', 'r', 't')",
        [],
    )
    .unwrap();
    assert_eq!(qmd.maintenance().clear_llm_cache().unwrap(), 1);
    let n: i64 = conn
        .query_row("SELECT COUNT(*) FROM llm_cache", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 0);

    conn.execute(
        "INSERT INTO content_vectors \
         (hash, seq, pos, model, embedded_at) VALUES ('x', 0, 0, 'm', 't')",
        [],
    )
    .unwrap();
    qmd.maintenance().clear_embeddings().unwrap();
    let n: i64 = conn
        .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 0);
    // `vectors_vec` was dropped if it existed.
    assert!(!qmd.status().unwrap().has_vector_index);
}

#[test]
fn preview_reports_without_writing() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, db) = indexed_db(tmp.path());
    fs::remove_file(tmp.path().join("docs/b.md")).unwrap();
    // `update` clears `llm_cache` first (upstream parity), so seed the
    // cache row after re-indexing.
    qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();
    let conn = raw_conn(&db);
    conn.execute(
        "INSERT INTO llm_cache (hash, result, created_at) VALUES ('h', 'r', 't')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO content (hash, doc, created_at) VALUES ('orphan', 'x', 't')",
        [],
    )
    .unwrap();

    let counts = qmd.maintenance().preview().unwrap();
    assert_eq!(counts.cache_count, 1);
    assert_eq!(counts.inactive_docs, 1);
    // `b.md`'s content is orphaned only once the tombstone is deleted;
    // upstream counts hashes with no active document.
    assert_eq!(counts.orphaned_content, 2); // 'orphan' + b.md's hash
    // Preview wrote nothing.
    let n: i64 = conn
        .query_row("SELECT COUNT(*) FROM llm_cache", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 1);
}

#[test]
fn maintenance_run_leaves_consistent_index() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, db) = indexed_db(tmp.path());
    fs::remove_file(tmp.path().join("docs/b.md")).unwrap();
    // `update` clears `llm_cache` first; seed it after re-indexing.
    qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();
    let conn = raw_conn(&db);
    conn.execute_batch(
        "INSERT INTO llm_cache (hash, result, created_at) VALUES ('h', 'r', 't');\
         INSERT INTO content (hash, doc, created_at) VALUES ('orphan', 'x', 't');\
         INSERT INTO content_vectors \
           (hash, seq, pos, model, embedded_at) VALUES ('deadbeef', 0, 0, 'm', 't');",
    )
    .unwrap();

    let counts = qmd.maintenance().run().unwrap();
    assert_eq!(counts.cache_count, 1);
    assert_eq!(counts.inactive_docs, 1);
    // vectors_vec is absent, so vector cleanup reports 0 as upstream.
    assert_eq!(counts.orphaned_vectors, 0);
    // Tombstone deleted first, so b.md's content became collectible.
    assert_eq!(counts.orphaned_content, 2);

    let s = qmd.status().unwrap();
    assert_eq!(s.total_documents, 1);
    // Upstream parity: without `vectors_vec`, the orphaned
    // `content_vectors` row survives cleanup and keeps reporting.
    assert_eq!(s.orphaned_vectors, 1);

    // FTS still answers after optimize + VACUUM.
    let hits = qmd
        .search_lex(
            "alpha",
            &LexOptions {
                limit: Some(5),
                collections: None,
            },
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].filepath, "qmd://docs/a.md");
}
