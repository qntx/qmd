//! Metadata persistence lifecycle tests — port of upstream
//! `test/metadata-store.test.ts`: schema wiring, replacement semantics,
//! reindex synchronization, and cleanup cascades.
//!
//! Upstream exercises `syncDocumentMetadata`/`replaceDocumentMetadata`
//! directly on the store handle; here the equivalent coverage goes
//! through the public `Qmd` API (`add_collection` + `update`, which run
//! the same per-document sync inside `reindex_collection`) plus direct
//! SQLite reads of `document_metadata` and `document_metadata_values`.

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

use std::fs;
use std::path::Path;

use qmd::{CollectionSpec, Environment, Qmd, UpdateOptions};

fn write(dir: &Path, rel: &str, content: &str) {
    let path = dir.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
}

fn open(db: &Path) -> Qmd {
    Qmd::builder(db)
        .environment(Environment::default())
        .build()
        .unwrap()
}

fn raw_conn(path: &Path) -> rusqlite::Connection {
    rusqlite::Connection::open(path).expect("open raw connection")
}

fn build_doc(frontmatter_yaml: Option<&str>, body: &str) -> String {
    frontmatter_yaml.map_or_else(
        || body.to_owned(),
        |yaml| format!("---\n{yaml}---\n\n{body}"),
    )
}

fn update(qmd: &Qmd) -> qmd::UpdateReport {
    qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap()
}

/// `qmd://<collection>/<path>` → `document_metadata` row
/// (`metadata_json`, `extraction_version`, `extraction_error`).
fn metadata_row(
    conn: &rusqlite::Connection,
    collection: &str,
    path: &str,
) -> Option<(String, i64, Option<String>)> {
    conn.query_row(
        "SELECT dm.metadata_json, dm.extraction_version, dm.extraction_error
         FROM document_metadata dm
         JOIN documents d ON d.id = dm.document_id
         WHERE d.collection = ? AND d.path = ?",
        rusqlite::params![collection, path],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )
    .ok()
}

fn count_value_rows(conn: &rusqlite::Connection, collection: &str, path: &str) -> i64 {
    conn.query_row(
        "SELECT COUNT(*) FROM document_metadata_values v
         JOIN documents d ON d.id = v.document_id
         WHERE d.collection = ? AND d.path = ?",
        rusqlite::params![collection, path],
        |r| r.get(0),
    )
    .unwrap()
}

fn count_table(conn: &rusqlite::Connection, table: &str) -> i64 {
    conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
        .unwrap()
}

fn document_id(conn: &rusqlite::Connection, collection: &str, path: &str) -> i64 {
    conn.query_row(
        "SELECT id FROM documents WHERE collection = ? AND path = ?",
        rusqlite::params![collection, path],
        |r| r.get(0),
    )
    .unwrap()
}

fn pending_metadata(conn: &rusqlite::Connection) -> i64 {
    conn.query_row(
        "SELECT COUNT(*) FROM documents d
         WHERE d.active = 1
           AND NOT EXISTS (
             SELECT 1 FROM document_metadata dm
             WHERE dm.document_id = d.id
               AND dm.extraction_version = ?
               AND dm.extraction_error IS NULL
           )",
        rusqlite::params![qmd::METADATA_EXTRACTION_VERSION],
        |r| r.get(0),
    )
    .unwrap()
}

// =============================================================================
// syncDocumentMetadata (via update → reindex_collection sync)
// =============================================================================

#[test]
fn persists_metadata_and_value_rows_for_a_new_document() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    let db = tmp.path().join("i.sqlite");
    write(
        &docs,
        "doc.md",
        &build_doc(
            Some("qmd:\n  metadata:\n    topics: [a, b]\n    priority: 3\n"),
            "# Doc\n",
        ),
    );
    let qmd = open(&db);
    qmd.add_collection(
        "notes",
        &CollectionSpec {
            path: docs.to_string_lossy().into_owned(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();
    update(&qmd);

    let conn = raw_conn(&db);
    let (json, version, error) = metadata_row(&conn, "notes", "doc.md").unwrap();
    let json: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(
        json,
        serde_json::json!({"topics": ["a", "b"], "priority": 3})
    );
    assert_eq!(version, qmd::METADATA_EXTRACTION_VERSION);
    assert!(error.is_none());
    assert_eq!(count_value_rows(&conn, "notes", "doc.md"), 3);
}

#[test]
fn records_extraction_state_for_documents_without_metadata() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    let db = tmp.path().join("i.sqlite");
    write(&docs, "plain.md", "# Plain doc\n");
    let qmd = open(&db);
    qmd.add_collection(
        "notes",
        &CollectionSpec {
            path: docs.to_string_lossy().into_owned(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();
    update(&qmd);

    let conn = raw_conn(&db);
    let (json, _, error) = metadata_row(&conn, "notes", "plain.md").unwrap();
    assert_eq!(json, "{}");
    assert!(error.is_none());
    assert_eq!(count_value_rows(&conn, "notes", "plain.md"), 0);
}

#[test]
fn only_if_stale_skips_documents_with_a_current_extraction_row() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    let db = tmp.path().join("i.sqlite");
    write(
        &docs,
        "doc.md",
        &build_doc(Some("qmd:\n  metadata:\n    status: ok\n"), "# Doc\n"),
    );
    let qmd = open(&db);
    qmd.add_collection(
        "notes",
        &CollectionSpec {
            path: docs.to_string_lossy().into_owned(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();
    update(&qmd);

    // Upstream's staleness check compares `extraction_version` only, so
    // a current-version row is skipped even if we corrupt it — the
    // second update must not rewrite the row.
    let conn = raw_conn(&db);
    conn.execute(
        "UPDATE document_metadata SET extraction_error = 'boom'
         WHERE document_id = (SELECT id FROM documents WHERE path = 'doc.md')",
        [],
    )
    .unwrap();
    let report = update(&qmd);
    assert_eq!(report.unchanged, 1);
    let (_, _, error) = metadata_row(&conn, "notes", "doc.md").unwrap();
    assert_eq!(error.as_deref(), Some("boom"));
    // …but a missing row is backfilled on the next pass.
    conn.execute("DELETE FROM document_metadata", []).unwrap();
    update(&qmd);
    let (json, _, error) = metadata_row(&conn, "notes", "doc.md").unwrap();
    assert_eq!(json, r#"{"status":"ok"}"#);
    assert!(error.is_none());
}

#[test]
fn invalid_metadata_replaces_prior_rows_with_empty_metadata_and_an_error() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    let db = tmp.path().join("i.sqlite");
    let file = docs.join("doc.md");
    write(
        &docs,
        "doc.md",
        &build_doc(Some("qmd:\n  metadata:\n    status: ok\n"), "# Doc\n"),
    );
    let qmd = open(&db);
    qmd.add_collection(
        "notes",
        &CollectionSpec {
            path: docs.to_string_lossy().into_owned(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();
    update(&qmd);

    let conn = raw_conn(&db);
    assert_eq!(count_value_rows(&conn, "notes", "doc.md"), 1);

    fs::write(
        &file,
        build_doc(Some("qmd:\n  metadata:\n    status: null\n"), "# Doc\n"),
    )
    .unwrap();
    let report = update(&qmd);
    assert_eq!(report.metadata_errors, 1);

    let (json, _, error) = metadata_row(&conn, "notes", "doc.md").unwrap();
    assert_eq!(json, "{}");
    assert!(error.unwrap().contains("null is not supported"));
    assert_eq!(count_value_rows(&conn, "notes", "doc.md"), 0);
}

#[test]
fn metadata_removal_clears_value_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    let db = tmp.path().join("i.sqlite");
    let file = docs.join("doc.md");
    write(
        &docs,
        "doc.md",
        &build_doc(Some("qmd:\n  metadata:\n    status: ok\n"), "# Doc\n"),
    );
    let qmd = open(&db);
    qmd.add_collection(
        "notes",
        &CollectionSpec {
            path: docs.to_string_lossy().into_owned(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();
    update(&qmd);

    let conn = raw_conn(&db);
    assert_eq!(count_value_rows(&conn, "notes", "doc.md"), 1);

    fs::write(&file, "# Doc\n").unwrap();
    update(&qmd);
    assert_eq!(count_value_rows(&conn, "notes", "doc.md"), 0);
    let (_, _, error) = metadata_row(&conn, "notes", "doc.md").unwrap();
    assert!(error.is_none());
}

#[test]
fn two_paths_sharing_one_content_hash_keep_separate_metadata_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    let db = tmp.path().join("i.sqlite");
    // Same bytes in two collections → one `content` row, two documents.
    let content = build_doc(
        Some("qmd:\n  metadata:\n    status: shared\n"),
        "# Same content\n",
    );
    write(&docs, "first.md", &content);
    let docs2 = tmp.path().join("docs2");
    write(&docs2, "second.md", &content);
    let qmd = open(&db);
    for (name, dir) in [("notes", &docs), ("other", &docs2)] {
        qmd.add_collection(
            name,
            &CollectionSpec {
                path: dir.to_string_lossy().into_owned(),
                ..CollectionSpec::default()
            },
        )
        .unwrap();
    }
    update(&qmd);

    let conn = raw_conn(&db);
    let (first, _, _) = metadata_row(&conn, "notes", "first.md").unwrap();
    let (second, _, _) = metadata_row(&conn, "other", "second.md").unwrap();
    assert_eq!(first, r#"{"status":"shared"}"#);
    assert_eq!(second, r#"{"status":"shared"}"#);
    assert_ne!(
        document_id(&conn, "notes", "first.md"),
        document_id(&conn, "other", "second.md")
    );

    // Editing only the first file rewrites its row alone — metadata is
    // per document identity, not per content hash.
    fs::write(
        docs.join("first.md"),
        build_doc(
            Some("qmd:\n  metadata:\n    status: overridden\n"),
            "# Same content\n",
        ),
    )
    .unwrap();
    update(&qmd);
    let (first, _, _) = metadata_row(&conn, "notes", "first.md").unwrap();
    let (second, _, _) = metadata_row(&conn, "other", "second.md").unwrap();
    assert_eq!(first, r#"{"status":"overridden"}"#);
    assert_eq!(second, r#"{"status":"shared"}"#);
}

// =============================================================================
// countDocumentsPendingMetadata
// =============================================================================

#[test]
fn counts_unextracted_stale_and_errored_documents() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    let db = tmp.path().join("i.sqlite");
    for (file, body) in [
        ("extracted.md", "# A\n"),
        ("pending.md", "# B\n"),
        ("errored.md", "# C\n"),
        ("stale.md", "# D\n"),
    ] {
        write(&docs, file, body);
    }
    let qmd = open(&db);
    qmd.add_collection(
        "notes",
        &CollectionSpec {
            path: docs.to_string_lossy().into_owned(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();
    update(&qmd);

    let conn = raw_conn(&db);
    // pending.md: no extraction row. errored.md: error at current
    // version. stale.md: older extraction version.
    conn.execute(
        "DELETE FROM document_metadata WHERE document_id = ?",
        [document_id(&conn, "notes", "pending.md")],
    )
    .unwrap();
    conn.execute(
        "UPDATE document_metadata SET extraction_error = 'boom' WHERE document_id = ?",
        [document_id(&conn, "notes", "errored.md")],
    )
    .unwrap();
    conn.execute(
        "UPDATE document_metadata SET extraction_version = ? WHERE document_id = ?",
        rusqlite::params![
            qmd::METADATA_EXTRACTION_VERSION - 1,
            document_id(&conn, "notes", "stale.md")
        ],
    )
    .unwrap();

    assert_eq!(pending_metadata(&conn), 3);
    assert_eq!(qmd.status().unwrap().pending_metadata, 3);

    // Deactivated documents leave the pending count.
    conn.execute(
        "UPDATE documents SET active = 0 WHERE id = ?",
        [document_id(&conn, "notes", "pending.md")],
    )
    .unwrap();
    assert_eq!(pending_metadata(&conn), 2);
}

// =============================================================================
// reindexCollection metadata synchronization (via update)
// =============================================================================

#[test]
fn extracts_metadata_for_new_changed_and_unchanged_documents() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    let db = tmp.path().join("i.sqlite");
    let file = docs.join("doc.md");
    write(
        &docs,
        "doc.md",
        &build_doc(Some("qmd:\n  metadata:\n    status: draft\n"), "# Doc\n"),
    );
    let qmd = open(&db);
    qmd.add_collection(
        "notes",
        &CollectionSpec {
            path: docs.to_string_lossy().into_owned(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();

    let first = update(&qmd);
    assert_eq!(first.indexed, 1);
    assert_eq!(first.metadata_errors, 0);
    let conn = raw_conn(&db);
    assert_eq!(
        metadata_row(&conn, "notes", "doc.md").unwrap().0,
        r#"{"status":"draft"}"#
    );
    assert_eq!(pending_metadata(&conn), 0);

    // Unchanged content on a later pass keeps extraction current.
    let second = update(&qmd);
    assert_eq!(second.unchanged, 1);
    assert_eq!(pending_metadata(&conn), 0);

    // Edited metadata replaces the previous rows.
    fs::write(
        &file,
        build_doc(
            Some("qmd:\n  metadata:\n    status: published\n"),
            "# Doc\n",
        ),
    )
    .unwrap();
    update(&qmd);
    assert_eq!(
        metadata_row(&conn, "notes", "doc.md").unwrap().0,
        r#"{"status":"published"}"#
    );
}

#[test]
fn backfills_documents_indexed_before_the_metadata_schema_existed() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    let db = tmp.path().join("i.sqlite");
    write(
        &docs,
        "doc.md",
        &build_doc(Some("qmd:\n  metadata:\n    status: ok\n"), "# Doc\n"),
    );
    let qmd = open(&db);
    qmd.add_collection(
        "notes",
        &CollectionSpec {
            path: docs.to_string_lossy().into_owned(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();
    update(&qmd);

    let conn = raw_conn(&db);
    // Simulate a pre-metadata index: drop the extraction rows.
    conn.execute("DELETE FROM document_metadata", []).unwrap();
    assert_eq!(pending_metadata(&conn), 1);

    let result = update(&qmd);
    assert_eq!(result.unchanged, 1);
    assert_eq!(pending_metadata(&conn), 0);
    assert_eq!(
        metadata_row(&conn, "notes", "doc.md").unwrap().0,
        r#"{"status":"ok"}"#
    );
}

#[test]
fn counts_extraction_errors_without_aborting_the_collection() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    let db = tmp.path().join("i.sqlite");
    write(
        &docs,
        "bad.md",
        &build_doc(Some("qmd:\n  metadata:\n    mixed: [1, two]\n"), "# Bad\n"),
    );
    write(
        &docs,
        "good.md",
        &build_doc(Some("qmd:\n  metadata:\n    status: ok\n"), "# Good\n"),
    );
    let qmd = open(&db);
    qmd.add_collection(
        "notes",
        &CollectionSpec {
            path: docs.to_string_lossy().into_owned(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();

    let result = update(&qmd);
    assert_eq!(result.indexed, 2);
    assert_eq!(result.metadata_errors, 1);
    let conn = raw_conn(&db);
    assert_eq!(
        metadata_row(&conn, "notes", "good.md").unwrap().0,
        r#"{"status":"ok"}"#
    );
    assert_eq!(metadata_row(&conn, "notes", "bad.md").unwrap().0, "{}");
    assert_eq!(pending_metadata(&conn), 1);
}

#[test]
fn deactivation_hides_then_reactivation_restores_metadata() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    let db = tmp.path().join("i.sqlite");
    write(
        &docs,
        "doc.md",
        &build_doc(Some("qmd:\n  metadata:\n    status: ok\n"), "# Doc\n"),
    );
    let qmd = open(&db);
    qmd.add_collection(
        "notes",
        &CollectionSpec {
            path: docs.to_string_lossy().into_owned(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();
    update(&qmd);

    fs::remove_file(docs.join("doc.md")).unwrap();
    let removed = update(&qmd);
    assert_eq!(removed.removed, 1);
    let conn = raw_conn(&db);
    // The row stays (upstream keeps it for potential reactivation) but
    // the document is inactive and leaves the pending count.
    let active: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM documents d
             JOIN document_metadata dm ON dm.document_id = d.id
             WHERE d.path = 'doc.md' AND d.active = 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(active, 0);
    assert_eq!(pending_metadata(&conn), 0);

    // Reactivation reuses the document id, and the metadata row is
    // synced again on the next pass.
    let old_id = document_id(&conn, "notes", "doc.md");
    write(
        &docs,
        "doc.md",
        &build_doc(Some("qmd:\n  metadata:\n    status: ok\n"), "# Doc\n"),
    );
    update(&qmd);
    assert_eq!(document_id(&conn, "notes", "doc.md"), old_id);
    assert_eq!(
        metadata_row(&conn, "notes", "doc.md").unwrap().0,
        r#"{"status":"ok"}"#
    );
}

#[test]
fn hard_deletion_cascades_metadata_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    let db = tmp.path().join("i.sqlite");
    write(
        &docs,
        "doc.md",
        &build_doc(Some("qmd:\n  metadata:\n    status: ok\n"), "# Doc\n"),
    );
    let qmd = open(&db);
    qmd.add_collection(
        "notes",
        &CollectionSpec {
            path: docs.to_string_lossy().into_owned(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();
    update(&qmd);

    fs::remove_file(docs.join("doc.md")).unwrap();
    update(&qmd);
    qmd.maintenance().delete_inactive_documents().unwrap();

    let conn = raw_conn(&db);
    assert_eq!(count_table(&conn, "document_metadata"), 0);
    assert_eq!(count_table(&conn, "document_metadata_values"), 0);
}

#[test]
fn collection_removal_cascades_metadata_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    let db = tmp.path().join("i.sqlite");
    write(
        &docs,
        "doc.md",
        &build_doc(Some("qmd:\n  metadata:\n    status: ok\n"), "# Doc\n"),
    );
    let qmd = open(&db);
    qmd.add_collection(
        "notes",
        &CollectionSpec {
            path: docs.to_string_lossy().into_owned(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();
    update(&qmd);

    qmd.remove_collection("notes").unwrap();
    // This implementation deactivates collection documents instead of
    // upstream's hard `DELETE` (store/collections.rs) — the FK cascade
    // runs when maintenance removes the inactive rows.
    qmd.maintenance().delete_inactive_documents().unwrap();

    let conn = raw_conn(&db);
    assert_eq!(count_table(&conn, "document_metadata"), 0);
    assert_eq!(count_table(&conn, "document_metadata_values"), 0);
}
