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

//! Storage-layer tests: schema creation, three-layer versioning,
//! foreign-index rejection, WAL/busy pragmas, sqlite-vec, concurrent
//! cold opens. Ports the semantic coverage of upstream `test/db` and
//! `test/store-concurrency`.

use std::path::Path;

use qmd::{Environment, Error, Qmd};

fn env() -> Environment {
    Environment::default()
}

fn open_qmd(path: &Path) -> qmd::Result<Qmd> {
    Qmd::builder(path).environment(env()).build()
}

fn raw_conn(path: &Path) -> rusqlite::Connection {
    rusqlite::Connection::open(path).expect("open raw connection")
}

fn table_exists(conn: &rusqlite::Connection, name: &str) -> bool {
    conn.query_row(
        "SELECT COUNT(*) > 0 FROM sqlite_master \
         WHERE type = 'table' AND name = ?",
        [name],
        |r| r.get(0),
    )
    .expect("query sqlite_master")
}

#[test]
fn creates_schema_and_marks_application_id() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("idx.sqlite");
    open_qmd(&db).expect("build").close().expect("close");

    let conn = raw_conn(&db);
    for t in [
        "store_collections",
        "store_config",
        "documents",
        "documents_fts",
        "content",
        "content_vectors",
        "llm_cache",
    ] {
        assert!(table_exists(&conn, t), "missing table {t}");
    }
    let app_id: i64 = conn
        .query_row("PRAGMA application_id", [], |r| r.get(0))
        .expect("app id");
    assert_eq!(app_id, 0x516D_6452);
    let journal: String = conn
        .query_row("PRAGMA journal_mode", [], |r| r.get(0))
        .expect("journal mode");
    assert_eq!(journal, "wal");
}

#[test]
fn vec_extension_works() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("idx.sqlite");
    let qmd = open_qmd(&db).expect("build");

    // The auto-extension is registered process-wide; any connection —
    // including this raw one — sees `vec_version()`.
    let conn = raw_conn(&db);
    let version: String = conn
        .query_row("SELECT vec_version()", [], |r| r.get(0))
        .expect("vec_version");
    assert!(version.starts_with('v'), "unexpected {version}");

    let dist: f64 = conn
        .query_row(
            "SELECT vec_distance_cosine('[1.0,0.0]', '[1.0,0.0]')",
            [],
            |r| r.get(0),
        )
        .expect("distance");
    assert_eq!(dist, 0.0);
    qmd.close().expect("close");
}

#[test]
fn rejects_foreign_index() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("idx.sqlite");
    {
        let conn = raw_conn(&db);
        conn.execute_batch("CREATE TABLE documents (id INTEGER PRIMARY KEY, x TEXT);")
            .expect("seed foreign schema");
    }
    let err = open_qmd(&db).expect_err("must reject foreign db");
    assert!(matches!(err, Error::ForeignIndex { .. }), "{err}");
}

#[test]
fn foreign_index_is_not_modified() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("idx.sqlite");
    {
        let conn = raw_conn(&db);
        conn.execute_batch("CREATE TABLE keepme (x TEXT);")
            .expect("seed");
    }
    let _ = open_qmd(&db).expect_err("reject");
    let conn = raw_conn(&db);
    assert!(table_exists(&conn, "keepme"));
    assert!(!table_exists(&conn, "store_collections"));
    let app_id: i64 = conn
        .query_row("PRAGMA application_id", [], |r| r.get(0))
        .expect("app id");
    assert_eq!(app_id, 0);
}

#[test]
fn structural_version_rebuild_keeps_config_layer() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("idx.sqlite");
    let qmd = open_qmd(&db).expect("build");
    qmd.add_collection(
        "docs",
        &qmd::CollectionSpec {
            path: "/data".to_owned(),
            ..Default::default()
        },
    )
    .expect("add");
    {
        let conn = raw_conn(&db);
        // Seed a documents row, then force the version mismatch.
        conn.execute_batch(
            "INSERT INTO documents \
             (collection, path, title, hash, created_at, modified_at) \
             VALUES ('docs', 'a.md', 'A', 'hash0', 't', 't');",
        )
        .expect("seed doc");
        conn.execute(
            "UPDATE store_config SET value = '0' WHERE key = 'schema_version'",
            [],
        )
        .expect("downgrade");
    }
    qmd.close().expect("close");

    let qmd = open_qmd(&db).expect("rebuild open");
    let conn = raw_conn(&db);
    // Structural data dropped, config layer preserved.
    let docs: i64 = conn
        .query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))
        .expect("count");
    assert_eq!(docs, 0);
    let colls: i64 = conn
        .query_row("SELECT COUNT(*) FROM store_collections", [], |r| r.get(0))
        .expect("count colls");
    assert_eq!(colls, 1);
    let reindex: String = conn
        .query_row(
            "SELECT value FROM store_config WHERE key = 'reindex_required'",
            [],
            |r| r.get(0),
        )
        .expect("reindex flag");
    assert_eq!(reindex, "1");
    let v: String = conn
        .query_row(
            "SELECT value FROM store_config WHERE key = 'schema_version'",
            [],
            |r| r.get(0),
        )
        .expect("version");
    assert_eq!(v, "1");
    qmd.close().expect("close");
}

#[test]
fn vector_version_rebuild_keeps_content() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("idx.sqlite");
    open_qmd(&db).expect("build").close().expect("close");
    {
        let conn = raw_conn(&db);
        conn.execute_batch(
            "INSERT INTO content (hash, doc, created_at) \
             VALUES ('h1', 'body', 't');
             INSERT INTO content_vectors \
             (hash, seq, pos, model, embedded_at) \
             VALUES ('h1', 0, 0, 'm', 't');
             UPDATE store_config SET value = '0' \
             WHERE key = 'vector_schema_version';",
        )
        .expect("seed + downgrade");
    }
    open_qmd(&db).expect("reopen").close().expect("close");
    let conn = raw_conn(&db);
    let content: i64 = conn
        .query_row("SELECT COUNT(*) FROM content", [], |r| r.get(0))
        .expect("count");
    assert_eq!(content, 1);
    let vectors: i64 = conn
        .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))
        .expect("count vectors");
    assert_eq!(vectors, 0);
    let v: String = conn
        .query_row(
            "SELECT value FROM store_config \
             WHERE key = 'vector_schema_version'",
            [],
            |r| r.get(0),
        )
        .expect("version");
    assert_eq!(v, "1");
}

#[test]
fn concurrent_cold_opens() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("idx.sqlite");
    let mut handles = Vec::new();
    for _ in 0..8 {
        let db = db.clone();
        handles.push(std::thread::spawn(move || open_qmd(&db)));
    }
    for h in handles {
        let qmd = h.join().expect("join").expect("open");
        qmd.close().expect("close");
    }
    // Schema initialized exactly once and correctly.
    let conn = raw_conn(&db);
    assert!(table_exists(&conn, "store_collections"));
}

#[test]
fn busy_timeout_zero_disables_waiting() {
    // `QMD_SQLITE_BUSY_TIMEOUT=0` restores fail-fast behavior: a write
    // under an external exclusive lock must error immediately instead of
    // waiting out the default 120 s budget.
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("idx.sqlite");
    let mut e = env();
    e.qmd_sqlite_busy_timeout = Some("0".to_owned());
    let qmd = Qmd::builder(&db).environment(e).build().expect("build");

    let blocker = raw_conn(&db);
    blocker
        .execute_batch("BEGIN EXCLUSIVE")
        .expect("acquire write lock");
    let err = qmd
        .add_collection(
            "docs",
            &qmd::CollectionSpec {
                path: "/data".to_owned(),
                ..Default::default()
            },
        )
        .expect_err("must fail fast while write-locked");
    assert!(matches!(err, Error::Db(_)), "{err}");
    blocker.execute_batch("ROLLBACK").expect("rollback");
    qmd.close().expect("close");
}
