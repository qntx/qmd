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

//! `Qmd` handle tests: builder config sources, collection CRUD with CLI
//! semantics, context management, config write-back.
//!
//! Ports the semantic coverage of upstream `test/collections-config`,
//! `test/store-insert-context`, `test/store-concurrency`, and the
//! config/collection/context parts of `test/sdk`.

use std::path::Path;

use qmd::{CollectionSettings, CollectionSpec, Config, Environment, Error, Qmd};

fn spec(path: &str) -> CollectionSpec {
    CollectionSpec {
        path: path.to_owned(),
        ..Default::default()
    }
}

fn open(path: &Path) -> Qmd {
    Qmd::builder(path)
        .environment(Environment::default())
        .build()
        .expect("build")
}

fn open_with_config(db: &Path, cfg_path: &Path) -> Qmd {
    Qmd::builder(db)
        .environment(Environment::default())
        .config_file(cfg_path)
        .build()
        .expect("build")
}

#[test]
fn add_list_show_remove_collection() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let qmd = open(&tmp.path().join("i.sqlite"));

    qmd.add_collection("docs", &spec("/data/docs"))
        .expect("add");
    qmd.add_collection("notes", &spec("/data/notes"))
        .expect("add");

    let names: Vec<String> = qmd
        .collections()
        .expect("list")
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert_eq!(names, ["docs", "notes"]);

    let shown = qmd.collection("docs").expect("show").expect("exists");
    assert_eq!(shown.path, "/data/docs");

    let removal = qmd
        .remove_collection("docs")
        .expect("remove")
        .expect("found");
    assert_eq!(removal.documents_deactivated, 0);
    assert!(qmd.collection("docs").expect("show").is_none());
    assert!(qmd.remove_collection("docs").expect("remove").is_none());
}

#[test]
fn add_collection_rejects_duplicates() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let qmd = open(&tmp.path().join("i.sqlite"));
    qmd.add_collection("a", &spec("/data")).expect("add");

    let err = qmd
        .add_collection("a", &spec("/other"))
        .expect_err("name dup");
    assert!(matches!(err, Error::CollectionExists { .. }), "{err}");

    // Same path + same effective pattern under a different name.
    let err = qmd
        .add_collection("b", &spec("/data"))
        .expect_err("source dup");
    assert!(
        matches!(err, Error::DuplicateCollectionSource { .. }),
        "{err}"
    );

    // Same path but a different pattern is fine.
    qmd.add_collection(
        "b",
        &CollectionSpec {
            path: "/data".to_owned(),
            pattern: Some("**/*.txt".to_owned()),
            ..Default::default()
        },
    )
    .expect("different pattern");
}

#[test]
fn rename_updates_documents() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("i.sqlite");
    let qmd = open(&db);
    qmd.add_collection("old", &spec("/data")).expect("add");
    {
        // Simulate an indexed document.
        let conn = rusqlite::Connection::open(&db).expect("conn");
        conn.execute_batch(
            "INSERT INTO documents \
             (collection, path, title, hash, created_at, modified_at) \
             VALUES ('old', 'a.md', 'A', 'h', 't', 't');
             INSERT INTO documents_fts (rowid, filepath, title, body) \
             SELECT id, collection || '/' || path, title, 'x' \
             FROM documents;",
        )
        .expect("seed");
    }
    assert!(qmd.rename_collection("old", "new").expect("rename"));
    assert!(!qmd.rename_collection("old", "new2").expect("rename again"));

    let conn = rusqlite::Connection::open(&db).expect("conn");
    let collection: String = conn
        .query_row("SELECT collection FROM documents", [], |r| r.get(0))
        .expect("doc collection");
    assert_eq!(collection, "new");

    // Renaming onto an existing name is rejected.
    qmd.add_collection("other", &spec("/other"))
        .expect("add other");
    let err = qmd.rename_collection("other", "new").expect_err("conflict");
    assert!(matches!(err, Error::CollectionExists { .. }), "{err}");
}

#[test]
fn remove_collection_deactivates_documents() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("i.sqlite");
    let qmd = open(&db);
    qmd.add_collection("docs", &spec("/data")).expect("add");
    {
        let conn = rusqlite::Connection::open(&db).expect("conn");
        conn.execute_batch(
            "INSERT INTO documents \
             (collection, path, title, hash, created_at, modified_at) \
             VALUES ('docs', 'a.md', 'A', 'h', 't', 't');
             INSERT INTO documents_fts (rowid, filepath, title, body) \
             SELECT id, collection || '/' || path, title, 'x' \
             FROM documents;",
        )
        .expect("seed");
    }
    let removal = qmd
        .remove_collection("docs")
        .expect("remove")
        .expect("found");
    assert_eq!(removal.documents_deactivated, 1);
    let conn = rusqlite::Connection::open(&db).expect("conn");
    let active: i64 = conn
        .query_row("SELECT active FROM documents", [], |r| r.get(0))
        .expect("active");
    assert_eq!(active, 0);
    let fts: i64 = conn
        .query_row("SELECT COUNT(*) FROM documents_fts", [], |r| r.get(0))
        .expect("fts count");
    assert_eq!(fts, 0);
}

#[test]
fn update_collection_settings() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let qmd = open(&tmp.path().join("i.sqlite"));
    qmd.add_collection("docs", &spec("/data")).expect("add");

    qmd.update_collection_settings(
        "docs",
        &CollectionSettings {
            update: Some(Some("git pull".to_owned())),
            include_by_default: Some(false),
        },
    )
    .expect("settings");
    let coll = qmd.collection("docs").expect("show").expect("exists");
    assert_eq!(coll.update.as_deref(), Some("git pull"));
    assert_eq!(coll.include_by_default, Some(false));

    // Clear the hook, leave the flag alone.
    qmd.update_collection_settings(
        "docs",
        &CollectionSettings {
            update: Some(None),
            include_by_default: None,
        },
    )
    .expect("clear");
    let coll = qmd.collection("docs").expect("show").expect("exists");
    assert!(coll.update.is_none());
    assert_eq!(coll.include_by_default, Some(false));

    let err = qmd
        .update_collection_settings("nope", &CollectionSettings::default())
        .expect_err("missing");
    assert!(matches!(err, Error::CollectionNotFound { .. }), "{err}");
}

#[test]
fn default_collection_names_filters_excluded() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let qmd = open(&tmp.path().join("i.sqlite"));
    qmd.add_collection("a", &spec("/a")).expect("add a");
    qmd.add_collection("b", &spec("/b")).expect("add b");
    qmd.update_collection_settings(
        "b",
        &CollectionSettings {
            include_by_default: Some(false),
            ..Default::default()
        },
    )
    .expect("exclude");
    assert_eq!(qmd.default_collection_names().expect("names"), ["a"]);
}

#[test]
fn global_and_collection_contexts() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let qmd = open(&tmp.path().join("i.sqlite"));
    qmd.add_collection("docs", &spec("/data")).expect("add");

    qmd.set_global_context(Some("global words"))
        .expect("set global");
    assert_eq!(
        qmd.global_context().expect("get global").as_deref(),
        Some("global words")
    );

    qmd.add_context("docs", "2024", "journal notes")
        .expect("add ctx");
    let entries = qmd.contexts().expect("contexts");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].collection, "*");
    assert_eq!(entries[0].path, "/");
    assert_eq!(entries[1].collection, "docs");
    assert_eq!(entries[1].path, "2024");

    assert!(qmd.remove_context("docs", "2024").expect("rm ctx"));
    assert!(!qmd.remove_context("docs", "2024").expect("rm again"));
    assert!(!qmd.remove_context("nope", "x").expect("rm unknown"));

    qmd.set_global_context(None).expect("clear global");
    assert!(qmd.global_context().expect("get").is_none());

    let err = qmd.add_context("nope", "p", "t").expect_err("missing coll");
    assert!(matches!(err, Error::CollectionNotFound { .. }), "{err}");
}

#[test]
fn config_file_writeback() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("i.sqlite");
    let cfg_path = tmp.path().join("qmd.yml");
    let qmd = open_with_config(&db, &cfg_path);
    qmd.add_collection("docs", &spec("/data")).expect("add");
    qmd.add_context("docs", "sub", "ctx text").expect("ctx");
    qmd.set_global_context(Some("g")).expect("global");
    qmd.close().expect("close");

    // The YAML file was written and round-trips.
    let file_cfg = qmd::config::load(&cfg_path).expect("load yaml");
    assert!(file_cfg.collections.contains_key("docs"));
    assert_eq!(file_cfg.global_context.as_deref(), Some("g"));
    assert_eq!(
        file_cfg.collections["docs"]
            .context
            .as_ref()
            .and_then(|c| c.get("sub"))
            .map(String::as_str),
        Some("ctx text")
    );

    // Reopening with the same file sees the collection from YAML.
    let qmd = open_with_config(&db, &cfg_path);
    let coll = qmd.collection("docs").expect("show").expect("exists");
    assert_eq!(coll.path, "/data");
}

#[test]
fn db_only_mode_assembles_config_from_store() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("i.sqlite");
    let cfg_path = tmp.path().join("qmd.yml");
    {
        let qmd = open_with_config(&db, &cfg_path);
        qmd.add_collection("docs", &spec("/data")).expect("add");
        qmd.close().expect("close");
    }
    // Reopen without the config file: the DB mirror still knows the
    // collections (upstream parity: store_collections survives config loss).
    let qmd = open(&db);
    let names: Vec<String> = qmd
        .collections()
        .expect("list")
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert_eq!(names, ["docs"]);
    assert_eq!(
        qmd.collection("docs").expect("show").expect("x").path,
        "/data"
    );
}

#[test]
fn inline_config_source() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut config = Config::default();
    config.collections.insert(
        "inline".to_owned(),
        qmd::Collection {
            path: "/inl".to_owned(),
            ..Default::default()
        },
    );
    let qmd = Qmd::builder(tmp.path().join("i.sqlite"))
        .environment(Environment::default())
        .config(config)
        .build()
        .expect("build");
    assert!(qmd.collection("inline").expect("show").is_some());

    // Mutations land in memory + the DB mirror, no file is written.
    qmd.add_collection("extra", &spec("/e")).expect("add");
    let names: Vec<String> = qmd
        .collections()
        .expect("list")
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert_eq!(names, ["inline", "extra"]);
    assert!(!tmp.path().join("qmd.yml").exists());
}

#[test]
fn two_handles_mutate_without_corruption() {
    // Ported from upstream store-concurrency: two stores on the same file
    // both mutate config; WAL + busy timeout serialize them.
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("i.sqlite");
    let a = open(&db);
    let b = open(&db);
    a.add_collection("a", &spec("/a")).expect("add a");
    b.add_collection("b", &spec("/b")).expect("add b");
    a.set_global_context(Some("from a")).expect("global a");
    b.add_context("b", "p", "t").expect("ctx b");

    // DbOnly views resolve through store_collections on each handle.
    let names_a: Vec<String> = a
        .collections()
        .expect("list a")
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert!(names_a.contains(&"a".to_owned()));
    assert!(names_a.contains(&"b".to_owned()));
}

#[test]
fn detect_collection_by_path() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().join("notes");
    let nested = root.join("sub");
    std::fs::create_dir_all(&nested).expect("mkdir");
    let file = nested.join("a.md");
    std::fs::write(&file, "hi").expect("write");

    let qmd = open(&tmp.path().join("i.sqlite"));
    qmd.add_collection("notes", &spec(root.to_str().expect("utf8 path")))
        .expect("add");
    let hit = qmd
        .detect_collection(&file)
        .expect("detect")
        .expect("found");
    assert_eq!(hit.name, "notes");
    assert_eq!(hit.relative_path, "sub/a.md");
    assert!(
        qmd.detect_collection(tmp.path())
            .expect("detect none")
            .is_none()
    );
}

#[test]
fn local_config_discovery_flow() {
    // Mirrors `qmd init` semantics: a `.qmd/index.yaml` found by walking
    // up, DB lives next to it as `index-rs.sqlite`.
    let tmp = tempfile::tempdir().expect("tempdir");
    let proj = tmp.path().join("proj");
    let qmd_dir = proj.join(".qmd");
    std::fs::create_dir_all(&qmd_dir).expect("mkdir");
    let cfg_path = qmd_dir.join("index.yaml");
    std::fs::write(&cfg_path, "collections: {}\n").expect("write");

    let deep = proj.join("a/b/c");
    std::fs::create_dir_all(&deep).expect("mkdir");
    let found = qmd::paths::find_local_config(&deep).expect("find");
    assert_eq!(found, cfg_path);
    assert_eq!(
        qmd::paths::local_db_path(&found),
        qmd_dir.join("index-rs.sqlite")
    );

    let qmd = open_with_config(&qmd::paths::local_db_path(&found), &found);
    qmd.add_collection("docs", &spec("/d")).expect("add");
}
