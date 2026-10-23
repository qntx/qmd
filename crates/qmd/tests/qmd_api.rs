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

/// `spec` with the CLI's resolved default pattern, like `collection
/// add` without `--mask`.
fn md_spec(path: &str) -> CollectionSpec {
    CollectionSpec {
        pattern: Some("**/*.md".to_owned()),
        ..spec(path)
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
    assert_eq!(removal.deleted_docs, 0);
    assert!(qmd.collection("docs").expect("show").is_none());
    assert!(qmd.remove_collection("docs").expect("remove").is_none());
}

#[test]
fn add_collection_rejects_duplicates() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let qmd = open(&tmp.path().join("i.sqlite"));
    // The CLI always writes a concrete `pattern:` (its `--mask` default
    // resolves before the add), so the dup check compares stored
    // `Some("**/*.md")` rows against a `Some("**/*.md")` spec —
    // upstream `c.pattern === globPattern` (cli/qmd.ts:1845-1846).
    qmd.add_collection("a", &md_spec("/data")).expect("add");

    let err = qmd
        .add_collection("a", &md_spec("/other"))
        .expect_err("name dup");
    assert!(matches!(err, Error::CollectionExists { .. }), "{err}");

    // Same path + same effective pattern under a different name.
    let err = qmd
        .add_collection("b", &md_spec("/data"))
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
fn remove_collection_deletes_documents() {
    // Upstream `removeCollection` (store.ts:3680-3695): `documents` rows
    // are hard-deleted (with their FTS rows) and content hashes left
    // unreferenced by active documents are dropped.
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
    // Bun `run().changes` is `sqlite3_total_changes` — the one document
    // row delete plus the FTS shadow deletions it triggers count 8,
    // matching upstream's `Deleted 8 documents` for one file.
    assert_eq!(removal.deleted_docs, 8);
    assert_eq!(removal.cleaned_hashes, 0, "no content rows were seeded");
    let conn = rusqlite::Connection::open(&db).expect("conn");
    let docs: i64 = conn
        .query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))
        .expect("doc count");
    assert_eq!(docs, 0);
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

#[test]
fn ensure_models_configured_writes_resolved_back() {
    // Upstream `ensureModelsConfiguredForCli` (qmd.ts:2143-2163): when
    // the config's models block differs from the resolved values, the
    // resolved embed/generate/rerank are written back while other keys
    // (`embed_pooling` here) survive.
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("i.sqlite");
    let cfg_path = tmp.path().join("qmd.yml");
    std::fs::write(&cfg_path, "models:\n  embed_pooling: full_sequence\n").expect("seed config");

    let qmd = open_with_config(&db, &cfg_path);
    let resolved = qmd.ensure_models_configured();
    assert_eq!(resolved.embed, qmd::llm::DEFAULT_EMBED_MODEL);
    assert_eq!(resolved.generate, qmd::llm::DEFAULT_GENERATE_MODEL);
    assert_eq!(resolved.rerank, qmd::llm::DEFAULT_RERANK_MODEL);
    qmd.close().expect("close");

    let file_cfg = qmd::config::load(&cfg_path).expect("load yaml");
    let models = file_cfg.models.expect("models written");
    assert_eq!(models.embed.as_deref(), Some(qmd::llm::DEFAULT_EMBED_MODEL));
    assert_eq!(
        models.generate.as_deref(),
        Some(qmd::llm::DEFAULT_GENERATE_MODEL)
    );
    assert_eq!(
        models.rerank.as_deref(),
        Some(qmd::llm::DEFAULT_RERANK_MODEL)
    );
    assert_eq!(
        models.embed_pooling,
        Some(qmd::config::EmbedPooling::FullSequence),
        "sibling models keys must be preserved"
    );

    // Second call is a no-op write-wise and still resolves.
    let qmd = open_with_config(&db, &cfg_path);
    let again = qmd.ensure_models_configured();
    assert_eq!(again, resolved);
}

#[test]
fn ensure_models_configured_db_only_does_not_write() {
    // No config source to pin: resolved values are returned and nothing
    // fails (upstream has no YAML to persist in this mode either).
    let tmp = tempfile::tempdir().expect("tempdir");
    let qmd = open(&tmp.path().join("i.sqlite"));
    let resolved = qmd.ensure_models_configured();
    assert_eq!(resolved.embed, qmd::llm::DEFAULT_EMBED_MODEL);
}

#[test]
fn ensure_models_configured_env_override_wins() {
    // Resolution order env > default when the config field is unset,
    // matching `resolve_models`; the env value is what gets pinned.
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("i.sqlite");
    let cfg_path = tmp.path().join("qmd.yml");
    let mut env = Environment::default();
    env.qmd_embed_model = Some("hf:custom/embed.gguf".to_owned());
    let qmd = Qmd::builder(&db)
        .environment(env)
        .config_file(&cfg_path)
        .build()
        .expect("build");
    let resolved = qmd.ensure_models_configured();
    assert_eq!(resolved.embed, "hf:custom/embed.gguf");
    qmd.close().expect("close");

    let file_cfg = qmd::config::load(&cfg_path).expect("load yaml");
    assert_eq!(
        file_cfg.models.and_then(|m| m.embed).as_deref(),
        Some("hf:custom/embed.gguf")
    );
}

// --- project-local init + gated model resolution (P4.2) -------------------

/// Fresh `init`: `.qmd/index.yml` gets the byte-exact upstream
/// `saveConfig` output and the database opens at `index-rs.sqlite`.
#[test]
fn init_local_fresh_writes_defaults() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).expect("mkdir");
    let env = Environment::default();

    let config_path = Qmd::init_local(&env, &project).expect("init");
    assert_eq!(config_path, project.join(".qmd/index.yml"));
    let want = format!(
        "collections: {{}}\nmodels:\n  embed: {}\n  generate: {}\n  rerank: {}\n",
        qmd::llm::DEFAULT_EMBED_MODEL,
        qmd::llm::DEFAULT_GENERATE_MODEL,
        qmd::llm::DEFAULT_RERANK_MODEL,
    );
    assert_eq!(std::fs::read_to_string(&config_path).expect("read"), want);
    // Deliberate deviation: `.qmd/index-rs.sqlite`, not upstream's
    // `index.sqlite` (paths.rs `local_db_path`).
    assert!(project.join(".qmd/index-rs.sqlite").exists());
}

/// `index.yaml` wins over `index.yml` when it already exists; an
/// existing config goes through `ensure_models_configured` instead of
/// being rewritten from scratch.
#[test]
fn init_local_existing_yaml_wins_and_gets_models() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let project = tmp.path().join("proj");
    let qmd_dir = project.join(".qmd");
    std::fs::create_dir_all(&qmd_dir).expect("mkdir");
    std::fs::write(
        qmd_dir.join("index.yaml"),
        "collections:\n  docs:\n    path: ./docs\n    pattern: \"**/*.md\"\n",
    )
    .expect("seed yaml");
    let env = Environment::default();

    let config_path = Qmd::init_local(&env, &project).expect("init");
    assert_eq!(config_path, qmd_dir.join("index.yaml"));
    // Collections survive; `models:` was pinned by ensure.
    let yaml = std::fs::read_to_string(&config_path).expect("read");
    assert!(yaml.contains("docs:"), "{yaml}");
    assert!(yaml.contains("embed: "), "{yaml}");
    // No competing `.yml` was created.
    assert!(!qmd_dir.join("index.yml").exists());
}

/// An existing `index.yml` is adopted (not overwritten) and gets the
/// models block — upstream `ensureModelsConfiguredForCli` semantics.
#[test]
fn init_local_existing_yml_is_adopted() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let project = tmp.path().join("proj");
    let qmd_dir = project.join(".qmd");
    std::fs::create_dir_all(&qmd_dir).expect("mkdir");
    std::fs::write(qmd_dir.join("index.yml"), "collections: {}\n").expect("seed yml");
    let env = Environment::default();

    let config_path = Qmd::init_local(&env, &project).expect("init");
    assert_eq!(config_path, qmd_dir.join("index.yml"));
    let yaml = std::fs::read_to_string(&config_path).expect("read");
    assert!(yaml.contains("rerank: "), "{yaml}");
}

/// `init` in `$HOME` is refused before anything is created — upstream
/// `sameDirectory(cwd, homedir())` (cli/qmd.ts:440-453).
#[test]
fn init_local_refuses_home() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).expect("mkdir");
    let mut env = Environment::default();
    env.home = Some(project.clone());

    let err = Qmd::init_local(&env, &project).expect_err("must refuse");
    let Error::LocalInitRefused { reason } = err else {
        panic!("expected LocalInitRefused, got {err}");
    };
    assert!(
        reason.starts_with("Refusing to initialize a local index in $HOME"),
        "{reason}"
    );
    assert!(!project.join(".qmd").exists());
}

/// Running `init` twice is a no-op beyond `ensure_models_configured`:
/// same path, same bytes, no second `index.yaml`.
#[test]
fn init_local_is_idempotent() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).expect("mkdir");
    let env = Environment::default();

    let first = Qmd::init_local(&env, &project).expect("init");
    let bytes = std::fs::read_to_string(&first).expect("read");
    let second = Qmd::init_local(&env, &project).expect("init again");
    assert_eq!(first, second);
    assert_eq!(std::fs::read_to_string(&second).expect("read"), bytes);
    assert!(!project.join(".qmd/index.yaml").exists());
}

/// `use_config_models(false)` drops a declared `models:` block out of
/// *runtime* resolution — env/defaults win — while
/// `ensure_models_configured` still resolves the declared triple and
/// must not overwrite the custom URI (upstream
/// `resolveModelsForRuntime` vs `ensureModelsConfiguredForCli`,
/// qmd.ts:2183-2188 / 2143-2163).
#[test]
fn use_config_models_false_ignores_declared_models() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("i.sqlite");
    let cfg = tmp.path().join("index.yml");
    std::fs::write(
        &cfg,
        "collections: {}\nmodels:\n  embed: hf:custom/e.gguf\n",
    )
    .expect("write cfg");

    // Seed an active document plus a complete vector row recorded under
    // the *declared* model's fingerprint — `needs_embedding` then
    // exposes which model the runtime resolved.
    {
        let schema = open(&db);
        schema.close().expect("close");
        let fingerprint = qmd::llm::embedding_fingerprint("hf:custom/e.gguf", None);
        let conn = rusqlite::Connection::open(&db).expect("conn");
        conn.execute_batch(&format!(
            "INSERT INTO documents \
             (collection, path, title, hash, created_at, modified_at) \
             VALUES ('docs', 'a.md', 'A', 'h1', 't', 't'); \
             INSERT INTO content_vectors \
             (hash, seq, model, embed_fingerprint, total_chunks, embedded_at) \
             VALUES ('h1', 0, 'hf:custom/e.gguf', '{fingerprint}', 1, 't');"
        ))
        .expect("seed");
    }

    // Trusted: the declared URI steers runtime resolution, so the
    // recorded vector satisfies the pending count.
    let trusted = open_with_config(&db, &cfg);
    assert_eq!(trusted.ensure_models_configured().embed, "hf:custom/e.gguf");
    assert_eq!(
        trusted
            .update_tail_counts()
            .expect("tail counts")
            .needs_embedding,
        0
    );
    trusted.close().expect("close");

    // Gated: runtime resolution ignores the declared URI and falls back
    // to the default — the custom-model vector no longer satisfies.
    let gated = Qmd::builder(&db)
        .environment(Environment::default())
        .config_file(&cfg)
        .use_config_models(false)
        .build()
        .expect("build");
    assert_eq!(
        gated
            .update_tail_counts()
            .expect("tail counts")
            .needs_embedding,
        1
    );

    // `ensure_models_configured` is not gated: it resolves the declared
    // triple and the write-back pins missing fields while preserving
    // the custom `embed` URI rather than clobbering it with the
    // runtime default.
    assert_eq!(gated.ensure_models_configured().embed, "hf:custom/e.gguf");
    gated.close().expect("close");
    let after = std::fs::read_to_string(&cfg).expect("read cfg");
    assert!(after.contains("embed: hf:custom/e.gguf"), "{after}");
    assert!(
        !after.contains(&format!("embed: {}", qmd::llm::DEFAULT_EMBED_MODEL)),
        "{after}"
    );

    // An env override wins runtime resolution under the gate (the
    // declared value is invisible to it) — a vector recorded under the
    // env model now satisfies the pending count.
    let env_fp = qmd::llm::embedding_fingerprint("hf:env/model.gguf", None);
    let conn = rusqlite::Connection::open(&db).expect("conn");
    conn.execute(
        "UPDATE content_vectors SET model = 'hf:env/model.gguf', \
         embed_fingerprint = ? WHERE hash = 'h1'",
        [&env_fp],
    )
    .expect("re-tag vector");
    drop(conn);
    let mut env = Environment::default();
    env.qmd_embed_model = Some("hf:env/model.gguf".to_owned());
    let env_gated = Qmd::builder(&db)
        .environment(env)
        .config_file(&cfg)
        .use_config_models(false)
        .build()
        .expect("build");
    assert_eq!(
        env_gated
            .update_tail_counts()
            .expect("tail counts")
            .needs_embedding,
        0
    );
    // But `ensure_models_configured` still prefers declared > env.
    assert_eq!(
        env_gated.ensure_models_configured().embed,
        "hf:custom/e.gguf"
    );
}
