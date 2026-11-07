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

//! `Meme` handle tests: builder config sources, collection CRUD with CLI
//! semantics, context management, config write-back.
//!
//! Ports the semantic coverage of upstream `test/collections-config`,
//! `test/store-insert-context`, `test/store-concurrency`, and the
//! config/collection/context parts of `test/sdk`.

use std::path::Path;

use meme::{CollectionSettings, CollectionSpec, Config, Environment, Error, Meme};

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

fn open(path: &Path) -> Meme {
    Meme::builder(path)
        .environment(Environment::default())
        .build()
        .expect("build")
}

fn open_with_config(db: &Path, cfg_path: &Path) -> Meme {
    Meme::builder(db)
        .environment(Environment::default())
        .config_file(cfg_path)
        .build()
        .expect("build")
}

#[test]
fn add_list_show_remove_collection() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let meme = open(&tmp.path().join("i.sqlite"));

    meme.add_collection("docs", &spec("/data/docs"))
        .expect("add");
    meme.add_collection("notes", &spec("/data/notes"))
        .expect("add");

    let names: Vec<String> = meme
        .collections()
        .expect("list")
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert_eq!(names, ["docs", "notes"]);

    let shown = meme.collection("docs").expect("show").expect("exists");
    assert_eq!(shown.path, "/data/docs");

    let removal = meme
        .remove_collection("docs")
        .expect("remove")
        .expect("found");
    assert_eq!(removal.deleted_docs, 0);
    assert!(meme.collection("docs").expect("show").is_none());
    assert!(meme.remove_collection("docs").expect("remove").is_none());
}

#[test]
fn add_collection_rejects_duplicates() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let meme = open(&tmp.path().join("i.sqlite"));
    // The CLI always writes a concrete `pattern:` (its `--mask` default
    // resolves before the add), so the dup check compares stored
    // `Some("**/*.md")` rows against a `Some("**/*.md")` spec —
    // upstream `c.pattern === globPattern` (cli/qmd.ts:1845-1846).
    meme.add_collection("a", &md_spec("/data")).expect("add");

    let err = meme
        .add_collection("a", &md_spec("/other"))
        .expect_err("name dup");
    assert!(matches!(err, Error::CollectionExists { .. }), "{err}");

    // Same path + same effective pattern under a different name.
    let err = meme
        .add_collection("b", &md_spec("/data"))
        .expect_err("source dup");
    assert!(
        matches!(err, Error::DuplicateCollectionSource { .. }),
        "{err}"
    );

    // Same path but a different pattern is fine.
    meme.add_collection(
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
    let meme = open(&db);
    meme.add_collection("old", &spec("/data")).expect("add");
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
    assert!(meme.rename_collection("old", "new").expect("rename"));
    assert!(!meme.rename_collection("old", "new2").expect("rename again"));

    let conn = rusqlite::Connection::open(&db).expect("conn");
    let collection: String = conn
        .query_row("SELECT collection FROM documents", [], |r| r.get(0))
        .expect("doc collection");
    assert_eq!(collection, "new");

    // Renaming onto an existing name is rejected.
    meme.add_collection("other", &spec("/other"))
        .expect("add other");
    let err = meme
        .rename_collection("other", "new")
        .expect_err("conflict");
    assert!(matches!(err, Error::CollectionExists { .. }), "{err}");
}

#[test]
fn remove_collection_deletes_documents() {
    // Upstream `removeCollection` (store.ts:3680-3695): `documents` rows
    // are hard-deleted (with their FTS rows) and content hashes left
    // unreferenced by active documents are dropped.
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("i.sqlite");
    let meme = open(&db);
    meme.add_collection("docs", &spec("/data")).expect("add");
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
    let removal = meme
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
    let meme = open(&tmp.path().join("i.sqlite"));
    meme.add_collection("docs", &spec("/data")).expect("add");

    meme.update_collection_settings(
        "docs",
        &CollectionSettings {
            update: Some(Some("git pull".to_owned())),
            include_by_default: Some(false),
        },
    )
    .expect("settings");
    let coll = meme.collection("docs").expect("show").expect("exists");
    assert_eq!(coll.update.as_deref(), Some("git pull"));
    assert_eq!(coll.include_by_default, Some(false));

    // Clear the hook, leave the flag alone.
    meme.update_collection_settings(
        "docs",
        &CollectionSettings {
            update: Some(None),
            include_by_default: None,
        },
    )
    .expect("clear");
    let coll = meme.collection("docs").expect("show").expect("exists");
    assert!(coll.update.is_none());
    assert_eq!(coll.include_by_default, Some(false));

    let err = meme
        .update_collection_settings("nope", &CollectionSettings::default())
        .expect_err("missing");
    assert!(matches!(err, Error::CollectionNotFound { .. }), "{err}");
}

#[test]
fn default_collection_names_filters_excluded() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let meme = open(&tmp.path().join("i.sqlite"));
    meme.add_collection("a", &spec("/a")).expect("add a");
    meme.add_collection("b", &spec("/b")).expect("add b");
    meme.update_collection_settings(
        "b",
        &CollectionSettings {
            include_by_default: Some(false),
            ..Default::default()
        },
    )
    .expect("exclude");
    assert_eq!(meme.default_collection_names().expect("names"), ["a"]);
}

#[test]
fn global_and_collection_contexts() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let meme = open(&tmp.path().join("i.sqlite"));
    meme.add_collection("docs", &spec("/data")).expect("add");

    meme.set_global_context(Some("global words"))
        .expect("set global");
    assert_eq!(
        meme.global_context().expect("get global").as_deref(),
        Some("global words")
    );

    meme.add_context("docs", "2024", "journal notes")
        .expect("add ctx");
    let entries = meme.contexts().expect("contexts");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].collection, "*");
    assert_eq!(entries[0].path, "/");
    assert_eq!(entries[1].collection, "docs");
    assert_eq!(entries[1].path, "2024");

    assert!(meme.remove_context("docs", "2024").expect("rm ctx"));
    assert!(!meme.remove_context("docs", "2024").expect("rm again"));
    assert!(!meme.remove_context("nope", "x").expect("rm unknown"));

    meme.set_global_context(None).expect("clear global");
    assert!(meme.global_context().expect("get").is_none());

    let err = meme
        .add_context("nope", "p", "t")
        .expect_err("missing coll");
    assert!(matches!(err, Error::CollectionNotFound { .. }), "{err}");
}

#[test]
fn config_file_writeback() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("i.sqlite");
    let cfg_path = tmp.path().join("meme.yml");
    let meme = open_with_config(&db, &cfg_path);
    meme.add_collection("docs", &spec("/data")).expect("add");
    meme.add_context("docs", "sub", "ctx text").expect("ctx");
    meme.set_global_context(Some("g")).expect("global");
    meme.close().expect("close");

    // The YAML file was written and round-trips.
    let file_cfg = meme::config::load(&cfg_path).expect("load yaml");
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
    let meme = open_with_config(&db, &cfg_path);
    let coll = meme.collection("docs").expect("show").expect("exists");
    assert_eq!(coll.path, "/data");
}

#[test]
fn db_only_mode_assembles_config_from_store() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("i.sqlite");
    let cfg_path = tmp.path().join("meme.yml");
    {
        let meme = open_with_config(&db, &cfg_path);
        meme.add_collection("docs", &spec("/data")).expect("add");
        meme.close().expect("close");
    }
    // Reopen without the config file: the DB mirror still knows the
    // collections (upstream parity: store_collections survives config loss).
    let meme = open(&db);
    let names: Vec<String> = meme
        .collections()
        .expect("list")
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert_eq!(names, ["docs"]);
    assert_eq!(
        meme.collection("docs").expect("show").expect("x").path,
        "/data"
    );
}

#[test]
fn inline_config_source() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut config = Config::default();
    config.collections.insert(
        "inline".to_owned(),
        meme::Collection {
            path: "/inl".to_owned(),
            ..Default::default()
        },
    );
    let meme = Meme::builder(tmp.path().join("i.sqlite"))
        .environment(Environment::default())
        .config(config)
        .build()
        .expect("build");
    assert!(meme.collection("inline").expect("show").is_some());

    // Mutations land in memory + the DB mirror, no file is written.
    meme.add_collection("extra", &spec("/e")).expect("add");
    let names: Vec<String> = meme
        .collections()
        .expect("list")
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert_eq!(names, ["inline", "extra"]);
    assert!(!tmp.path().join("meme.yml").exists());
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

    let meme = open(&tmp.path().join("i.sqlite"));
    meme.add_collection("notes", &spec(root.to_str().expect("utf8 path")))
        .expect("add");
    let hit = meme
        .detect_collection(&file)
        .expect("detect")
        .expect("found");
    assert_eq!(hit.name, "notes");
    assert_eq!(hit.relative_path, "sub/a.md");
    assert!(
        meme.detect_collection(tmp.path())
            .expect("detect none")
            .is_none()
    );
}

#[test]
fn local_config_discovery_flow() {
    // Mirrors `meme init` semantics: a `.meme/index.yaml` found by walking
    // up, DB lives next to it as `index.sqlite`.
    let tmp = tempfile::tempdir().expect("tempdir");
    let proj = tmp.path().join("proj");
    let meme_dir = proj.join(".meme");
    std::fs::create_dir_all(&meme_dir).expect("mkdir");
    let cfg_path = meme_dir.join("index.yaml");
    std::fs::write(&cfg_path, "collections: {}\n").expect("write");

    let deep = proj.join("a/b/c");
    std::fs::create_dir_all(&deep).expect("mkdir");
    let found = meme::paths::find_local_config(&deep).expect("find");
    assert_eq!(found, cfg_path);
    assert_eq!(
        meme::paths::local_db_path(&found),
        meme_dir.join("index.sqlite")
    );

    let meme = open_with_config(&meme::paths::local_db_path(&found), &found);
    meme.add_collection("docs", &spec("/d")).expect("add");
}

#[test]
fn ensure_models_configured_writes_resolved_back() {
    // Upstream `ensureModelsConfiguredForCli` (qmd.ts:2143-2163): when
    // the config's models block differs from the resolved values, the
    // resolved embed/generate/rerank are written back while other keys
    // (`embed_pooling` here) survive.
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("i.sqlite");
    let cfg_path = tmp.path().join("meme.yml");
    std::fs::write(&cfg_path, "models:\n  embed_pooling: full_sequence\n").expect("seed config");

    let meme = open_with_config(&db, &cfg_path);
    let resolved = meme.ensure_models_configured();
    assert_eq!(resolved.embed, meme::llm::DEFAULT_EMBED_MODEL);
    assert_eq!(resolved.generate, meme::llm::DEFAULT_GENERATE_MODEL);
    assert_eq!(resolved.rerank, meme::llm::DEFAULT_RERANK_MODEL);
    meme.close().expect("close");

    let file_cfg = meme::config::load(&cfg_path).expect("load yaml");
    let models = file_cfg.models.expect("models written");
    assert_eq!(
        models.embed.as_deref(),
        Some(meme::llm::DEFAULT_EMBED_MODEL)
    );
    assert_eq!(
        models.generate.as_deref(),
        Some(meme::llm::DEFAULT_GENERATE_MODEL)
    );
    assert_eq!(
        models.rerank.as_deref(),
        Some(meme::llm::DEFAULT_RERANK_MODEL)
    );
    assert_eq!(
        models.embed_pooling,
        Some(meme::config::EmbedPooling::FullSequence),
        "sibling models keys must be preserved"
    );

    // Second call is a no-op write-wise and still resolves.
    let meme = open_with_config(&db, &cfg_path);
    let again = meme.ensure_models_configured();
    assert_eq!(again, resolved);
}

#[test]
fn ensure_models_configured_db_only_does_not_write() {
    // No config source to pin: resolved values are returned and nothing
    // fails (upstream has no YAML to persist in this mode either).
    let tmp = tempfile::tempdir().expect("tempdir");
    let meme = open(&tmp.path().join("i.sqlite"));
    let resolved = meme.ensure_models_configured();
    assert_eq!(resolved.embed, meme::llm::DEFAULT_EMBED_MODEL);
}

#[test]
fn ensure_models_configured_env_override_wins() {
    // Resolution order env > default when the config field is unset,
    // matching `resolve_models`; the env value is what gets pinned.
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("i.sqlite");
    let cfg_path = tmp.path().join("meme.yml");
    let mut env = Environment::default();
    env.meme_embed_model = Some("hf:custom/embed.gguf".to_owned());
    let meme = Meme::builder(&db)
        .environment(env)
        .config_file(&cfg_path)
        .build()
        .expect("build");
    let resolved = meme.ensure_models_configured();
    assert_eq!(resolved.embed, "hf:custom/embed.gguf");
    meme.close().expect("close");

    let file_cfg = meme::config::load(&cfg_path).expect("load yaml");
    assert_eq!(
        file_cfg.models.and_then(|m| m.embed).as_deref(),
        Some("hf:custom/embed.gguf")
    );
}

// --- project-local init + gated model resolution (P4.2) -------------------

/// Fresh `init`: `.meme/index.yml` gets the byte-exact upstream
/// `saveConfig` output and the database opens at `index.sqlite`.
#[test]
fn init_local_fresh_writes_defaults() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).expect("mkdir");
    let env = Environment::default();

    let config_path = Meme::init_local(&env, &project).expect("init");
    assert_eq!(config_path, project.join(".meme/index.yml"));
    let want = format!(
        "collections: {{}}\nmodels:\n  embed: {}\n  generate: {}\n  rerank: {}\n",
        meme::llm::DEFAULT_EMBED_MODEL,
        meme::llm::DEFAULT_GENERATE_MODEL,
        meme::llm::DEFAULT_RERANK_MODEL,
    );
    assert_eq!(std::fs::read_to_string(&config_path).expect("read"), want);
    // Deliberate deviation: `.meme/index.sqlite`, not upstream's
    // `index.sqlite` (paths.rs `local_db_path`).
    assert!(project.join(".meme/index.sqlite").exists());
}

/// `index.yaml` wins over `index.yml` when it already exists; an
/// existing config goes through `ensure_models_configured` instead of
/// being rewritten from scratch.
#[test]
fn init_local_existing_yaml_wins_and_gets_models() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let project = tmp.path().join("proj");
    let meme_dir = project.join(".meme");
    std::fs::create_dir_all(&meme_dir).expect("mkdir");
    std::fs::write(
        meme_dir.join("index.yaml"),
        "collections:\n  docs:\n    path: ./docs\n    pattern: \"**/*.md\"\n",
    )
    .expect("seed yaml");
    let env = Environment::default();

    let config_path = Meme::init_local(&env, &project).expect("init");
    assert_eq!(config_path, meme_dir.join("index.yaml"));
    // Collections survive; `models:` was pinned by ensure.
    let yaml = std::fs::read_to_string(&config_path).expect("read");
    assert!(yaml.contains("docs:"), "{yaml}");
    assert!(yaml.contains("embed: "), "{yaml}");
    // No competing `.yml` was created.
    assert!(!meme_dir.join("index.yml").exists());
}

/// An existing `index.yml` is adopted (not overwritten) and gets the
/// models block — upstream `ensureModelsConfiguredForCli` semantics.
#[test]
fn init_local_existing_yml_is_adopted() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let project = tmp.path().join("proj");
    let meme_dir = project.join(".meme");
    std::fs::create_dir_all(&meme_dir).expect("mkdir");
    std::fs::write(meme_dir.join("index.yml"), "collections: {}\n").expect("seed yml");
    let env = Environment::default();

    let config_path = Meme::init_local(&env, &project).expect("init");
    assert_eq!(config_path, meme_dir.join("index.yml"));
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

    let err = Meme::init_local(&env, &project).expect_err("must refuse");
    let Error::LocalInitRefused { reason } = err else {
        panic!("expected LocalInitRefused, got {err}");
    };
    assert!(
        reason.starts_with("Refusing to initialize a local index in $HOME"),
        "{reason}"
    );
    assert!(!project.join(".meme").exists());
}

/// Running `init` twice is a no-op beyond `ensure_models_configured`:
/// same path, same bytes, no second `index.yaml`.
#[test]
fn init_local_is_idempotent() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).expect("mkdir");
    let env = Environment::default();

    let first = Meme::init_local(&env, &project).expect("init");
    let bytes = std::fs::read_to_string(&first).expect("read");
    let second = Meme::init_local(&env, &project).expect("init again");
    assert_eq!(first, second);
    assert_eq!(std::fs::read_to_string(&second).expect("read"), bytes);
    assert!(!project.join(".meme/index.yaml").exists());
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
        let fingerprint = meme::llm::embedding_fingerprint("hf:custom/e.gguf", None);
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
    let gated = Meme::builder(&db)
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
        !after.contains(&format!("embed: {}", meme::llm::DEFAULT_EMBED_MODEL)),
        "{after}"
    );

    // An env override wins runtime resolution under the gate (the
    // declared value is invisible to it) — a vector recorded under the
    // env model now satisfies the pending count.
    let env_fp = meme::llm::embedding_fingerprint("hf:env/model.gguf", None);
    let conn = rusqlite::Connection::open(&db).expect("conn");
    conn.execute(
        "UPDATE content_vectors SET model = 'hf:env/model.gguf', \
         embed_fingerprint = ? WHERE hash = 'h1'",
        [&env_fp],
    )
    .expect("re-tag vector");
    drop(conn);
    let mut env = Environment::default();
    env.meme_embed_model = Some("hf:env/model.gguf".to_owned());
    let env_gated = Meme::builder(&db)
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
