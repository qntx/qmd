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

//! Path helpers and YAML config round-trip tests.
//!
//! Ports the semantic coverage of upstream `test/collections-config` and
//! `test/local-config` plus the Rust-specific round-trip requirements
//! (unknown keys, key order, editor_uri aliases, empty file).

use std::path::{Path, PathBuf};

use qmd::config::{self, Collection, Config, ModelsConfig, YamlValue};
use qmd::{Environment, paths};

fn env(fields: impl FnOnce(&mut Environment)) -> Environment {
    let mut e = Environment::default();
    fields(&mut e);
    e
}

#[test]
fn config_dir_prefers_qmd_config_dir() {
    let e = env(|e| {
        e.qmd_config_dir = Some(PathBuf::from("/custom"));
        e.xdg_config_home = Some(PathBuf::from("/xdg"));
        e.home = Some(PathBuf::from("/home"));
    });
    assert_eq!(paths::config_dir(&e), PathBuf::from("/custom"));
}

#[test]
fn config_dir_uses_xdg_then_home() {
    let e = env(|e| {
        e.xdg_config_home = Some(PathBuf::from("/xdg"));
        e.home = Some(PathBuf::from("/home"));
    });
    assert_eq!(paths::config_dir(&e), PathBuf::from("/xdg/qmd"));

    let e = env(|e| e.home = Some(PathBuf::from("/home")));
    assert_eq!(paths::config_dir(&e), PathBuf::from("/home/.config/qmd"));
}

#[test]
fn default_db_path_prefers_index_path() {
    let e = env(|e| {
        e.index_path = Some(PathBuf::from("/idx/my.sqlite"));
        e.xdg_cache_home = Some(PathBuf::from("/cache"));
    });
    assert_eq!(
        paths::default_db_path(&e, "main"),
        PathBuf::from("/idx/my.sqlite")
    );
}

#[test]
fn default_db_path_uses_rs_suffix() {
    // Rust indexes get `<index>-rs.sqlite` so they never collide with an
    // upstream `<index>.sqlite` file.
    let e = env(|e| e.xdg_cache_home = Some(PathBuf::from("/cache")));
    assert_eq!(
        paths::default_db_path(&e, "main"),
        PathBuf::from("/cache/qmd/main-rs.sqlite")
    );
}

#[test]
fn find_local_config_walks_upward() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let qmd_dir = tmp.path().join("proj/sub/.qmd");
    std::fs::create_dir_all(&qmd_dir).expect("mkdir");
    // .qmd/index.yml lives in proj/, not in proj/sub/
    let root_cfg = tmp.path().join("proj/.qmd");
    std::fs::create_dir_all(&root_cfg).expect("mkdir");
    let cfg_file = root_cfg.join("index.yml");
    std::fs::write(&cfg_file, "collections: {}\n").expect("write");

    let start = tmp.path().join("proj/sub/deep");
    std::fs::create_dir_all(&start).expect("mkdir");
    assert_eq!(
        paths::find_local_config(&start).as_deref(),
        Some(cfg_file.as_path())
    );
    assert_eq!(paths::find_local_config(tmp.path()), None);
}

#[test]
fn find_local_config_prefers_yaml_over_yml() {
    // Upstream checks `.qmd/index.yaml` first, then `.qmd/index.yml`.
    let tmp = tempfile::tempdir().expect("tempdir");
    let qmd_dir = tmp.path().join("proj/.qmd");
    std::fs::create_dir_all(&qmd_dir).expect("mkdir");
    let yaml = qmd_dir.join("index.yaml");
    let yml = qmd_dir.join("index.yml");
    std::fs::write(&yaml, "collections: {}\n").expect("write yaml");
    std::fs::write(&yml, "collections: {}\n").expect("write yml");

    let start = tmp.path().join("proj").join("deeper");
    std::fs::create_dir_all(&start).expect("mkdir");
    assert_eq!(
        paths::find_local_config(&start).as_deref(),
        Some(yaml.as_path())
    );
}

#[test]
fn normalize_index_name_cases() {
    let cwd = Path::new("/Users/me/work");
    assert_eq!(paths::normalize_index_name("myindex", cwd), "myindex");
    assert_eq!(
        paths::normalize_index_name("./notes", cwd),
        "Users_me_work_notes"
    );
    assert_eq!(paths::normalize_index_name("/abs/path", cwd), "abs_path");
    assert_eq!(
        paths::normalize_index_name("a/b/c", cwd),
        "Users_me_work_a_b_c"
    );
    assert_eq!(
        paths::normalize_index_name("C:\\Users\\me", cwd),
        "C_Users_me"
    );
}

#[test]
fn expand_home_handles_tilde() {
    let e = env(|e| e.home = Some(PathBuf::from("/home/me")));
    assert_eq!(
        paths::expand_home("~/docs", &e),
        PathBuf::from("/home/me/docs")
    );
    assert_eq!(paths::expand_home("~", &e), PathBuf::from("/home/me"));
    assert_eq!(paths::expand_home("/abs/x", &e), PathBuf::from("/abs/x"));
}

#[test]
fn busy_timeout_parsing() {
    assert_eq!(
        paths::busy_timeout_ms(&Environment::default()),
        paths::DEFAULT_BUSY_TIMEOUT_MS
    );
    let with = |v: &str| env(|e| e.qmd_sqlite_busy_timeout = Some(v.to_owned()));
    assert_eq!(paths::busy_timeout_ms(&with("0")), 0);
    assert_eq!(paths::busy_timeout_ms(&with("5000")), 5000);
    assert_eq!(paths::busy_timeout_ms(&with("1500.9")), 1500);
    assert_eq!(
        paths::busy_timeout_ms(&with("abc")),
        paths::DEFAULT_BUSY_TIMEOUT_MS
    );
    assert_eq!(
        paths::busy_timeout_ms(&with("-5")),
        paths::DEFAULT_BUSY_TIMEOUT_MS
    );
    assert_eq!(
        paths::busy_timeout_ms(&with("")),
        paths::DEFAULT_BUSY_TIMEOUT_MS
    );
}

// ---- config round trip ----

#[test]
fn empty_and_null_configs_load_as_empty() {
    let tmp = tempfile::tempdir().expect("tempdir");
    for (name, body) in [
        ("missing.yml", ""),
        ("empty.yml", ""),
        ("blank.yml", "   \n"),
        ("null.yml", "null\n"),
        ("tilde.yml", "~\n"),
    ] {
        let p = tmp.path().join(name);
        if name != "missing.yml" {
            std::fs::write(&p, body).expect("write");
        }
        let cfg = config::load(&p).expect("load");
        assert!(cfg.collections.is_empty(), "{name}");
    }
}

#[test]
fn config_round_trip_preserves_unknown_order_aliases() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("qmd.yml");
    let src = "\
global_context: global words
editorUri: obsidian://open?vault=x&file={file}
unknown_top: keepme
models:
  embed: my-embed
collections:
  zeta:
    path: /data/z
    ignore:
      - drafts/**
    custom_key: 42
  '2024':
    path: /data/y
    context:
      '2024': year context
      /sub: sub context
    includeByDefault: false
  alpha:
    path: /data/a
    pattern: '**/*.txt'
";
    std::fs::write(&path, src).expect("write");
    let cfg = config::load(&path).expect("load");

    // Key order inside `collections` preserved.
    let names: Vec<_> = cfg.collections.keys().collect();
    assert_eq!(names, ["zeta", "2024", "alpha"]);

    // Numeric collection and context keys coerced to strings.
    let y2024 = &cfg.collections["2024"];
    assert_eq!(
        y2024
            .context
            .as_ref()
            .and_then(|c| c.get("2024"))
            .map(String::as_str),
        Some("year context")
    );

    // Alias spelling read.
    assert_eq!(
        cfg.editor_uri_value(),
        Some("obsidian://open?vault=x&file={file}")
    );

    // Unknown keys landed in `extra`.
    assert!(cfg.extra.contains_key("unknown_top"));
    assert!(cfg.collections["zeta"].extra.contains_key("custom_key"));

    // Serialize, reload, compare structurally.
    config::save(&path, &cfg).expect("save");
    let reloaded = config::load(&path).expect("reload");
    assert_eq!(reloaded, cfg);

    let emitted = std::fs::read_to_string(&path).expect("read");
    // Collection order still zeta, 2024, alpha in the emitted file.
    let p_zeta = emitted.find("zeta:").expect("zeta");
    let p_2024 = emitted
        .find("'2024':")
        .or_else(|| emitted.find("2024:"))
        .expect("2024");
    let p_alpha = emitted.find("alpha:").expect("alpha");
    assert!(p_zeta < p_2024 && p_2024 < p_alpha);
    // Unknown keys still present.
    assert!(emitted.contains("unknown_top"));
    assert!(emitted.contains("custom_key"));
    // Alias spelling preserved (editorUri, not editor_uri).
    assert!(emitted.contains("editorUri:"));
}

#[test]
fn editor_uri_precedence() {
    let mut cfg = Config::default();
    cfg.editor_uri_kebab = Some("kebab".to_owned());
    cfg.editor_uri_camel = Some("camel".to_owned());
    cfg.editor_uri_template = Some("template".to_owned());
    cfg.editor_uri = Some("snake".to_owned());
    assert_eq!(cfg.editor_uri_value(), Some("snake"));
    cfg.editor_uri = None;
    assert_eq!(cfg.editor_uri_value(), Some("template"));
    cfg.editor_uri_template = None;
    assert_eq!(cfg.editor_uri_value(), Some("camel"));
    cfg.editor_uri_camel = None;
    assert_eq!(cfg.editor_uri_value(), Some("kebab"));
}

#[test]
fn models_embed_pooling_roundtrip() {
    let mut cfg = Config::default();
    cfg.models = Some(ModelsConfig {
        embed_pooling: Some(qmd::EmbedPooling::FullSequence),
        ..ModelsConfig::default()
    });
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("qmd.yml");
    config::save(&path, &cfg).expect("save");
    let emitted = std::fs::read_to_string(&path).expect("read");
    assert!(emitted.contains("embed_pooling: full_sequence"));
    assert_eq!(config::load(&path).expect("load"), cfg);
}

#[test]
fn collection_defaults() {
    let coll = Collection {
        path: "/x".to_owned(),
        ..Collection::default()
    };
    assert_eq!(coll.pattern_or_default(), config::DEFAULT_PATTERN);
    assert!(coll.include_by_default());
    assert!(coll.ignore_patterns().is_empty());
}

#[test]
#[ignore = "requires bun and an upstream checkout (set QMD_UPSTREAM)"]
fn config_interop_upstream_round_trip() {
    // AC4: YAML written by Rust must load under upstream `loadConfig`
    // and survive upstream `saveConfig`; upstream's output must load
    // under Rust — same file shared by both implementations.
    //   QMD_UPSTREAM=/path/to/tobi/qmd \
    //     cargo test -p qmd --test paths_config -- --ignored config_interop
    let upstream = std::env::var("QMD_UPSTREAM").expect("QMD_UPSTREAM not set");
    let tmp = tempfile::tempdir().expect("tempdir");

    // 1. Rust writes a config exercising every feature upstream must
    //    tolerate (unknown keys, models, collection extras).
    let rust_yaml = tmp.path().join("rust.yaml");
    let mut cfg = Config::default();
    let mut coll = Collection {
        path: "/data/docs".to_owned(),
        pattern: Some("**/*.md".to_owned()),
        ignore: Some(vec!["drafts/**".to_owned()]),
        include_by_default: Some(false),
        ..Collection::default()
    };
    coll.context = Some(indexmap::IndexMap::from([(
        "/".to_owned(),
        "ctx".to_owned(),
    )]));
    coll.extra
        .insert("custom_key".to_owned(), YamlValue::from(42));
    cfg.collections.insert("docs".to_owned(), coll);
    cfg.models = Some(ModelsConfig {
        embed_pooling: Some(qmd::EmbedPooling::FullSequence),
        ..ModelsConfig::default()
    });
    cfg.extra
        .insert("unknown_top".to_owned(), YamlValue::from("keepme"));
    config::save(&rust_yaml, &cfg).expect("save rust yaml");

    // 2. Upstream loads it and re-serializes via its own saveConfig.
    let upstream_yaml = tmp.path().join("upstream.yaml");
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/interop_yaml.ts");
    let out = std::process::Command::new("bun")
        .arg(script)
        .arg(&rust_yaml)
        .arg(&upstream_yaml)
        .env("QMD_UPSTREAM", &upstream)
        .output()
        .expect("spawn bun");
    assert!(
        out.status.success(),
        "upstream loadConfig failed on Rust YAML: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // 3. Upstream's parse sees our data; its re-serialization keeps the
    //    keys it does not model (upstream preserves unknown keys —
    //    p1-config-probe).
    let seen: serde_json::Value = serde_json::from_slice(&out.stdout).expect("upstream json");
    assert_eq!(seen["collections"]["docs"]["path"], "/data/docs");
    let emitted = std::fs::read_to_string(&upstream_yaml).expect("upstream yaml");
    assert!(
        emitted.contains("embed_pooling"),
        "upstream dropped models.embed_pooling:\n{emitted}"
    );
    assert!(emitted.contains("unknown_top"));
    assert!(emitted.contains("custom_key"));

    // 4. Rust loads upstream's output back without losing anything.
    let back = config::load(&upstream_yaml).expect("rust reload");
    assert_eq!(back.collections["docs"].path, "/data/docs");
    assert_eq!(
        back.collections["docs"].ignore,
        Some(vec!["drafts/**".to_owned()])
    );
    assert_eq!(
        back.extra.get("unknown_top"),
        Some(&YamlValue::from("keepme"))
    );
    assert_eq!(
        back.models.as_ref().and_then(|m| m.embed_pooling),
        Some(qmd::EmbedPooling::FullSequence)
    );
}
