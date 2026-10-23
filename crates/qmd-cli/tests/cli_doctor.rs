//! Ports of upstream `test/cli.test.ts` `CLI Status Command`'s `qmd
//! doctor` block (cli.test.ts:702-940 → `showDoctor`, cli/qmd.ts:
//! 4205-4323). The two `status` tests of that describe live in
//! `cli_maintenance.rs`.
//!
//! Setup differences from upstream: collections/index are populated
//! through the library (`CliEnv::update`), and the model cache is
//! `$XDG_CACHE_HOME/qmd/models` — both equivalent to upstream's
//! `createIsolatedTestEnv` + `collection add`.
//!
//! All cases are model-free: `vectors_vec` never exists here, so the
//! sample probe reports "no vector table" and the lazy embedder is
//! never asked to embed. `QMD_DOCTOR_DEVICE_PROBE=0` (harness default)
//! skips the native device probe.

#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::missing_assert_message,
    clippy::panic,
    reason = "idiomatic test-code patterns"
)]

mod common;

use common::{CliEnv, collection_yaml, fixtures_corpus};

fn env_with_fixtures() -> CliEnv {
    let env = CliEnv::new();
    fixtures_corpus(&env);
    env.write_config(
        "index",
        &collection_yaml("fixtures", &env.fixtures, "**/*.md"),
    );
    env.update();
    env
}

/// `$XDG_CACHE_HOME/qmd/models` — `model_cache_dir` (download.rs:372).
fn model_cache(env: &CliEnv) -> std::path::PathBuf {
    env.root.join("xdg-cache").join("qmd").join("models")
}

// ── Upstream ports (cli.test.ts:707-939) ─────────────────────────────

#[test]
fn doctor_reports_core_index_health_checks() {
    let env = env_with_fixtures();
    let out = env.run(&["doctor"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    for needle in [
        "QMD Doctor",
        "SQLite runtime",
        "sqlite-vec",
        "environment overrides",
        "INDEX_PATH",
        "overrides the SQLite index path",
        "QMD_CONFIG_DIR",
        "overrides the QMD config directory",
        "model defaults",
        "model cache",
        "device mode",
        "device probe",
        "embedding freshness",
        "embedding fingerprints",
        "embedding vector sample",
        "please run qmd embed again",
    ] {
        assert!(
            out.stdout.contains(needle),
            "missing '{needle}'\n{}",
            out.stdout
        );
    }

    // `ensureModelsConfigured` pinned the resolved triple into index.yml.
    let cfg = std::fs::read_to_string(env.config_dir.join("index.yml")).unwrap();
    assert!(cfg.contains("models:"), "{cfg}");
    assert!(cfg.contains("embed:"), "{cfg}");
    assert!(cfg.contains("generate:"), "{cfg}");
    assert!(cfg.contains("rerank:"), "{cfg}");
}

#[test]
fn doctor_warns_when_no_collections_configured() {
    let env = CliEnv::new();
    env.write_config("index", "collections: {}\n");
    let out = env.run(&["doctor"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("index config"), "{}", out.stdout);
    assert!(
        out.stdout.contains("no collections configured"),
        "{}",
        out.stdout
    );
    assert!(
        out.stdout.contains("qmd collection add ."),
        "{}",
        out.stdout
    );
}

#[test]
fn doctor_reports_invalid_index_yml_without_crashing() {
    let env = CliEnv::new();
    env.write_config("index", "collections:\n  bad: [unterminated\n");
    let out = env.run(&["doctor"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("index config"), "{}", out.stdout);
    assert!(
        out.stdout.contains("invalid index.yml at"),
        "{}",
        out.stdout
    );
    assert!(
        out.stdout.contains(
            &env.config_dir
                .join("index.yml")
                .to_string_lossy()
                .into_owned()
        ),
        "{}",
        out.stdout
    );
    assert!(out.stdout.contains("fix the YAML"), "{}", out.stdout);
}

#[test]
fn doctor_warns_when_configured_models_differ_from_defaults() {
    let env = CliEnv::new();
    env.write_config(
        "index",
        &format!(
            "collections: {{}}\nmodels:\n  embed: hf:example/custom-embed/custom.gguf\n  generate: {}\n  rerank: {}\n",
            qmd::llm::DEFAULT_GENERATE_MODEL,
            qmd::llm::DEFAULT_RERANK_MODEL,
        ),
    );
    let out = env.run(&["doctor"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("model defaults"), "{}", out.stdout);
    assert!(
        out.stdout.contains("non-default model configuration"),
        "{}",
        out.stdout
    );
    assert!(
        out.stdout
            .contains("index hf:example/custom-embed/custom.gguf"),
        "{}",
        out.stdout
    );
    assert!(out.stdout.contains("might be ok"), "{}", out.stdout);
    assert!(out.stdout.contains("qmd pull"), "{}", out.stdout);
}

#[test]
fn doctor_identifies_cached_non_gguf_model_files() {
    // The model URI's filename is probed inside the model cache; an
    // HTML body fails the GGUF magic check (llm.ts:361-370, 3810-3835).
    let env = CliEnv::new();
    let model = "hf:example/custom-model/custom.gguf";
    env.write_config(
        "index",
        &format!(
            "collections: {{}}\nmodels:\n  embed: {model}\n  generate: {model}\n  rerank: {model}\n"
        ),
    );
    let cache = model_cache(&env);
    std::fs::create_dir_all(&cache).unwrap();
    std::fs::write(
        cache.join("custom.gguf"),
        "<!doctype html><html>blocked</html>",
    )
    .unwrap();

    let out = env.run(&["doctor"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("model cache"), "{}", out.stdout);
    assert!(out.stdout.contains("invalid 1"), "{}", out.stdout);
    assert!(
        out.stdout.contains("HTML page, not a GGUF model"),
        "{}",
        out.stdout
    );
    assert!(out.stdout.contains("qmd pull --refresh"), "{}", out.stdout);
}

#[test]
fn doctor_ignores_etag_sidecars_beside_valid_cached_model() {
    // Upstream #812: `<file>.gguf.etag` sorts before the blob but only
    // real `.gguf` files may be inspected.
    let env = CliEnv::new();
    let model = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
    env.write_config(
        "index",
        &format!(
            "collections: {{}}\nmodels:\n  embed: {model}\n  generate: {model}\n  rerank: {model}\n"
        ),
    );
    let cache = model_cache(&env);
    std::fs::create_dir_all(&cache).unwrap();
    std::fs::write(
        cache.join("embeddinggemma-300M-Q8_0.gguf.etag"),
        "\"9d3a1b2c3d4e5\"\n",
    )
    .unwrap();
    let mut blob = b"GGUF".to_vec();
    blob.extend(std::iter::repeat_n(0u8, 60));
    std::fs::write(
        cache.join("hf_ggml-org_embeddinggemma-300M-Q8_0.gguf"),
        blob,
    )
    .unwrap();

    let out = env.run(&["doctor"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("model cache"), "{}", out.stdout);
    assert!(
        out.stdout.contains("downloaded and valid GGUF"),
        "{}",
        out.stdout
    );
    assert!(!out.stdout.contains("not valid GGUF"), "{}", out.stdout);
    assert!(!out.stdout.contains(".etag"), "{}", out.stdout);
}

#[test]
fn doctor_says_when_models_are_overridden_by_env() {
    let env = CliEnv::new();
    env.write_config("index", "collections: {}\n");
    let custom = "hf:example/env-embed/custom.gguf";
    let out = env.run_opts(
        &["doctor"],
        &env.fixtures,
        true,
        &[("QMD_EMBED_MODEL", custom)],
    );
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("model defaults"), "{}", out.stdout);
    assert!(
        out.stdout
            .contains(&format!("env QMD_EMBED_MODEL={custom}")),
        "{}",
        out.stdout
    );
    assert!(out.stdout.contains("might be ok"), "{}", out.stdout);
    assert!(
        out.stdout.contains("environment overrides"),
        "{}",
        out.stdout
    );
    assert!(
        out.stdout.contains(&format!("QMD_EMBED_MODEL={custom}")),
        "{}",
        out.stdout
    );
    assert!(
        out.stdout.contains("sets the active embed model"),
        "{}",
        out.stdout
    );
}

#[test]
fn doctor_shows_cpu_forced_device_mode() {
    let env = CliEnv::new();
    env.write_config("index", "collections: {}\n");
    let out = env.run_opts(&["doctor"], &env.fixtures, true, &[("QMD_FORCE_CPU", "1")]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("QMD_FORCE_CPU=1"), "{}", out.stdout);
    assert!(
        out.stdout
            .contains("forces llama.cpp to bypass GPU backends"),
        "{}",
        out.stdout
    );
    assert!(
        out.stdout
            .contains("device mode: CPU forced (QMD_FORCE_CPU)"),
        "{}",
        out.stdout
    );
}

#[test]
fn doctor_lists_known_environment_overrides_and_consequences() {
    // `INDEX_PATH`, `QMD_CONFIG_DIR`, `XDG_*`, `NO_COLOR` and
    // `QMD_DOCTOR_DEVICE_PROBE` come from the harness env — upstream's
    // list also includes them and asserts every name appears.
    let env = CliEnv::new();
    env.write_config("index", "collections: {}\n");
    let extra: Vec<(&str, &str)> = vec![
        ("QMD_FORCE_CPU", "1"),
        ("QMD_LLAMA_GPU", "metal"),
        ("QMD_EMBED_PARALLELISM", "2"),
        ("QMD_EXPAND_CONTEXT_SIZE", "4096"),
        ("QMD_RERANK_CONTEXT_SIZE", "8192"),
        ("QMD_EMBED_CONTEXT_SIZE", "1024"),
        ("QMD_EDITOR_URI", "vscode://file/{file}:{line}:{col}"),
        ("QMD_SKILLS_DIR", "/tmp/qmd-skills"),
        ("QMD_METAL_KEEP_RESIDENCY", "1"),
        ("CI", "1"),
        ("HF_ENDPOINT", "https://hf-mirror.com"),
        ("WSL_DISTRO_NAME", "Ubuntu"),
        ("WSL_INTEROP", "1"),
    ];
    let out = env.run_opts(&["doctor"], &env.fixtures, true, &extra);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let harness_names = [
        "INDEX_PATH",
        "QMD_CONFIG_DIR",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "NO_COLOR",
        "QMD_DOCTOR_DEVICE_PROBE",
    ];
    for name in extra.iter().map(|(n, _)| *n).chain(harness_names) {
        assert!(
            out.stdout.contains(name),
            "missing '{name}'\n{}",
            out.stdout
        );
    }
    for needle in [
        "forces llama.cpp to bypass GPU backends",
        "moves the default index cache",
        "disables real LLM operations",
        "changes Hugging Face download endpoint",
    ] {
        assert!(
            out.stdout.contains(needle),
            "missing '{needle}'\n{}",
            out.stdout
        );
    }
}

#[test]
fn doctor_flags_mixed_embedding_fingerprints() {
    // Upstream inserts `content_vectors` rows directly (better-sqlite3);
    // the fingerprints check is pure SQL — no embedder is touched
    // because `vectors_vec` does not exist (cli/qmd.ts:4001-4066).
    let env = env_with_fixtures();
    let conn = rusqlite::Connection::open(&env.db_path).unwrap();
    let model: String = qmd::llm::DEFAULT_EMBED_MODEL.to_owned();
    let now = "2026-01-01T00:00:00.000Z";
    conn.execute(
        "INSERT OR REPLACE INTO content_vectors (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at) \
         SELECT hash, 0, 0, ?1, 'stale1', 2, ?2 FROM documents WHERE active = 1 LIMIT 1",
        rusqlite::params![model, now],
    )
    .unwrap();
    conn.execute(
        "INSERT OR REPLACE INTO content_vectors (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at) \
         SELECT hash, 1, 1, ?1, 'stale2', 2, ?2 FROM documents WHERE active = 1 LIMIT 1",
        rusqlite::params![model, now],
    )
    .unwrap();
    drop(conn);

    let out = env.run(&["doctor"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("embedding fingerprints"),
        "{}",
        out.stdout
    );
    assert!(
        out.stdout.contains("mixed named embedding fingerprints"),
        "{}",
        out.stdout
    );
    assert!(out.stdout.contains("stale1"), "{}", out.stdout);
}
