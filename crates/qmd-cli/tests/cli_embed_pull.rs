//! Ports of upstream `test/cli.test.ts` `CLI Embed` describe (the two
//! `runQmd` cases; `resolveEmbedModelForCli` env-preference is covered
//! at library level by `qmd/tests/qmd_api.rs::ensure_models_configured_*`)
//! plus CLI-path coverage for `qmd embed`'s remaining upstream behaviors
//! (`vectorIndex`, cli/qmd.ts:2189-2320) and `qmd pull` (`pullModels`,
//! llm.ts:523-580).
//!
//! Model-dependent paths stay deterministic: `embed` on a pending index
//! is exercised under `CI=1`, where the lazy backend fails fast with the
//! upstream `_ciMode` message before any download; `pull` is exercised
//! with local-path model URIs, which upstream resolves as plain files —
//! no `hf:` download is ever attempted.
//!
//! Upstream differences noted inline: a busy embed lock is an OS
//! `try_lock` on `<db>.embed.lock` (P4-D4), and `--chunk-strategy`
//! reaches our validator inside the command while upstream's
//! `parseCLI`-time throw produces a Bun dump (golden `record`).

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

use std::fs::OpenOptions;

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

// ── CLI Embed (upstream cli.test.ts:316-355) ─────────────────────────

#[test]
fn rejects_invalid_max_docs_per_batch() {
    let env = CliEnv::new();
    env.write_config("index", "collections: {}\n");
    let out = env.run(&["embed", "--max-docs-per-batch", "0"]);
    assert_eq!(out.exit_code, 1);
    assert!(out.stderr.contains("maxDocsPerBatch"), "{}", out.stderr);
}

#[test]
fn rejects_invalid_max_batch_mb() {
    let env = CliEnv::new();
    env.write_config("index", "collections: {}\n");
    let out = env.run(&["embed", "--max-batch-mb", "0"]);
    assert_eq!(out.exit_code, 1);
    assert!(out.stderr.contains("maxBatchBytes"), "{}", out.stderr);
}

#[test]
fn rejects_invalid_timeout() {
    let env = CliEnv::new();
    env.write_config("index", "collections: {}\n");
    for value in ["-1", "abc"] {
        let out = env.run(&["embed", "--timeout", value]);
        assert_eq!(out.exit_code, 1, "--timeout {value}");
        assert!(
            out.stderr
                .contains("--timeout must be a non-negative number of minutes"),
            "{}",
            out.stderr
        );
    }
}

// ── vectorIndex ordering (cli/qmd.ts:2189-2320) ──────────────────────

#[test]
fn embed_rejects_unknown_collection_without_doctor_hint() {
    // Upstream `resolveCollectionFilter` prints + `process.exit(1)` —
    // no `exitWithError`, so no doctor hint (qmd.ts:4716 → 2165-2180).
    let env = env_with_fixtures();
    let out = env.run(&["embed", "-c", "missing"]);
    assert_eq!(out.exit_code, 1);
    assert!(out.stderr.contains("Collection not found: missing"));
    assert!(!out.stderr.contains("qmd doctor"), "{}", out.stderr);
}

#[test]
fn embed_empty_index_reports_nothing_pending() {
    let env = CliEnv::new();
    env.write_config("index", "collections: {}\n");
    env.update();
    let out = env.run(&["embed"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout
            .contains("All content hashes already have embeddings"),
        "{}",
        out.stdout
    );
}

#[test]
fn embed_force_empty_index_skips_after_model_banner() {
    // `vectorIndex`: force prints the banner and reaches the model line,
    // then `generateEmbeddings` finds no non-empty docs — the model is
    // resolved lazily, so nothing downloads (cli/qmd.ts:2267-2286).
    let env = CliEnv::new();
    env.write_config("index", "collections: {}\n");
    env.update();
    let out = env.run(&["embed", "--force"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout
            .contains("Force re-indexing: clearing all vectors"),
        "{}",
        out.stdout
    );
    assert!(out.stdout.contains("Model: "), "{}", out.stdout);
    assert!(
        out.stdout.contains("No non-empty documents to embed"),
        "{}",
        out.stdout
    );
}

#[test]
fn embed_pending_work_reaches_model_resolution() {
    // Pending docs → `generateEmbeddings` → tokenizer init →
    // `resolveModelFile` — upstream's `_ciMode` gate is inside
    // `embed`/`embedBatch` (llm.ts:1503), NOT `tokenize` (llm.ts:1410),
    // so chunking reaches model resolution even under CI; a missing
    // local path is the deterministic way to observe that ordering
    // without a download.
    let env = env_with_fixtures();
    let missing = env.root.join("nope.gguf").to_string_lossy().into_owned();
    let out = env.run_opts(
        &["embed"],
        &env.fixtures,
        true,
        &[("QMD_EMBED_MODEL", missing.as_str())],
    );
    assert_eq!(out.exit_code, 1, "stderr: {}", out.stderr);
    assert!(
        out.stderr.contains("No model file found"),
        "stderr: {}",
        out.stderr
    );
}

#[test]
fn embed_skips_when_lock_is_busy() {
    // Upstream `tryAcquireEmbedLock` busy path (embed-lock.ts:65-97):
    // quiet skip on stdout, exit 0. Our lock is an OS `try_lock` on
    // `<db>.embed.lock` (P4-D4) — holding it here blocks the subprocess.
    let env = env_with_fixtures();
    let lock_path = env.db_path.with_file_name("index.sqlite.embed.lock");
    let lock_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&lock_path)
        .unwrap();
    lock_file.try_lock().unwrap();

    let out = env.run(&["embed"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout
            .contains("Another embed process is already running. Skipping."),
        "{}",
        out.stdout
    );
    drop(lock_file);
}

// ── qmd pull (llm.ts:523-580) ────────────────────────────────────────

/// Write a fake-but-valid GGUF blob (magic + padding — `inspectGgufFile`
/// only checks the 4-byte magic, llm.ts:361-370) and return its path.
fn fake_gguf(env: &CliEnv, name: &str) -> std::path::PathBuf {
    let path = env.root.join(name);
    let mut blob = b"GGUF".to_vec();
    blob.extend(std::iter::repeat_n(0u8, 64));
    std::fs::write(&path, blob).unwrap();
    path
}

#[test]
fn pull_with_local_model_paths_is_offline() {
    // Local-path URIs resolve via `path.resolve(cacheDir, spec)`
    // (resolveModelFile.js) — no network. Three distinct models → three
    // result lines; the config pin lands before downloading.
    let env = CliEnv::new();
    env.write_config("index", "collections: {}\n");
    let embed = fake_gguf(&env, "embed.gguf");
    let generate = fake_gguf(&env, "generate.gguf");
    let rerank = fake_gguf(&env, "rerank.gguf");
    let extra: Vec<(&str, String)> = [
        ("QMD_EMBED_MODEL", embed.to_string_lossy().into_owned()),
        (
            "QMD_GENERATE_MODEL",
            generate.to_string_lossy().into_owned(),
        ),
        ("QMD_RERANK_MODEL", rerank.to_string_lossy().into_owned()),
    ]
    .into_iter()
    .collect();
    let refs: Vec<(&str, &str)> = extra.iter().map(|(k, v)| (*k, v.as_str())).collect();

    let out = env.run_opts(&["pull"], &env.fixtures, true, &refs);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Pulling models"), "{}", out.stdout);
    for model in [&embed, &generate, &rerank] {
        let display = model.to_string_lossy().into_owned();
        assert!(
            out.stdout.contains(&display),
            "{}\nmodel: {}",
            out.stdout,
            display
        );
        assert!(out.stdout.contains("cached/checked"), "{}", out.stdout);
    }

    // `ensureModelsConfiguredForCli` pinned the env-resolved triple.
    let cfg = std::fs::read_to_string(env.config_dir.join("index.yml")).unwrap();
    assert!(cfg.contains("models:"), "{cfg}");
    assert!(cfg.contains(&embed.to_string_lossy().into_owned()), "{cfg}");
}

#[test]
fn pull_errors_on_missing_local_model_file() {
    // `resolveModelFile` for a non-`hf:` path that doesn't exist → the
    // upstream `No model file found at "..."` error via `exitWithError`.
    let env = CliEnv::new();
    env.write_config("index", "collections: {}\n");
    let missing = env.root.join("nope.gguf").to_string_lossy().into_owned();
    let out = env.run_opts(
        &["pull"],
        &env.fixtures,
        true,
        &[
            ("QMD_EMBED_MODEL", missing.as_str()),
            ("QMD_GENERATE_MODEL", missing.as_str()),
            ("QMD_RERANK_MODEL", missing.as_str()),
        ],
    );
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr.contains("No model file found"),
        "stderr: {}",
        out.stderr
    );
}

#[test]
fn pull_refresh_rechecks_local_models() {
    // `--refresh` on a local path only sweeps cache-dir matches — the
    // file itself is re-validated and reported as cached/checked.
    let env = CliEnv::new();
    env.write_config("index", "collections: {}\n");
    let model = fake_gguf(&env, "model.gguf");
    let model_str = model.to_string_lossy().into_owned();
    let out = env.run_opts(
        &["pull", "--refresh"],
        &env.fixtures,
        true,
        &[
            ("QMD_EMBED_MODEL", model_str.as_str()),
            ("QMD_GENERATE_MODEL", model_str.as_str()),
            ("QMD_RERANK_MODEL", model_str.as_str()),
        ],
    );
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Pulling models"), "{}", out.stdout);
    assert!(out.stdout.contains(&model_str), "{}", out.stdout);
}
