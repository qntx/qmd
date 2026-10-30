//! `qmd bench` CLI surface — ports the upstream `qmd.ts:4788-4810`
//! dispatch: usage on a missing fixture arg, fixture read/parse errors,
//! collection guard errors, and `--format` rejection. Backend-reaching
//! paths are deliberately untested here (vector/hybrid/full load
//! models).

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

use common::CliEnv;

/// A fixture file + its path inside `env.fixtures`.
fn fixture(env: &CliEnv, name: &str, body: &str) -> String {
    let p = env.write_file(name, body);
    p.to_string_lossy().into_owned()
}

const MINIMAL: &str =
    r#"{"queries":[{"id":"q","query":"x","expected_files":[],"expected_in_top_k":1}]}"#;

#[test]
fn bench_usage_no_arg() {
    // Upstream `qmd.ts:4790-4795` — 4 stderr lines, exit 1, no hint.
    let env = CliEnv::new();
    let out = env.run(&["bench"]);
    assert_eq!(out.exit_code, 1);
    assert_eq!(
        out.stderr,
        "Usage: qmd bench <fixture.json> [--json] [-c collection]\n\nRun search quality benchmarks against a fixture file.\nSee src/bench/fixtures/example.json for the fixture format.\n"
    );
    assert!(!out.stderr.contains("doctor"), "{:?}", out.stderr);
}

#[test]
fn bench_missing_file() {
    let env = CliEnv::new();
    let out = env.run(&["bench", "/nonexistent-xyz.json"]);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr
            .contains("Failed to read fixture /nonexistent-xyz.json:"),
        "{:?}",
        out.stderr
    );
}

#[test]
fn bench_invalid_json() {
    let env = CliEnv::new();
    let p = fixture(&env, "bad.json", "not json");
    let out = env.run(&["bench", &p]);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr.contains("Invalid fixture JSON:"),
        "{:?}",
        out.stderr
    );
}

#[test]
fn bench_missing_queries() {
    let env = CliEnv::new();
    let p = fixture(&env, "nq.json", r#"{"version":1}"#);
    let out = env.run(&["bench", &p]);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr
            .contains("Invalid fixture: missing 'queries' array"),
        "{:?}",
        out.stderr
    );
}

#[test]
fn bench_collection_not_found() {
    // A store with collection `other`; the fixture asks for `missing`.
    let env = CliEnv::new();
    env.write_file("docs/a.md", "# Doc\ncontent\n");
    env.write_config(
        "index",
        "collections:\n  other:\n    path: {root}/fixtures/docs\n    pattern: '**/*.md'\n",
    );
    env.update();
    let p = fixture(
        &env,
        "f.json",
        r#"{"collection":"missing","queries":[{"id":"q","query":"x","expected_files":[],"expected_in_top_k":1}]}"#,
    );
    let out = env.run(&["bench", &p]);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr
            .contains("Collection not found: missing\nAvailable: other"),
        "{:?}",
        out.stderr
    );
}

#[test]
fn bench_empty_collection() {
    let env = CliEnv::new();
    // An `empty` collection whose dir has no markdown.
    std::fs::create_dir_all(env.fixtures.join("empty")).unwrap();
    env.write_config(
        "index",
        "collections:\n  empty:\n    path: {root}/fixtures/empty\n    pattern: '**/*.md'\n",
    );
    env.update();
    let p = fixture(
        &env,
        "f.json",
        r#"{"collection":"empty","queries":[{"id":"q","query":"x","expected_files":[],"expected_in_top_k":1}]}"#,
    );
    let out = env.run(&["bench", &p]);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr
            .contains("Collection 'empty' has no indexed documents"),
        "{:?}",
        out.stderr
    );
}

#[test]
fn bench_c_overrides_fixture_collection() {
    let env = CliEnv::new();
    env.write_file("docs/a.md", "# Doc\ncontent\n");
    env.write_config(
        "index",
        "collections:\n  eval-docs:\n    path: {root}/fixtures/docs\n    pattern: '**/*.md'\n",
    );
    env.update();
    let p = fixture(
        &env,
        "f.json",
        r#"{"collection":"eval-docs","queries":[{"id":"q","query":"x","expected_files":[],"expected_in_top_k":1}]}"#,
    );
    // `-c nothere` wins over the fixture's `eval-docs` → error names it.
    let out = env.run(&["bench", &p, "-c", "nothere"]);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr.contains("Collection not found: nothere"),
        "{:?}",
        out.stderr
    );
}

#[test]
fn bench_format_rejected() {
    // Strict parsing (P4-D2): `--format` is not a bench flag → exit 1.
    let env = CliEnv::new();
    let p = fixture(&env, "f.json", MINIMAL);
    let out = env.run(&["bench", &p, "--format", "json"]);
    assert_eq!(out.exit_code, 1);
}

fn by_id(v: &serde_json::Value) -> std::collections::HashMap<String, &serde_json::Value> {
    v["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| (r["id"].as_str().unwrap().to_owned(), r))
        .collect()
}

/// Ports upstream's real-model `bench` run over the eval-docs corpus —
/// indexes the vendored docs, embeds with the cached GGUF triple, then
/// compares `qmd bench --json` against the recorded upstream baseline
/// (`fixtures/bench/baseline-v2.8.3.json`): bm25 + vector `top_files`
/// identical per query, hybrid/full metric fields equal.
#[test]
#[ignore = "requires cached GGUF models"]
fn bench_real_models_matches_baseline() {
    let env = CliEnv::new();
    let corpus =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../qmd/tests/fixtures/eval-docs");
    env.write_config(
        "index",
        &format!(
            "collections:\n  eval-docs:\n    path: {}\n    pattern: '**/*.md'\n",
            corpus.display()
        ),
    );
    env.update();

    let cache = qmd::llm::download::model_cache_dir(&qmd::Environment::from_process());
    let models: Vec<(&str, String)> = [
        (
            "QMD_EMBED_MODEL",
            "hf_ggml-org_embeddinggemma-300M-Q8_0.gguf",
        ),
        (
            "QMD_GENERATE_MODEL",
            "hf_tobil_qmd-query-expansion-1.7B-q4_k_m.gguf",
        ),
        (
            "QMD_RERANK_MODEL",
            "hf_ggml-org_qwen3-reranker-0.6b-q8_0.gguf",
        ),
    ]
    .into_iter()
    .map(|(k, f)| (k, cache.join(f).to_string_lossy().into_owned()))
    .collect();
    let model_env: Vec<(&str, &str)> = models.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let embed = env.run_opts(&["embed"], &env.fixtures, true, &model_env);
    assert_eq!(embed.exit_code, 0, "embed failed: {}", embed.stderr);

    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/bench/example.json")
        .to_string_lossy()
        .into_owned();
    let out = env.run_opts(
        &["bench", &fixture, "--json"],
        &env.fixtures,
        true,
        &model_env,
    );
    assert_eq!(out.exit_code, 0, "bench failed: {}", out.stderr);
    let got: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();

    let baseline: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/bench/baseline-v2.8.3.json")).unwrap();
    let exp = by_id(&baseline);
    for g in got["results"].as_array().unwrap() {
        let id = g["id"].as_str().unwrap();
        let e = exp[id];
        for backend in ["bm25", "vector"] {
            assert_eq!(
                g["backends"][backend]["top_files"], e["backends"][backend]["top_files"],
                "{id}/{backend} top_files"
            );
        }
    }
    // hybrid/full run query expansion, which samples (temp 0.7, top_k 20,
    // top_p 0.8, time-seeded — upstream llm.ts:1584-1594 and ours alike),
    // so per-query results vary run to run on both sides: upstream's own
    // re-run on 2026-10-30 scored hybrid R@1 0.9 against the baseline's
    // 1.0. Only bm25/vector are deterministic; hybrid/full are held to the
    // research-report §10.15 quality bar on the summary instead.
    for backend in ["hybrid", "full"] {
        let s = &got["summary"][backend];
        for k in ["avg_precision", "avg_recall"] {
            let v = s[k].as_f64().unwrap();
            assert!(v >= 0.9, "{backend}/{k} = {v} below the 0.9 quality bar");
        }
    }
    // Warm latency, printed for the runbook.
    for (name, s) in got["summary"].as_object().unwrap() {
        eprintln!("warm {name}: avg_latency_ms={}", s["avg_latency_ms"]);
    }
}

#[test]
fn bench_does_not_write_models_to_config() {
    // R4(a): index.yml with a collection and NO `models:` block —
    // running bench to the (empty-collection) guard must leave the
    // file byte-identical; upstream's SDK `createStore` never writes.
    let env = CliEnv::new();
    std::fs::create_dir_all(env.fixtures.join("empty")).unwrap();
    let cfg = env.write_config(
        "index",
        "collections:\n  empty:\n    path: {root}/fixtures/empty\n    pattern: '**/*.md'\n",
    );
    let before = std::fs::read_to_string(&cfg).unwrap();
    let p = fixture(
        &env,
        "f.json",
        r#"{"collection":"empty","queries":[{"id":"q","query":"x","expected_files":[],"expected_in_top_k":1}]}"#,
    );
    let out = env.run(&["bench", &p]);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr
            .contains("Collection 'empty' has no indexed documents"),
        "{:?}",
        out.stderr
    );
    assert_eq!(
        std::fs::read_to_string(&cfg).unwrap(),
        before,
        "index.yml mutated by bench"
    );
}

#[test]
fn bench_without_config_reports_no_documents() {
    // R4(b): no index.yml at all → the guard's "nothing indexed"
    // message, and the file is not created either.
    let env = CliEnv::new();
    let cfg = env.config_dir.join("index.yml");
    assert!(!cfg.exists());
    let p = fixture(&env, "f.json", MINIMAL);
    let out = env.run(&["bench", &p]);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr.contains("No indexed documents found."),
        "{:?}",
        out.stderr
    );
    assert!(!cfg.exists(), "index.yml created by bench");
}
