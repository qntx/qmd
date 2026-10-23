//! CLI-path coverage of `qmd vsearch`/`vector-search` and
//! `qmd query`/`deep-search` — upstream `vectorSearch` (cli/qmd.ts
//! `case "vsearch"`) and `querySearch` (`case "query"`), complementing
//! `cli_golden.rs`'s byte comparisons with behavioral assertions that
//! don't depend on the golden corpus.
//!
//! Upstream test debt acknowledged: `cli.test.ts` has no vsearch/query
//! describe blocks, and `structured-search.test.ts`'s `structuredSearch`
//! section exercises the store directly — ported in
//! `qmd/tests/hybrid.rs` and `qmd/tests/query_doc.rs` (parser). What is
//! unique here is the CLI glue: format/usage errors, collection and
//! `--filter` validation, index-health warnings, structured-search
//! headers, `--no-rerank`/`--explain`, aliases, and the `_ciMode` gate
//! surfacing on the expansion path.
//!
//! Determinism: no test needs a model — `lex:`/empty-vector paths never
//! reach a backend, and the CI cases rely on the lazy backends failing
//! fast with `LLM operations are disabled in CI` before any download.

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

// ── Usage / argument validation ──────────────────────────────────────

#[test]
fn vsearch_requires_query() {
    let env = CliEnv::new();
    let out = env.run(&["vsearch"]);
    assert_eq!(out.exit_code, 1);
    assert!(out.stderr.contains("Usage: qmd vsearch"), "{}", out.stderr);
}

#[test]
fn vector_search_alias_requires_query() {
    // Alias resolves to the same command; upstream's usage line names
    // the canonical `vsearch` (qmd.ts usage table).
    let env = CliEnv::new();
    let out = env.run(&["vector-search"]);
    assert_eq!(out.exit_code, 1);
    assert!(out.stderr.contains("Usage: qmd vsearch"), "{}", out.stderr);
}

#[test]
fn query_requires_query() {
    let env = CliEnv::new();
    let out = env.run(&["query"]);
    assert_eq!(out.exit_code, 1);
    assert!(out.stderr.contains("Usage: qmd query"), "{}", out.stderr);
}

#[test]
fn deep_search_alias_requires_query() {
    let env = CliEnv::new();
    let out = env.run(&["deep-search"]);
    assert_eq!(out.exit_code, 1);
    assert!(out.stderr.contains("Usage: qmd query"), "{}", out.stderr);
}

#[test]
fn query_rejects_unknown_format() {
    // Format resolution precedes the missing-query usage check
    // (qmd.ts `resolveSearchFormat`), so the message wins even with a
    // valid positional arg present.
    let env = CliEnv::new();
    let out = env.run(&["query", "--format", "bogus", "index"]);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr.contains("Unknown --format value: bogus"),
        "{}",
        out.stderr
    );
    assert!(out.stderr.contains("json, csv, md, xml, files"));
}

#[test]
fn vsearch_rejects_bad_filter_json() {
    // `parseCliMetadataFilter` prints + exit(1) — no doctor hint.
    let env = env_with_fixtures();
    let out = env.run(&["vsearch", "index", "--filter", "{not json"]);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr.contains("Invalid --filter JSON"),
        "{}",
        out.stderr
    );
    assert!(out.stderr.contains("Example: --filter"), "{}", out.stderr);
}

// ── vsearch behavior on an index without vectors ─────────────────────

#[test]
fn vsearch_empty_vectors_returns_no_results() {
    // No `vectors_vec` table → upstream returns [] before touching the
    // embedder; the pending-embeddings warning still prints on stderr.
    let env = env_with_fixtures();
    let out = env.run(&["vsearch", "meeting"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.stdout, "No results found.\n");
    assert!(out.stderr.contains("need embeddings"), "{}", out.stderr);
}

#[test]
fn vsearch_empty_json_is_empty_array() {
    let env = env_with_fixtures();
    let out = env.run(&["vsearch", "meeting", "--json"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&out.stdout).unwrap(),
        serde_json::json!([])
    );
}

#[test]
fn vsearch_rejects_unknown_collection() {
    // `resolveCollectionFilter` → `Collection not found` + exit(1),
    // no doctor hint (qmd.ts:2165-2180).
    let env = env_with_fixtures();
    let out = env.run(&["vsearch", "index", "-c", "missing"]);
    assert_eq!(out.exit_code, 1);
    assert!(out.stderr.contains("Collection not found: missing"));
    assert!(!out.stderr.contains("qmd doctor"), "{}", out.stderr);
}

#[test]
fn vector_search_alias_behaves_like_vsearch() {
    let env = env_with_fixtures();
    let a = env.run(&["vsearch", "meeting"]);
    let b = env.run(&["vector-search", "meeting"]);
    assert_eq!(a.exit_code, b.exit_code);
    assert_eq!(a.stdout, b.stdout);
}

// ── query: structured path (no LLM needed) ───────────────────────────

#[test]
fn query_lex_runs_without_llm_and_prints_structured_header() {
    // `lex:` searches hit FTS only — no expansion, no embed, no rerank
    // (structuredSearch, store.ts:5990-6120). The stderr header mirrors
    // upstream's `Structured search: N queries (kinds)` + tree.
    let env = env_with_fixtures();
    let out = env.run(&["query", "lex: meeting", "--no-rerank"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("qmd://"), "{}", out.stdout);
    assert!(out.stdout.contains("Score:"), "{}", out.stdout);
    assert!(
        out.stderr.contains("Structured search: 1 queries (lex)"),
        "{}",
        out.stderr
    );
    assert!(out.stderr.contains("lex: meeting"), "{}", out.stderr);
    // No LLM hooks fire on the lex-only path.
    assert!(!out.stderr.contains("Expanding query"), "{}", out.stderr);
    assert!(!out.stderr.contains("Reranking"), "{}", out.stderr);
}

#[test]
fn query_lex_explain_shows_rrf_breakdown() {
    let env = env_with_fixtures();
    let out = env.run(&["query", "lex: meeting", "--no-rerank", "--explain"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Explain: fts=["), "{}", out.stdout);
    assert!(out.stdout.contains("RRF:"), "{}", out.stdout);
}

#[test]
fn query_lex_collection_scope_restricts_results() {
    let env = CliEnv::new();
    env.write_file(
        "notes/meeting.md",
        "# Team Meeting\n\nmeeting notes search\n",
    );
    env.write_file("docs/api.md", "# API\n\nsearch reference docs\n");
    env.write_config(
        "index",
        "collections:\n  notes:\n    path: {root}/fixtures\n    pattern: \"notes/*.md\"\n  docs:\n    path: {root}/fixtures\n    pattern: \"docs/*.md\"\n",
    );
    env.update();

    let out = env.run(&["query", "lex: search", "-c", "notes", "--no-rerank"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("qmd://notes/"), "{}", out.stdout);
    assert!(!out.stdout.contains("qmd://docs/"), "{}", out.stdout);
}

#[test]
fn query_vec_without_vectors_returns_no_results() {
    // `structuredSearch` gates vec/hyde embedding on the vec table
    // (hybrid.rs `has_vectors`) — no model call, just empty lists.
    let env = env_with_fixtures();
    let out = env.run(&["query", "vec: meeting", "--no-rerank"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.stdout, "No results found.\n");
    assert!(
        out.stderr.contains("Structured search: 1 queries (vec)"),
        "{}",
        out.stderr
    );
}

#[test]
fn query_rejects_mixed_plain_and_prefixed_lines() {
    // Upstream's `parseStructuredQuery` throw is an uncaught Bun dump;
    // ours is `exitWithError` — same message text, registered deviation.
    let env = env_with_fixtures();
    let out = env.run(&["query", "meeting\nsearch"]);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr.contains("missing a lex:/vec:/hyde:"),
        "{}",
        out.stderr
    );
}

// ── query: expansion path and the CI gate ────────────────────────────

#[test]
fn query_expansion_blocked_in_ci() {
    // A plain query routes to `expandQuery` → `LazyLlamaGenerator` →
    // the `_ciMode` gate (llm.ts:1631) throws before model init.
    // `Expanding query...` fires first — the hook precedes the throw.
    let env = env_with_fixtures();
    let out = env.run_opts(&["query", "meeting"], &env.fixtures, true, &[("CI", "1")]);
    assert_eq!(out.exit_code, 1, "stderr: {}", out.stderr);
    assert!(out.stderr.contains("Expanding query"), "{}", out.stderr);
    assert!(out.stderr.contains("disabled in CI"), "{}", out.stderr);
}

#[test]
fn deep_search_alias_runs_structured_path() {
    let env = env_with_fixtures();
    let out = env.run(&["deep-search", "lex: meeting", "--no-rerank"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("qmd://"), "{}", out.stdout);
    assert!(
        out.stderr.contains("Structured search: 1 queries (lex)"),
        "{}",
        out.stderr
    );
}
