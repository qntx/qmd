//! Ports of upstream `test/cli.test.ts` describe blocks that are in
//! P4.1 scope: `CLI Help`, `CLI Search Command`, `CLI Error Handling`,
//! `CLI Output Formats` and `CLI Search with Collection Filter` (plus
//! the parse-level collection-flag tests of
//! `test/multi-collection-filter.test.ts`, which our strict clap
//! parsing makes an integration concern — see `collection_flag.rs`
//! coverage in `src/args.rs` unit tests instead).
//!
//! Upstream relies on `collection add`/`update` (P4.2 commands here);
//! state is set up through the library (`common::CliEnv`) instead.
//! Deferred upstream assertions, noted at each port:
//! `CLI Help` also asserts `qmd collection add` and `qmd skill
//! show/install` lines — both commands land in P4.2/P4.4, so our help
//! lists implemented commands only.
//!
//! `CLI Skills`, `CLI Embed`, `CLI Skill Commands`, `CLI Init`, `CLI
//! Add`, `CLI Status`/`doctor`, `CLI Update`, `CLI Add-Context`, `CLI
//! Cleanup`, `CLI Context Management`, `CLI Collection Commands`,
//! `collection ignore patterns` (update-side), `status and collection
//! list hide filesystem paths`, `mcp http daemon` and `mcp stdio
//! launcher` are all out of P4.1 scope.

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

use common::{CliEnv, collection_yaml};

/// The corpus upstream's `beforeAll` writes plus a `fixtures`
/// collection over it.
fn env_with_fixtures() -> CliEnv {
    let env = CliEnv::new();
    env.write_file(
        "README.md",
        "# Test Project\n\nThis is a test project for QMD CLI testing.\n\n## Features\n\n- Full-text search with BM25\n- Vector similarity search\n- Hybrid search with reranking\n",
    );
    env.write_file(
        "notes/meeting.md",
        "# Team Meeting Notes\n\nDate: 2024-01-15\n\n## Attendees\n- Alice\n- Bob\n- Charlie\n\n## Discussion Topics\n- Project timeline review\n- Resource allocation\n- Technical debt prioritization\n\n## Action Items\n1. Alice to update documentation\n2. Bob to fix authentication bug\n3. Charlie to review pull requests\n",
    );
    env.write_file(
        "notes/ideas.md",
        "# Product Ideas\n\n## Feature Requests\n- Dark mode support\n- Keyboard shortcuts\n- Export to PDF\n\n## Technical Improvements\n- Improve search performance\n- Add caching layer\n- Optimize database queries\n",
    );
    env.write_file(
        "docs/api.md",
        "# API Documentation\n\n## Endpoints\n\n### GET /search\nSearch for documents.\n\nParameters:\n- q: Search query (required)\n- limit: Max results (default: 10)\n\n### GET /document/:id\nRetrieve a specific document.\n\n### POST /index\nIndex new documents.\n",
    );
    env.write_file(
        "test1.md",
        "# Test Document 1\n\nThis is the first test document.\n\nIt has multiple lines for testing line numbers.\nLine 6 is here.\nLine 7 is here.\n",
    );
    env.write_file(
        "test2.md",
        "# Test Document 2\n\nThis is the second test document.\n",
    );
    env.write_config(
        "index",
        &collection_yaml("fixtures", &env.fixtures, "**/*.md"),
    );
    env.update();
    env
}

// ── CLI Help ─────────────────────────────────────────────────────────

#[test]
fn shows_help_with_help_flag() {
    let env = CliEnv::new();
    let out = env.run(&["--help"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Usage:"));
    assert!(out.stdout.contains("qmd search"));
    assert!(out.stdout.contains("--no-gpu"));
    // Deferred: upstream also asserts `qmd collection add` and
    // `qmd skill show/install` — those commands are P4.2/P4.4 and our
    // help lists implemented commands only (P4-D2).
}

#[test]
fn shows_help_with_no_arguments() {
    let env = CliEnv::new();
    let out = env.run(&[]);
    assert_eq!(out.exit_code, 1);
    assert!(out.stdout.contains("Usage:"));
}

// ── CLI Search Command ───────────────────────────────────────────────

#[test]
fn searches_for_documents_with_bm25() {
    let env = env_with_fixtures();
    let out = env.run(&["search", "meeting"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.to_lowercase().contains("meeting"));
}

#[test]
fn searches_with_limit_option() {
    let env = env_with_fixtures();
    let out = env.run(&["search", "-n", "1", "test"]);
    assert_eq!(out.exit_code, 0);
}

#[test]
fn searches_with_all_results_option() {
    let env = env_with_fixtures();
    let out = env.run(&["search", "--all", "the"]);
    assert_eq!(out.exit_code, 0);
}

#[test]
fn returns_no_results_message_for_non_matching_query() {
    let env = env_with_fixtures();
    let out = env.run(&["search", "xyznonexistent123"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("No results"));
}

#[test]
fn returns_empty_json_array_for_non_matching_query() {
    let env = env_with_fixtures();
    let out = env.run(&["search", "xyznonexistent123", "--json"]);
    assert_eq!(out.exit_code, 0);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&out.stdout).unwrap(),
        serde_json::json!([])
    );
}

#[test]
fn returns_csv_header_only_for_non_matching_query() {
    let env = env_with_fixtures();
    let out = env.run(&["search", "xyznonexistent123", "--csv"]);
    assert_eq!(out.exit_code, 0);
    assert_eq!(
        out.stdout.trim(),
        "docid,score,file,title,context,line,snippet"
    );
}

#[test]
fn returns_empty_xml_container_for_non_matching_query() {
    let env = env_with_fixtures();
    let out = env.run(&["search", "xyznonexistent123", "--xml"]);
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.trim(), "<results></results>");
}

#[test]
fn returns_empty_output_for_non_matching_query_md() {
    let env = env_with_fixtures();
    let out = env.run(&["search", "xyznonexistent123", "--md"]);
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.trim(), "");
}

#[test]
fn returns_empty_output_for_non_matching_query_files() {
    let env = env_with_fixtures();
    let out = env.run(&["search", "xyznonexistent123", "--files"]);
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.trim(), "");
}

#[test]
fn returns_min_score_threshold_message_for_default_cli_output() {
    let env = env_with_fixtures();
    let out = env.run(&["search", "test", "--min-score", "2"]);
    assert_eq!(out.exit_code, 0);
    assert!(
        out.stdout
            .contains("No results found above minimum score threshold.")
    );
}

#[test]
fn returns_format_safe_empty_output_when_min_score_filters_all() {
    let env = env_with_fixtures();
    let json = env.run(&["search", "test", "--json", "--min-score", "2"]);
    assert_eq!(json.exit_code, 0);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&json.stdout).unwrap(),
        serde_json::json!([])
    );
    let csv = env.run(&["search", "test", "--csv", "--min-score", "2"]);
    assert_eq!(
        csv.stdout.trim(),
        "docid,score,file,title,context,line,snippet"
    );
    let xml = env.run(&["search", "test", "--xml", "--min-score", "2"]);
    assert_eq!(xml.stdout.trim(), "<results></results>");
    let md = env.run(&["search", "test", "--md", "--min-score", "2"]);
    assert_eq!(md.stdout.trim(), "");
    let files = env.run(&["search", "test", "--files", "--min-score", "2"]);
    assert_eq!(files.stdout.trim(), "");
}

#[test]
fn requires_query_argument() {
    let env = env_with_fixtures();
    let out = env.run(&["search"]);
    assert_eq!(out.exit_code, 1);
    assert!(out.stderr.contains("Usage:"));
}

#[test]
fn json_full_includes_line_field_for_round_tripping_to_get() {
    let env = env_with_fixtures();
    let out = env.run(&["search", "meeting", "--json", "--full", "-n", "1"]);
    assert_eq!(out.exit_code, 0);
    let results: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let first = &results[0];
    assert!(first["line"].as_u64().unwrap() > 0);
    assert!(first["body"].is_string());
}

// ── CLI Error Handling ───────────────────────────────────────────────

#[test]
fn handles_unknown_command() {
    let env = env_with_fixtures();
    let out = env.run(&["unknowncommand"]);
    assert_eq!(out.exit_code, 1);
    assert!(out.stderr.contains("Unknown command"));
    assert!(out.stderr.contains("qmd doctor"));
}

#[test]
fn uses_index_path_environment_variable() {
    // Upstream proves INDEX_PATH via `collection add`; the equivalent
    // here is a library-driven index, then any CLI run against it.
    let env = env_with_fixtures();
    let custom_db = env.root.join("custom.sqlite");
    let out = env.run_opts(
        &["search", "meeting"],
        &env.fixtures,
        false,
        &[("INDEX_PATH", &custom_db.to_string_lossy())],
    );
    assert_eq!(out.exit_code, 0);
    // Opening the store creates the db at the overridden path.
    assert!(custom_db.exists());
}

// ── CLI Output Formats ───────────────────────────────────────────────

#[test]
fn search_with_json_flag_outputs_json() {
    let env = env_with_fixtures();
    let out = env.run(&["search", "--json", "test"]);
    assert_eq!(out.exit_code, 0);
    assert!(
        serde_json::from_str::<serde_json::Value>(&out.stdout)
            .unwrap()
            .is_array()
    );
}

#[test]
fn search_with_files_flag_outputs_file_paths() {
    let env = env_with_fixtures();
    let out = env.run(&["search", "--files", "meeting"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains(".md"));
}

#[test]
fn search_output_includes_snippets_by_default() {
    let env = env_with_fixtures();
    let out = env.run(&["search", "API"]);
    assert_eq!(out.exit_code, 0);
    if !out.stdout.contains("No results") {
        assert!(out.stdout.to_lowercase().contains("api"));
    }
}

// ── CLI Search with Collection Filter ────────────────────────────────

#[test]
fn filters_search_by_collection_name() {
    // Upstream adds two collections via `collection add --mask`;
    // equivalent config here, indexed through the library.
    let env = CliEnv::new();
    env.write_file("notes/meeting.md", "# Team Meeting\n\nmeeting notes\n");
    env.write_file("docs/api.md", "# API\n\nreference\n");
    env.write_config(
        "index",
        "collections:\n  notes:\n    path: {root}/fixtures\n    pattern: \"notes/*.md\"\n  docs:\n    path: {root}/fixtures\n    pattern: \"docs/*.md\"\n",
    );
    env.update();

    let out = env.run(&["search", "-c", "notes", "meeting"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
}
