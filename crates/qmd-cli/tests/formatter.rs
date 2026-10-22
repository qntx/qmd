//! Port of upstream `test/formatter.test.ts` — context must appear in
//! every output format and be omitted when null. Upstream tests call
//! the (dead) `formatter.ts` functions directly; here the same
//! assertions run against the real renderers through the CLI.
//!
//! Upstream's single-document formatters (`documentToJson` /
//! `documentToMarkdown` / `documentToXml` and their `formatDocument`
//! dispatch) exist only for the MCP server — `get` takes no `--format`
//! upstream either — so the `single document …` describes are N/A here
//! (MCP is a later phase). `searchResultsToMcpCsv` likewise belongs to
//! the MCP transport.

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

const CONTEXT: &str = "Internal engineering keynotes from company summit events";

/// One `archive` collection (upstream fixture path) with and without
/// collection context.
fn env(context: bool) -> CliEnv {
    let env = CliEnv::new();
    env.write_file(
        "summit/keynote.md",
        "---\ntitle: Summit Keynote\n---\n\nThis is the keynote content.\n",
    );
    let ctx = if context {
        format!("    context:\n      /: {CONTEXT}\n")
    } else {
        String::new()
    };
    env.write_config(
        "index",
        &format!(
            "collections:\n  archive:\n    path: {{root}}/fixtures\n    pattern: \"**/*.md\"\n{ctx}"
        ),
    );
    env.update();
    env
}

// ── search results include context in all formats ────────────────────

#[test]
fn search_json_format_includes_context() {
    let env = env(true);
    let out = env.run(&["search", "keynote", "--json", "-n", "1"]);
    assert_eq!(out.exit_code, 0);
    let parsed: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(parsed[0]["context"], CONTEXT);
}

#[test]
fn search_json_format_includes_line() {
    let env = env(true);
    let out = env.run(&["search", "keynote", "--json", "-n", "1"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("\"line\""));
    let parsed: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert!(parsed[0]["line"].as_u64().unwrap() > 0);
}

#[test]
fn search_json_format_includes_line_with_full() {
    let env = env(true);
    let out = env.run(&["search", "keynote", "--json", "--full", "-n", "1"]);
    assert_eq!(out.exit_code, 0);
    let parsed: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert!(parsed[0]["line"].as_u64().unwrap() > 0);
}

#[test]
fn search_csv_format_includes_context() {
    let env = env(true);
    let out = env.run(&["search", "keynote", "--csv", "-n", "1"]);
    assert_eq!(out.exit_code, 0);
    let first = out.stdout.lines().next().unwrap();
    assert!(first.contains("context"));
    assert!(out.stdout.contains(CONTEXT));
}

#[test]
fn search_files_format_includes_context() {
    let env = env(true);
    let out = env.run(&["search", "keynote", "--files", "-n", "1"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains(CONTEXT));
}

#[test]
fn search_markdown_format_includes_context() {
    let env = env(true);
    let out = env.run(&["search", "keynote", "--md", "-n", "1"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains(CONTEXT));
}

#[test]
fn search_xml_format_includes_context() {
    let env = env(true);
    let out = env.run(&["search", "keynote", "--xml", "-n", "1"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains(CONTEXT));
}

// ── search results omit context when null ────────────────────────────

#[test]
fn search_json_format_omits_context_field_when_null() {
    let env = env(false);
    let out = env.run(&["search", "keynote", "--json", "-n", "1"]);
    assert_eq!(out.exit_code, 0);
    let parsed: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert!(parsed[0].get("context").is_none());
}

#[test]
fn search_files_format_has_no_trailing_context_when_null() {
    let env = env(false);
    let out = env.run(&["search", "keynote", "--files", "-n", "1"]);
    assert_eq!(out.exit_code, 0);
    // Just docid,score,path — no trailing comma/context.
    assert!(!out.stdout.contains(",\""));
}

// ── multi-get documents include context in all formats ───────────────

#[test]
fn multi_get_json_format_includes_context() {
    let env = env(true);
    let out = env.run(&["multi-get", "summit/keynote.md", "--json"]);
    assert_eq!(out.exit_code, 0);
    let parsed: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(parsed[0]["context"], CONTEXT);
}

#[test]
fn multi_get_csv_format_includes_context() {
    let env = env(true);
    let out = env.run(&["multi-get", "summit/keynote.md", "--csv"]);
    assert_eq!(out.exit_code, 0);
    let first = out.stdout.lines().next().unwrap();
    assert!(first.contains("context"));
    assert!(out.stdout.contains(CONTEXT));
}

#[test]
fn multi_get_files_format_includes_context() {
    let env = env(true);
    let out = env.run(&["multi-get", "summit/keynote.md", "--files"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains(CONTEXT));
}

#[test]
fn multi_get_markdown_format_includes_context() {
    let env = env(true);
    let out = env.run(&["multi-get", "summit/keynote.md", "--md"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains(CONTEXT));
}

#[test]
fn multi_get_xml_format_includes_context() {
    let env = env(true);
    let out = env.run(&["multi-get", "summit/keynote.md", "--xml"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains(CONTEXT));
}
