//! Port of upstream `test/metadata-cli.test.ts` — `search --filter`
//! parsing, propagation, JSON `metadata` field, and error behavior.
//! State (corpus + index) is set up through the library because
//! `collection add`/`update` are P4.2 commands.

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

/// `notes` collection over two frontmatter-tagged documents.
fn env_with_metadata() -> CliEnv {
    let env = CliEnv::new();
    env.write_file(
        "published.md",
        "---\nqmd:\n  metadata:\n    status: published\n    topics: [typescript, programming]\n---\n\n# Published doc\n\ncli filter keyword body\n",
    );
    env.write_file(
        "draft.md",
        "---\nqmd:\n  metadata:\n    status: draft\n---\n\n# Draft doc\n\ncli filter keyword body\n",
    );
    env.write_config("index", &collection_yaml("notes", &env.fixtures, "**/*.md"));
    env.update();
    env
}

#[test]
fn returns_only_matching_documents_and_includes_metadata_in_json() {
    let env = env_with_metadata();
    let out = env.run(&[
        "search",
        "cli filter keyword",
        "--format",
        "json",
        "--filter",
        r#"{"key":"status","operator":"eq","value":"published"}"#,
    ]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);

    let results: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let arr = results.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["file"], "qmd://notes/published.md");
    assert_eq!(
        arr[0]["metadata"],
        serde_json::json!({"status": "published", "topics": ["typescript", "programming"]})
    );
}

#[test]
fn supports_nested_filters() {
    let env = env_with_metadata();
    let filter = r#"{"operator":"and","operands":[{"key":"topics","operator":"all","value":["typescript","programming"]},{"operator":"not","operand":{"key":"status","operator":"eq","value":"draft"}}]}"#;
    let out = env.run(&[
        "search",
        "cli filter keyword",
        "--format",
        "json",
        "--filter",
        filter,
    ]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let results: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let files: Vec<&str> = results
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["file"].as_str().unwrap())
        .collect();
    assert_eq!(files, ["qmd://notes/published.md"]);
}

#[test]
fn returns_format_safe_empty_output_when_nothing_matches() {
    let env = env_with_metadata();
    let out = env.run(&[
        "search",
        "cli filter keyword",
        "--format",
        "json",
        "--filter",
        r#"{"key":"status","operator":"eq","value":"missing"}"#,
    ]);
    assert_eq!(out.exit_code, 0);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&out.stdout).unwrap(),
        serde_json::json!([])
    );
}

#[test]
fn omits_metadata_from_json_output_when_a_document_has_none() {
    let env = env_with_metadata();
    env.write_file("plain.md", "# Plain doc\n\ncli filter keyword body\n");
    // Re-index through the library (upstream ran `qmd update`).
    env.update();

    let out = env.run(&["search", "cli filter keyword", "--format", "json"]);
    assert_eq!(out.exit_code, 0);
    let results: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let plain = results
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["file"] == "qmd://notes/plain.md")
        .expect("plain.md should appear in results");
    assert!(plain.get("metadata").is_none());
}

#[test]
fn rejects_malformed_filter_json_with_non_zero_exit() {
    let env = env_with_metadata();
    let out = env.run(&["search", "cli filter keyword", "--filter", "{not json"]);
    assert_eq!(out.exit_code, 1);
    assert!(out.stderr.contains("Invalid --filter JSON"));
}

#[test]
fn rejects_valid_json_with_an_invalid_filter_ast() {
    let env = env_with_metadata();
    let out = env.run(&[
        "search",
        "cli filter keyword",
        "--filter",
        r#"{"key":"status","operator":"equal","value":"x"}"#,
    ]);
    assert_eq!(out.exit_code, 1);
    assert!(out.stderr.contains("Invalid metadata filter at $"));
    assert!(out.stderr.contains("unknown operator 'equal'"));
}
