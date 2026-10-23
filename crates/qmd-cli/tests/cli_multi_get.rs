//! Port of upstream `test/cli.test.ts` describe `CLI Multi-Get
//! Command`. Each upstream test used a fresh db; here every test gets a
//! fresh `CliEnv` — indexing goes through the library (`collection add`
//! is P4.2).

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

use std::path::Path;

use common::{CliEnv, collection_yaml};

/// Upstream `beforeEach`: `collection add . --name fixtures` over the
/// standard corpus.
fn env_with_fixtures() -> CliEnv {
    let env = CliEnv::new();
    env.write_file(
        "README.md",
        "# Test Project\n\nThis is a test project for QMD CLI testing.\n",
    );
    env.write_file(
        "notes/meeting.md",
        "# Team Meeting Notes\n\nDate: 2024-01-15\n\n## Attendees\n- Alice\n- Bob\n",
    );
    env.write_file(
        "notes/ideas.md",
        "# Product Ideas\n\n## Feature Requests\n- Dark mode support\n",
    );
    env.write_file(
        "docs/api.md",
        "# API Documentation\n\n## Endpoints\n\n### GET /search\n",
    );
    env.write_config(
        "index",
        &collection_yaml("fixtures", &env.fixtures, "**/*.md"),
    );
    env.update();
    env
}

#[test]
fn retrieves_multiple_documents_by_pattern() {
    let env = env_with_fixtures();
    let out = env.run(&["multi-get", "notes/*.md"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Meeting"));
    assert!(out.stdout.contains("Ideas"));
}

#[test]
fn retrieves_documents_by_comma_separated_paths() {
    let env = env_with_fixtures();
    let out = env.run(&["multi-get", "README.md,notes/meeting.md"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Test Project"));
    assert!(out.stdout.contains("Team Meeting"));
}

#[test]
fn comma_list_accepts_collection_prefixed_paths() {
    // tobi/qmd#759
    let env = env_with_fixtures();
    let out = env.run(&["multi-get", "fixtures/docs/api.md, fixtures/README.md"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(!out.stderr.contains("File not found"));
    assert!(out.stdout.contains("API Documentation"));
    assert!(out.stdout.contains("Test Project"));
}

#[test]
fn comma_list_does_not_fetch_a_document_from_a_filename_fragment() {
    // tobi/qmd#759 — `pi.md` must not fuzzy-match `api.md`.
    let env = env_with_fixtures();
    let out = env.run(&["multi-get", "pi.md, ZZZ-nonexistent.md"]);
    assert!(!out.stdout.contains("API Documentation"));
    assert!(out.stderr.contains("File not found: pi.md"));
    assert!(out.stderr.contains("File not found: ZZZ-nonexistent.md"));
    assert!(!out.stderr.to_lowercase().contains("api.md"));
}

#[test]
fn md_output_includes_a_docid_for_each_file() {
    let env = env_with_fixtures();
    let out = env.run(&["multi-get", "notes/*.md", "--md"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("**docid:** `#"));
}

#[test]
fn json_output_includes_a_docid_for_each_file() {
    let env = env_with_fixtures();
    let out = env.run(&["multi-get", "notes/*.md", "--json"]);
    assert_eq!(out.exit_code, 0);
    let parsed: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let arr = parsed.as_array().unwrap();
    assert!(!arr.is_empty());
    for entry in arr {
        let docid = entry["docid"].as_str().unwrap();
        assert!(docid.starts_with('#') && docid.len() == 7);
    }
}

#[test]
fn format_files_emits_docid_as_its_own_csv_field() {
    // tobi/qmd#760
    let env = env_with_fixtures();
    let out = env.run(&["multi-get", "README.md", "--format", "files"]);
    assert_eq!(out.exit_code, 0);
    let line = out.stdout.trim().lines().next().unwrap();
    // Shape: #docid,path[,"context"] — not "#docid path,…".
    assert!(line.starts_with('#'));
    let first = line.split(',').next().unwrap();
    assert!(first.starts_with('#') && first.len() == 7);
    assert!(!first.contains(' '));
}

#[test]
fn retrieves_a_document_by_docid() {
    let env = env_with_fixtures();
    let lookup = env.run(&["multi-get", "README.md", "--json"]);
    assert_eq!(lookup.exit_code, 0);
    let parsed: serde_json::Value = serde_json::from_str(&lookup.stdout).unwrap();
    let docid = parsed[0]["docid"].as_str().unwrap().to_owned();

    let out = env.run(&["multi-get", &docid, "--json"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(!out.stderr.contains("No files matched pattern"));
    let by_docid: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let arr = by_docid.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["docid"].as_str().unwrap(), docid);
    assert!(arr[0]["body"].as_str().unwrap().contains("Test Project"));
}

#[test]
fn fails_for_a_missing_single_docid() {
    let env = env_with_fixtures();
    let out = env.run(&["multi-get", "#000000", "--json"]);
    assert_eq!(out.exit_code, 1);
    assert_eq!(out.stdout, "");
    assert!(out.stderr.contains("File not found: #000000"));
}

#[test]
fn non_numeric_max_bytes_skips_no_files() {
    // parseInt("abc") is NaN upstream and `bodyLength > NaN` is false,
    // so a NaN --max-bytes disables skipping entirely (qmd.ts:1444-1448).
    let env = env_with_fixtures();
    let out = env.run(&[
        "multi-get",
        "README.md,notes/meeting.md",
        "--max-bytes",
        "abc",
    ]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Test Project"));
    assert!(out.stdout.contains("Team Meeting"));
    assert!(!out.stdout.contains("SKIPPED"));
}

#[test]
fn retrieves_docids_in_comma_separated_lists() {
    let env = env_with_fixtures();
    let lookup = env.run(&["multi-get", "README.md", "--json"]);
    let parsed: serde_json::Value = serde_json::from_str(&lookup.stdout).unwrap();
    let docid = parsed[0]["docid"].as_str().unwrap().to_owned();

    let out = env.run(&["multi-get", &format!("{docid},notes/meeting.md"), "--json"]);
    assert_eq!(out.exit_code, 0);
    let listed: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let arr = listed.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["docid"].as_str().unwrap(), docid);
    assert_eq!(
        arr[1]["file"].as_str().unwrap(),
        "qmd://fixtures/notes/meeting.md"
    );
}

#[test]
fn shows_line_numbers_by_default_and_no_line_numbers_disables_them() {
    let env = env_with_fixtures();
    let with_nums = env.run(&["multi-get", "README.md"]);
    assert_eq!(with_nums.exit_code, 0);
    assert!(with_nums.stdout.lines().any(|l| l.starts_with("1: ")));

    let raw = env.run(&["multi-get", "README.md", "--no-line-numbers"]);
    assert_eq!(raw.exit_code, 0);
    assert!(!raw.stdout.lines().any(|l| l.starts_with("1: ")));
}

#[test]
fn full_path_md_shows_dot_slash_paths_and_drops_the_docid() {
    let env = env_with_fixtures();
    let out = env.run(&["multi-get", "notes/*.md", "--md", "--full-path"]);
    assert_eq!(out.exit_code, 0);
    assert!(
        out.stdout
            .lines()
            .any(|l| l.starts_with("## ./notes/") && l.to_lowercase().ends_with(".md"))
    );
    assert!(!out.stdout.contains("qmd://"));
    assert!(!out.stdout.contains("**docid:**"));
}

#[test]
fn full_path_json_puts_dot_slash_path_in_file_and_omits_docid() {
    let env = env_with_fixtures();
    let out = env.run(&["multi-get", "notes/*.md", "--json", "--full-path"]);
    assert_eq!(out.exit_code, 0);
    let parsed: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let arr = parsed.as_array().unwrap();
    assert!(!arr.is_empty());
    for entry in arr {
        assert!(entry["file"].as_str().unwrap().starts_with("./notes/"));
        assert!(entry.get("docid").is_none());
    }
}

#[test]
fn full_path_json_uses_absolute_path_when_files_are_outside_pwd() {
    let env = env_with_fixtures();
    let out = env.run_at(
        &["multi-get", "notes/*.md", "--json", "--full-path"],
        Path::new("/"),
    );
    assert_eq!(out.exit_code, 0);
    let parsed: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let arr = parsed.as_array().unwrap();
    assert!(!arr.is_empty());
    for entry in arr {
        let file = entry["file"].as_str().unwrap();
        assert!(file.starts_with('/'));
        assert!(!file.starts_with("./"));
        assert!(entry.get("docid").is_none());
    }
}
