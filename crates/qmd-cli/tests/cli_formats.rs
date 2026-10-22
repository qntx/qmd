//! Port of upstream `test/cli.test.ts` describe `search output
//! formats` — qmd:// paths, docids and context in every format,
//! `--full-path` rendering, `--format` equivalence to the legacy
//! aliases, and the custom-index `?index=` round trip through `get`.

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

use common::{CliEnv, strip_ansi};

/// `fixtures` collection with collection-root context ("Test fixtures
/// for QMD" — upstream `context add qmd://fixtures/ …`).
fn env_with_context() -> CliEnv {
    let env = CliEnv::new();
    env.write_file(
        "README.md",
        "# Test Project\n\nThis is a test project for QMD CLI testing.\n",
    );
    env.write_file(
        "notes/meeting.md",
        "# Team Meeting Notes\n\nDate: 2024-01-15\n\n## Attendees\n- Alice\n- Bob\n",
    );
    env.write_file("test1.md", "# Test Document 1\n\ntest body\n");
    env.write_config(
        "index",
        "collections:\n  fixtures:\n    path: {root}/fixtures\n    pattern: \"**/*.md\"\n    context:\n      /: Test fixtures for QMD\n",
    );
    env.update();
    env
}

#[test]
fn search_json_includes_qmd_path_docid_and_context() {
    let env = env_with_context();
    let out = env.run(&["search", "test", "--json", "-n", "1"]);
    assert_eq!(out.exit_code, 0);
    let results: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let result = &results[0];
    assert!(
        result["file"]
            .as_str()
            .unwrap()
            .starts_with("qmd://fixtures/")
    );
    assert!(result["docid"].as_str().unwrap().starts_with('#'));
    assert_eq!(result["context"].as_str().unwrap(), "Test fixtures for QMD");
    // No full filesystem paths.
    assert!(!result["file"].as_str().unwrap().starts_with('/'));
}

#[test]
fn custom_index_search_links_include_index_and_round_trip_through_get() {
    let env = CliEnv::new();
    env.write_file("test1.md", "# Test Document\n\ntest body\n");
    // `--index release-notes` → config `release-notes.yml`, db under
    // XDG_CACHE_HOME (no INDEX_PATH).
    env.write_config(
        "release-notes",
        "collections:\n  fixtures-alt:\n    path: {root}/fixtures\n    pattern: \"**/*.md\"\n",
    );
    env.reindex("release-notes", "release-notes", false);

    let search = env.run_opts(
        &[
            "--index",
            "release-notes",
            "search",
            "test",
            "--json",
            "-n",
            "1",
        ],
        &env.fixtures,
        false,
        &[],
    );
    assert_eq!(search.exit_code, 0, "stderr: {}", search.stderr);
    let results: serde_json::Value = serde_json::from_str(&search.stdout).unwrap();
    let file = results[0]["file"].as_str().unwrap();
    assert!(
        file.starts_with("qmd://fixtures-alt/") && file.ends_with("?index=release-notes"),
        "file was {file}"
    );

    let get = env.run_opts(&["get", file, "-l", "2"], &env.fixtures, false, &[]);
    assert_eq!(get.exit_code, 0, "stderr: {}", get.stderr);
    assert!(!get.stdout.trim().is_empty());
}

#[test]
fn search_files_includes_qmd_path_docid_and_context() {
    let env = env_with_context();
    let out = env.run(&["search", "test", "--files", "-n", "1"]);
    assert_eq!(out.exit_code, 0);
    // Format: #docid,score,qmd://collection/path,"context"
    let line = out.stdout.lines().next().unwrap();
    assert!(line.starts_with('#'));
    assert!(line.contains(",qmd://fixtures/"));
    assert!(out.stdout.contains("Test fixtures for QMD"));
    assert!(!out.stdout.contains(env.root.to_string_lossy().as_ref()));
}

#[test]
fn search_csv_includes_qmd_path_docid_and_context() {
    let env = env_with_context();
    let out = env.run(&["search", "test", "--csv", "-n", "1"]);
    assert_eq!(out.exit_code, 0);
    assert!(
        out.stdout
            .contains("docid,score,file,title,context,line,snippet")
    );
    assert!(out.stdout.contains(",qmd://fixtures/"));
    assert!(out.stdout.contains("Test fixtures for QMD"));
}

#[test]
fn search_md_includes_docid_context_and_qmd_file_line() {
    let env = env_with_context();
    let out = env.run(&["search", "test", "--md", "-n", "1"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("**docid:** `#"));
    assert!(out.stdout.contains("**context:** Test fixtures for QMD"));
    assert!(out.stdout.contains("**file:** `qmd://fixtures/"));
}

#[test]
fn search_xml_includes_qmd_path_docid_and_context() {
    let env = env_with_context();
    let out = env.run(&["search", "test", "--xml", "-n", "1"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("docid=\"#"));
    assert!(out.stdout.contains("name=\"qmd://fixtures/"));
    assert!(out.stdout.contains("context=\"Test fixtures for QMD\""));
}

#[test]
fn search_full_path_json_swaps_qmd_uri_for_absolute_realpath() {
    let env = env_with_context();
    let out = env.run_at(
        &["search", "test", "--full-path", "--json", "-n", "1"],
        Path::new("/"),
    );
    assert_eq!(out.exit_code, 0);
    let results: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let result = &results[0];
    let file = result["file"].as_str().unwrap();
    assert!(!file.starts_with("qmd://"));
    assert!(file.starts_with('/'));
    assert!(
        Path::new(file)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("md"))
    );
    assert!(result.get("docid").is_none());
}

#[test]
fn search_full_path_json_uses_dot_slash_relative_path_under_pwd() {
    let env = env_with_context();
    let out = env.run(&["search", "test", "--full-path", "--json", "-n", "1"]);
    assert_eq!(out.exit_code, 0);
    let results: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let file = results[0]["file"].as_str().unwrap();
    assert!(!file.starts_with("qmd://"));
    assert!(file.starts_with("./"));
    assert!(!file.starts_with("../"));
    assert!(
        Path::new(file)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("md"))
    );
}

#[test]
fn search_full_path_cli_shows_on_disk_path_and_drops_docid() {
    let env = env_with_context();
    let out = env.run_at(
        &["search", "test", "--full-path", "-n", "1"],
        Path::new("/"),
    );
    assert_eq!(out.exit_code, 0);
    let plain = strip_ansi(&out.stdout);
    assert!(!plain.contains("qmd://"));
    // Result lines are `<abs path>:<line>` (upstream `/^\/.+\.md/m`).
    assert!(
        plain
            .lines()
            .any(|l| l.starts_with('/') && l.contains(".md"))
    );
    // No `#docid` suffix when --full-path is set.
    assert!(!plain.lines().any(|l| {
        let t = l.trim_end();
        t.rsplit_once('#')
            .is_some_and(|(_, s)| s.len() == 6 && s.chars().all(|c| c.is_ascii_hexdigit()))
    }));
}

#[test]
fn search_full_path_md_uses_on_disk_path_and_drops_docid() {
    let env = env_with_context();
    let out = env.run_at(
        &["search", "test", "--full-path", "--md", "-n", "1"],
        Path::new("/"),
    );
    assert_eq!(out.exit_code, 0);
    assert!(!out.stdout.contains("qmd://"));
    assert!(!out.stdout.contains("**docid:**"));
    assert!(out.stdout.contains("**file:** `/"));
}

#[test]
fn format_json_matches_legacy_json() {
    let env = env_with_context();
    let a = env.run(&["search", "test", "--format", "json", "-n", "1"]);
    let b = env.run(&["search", "test", "--json", "-n", "1"]);
    assert_eq!(a.exit_code, 0);
    assert_eq!(b.exit_code, 0);
    let ar: serde_json::Value = serde_json::from_str(&a.stdout).unwrap();
    let br: serde_json::Value = serde_json::from_str(&b.stdout).unwrap();
    assert!(!ar.as_array().unwrap().is_empty());
    assert_eq!(ar[0]["file"], br[0]["file"]);
}

#[test]
fn format_md_works_equivalent_to_legacy_md() {
    let env = env_with_context();
    let out = env.run(&["search", "test", "--format", "md", "-n", "1"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("**docid:** `#"));
    assert!(out.stdout.contains("**file:** `qmd://fixtures/"));
}

#[test]
fn format_with_an_unknown_kind_fails_cleanly() {
    let env = env_with_context();
    let out = env.run(&["search", "test", "--format", "yaml", "-n", "1"]);
    assert_ne!(out.exit_code, 0);
    assert!(out.stderr.contains("Unknown --format value"));
}

#[test]
fn default_cli_format_includes_qmd_path_docid_and_context_non_tty() {
    let env = env_with_context();
    let out = env.run(&["search", "test", "-n", "1"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("qmd://fixtures/"));
    assert!(out.stdout.contains('#'));
    assert!(out.stdout.contains("Context: Test fixtures for QMD"));
    assert!(!out.stdout.contains("\x1b]8;;"));
    let plain = strip_ansi(&out.stdout);
    assert!(!plain.lines().any(|l| l.starts_with("fixtures/")));
}
