//! Ports of upstream `test/cli.test.ts` describe blocks `CLI Get
//! Command`, `get command path normalization` and `get command missing
//! and ignored paths`. Indexing goes through the library since
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

use std::path::Path;

use common::{CliEnv, collection_yaml};

/// `test1.md`/`test2.md`/`README.md` — the files the upstream
/// path-normalization tests address.
fn env_with_get_fixtures() -> CliEnv {
    let env = CliEnv::new();
    env.write_file(
        "README.md",
        "# Test Project\n\nThis is a test project for QMD CLI testing.\n",
    );
    env.write_file(
        "notes/meeting.md",
        "# Team Meeting Notes\n\nDate: 2024-01-15\n",
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

// ── CLI Get Command ──────────────────────────────────────────────────

#[test]
fn retrieves_document_content_by_path() {
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "README.md"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Test Project"));
}

#[test]
fn retrieves_document_from_subdirectory() {
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "notes/meeting.md"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Team Meeting"));
}

#[test]
fn handles_non_existent_file() {
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "nonexistent.md"]);
    assert_eq!(out.exit_code, 1);
}

#[test]
fn clamps_negative_from_to_top_of_file() {
    let env = env_with_get_fixtures();
    let baseline = env.run(&["get", "README.md"]);
    let negative = env.run(&["get", "README.md", "--from", "-19"]);
    assert_eq!(negative.exit_code, 0);
    assert_eq!(negative.stdout, baseline.stdout);
}

#[test]
fn non_numeric_from_prints_the_full_body_numbered_from_1() {
    // parseInt("abc") is NaN upstream: Math.max(1, NaN) is NaN, slice(NaN)
    // starts at 0 and addLineNumbers' `startLine || 1` is 1 — a
    // non-numeric --from behaves like no --from at all (qmd.ts:1330-1345).
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "test2.md", "--from", "abc"]);
    assert_eq!(out.exit_code, 0);
    let header = out.stdout.lines().next().unwrap();
    assert!(header.starts_with("qmd://fixtures/test2.md  #"));
    assert_eq!(
        out.stdout,
        format!(
            "{header}\n---\n\n1: # Test Document 2\n2: \n3: This is the second test document.\n4: \n"
        )
    );
}

#[test]
fn non_numeric_l_prints_an_empty_numbered_body() {
    // slice(0, NaN) is empty upstream; addLineNumbers("") still emits a
    // single "1: " line (qmd.ts:1337-1345).
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "test2.md", "-l", "abc"]);
    assert_eq!(out.exit_code, 0);
    let header = out.stdout.lines().next().unwrap();
    assert!(header.starts_with("qmd://fixtures/test2.md  #"));
    assert_eq!(out.stdout, format!("{header}\n---\n\n1: \n"));
}

// ── get command path normalization ───────────────────────────────────

#[test]
fn get_with_qmd_collection_path_format() {
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "qmd://fixtures/test1.md", "-l", "3"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Test Document 1"));
}

#[test]
fn get_with_collection_path_format_no_scheme() {
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "fixtures/test1.md", "-l", "3"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Test Document 1"));
}

#[test]
fn get_with_double_slash_collection_path_format() {
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "//fixtures/test1.md", "-l", "3"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Test Document 1"));
}

#[test]
fn get_with_qmd_extra_slashes_format() {
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "qmd:////fixtures/test1.md", "-l", "3"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Test Document 1"));
}

#[test]
fn get_with_path_line_format() {
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "fixtures/test1.md:3", "-l", "2"]);
    assert_eq!(out.exit_code, 0);
    // Starts from line 3, not line 1.
    assert!(!out.stdout.contains("# Test Document 1\n"));
    assert!(out.stdout.contains("This is the first test document."));
}

#[test]
fn get_with_qmd_path_line_format() {
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "qmd://fixtures/test1.md:3", "-l", "2"]);
    assert_eq!(out.exit_code, 0);
    assert!(!out.stdout.contains("# Test Document 1\n"));
}

#[test]
fn get_with_path_from_count_format_reads_a_bounded_range() {
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "fixtures/test1.md:5:2"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("It has multiple lines"));
    assert!(out.stdout.contains("Line 6 is here."));
    assert!(!out.stdout.contains("1: # Test Document 1"));
    assert!(!out.stdout.contains("Line 7 is here."));
}

#[test]
fn get_with_qmd_path_from_count_format_reads_a_bounded_range() {
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "qmd://fixtures/test1.md:5:2"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("It has multiple lines"));
    assert!(out.stdout.contains("Line 6 is here."));
    assert!(!out.stdout.contains("Line 7 is here."));
}

#[test]
fn explicit_l_overrides_the_count_in_path_from_count() {
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "fixtures/test1.md:5:2", "-l", "1"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("It has multiple lines"));
    assert!(!out.stdout.contains("Line 6 is here."));
}

#[test]
fn get_header_includes_canonical_qmd_path_and_docid() {
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "fixtures/test1.md"]);
    assert_eq!(out.exit_code, 0);
    let first = out.stdout.lines().next().unwrap();
    assert!(first.starts_with("qmd://fixtures/test1.md"));
    assert!(first.contains('#'));
}

#[test]
fn get_shows_line_numbers_by_default() {
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "fixtures/test1.md"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("1: # Test Document 1"));
    assert!(out.stdout.contains("6: Line 6 is here."));
}

#[test]
fn get_no_line_numbers_returns_raw_content() {
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "fixtures/test1.md", "--no-line-numbers"]);
    assert_eq!(out.exit_code, 0);
    assert!(!out.stdout.lines().any(|l| l.starts_with("1: ")));
    assert!(out.stdout.contains("# Test Document 1"));
}

#[test]
fn get_line_numbers_reflect_the_start_line_of_a_range() {
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "fixtures/test1.md:5:2"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("5: It has multiple lines"));
    assert!(!out.stdout.lines().any(|l| l.starts_with("1: ")));
}

#[test]
fn get_full_path_shows_dot_slash_path_when_file_is_under_pwd() {
    let env = env_with_get_fixtures();
    let out = env.run(&["get", "fixtures/test1.md", "--full-path"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.lines().any(|l| l == "./test1.md"));
    assert!(!out.stdout.contains("qmd://"));
    // Body still present and line-numbered.
    assert!(out.stdout.contains("1: # Test Document 1"));
}

#[test]
fn get_full_path_shows_absolute_path_when_file_is_outside_pwd() {
    let env = env_with_get_fixtures();
    let out = env.run_at(&["get", "fixtures/test1.md", "--full-path"], Path::new("/"));
    assert_eq!(out.exit_code, 0);
    assert!(
        out.stdout
            .lines()
            .any(|l| l.starts_with('/') && l.ends_with("/test1.md"))
    );
    assert!(!out.stdout.contains("qmd://"));
}

#[test]
fn get_full_path_falls_back_to_qmd_uri_plus_docid_when_file_is_gone() {
    let env = CliEnv::new();
    let corpus = env.root.join("gone-fixtures");
    std::fs::create_dir_all(&corpus).unwrap();
    let gone = corpus.join("gone.md");
    std::fs::write(&gone, "# Gone\n\nbody line\n").unwrap();
    env.write_config("index", &collection_yaml("gonecoll", &corpus, "**/*.md"));
    env.update();
    std::fs::remove_file(&gone).unwrap();

    let out = env.run(&["get", "gonecoll/gone.md", "--full-path"]);
    assert_eq!(out.exit_code, 0);
    let first = out.stdout.lines().next().unwrap();
    assert!(first.starts_with("qmd://gonecoll/gone.md"));
    assert!(first.contains('#'));
}

// ── get command missing and ignored paths ────────────────────────────

/// Environment with `gettst` over a corpus where `private/` is ignored.
fn env_with_ignored() -> CliEnv {
    let env = CliEnv::new();
    let corpus = env.root.join("get-missing-ignored-fixtures");
    std::fs::create_dir_all(corpus.join("public")).unwrap();
    std::fs::create_dir_all(corpus.join("private")).unwrap();
    std::fs::write(
        corpus.join("public/prompt-log.md"),
        "# Public Log\nThis indexed file must not be returned for missing paths.",
    )
    .unwrap();
    std::fs::write(
        corpus.join("private/log.md"),
        "# Private Log\nThis ignored file must not be indexed.",
    )
    .unwrap();
    env.write_config(
        "index",
        &format!(
            "collections:\n  gettst:\n    path: {}\n    pattern: \"**/*.md\"\n    ignore:\n      - \"private/**\"\n",
            corpus.display()
        ),
    );
    env.update();
    env
}

#[test]
fn get_reports_ignored_paths_instead_of_fuzzy_match() {
    let env = env_with_ignored();
    let out = env.run_at(
        &["get", "private/log.md"],
        &env.root.join("get-missing-ignored-fixtures"),
    );
    assert_eq!(out.exit_code, 1);
    assert!(!out.stdout.contains("Public Log"));
    assert!(out.stderr.contains("excluded by ignore rule"));
    assert!(out.stderr.contains("private/log.md"));
    assert!(out.stderr.contains("private/**"));
}

#[test]
fn get_does_not_fall_back_to_same_basename_fuzzy_match_for_missing() {
    let env = env_with_ignored();
    let out = env.run_at(
        &["get", "missing/log.md"],
        &env.root.join("get-missing-ignored-fixtures"),
    );
    assert_eq!(out.exit_code, 1);
    assert!(!out.stdout.contains("Public Log"));
    assert!(out.stderr.contains("Document not found"));
}
