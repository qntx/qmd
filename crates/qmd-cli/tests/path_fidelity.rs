//! Port of upstream `test/path-fidelity.test.ts` describe `Path
//! fidelity — CLI integration` — literal paths with spaces, `#`, `&`,
//! `@`, `[]`, `()` round-trip through index → search → get →
//! `--full-path` → `ls` (never handalized slugs). The store-level and
//! migration describes of that file are covered by library tests;
//! `collection add` is replaced by library-driven indexing.

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

use std::path::PathBuf;

use common::{CliEnv, collection_yaml};

/// The crazy-filename corpus of upstream `createCrazyCollection`.
fn env_with_crazy() -> CliEnv {
    let env = CliEnv::new();
    env.write_file(
        "# Meeting - 234232 3432 __ 5.md",
        "# Meeting - 234232 3432 // 5\n\nSome meeting content with searchterm-alpha.\n",
    );
    env.write_file(
        "Budget & Revenue (Q4) [2024].md",
        "# Budget & Revenue Q4 2024\n\nFinancial overview searchterm-beta.\n",
    );
    env.write_file(
        "normal-file.md",
        "# Normal File\n\nPlain filename, should always work.\n",
    );
    env.write_file(
        "subdir/Notes #42 - foo@bar.md",
        "# Notes #42\n\nSubdir file with searchterm-gamma.\n",
    );
    env.write_config(
        "index",
        &collection_yaml("crazytest", &env.fixtures, "**/*.md"),
    );
    env.update();
    env
}

#[test]
fn search_json_file_field_contains_literal_path_not_slug() {
    let env = env_with_crazy();
    let out = env.run(&["search", "searchterm-alpha", "--json"]);
    assert_eq!(out.exit_code, 0);
    let results: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let meeting = results
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["file"].as_str().unwrap().contains("Meeting"))
        .expect("meeting result");
    let file = meeting["file"].as_str().unwrap();
    assert!(file.contains("# Meeting - 234232 3432 __ 5.md"));
    assert!(!file.contains("Meeting-234232-3432-5.md"));
}

#[test]
fn get_full_path_resolves_to_real_filesystem_path_for_crazy_named_file() {
    let env = env_with_crazy();
    let out = env.run(&[
        "get",
        "qmd://crazytest/Budget & Revenue (Q4) [2024].md",
        "--full-path",
    ]);
    assert_eq!(out.exit_code, 0, "stdout: {}", out.stdout);

    let header = out.stdout.lines().next().unwrap();
    assert!(!header.starts_with("qmd://"));
    assert!(header.contains("Budget & Revenue (Q4) [2024].md"));
    // The resolved path must exist — strip a trailing `#docid` if any.
    let fs_path = header.trim().split(" #").next().unwrap();
    let abs = if fs_path.starts_with('/') {
        PathBuf::from(fs_path)
    } else {
        env.fixtures
            .join(fs_path.strip_prefix("./").unwrap_or(fs_path))
    };
    assert!(
        abs.exists(),
        "resolved path does not exist: {}",
        abs.display()
    );
}

#[test]
fn get_actual_fs_path_finds_the_document() {
    let env = env_with_crazy();
    let fs_path = env.fixtures.join("Budget & Revenue (Q4) [2024].md");
    let out = env.run(&["get", &fs_path.to_string_lossy()]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Budget & Revenue (Q4) [2024].md"));
}

#[test]
fn get_actual_fs_path_finds_subdir_file_with_crazy_name() {
    let env = env_with_crazy();
    let fs_path = env.fixtures.join("subdir/Notes #42 - foo@bar.md");
    let out = env.run(&["get", &fs_path.to_string_lossy()]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Notes #42 - foo@bar.md"));
}

#[test]
fn ls_shows_literal_paths_not_slugs() {
    let env = env_with_crazy();
    let out = env.run(&["ls", "crazytest"]);
    assert_eq!(out.exit_code, 0);

    assert!(out.stdout.contains("# Meeting - 234232 3432 __ 5.md"));
    assert!(out.stdout.contains("Budget & Revenue (Q4) [2024].md"));
    assert!(out.stdout.contains("Notes #42 - foo@bar.md"));

    assert!(!out.stdout.contains("Meeting-234232-3432-5.md"));
    assert!(!out.stdout.contains("Budget-Revenue-Q4-2024.md"));
    assert!(!out.stdout.contains("Notes-42-foo-bar.md"));
}

#[test]
fn search_json_returns_docid_that_can_be_fetched_back() {
    let env = env_with_crazy();
    let search = env.run(&["search", "searchterm-beta", "--json"]);
    assert_eq!(search.exit_code, 0);
    let results: serde_json::Value = serde_json::from_str(&search.stdout).unwrap();
    let hit = &results.as_array().unwrap()[0];
    let docid = hit["docid"].as_str().unwrap();
    assert!(docid.starts_with('#') && docid.len() == 7);

    let get = env.run(&["get", docid]);
    assert_eq!(get.exit_code, 0);
    assert!(get.stdout.contains("Budget & Revenue (Q4) [2024].md"));
}

#[test]
fn normal_filenames_are_still_stored_correctly() {
    let env = env_with_crazy();
    let out = env.run(&["search", "Plain filename", "--json"]);
    assert_eq!(out.exit_code, 0);
    let results: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let hit = results
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["file"].as_str().unwrap().contains("normal-file"))
        .expect("normal-file result");
    assert!(hit["file"].as_str().unwrap().contains("normal-file.md"));
}
