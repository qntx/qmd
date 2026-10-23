//! Port of upstream `test/cli.test.ts` describe `CLI ls Command`,
//! including the two absolute-path-collection cases (collection *names*
//! that are filesystem paths). Indexing goes through the library.

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

/// `fixtures` collection over a small corpus (upstream `collection add .`).
fn env_with_fixtures() -> CliEnv {
    let env = CliEnv::new();
    env.write_file("README.md", "# Test Project\n");
    env.write_file("notes/meeting.md", "# Team Meeting\n");
    env.write_file("notes/ideas.md", "# Product Ideas\n");
    env.write_file("docs/api.md", "# API Documentation\n");
    env.write_config(
        "index",
        &collection_yaml("fixtures", &env.fixtures, "**/*.md"),
    );
    env.update();
    env
}

#[test]
fn lists_all_collections() {
    let env = env_with_fixtures();
    let out = env.run(&["ls"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Collections:"));
    assert!(out.stdout.contains("qmd://fixtures/"));
}

#[test]
fn lists_files_in_a_collection() {
    let env = env_with_fixtures();
    let out = env.run(&["ls", "fixtures"]);
    assert_eq!(out.exit_code, 0);
    // Original case preserved.
    assert!(out.stdout.contains("qmd://fixtures/README.md"));
    assert!(out.stdout.contains("qmd://fixtures/notes/meeting.md"));
}

#[test]
fn lists_files_with_path_prefix() {
    let env = env_with_fixtures();
    let out = env.run(&["ls", "fixtures/notes"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("qmd://fixtures/notes/meeting.md"));
    assert!(out.stdout.contains("qmd://fixtures/notes/ideas.md"));
    assert!(!out.stdout.contains("qmd://fixtures/README.md"));
}

#[test]
fn lists_files_with_virtual_path() {
    let env = env_with_fixtures();
    let out = env.run(&["ls", "qmd://fixtures/docs"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("qmd://fixtures/docs/api.md"));
}

#[test]
fn continues_to_normalize_extra_slashes_for_normal_collection_virtual_paths() {
    let env = env_with_fixtures();
    let out = env.run(&["ls", "qmd:///fixtures/docs"]);
    assert_eq!(out.stderr, "");
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("qmd://fixtures/docs/api.md"));
}

#[test]
fn lists_an_absolute_path_collection_from_a_qmd_uri() {
    let env = CliEnv::new();
    let absolute_dir = env.root.join("absolute-collection");
    std::fs::create_dir_all(&absolute_dir).unwrap();
    std::fs::write(absolute_dir.join("root.md"), "# Absolute collection\n").unwrap();
    // The collection name IS the absolute path.
    env.write_config(
        "index",
        &collection_yaml(&absolute_dir.to_string_lossy(), &absolute_dir, "**/*.md"),
    );
    env.update();

    let out = env.run_at(
        &["ls", &format!("qmd://{}/", absolute_dir.display())],
        &absolute_dir,
    );
    assert_eq!(out.stderr, "");
    assert_eq!(out.exit_code, 0);
    assert!(
        out.stdout
            .contains(&format!("qmd://{}/root.md", absolute_dir.display()))
    );
}

#[test]
fn lists_an_absolute_path_collection_from_a_raw_path_longest_prefix() {
    let env = CliEnv::new();
    // Two collections whose names are absolute paths, one nested under
    // the other's prefix — the longest one must win.
    let parent_name = env.root.join("absolute-parent-name");
    let child_name = parent_name.join("nested");
    let parent_data = env.root.join("absolute-parent-data");
    let child_data = env.root.join("absolute-child-data");
    for dir in [&parent_data, &child_data] {
        std::fs::create_dir_all(dir).unwrap();
    }
    std::fs::create_dir_all(&child_name).unwrap();
    std::fs::write(parent_data.join("parent.md"), "# Parent collection\n").unwrap();
    std::fs::write(child_data.join("child.md"), "# Child collection\n").unwrap();
    env.write_config(
        "index",
        &format!(
            "collections:\n  \"{}\":\n    path: \"{}\"\n    pattern: \"**/*.md\"\n  \"{}\":\n    path: \"{}\"\n    pattern: \"**/*.md\"\n",
            parent_name.display(),
            parent_data.display(),
            child_name.display(),
            child_data.display(),
        ),
    );
    env.update();

    let out = env.run_at(&["ls", &format!("{}/", child_name.display())], &child_data);
    assert_eq!(out.stderr, "");
    assert_eq!(out.exit_code, 0);
    assert!(
        out.stdout
            .contains(&format!("qmd://{}/child.md", child_name.display()))
    );
    assert!(!out.stdout.contains("No files found"));
    assert!(
        !out.stdout
            .contains(&format!("qmd://{}/parent.md", parent_name.display()))
    );
}

#[test]
fn handles_non_existent_collection() {
    let env = env_with_fixtures();
    let out = env.run(&["ls", "nonexistent"]);
    assert_eq!(out.exit_code, 1);
    assert!(out.stderr.contains("Collection not found"));
}
