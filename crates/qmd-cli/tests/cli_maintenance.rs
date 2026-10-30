//! Ports of upstream `test/cli.test.ts` describe blocks `CLI Init
//! Command`, `CLI Update Command`, `CLI Cleanup Command`, the two
//! non-`doctor` `CLI Status Command` assertions, and the `status` half
//! of `status and collection list hide filesystem paths`.
//!
//! Deviation: the project-local database is `.qmd/index-rs.sqlite`
//! (upstream `.qmd/index.sqlite`) so Rust and Bun indexes do not
//! collide in one checkout.

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

use std::fs;

use common::{CliEnv, fixtures_corpus};

/// Upstream `beforeEach`: corpus plus `collection add .`.
fn env_with_fixtures_collection() -> CliEnv {
    let env = CliEnv::new();
    fixtures_corpus(&env);
    let out = env.run(&["collection", "add", "."]);
    assert_eq!(out.exit_code, 0, "setup stderr: {}", out.stderr);
    env
}

// ── CLI Init Command ────────────────────────────────────────────────

#[test]
fn init_creates_a_project_local_qmd_index() {
    let env = CliEnv::new();
    let project = env.root.join("init-project");
    fs::create_dir_all(&project).unwrap();

    let out = env.run_at(&["init"], &project);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.stdout.trim(), "ready to go with new local index");
    assert!(project.join(".qmd/index.yml").exists());
    assert!(project.join(".qmd/index-rs.sqlite").exists());
    let config = fs::read_to_string(project.join(".qmd/index.yml")).unwrap();
    assert!(config.contains("collections: {}"), "{config}");
    assert!(config.contains("models:"), "{config}");
}

#[test]
fn init_refuses_to_initialize_in_home() {
    let env = CliEnv::new();
    let fake_home = env.root.join("init-home");
    fs::create_dir_all(&fake_home).unwrap();

    let out = env.run_opts(
        &["init"],
        &fake_home,
        true,
        &[("HOME", &fake_home.to_string_lossy())],
    );
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr
            .contains("Refusing to initialize a local index in $HOME"),
        "{}",
        out.stderr
    );
    assert!(
        out.stderr.contains("global index is automatically created"),
        "{}",
        out.stderr
    );
    assert!(!fake_home.join(".qmd/index.yml").exists());
}

// ── CLI Status Command ──────────────────────────────────────────────

#[test]
fn shows_index_status() {
    let env = env_with_fixtures_collection();
    let out = env.run(&["status"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Collection"), "{}", out.stdout);
}

/// The `AST Chunking` block (qmd.ts:600-626): statically linked
/// grammars are all `available`, so the `active` branch prints —
/// upstream's `Unavailable:`/`unavailable` rows are unreachable.
#[cfg(feature = "ast")]
#[test]
fn status_reports_ast_chunking_active() {
    let env = env_with_fixtures_collection();
    let out = env.run(&["status"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("AST Chunking"), "{}", out.stdout);
    assert!(out.stdout.contains("Status:   active"), "{}", out.stdout);
    assert!(
        out.stdout
            .contains("Languages: typescript, tsx, javascript, python, go, rust"),
        "{}",
        out.stdout
    );
}

/// `doctor` owns GPU diagnostics — status stays silent about devices.
#[test]
fn status_omits_device_probing_details() {
    let env = env_with_fixtures_collection();
    let out = env.run(&["status"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(!out.stdout.contains("Device"), "{}", out.stdout);
    assert!(
        !out.stdout.contains("QMD_STATUS_DEVICE_PROBE"),
        "{}",
        out.stdout
    );
    assert!(!out.stdout.contains("not probed"), "{}", out.stdout);
}

// ── CLI Update Command ──────────────────────────────────────────────

#[test]
fn updates_all_collections() {
    let env = env_with_fixtures_collection();
    let out = env.run(&["update"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Updating"), "{}", out.stdout);
}

#[test]
fn update_deactivates_stale_docs_when_collection_has_zero_matching_files() {
    let env = CliEnv::new();
    let collection_dir = env.root.join("update-empty");
    fs::create_dir_all(&collection_dir).unwrap();
    let doc = collection_dir.join("only.md");
    let token = "stale-proof-token";
    fs::write(
        &doc,
        format!("---\ndate: 2026-03-06\n---\n# Empty Collection Deactivation\n{token}\n"),
    )
    .unwrap();

    let dir_arg = collection_dir.to_string_lossy().into_owned();
    let add = env.run(&["collection", "add", &dir_arg, "--name", "empty-check"]);
    assert_eq!(add.exit_code, 0, "stderr: {}", add.stderr);

    let before = env.run(&["get", "qmd://empty-check/only.md"]);
    assert_eq!(before.exit_code, 0, "stderr: {}", before.stderr);
    assert!(before.stdout.contains(token), "{}", before.stdout);

    fs::remove_file(&doc).unwrap();

    let update = env.run(&["update"]);
    assert_eq!(update.exit_code, 0, "stderr: {}", update.stderr);
    assert!(
        update
            .stdout
            .contains("0 new, 0 updated, 0 unchanged, 1 removed"),
        "{}",
        update.stdout
    );

    let after = env.run(&["get", "qmd://empty-check/only.md"]);
    assert_eq!(after.exit_code, 1);
}

// ── CLI Cleanup Command ─────────────────────────────────────────────

#[test]
fn cleans_up_orphaned_entries() {
    let env = env_with_fixtures_collection();
    let out = env.run(&["cleanup"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
}

// ── status and collection list hide filesystem paths ────────────────

#[test]
fn status_hides_filesystem_paths() {
    let env = env_with_fixtures_collection();
    let out = env.run(&["status"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("qmd://fixtures/"), "{}", out.stdout);
    // Every line except the `Index:` location must use qmd:// URIs.
    let path_lines: Vec<&str> = out
        .stdout
        .lines()
        .filter(|l| !l.contains("Index:"))
        .filter(|l| l.contains("/Users/") || l.contains("/home/") || l.contains("/tmp/"))
        .collect();
    assert!(path_lines.is_empty(), "{path_lines:?}");
}
