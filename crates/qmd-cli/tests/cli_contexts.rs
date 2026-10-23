//! Ports of upstream `test/cli.test.ts` describe blocks `CLI
//! Add-Context Command` and `CLI Context Management`.

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

/// Upstream `beforeEach`: corpus plus `collection add .` so the
/// `fixtures` collection exists.
fn env_with_fixtures_collection() -> CliEnv {
    let env = CliEnv::new();
    fixtures_corpus(&env);
    let out = env.run(&["collection", "add", "."]);
    assert_eq!(out.exit_code, 0, "setup stderr: {}", out.stderr);
    env
}

// ── CLI Add-Context Command ─────────────────────────────────────────

#[test]
fn adds_context_to_collection_root_via_virtual_path() {
    let env = env_with_fixtures_collection();
    let out = env.run(&[
        "context",
        "add",
        "qmd://fixtures/",
        "Personal notes and meeting logs",
    ]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("✓ Added context"), "{}", out.stdout);
}

#[test]
fn context_add_requires_text_argument() {
    let env = env_with_fixtures_collection();
    let out = env.run(&["context", "add"]);
    assert_eq!(out.exit_code, 1);
    assert!(out.stderr.contains("Usage:"), "{}", out.stderr);
}

// ── CLI Context Management ──────────────────────────────────────────

#[test]
fn add_global_context_with_slash() {
    let env = env_with_fixtures_collection();
    let out = env.run(&["context", "add", "/", "Global system context"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("✓ Set global context"),
        "{}",
        out.stdout
    );
    assert!(
        out.stdout.contains("Global system context"),
        "{}",
        out.stdout
    );
}

#[test]
fn list_contexts() {
    let env = env_with_fixtures_collection();
    let add = env.run(&["context", "add", "/", "Test context"]);
    assert_eq!(add.exit_code, 0, "stderr: {}", add.stderr);

    let out = env.run(&["context", "list"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Configured Contexts"), "{}", out.stdout);
    assert!(out.stdout.contains("Test context"), "{}", out.stdout);
}

#[test]
fn add_context_to_virtual_path() {
    let env = env_with_fixtures_collection();
    let out = env.run(&[
        "context",
        "add",
        "qmd://fixtures/notes",
        "Context for notes subdirectory",
    ]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout
            .contains("✓ Added context for: qmd://fixtures/notes"),
        "{}",
        out.stdout
    );
}

#[test]
fn remove_global_context() {
    let env = env_with_fixtures_collection();
    let add = env.run(&["context", "add", "/", "Global context to remove"]);
    assert_eq!(add.exit_code, 0, "stderr: {}", add.stderr);

    let out = env.run(&["context", "rm", "/"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("✓ Removed"), "{}", out.stdout);
}

#[test]
fn remove_virtual_path_context() {
    let env = env_with_fixtures_collection();
    let add = env.run(&[
        "context",
        "add",
        "qmd://fixtures/notes",
        "Context to remove",
    ]);
    assert_eq!(add.exit_code, 0, "stderr: {}", add.stderr);

    let out = env.run(&["context", "rm", "qmd://fixtures/notes"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout
            .contains("✓ Removed context for: qmd://fixtures/notes"),
        "{}",
        out.stdout
    );
}

#[test]
fn fails_to_remove_nonexistent_context() {
    let env = env_with_fixtures_collection();
    let out = env.run(&["context", "rm", "qmd://nonexistent/path"]);
    assert_eq!(out.exit_code, 1);
    let combined = format!("{}{}", out.stderr, out.stdout);
    assert!(combined.contains("not found"), "{combined}");
}

// ── lazy store opening (upstream contextRemove) ─────────────────────

/// The `qmd://` branch of `contextRemove` is pure `yamlRemoveContext`
/// (qmd.ts:1165-1186, collections.ts:422-441) — no `getDb`, no
/// `resyncConfig` — so the index file must never appear.
#[test]
fn context_rm_virtual_path_does_not_create_the_index() {
    let env = CliEnv::new();
    env.write_file("a.md", "# A\n");
    env.write_config(
        "index",
        &format!(
            "collections:\n  notes:\n    path: {}\n    pattern: \"**/*.md\"\n    context:\n      guide: Some context\n      sub/dir: Other\n",
            env.fixtures.display()
        ),
    );

    let out = env.run(&["context", "rm", "qmd://notes/guide"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout
            .contains("✓ Removed context for: qmd://notes/guide"),
        "{}",
        out.stdout
    );
    assert!(!env.db_path.exists(), "created {}", env.db_path.display());

    // Missing collection / missing prefix error before any store open.
    let missing_coll = env.run(&["context", "rm", "qmd://nope/x"]);
    assert_eq!(missing_coll.exit_code, 1);
    assert!(
        missing_coll.stderr.contains("Collection not found: nope"),
        "{}",
        missing_coll.stderr
    );
    let missing_prefix = env.run(&["context", "rm", "qmd://notes/nowhere"]);
    assert_eq!(missing_prefix.exit_code, 1);
    assert!(
        missing_prefix.stderr.contains("No context found"),
        "{}",
        missing_prefix.stderr
    );
    assert!(!env.db_path.exists());

    // Upstream `delete collection.context` once the map is empty.
    let last = env.run(&["context", "rm", "qmd://notes/sub/dir"]);
    assert_eq!(last.exit_code, 0, "stderr: {}", last.stderr);
    let yaml = fs::read_to_string(env.config_dir.join("index.yml")).unwrap();
    assert!(!yaml.contains("context"), "{yaml}");
    assert!(!env.db_path.exists());
}
