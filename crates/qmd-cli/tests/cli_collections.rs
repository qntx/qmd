//! Ports of upstream `test/cli.test.ts` describe blocks `CLI Add
//! Command`, `CLI Collection Commands` and `collection ignore
//! patterns`, plus the `collection list` half of `status and
//! collection list hide filesystem paths`.
//!
//! Upstream's `createIsolatedTestEnv` writes `collections: {}\n` into
//! the config dir before each run; `CliEnv::write_config` does the
//! same here. `INDEX_PATH` stands in for the upstream `dbPath`
//! override.

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

use common::{CliEnv, collection_yaml, fixtures_corpus};

/// Upstream `testDir` corpus under `fixtures/` — the `collection add .`
/// subject most tests share.
fn env_with_corpus() -> CliEnv {
    let env = CliEnv::new();
    fixtures_corpus(&env);
    env
}

/// Upstream `createIsolatedTestEnv`: corpus dir plus a
/// `collections: {}` config file.
fn isolated_env() -> CliEnv {
    let env = env_with_corpus();
    env.write_config("index", "collections: {}\n");
    env
}

// ── CLI Add Command ─────────────────────────────────────────────────

#[test]
fn adds_files_from_current_directory() {
    let env = isolated_env();
    let out = env.run(&["collection", "add", "."]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Collection:"), "{}", out.stdout);
    assert!(out.stdout.contains("Indexed:"), "{}", out.stdout);
    assert!(
        out.stdout
            .contains("Collection 'fixtures' created successfully"),
        "{}",
        out.stdout
    );
}

/// #684 — a bare `collection add` must not silently index CWD.
#[test]
fn requires_a_path_argument() {
    let env = isolated_env();
    let out = env.run(&["collection", "add"]);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr.contains("Usage: qmd collection add <path>"),
        "{}",
        out.stderr
    );
    assert!(
        out.stderr
            .contains("Pass '.' to index the current directory"),
        "{}",
        out.stderr
    );
    assert!(!out.stdout.contains("created successfully"));
    assert_eq!(
        fs::read_to_string(env.config_dir.join("index.yml")).unwrap(),
        "collections: {}\n"
    );

    let list = env.run(&["collection", "list"]);
    assert!(
        list.stdout.contains("No collections found"),
        "{}",
        list.stdout
    );
}

#[test]
fn rejects_a_nonexistent_collection_path_without_persisting_it() {
    let env = isolated_env();
    let missing = env.fixtures.join("does-not-exist");
    let missing = missing.to_string_lossy().into_owned();

    let out = env.run(&["collection", "add", &missing]);
    assert_ne!(out.exit_code, 0);
    assert!(
        out.stderr.contains("Collection path does not exist"),
        "{}",
        out.stderr
    );
    assert!(
        out.stderr.contains(&format!("Received: {missing}")),
        "{}",
        out.stderr
    );
    assert!(out.stderr.contains("Resolved:"), "{}", out.stderr);
    assert!(
        out.stderr.contains("Provide an existing directory"),
        "{}",
        out.stderr
    );
    assert!(!out.stdout.contains("created successfully"));
    assert_eq!(
        fs::read_to_string(env.config_dir.join("index.yml")).unwrap(),
        "collections: {}\n"
    );

    let list = env.run(&["collection", "list"]);
    assert!(!list.stdout.contains("does-not-exist"), "{}", list.stdout);
}

/// A path string that merely *looks* like flags is a path, not options.
#[test]
fn rejects_swallowed_collection_flags_without_persisting() {
    let env = isolated_env();
    let malformed = format!(
        "{} --name swallowed --mask *.md",
        env.fixtures.join("missing").display()
    );
    let derived_name = "missing --name swallowed --mask *.md";

    let out = env.run(&["collection", "add", &malformed]);
    assert_ne!(out.exit_code, 0);
    assert!(
        out.stderr.contains("Collection path does not exist"),
        "{}",
        out.stderr
    );
    assert!(
        out.stderr.contains(&format!("Received: {malformed}")),
        "{}",
        out.stderr
    );
    assert!(out.stderr.contains("Resolved:"), "{}", out.stderr);
    assert!(!out.stdout.contains("created successfully"));
    assert_eq!(
        fs::read_to_string(env.config_dir.join("index.yml")).unwrap(),
        "collections: {}\n"
    );

    let list = env.run(&["collection", "list"]);
    assert!(!list.stdout.contains(derived_name), "{}", list.stdout);
}

#[test]
fn rejects_a_regular_file_as_collection_path() {
    let env = isolated_env();
    let file = env.fixtures.join("README.md");
    let file = file.to_string_lossy().into_owned();

    let out = env.run(&["collection", "add", &file]);
    assert_ne!(out.exit_code, 0);
    assert!(
        out.stderr.contains("Collection path is not a directory"),
        "{}",
        out.stderr
    );
    assert!(
        out.stderr.contains(&format!("Received: {file}")),
        "{}",
        out.stderr
    );
    assert!(out.stderr.contains("Resolved:"), "{}", out.stderr);
    assert!(out.stderr.contains("Choose a directory"), "{}", out.stderr);
    assert!(!out.stdout.contains("created successfully"));
    assert_eq!(
        fs::read_to_string(env.config_dir.join("index.yml")).unwrap(),
        "collections: {}\n"
    );

    let list = env.run(&["collection", "list"]);
    assert!(!list.stdout.contains("README.md"), "{}", list.stdout);
}

#[test]
fn adds_files_with_custom_glob_pattern() {
    let env = isolated_env();
    let out = env.run(&["collection", "add", ".", "--mask", "notes/*.md"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Collection:"), "{}", out.stdout);
    assert!(out.stdout.contains("notes/*.md"), "{}", out.stdout);
}

/// #536 — `--glob` aliases `--mask`; same path with a different pattern
/// is a different collection.
#[test]
fn accepts_glob_alias_for_same_path_collections() {
    let env = isolated_env();
    let dir = env.root.join("glob-alias");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("MEMORY.md"), "root memory\n").unwrap();
    fs::write(dir.join("memory.md"), "alt memory\n").unwrap();
    fs::write(dir.join("other.md"), "other\n").unwrap();
    let dir_arg = dir.to_string_lossy().into_owned();

    let root = env.run(&["collection", "add", &dir_arg, "--name", "memory-root"]);
    assert_eq!(root.exit_code, 0, "stderr: {}", root.stderr);

    let alt = env.run(&[
        "collection",
        "add",
        &dir_arg,
        "--name",
        "memory-alt",
        "--glob",
        "memory.md",
    ]);
    assert_eq!(alt.exit_code, 0, "stderr: {}", alt.stderr);
    assert!(
        !alt.stderr
            .contains("already exists for this path and pattern"),
        "{}",
        alt.stderr
    );
    assert!(alt.stdout.contains("memory.md"), "{}", alt.stdout);

    let list = env.run(&["collection", "list"]);
    assert!(list.stdout.contains("memory-root"), "{}", list.stdout);
    assert!(list.stdout.contains("memory-alt"), "{}", list.stdout);
    assert!(list.stdout.contains("memory.md"), "{}", list.stdout);
}

/// #557 — a comma-separated mask is a union of patterns.
#[test]
fn indexes_comma_separated_mask_patterns_as_union() {
    let env = isolated_env();
    let dir = env.root.join("comma-mask");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("a.md"), "alpha\n").unwrap();
    fs::write(dir.join("X - b.md"), "bravo\n").unwrap();
    fs::write(dir.join("skip.txt"), "nope\n").unwrap();
    let dir_arg = dir.to_string_lossy().into_owned();

    let out = env.run(&[
        "collection",
        "add",
        &dir_arg,
        "--name",
        "bug-demo",
        "--mask",
        "a.md,X - *.md",
    ]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Indexed: 2 new"), "{}", out.stdout);
}

/// #557 — brace expansion still works alongside comma masks.
#[test]
fn still_accepts_brace_expansion_mask() {
    let env = isolated_env();
    let dir = env.root.join("brace-mask");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("a.md"), "alpha\n").unwrap();
    fs::write(dir.join("X - b.md"), "bravo\n").unwrap();
    let dir_arg = dir.to_string_lossy().into_owned();

    let out = env.run(&[
        "collection",
        "add",
        &dir_arg,
        "--name",
        "bug-demo",
        "--mask",
        "{a.md,X - *.md}",
    ]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Indexed: 2 new"), "{}", out.stdout);
}

/// #460 — an unreadable file warns on stderr and indexing continues.
#[test]
#[cfg(unix)]
fn warns_and_continues_when_a_file_is_unreadable() {
    use std::os::unix::fs::PermissionsExt;

    let env = isolated_env();
    let dir = env.root.join("skip-unreadable");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("good.md"), "alpha\n").unwrap();
    let bad = dir.join("bad.md");
    fs::write(&bad, "bravo\n").unwrap();
    fs::set_permissions(&bad, fs::Permissions::from_mode(0o000)).unwrap();

    if fs::read_to_string(&bad).is_ok() {
        // Root (or ACL override) reads mode-0 files — same early return
        // upstream makes.
        fs::set_permissions(&bad, fs::Permissions::from_mode(0o644)).ok();
        return;
    }

    let dir_arg = dir.to_string_lossy().into_owned();
    let out = env.run(&["collection", "add", &dir_arg, "--name", "skip-unreadable"]);
    fs::set_permissions(&bad, fs::Permissions::from_mode(0o644)).ok();
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Indexed: 1 new"), "{}", out.stdout);
    let unreadable =
        regex::Regex::new(r"Skipped unreadable file: bad\.md \((?:EACCES|EPERM)\)").unwrap();
    assert!(unreadable.is_match(&out.stderr), "{}", out.stderr);
    assert!(
        out.stderr.contains("Skipped 1 unreadable file(s)"),
        "{}",
        out.stderr
    );
}

#[test]
fn can_recreate_collection_with_remove_and_add() {
    let env = isolated_env();
    let first = env.run(&["collection", "add", "."]);
    assert_eq!(first.exit_code, 0, "stderr: {}", first.stderr);
    let rm = env.run(&["collection", "remove", "fixtures"]);
    assert_eq!(rm.exit_code, 0, "stderr: {}", rm.stderr);
    let out = env.run(&["collection", "add", "."]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout
            .contains("Collection 'fixtures' created successfully"),
        "{}",
        out.stdout
    );
}

// ── CLI Collection Commands ─────────────────────────────────────────

/// `runQmd(["collection","add","."])` beforeEach.
fn env_with_fixtures_collection() -> CliEnv {
    let env = env_with_corpus();
    let out = env.run(&["collection", "add", "."]);
    assert_eq!(out.exit_code, 0, "setup stderr: {}", out.stderr);
    env
}

#[test]
fn lists_collections() {
    let env = env_with_fixtures_collection();
    let out = env.run(&["collection", "list"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Collections"), "{}", out.stdout);
    assert!(out.stdout.contains("fixtures"), "{}", out.stdout);
    assert!(out.stdout.contains("qmd://fixtures/"), "{}", out.stdout);
    assert!(out.stdout.contains("Pattern:"), "{}", out.stdout);
    assert!(out.stdout.contains("Files:"), "{}", out.stdout);
}

#[test]
fn removes_a_collection() {
    let env = env_with_fixtures_collection();
    let before = env.run(&["collection", "list"]);
    assert!(before.stdout.contains("fixtures"));

    let out = env.run(&["collection", "remove", "fixtures"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("✓ Removed collection 'fixtures'"),
        "{}",
        out.stdout
    );
    assert!(out.stdout.contains("Deleted"), "{}", out.stdout);

    let after = env.run(&["collection", "list"]);
    assert!(!after.stdout.contains("fixtures"), "{}", after.stdout);
}

#[test]
fn handles_removing_nonexistent_collection() {
    let env = env_with_fixtures_collection();
    let out = env.run(&["collection", "remove", "nonexistent"]);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr.contains("Collection not found"),
        "{}",
        out.stderr
    );
}

#[test]
fn handles_missing_remove_argument() {
    let env = env_with_fixtures_collection();
    let out = env.run(&["collection", "remove"]);
    assert_eq!(out.exit_code, 1);
    assert!(out.stderr.contains("Usage:"), "{}", out.stderr);
}

#[test]
fn handles_unknown_subcommand() {
    let env = env_with_fixtures_collection();
    let out = env.run(&["collection", "invalid"]);
    assert_eq!(out.exit_code, 1);
    assert!(out.stderr.contains("Unknown subcommand"), "{}", out.stderr);
}

#[test]
fn renames_a_collection() {
    let env = env_with_fixtures_collection();
    let out = env.run(&["collection", "rename", "fixtures", "my-fixtures"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout
            .contains("✓ Renamed collection 'fixtures' to 'my-fixtures'"),
        "{}",
        out.stdout
    );
    assert!(out.stdout.contains("qmd://fixtures/"), "{}", out.stdout);
    assert!(out.stdout.contains("qmd://my-fixtures/"), "{}", out.stdout);

    let after = env.run(&["collection", "list"]);
    assert!(
        after.stdout.contains("qmd://my-fixtures/"),
        "{}",
        after.stdout
    );
    assert!(
        !after.stdout.contains("qmd://fixtures/"),
        "{}",
        after.stdout
    );
}

#[test]
fn handles_renaming_nonexistent_collection() {
    let env = env_with_fixtures_collection();
    let out = env.run(&["collection", "rename", "nonexistent", "newname"]);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr.contains("Collection not found"),
        "{}",
        out.stderr
    );
}

#[test]
fn handles_renaming_to_existing_collection_name() {
    let env = env_with_fixtures_collection();
    let second = env.root.join("second");
    fs::create_dir_all(&second).unwrap();
    fs::write(second.join("test.md"), "# Test\n").unwrap();
    let second_arg = second.to_string_lossy().into_owned();
    let add = env.run(&["collection", "add", &second_arg, "--name", "second"]);
    assert_eq!(add.exit_code, 0, "stderr: {}", add.stderr);

    let both = env.run(&["collection", "list"]);
    assert!(both.stdout.contains("qmd://fixtures/"), "{}", both.stdout);
    assert!(both.stdout.contains("qmd://second/"), "{}", both.stdout);

    let out = env.run(&["collection", "rename", "fixtures", "second"]);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr.contains("Collection name already exists"),
        "{}",
        out.stderr
    );
}

#[test]
fn handles_missing_rename_arguments() {
    let env = env_with_fixtures_collection();
    let bare = env.run(&["collection", "rename"]);
    assert_eq!(bare.exit_code, 1);
    assert!(bare.stderr.contains("Usage:"), "{}", bare.stderr);

    let one_arg = env.run(&["collection", "rename", "fixtures"]);
    assert_eq!(one_arg.exit_code, 1);
    assert!(one_arg.stderr.contains("Usage:"), "{}", one_arg.stderr);
}

// ── collection ignore patterns ──────────────────────────────────────

/// Upstream `ignore-fixtures` tree plus a `ignoretst` collection with
/// `sessions/**`/`archive/**` ignores.
fn env_with_ignore_collection() -> CliEnv {
    let env = CliEnv::new();
    let dir = env.root.join("ignore-fixtures");
    fs::create_dir_all(dir.join("notes")).unwrap();
    fs::create_dir_all(dir.join("sessions/2026-03")).unwrap();
    fs::create_dir_all(dir.join("archive")).unwrap();
    fs::write(
        dir.join("readme.md"),
        "# Main readme\nThis should be indexed.",
    )
    .unwrap();
    fs::write(
        dir.join("notes/note1.md"),
        "# Note 1\nThis is a personal note.",
    )
    .unwrap();
    fs::write(
        dir.join("sessions/session1.md"),
        "# Session 1\nThis session should be ignored.",
    )
    .unwrap();
    fs::write(
        dir.join("sessions/2026-03/session2.md"),
        "# Session 2\nNested session should also be ignored.",
    )
    .unwrap();
    fs::write(
        dir.join("archive/old.md"),
        "# Old stuff\nThis archive file should be ignored.",
    )
    .unwrap();
    env.write_config(
        "index",
        &format!(
            "collections:\n  ignoretst:\n    path: {}\n    pattern: \"**/*.md\"\n    ignore:\n      - \"sessions/**\"\n      - \"archive/**\"\n",
            dir.display()
        ),
    );
    env
}

#[test]
fn ignore_patterns_exclude_matching_files_from_indexing() {
    let env = env_with_ignore_collection();
    let out = env.run(&["update"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("2 new"), "{}", out.stdout);
}

#[test]
fn ignored_files_are_not_searchable() {
    let env = env_with_ignore_collection();
    let update = env.run(&["update"]);
    assert_eq!(update.exit_code, 0, "stderr: {}", update.stderr);
    let out = env.run(&["search", "session", "-n", "10"]);
    if out.exit_code == 0 {
        assert!(!out.stdout.contains("session1"), "{}", out.stdout);
        assert!(!out.stdout.contains("session2"), "{}", out.stdout);
    }
}

#[test]
fn non_ignored_files_are_searchable() {
    let env = env_with_ignore_collection();
    let update = env.run(&["update"]);
    assert_eq!(update.exit_code, 0, "stderr: {}", update.stderr);
    let out = env.run(&["search", "personal note", "-n", "10"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("note1"), "{}", out.stdout);
}

#[test]
fn collection_list_shows_ignore_patterns() {
    let env = env_with_ignore_collection();
    let update = env.run(&["update"]);
    assert_eq!(update.exit_code, 0, "stderr: {}", update.stderr);
    let out = env.run(&["collection", "list"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Ignore:"), "{}", out.stdout);
    assert!(out.stdout.contains("sessions/**"), "{}", out.stdout);
    assert!(out.stdout.contains("archive/**"), "{}", out.stdout);
}

#[test]
fn collection_without_ignore_indexes_all_files() {
    let env = env_with_ignore_collection();
    env.write_config(
        "index",
        &format!(
            "collections:\n  allfiles:\n    path: {}\n    pattern: \"**/*.md\"\n",
            env.root.join("ignore-fixtures").display()
        ),
    );
    let out = env.run(&["update"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("5 new"), "{}", out.stdout);
}

// ── status and collection list hide filesystem paths ────────────────

#[test]
fn collection_list_hides_filesystem_paths() {
    let env = env_with_fixtures_collection();
    let out = env.run(&["collection", "list"]);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("qmd://fixtures/"), "{}", out.stdout);
    let path_line = regex::Regex::new(r"Path:\s+/").unwrap();
    assert!(!path_line.is_match(&out.stdout), "{}", out.stdout);
}

// ── lazy store opening (upstream getStore placement) ────────────────

/// `update-cmd`/`include`/`exclude`/`show` are pure-YAML upstream
/// (`getCollection`/`updateCollectionSettings`, collections.ts:244-311)
/// — the database must never appear, success or failure.
#[test]
fn yaml_only_collection_commands_do_not_create_the_index() {
    let env = CliEnv::new();
    env.write_file("a.md", "# A\n");
    env.write_config("index", &collection_yaml("notes", &env.fixtures, "**/*.md"));

    for args in [
        ["collection", "update-cmd", "notes", "echo hi"].as_slice(),
        ["collection", "update-cmd", "notes"].as_slice(),
        ["collection", "exclude", "notes"].as_slice(),
        ["collection", "include", "notes"].as_slice(),
        ["collection", "show", "notes"].as_slice(),
    ] {
        let out = env.run(args);
        assert_eq!(out.exit_code, 0, "{args:?}: {}", out.stderr);
        assert!(
            !env.db_path.exists(),
            "{args:?} created {}",
            env.db_path.display()
        );
    }
    let yaml = fs::read_to_string(env.config_dir.join("index.yml")).unwrap();
    assert!(!yaml.contains("update:"), "{yaml}");
    assert!(!yaml.contains("includeByDefault"), "{yaml}");

    // Missing-collection variants also stop before `getDb`.
    for args in [
        ["collection", "update-cmd", "missing", "x"].as_slice(),
        ["collection", "show", "missing"].as_slice(),
        ["collection", "remove", "missing"].as_slice(),
        ["collection", "rename", "missing", "other"].as_slice(),
        ["collection", "rename", "notes", "notes"].as_slice(),
    ] {
        let out = env.run(args);
        assert_eq!(out.exit_code, 1, "{args:?}: {}", out.stdout);
        assert!(
            !env.db_path.exists(),
            "{args:?} created {}",
            env.db_path.display()
        );
    }
}

/// `collectionAdd`'s validation and YAML duplicate checks run before
/// `resyncConfig` (qmd.ts:4529-4547, 1837-1854) — a rejected add never
/// creates the index.
#[test]
fn collection_add_errors_do_not_create_the_index() {
    let env = CliEnv::new();
    env.write_file("a.md", "# A\n");
    env.write_config("index", &collection_yaml("notes", &env.fixtures, "**/*.md"));

    let cases: [&[&str]; 4] = [
        &["collection", "add", "no-such-dir"],
        &["collection", "add", "a.md"],
        &["collection", "add", ".", "--name", "notes"],
        &["collection", "add", ".", "--name", "fresh"],
    ];
    for args in cases {
        let out = env.run(args);
        assert_eq!(out.exit_code, 1, "{args:?}: {}", out.stdout);
        assert!(
            !env.db_path.exists(),
            "{args:?} created {}",
            env.db_path.display()
        );
    }
    // ...and each failure text is the upstream one.
    let missing = env.run(&["collection", "add", "no-such-dir"]);
    assert!(
        missing.stderr.contains("does not exist"),
        "{}",
        missing.stderr
    );
    let dup = env.run(&["collection", "add", ".", "--name", "fresh"]);
    assert!(
        dup.stderr
            .contains("already exists for this path and pattern"),
        "{}",
        dup.stderr
    );
}

/// `--name ""` is falsy upstream (`if (!collName)`, qmd.ts:1831) and
/// falls back to the basename. The `/` → `root` edge is covered by the
/// `collection_name` unit test (indexing `/` in a test would scan the
/// filesystem).
#[test]
fn collection_add_empty_name_falls_back_to_basename() {
    let env = isolated_env();
    let out = env.run(&["collection", "add", ".", "--name", ""]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("Creating collection 'fixtures'"),
        "{}",
        out.stdout
    );
    let show = env.run(&["collection", "show", "fixtures"]);
    assert_eq!(show.exit_code, 0, "stderr: {}", show.stderr);
}

/// `if (col.update)` is JS truthiness (qmd.ts:4640): an empty
/// `update: ""` prints no `Update:` line.
#[test]
fn collection_show_skips_empty_update() {
    let env = CliEnv::new();
    env.write_file("a.md", "# A\n");
    env.write_config(
        "index",
        &format!(
            "collections:\n  notes:\n    path: {}\n    pattern: \"**/*.md\"\n    update: \"\"\n",
            env.fixtures.display()
        ),
    );
    let out = env.run(&["collection", "show", "notes"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(!out.stdout.contains("Update:"), "{}", out.stdout);
    assert!(!env.db_path.exists());
}
