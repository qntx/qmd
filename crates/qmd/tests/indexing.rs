//! Integration tests for collection scanning and incremental indexing
//! (P1.2): `Qmd::update`, `split_glob_mask`, `extract_title`, symlink
//! boundaries, masks/ignores, and add/modify/delete counting.
//!
//! Ported upstream coverage: `test/store.test.ts` "Reindex Collection"
//! and "Document Helpers" groups, `test/path-fidelity.test.ts` (store
//! level, incl. the #717 separator conflict), `splitGlobMask`, and the
//! `update` parts of `test/sdk.test.ts`.

#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::shadow_unrelated,
    clippy::field_reassign_with_default,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_assert_message,
    clippy::panic,
    reason = "idiomatic test-code patterns"
)]

use std::fs;
use std::path::Path;

use qmd::{CollectionSpec, Config, Environment, Qmd, UpdateOptions};

fn write(dir: &Path, rel: &str, content: &str) {
    let path = dir.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
}

fn open(db: &Path) -> Qmd {
    Qmd::builder(db)
        .environment(Environment::default())
        .build()
        .unwrap()
}

fn open_with_docs(docs: &Path, db: &Path) -> Qmd {
    let qmd = open(db);
    qmd.add_collection(
        "docs",
        &CollectionSpec {
            path: docs.to_string_lossy().into_owned(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();
    qmd
}

fn update(qmd: &Qmd) -> qmd::UpdateReport {
    let mut events = 0usize;
    qmd.update(&UpdateOptions::default(), &mut |_| events += 1)
        .unwrap()
}

// --- splitGlobMask (upstream store.helpers) -------------------------------

#[test]
fn split_glob_mask_splits_on_top_level_commas() {
    assert_eq!(
        qmd::split_glob_mask("**/*.md,**/*.txt"),
        vec!["**/*.md", "**/*.txt"]
    );
}

#[test]
fn split_glob_mask_ignores_commas_in_braces_and_brackets() {
    assert_eq!(qmd::split_glob_mask("**/*.{md,txt}"), vec!["**/*.{md,txt}"]);
    assert_eq!(
        qmd::split_glob_mask("[a,b].md,x.md"),
        vec!["[a,b].md", "x.md"]
    );
    assert_eq!(qmd::split_glob_mask(""), vec![""]);
}

// --- extractTitle (upstream store.test.ts Document Helpers) ---------------

#[test]
fn extract_title_markdown_heading() {
    assert_eq!(
        qmd::extract_title("intro\n\n# Hello World\nbody", "doc.md"),
        "Hello World"
    );
    assert_eq!(qmd::extract_title("## Second\n", "doc.md"), "Second");
}

#[test]
fn extract_title_notes_falls_back_to_h2() {
    let content = "# Notes\n\n## Actual Title\nbody";
    assert_eq!(qmd::extract_title(content, "doc.md"), "Actual Title");
}

#[test]
fn extract_title_org() {
    assert_eq!(
        qmd::extract_title("#+TITLE: My Org Doc\n* Heading", "doc.org"),
        "My Org Doc"
    );
    assert_eq!(qmd::extract_title("* Heading\nbody", "doc.org"), "Heading");
}

#[test]
fn extract_title_filename_fallback() {
    assert_eq!(qmd::extract_title("no heading", "dir/my-doc.md"), "my-doc");
    assert_eq!(qmd::extract_title("", "plain"), "plain");
}

// --- update: basic pass (upstream Reindex Collection) ---------------------

#[test]
fn update_indexes_markdown_files() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "a.md", "# Alpha\nbody");
    write(&docs, "sub/b.md", "# Beta\nbody");
    let db = tmp.path().join("i.sqlite");
    let qmd = open_with_docs(&docs, &db);

    let report = update(&qmd);
    assert_eq!(report.collections, 1);
    assert_eq!(report.indexed, 2);
    assert_eq!((report.updated, report.removed, report.skipped), (0, 0, 0));
    assert_eq!(report.needs_embedding, 2);

    let colls = qmd.collections().unwrap();
    assert_eq!(colls[0].doc_count, 2);
    assert_eq!(colls[0].active_count, 2);
}

#[test]
fn second_update_reports_all_unchanged() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "a.md", "# Alpha\nbody");
    write(&docs, "b.md", "# Beta\nbody");
    let db = tmp.path().join("i.sqlite");
    let qmd = open_with_docs(&docs, &db);

    update(&qmd);
    let report = update(&qmd);
    assert_eq!(report.unchanged, 2);
    assert_eq!((report.indexed, report.updated, report.removed), (0, 0, 0));
}

#[test]
fn update_handles_add_modify_delete() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "keep.md", "# Keep\nbody");
    write(&docs, "gone.md", "# Gone\nbody");
    let db = tmp.path().join("i.sqlite");
    let qmd = open_with_docs(&docs, &db);
    update(&qmd);

    fs::remove_file(docs.join("gone.md")).unwrap();
    write(&docs, "keep.md", "# Keep\nchanged body");
    write(&docs, "new.md", "# New\nbody");

    let report = update(&qmd);
    assert_eq!(
        (
            report.indexed,
            report.updated,
            report.removed,
            report.unchanged
        ),
        (1, 1, 1, 0)
    );
    let colls = qmd.collections().unwrap();
    // `gone.md` is deactivated, not deleted.
    assert_eq!((colls[0].doc_count, colls[0].active_count), (3, 2));
}

#[test]
fn update_collection_filter_limits_scope() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a");
    let b = tmp.path().join("b");
    write(&a, "a.md", "# A");
    write(&b, "b.md", "# B");
    let db = tmp.path().join("i.sqlite");
    let qmd = open(&db);
    for (name, dir) in [("aa", &a), ("bb", &b)] {
        qmd.add_collection(
            name,
            &CollectionSpec {
                path: dir.to_string_lossy().into_owned(),
                ..CollectionSpec::default()
            },
        )
        .unwrap();
    }

    let report = qmd
        .update(
            &UpdateOptions {
                collections: Some(vec!["aa".to_owned()]),
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!((report.collections, report.indexed), (1, 1));
    let colls = qmd.collections().unwrap();
    assert_eq!(colls[0].active_count + colls[1].active_count, 1);
}

#[test]
fn update_emits_progress_for_every_file() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "a.md", "# A");
    write(&docs, "b.md", "# B");
    let db = tmp.path().join("i.sqlite");
    let qmd = open_with_docs(&docs, &db);

    let mut seen: Vec<(String, usize, usize)> = Vec::new();
    qmd.update(&UpdateOptions::default(), &mut |p| {
        seen.push((p.file.clone(), p.current, p.total));
        assert_eq!(p.collection, "docs");
    })
    .unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].1, 1);
    assert_eq!(seen[1].1, 2);
    assert_eq!(seen[0].2, 2);
}

// --- scanning: masks, excludes, hidden, ignores ---------------------------

#[test]
fn comma_and_brace_masks_index_matching_files() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "a.md", "# A");
    write(&docs, "b.txt", "plain text");
    write(&docs, "c.rs", "fn main() {}");
    let db = tmp.path().join("i.sqlite");
    let qmd = open(&db);

    for (name, pattern) in [("comma", "**/*.md,**/*.txt"), ("brace", "**/*.{md,txt}")] {
        qmd.add_collection(
            name,
            &CollectionSpec {
                path: docs.to_string_lossy().into_owned(),
                pattern: Some(pattern.to_owned()),
                ..CollectionSpec::default()
            },
        )
        .unwrap();
    }

    let report = update(&qmd);
    // Both collections index a.md + b.txt; c.rs never matches.
    assert_eq!(report.indexed, 4);
    assert_eq!(report.skipped, 0);
}

#[test]
fn excluded_dirs_and_hidden_files_are_skipped() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "ok.md", "# OK");
    write(&docs, "node_modules/x/y.md", "# NM");
    write(&docs, ".git/z.md", "# GIT");
    write(&docs, "dist/d.md", "# DIST");
    write(&docs, ".hidden.md", "# HIDDEN");
    write(&docs, ".hid/inner.md", "# INNER");
    let db = tmp.path().join("i.sqlite");
    let qmd = open_with_docs(&docs, &db);

    let report = update(&qmd);
    assert_eq!(report.indexed, 1);
    let colls = qmd.collections().unwrap();
    assert_eq!(colls[0].active_count, 1);
}

#[test]
fn ignore_patterns_exclude_files() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "keep.md", "# Keep");
    write(&docs, "skip/draft.md", "# Draft");
    let db = tmp.path().join("i.sqlite");
    let qmd = open(&db);
    qmd.add_collection(
        "docs",
        &CollectionSpec {
            path: docs.to_string_lossy().into_owned(),
            ignore: vec!["skip/**".to_owned()],
            ..CollectionSpec::default()
        },
    )
    .unwrap();

    let report = update(&qmd);
    assert_eq!(report.indexed, 1);
}

#[test]
fn bare_dir_ignore_pattern_excludes_subtree() {
    // Upstream fast-glob evaluates ignores against every entry, so a bare
    // directory name in `ignore` prunes the whole subtree.
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "keep.md", "# Keep");
    write(&docs, "drafts/a.md", "# A");
    write(&docs, "drafts/deep/b.md", "# B");
    let db = tmp.path().join("i.sqlite");
    let qmd = open(&db);
    qmd.add_collection(
        "docs",
        &CollectionSpec {
            path: docs.to_string_lossy().into_owned(),
            ignore: vec!["drafts".to_owned()],
            ..CollectionSpec::default()
        },
    )
    .unwrap();

    let report = update(&qmd);
    assert_eq!(report.indexed, 1);
}

#[test]
#[cfg(unix)]
fn dir_glob_ignore_pattern_prunes_subtree() {
    // `dir/**` must match the directory itself, not just its contents:
    // globset's `dir/**` compiles to `^dir/.*$`, so `keep_entry` tests
    // `dir/` too. An unreadable ignored dir proves the walker prunes it
    // instead of descending (which would error out the whole update).
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "keep.md", "# Keep");
    write(&docs, "blocked/a.md", "# A");
    let blocked = docs.join("blocked");
    fs::set_permissions(&blocked, fs::Permissions::from_mode(0o000)).unwrap();
    let db = tmp.path().join("i.sqlite");
    let qmd = open(&db);
    qmd.add_collection(
        "docs",
        &CollectionSpec {
            path: docs.to_string_lossy().into_owned(),
            ignore: vec!["blocked/**".to_owned()],
            ..CollectionSpec::default()
        },
    )
    .unwrap();

    let report = update(&qmd);
    fs::set_permissions(&blocked, fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(report.indexed, 1);
}

// --- path fidelity & file-state edge cases --------------------------------

#[test]
fn separator_conflict_files_both_indexed() {
    // Upstream path-fidelity #717: literal paths mean "a b.md" and
    // "a_b.md" stay distinct documents.
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "a b.md", "# Space");
    write(&docs, "a_b.md", "# Underscore");
    write(&docs, "a-b.md", "# Hyphen");
    let db = tmp.path().join("i.sqlite");
    let qmd = open_with_docs(&docs, &db);

    let report = update(&qmd);
    assert_eq!(report.indexed, 3);
}

#[test]
fn empty_file_is_not_indexed() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "a.md", "# A");
    write(&docs, "empty.md", "   \n  \n");
    let db = tmp.path().join("i.sqlite");
    let qmd = open_with_docs(&docs, &db);

    let report = update(&qmd);
    assert_eq!(report.indexed, 1);
    assert_eq!(report.skipped, 0);
}

#[test]
fn file_emptied_after_indexing_keeps_document_active() {
    // Upstream marks the file seen before reading, so an emptied file
    // does not deactivate its document (store.ts:1685-1687).
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "a.md", "# A\nbody");
    let db = tmp.path().join("i.sqlite");
    let qmd = open_with_docs(&docs, &db);
    update(&qmd);

    write(&docs, "a.md", "   ");
    let report = update(&qmd);
    assert_eq!(
        (report.removed, report.unchanged, report.indexed),
        (0, 0, 0)
    );
    assert_eq!(qmd.collections().unwrap()[0].active_count, 1);
}

#[cfg(unix)]
#[test]
fn unreadable_file_is_skipped_not_fatal() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "ok.md", "# OK");
    let bad = docs.join("bad.md");
    write(&docs, "bad.md", "# Bad");
    fs::set_permissions(&bad, fs::Permissions::from_mode(0o000)).unwrap();
    let db = tmp.path().join("i.sqlite");
    let qmd = open_with_docs(&docs, &db);

    let report = update(&qmd);
    assert_eq!(report.indexed, 1);
    assert_eq!(report.skipped, 1);
    fs::set_permissions(&bad, fs::Permissions::from_mode(0o644)).unwrap();
}

#[cfg(unix)]
#[test]
fn file_symlink_escaping_root_is_skipped() {
    use std::os::unix::fs::symlink;
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    fs::create_dir_all(&docs).unwrap();
    let outside = tmp.path().join("secret.md");
    fs::write(&outside, "# Secret").unwrap();
    symlink(&outside, docs.join("link.md")).unwrap();
    write(&docs, "ok.md", "# OK");
    let db = tmp.path().join("i.sqlite");
    let qmd = open_with_docs(&docs, &db);

    let report = update(&qmd);
    assert_eq!(report.indexed, 1);
    assert_eq!(report.skipped, 1);
}

#[cfg(unix)]
#[test]
fn file_symlink_inside_root_is_indexed() {
    use std::os::unix::fs::symlink;
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "real.md", "# Real");
    symlink(docs.join("real.md"), docs.join("alias.md")).unwrap();
    let db = tmp.path().join("i.sqlite");
    let qmd = open_with_docs(&docs, &db);

    let report = update(&qmd);
    // Same content under two paths: both index; the hash is shared.
    assert_eq!(report.indexed, 2);
}

#[cfg(unix)]
#[test]
fn dir_symlink_is_not_traversed() {
    use std::os::unix::fs::symlink;
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "ok.md", "# OK");
    let outside = tmp.path().join("elsewhere");
    write(&outside, "x.md", "# X");
    symlink(&outside, docs.join("else")).unwrap();
    let db = tmp.path().join("i.sqlite");
    let qmd = open_with_docs(&docs, &db);

    let report = update(&qmd);
    assert_eq!(report.indexed, 1);
}

#[test]
fn update_indexes_cjk_file() {
    // Exercises normalize_cjk_for_fts through rebuild_document_fts.
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "cjk.md", "# 你好世界\n正文 テスト 내용");
    let db = tmp.path().join("i.sqlite");
    let qmd = open_with_docs(&docs, &db);

    let report = update(&qmd);
    assert_eq!(report.indexed, 1);
}

// --- db-only reopen still updates -----------------------------------------

#[test]
fn db_only_reopen_can_update() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "a.md", "# A");
    let db = tmp.path().join("i.sqlite");
    let inline = {
        let mut cfg = Config::default();
        cfg.collections.insert(
            "docs".to_owned(),
            qmd::Collection {
                path: docs.to_string_lossy().into_owned(),
                ..qmd::Collection::default()
            },
        );
        cfg
    };
    {
        let qmd = Qmd::builder(&db)
            .environment(Environment::default())
            .config(inline)
            .build()
            .unwrap();
        update(&qmd);
        qmd.close().unwrap();
    }
    // Reopen without config: collections come from store_collections.
    let qmd = open(&db);
    write(&docs, "b.md", "# B");
    let report = update(&qmd);
    assert_eq!((report.indexed, report.unchanged), (1, 1));
}
