//! Port of upstream `test/full-path-fallback.test.ts` — `--full-path`
//! falls back to the canonical `qmd://` URI plus docid when the indexed
//! file moved, and warns on stderr; stdout stays machine-clean.
//!
//! Upstream builds the index with `collection add` then `rename`s
//! `beta.md` away; here the index is populated through the library and
//! the file is renamed afterwards, leaving a stale row.

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

use common::{CliEnv, collection_yaml, strip_ansi};

/// `stale` collection over alpha.md + beta.md, with beta.md moved out
/// after indexing.
fn env_with_stale() -> CliEnv {
    let env = CliEnv::new();
    env.write_file("alpha.md", "# Alpha\n\nsearchterm-stale alpha\n");
    let beta = env.write_file("beta.md", "# Beta\n\nsearchterm-stale beta\n");
    env.write_config("index", &collection_yaml("stale", &env.fixtures, "**/*.md"));
    env.update();
    std::fs::rename(&beta, env.root.join("beta-moved.md")).unwrap();
    env
}

fn has_full_path_warning(stderr: &str) -> bool {
    stderr.contains("--full-path could not resolve")
}

#[test]
fn search_json_keeps_the_docid_on_the_row_it_could_not_resolve() {
    let env = env_with_stale();
    let out = env.run(&["search", "searchterm-stale", "--full-path", "--json"]);
    assert_eq!(out.exit_code, 0);

    let results: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let arr = results.as_array().unwrap();
    assert_eq!(arr.len(), 2);

    let resolved = arr
        .iter()
        .find(|r| !r["file"].as_str().unwrap().starts_with("qmd://"))
        .expect("alpha.md should resolve on disk");
    let unresolved = arr
        .iter()
        .find(|r| r["file"].as_str().unwrap().starts_with("qmd://"))
        .expect("beta.md should fall back to its qmd:// URI");

    // Resolved row: the path is the identifier, so no docid.
    assert!(resolved["file"].as_str().unwrap().contains("alpha.md"));
    assert!(resolved.get("docid").is_none());
    // Unresolved row: the docid is all that is left to address it by.
    let docid = unresolved["docid"].as_str().unwrap();
    assert!(docid.starts_with('#') && docid.len() == 7);

    assert!(has_full_path_warning(&out.stderr));
    assert!(out.stderr.contains("qmd update"));
}

#[test]
fn search_format_csv_always_emits_the_docid_column() {
    let env = env_with_stale();
    let out = env.run(&[
        "search",
        "searchterm-stale",
        "--full-path",
        "--format",
        "csv",
    ]);
    assert_eq!(out.exit_code, 0);

    let lines: Vec<&str> = out.stdout.trim().lines().collect();
    assert_eq!(lines[0], "docid,score,file,title,context,line,snippet");

    let resolved_row = lines.iter().find(|l| l.contains("alpha.md")).unwrap();
    let unresolved_row = lines
        .iter()
        .find(|l| l.contains("qmd://stale/beta.md"))
        .unwrap();
    // Resolved rows leave the column empty; the unresolved row fills it.
    assert!(resolved_row.starts_with(','));
    assert!(unresolved_row.starts_with('#'));
}

#[test]
fn search_default_cli_format_keeps_the_docid_next_to_the_fallback_uri() {
    let env = env_with_stale();
    let out = env.run(&["search", "searchterm-stale", "--full-path"]);
    assert_eq!(out.exit_code, 0);
    let plain = strip_ansi(&out.stdout);

    let beta_line = plain
        .lines()
        .find(|l| l.contains("qmd://stale/beta.md"))
        .expect("beta should fall back to its qmd:// URI");
    let tail = beta_line.trim_end();
    assert!(
        tail.rsplit_once('#')
            .is_some_and(|(_, s)| s.len() == 6 && s.chars().all(|c| c.is_ascii_hexdigit())),
        "beta line should end with #docid: {tail:?}"
    );

    let alpha_line = plain
        .lines()
        .find(|l| l.contains("alpha.md") && !l.starts_with("Title"))
        .unwrap();
    assert!(
        !alpha_line.contains('#')
            || !alpha_line
                .split('#')
                .nth(1)
                .is_some_and(|s| { s.chars().take(6).all(|c| c.is_ascii_hexdigit()) })
    );

    assert!(has_full_path_warning(&out.stderr));
}

#[test]
fn get_warns_and_falls_back_to_qmd_uri_plus_docid() {
    let env = env_with_stale();
    let out = env.run(&["get", "beta.md", "--full-path"]);
    assert_eq!(out.exit_code, 0);
    let first = out.stdout.lines().next().unwrap();
    assert!(first.starts_with("qmd://stale/beta.md"));
    assert!(
        first
            .rsplit_once('#')
            .is_some_and(|(_, s)| s.len() == 6 && s.chars().all(|c| c.is_ascii_hexdigit()))
    );
    assert!(has_full_path_warning(&out.stderr));
}

#[test]
fn multi_get_warns_when_a_requested_file_is_gone_from_disk() {
    let env = env_with_stale();
    let out = env.run(&[
        "multi-get",
        "alpha.md,beta.md",
        "--full-path",
        "--format",
        "files",
    ]);
    assert_eq!(out.exit_code, 0);
    assert!(
        out.stdout
            .lines()
            .any(|l| l.starts_with('#') && l.contains(",qmd://stale/beta.md"))
    );
    assert!(has_full_path_warning(&out.stderr));
}

#[test]
fn no_warning_when_every_result_resolves() {
    let env = env_with_stale();
    let out = env.run(&["search", "alpha", "--full-path", "--json"]);
    assert_eq!(out.exit_code, 0);
    let results: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let arr = results.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert!(!arr[0]["file"].as_str().unwrap().starts_with("qmd://"));
    assert!(arr[0].get("docid").is_none());
    assert!(!has_full_path_warning(&out.stderr));
}

#[test]
fn without_full_path_nothing_changes_and_nothing_is_warned() {
    let env = env_with_stale();
    let out = env.run(&["search", "searchterm-stale", "--json"]);
    assert_eq!(out.exit_code, 0);
    let results: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    for r in results.as_array().unwrap() {
        assert!(r["file"].as_str().unwrap().starts_with("qmd://stale/"));
        let docid = r["docid"].as_str().unwrap();
        assert!(docid.starts_with('#') && docid.len() == 7);
    }
    assert!(!has_full_path_warning(&out.stderr));
}
