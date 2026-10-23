//! Build script: stamp the commit for `--version` (P4-D6).
//!
//! Only when `git rev-parse --show-toplevel` realpaths to the workspace
//! root does `git rev-parse --short HEAD` become the stamp — a checkout
//! nested inside an unrelated repository reports nothing (upstream
//! tobi/qmd#787), and crates.io builds (no `.git`) get no commit at
//! all. A dirty worktree appends `-dirty`. `git` is invoked with an
//! argument list, never a shell, so paths containing spaces work.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "src/commit_stamp.rs"]
mod commit_stamp;

#[allow(
    clippy::expect_used,
    reason = "CARGO_MANIFEST_DIR is always set by cargo for build scripts"
)]
fn main() {
    let manifest = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by cargo"),
    );
    let workspace_root = manifest
        .parent()
        .and_then(Path::parent)
        .map_or_else(|| manifest.clone(), Path::to_path_buf);
    let workspace_root = std::fs::canonicalize(&workspace_root).unwrap_or(workspace_root);

    let toplevel = git(&workspace_root, &["rev-parse", "--show-toplevel"]);
    let head = git(&workspace_root, &["rev-parse", "--short", "HEAD"]);
    let dirty =
        git(&workspace_root, &["status", "--porcelain"]).is_some_and(|s| !s.trim().is_empty());
    let commit =
        commit_stamp::resolve_commit(toplevel.as_deref(), &workspace_root, head.as_deref(), dirty)
            .unwrap_or_default();
    println!("cargo:rustc-env=QMD_CLI_COMMIT={commit}");

    // Rebuild when the checked-out commit, its ref, or the index
    // (the `-dirty` signal) changes — best effort.
    let head_file = workspace_root.join(".git/HEAD");
    println!("cargo:rerun-if-changed={}", head_file.display());
    println!(
        "cargo:rerun-if-changed={}",
        workspace_root.join(".git/index").display()
    );
    if let Ok(content) = std::fs::read_to_string(&head_file)
        && let Some(reff) = content.trim().strip_prefix("ref: ")
    {
        println!(
            "cargo:rerun-if-changed={}",
            workspace_root.join(".git").join(reff).display()
        );
    }
}

/// `git <args>` in `dir` → trimmed stdout on success.
fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}
