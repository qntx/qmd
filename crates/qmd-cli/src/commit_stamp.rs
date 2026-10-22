//! Pure decision for the `--version` commit stamp (P4-D6).
//!
//! Mirrors upstream `resolveCommit` (version.ts:56-83, tobi/qmd#787):
//! report a commit only when the git toplevel *is* this workspace — a
//! binary built inside an unrelated enclosing repository must not claim
//! that repo's HEAD. `-dirty` marks a dirty worktree.

use std::path::Path;

/// `Some(commit)` when `git rev-parse --show-toplevel` (already
/// canonicalized inside) equals the workspace root; `None` for foreign
/// checkouts or non-repo builds (crates.io tarballs have no `.git`).
pub(crate) fn resolve_commit(
    git_toplevel: Option<&str>,
    workspace_root: &Path,
    head_short: Option<&str>,
    dirty: bool,
) -> Option<String> {
    let toplevel = git_toplevel?;
    let canonical = std::fs::canonicalize(toplevel).ok()?;
    if canonical != workspace_root {
        return None;
    }
    let head = head_short?.trim().to_owned();
    if head.is_empty() {
        return None;
    }
    Some(if dirty { format!("{head}-dirty") } else { head })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamps_commit_when_toplevel_is_workspace() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let root = std::fs::canonicalize(root.parent().unwrap().parent().unwrap()).unwrap();
        let toplevel = root.to_string_lossy().into_owned();
        assert_eq!(
            resolve_commit(Some(&toplevel), &root, Some("abc1234"), false).as_deref(),
            Some("abc1234")
        );
    }

    #[test]
    fn dirty_worktree_gets_suffix() {
        let dir = tempfile::tempdir().unwrap();
        // macOS /var is a symlink to /private/var — canonicalize like
        // `git rev-parse --show-toplevel` output.
        let root = dir.path().canonicalize().unwrap();
        let top = root.to_string_lossy().into_owned();
        assert_eq!(
            resolve_commit(Some(&top), &root, Some("def5678"), true).as_deref(),
            Some("def5678-dirty")
        );
    }

    #[test]
    fn foreign_enclosing_repo_stamps_nothing() {
        // tobi/qmd#787: a package installed inside an unrelated repo
        // must not report that repo's HEAD.
        let outer = tempfile::tempdir().unwrap();
        let inner = outer.path().join("vendor/qmd");
        std::fs::create_dir_all(&inner).unwrap();
        let outer_top = outer.path().to_string_lossy().into_owned();
        assert_eq!(
            resolve_commit(Some(&outer_top), &inner, Some("abc"), false),
            None
        );
    }

    #[test]
    fn outside_any_repo_stamps_nothing() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(resolve_commit(None, dir.path(), Some("abc"), false), None);
        assert_eq!(
            resolve_commit(Some("/"), dir.path(), Some("abc"), false),
            None
        );
    }

    #[test]
    fn path_with_space_resolves() {
        let dir = tempfile::tempdir().unwrap();
        let spaced = dir.path().join("dir with space");
        std::fs::create_dir(&spaced).unwrap();
        let spaced = spaced.canonicalize().unwrap();
        let top = spaced.to_string_lossy().into_owned();
        assert_eq!(
            resolve_commit(Some(&top), &spaced, Some("abc"), false).as_deref(),
            Some("abc")
        );
    }
}
