//! Filesystem scanning for collections: upstream `splitGlobMask`
//! (store.ts:70-101) plus the `fast-glob` traversal inside
//! `reindexCollection` (store.ts:1635-1649).
//!
//! Deviations: `walkdir` replaces `fast-glob`. Directory symlinks are not
//! followed in either implementation; file symlinks are listed and the
//! out-of-root check in `index` rejects escapes, matching upstream.

use std::path::Path;

use globset::{Glob, GlobSet, GlobSetBuilder};
use walkdir::{DirEntry, WalkDir};

use crate::error::{Error, Result};

/// Directories never descended into (upstream `excludeDirs`,
/// store.ts:1627). A *file* carrying one of these names is still a
/// candidate, same as upstream's `**/{d}/**` ignore patterns.
const EXCLUDE_DIRS: &[&str] = &["node_modules", ".git", ".cache", "vendor", "dist", "build"];

/// Upstream `splitGlobMask` (store.ts:70-101): split on commas, but not on
/// commas inside `{}` alternations or `[]` character classes. An empty or
/// all-separator mask yields the original string.
#[must_use]
pub fn split_glob_mask(mask: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut brace_depth = 0u32;
    let mut bracket_depth = 0u32;

    for ch in mask.chars() {
        match ch {
            '{' => {
                brace_depth += 1;
                current.push(ch);
            }
            '}' if brace_depth > 0 => {
                brace_depth -= 1;
                current.push(ch);
            }
            '[' => {
                bracket_depth += 1;
                current.push(ch);
            }
            ']' if bracket_depth > 0 => {
                bracket_depth -= 1;
                current.push(ch);
            }
            ',' if brace_depth == 0 && bracket_depth == 0 => {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    parts.push(trimmed.to_owned());
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    let trimmed = current.trim();
    if !trimmed.is_empty() {
        parts.push(trimmed.to_owned());
    }
    if parts.is_empty() {
        parts.push(mask.to_owned());
    }
    parts
}

fn build_globset(patterns: &[String]) -> Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        let glob = Glob::new(pattern).map_err(|e| Error::InvalidInput {
            reason: format!("invalid glob pattern '{pattern}': {e}"),
        })?;
        builder.add(glob);
    }
    builder.build().map_err(|e| Error::InvalidInput {
        reason: format!("invalid glob set: {e}"),
    })
}

/// Hidden entries (`.*`) are never indexed upstream (`dot: false` plus
/// the any-segment filter), excluded directory names are pruned rather
/// than matched inside, and directories matching a user `ignore` pattern
/// are pruned too — upstream `fast-glob` evaluates ignores against every
/// entry, so a bare `ignore: ["dirname"]` excludes its whole subtree.
fn keep_entry(entry: &DirEntry, root: &Path, ignore: &GlobSet) -> bool {
    if entry.depth() == 0 {
        return true;
    }
    let name = entry.file_name().to_string_lossy();
    if name.starts_with('.') {
        return false;
    }
    let ft = entry.file_type();
    if ft.is_dir() && EXCLUDE_DIRS.contains(&name.as_ref()) {
        return false;
    }
    if (ft.is_dir() || ft.is_symlink())
        && let Ok(rel) = entry.path().strip_prefix(root)
    {
        return !ignore.is_match(to_slash_path(rel));
    }
    true
}

/// A symlink candidate counts only when its target is an existing file;
/// a symlink to a directory or a dangling one is never a document —
/// upstream `onlyFiles` (`fs.stat`) yields the same behaviour.
fn is_file_candidate(entry: &DirEntry) -> bool {
    let ft = entry.file_type();
    if ft.is_file() {
        return true;
    }
    ft.is_symlink() && std::fs::metadata(entry.path()).is_ok_and(|m| m.is_file())
}

fn to_slash_path(path: &Path) -> String {
    path.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// Scan `root` for files matching `mask` minus `ignore` patterns.
/// Returns relative paths with `/` separators, sorted for determinism.
///
/// # Errors
/// [`Error::InvalidInput`] when a mask or ignore pattern fails to parse,
/// [`Error::Io`] when the traversal itself fails.
pub(crate) fn scan_collection(root: &Path, mask: &str, ignore: &[String]) -> Result<Vec<String>> {
    let masks = split_glob_mask(mask);
    let mask_set = build_globset(&masks)?;
    let ignore_set = build_globset(ignore)?;

    let mut files = Vec::new();
    for entry in WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| keep_entry(e, root, &ignore_set))
    {
        let entry = entry.map_err(|e| {
            let path = e
                .path()
                .map_or_else(|| root.to_path_buf(), Path::to_path_buf);
            let source = e
                .into_io_error()
                .unwrap_or_else(|| std::io::Error::other("walkdir error"));
            Error::Io { path, source }
        })?;
        if !is_file_candidate(&entry) {
            continue;
        }
        let Ok(rel) = entry.path().strip_prefix(root) else {
            continue;
        };
        let rel_slash = to_slash_path(rel);
        // Upstream's post-filter (store.ts:1641-1645): any path segment
        // starting with '.' is dropped. Traversal already prunes these,
        // but a defensive check keeps the guarantee at the boundary.
        if rel
            .components()
            .any(|c| c.as_os_str().to_string_lossy().starts_with('.'))
        {
            continue;
        }
        if ignore_set.is_match(&rel_slash) || !mask_set.is_match(&rel_slash) {
            continue;
        }
        files.push(rel_slash);
    }
    files.sort();
    Ok(files)
}
