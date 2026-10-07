//! Per-collection reindex pipeline, ported from upstream
//! `reindexCollection` (store.ts:1617-1747).
//!
//! Deviations:
//! - The whole pass runs inside one transaction; upstream's statements
//!   autocommit. A mid-pass failure here rolls back instead of leaving a
//!   half-indexed collection.
//! - `syncDocumentMetadata` / `metadataErrors` are deferred to P3.
//! - `findOrMigrateLegacyDocument` is not ported (no handelized legacy
//!   paths exist in this store).
//! - Error codes map [`std::io::ErrorKind`] to the closest Node errno
//!   names; upstream passes through `err.code` verbatim.

use std::collections::HashSet;
use std::io::ErrorKind;
use std::path::Path;

use rusqlite::Connection;

use super::scan::scan_collection;
use crate::error::{DbError, Result};
use crate::paths::is_path_inside_dir;
use crate::store::documents;

/// Progress callback payload for a single file.
pub(crate) struct FileProgress<'a> {
    /// Relative file path, `/`-separated.
    pub(crate) file: &'a str,
    /// Files processed so far (including this one).
    pub(crate) current: usize,
    /// Total files in this scan.
    pub(crate) total: usize,
}

/// A file that could not be indexed.
#[derive(Debug)]
pub(crate) struct SkippedFile {
    /// Relative file path, `/`-separated.
    #[allow(dead_code, reason = "kept for per-file diagnostics in P4 CLI")]
    pub(crate) file: String,
    /// Skip reason: `OUTSIDE_COLLECTION` or an errno-style code.
    #[allow(dead_code, reason = "kept for per-file diagnostics in P4 CLI")]
    pub(crate) code: String,
}

/// Per-collection result, mirroring upstream `ReindexResult` minus
/// `metadataErrors` (deferred to P3).
#[derive(Debug, Default)]
pub(crate) struct ReindexReport {
    /// Newly indexed files.
    pub(crate) indexed: usize,
    /// Re-indexed files whose hash or title changed.
    pub(crate) updated: usize,
    /// Files already up to date.
    pub(crate) unchanged: usize,
    /// Documents deactivated because the file vanished.
    pub(crate) removed: usize,
    /// Orphan `content` rows removed.
    pub(crate) orphaned_cleaned: usize,
    /// Files skipped (see [`SkippedFile::code`]).
    pub(crate) skipped_files: Vec<SkippedFile>,
}

/// Upstream `fsErrorCode` (store.ts:1582-1589): Node errno names for the
/// common kinds, `ERROR` otherwise.
fn fs_error_code(err: &std::io::Error) -> String {
    match err.kind() {
        ErrorKind::PermissionDenied => "EACCES",
        ErrorKind::NotFound => "ENOENT",
        ErrorKind::TimedOut => "ETIMEDOUT",
        ErrorKind::WouldBlock => "EAGAIN",
        ErrorKind::IsADirectory => "EISDIR",
        _ => "ERROR",
    }
    .to_owned()
}

/// `fs.statSync` timestamps; upstream uses `birthtime` for `created_at`
/// and `mtime` for `modified_at`. `created()` is not supported on every
/// filesystem, so it falls back to `modified()` then `now`.
fn file_timestamps(path: &Path, now: &str) -> (String, String) {
    let Ok(meta) = std::fs::metadata(path) else {
        return (now.to_owned(), now.to_owned());
    };
    let modified = meta
        .modified()
        .map_or_else(|_| now.to_owned(), documents::iso_timestamp);
    let created = meta
        .created()
        .map_or_else(|_| modified.clone(), documents::iso_timestamp);
    (created, modified)
}

/// Upstream `reindexCollection` (store.ts:1617-1747): scan, diff by
/// content hash, upsert into `documents`/`content`, maintain FTS, then
/// deactivate unseen rows and collect orphans.
///
/// # Errors
/// [`Error::Db`] on SQLite failures, [`Error::InvalidInput`] on a bad
/// glob, [`Error::Io`] on traversal failure.
pub(crate) fn reindex_collection(
    conn: &mut Connection,
    collection_path: &Path,
    glob_pattern: &str,
    collection_name: &str,
    ignore: &[String],
    on_progress: &mut dyn FnMut(&FileProgress<'_>),
) -> Result<ReindexReport> {
    let files = scan_collection(collection_path, glob_pattern, ignore)?;
    let total = files.len();
    let tx = conn.transaction().map_err(DbError::from)?;
    let now = documents::now_iso();

    let mut report = ReindexReport::default();
    let mut seen_paths: HashSet<String> = HashSet::with_capacity(total);
    let mut processed = 0usize;

    for relative_file in &files {
        let filepath = collection_path.join(relative_file);
        // Glob `../` segments and file symlinks can resolve outside the
        // collection root. Do not ingest them, and do not mark them seen
        // so a previous escaped row is deactivated on this pass
        // (store.ts:1660-1669).
        if !is_path_inside_dir(collection_path, &filepath) {
            processed += 1;
            report.skipped_files.push(SkippedFile {
                file: relative_file.clone(),
                code: "OUTSIDE_COLLECTION".to_owned(),
            });
            on_progress(&FileProgress {
                file: relative_file,
                current: processed,
                total,
            });
            continue;
        }
        seen_paths.insert(relative_file.clone());

        // Upstream reads UTF-8; invalid byte sequences are replaced.
        let content = match std::fs::read(&filepath) {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Err(err) => {
                processed += 1;
                report.skipped_files.push(SkippedFile {
                    file: relative_file.clone(),
                    code: fs_error_code(&err),
                });
                on_progress(&FileProgress {
                    file: relative_file,
                    current: processed,
                    total,
                });
                continue;
            }
        };

        // Whitespace-only files are skipped *after* being marked seen:
        // upstream keeps the previous document active (store.ts:1685-1687).
        if content.trim().is_empty() {
            processed += 1;
            on_progress(&FileProgress {
                file: relative_file,
                current: processed,
                total,
            });
            continue;
        }

        let hash = documents::hash_content(&content);
        let title = documents::extract_title(&content, relative_file);
        let existing = documents::find_active_document(&tx, collection_name, relative_file)?;

        match existing {
            Some(doc) if doc.hash == hash => {
                if doc.title == title {
                    report.unchanged += 1;
                } else {
                    documents::update_document_title(&tx, doc.id, &title, &now)?;
                    report.updated += 1;
                }
            }
            Some(doc) => {
                documents::insert_content(&tx, &hash, &content, &now)?;
                let (_, modified) = file_timestamps(&filepath, &now);
                documents::update_document(&tx, doc.id, &title, &hash, &modified)?;
                report.updated += 1;
            }
            None => {
                documents::insert_content(&tx, &hash, &content, &now)?;
                let (created, modified) = file_timestamps(&filepath, &now);
                documents::insert_document(
                    &tx,
                    collection_name,
                    relative_file,
                    &title,
                    &hash,
                    &created,
                    &modified,
                )?;
                report.indexed += 1;
            }
        }

        processed += 1;
        on_progress(&FileProgress {
            file: relative_file,
            current: processed,
            total,
        });
    }

    // Deactivate documents that no longer exist (store.ts:1732-1740).
    for path in documents::active_document_paths(&tx, collection_name)? {
        if !seen_paths.contains(&path) {
            documents::deactivate_document(&tx, collection_name, &path)?;
            report.removed += 1;
        }
    }

    report.orphaned_cleaned = documents::cleanup_orphaned_content(&tx)?;
    tx.commit().map_err(DbError::from)?;
    Ok(report)
}
