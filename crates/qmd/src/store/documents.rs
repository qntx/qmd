//! Document and content indexing operations, ported from upstream
//! `src/store.ts`: `insertContent` (2948), `insertDocument` (2979),
//! `updateDocumentTitle` (3101), `updateDocument` (3116),
//! `deactivateDocument` (3131), `getActiveDocumentPaths` (3139),
//! `findActiveDocument` (3005), `cleanupOrphanedContent` (2733),
//! `extractTitle` (2930), `hashContent` (2902) and `clearCache` (2700).
//!
//! Deviations:
//! - `deactivate_document` also removes the FTS row; upstream relies on a
//!   delete trigger this implementation intentionally does not create.
//! - `findOrMigrateLegacyDocument` is not ported: this store never wrote
//!   handelized paths, so there is no legacy row to adopt.
//! - `syncDocumentMetadata` / `metadataErrors` are deferred to P3 (document
//!   metadata extraction).

use std::sync::LazyLock;
use std::time::SystemTime;

use regex::Regex;
use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};
use time::OffsetDateTime;
use time::macros::format_description;

use super::fts::rebuild_document_fts;
use crate::error::{DbError, Result};

/// Upstream `hashContent` (store.ts:2902-2906): lowercase hex SHA-256.
pub(crate) fn hash_content(content: &str) -> String {
    // Manually nibble-split to keep indexing-slicing happy; `b >> 4` and
    // `b & 0xf` are always < 16 but the lint cannot see it.
    let mut s = String::with_capacity(64);
    for b in Sha256::digest(content.as_bytes()) {
        s.push(char::from_digit(u32::from(b >> 4), 16).unwrap_or('0'));
        s.push(char::from_digit(u32::from(b & 0x0f), 16).unwrap_or('0'));
    }
    s
}

/// ISO-8601 millisecond timestamp (`YYYY-MM-DDTHH:MM:SS.sssZ`), matching
/// upstream `new Date().toISOString()`.
pub(crate) fn iso_timestamp(t: SystemTime) -> String {
    const FMT: &[time::format_description::BorrowedFormatItem<'_>] =
        format_description!("[year]-[month]-[day]T[hour]:[minute]:[second].[subsecond digits:3]Z");
    OffsetDateTime::from(t)
        .format(FMT)
        .unwrap_or_else(|_| "1970-01-01T00:00:00.000Z".to_owned())
}

/// `new Date().toISOString()` for the current time.
pub(crate) fn now_iso() -> String {
    iso_timestamp(SystemTime::now())
}

macro_rules! static_regex {
    ($name:ident, $pat:literal) => {
        static $name: LazyLock<Regex> = LazyLock::new(|| {
            #[allow(clippy::expect_used, reason = "the pattern is a compile-time constant")]
            Regex::new($pat).expect("static regex pattern is valid")
        });
    };
}

static_regex!(MD_HEADING, r"(?m)^##?\s+(.+)$");
static_regex!(MD_H2_HEADING, r"(?m)^##\s+(.+)$");
static_regex!(ORG_TITLE, r"(?im)^#\+TITLE:\s*(.+)$");
static_regex!(ORG_HEADING, r"(?m)^\*+\s+(.+)$");
static_regex!(LAST_EXT, r"\.[^.]+$");

/// Upstream `.md` extractor (store.ts:2909-2922): first `#`/`##` heading;
/// when it is the boilerplate "📝 Notes"/"Notes" the next `##` wins.
fn md_title(content: &str) -> Option<String> {
    let cap = MD_HEADING.captures(content)?;
    let title = cap[1].trim();
    if (title == "📝 Notes" || title == "Notes")
        && let Some(next) = MD_H2_HEADING.captures(content)
    {
        return Some(next[1].trim().to_owned());
    }
    Some(title.to_owned())
}

/// Upstream `.org` extractor (store.ts:2923-2928): `#+TITLE:` property,
/// else the first `*`-heading.
fn org_title(content: &str) -> Option<String> {
    ORG_TITLE
        .captures(content)
        .or_else(|| ORG_HEADING.captures(content))
        .map(|c| c[1].trim().to_owned())
}

/// Upstream `extractTitle` (store.ts:2930-2938) and its `titleExtractors`
/// (store.ts:2908-2928).
///
/// Markdown `#`/`##` heading (with the upstream "📝 Notes"/"Notes"
/// level-2 fallback), org `#+TITLE:`/heading, then the
/// extension-stripped basename.
#[must_use]
pub fn extract_title(content: &str, filename: &str) -> String {
    let ext = filename.rfind('.').map(|i| filename[i..].to_lowercase());
    let extracted = match ext.as_deref() {
        Some(".md") => md_title(content),
        Some(".org") => org_title(content),
        _ => None,
    };
    if let Some(title) = extracted {
        return title;
    }
    let stem = LAST_EXT.replace(filename, "");
    stem.rsplit('/')
        .next()
        .filter(|s| !s.is_empty())
        .map_or_else(|| filename.to_owned(), ToOwned::to_owned)
}

/// A live `documents` row as used by the reindex loop.
pub(crate) struct ActiveDocument {
    /// Row id.
    pub(crate) id: i64,
    /// Current content hash.
    pub(crate) hash: String,
    /// Current title.
    pub(crate) title: String,
}

/// Upstream `findActiveDocument` (store.ts:3005-3019).
pub(crate) fn find_active_document(
    conn: &Connection,
    collection: &str,
    path: &str,
) -> Result<Option<ActiveDocument>> {
    conn.query_row(
        "SELECT id, hash, title FROM documents \
         WHERE collection = ? AND path = ? AND active = 1",
        params![collection, path],
        |r| {
            Ok(ActiveDocument {
                id: r.get(0)?,
                hash: r.get(1)?,
                title: r.get(2)?,
            })
        },
    )
    .optional()
    .map_err(|e| DbError::from(e).into())
}

/// Upstream `insertContent` (store.ts:2948-2951): content-addressable
/// storage, `INSERT OR IGNORE` on the hash primary key.
pub(crate) fn insert_content(
    conn: &Connection,
    hash: &str,
    doc: &str,
    created_at: &str,
) -> Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?, ?, ?)",
        params![hash, doc, created_at],
    )
    .map_err(DbError::from)?;
    Ok(())
}

/// Upstream `insertDocument` (store.ts:2979-3000): upsert keyed on
/// `(collection, path)`, reactivating the row, then rebuild FTS.
/// Returns the document id.
pub(crate) fn insert_document(
    conn: &Connection,
    collection: &str,
    path: &str,
    title: &str,
    hash: &str,
    created_at: &str,
    modified_at: &str,
) -> Result<i64> {
    conn.execute(
        "INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active) \
         VALUES (?, ?, ?, ?, ?, ?, 1) \
         ON CONFLICT(collection, path) DO UPDATE SET \
           title = excluded.title, \
           hash = excluded.hash, \
           modified_at = excluded.modified_at, \
           active = 1",
        params![collection, path, title, hash, created_at, modified_at],
    )
    .map_err(DbError::from)?;

    let id = conn
        .query_row(
            "SELECT id FROM documents WHERE collection = ? AND path = ?",
            params![collection, path],
            |r| r.get(0),
        )
        .map_err(DbError::from)?;
    rebuild_document_fts(conn, id)?;
    Ok(id)
}

/// Upstream `updateDocumentTitle` (store.ts:3101-3109).
pub(crate) fn update_document_title(
    conn: &Connection,
    document_id: i64,
    title: &str,
    modified_at: &str,
) -> Result<()> {
    conn.execute(
        "UPDATE documents SET title = ?, modified_at = ? WHERE id = ?",
        params![title, modified_at, document_id],
    )
    .map_err(DbError::from)?;
    rebuild_document_fts(conn, document_id)
}

/// Upstream `updateDocument` (store.ts:3116-3126): content changed under
/// the same path.
pub(crate) fn update_document(
    conn: &Connection,
    document_id: i64,
    title: &str,
    hash: &str,
    modified_at: &str,
) -> Result<()> {
    conn.execute(
        "UPDATE documents SET title = ?, hash = ?, modified_at = ? WHERE id = ?",
        params![title, hash, modified_at, document_id],
    )
    .map_err(DbError::from)?;
    rebuild_document_fts(conn, document_id)
}

/// Upstream `deactivateDocument` (store.ts:3131-3135) plus the FTS-row
/// removal upstream performs via trigger.
pub(crate) fn deactivate_document(conn: &Connection, collection: &str, path: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM documents_fts WHERE rowid IN \
         (SELECT id FROM documents WHERE collection = ? AND path = ? AND active = 1)",
        params![collection, path],
    )
    .map_err(DbError::from)?;
    conn.execute(
        "UPDATE documents SET active = 0 \
         WHERE collection = ? AND path = ? AND active = 1",
        params![collection, path],
    )
    .map_err(DbError::from)?;
    Ok(())
}

/// Upstream `getActiveDocumentPaths` (store.ts:3139-3145).
pub(crate) fn active_document_paths(conn: &Connection, collection: &str) -> Result<Vec<String>> {
    let mut stmt = conn
        .prepare("SELECT path FROM documents WHERE collection = ? AND active = 1")
        .map_err(DbError::from)?;
    let rows = stmt
        .query_map([collection], |r| r.get(0))
        .map_err(DbError::from)?
        .collect::<rusqlite::Result<Vec<String>>>()
        .map_err(DbError::from)?;
    Ok(rows)
}

/// Upstream `cleanupOrphanedContent` (store.ts:2733-2738): drop `content`
/// rows no longer referenced by any document. Returns rows removed.
pub(crate) fn cleanup_orphaned_content(conn: &Connection) -> Result<usize> {
    conn.execute(
        "DELETE FROM content WHERE hash NOT IN (SELECT DISTINCT hash FROM documents)",
        [],
    )
    .map_err(|e| DbError::from(e).into())
}

/// Upstream `clearCache` (store.ts:2700-2702): `update` wipes the LLM
/// cache before re-indexing.
pub(crate) fn clear_llm_cache(conn: &Connection) -> Result<()> {
    conn.execute_batch("DELETE FROM llm_cache")
        .map_err(|e| DbError::from(e).into())
}

/// Count of distinct active content hashes with no vector rows.
///
/// Simplified version of upstream `getHashesNeedingEmbedding`
/// (store.ts:2800-2820): model and embedding-fingerprint filtering arrive
/// with the embedding pipeline in P2, so for now every active hash counts
/// as needing embedding.
pub(crate) fn hashes_needing_embedding(conn: &Connection) -> Result<usize> {
    conn.query_row(
        "SELECT COUNT(DISTINCT d.hash) FROM documents d \
         WHERE d.active = 1 AND NOT EXISTS \
           (SELECT 1 FROM content_vectors cv WHERE cv.hash = d.hash)",
        [],
        |r| r.get::<_, i64>(0),
    )
    .map(|n| usize::try_from(n).unwrap_or(0))
    .map_err(|e| DbError::from(e).into())
}
