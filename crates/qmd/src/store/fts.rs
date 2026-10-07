//! Write-side FTS5 maintenance, ported from upstream `src/store.ts`
//! (`normalizeCjkForFTS` at store.ts:883-885, `rebuildDocumentFTS` at
//! store.ts:2954-2976).
//!
//! Deviation: upstream additionally keeps FTS in sync through triggers
//! installed by `applyFtsSyncTriggers`; this implementation updates the FTS
//! row explicitly inside each document mutation instead (P1 T3).

use std::sync::LazyLock;

use regex::Regex;
use rusqlite::{Connection, OptionalExtension, params};

use crate::error::{DbError, Result};

/// Upstream `CJK_RUN_PATTERN` (store.ts:871): runs of Han, Hiragana,
/// Katakana or Hangul characters.
static CJK_RUN: LazyLock<Regex> = LazyLock::new(|| {
    #[allow(clippy::expect_used, reason = "the pattern is a compile-time constant")]
    Regex::new(r"[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]+")
        .expect("CJK run pattern is valid")
});

/// Upstream `normalizeCjkForFTS` (store.ts:883-885).
///
/// FTS5's unicode61 tokenizer does not segment CJK text into searchable
/// words, so CJK runs are spaced character by character; the matching
/// query-side rewrite lives in P1.3.
pub(crate) fn normalize_cjk_for_fts(text: &str) -> String {
    CJK_RUN
        .replace_all(text, |caps: &regex::Captures<'_>| {
            let spaced = caps[0]
                .chars()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            format!(" {spaced} ")
        })
        .into_owned()
}

/// Upstream `rebuildDocumentFTS` (store.ts:2954-2976): delete the FTS row
/// and reinsert it for an active document; inactive or missing rows leave
/// `documents_fts` empty.
pub(crate) fn rebuild_document_fts(conn: &Connection, document_id: i64) -> Result<()> {
    let row = conn
        .query_row(
            "SELECT d.id, d.collection, d.path, d.title, content.doc \
             FROM documents d \
             JOIN content ON content.hash = d.hash \
             WHERE d.id = ? AND d.active = 1",
            [document_id],
            |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                ))
            },
        )
        .optional()
        .map_err(DbError::from)?;

    conn.execute("DELETE FROM documents_fts WHERE rowid = ?", [document_id])
        .map_err(DbError::from)?;

    if let Some((id, collection, path, title, body)) = row {
        conn.execute(
            "INSERT INTO documents_fts(rowid, filepath, title, body) \
             VALUES (?, ?, ?, ?)",
            params![
                id,
                normalize_cjk_for_fts(&format!("{collection}/{path}")),
                normalize_cjk_for_fts(&title),
                normalize_cjk_for_fts(&body),
            ],
        )
        .map_err(DbError::from)?;
    }
    Ok(())
}
