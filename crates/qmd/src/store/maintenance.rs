//! Cleanup and maintenance operations, ported from upstream
//! `src/store.ts`: `deleteLLMCache` (2713), `deleteInactiveDocuments`
//! (2722), `cleanupOrphanedContent`/`countOrphanedContent` (2733-2751),
//! `cleanupOrphanedVectors`/`countOrphanedVectors` (2753-2844),
//! `vacuumDatabase` (2850), `optimizeDocumentsFts` (2858-2866),
//! `clearAllEmbeddings` (4437-4441, whole-index variant), and
//! `previewCleanup`/`runCleanup` (2868-2896).

use rusqlite::{Connection, TransactionBehavior};

use super::status::count_orphaned_vectors;
use crate::error::{DbError, Result};

/// Counts mirroring upstream `CleanupStats` (store.ts:2868-2873) —
/// what `run` removes.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CleanupCounts {
    /// `llm_cache` rows deleted.
    pub cache_count: usize,
    /// Orphaned `content_vectors` rows deleted.
    pub orphaned_vectors: usize,
    /// Inactive `documents` rows deleted.
    pub inactive_docs: usize,
    /// Orphaned `content` rows deleted.
    pub orphaned_content: usize,
}

/// Upstream `deleteLLMCache` (store.ts:2713-2716): rows removed.
pub(crate) fn delete_llm_cache(conn: &Connection) -> Result<usize> {
    conn.execute("DELETE FROM llm_cache", [])
        .map_err(|e| DbError::from(e).into())
}

/// Upstream `deleteInactiveDocuments` (store.ts:2722-2725): tombstone
/// rows (`active = 0`) hard-deleted. Their `documents_fts` rows are
/// already gone (deactivation removes them explicitly).
pub(crate) fn delete_inactive_documents(conn: &Connection) -> Result<usize> {
    conn.execute("DELETE FROM documents WHERE active = 0", [])
        .map_err(|e| DbError::from(e).into())
}

/// Upstream `countOrphanedContent` (store.ts:2745-2751): `content`
/// hashes only referenced by *inactive* documents — what
/// [`cleanup_orphaned_content`] frees once tombstones are deleted.
pub(crate) fn count_orphaned_content(conn: &Connection) -> Result<usize> {
    let n: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM content \
             WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
            [],
            |r| r.get(0),
        )
        .map_err(DbError::from)?;
    Ok(usize::try_from(n).unwrap_or(0))
}

/// Upstream `cleanupOrphanedVectors` (store.ts:2781-2844): one
/// `BEGIN IMMEDIATE` transaction covering the orphan count and both
/// DELETEs so the two tables cannot desync, and so the count returned
/// matches the rows removed under concurrent writers.
///
/// When `vectors_vec` is absent (embeddings never ran), upstream's
/// `SELECT 1 FROM vectors_vec LIMIT 0` probe throws and the whole call
/// reports `0` — mirrored by the existence check.
pub(crate) fn cleanup_orphaned_vectors(conn: &mut Connection) -> Result<usize> {
    if !super::status::has_vectors_vec(conn)? {
        return Ok(0);
    }
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(DbError::from)?;
    let orphaned = count_orphaned_vectors(&tx)?;
    if orphaned == 0 {
        return Ok(0);
    }
    tx.execute_batch(
        "DELETE FROM vectors_vec WHERE hash_seq IN ( \
           SELECT cv.hash || '_' || cv.seq FROM content_vectors cv \
           WHERE NOT EXISTS ( \
             SELECT 1 FROM documents d WHERE d.hash = cv.hash AND d.active = 1 \
           ) \
         ); \
         DELETE FROM content_vectors WHERE hash NOT IN ( \
           SELECT hash FROM documents WHERE active = 1 \
         );",
    )
    .map_err(DbError::from)?;
    tx.commit().map_err(DbError::from)?;
    Ok(orphaned)
}

/// Upstream `vacuumDatabase` (store.ts:2850-2852).
pub(crate) fn vacuum_database(conn: &Connection) -> Result<()> {
    conn.execute_batch("VACUUM")
        .map_err(|e| DbError::from(e).into())
}

/// Upstream `optimizeDocumentsFts` (store.ts:2858-2866): merge FTS5
/// b-trees so hard-deleted rows leave `documents_fts_data` (#550).
/// A missing `documents_fts` table degrades to a no-op, as upstream.
pub(crate) fn optimize_documents_fts(conn: &Connection) -> Result<()> {
    match conn.execute_batch("INSERT INTO documents_fts(documents_fts) VALUES('optimize')") {
        Ok(()) => Ok(()),
        Err(e) => {
            let msg = e.to_string();
            if msg.to_lowercase().contains("no such table") {
                Ok(())
            } else {
                Err(DbError::from(e).into())
            }
        }
    }
}

/// Upstream `clearAllEmbeddings` whole-index variant
/// (store.ts:4437-4441): empty `content_vectors` and drop
/// `vectors_vec` (recreated with the right dimensions on next embed).
pub(crate) fn clear_all_embeddings(conn: &Connection) -> Result<()> {
    conn.execute_batch("DELETE FROM content_vectors; DROP TABLE IF EXISTS vectors_vec")
        .map_err(|e| DbError::from(e).into())
}

/// Upstream `previewCleanup` (store.ts:2876-2882): what `run` would
/// remove, without writing.
pub(crate) fn preview_cleanup(conn: &Connection) -> Result<CleanupCounts> {
    let cache_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM llm_cache", [], |r| r.get(0))
        .map_err(DbError::from)?;
    let inactive_docs: i64 = conn
        .query_row("SELECT COUNT(*) FROM documents WHERE active = 0", [], |r| {
            r.get(0)
        })
        .map_err(DbError::from)?;
    Ok(CleanupCounts {
        cache_count: usize::try_from(cache_count).unwrap_or(0),
        orphaned_vectors: count_orphaned_vectors(conn)?,
        inactive_docs: usize::try_from(inactive_docs).unwrap_or(0),
        orphaned_content: count_orphaned_content(conn)?,
    })
}

/// Upstream `runCleanup` (store.ts:2888-2896): cache → orphaned vectors →
/// inactive documents → orphaned content → FTS optimize → VACUUM.
pub(crate) fn run_cleanup(conn: &mut Connection) -> Result<CleanupCounts> {
    let cache_count = delete_llm_cache(conn)?;
    let orphaned_vectors = cleanup_orphaned_vectors(conn)?;
    let inactive_docs = delete_inactive_documents(conn)?;
    let orphaned_content = super::documents::cleanup_orphaned_content(conn)?;
    optimize_documents_fts(conn)?;
    vacuum_database(conn)?;
    Ok(CleanupCounts {
        cache_count,
        orphaned_vectors,
        inactive_docs,
        orphaned_content,
    })
}
