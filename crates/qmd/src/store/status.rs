//! Index status and health queries, ported from upstream
//! `src/store.ts`: `getStatus` (5239-5284), `getIndexHealth`
//! (2656-2668), `getHashesNeedingEmbedding` (2546-2566) and the CLI
//! `status` aggregates (cli/qmd.ts:540-583).

use rusqlite::{Connection, OptionalExtension};
use time::OffsetDateTime;

use super::collections::get_store_collections;
use crate::error::{DbError, Result};
use crate::llm::embedding_fingerprint;

/// One `status` collection row — upstream `CollectionInfo`
/// (store.ts:2525-2531). `path`/`pattern` are `None` when the collection
/// exists only as document rows (no `store_collections` entry).
pub(crate) struct StatusCollection {
    /// Collection name.
    pub(crate) name: String,
    /// Filesystem path, if known.
    pub(crate) path: Option<String>,
    /// Glob pattern, if known.
    pub(crate) pattern: Option<String>,
    /// Active document count.
    pub(crate) documents: usize,
    /// Latest `modified_at`, or the current time when empty.
    pub(crate) last_updated: String,
}

/// Upstream `IndexStatus` (store.ts:2533-2540) plus the extra aggregates
/// the CLI `status` command queries directly (vector counts, orphans,
/// `latest_modified`) and this implementation's `reindex_required`
/// flag. `pendingMetadata` arrives with P3.
pub(crate) struct StatusRow {
    /// Active documents.
    pub(crate) total_documents: usize,
    /// Distinct active content hashes lacking current vectors.
    pub(crate) needs_embedding: usize,
    /// Whether `vectors_vec` exists.
    pub(crate) has_vector_index: bool,
    /// Total `content_vectors` rows (cli/qmd.ts:541).
    pub(crate) vector_count: usize,
    /// `content_vectors` rows with no active document (cli/qmd.ts:565).
    pub(crate) orphaned_vectors: usize,
    /// `MAX(modified_at)` of active documents (cli/qmd.ts:546).
    pub(crate) latest_modified: Option<String>,
    /// Schema rebuild flagged a required re-index (`store_config`).
    pub(crate) reindex_required: bool,
    /// Per-collection rows, most recently updated first.
    pub(crate) collections: Vec<StatusCollection>,
}

/// Upstream `IndexHealthInfo` (store.ts:2568-2572).
pub(crate) struct IndexHealthRow {
    /// Distinct active content hashes lacking current vectors.
    pub(crate) needs_embedding: usize,
    /// Active document count.
    pub(crate) total_docs: usize,
    /// Whole days since the newest `modified_at`; `None` when empty.
    pub(crate) days_stale: Option<i64>,
}

/// Upstream `getHashesNeedingEmbedding` (store.ts:2546-2566): distinct
/// active content hashes whose `content_vectors` rows for `model` are
/// missing, under a different `embed_fingerprint`, or incomplete
/// (`chunk_count < expected_chunks`). The lazy column-migration wrapper
/// is unnecessary — this implementation's schema always has the columns.
pub(crate) fn hashes_needing_embedding(
    conn: &Connection,
    collection: Option<&str>,
    model: &str,
    fingerprint_extra: Option<&str>,
) -> Result<usize> {
    let fingerprint = embedding_fingerprint(model, fingerprint_extra);
    let sql = format!(
        "SELECT COUNT(DISTINCT d.hash) FROM documents d \
         LEFT JOIN ( \
           SELECT hash, model, COUNT(*) AS chunk_count, MAX(total_chunks) AS expected_chunks \
           FROM content_vectors \
           WHERE model = ? AND embed_fingerprint = ? \
           GROUP BY hash, model, embed_fingerprint \
         ) v ON d.hash = v.hash \
         WHERE d.active = 1 AND (v.hash IS NULL OR v.chunk_count < v.expected_chunks) {}",
        if collection.is_some() {
            "AND d.collection = ?"
        } else {
            ""
        }
    );
    let mut params: Vec<rusqlite::types::Value> = vec![model.to_owned().into(), fingerprint.into()];
    if let Some(c) = collection {
        params.push(c.to_owned().into());
    }
    let count = conn
        .query_row(&sql, rusqlite::params_from_iter(params), |r| {
            r.get::<_, i64>(0)
        })
        .map_err(DbError::from)?;
    Ok(usize::try_from(count).unwrap_or(0))
}

/// `MAX(modified_at)` over active documents.
fn latest_modified(conn: &Connection) -> Result<Option<String>> {
    conn.query_row(
        "SELECT MAX(modified_at) FROM documents WHERE active = 1",
        [],
        |r| r.get(0),
    )
    .map_err(|e| DbError::from(e).into())
}

/// Upstream `countOrphanedVectors` (store.ts:2753-2775): `content_vectors`
/// rows whose hash no active document references.
pub(crate) fn count_orphaned_vectors(conn: &Connection) -> Result<usize> {
    let n: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM content_vectors cv WHERE NOT EXISTS \
             (SELECT 1 FROM documents d WHERE d.hash = cv.hash AND d.active = 1)",
            [],
            |r| r.get(0),
        )
        .map_err(DbError::from)?;
    Ok(usize::try_from(n).unwrap_or(0))
}

/// Whether the `vectors_vec` shadow table exists (upstream checks
/// `sqlite_master` for `hasVectorIndex`).
pub(crate) fn has_vectors_vec(conn: &Connection) -> Result<bool> {
    conn.query_row(
        "SELECT COUNT(*) > 0 FROM sqlite_master \
         WHERE type = 'table' AND name = 'vectors_vec'",
        [],
        |r| r.get(0),
    )
    .map_err(|e| DbError::from(e).into())
}

/// Upstream `getStatus` (store.ts:5239-5284) plus the CLI `status`
/// aggregates (cli/qmd.ts:540-583). Collections are sorted by
/// `last_updated` descending.
pub(crate) fn get_status(
    conn: &Connection,
    model: &str,
    fingerprint_extra: Option<&str>,
) -> Result<StatusRow> {
    let mut stmt = conn
        .prepare(
            "SELECT collection AS name, COUNT(*) AS active_count, \
             MAX(modified_at) AS last_doc_update \
             FROM documents WHERE active = 1 GROUP BY collection",
        )
        .map_err(DbError::from)?;
    let doc_rows = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, Option<String>>(2)?,
            ))
        })
        .map_err(DbError::from)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)?;

    let store_collections = get_store_collections(conn)?;
    let mut collections: Vec<StatusCollection> = doc_rows
        .into_iter()
        .map(|(name, count, last)| {
            let cfg = store_collections.iter().find(|c| c.name == name);
            StatusCollection {
                name,
                path: cfg.map(|c| c.path.clone()),
                pattern: cfg.map(|c| c.pattern.clone()),
                documents: usize::try_from(count).unwrap_or(0),
                last_updated: last.unwrap_or_else(super::documents::now_iso),
            }
        })
        .collect();
    // ISO-8601 `Z`-suffixed timestamps sort correctly as strings.
    collections.sort_by(|a, b| b.last_updated.cmp(&a.last_updated));

    let total_docs: i64 = conn
        .query_row("SELECT COUNT(*) FROM documents WHERE active = 1", [], |r| {
            r.get(0)
        })
        .map_err(DbError::from)?;
    let vector_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))
        .map_err(DbError::from)?;

    Ok(StatusRow {
        total_documents: usize::try_from(total_docs).unwrap_or(0),
        needs_embedding: hashes_needing_embedding(conn, None, model, fingerprint_extra)?,
        has_vector_index: has_vectors_vec(conn)?,
        vector_count: usize::try_from(vector_count).unwrap_or(0),
        orphaned_vectors: count_orphaned_vectors(conn)?,
        latest_modified: latest_modified(conn)?,
        reindex_required: super::config_get(conn, "reindex_required")?.as_deref() == Some("1"),
        collections,
    })
}

/// Upstream `getIndexHealth` (store.ts:2656-2668): pending embeddings,
/// active document count, and whole days since the newest document.
pub(crate) fn get_index_health(
    conn: &Connection,
    model: &str,
    fingerprint_extra: Option<&str>,
) -> Result<IndexHealthRow> {
    let latest = latest_modified(conn)?;
    // Upstream `Math.floor((now - last) / 86400000)`; `whole_days`
    // truncates toward zero, which only differs for a future-dated
    // `modified_at` (upstream would floor to -1, we report 0).
    // Rfc3339 (not a custom description with a literal `Z`) so the parsed
    // value actually carries its UTC offset.
    let days_stale = latest.as_deref().and_then(|s| {
        OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339)
            .ok()
            // Upstream `Math.floor((now - last) / 86400000)`: `div_euclid`
            // keeps floor semantics for future-dated timestamps (-1)
            // where `whole_days()` would truncate toward zero.
            .map(|t| {
                (OffsetDateTime::now_utc() - t)
                    .whole_seconds()
                    .div_euclid(86_400)
            })
    });
    let total_docs: i64 = conn
        .query_row("SELECT COUNT(*) FROM documents WHERE active = 1", [], |r| {
            r.get(0)
        })
        .map_err(DbError::from)?;
    Ok(IndexHealthRow {
        needs_embedding: hashes_needing_embedding(conn, None, model, fingerprint_extra)?,
        total_docs: usize::try_from(total_docs).unwrap_or(0),
        days_stale,
    })
}

/// `content.doc` for a hash — helper used by vector-cleanup tests and
/// the P2 embedding pipeline.
#[allow(dead_code, reason = "used by tests and the P2 embedding pipeline")]
pub(crate) fn content_for_hash(conn: &Connection, hash: &str) -> Result<Option<String>> {
    conn.query_row("SELECT doc FROM content WHERE hash = ?", [hash], |r| {
        r.get(0)
    })
    .optional()
    .map_err(|e| DbError::from(e).into())
}
