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
/// flag.
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
    /// `countDocumentsPendingMetadata` (metadata-store.ts:167-179).
    pub(crate) pending_metadata: usize,
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

/// Total `content_vectors` rows — the CLI `status` / `update` epilogue
/// `vector_total` (cli/qmd.ts:1011).
pub(crate) fn count_content_vectors(conn: &Connection) -> Result<usize> {
    let n: i64 = conn
        .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))
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

    Ok(StatusRow {
        total_documents: usize::try_from(total_docs).unwrap_or(0),
        needs_embedding: hashes_needing_embedding(conn, None, model, fingerprint_extra)?,
        has_vector_index: has_vectors_vec(conn)?,
        vector_count: count_content_vectors(conn)?,
        orphaned_vectors: count_orphaned_vectors(conn)?,
        latest_modified: latest_modified(conn)?,
        reindex_required: super::config_get(conn, "reindex_required")?.as_deref() == Some("1"),
        pending_metadata: super::metadata::count_documents_pending_metadata(conn)?,
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

// =============================================================================
// `qmd doctor` probes (P4.3) — the data side of upstream `showDoctor`
// (cli/qmd.ts:4210-4310) and `maybeAdoptLegacyEmbeddingFingerprint`
// (store.ts:2581-2650).
// =============================================================================

/// `SELECT sqlite_version()` — upstream's "SQLite runtime" check
/// (cli/qmd.ts:4219); a failing query surfaces to the caller like
/// upstream's try/catch.
pub(crate) fn sqlite_version(conn: &Connection) -> Result<String> {
    conn.query_row("SELECT sqlite_version()", [], |r| r.get(0))
        .map_err(|e| DbError::from(e).into())
}

/// `SELECT vec_version()` — upstream's "sqlite-vec" check
/// (cli/qmd.ts:4229); errors (extension not loaded) surface to the
/// caller like upstream's try/catch.
pub(crate) fn vec_version(conn: &Connection) -> Result<String> {
    conn.query_row("SELECT vec_version()", [], |r| r.get(0))
        .map_err(|e| DbError::from(e).into())
}

/// `SELECT COUNT(*) FROM documents WHERE active = 1` — upstream's
/// `activeDocs`/`totalDocs` probes (cli/qmd.ts:4003, store.ts:2658).
pub(crate) fn active_doc_count(conn: &Connection) -> Result<usize> {
    let n: i64 = conn
        .query_row("SELECT COUNT(*) FROM documents WHERE active = 1", [], |r| {
            r.get(0)
        })
        .map_err(DbError::from)?;
    Ok(usize::try_from(n).unwrap_or(0))
}

/// One `(model, embed_fingerprint)` group of `content_vectors` —
/// upstream's fingerprint-summary row (cli/qmd.ts:4264-4268).
#[derive(Debug)]
pub(crate) struct FingerprintGroupRow {
    /// `content_vectors.model`.
    pub(crate) model: String,
    /// `content_vectors.embed_fingerprint` (`""` = legacy rows).
    pub(crate) fingerprint: String,
    /// `COUNT(DISTINCT hash)` — distinct documents in the group.
    pub(crate) docs: usize,
    /// `COUNT(*)` — embedded chunks in the group.
    pub(crate) chunks: usize,
}

/// Upstream's fingerprint-groups query (cli/qmd.ts:4263-4269):
/// `GROUP BY model, embed_fingerprint`, most chunks first.
pub(crate) fn fingerprint_groups(conn: &Connection) -> Result<Vec<FingerprintGroupRow>> {
    let rows = conn
        .prepare(
            "SELECT model, embed_fingerprint AS fingerprint, \
             COUNT(DISTINCT hash) AS docs, COUNT(*) AS chunks \
             FROM content_vectors \
             GROUP BY model, embed_fingerprint \
             ORDER BY chunks DESC, model, embed_fingerprint",
        )
        .and_then(|mut stmt| {
            stmt.query_map([], |r| {
                Ok(FingerprintGroupRow {
                    model: r.get(0)?,
                    fingerprint: r.get(1)?,
                    docs: usize::try_from(r.get::<_, i64>(2)?).unwrap_or(0),
                    chunks: usize::try_from(r.get::<_, i64>(3)?).unwrap_or(0),
                })
            })
            .and_then(|m| m.collect::<rusqlite::Result<Vec<_>>>())
        })
        .map_err(DbError::from)?;
    Ok(rows)
}

/// `COUNT(DISTINCT hash)` of legacy empty-fingerprint rows for `model`
/// — upstream's `legacyCount` (store.ts:2586-2588).
pub(crate) fn legacy_fingerprint_doc_count(conn: &Connection, model: &str) -> Result<usize> {
    let n: i64 = conn
        .query_row(
            "SELECT COUNT(DISTINCT hash) AS count FROM content_vectors \
             WHERE model = ? AND embed_fingerprint = ''",
            [model],
            |r| r.get(0),
        )
        .map_err(DbError::from)?;
    Ok(usize::try_from(n).unwrap_or(0))
}

/// A chunk row re-embeddable from `content` — the shared shape of the
/// legacy-adoption sample (store.ts:2591-2602) and the doctor
/// vector-sample probe (cli/qmd.ts:4010-4018).
#[derive(Debug)]
pub(crate) struct SampleRow {
    /// `content_vectors.hash`.
    pub(crate) hash: String,
    /// `content_vectors.seq`.
    pub(crate) seq: usize,
    /// `content.doc` — full document body.
    pub(crate) body: String,
    /// `MIN(documents.path)` over active docs sharing the hash.
    pub(crate) path: String,
}

/// Upstream's deterministic legacy sample (store.ts:2591-2603): one
/// active document's chunk with an empty `embed_fingerprint`, picked by
/// `ORDER BY cv.hash, cv.seq LIMIT 1`. Upstream wraps the query in
/// `withLazyContentVectorMigration`; this schema always has the
/// columns, so the wrapper is a no-op here.
pub(crate) fn legacy_fingerprint_sample(
    conn: &Connection,
    model: &str,
) -> Result<Option<SampleRow>> {
    // `cv.pos`/`cv.total_chunks` stay in GROUP BY (upstream's keys)
    // though only hash/seq/body/path are selected.
    conn.query_row(
        "SELECT cv.hash, cv.seq, c.doc AS body, MIN(d.path) AS path \
         FROM content_vectors cv \
         JOIN documents d ON d.hash = cv.hash AND d.active = 1 \
         JOIN content c ON c.hash = cv.hash \
         WHERE cv.model = ? AND cv.embed_fingerprint = '' \
         GROUP BY cv.hash, cv.seq, cv.pos, cv.total_chunks, c.doc \
         ORDER BY cv.hash, cv.seq \
         LIMIT 1",
        [model],
        |r| {
            Ok(SampleRow {
                hash: r.get(0)?,
                seq: usize::try_from(r.get::<_, i64>(1)?).unwrap_or(0),
                body: r.get(2)?,
                path: r.get(3)?,
            })
        },
    )
    .optional()
    .map_err(|e| DbError::from(e).into())
}

/// Adopt every empty fingerprint for `model` — upstream's `UPDATE
/// content_vectors SET embed_fingerprint = ? …` (store.ts:2645).
/// Returns rows changed.
pub(crate) fn adopt_legacy_fingerprints(
    conn: &Connection,
    model: &str,
    fingerprint: &str,
) -> Result<usize> {
    let changed = conn
        .execute(
            "UPDATE content_vectors SET embed_fingerprint = ? \
             WHERE model = ? AND embed_fingerprint = ''",
            rusqlite::params![fingerprint, model],
        )
        .map_err(DbError::from)?;
    Ok(changed)
}

/// Upstream's doctor vector-sample probe (cli/qmd.ts:4010-4018):
/// `LIMIT` random current-fingerprint chunks joined to active docs and
/// `content`.
pub(crate) fn current_fingerprint_samples(
    conn: &Connection,
    model: &str,
    fingerprint: &str,
    limit: usize,
) -> Result<Vec<SampleRow>> {
    let rows = conn
        .prepare(
            "SELECT cv.hash, cv.seq, c.doc AS body, MIN(d.path) AS path \
             FROM content_vectors cv \
             JOIN documents d ON d.hash = cv.hash AND d.active = 1 \
             JOIN content c ON c.hash = cv.hash \
             WHERE cv.model = ? AND cv.embed_fingerprint = ? \
             GROUP BY cv.hash, cv.seq, c.doc \
             ORDER BY random() \
             LIMIT ?",
        )
        .and_then(|mut stmt| {
            stmt.query_map(
                rusqlite::params![model, fingerprint, i64::try_from(limit).unwrap_or(i64::MAX)],
                |r| {
                    Ok(SampleRow {
                        hash: r.get(0)?,
                        seq: usize::try_from(r.get::<_, i64>(1)?).unwrap_or(0),
                        body: r.get(2)?,
                        path: r.get(3)?,
                    })
                },
            )
            .and_then(|m| m.collect::<rusqlite::Result<Vec<_>>>())
        })
        .map_err(DbError::from)?;
    Ok(rows)
}

/// Nearest `vectors_vec` row to `embedding` — upstream's
/// `embedding MATCH ? AND k = 1` probe (store.ts:2629-2633). Returns
/// `(hash_seq, cosine distance)`.
///
/// # Errors
/// [`DbError`] — including when `vectors_vec` is missing; the caller
/// checks [`has_vectors_vec`] first, like upstream.
pub(crate) fn vec_nearest(conn: &Connection, embedding: &[f32]) -> Result<Option<(String, f64)>> {
    conn.query_row(
        "SELECT hash_seq, distance FROM vectors_vec WHERE embedding MATCH ? AND k = 1",
        [rusqlite::types::Value::Blob(super::vec::vec_blob(
            embedding,
        ))],
        |r| Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?)),
    )
    .optional()
    .map_err(|e| DbError::from(e).into())
}

/// Stored `vectors_vec.embedding` blob for `hash_seq` — upstream's
/// per-sample fetch (cli/qmd.ts:4046).
pub(crate) fn stored_embedding(conn: &Connection, hash_seq: &str) -> Result<Option<Vec<u8>>> {
    conn.query_row(
        "SELECT embedding FROM vectors_vec WHERE hash_seq = ?",
        [hash_seq],
        |r| r.get(0),
    )
    .optional()
    .map_err(|e| DbError::from(e).into())
}
