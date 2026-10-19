//! `document_metadata` / `document_metadata_values` persistence — a
//! port of upstream `src/metadata-store.ts`.
//!
//! Metadata attaches to document identity (`documents.id`), not content
//! identity: two paths can share one content hash while carrying
//! different metadata. `document_metadata` records extraction state per
//! document (including successful-but-empty extraction) so filtered
//! search can distinguish "extracted with no metadata" from "not yet
//! extracted" and "extraction failed"; `document_metadata_values` holds
//! one indexed row per scalar value for filtering.

use rusqlite::{Connection, OptionalExtension, params};

use crate::error::{DbError, Result};
use crate::metadata::{
    DocumentMetadata, METADATA_EXTRACTION_VERSION, MetadataExtraction, MetadataValue,
    extract_document_metadata,
};
use crate::store::documents::now_iso;

/// Upstream `initializeMetadataSchema` DDL (metadata-store.ts:29-78):
/// both tables, the `ON DELETE CASCADE` foreign keys, the per-type
/// CHECK constraint, and the three partial indexes.
pub(crate) const METADATA_DDL: &[&str] = &[
    r"
        CREATE TABLE IF NOT EXISTS document_metadata (
            document_id INTEGER PRIMARY KEY,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            extraction_version INTEGER NOT NULL,
            extraction_error TEXT,
            extracted_at TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        )
    ",
    r"
        CREATE TABLE IF NOT EXISTS document_metadata_values (
            document_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            value_type TEXT NOT NULL,
            text_value TEXT,
            number_value REAL,
            boolean_value INTEGER,
            PRIMARY KEY (document_id, key, ordinal),
            FOREIGN KEY (document_id)
                REFERENCES document_metadata(document_id)
                ON DELETE CASCADE,
            CHECK (value_type IN ('string', 'number', 'boolean')),
            CHECK (
                (value_type = 'string' AND text_value IS NOT NULL AND number_value IS NULL AND boolean_value IS NULL)
                OR (value_type = 'number' AND number_value IS NOT NULL AND text_value IS NULL AND boolean_value IS NULL)
                OR (value_type = 'boolean' AND boolean_value IN (0, 1) AND text_value IS NULL AND number_value IS NULL)
            )
        )
    ",
    r"
        CREATE INDEX IF NOT EXISTS idx_metadata_text_lookup
        ON document_metadata_values(key, text_value, document_id)
        WHERE value_type = 'string'
    ",
    r"
        CREATE INDEX IF NOT EXISTS idx_metadata_number_lookup
        ON document_metadata_values(key, number_value, document_id)
        WHERE value_type = 'number'
    ",
    r"
        CREATE INDEX IF NOT EXISTS idx_metadata_boolean_lookup
        ON document_metadata_values(key, boolean_value, document_id)
        WHERE value_type = 'boolean'
    ",
];

/// Apply [`METADATA_DDL`] — the tables belong to the structural layer,
/// so `store::initialize` calls this right after the structural DDL in
/// both the rebuild and fresh-database branches.
///
/// # Errors
/// Database failures.
pub(crate) fn create_tables(conn: &Connection) -> Result<()> {
    for statement in METADATA_DDL {
        conn.execute_batch(statement).map_err(DbError::from)?;
    }
    Ok(())
}

/// Upstream `syncDocumentMetadata` (metadata-store.ts:89-102): extract
/// and persist unless `only_if_stale` and the stored row is current —
/// the row exists and its `extraction_version` matches
/// [`METADATA_EXTRACTION_VERSION`].
///
/// Returns `None` when the document was already current, otherwise the
/// extraction that was persisted.
///
/// # Errors
/// Database failures.
pub(crate) fn sync_document_metadata(
    conn: &Connection,
    document_id: i64,
    content: &str,
    path: &str,
    only_if_stale: bool,
) -> Result<Option<MetadataExtraction>> {
    if only_if_stale && is_document_metadata_current(conn, document_id)? {
        return Ok(None);
    }
    let extraction = extract_document_metadata(content, path);
    replace_document_metadata(conn, document_id, &extraction)?;
    Ok(Some(extraction))
}

/// Upstream `isDocumentMetadataCurrent` (metadata-store.ts:140-144) —
/// compares the extraction version only; a current-version error row
/// still counts as current.
fn is_document_metadata_current(conn: &Connection, document_id: i64) -> Result<bool> {
    Ok(conn
        .query_row(
            "SELECT extraction_version FROM document_metadata WHERE document_id = ?",
            params![document_id],
            |r| r.get::<_, i64>(0),
        )
        .optional()
        .map_err(DbError::from)?
        == Some(METADATA_EXTRACTION_VERSION))
}

/// Upstream `replaceDocumentMetadata` (metadata-store.ts:104-138):
/// upsert the JSON row and rewrite the scalar value rows — atomic via
/// the caller's update transaction (upstream `db.transaction`). A failed
/// extraction persists empty metadata plus the error, so stale metadata
/// never survives a bad edit.
///
/// # Errors
/// Database failures.
fn replace_document_metadata(
    conn: &Connection,
    document_id: i64,
    extraction: &MetadataExtraction,
) -> Result<()> {
    conn.execute(
        "INSERT INTO document_metadata
            (document_id, metadata_json, extraction_version, extraction_error, extracted_at)
         VALUES (?, ?, ?, ?, ?)
         ON CONFLICT(document_id) DO UPDATE SET
            metadata_json = excluded.metadata_json,
            extraction_version = excluded.extraction_version,
            extraction_error = excluded.extraction_error,
            extracted_at = excluded.extracted_at",
        params![
            document_id,
            extraction.metadata.to_json(),
            extraction.extraction_version,
            extraction.error.as_deref(),
            now_iso(),
        ],
    )
    .map_err(DbError::from)?;
    conn.execute(
        "DELETE FROM document_metadata_values WHERE document_id = ?",
        params![document_id],
    )
    .map_err(DbError::from)?;
    write_metadata_values(conn, document_id, &extraction.metadata)?;
    Ok(())
}

/// Expand each metadata entry into `document_metadata_values`
/// (metadata-store.ts:131-134): `const scalars = Array.isArray(value)
/// ? value : [value]` — array elements each get a row with their
/// ordinal; scalars use ordinal 0.
fn write_metadata_values(
    conn: &Connection,
    document_id: i64,
    metadata: &DocumentMetadata,
) -> Result<()> {
    for (key, value) in metadata.iter() {
        let scalars: Vec<Scalar<'_>> = match value {
            MetadataValue::String(s) => vec![Scalar::Text(s)],
            MetadataValue::Number(n) => vec![Scalar::Number(*n)],
            MetadataValue::Bool(b) => vec![Scalar::Bool(*b)],
            MetadataValue::Strings(items) => items.iter().map(|s| Scalar::Text(s)).collect(),
            MetadataValue::Numbers(items) => items.iter().map(|n| Scalar::Number(*n)).collect(),
            MetadataValue::Bools(items) => items.iter().map(|b| Scalar::Bool(*b)).collect(),
        };
        for (ordinal, scalar) in scalars.iter().enumerate() {
            let ordinal = i64::try_from(ordinal).unwrap_or(i64::MAX);
            match scalar {
                Scalar::Text(v) => conn
                    .execute(
                        "INSERT INTO document_metadata_values
                            (document_id, key, ordinal, value_type, text_value)
                         VALUES (?, ?, ?, 'string', ?)",
                        params![document_id, key, ordinal, v],
                    )
                    .map_err(DbError::from)?,
                Scalar::Number(v) => conn
                    .execute(
                        "INSERT INTO document_metadata_values
                            (document_id, key, ordinal, value_type, number_value)
                         VALUES (?, ?, ?, 'number', ?)",
                        params![document_id, key, ordinal, v],
                    )
                    .map_err(DbError::from)?,
                Scalar::Bool(v) => conn
                    .execute(
                        "INSERT INTO document_metadata_values
                            (document_id, key, ordinal, value_type, boolean_value)
                         VALUES (?, ?, ?, 'boolean', ?)",
                        params![document_id, key, ordinal, i64::from(*v)],
                    )
                    .map_err(DbError::from)?,
            };
        }
    }
    Ok(())
}

/// One scalar slot of a metadata value.
enum Scalar<'a> {
    /// `string`
    Text(&'a str),
    /// `number`
    Number(f64),
    /// `boolean`
    Bool(bool),
}

/// Upstream `countDocumentsPendingMetadata` (metadata-store.ts:167-179):
/// active documents lacking a current-version, error-free extraction.
///
/// # Errors
/// Database failures.
pub(crate) fn count_documents_pending_metadata(conn: &Connection) -> Result<usize> {
    Ok(conn
        .query_row(
            "SELECT COUNT(*) FROM documents d
             WHERE d.active = 1
               AND NOT EXISTS (
                 SELECT 1 FROM document_metadata dm
                 WHERE dm.document_id = d.id
                   AND dm.extraction_version = ?
                   AND dm.extraction_error IS NULL
               )",
            params![METADATA_EXTRACTION_VERSION],
            |row| row.get::<_, i64>(0),
        )
        .map(|n| usize::try_from(n).unwrap_or(0))
        .map_err(DbError::from)?)
}
