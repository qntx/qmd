//! `store_collections` / `store_config` persistence and the
//! config-to-database sync, ported from upstream `src/collections.ts`.
//!
//! Deviation from upstream: when a collection disappears from the config,
//! its documents are deactivated (`active = 0`, FTS rows removed) rather
//! than left orphaned — upstream keeps them searchable, which P1 treats as
//! a bug. Tombstones keep `content` rows reachable and are reclaimed by
//! `cleanup` (P1.4), so removal is recoverable until then.

use indexmap::IndexMap;
use rusqlite::{Connection, TransactionBehavior, params};

use crate::config::{Collection, Config};
use crate::error::{DbError, Result};

/// A `store_collections` row.
pub(crate) struct DbCollection {
    /// Collection name (primary key).
    pub(crate) name: String,
    /// Filesystem path, verbatim.
    pub(crate) path: String,
    /// Glob pattern (never NULL in the DB).
    pub(crate) pattern: String,
    /// Decoded `ignore_patterns` JSON array.
    pub(crate) ignore: Option<Vec<String>>,
    /// `include_by_default` flag.
    pub(crate) include_by_default: bool,
    /// Pre-update hook command.
    pub(crate) update: Option<String>,
    /// Decoded `context` JSON object.
    pub(crate) context: Option<IndexMap<String, String>>,
}

fn json_err(e: serde_json::Error) -> crate::error::Error {
    // serde_json errors here indicate corrupt DB content written by us;
    // surface them as database errors rather than a new variant.
    DbError::from(rusqlite::Error::ToSqlConversionFailure(Box::new(e))).into()
}

fn de_json<T: serde::de::DeserializeOwned>(
    json: Option<String>,
    column: usize,
    _name: &'static str,
) -> rusqlite::Result<Option<T>> {
    json.map(|s| {
        serde_json::from_str(&s).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(
                column,
                rusqlite::types::Type::Text,
                Box::new(e),
            )
        })
    })
    .transpose()
}

fn row_to_collection(row: &rusqlite::Row<'_>) -> rusqlite::Result<DbCollection> {
    let ignore_json: Option<String> = row.get(3)?;
    let context_json: Option<String> = row.get(6)?;
    Ok(DbCollection {
        name: row.get(0)?,
        path: row.get(1)?,
        pattern: row.get(2)?,
        ignore: de_json(ignore_json, 3, "ignore_patterns")?,
        include_by_default: row.get::<_, i64>(4)? != 0,
        update: row.get(5)?,
        context: de_json(context_json, 6, "context")?,
    })
}

const SELECT_COLS: &str = "name, path, pattern, ignore_patterns, include_by_default, \
     update_command, context";

/// All `store_collections` rows in insertion order (`rowid`).
pub(crate) fn get_store_collections(conn: &Connection) -> Result<Vec<DbCollection>> {
    let mut stmt = conn
        .prepare(&format!(
            "SELECT {SELECT_COLS} FROM store_collections ORDER BY rowid"
        ))
        .map_err(DbError::from)?;
    let rows = stmt
        .query_map([], row_to_collection)
        .map_err(DbError::from)?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(DbError::from)?;
    Ok(rows)
}

/// Insert or update one `store_collections` row from config data
/// (`upsertStoreCollection` in upstream). Accepts `&Collection` plus name.
pub(crate) fn upsert_store_collection(
    conn: &Connection,
    name: &str,
    coll: &Collection,
) -> Result<()> {
    let ignore_json = coll
        .ignore
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(json_err)?;
    let context_json = coll
        .context
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(json_err)?;
    conn.execute(
        "INSERT INTO store_collections
           (name, path, pattern, ignore_patterns, include_by_default,
            update_command, context)
         VALUES (?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(name) DO UPDATE SET
           path = excluded.path,
           pattern = excluded.pattern,
           ignore_patterns = excluded.ignore_patterns,
           include_by_default = excluded.include_by_default,
           update_command = excluded.update_command,
           context = excluded.context,
           updated_at = CURRENT_TIMESTAMP",
        params![
            name,
            coll.path,
            coll.pattern_or_default(),
            ignore_json,
            i64::from(coll.include_by_default()),
            coll.update,
            context_json,
        ],
    )
    .map_err(DbError::from)?;
    Ok(())
}

/// Delete a `store_collections` row.
pub(crate) fn delete_store_collection(conn: &Connection, name: &str) -> Result<()> {
    conn.execute("DELETE FROM store_collections WHERE name = ?", [name])
        .map_err(DbError::from)?;
    Ok(())
}

/// Global context text (`getStoreGlobalContext`).
pub(crate) fn get_store_global_context(conn: &Connection) -> Result<Option<String>> {
    super::config_get(conn, "global_context")
}

/// Store the global context (`setStoreGlobalContext`); `None` removes it.
pub(crate) fn set_store_global_context(conn: &Connection, text: Option<&str>) -> Result<()> {
    if let Some(text) = text {
        conn.execute(
            "INSERT INTO store_config (key, value) VALUES ('global_context', ?) \
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [text],
        )
        .map_err(DbError::from)?;
    } else {
        conn.execute("DELETE FROM store_config WHERE key = 'global_context'", [])
            .map_err(DbError::from)?;
    }
    Ok(())
}

/// One entry of [`get_store_contexts`]: global context first (path `"/"`),
/// then each collection context in `store_collections` order.
pub(crate) struct ContextRow {
    /// Collection name or `"*"`.
    pub(crate) collection: String,
    /// Context path prefix (`"/"` for the global context).
    pub(crate) path: String,
    /// Context text.
    pub(crate) context: String,
}

/// All contexts (`getStoreContexts`), global first.
pub(crate) fn get_store_contexts(conn: &Connection) -> Result<Vec<ContextRow>> {
    let mut out = Vec::new();
    if let Some(global) = get_store_global_context(conn)? {
        out.push(ContextRow {
            collection: "*".to_owned(),
            path: "/".to_owned(),
            context: global,
        });
    }
    for coll in get_store_collections(conn)? {
        if let Some(contexts) = coll.context {
            out.extend(contexts.into_iter().map(|(path, context)| ContextRow {
                collection: coll.name.clone(),
                path,
                context,
            }));
        }
    }
    Ok(out)
}

/// Remove FTS rows for documents of a collection (deactivation deletes
/// them upstream via triggers; we maintain FTS explicitly).
fn delete_fts_for_collection(conn: &Connection, name: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM documents_fts WHERE rowid IN \
         (SELECT id FROM documents WHERE collection = ?)",
        [name],
    )
    .map_err(DbError::from)?;
    Ok(())
}

/// Deactivate all documents of a collection and drop their FTS rows.
/// Returns the number of documents deactivated by this call (rows that
/// were already inactive are not counted). The FTS delete covers every
/// row of the collection regardless, in case stale rows linger.
pub(crate) fn deactivate_documents(conn: &Connection, collection: &str) -> Result<usize> {
    delete_fts_for_collection(conn, collection)?;
    let n = conn
        .execute(
            "UPDATE documents SET active = 0 \
             WHERE collection = ? AND active = 1",
            [collection],
        )
        .map_err(DbError::from)?;
    Ok(n)
}

/// Rename all documents of a collection (P1-D1 CLI semantics; the
/// `documents_fts.filepath` column intentionally keeps the old name, as
/// upstream's `renameCollection` does).
pub(crate) fn rename_documents(conn: &Connection, old: &str, new: &str) -> Result<()> {
    conn.execute(
        "UPDATE documents SET collection = ? WHERE collection = ?",
        params![new, old],
    )
    .map_err(DbError::from)?;
    Ok(())
}

/// Outcome of [`sync_config_to_db`].
#[derive(Debug, Default)]
pub(crate) struct SyncOutcome {
    /// Documents deactivated because their collection left the config.
    pub(crate) documents_deactivated: usize,
}

/// Mirror config into `store_collections`/`store_config`
/// (`syncConfigToDb`). Skipped when the serialized config hash is
/// unchanged, unless `force` is set.
///
/// Collections present in the DB but absent from the config lose their
/// `store_collections` row and have their documents deactivated.
pub(crate) fn sync_config_to_db(
    conn: &mut Connection,
    config: &Config,
    force: bool,
) -> Result<SyncOutcome> {
    use sha2::Digest as _;
    let json = serde_json::to_string(config).map_err(json_err)?;
    let digest = sha2::Sha256::digest(json.as_bytes());
    let hash: String = digest
        .iter()
        .flat_map(|b| {
            [
                char::from_digit(u32::from(b >> 4), 16).unwrap_or_default(),
                char::from_digit(u32::from(b & 0x0f), 16).unwrap_or_default(),
            ]
        })
        .collect();
    if !force && super::config_get(conn, "config_hash")?.as_deref() == Some(hash.as_str()) {
        return Ok(SyncOutcome::default());
    }

    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(DbError::from)?;
    let mut outcome = SyncOutcome::default();

    let existing: Vec<String> = get_store_collections(&tx)?
        .into_iter()
        .map(|c| c.name)
        .collect();
    for (name, coll) in &config.collections {
        upsert_store_collection(&tx, name, coll)?;
    }
    for name in existing {
        if !config.collections.contains_key(&name) {
            outcome.documents_deactivated += deactivate_documents(&tx, &name)?;
            delete_store_collection(&tx, &name)?;
        }
    }
    set_store_global_context(&tx, config.global_context.as_deref())?;
    super::config_set(&tx, "config_hash", &hash)?;

    tx.commit().map_err(DbError::from)?;
    Ok(outcome)
}
