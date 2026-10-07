//! SQLite storage layer.
//!
//! Schema versioning uses three layers, per the P1 design:
//!
//! - **config layer** (`store_collections`, `store_config`) is never
//!   dropped: it is the only copy of the collection definitions when a
//!   database is reopened without its config file.
//! - **structural layer** (`content`, `documents`, `documents_fts`,
//!   `llm_cache`) is dropped and recreated when `schema_version` changes,
//!   and `reindex_required` is flagged so callers re-index.
//! - **vector layer** (`content_vectors`, `vectors_vec`) is recreated when
//!   `vector_schema_version` changes; content is preserved so vectors can
//!   be regenerated without re-scanning files.
//!
//! The `application_id` PRAGMA carries [`APPLICATION_ID`] to reject foreign
//! databases (including upstream `qmd` indexes) before any write.

pub(crate) mod collections;
pub(crate) mod documents;
pub(crate) mod fts;
pub(crate) mod vec;

use std::path::Path;
use std::time::{Duration, Instant};

use rusqlite::{Connection, OptionalExtension, TransactionBehavior};

use crate::env::Environment;
use crate::error::{DbError, Error, Result};
use crate::paths::busy_timeout_ms;

/// `application_id` marking databases created by this implementation
/// (`"QmdR"` in ASCII).
const APPLICATION_ID: i64 = 0x516D_6452;

/// Structural schema version for this implementation (independent from
/// upstream's 15).
const SCHEMA_VERSION: &str = "1";
/// Vector schema version for this implementation.
const VECTOR_SCHEMA_VERSION: &str = "1";

/// Full structural DDL, applied to fresh databases and on
/// `schema_version` mismatch rebuilds.
const STRUCTURAL_DDL: &str = "
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY,
  collection TEXT NOT NULL,
  path TEXT NOT NULL,
  title TEXT NOT NULL,
  hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  modified_at TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,
  UNIQUE(collection, path)
);
CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection, active);
CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path, active);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);

CREATE TABLE IF NOT EXISTS content (
  hash TEXT PRIMARY KEY,
  doc TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
  filepath,
  title,
  body,
  tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS llm_cache (
  hash TEXT PRIMARY KEY,
  result TEXT NOT NULL,
  created_at TEXT NOT NULL
);
";

/// Vector-layer DDL.
const VECTOR_DDL: &str = "
CREATE TABLE IF NOT EXISTS content_vectors (
  hash TEXT NOT NULL,
  seq INTEGER NOT NULL DEFAULT 0,
  pos INTEGER NOT NULL DEFAULT 0,
  model TEXT NOT NULL,
  embedded_at TEXT NOT NULL,
  PRIMARY KEY (hash, seq)
);
CREATE INDEX IF NOT EXISTS idx_vectors_model ON content_vectors(model);
";

/// Structural tables dropped on a `schema_version` mismatch.
const STRUCTURAL_TABLES: &[&str] = &["documents_fts", "documents", "content", "llm_cache"];

/// Vector tables dropped on a `vector_schema_version` mismatch.
/// `vectors_vec` may be a virtual shadow table owned by `vec0`; it is
/// dropped first and recreated by the DDL below.
const VECTOR_TABLES: &[&str] = &["vectors_vec", "content_vectors"];

/// Config-layer DDL.
const CONFIG_DDL: &str = "
CREATE TABLE IF NOT EXISTS store_collections (
  name TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  pattern TEXT NOT NULL DEFAULT '**/*.md',
  ignore_patterns TEXT,
  include_by_default INTEGER NOT NULL DEFAULT 1,
  update_command TEXT,
  context TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS store_config (
  key TEXT PRIMARY KEY,
  value TEXT
);
";

/// Open a database at `path` with upstream-equivalent pragmas
/// (`busy_timeout`, WAL with busy-retry, `foreign_keys`).
///
/// sqlite-vec is registered once per process before the first connection.
///
/// # Errors
/// [`Error::Db`] on any SQLite failure, [`Error::ForeignIndex`] if the
/// file is a non-empty database not created by this application.
pub(crate) fn open(env: &Environment, path: &Path) -> Result<Connection> {
    vec::register().map_err(DbError::from)?;
    let mut conn = Connection::open(path).map_err(DbError::from)?;
    let timeout = busy_timeout_ms(env);
    conn.execute_batch(&format!("PRAGMA busy_timeout = {timeout}"))
        .map_err(DbError::from)?;
    enable_wal(&conn, timeout)?;
    conn.execute_batch("PRAGMA foreign_keys = ON")
        .map_err(DbError::from)?;
    initialize(&mut conn, path)?;
    Ok(conn)
}

fn is_busy(err: &rusqlite::Error) -> bool {
    // `extended_code & 0xFF` strips extended bits so SQLITE_BUSY_SNAPSHOT
    // and friends also match the primary code.
    matches!(
        err,
        rusqlite::Error::SqliteFailure(e, _)
            if e.code == rusqlite::ffi::ErrorCode::DatabaseBusy
                || e.extended_code & 0xFF == rusqlite::ffi::SQLITE_BUSY
    )
}

/// `PRAGMA journal_mode = WAL` with the upstream retry loop
/// (`db.ts:84-95`): concurrent cold opens can hit `SQLITE_BUSY`; retry
/// inside the busy-timeout budget.
fn enable_wal(conn: &Connection, budget_ms: u64) -> Result<()> {
    let deadline = Instant::now() + Duration::from_millis(budget_ms);
    for attempt in 0u64.. {
        match conn.query_row("PRAGMA journal_mode = WAL", [], |r| r.get::<_, String>(0)) {
            Ok(_) => return Ok(()),
            Err(e) if is_busy(&e) && Instant::now() < deadline => {
                #[allow(
                    clippy::disallowed_methods,
                    reason = "bounded busy-retry mirrors upstream WAL cold-open loop"
                )]
                std::thread::sleep(Duration::from_millis((5 + attempt).min(25)));
            }
            Err(e) => return Err(DbError::from(e).into()),
        }
    }
    Ok(())
}

/// Read a `store_config` value. Returns `None` when the table or key does
/// not exist.
fn config_get(conn: &Connection, key: &str) -> Result<Option<String>> {
    let has_table: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master \
             WHERE type='table' AND name='store_config'",
            [],
            |r| r.get(0),
        )
        .map_err(DbError::from)?;
    if !has_table {
        return Ok(None);
    }
    conn.query_row("SELECT value FROM store_config WHERE key = ?", [key], |r| {
        r.get(0)
    })
    .optional()
    .map_err(|e| DbError::from(e).into())
}

fn config_set(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO store_config (key, value) VALUES (?, ?) \
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        rusqlite::params![key, value],
    )
    .map_err(DbError::from)?;
    Ok(())
}

fn application_id(conn: &Connection) -> Result<i64> {
    conn.query_row("PRAGMA application_id", [], |r| r.get(0))
        .map_err(|e| DbError::from(e).into())
}

fn has_tables(conn: &Connection) -> Result<bool> {
    conn.query_row(
        "SELECT COUNT(*) > 0 FROM sqlite_master \
         WHERE type IN ('table','view')",
        [],
        |r| r.get(0),
    )
    .map_err(|e| DbError::from(e).into())
}

/// Initialize schema, applying the three-layer versioning rules.
fn initialize(conn: &mut Connection, path: &Path) -> Result<()> {
    // Fast path: our database with current versions needs no transaction.
    if application_id(conn)? == APPLICATION_ID {
        let structural = config_get(conn, "schema_version")?;
        let vector = config_get(conn, "vector_schema_version")?;
        if structural.as_deref() == Some(SCHEMA_VERSION)
            && vector.as_deref() == Some(VECTOR_SCHEMA_VERSION)
        {
            return Ok(());
        }
    } else if application_id(conn)? != 0 || has_tables(conn)? {
        return Err(Error::ForeignIndex {
            path: path.to_path_buf(),
        });
    }

    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(DbError::from)?;

    let app_id = application_id(&tx)?;
    let existing_tables = has_tables(&tx)?;
    if app_id != APPLICATION_ID && (app_id != 0 || existing_tables) {
        // A foreign database appeared between the fast-path check and the
        // write transaction (concurrent opener). Never touch it.
        return Err(Error::ForeignIndex {
            path: path.to_path_buf(),
        });
    }

    tx.execute_batch(CONFIG_DDL).map_err(DbError::from)?;
    tx.execute_batch(&format!("PRAGMA application_id = {APPLICATION_ID}"))
        .map_err(DbError::from)?;

    if existing_tables {
        if config_get(&tx, "schema_version")?.as_deref() != Some(SCHEMA_VERSION) {
            drop_tables(&tx, STRUCTURAL_TABLES)?;
            tx.execute_batch(STRUCTURAL_DDL).map_err(DbError::from)?;
            config_set(&tx, "schema_version", SCHEMA_VERSION)?;
            // Structural rebuild loses the document index; callers must
            // re-index before searching.
            config_set(&tx, "reindex_required", "1")?;
        }
        if config_get(&tx, "vector_schema_version")?.as_deref() != Some(VECTOR_SCHEMA_VERSION) {
            drop_tables(&tx, VECTOR_TABLES)?;
            tx.execute_batch(VECTOR_DDL).map_err(DbError::from)?;
            config_set(&tx, "vector_schema_version", VECTOR_SCHEMA_VERSION)?;
        }
    } else {
        tx.execute_batch(STRUCTURAL_DDL).map_err(DbError::from)?;
        tx.execute_batch(VECTOR_DDL).map_err(DbError::from)?;
        config_set(&tx, "schema_version", SCHEMA_VERSION)?;
        config_set(&tx, "vector_schema_version", VECTOR_SCHEMA_VERSION)?;
        config_set(&tx, "reindex_required", "0")?;
    }

    tx.commit().map_err(|e| DbError::from(e).into())
}

fn drop_tables(conn: &Connection, tables: &[&str]) -> Result<()> {
    for table in tables {
        conn.execute_batch(&format!("DROP TABLE IF EXISTS {table}"))
            .map_err(DbError::from)?;
    }
    Ok(())
}

pub(crate) use collections::*;
