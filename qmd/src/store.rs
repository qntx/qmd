//! Database store for document indexing and retrieval.
//!
//! This module provides all database operations, search functions, and document
//! retrieval for QMD.

use crate::collections::{find_context_for_path, list_collections as yaml_list_collections};
use crate::config::{EXCLUDE_DIRS, get_default_db_path};
use crate::error::{QmdError, Result};
use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

/// Document result with all metadata.
#[derive(Debug, Clone)]
pub struct DocumentResult {
    /// Full filesystem path.
    pub filepath: String,
    /// Short display path.
    pub display_path: String,
    /// Document title.
    pub title: String,
    /// Folder context description if configured.
    pub context: Option<String>,
    /// Content hash.
    pub hash: String,
    /// Short docid (first 6 chars of hash).
    pub docid: String,
    /// Parent collection name.
    pub collection_name: String,
    /// Relative path within collection.
    pub path: String,
    /// Last modification timestamp.
    pub modified_at: String,
    /// Body length in bytes.
    pub body_length: usize,
    /// Document body (optional).
    pub body: Option<String>,
}

/// Search result extends `DocumentResult` with score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The document.
    pub doc: DocumentResult,
    /// Relevance score (0-1).
    pub score: f64,
    /// Search source.
    pub source: SearchSource,
}

/// Search source type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchSource {
    /// Full-text search.
    Fts,
    /// Vector similarity search.
    Vec,
}

/// Collection info from database.
#[derive(Debug, Clone)]
pub struct CollectionInfo {
    /// Collection name.
    pub name: String,
    /// Working directory path.
    pub pwd: String,
    /// Glob pattern.
    pub glob_pattern: String,
    /// Number of active documents.
    pub active_count: usize,
    /// Last modification timestamp.
    pub last_modified: Option<String>,
}

/// Index status information.
#[derive(Debug, Clone)]
pub struct IndexStatus {
    /// Total active documents.
    pub total_documents: usize,
    /// Documents needing embedding.
    pub needs_embedding: usize,
    /// Whether vector index exists.
    pub has_vector_index: bool,
    /// Collection information.
    pub collections: Vec<CollectionInfo>,
}

/// The database store.
#[derive(Debug)]
pub struct Store {
    /// Database connection.
    conn: Connection,
    /// Database file path.
    db_path: PathBuf,
}

impl Store {
    /// Create a new store with default database path.
    pub fn new() -> Result<Self> {
        let db_path = get_default_db_path("index")
            .ok_or_else(|| QmdError::Config("Could not determine database path".to_string()))?;
        Self::open(&db_path)
    }

    /// Create a new store with explicit database path.
    pub fn open(db_path: &Path) -> Result<Self> {
        // Ensure parent directory exists.
        if let Some(parent) = db_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(db_path)?;
        let mut store = Self {
            conn,
            db_path: db_path.to_path_buf(),
        };
        store.initialize()?;
        Ok(store)
    }

    /// Get the database path.
    #[must_use]
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Initialize database schema.
    fn initialize(&mut self) -> Result<()> {
        self.conn.execute_batch(
            r"
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;

            -- Content-addressable storage
            CREATE TABLE IF NOT EXISTS content (
                hash TEXT PRIMARY KEY,
                doc TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            -- Documents table
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection TEXT NOT NULL,
                path TEXT NOT NULL,
                title TEXT NOT NULL,
                hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY (hash) REFERENCES content(hash) ON DELETE CASCADE,
                UNIQUE(collection, path)
            );

            CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection, active);
            CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
            CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path, active);

            -- FTS index
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                filepath, title, body,
                tokenize='porter unicode61'
            );

            -- LLM cache
            CREATE TABLE IF NOT EXISTS llm_cache (
                hash TEXT PRIMARY KEY,
                result TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            -- Content vectors metadata
            CREATE TABLE IF NOT EXISTS content_vectors (
                hash TEXT NOT NULL,
                seq INTEGER NOT NULL DEFAULT 0,
                pos INTEGER NOT NULL DEFAULT 0,
                model TEXT NOT NULL,
                embedded_at TEXT NOT NULL,
                PRIMARY KEY (hash, seq)
            );
            ",
        )?;

        // Create FTS triggers.
        self.create_fts_triggers()?;

        Ok(())
    }

    /// Create FTS synchronization triggers.
    fn create_fts_triggers(&self) -> Result<()> {
        // Check if triggers exist.
        let trigger_exists: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name='documents_ai'",
                [],
                |_| Ok(true),
            )
            .unwrap_or(false);

        if !trigger_exists {
            self.conn.execute_batch(
                r"
                CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents
                WHEN new.active = 1
                BEGIN
                    INSERT INTO documents_fts(rowid, filepath, title, body)
                    SELECT
                        new.id,
                        new.collection || '/' || new.path,
                        new.title,
                        (SELECT doc FROM content WHERE hash = new.hash)
                    WHERE new.active = 1;
                END;

                CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                    DELETE FROM documents_fts WHERE rowid = old.id;
                END;

                CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents
                BEGIN
                    DELETE FROM documents_fts WHERE rowid = old.id AND new.active = 0;
                    INSERT OR REPLACE INTO documents_fts(rowid, filepath, title, body)
                    SELECT
                        new.id,
                        new.collection || '/' || new.path,
                        new.title,
                        (SELECT doc FROM content WHERE hash = new.hash)
                    WHERE new.active = 1;
                END;
                ",
            )?;
        }

        Ok(())
    }

    /// Hash content using SHA256.
    #[must_use]
    pub fn hash_content(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Get short docid from hash (first 6 characters).
    #[must_use]
    pub fn get_docid(hash: &str) -> String {
        hash.chars().take(6).collect()
    }

    /// Extract title from markdown content.
    #[must_use]
    pub fn extract_title(content: &str) -> String {
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("# ") {
                return trimmed[2..].trim().to_string();
            }
            if trimmed.starts_with("## ") {
                return trimmed[3..].trim().to_string();
            }
        }
        String::new()
    }

    /// Handelize a path to be more token-friendly.
    #[must_use]
    pub fn handelize(path: &str) -> String {
        path.replace("___", "/")
            .to_lowercase()
            .split('/')
            .filter(|s| !s.is_empty())
            .map(|segment| {
                let cleaned: String = segment
                    .chars()
                    .map(|c| if c.is_alphanumeric() { c } else { '-' })
                    .collect();
                cleaned.trim_matches('-').to_string()
            })
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("/")
    }

    /// Insert content into content-addressable storage.
    pub fn insert_content(&self, hash: &str, content: &str, created_at: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?1, ?2, ?3)",
            params![hash, content, created_at],
        )?;
        Ok(())
    }

    /// Insert a document record.
    pub fn insert_document(
        &self,
        collection: &str,
        path: &str,
        title: &str,
        hash: &str,
        created_at: &str,
        modified_at: &str,
    ) -> Result<()> {
        self.conn.execute(
            r"
            INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1)
            ON CONFLICT(collection, path) DO UPDATE SET
                title = excluded.title,
                hash = excluded.hash,
                modified_at = excluded.modified_at,
                active = 1
            ",
            params![collection, path, title, hash, created_at, modified_at],
        )?;
        Ok(())
    }

    /// Find an active document by collection and path.
    pub fn find_active_document(
        &self,
        collection: &str,
        path: &str,
    ) -> Result<Option<(i64, String, String)>> {
        let result = self
            .conn
            .query_row(
                "SELECT id, hash, title FROM documents WHERE collection = ?1 AND path = ?2 AND active = 1",
                params![collection, path],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        Ok(result)
    }

    /// Update document title.
    pub fn update_document_title(
        &self,
        document_id: i64,
        title: &str,
        modified_at: &str,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE documents SET title = ?1, modified_at = ?2 WHERE id = ?3",
            params![title, modified_at, document_id],
        )?;
        Ok(())
    }

    /// Update document hash and title.
    pub fn update_document(
        &self,
        document_id: i64,
        title: &str,
        hash: &str,
        modified_at: &str,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE documents SET title = ?1, hash = ?2, modified_at = ?3 WHERE id = ?4",
            params![title, hash, modified_at, document_id],
        )?;
        Ok(())
    }

    /// Deactivate a document.
    pub fn deactivate_document(&self, collection: &str, path: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE documents SET active = 0 WHERE collection = ?1 AND path = ?2",
            params![collection, path],
        )?;
        Ok(())
    }

    /// Get all active document paths for a collection.
    pub fn get_active_document_paths(&self, collection: &str) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path FROM documents WHERE collection = ?1 AND active = 1")?;
        let paths = stmt
            .query_map(params![collection], |row| row.get(0))?
            .collect::<std::result::Result<Vec<String>, _>>()?;
        Ok(paths)
    }

    /// Full-text search using FTS5.
    pub fn search_fts(
        &self,
        query: &str,
        limit: usize,
        collection: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let sql = if collection.is_some() {
            r"
            SELECT
                d.collection,
                d.path,
                d.title,
                d.hash,
                d.modified_at,
                bm25(documents_fts) as score,
                LENGTH(c.doc) as body_length
            FROM documents_fts fts
            JOIN documents d ON d.id = fts.rowid
            JOIN content c ON c.hash = d.hash
            WHERE documents_fts MATCH ?1
              AND d.collection = ?2
              AND d.active = 1
            ORDER BY score
            LIMIT ?3
            "
        } else {
            r"
            SELECT
                d.collection,
                d.path,
                d.title,
                d.hash,
                d.modified_at,
                bm25(documents_fts) as score,
                LENGTH(c.doc) as body_length
            FROM documents_fts fts
            JOIN documents d ON d.id = fts.rowid
            JOIN content c ON c.hash = d.hash
            WHERE documents_fts MATCH ?1
              AND d.active = 1
            ORDER BY score
            LIMIT ?2
            "
        };

        let mut stmt = self.conn.prepare(sql)?;

        let results: Vec<SearchResult> = if let Some(coll) = collection {
            stmt.query_map(params![query, coll, limit as i64], |row| {
                let collection_name: String = row.get(0)?;
                let path: String = row.get(1)?;
                let title: String = row.get(2)?;
                let hash: String = row.get(3)?;
                let modified_at: String = row.get(4)?;
                let score: f64 = row.get(5)?;
                let body_length: i64 = row.get(6)?;
                let body_length = body_length as usize;

                Ok(SearchResult {
                    doc: DocumentResult {
                        filepath: format!("qmd://{collection_name}/{path}"),
                        display_path: format!("{collection_name}/{path}"),
                        title,
                        context: None,
                        hash: hash.clone(),
                        docid: Self::get_docid(&hash),
                        collection_name,
                        path,
                        modified_at,
                        body_length,
                        body: None,
                    },
                    score: -score, // BM25 returns negative scores, higher is better.
                    source: SearchSource::Fts,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            stmt.query_map(params![query, limit as i64], |row| {
                let collection_name: String = row.get(0)?;
                let path: String = row.get(1)?;
                let title: String = row.get(2)?;
                let hash: String = row.get(3)?;
                let modified_at: String = row.get(4)?;
                let score: f64 = row.get(5)?;
                let body_length: i64 = row.get(6)?;
                let body_length = body_length as usize;

                Ok(SearchResult {
                    doc: DocumentResult {
                        filepath: format!("qmd://{collection_name}/{path}"),
                        display_path: format!("{collection_name}/{path}"),
                        title,
                        context: None,
                        hash: hash.clone(),
                        docid: Self::get_docid(&hash),
                        collection_name,
                        path,
                        modified_at,
                        body_length,
                        body: None,
                    },
                    score: -score,
                    source: SearchSource::Fts,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?
        };

        // Add context to results.
        let results_with_context: Vec<SearchResult> = results
            .into_iter()
            .map(|mut r| {
                r.doc.context =
                    find_context_for_path(&r.doc.collection_name, &r.doc.path).unwrap_or(None);
                r
            })
            .collect();

        Ok(results_with_context)
    }

    /// Get document by collection and path.
    pub fn get_document(&self, collection: &str, path: &str) -> Result<Option<DocumentResult>> {
        let result = self
            .conn
            .query_row(
                r"
                SELECT
                    d.title,
                    d.hash,
                    d.modified_at,
                    c.doc,
                    LENGTH(c.doc) as body_length
                FROM documents d
                JOIN content c ON c.hash = d.hash
                WHERE d.collection = ?1 AND d.path = ?2 AND d.active = 1
                ",
                params![collection, path],
                |row| {
                    let title: String = row.get(0)?;
                    let hash: String = row.get(1)?;
                    let modified_at: String = row.get(2)?;
                    let body: String = row.get(3)?;
                    let body_length: i64 = row.get(4)?;
                    let body_length = body_length as usize;

                    Ok(DocumentResult {
                        filepath: format!("qmd://{collection}/{path}"),
                        display_path: format!("{collection}/{path}"),
                        title,
                        context: None,
                        hash: hash.clone(),
                        docid: Self::get_docid(&hash),
                        collection_name: collection.to_string(),
                        path: path.to_string(),
                        modified_at,
                        body_length,
                        body: Some(body),
                    })
                },
            )
            .optional()?;

        // Add context if document found.
        let result = result.map(|mut doc| {
            doc.context = find_context_for_path(collection, path).unwrap_or(None);
            doc
        });

        Ok(result)
    }

    /// Get document by docid (first 6 chars of hash).
    pub fn find_document_by_docid(&self, docid: &str) -> Result<Option<(String, String)>> {
        let clean_docid = docid.trim_start_matches('#');
        let result = self
            .conn
            .query_row(
                r"
                SELECT d.collection, d.path
                FROM documents d
                WHERE d.hash LIKE ?1 || '%' AND d.active = 1
                LIMIT 1
                ",
                params![clean_docid],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        Ok(result)
    }

    /// List collections with stats from database.
    pub fn list_collections(&self) -> Result<Vec<CollectionInfo>> {
        let yaml_collections = yaml_list_collections()?;

        let mut collections = Vec::new();

        for coll in yaml_collections {
            let stats: (i64, Option<String>) = self
                .conn
                .query_row(
                    r"
                    SELECT COUNT(*) as count, MAX(modified_at) as last_modified
                    FROM documents
                    WHERE collection = ?1 AND active = 1
                    ",
                    params![coll.name],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap_or((0, None));

            collections.push(CollectionInfo {
                name: coll.name,
                pwd: coll.path,
                glob_pattern: coll.pattern,
                active_count: stats.0 as usize,
                last_modified: stats.1,
            });
        }

        Ok(collections)
    }

    /// Get index status.
    pub fn get_status(&self) -> Result<IndexStatus> {
        let total_documents: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM documents WHERE active = 1",
            [],
            |row| row.get(0),
        )?;
        let total_documents = total_documents as usize;

        let needs_embedding: i64 = self.conn.query_row(
            r"
            SELECT COUNT(DISTINCT d.hash)
            FROM documents d
            LEFT JOIN content_vectors v ON d.hash = v.hash AND v.seq = 0
            WHERE d.active = 1 AND v.hash IS NULL
            ",
            [],
            |row| row.get(0),
        )?;
        let needs_embedding = needs_embedding as usize;

        let has_vector_index: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vectors_vec'",
                [],
                |_| Ok(true),
            )
            .unwrap_or(false);

        let collections = self.list_collections()?;

        Ok(IndexStatus {
            total_documents,
            needs_embedding,
            has_vector_index,
            collections,
        })
    }

    /// Remove a collection and its documents from the database.
    pub fn remove_collection_documents(&self, name: &str) -> Result<(usize, usize)> {
        // Get count before deletion.
        let doc_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM documents WHERE collection = ?1",
            params![name],
            |row| row.get(0),
        )?;
        let doc_count = doc_count as usize;

        // Delete documents.
        self.conn
            .execute("DELETE FROM documents WHERE collection = ?1", params![name])?;

        // Cleanup orphaned content.
        let cleaned = self.cleanup_orphaned_content()?;

        Ok((doc_count, cleaned))
    }

    /// Rename collection in database.
    pub fn rename_collection_documents(&self, old_name: &str, new_name: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE documents SET collection = ?1 WHERE collection = ?2",
            params![new_name, old_name],
        )?;
        Ok(())
    }

    /// Cleanup orphaned content (not referenced by any active document).
    pub fn cleanup_orphaned_content(&self) -> Result<usize> {
        let changes = self.conn.execute(
            "DELETE FROM content WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
            [],
        )?;
        Ok(changes)
    }

    /// Cleanup orphaned vectors.
    pub fn cleanup_orphaned_vectors(&self) -> Result<usize> {
        let changes = self.conn.execute(
            r"
            DELETE FROM content_vectors
            WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
            ",
            [],
        )?;
        Ok(changes)
    }

    /// Delete inactive documents.
    pub fn delete_inactive_documents(&self) -> Result<usize> {
        let changes = self
            .conn
            .execute("DELETE FROM documents WHERE active = 0", [])?;
        Ok(changes)
    }

    /// Clear LLM cache.
    pub fn clear_cache(&self) -> Result<usize> {
        let changes = self.conn.execute("DELETE FROM llm_cache", [])?;
        Ok(changes)
    }

    /// Vacuum database.
    pub fn vacuum(&self) -> Result<()> {
        self.conn.execute("VACUUM", [])?;
        Ok(())
    }

    /// Ensure the vector table exists with the correct dimensions.
    pub fn ensure_vector_table(&self, _dimensions: usize) -> Result<()> {
        // Create vectors_vec table for storing embeddings
        self.conn.execute(
            &format!(
                r"
                CREATE TABLE IF NOT EXISTS vectors_vec (
                    hash_seq TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL
                )
                "
            ),
            [],
        )?;
        Ok(())
    }

    /// Insert an embedding for a content hash.
    pub fn insert_embedding(
        &self,
        hash: &str,
        seq: usize,
        pos: usize,
        embedding: &[f32],
        model: &str,
        embedded_at: &str,
    ) -> Result<()> {
        // Insert metadata
        self.conn.execute(
            r"
            INSERT OR REPLACE INTO content_vectors (hash, seq, pos, model, embedded_at)
            VALUES (?1, ?2, ?3, ?4, ?5)
            ",
            params![hash, seq as i64, pos as i64, model, embedded_at],
        )?;

        // Insert vector data
        let hash_seq = format!("{hash}_{seq}");
        let embedding_bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

        self.conn.execute(
            "INSERT OR REPLACE INTO vectors_vec (hash_seq, embedding) VALUES (?1, ?2)",
            params![hash_seq, embedding_bytes],
        )?;

        Ok(())
    }

    /// Get hashes that need embedding.
    pub fn get_hashes_needing_embedding(&self) -> Result<Vec<(String, String, String)>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT DISTINCT d.hash, d.path, c.doc
            FROM documents d
            JOIN content c ON c.hash = d.hash
            LEFT JOIN content_vectors v ON d.hash = v.hash AND v.seq = 0
            WHERE d.active = 1 AND v.hash IS NULL
            ",
        )?;

        let results = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(results)
    }

    /// Get embedding for a hash.
    pub fn get_embedding(&self, hash: &str, seq: usize) -> Result<Option<Vec<f32>>> {
        let hash_seq = format!("{hash}_{seq}");
        let result: Option<Vec<u8>> = self
            .conn
            .query_row(
                "SELECT embedding FROM vectors_vec WHERE hash_seq = ?1",
                params![hash_seq],
                |row| row.get(0),
            )
            .optional()?;

        Ok(result.map(|bytes| {
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        }))
    }

    /// Vector similarity search.
    pub fn search_vec(
        &self,
        query_embedding: &[f32],
        limit: usize,
        collection: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        // Get all embeddings and compute similarity
        let sql = if collection.is_some() {
            r"
            SELECT DISTINCT
                d.collection,
                d.path,
                d.title,
                d.hash,
                d.modified_at,
                LENGTH(c.doc) as body_length,
                v.hash_seq
            FROM documents d
            JOIN content c ON c.hash = d.hash
            JOIN vectors_vec v ON v.hash_seq = d.hash || '_0'
            WHERE d.active = 1 AND d.collection = ?1
            "
        } else {
            r"
            SELECT DISTINCT
                d.collection,
                d.path,
                d.title,
                d.hash,
                d.modified_at,
                LENGTH(c.doc) as body_length,
                v.hash_seq
            FROM documents d
            JOIN content c ON c.hash = d.hash
            JOIN vectors_vec v ON v.hash_seq = d.hash || '_0'
            WHERE d.active = 1
            "
        };

        let mut stmt = self.conn.prepare(sql)?;

        let rows: Vec<(String, String, String, String, String, usize, String)> =
            if let Some(coll) = collection {
                stmt.query_map(params![coll], |row| {
                    let body_length: i64 = row.get(5)?;
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        body_length as usize,
                        row.get(6)?,
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?
            } else {
                stmt.query_map([], |row| {
                    let body_length: i64 = row.get(5)?;
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        body_length as usize,
                        row.get(6)?,
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?
            };

        // Compute similarities
        let mut results: Vec<SearchResult> = Vec::new();

        for (collection_name, path, title, hash, modified_at, body_length, _hash_seq) in rows {
            if let Some(doc_embedding) = self.get_embedding(&hash, 0)? {
                let similarity = crate::llm::cosine_similarity(query_embedding, &doc_embedding);

                results.push(SearchResult {
                    doc: DocumentResult {
                        filepath: format!("qmd://{collection_name}/{path}"),
                        display_path: format!("{collection_name}/{path}"),
                        title,
                        context: None,
                        hash: hash.clone(),
                        docid: Self::get_docid(&hash),
                        collection_name: collection_name.clone(),
                        path: path.clone(),
                        modified_at,
                        body_length,
                        body: None,
                    },
                    score: similarity as f64,
                    source: SearchSource::Vec,
                });
            }
        }

        // Sort by similarity (descending) and limit
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        // Add context
        let results_with_context: Vec<SearchResult> = results
            .into_iter()
            .map(|mut r| {
                r.doc.context =
                    find_context_for_path(&r.doc.collection_name, &r.doc.path).unwrap_or(None);
                r
            })
            .collect();

        Ok(results_with_context)
    }

    /// Clear all embeddings.
    pub fn clear_embeddings(&self) -> Result<usize> {
        let changes1 = self.conn.execute("DELETE FROM content_vectors", [])?;
        let _ = self.conn.execute("DELETE FROM vectors_vec", []);
        Ok(changes1)
    }

    /// List files in a collection.
    pub fn list_files(
        &self,
        collection: &str,
        path_prefix: Option<&str>,
    ) -> Result<Vec<(String, String, String, usize)>> {
        let mut stmt;
        let files: Vec<(String, String, String, usize)> = if let Some(prefix) = path_prefix {
            let prefix_pattern = format!("{prefix}%");
            stmt = self.conn.prepare(
                r"
                SELECT d.path, d.title, d.modified_at, LENGTH(c.doc) as size
                FROM documents d
                JOIN content c ON d.hash = c.hash
                WHERE d.collection = ?1 AND d.path LIKE ?2 AND d.active = 1
                ORDER BY d.path
                ",
            )?;
            stmt.query_map(params![collection, prefix_pattern], |row| {
                let size: i64 = row.get(3)?;
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, size as usize))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            stmt = self.conn.prepare(
                r"
                SELECT d.path, d.title, d.modified_at, LENGTH(c.doc) as size
                FROM documents d
                JOIN content c ON d.hash = c.hash
                WHERE d.collection = ?1 AND d.active = 1
                ORDER BY d.path
                ",
            )?;
            stmt.query_map(params![collection], |row| {
                let size: i64 = row.get(3)?;
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, size as usize))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?
        };

        Ok(files)
    }
}

/// Check if a path should be excluded from indexing.
pub fn should_exclude(path: &Path) -> bool {
    for component in path.components() {
        if let std::path::Component::Normal(name) = component {
            let name_str = name.to_string_lossy();
            if name_str.starts_with('.') || EXCLUDE_DIRS.contains(&name_str.as_ref()) {
                return true;
            }
        }
    }
    false
}

/// Check if a string looks like a docid.
pub fn is_docid(s: &str) -> bool {
    let clean = s.trim_start_matches('#');
    clean.len() == 6 && clean.chars().all(|c| c.is_ascii_hexdigit())
}

/// Parse a virtual path like "qmd://collection/path".
pub fn parse_virtual_path(path: &str) -> Option<(String, String)> {
    let normalized = normalize_virtual_path(path);
    let stripped = normalized.strip_prefix("qmd://")?;
    let mut parts = stripped.splitn(2, '/');
    let collection = parts.next()?.to_string();
    let file_path = parts.next().unwrap_or("").to_string();
    Some((collection, file_path))
}

/// Build a virtual path from collection and path.
pub fn build_virtual_path(collection: &str, path: &str) -> String {
    format!("qmd://{collection}/{path}")
}

/// Check if a path is a virtual path.
pub fn is_virtual_path(path: &str) -> bool {
    let trimmed = path.trim();
    trimmed.starts_with("qmd:") || trimmed.starts_with("//")
}

/// Normalize virtual path format.
pub fn normalize_virtual_path(input: &str) -> String {
    let path = input.trim();

    if let Some(rest) = path.strip_prefix("qmd:") {
        let rest = rest.trim_start_matches('/');
        return format!("qmd://{rest}");
    }

    if path.starts_with("//") {
        let rest = path.trim_start_matches('/');
        return format!("qmd://{rest}");
    }

    path.to_string()
}
