//! The [`Qmd`] handle and its builder: open a database, wire the config
//! source, and expose collection/context management.
//!
//! Mutation flow is uniform: config is the source of truth, the database
//! mirrors it. Each mutation loads the config (file, inline, or
//! DB-assembled), edits it, persists it (for file and inline sources),
//! then syncs the `store_collections`/`store_config` mirror.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use rusqlite::Connection;

use crate::config::{Collection, Config};
use crate::document::{Document, DocumentEntry, GetOptions, LineRange, MultiGet, MultiGetOptions};
use crate::env::Environment;
use crate::error::{Error, Result};
use crate::maintenance::Maintenance;
use crate::store;
use crate::store::search::SearchResult;

/// How the [`Qmd`] handle resolves its configuration.
#[derive(Debug)]
enum ConfigSource {
    /// Load and persist `index-<name>.yml` / `.qmd/index.yaml`.
    File(PathBuf),
    /// Caller-supplied config, mutated in memory only.
    Inline(Box<Config>),
    /// No config source; the database `store_collections` mirror is the
    /// only source of truth.
    DbOnly,
}

/// Parameters for [`Qmd::add_collection`].
#[derive(Debug, Clone, Default)]
pub struct CollectionSpec {
    /// Filesystem path, stored verbatim (the CLI resolves it first).
    pub path: String,
    /// Glob pattern; `None` stores no key and applies the default.
    pub pattern: Option<String>,
    /// Ignore patterns; an empty list stores no key.
    pub ignore: Vec<String>,
}

impl CollectionSpec {
    /// Effective glob pattern (`None` resolves to
    /// [`crate::config::DEFAULT_PATTERN`]).
    #[must_use]
    pub fn pattern_or_default(&self) -> &str {
        self.pattern
            .as_deref()
            .unwrap_or(crate::config::DEFAULT_PATTERN)
    }
}

/// Settings updatable via [`Qmd::update_collection_settings`].
#[derive(Debug, Clone, Default)]
pub struct CollectionSettings {
    /// `Some(Some(cmd))` sets the pre-update hook, `Some(None)` clears it,
    /// `None` leaves it unchanged.
    #[allow(
        clippy::option_option,
        reason = "three-way semantics: set / clear / leave unchanged"
    )]
    pub update: Option<Option<String>>,
    /// `Some(_)` replaces the include-by-default flag.
    pub include_by_default: Option<bool>,
}

/// Result of [`Qmd::remove_collection`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CollectionRemoval {
    /// Documents deactivated (removed from search) by the removal.
    pub documents_deactivated: usize,
}

/// One row of [`Qmd::collections`], mirroring upstream `listCollections`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CollectionInfo {
    /// Collection name.
    pub name: String,
    /// Filesystem path.
    pub path: String,
    /// Glob pattern.
    pub pattern: String,
    /// Total document rows (including inactive).
    pub doc_count: usize,
    /// Active document rows.
    pub active_count: usize,
    /// `MAX(modified_at)` of active documents, as stored.
    pub last_modified: Option<String>,
    /// Include-in-default-search flag.
    pub include_by_default: bool,
}

/// One entry of [`Qmd::contexts`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ContextEntry {
    /// Collection name, or `"*"` for the global context.
    pub collection: String,
    /// Path prefix (`"/"` for the global context).
    pub path: String,
    /// Context text.
    pub context: String,
}

/// Result of [`Qmd::detect_collection`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CollectionPath {
    /// Collection covering the filesystem path.
    pub name: String,
    /// Path relative to the collection root.
    pub relative_path: String,
}

/// Options for [`Qmd::update`], mirroring upstream `UpdateOptions`
/// (index.ts:536-561).
#[derive(Debug, Clone, Default)]
pub struct UpdateOptions {
    /// Restrict the update to these collection names; `None` or empty
    /// updates all collections. Unknown names are ignored, like
    /// upstream's `includes()` filter.
    pub collections: Option<Vec<String>>,
}

/// One collection row of [`Qmd::status`] — upstream `CollectionInfo`
/// (store.ts:2525-2531).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StatusCollection {
    /// Collection name.
    pub name: String,
    /// Filesystem path (`None` when only document rows exist).
    pub path: Option<String>,
    /// Glob pattern, if known.
    pub pattern: Option<String>,
    /// Active document count.
    pub documents: usize,
    /// Latest `modified_at` (current time when the collection is empty).
    pub last_updated: String,
}

/// [`Qmd::status`] result — upstream `IndexStatus` (store.ts:2533-2540)
/// plus the CLI `status` aggregates (cli/qmd.ts:540-583) and the
/// `reindex_required` flag (P1-D2). `pending_metadata` arrives with P3.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Status {
    /// Active documents.
    pub total_documents: usize,
    /// Distinct active content hashes lacking vectors for the resolved
    /// embed model.
    pub needs_embedding: usize,
    /// Whether the `vectors_vec` table exists.
    pub has_vector_index: bool,
    /// Total `content_vectors` rows.
    pub vector_count: usize,
    /// `content_vectors` rows referenced by no active document.
    pub orphaned_vectors: usize,
    /// `MAX(modified_at)` of active documents.
    pub latest_modified: Option<String>,
    /// Schema rebuild requires re-indexing before search works.
    pub reindex_required: bool,
    /// Per-collection rows, most recently updated first.
    pub collections: Vec<StatusCollection>,
}

/// [`Qmd::index_health`] result — upstream `IndexHealthInfo`
/// (store.ts:2568-2572).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct IndexHealth {
    /// Distinct active content hashes lacking current vectors.
    pub needs_embedding: usize,
    /// Active document count.
    pub total_docs: usize,
    /// Whole days since the newest `modified_at` (`None` when empty).
    pub days_stale: Option<i64>,
}

/// Options for [`Qmd::search_lex`], mirroring the `limit` and
/// `collection` fields of upstream `searchLex` options
/// (index.ts:474-477). The upstream `filter` field is metadata-based and
/// arrives with P3.
#[derive(Debug, Clone, Default)]
pub struct LexOptions {
    /// Maximum results (upstream default: 20).
    pub limit: Option<usize>,
    /// Restrict the search to these collection names; `None` or empty
    /// searches all collections. Several names are queried separately
    /// and merged by score (upstream #775).
    pub collections: Option<Vec<String>>,
}

/// One progress event from [`Qmd::update`], mirroring upstream
/// `UpdateProgress`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct UpdateProgress {
    /// Collection currently being indexed.
    pub collection: String,
    /// Relative file path, `/`-separated.
    pub file: String,
    /// Files processed in this collection so far (including this one).
    pub current: usize,
    /// Total files in this collection's scan.
    pub total: usize,
}

/// Aggregate result of [`Qmd::update`], mirroring upstream
/// `UpdateResult` minus per-collection detail.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct UpdateReport {
    /// Collections processed.
    pub collections: usize,
    /// Newly indexed files.
    pub indexed: usize,
    /// Re-indexed files whose hash or title changed.
    pub updated: usize,
    /// Files already up to date.
    pub unchanged: usize,
    /// Documents deactivated because the file vanished.
    pub removed: usize,
    /// Files skipped (unreadable, out of root, ...).
    pub skipped: usize,
    /// Distinct active content hashes with no vector rows; the embedding
    /// pipeline arrives in P2, so every active hash counts for now.
    pub needs_embedding: usize,
}

/// Open handle to a qmd index.
///
/// `Send + Sync`: the connection and config are each guarded by a
/// [`Mutex`]. Dropping the handle closes the database; use
/// [`Qmd::close`] to observe shutdown errors.
pub struct Qmd {
    conn: Mutex<Connection>,
    config: Mutex<ConfigSource>,
    db_path: PathBuf,
    env: Environment,
}

impl std::fmt::Debug for Qmd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qmd")
            .field("db_path", &self.db_path)
            .finish_non_exhaustive()
    }
}

/// Recoverable-poison lock: a panicking writer between calls does not
/// corrupt state because every mutation is wrapped in a transaction or a
/// full config rewrite, so keeping the guard is safe.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Builder for [`Qmd`], mirroring upstream `createStore(options)`.
#[derive(Debug)]
#[must_use]
pub struct QmdBuilder {
    db_path: PathBuf,
    config: Option<ConfigSource>,
    env: Environment,
}

impl QmdBuilder {
    /// Override the injected environment (tests and embedders).
    pub fn environment(mut self, env: Environment) -> Self {
        self.env = env;
        self
    }

    /// Load and persist configuration from this YAML file
    /// (`configPath` in upstream). Overrides any inline config.
    pub fn config_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.config = Some(ConfigSource::File(path.into()));
        self
    }

    /// Use this configuration in memory only; file I/O is skipped
    /// (`config` in upstream).
    pub fn config(mut self, config: Config) -> Self {
        self.config = Some(ConfigSource::Inline(Box::new(config)));
        self
    }

    /// Open the database, initialize schema/versioning, and sync the
    /// config source into `store_collections`.
    ///
    /// # Errors
    /// [`Error::ForeignIndex`] for non-qmd-rs databases, [`Error::Config`]
    /// for malformed config files, [`Error::Db`] for SQLite failures.
    pub fn build(self) -> Result<Qmd> {
        let mut conn = store::open(&self.env, &self.db_path)?;
        let source = self.config.unwrap_or(ConfigSource::DbOnly);
        match &source {
            ConfigSource::File(path) => {
                let config = crate::config::load(path)?;
                store::sync_config_to_db(&mut conn, &config, false)?;
            }
            ConfigSource::Inline(config) => {
                store::sync_config_to_db(&mut conn, config, false)?;
            }
            ConfigSource::DbOnly => {}
        }
        Ok(Qmd {
            conn: Mutex::new(conn),
            config: Mutex::new(source),
            db_path: self.db_path,
            env: self.env,
        })
    }
}

impl Qmd {
    /// Start building a handle for the database at `db_path`.
    pub fn builder(db_path: impl Into<PathBuf>) -> QmdBuilder {
        QmdBuilder {
            db_path: db_path.into(),
            config: None,
            env: Environment::from_process(),
        }
    }

    /// Path of the underlying database file.
    #[must_use]
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Connection guard for [`Maintenance`] and internal helpers.
    pub(crate) fn lock_conn(&self) -> MutexGuard<'_, Connection> {
        lock(&self.conn)
    }

    /// The resolved embed model: `models.embed` → `QMD_EMBED_MODEL` →
    /// upstream default (llm.ts:303-305).
    fn embed_model(&self, config: &Config) -> String {
        crate::llm::resolve_embed_model(config.models.as_ref(), &self.env)
    }

    /// Close the handle, surfacing connection errors that `Drop` would
    /// otherwise swallow.
    ///
    /// # Errors
    /// [`Error::Db`] if SQLite reports a shutdown failure.
    pub fn close(self) -> Result<()> {
        let conn = match self.conn.into_inner() {
            Ok(conn) => conn,
            Err(e) => e.into_inner(),
        };
        conn.close()
            .map_err(|(_c, e)| crate::error::DbError::from(e))?;
        Ok(())
    }

    /// Current configuration: loaded from the file or inline source, or
    /// assembled from `store_collections` in DB-only mode.
    ///
    /// # Errors
    /// [`Error::Config`]/[`Error::Io`] when a file source cannot be read.
    pub fn config(&self) -> Result<Config> {
        let guard = lock(&self.config);
        self.load_config(&guard)
    }

    /// Add a collection. Applies upstream CLI semantics: the name must be
    /// free and no other collection may cover the same path + pattern.
    ///
    /// # Errors
    /// [`Error::CollectionExists`], [`Error::DuplicateCollectionSource`],
    /// [`Error::InvalidInput`] on an empty name.
    pub fn add_collection(&self, name: &str, spec: &CollectionSpec) -> Result<()> {
        if name.is_empty() {
            return Err(Error::InvalidInput {
                reason: "collection name must not be empty".to_owned(),
            });
        }
        self.mutate_config(|cfg| {
            if cfg.collections.contains_key(name) {
                return Err(Error::CollectionExists {
                    name: name.to_owned(),
                });
            }
            let duplicate = cfg.collections.iter().find(|(_, coll)| {
                coll.path == spec.path && coll.pattern_or_default() == spec.pattern_or_default()
            });
            if let Some((existing, _)) = duplicate {
                return Err(Error::DuplicateCollectionSource {
                    existing: existing.clone(),
                });
            }
            cfg.collections.insert(
                name.to_owned(),
                Collection {
                    path: spec.path.clone(),
                    pattern: spec.pattern.clone(),
                    ignore: (!spec.ignore.is_empty()).then(|| spec.ignore.clone()),
                    ..Collection::default()
                },
            );
            Ok(Some(()))
        })?;
        Ok(())
    }

    /// Remove a collection by name. The collection's documents are
    /// deactivated (hidden from search, reclaimable by `cleanup`), which
    /// matches CLI semantics rather than the upstream SDK behavior of
    /// leaving them searchable.
    ///
    /// Returns `Ok(None)` when the name is unknown.
    ///
    /// # Errors
    /// Propagates config I/O and database errors.
    pub fn remove_collection(&self, name: &str) -> Result<Option<CollectionRemoval>> {
        Ok(self
            .mutate_config(|cfg| Ok(cfg.collections.shift_remove(name).map(|_| ())))?
            .map(|((), outcome)| CollectionRemoval {
                documents_deactivated: outcome.documents_deactivated,
            }))
    }

    /// Rename a collection, updating its documents' `collection` field
    /// (CLI semantics; the upstream SDK leaves them under the old name).
    ///
    /// Returns `Ok(false)` when `old` does not exist.
    ///
    /// # Errors
    /// [`Error::CollectionExists`] when `new` is already taken.
    pub fn rename_collection(&self, old: &str, new: &str) -> Result<bool> {
        // After the config is persisted, `documents` rows are renamed
        // before the sync pass runs — otherwise the sync would see `old`
        // as removed and deactivate the documents we just moved.
        Ok(self
            .mutate_config_then(
                |cfg| {
                    if !cfg.collections.contains_key(old) {
                        return Ok(None);
                    }
                    if cfg.collections.contains_key(new) {
                        return Err(Error::CollectionExists {
                            name: new.to_owned(),
                        });
                    }
                    if let Some(coll) = cfg.collections.shift_remove(old) {
                        cfg.collections.insert(new.to_owned(), coll);
                    }
                    Ok(Some(()))
                },
                |conn| store::rename_documents(conn, old, new),
            )?
            .is_some())
    }

    /// Update a collection's settings (`update` hook and/or
    /// `includeByDefault`).
    ///
    /// # Errors
    /// [`Error::CollectionNotFound`] when the name is unknown.
    pub fn update_collection_settings(
        &self,
        name: &str,
        settings: &CollectionSettings,
    ) -> Result<()> {
        self.mutate_config(|cfg| {
            let Some(coll) = cfg.collections.get_mut(name) else {
                return Err(Error::CollectionNotFound {
                    name: name.to_owned(),
                });
            };
            if let Some(update) = &settings.update {
                coll.update.clone_from(update);
            }
            if let Some(include) = settings.include_by_default {
                coll.include_by_default = Some(include);
            }
            Ok(Some(()))
        })?;
        Ok(())
    }

    /// One collection's configuration (`collection show`).
    ///
    /// # Errors
    /// Propagates config load errors.
    pub fn collection(&self, name: &str) -> Result<Option<Collection>> {
        let guard = lock(&self.config);
        Ok(self.load_config(&guard)?.collections.get(name).cloned())
    }

    /// All collections with document statistics (`collection list`),
    /// in `store_collections` insertion order.
    ///
    /// # Errors
    /// [`Error::Db`] on query failure.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "the connection guard must live until the last query"
    )]
    pub fn collections(&self) -> Result<Vec<CollectionInfo>> {
        let conn = lock(&self.conn);
        let mut out = Vec::new();
        for coll in store::get_store_collections(&conn)? {
            let (doc_count, active_count, last_modified): (i64, i64, Option<String>) = conn
                .query_row(
                    "SELECT COUNT(*), \
                     SUM(CASE WHEN active = 1 THEN 1 ELSE 0 END), \
                     MAX(CASE WHEN active = 1 THEN modified_at END) \
                     FROM documents WHERE collection = ?",
                    [&coll.name],
                    |r| {
                        Ok((
                            r.get(0)?,
                            r.get::<_, Option<i64>>(1)?.unwrap_or(0),
                            r.get(2)?,
                        ))
                    },
                )
                .map_err(crate::error::DbError::from)?;
            out.push(CollectionInfo {
                name: coll.name,
                path: coll.path,
                pattern: coll.pattern,
                doc_count: doc_count.try_into().unwrap_or(0),
                active_count: active_count.try_into().unwrap_or(0),
                last_modified,
                include_by_default: coll.include_by_default,
            });
        }
        Ok(out)
    }

    /// Names of collections included in unscoped search
    /// (`getDefaultCollectionNames`).
    ///
    /// # Errors
    /// [`Error::Db`] on query failure.
    pub fn default_collection_names(&self) -> Result<Vec<String>> {
        Ok(self
            .collections()?
            .into_iter()
            .filter(|c| c.include_by_default)
            .map(|c| c.name)
            .collect())
    }

    /// Global context text, or `None` when unset.
    ///
    /// # Errors
    /// [`Error::Db`] on query failure.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "the connection guard must live until the query completes"
    )]
    pub fn global_context(&self) -> Result<Option<String>> {
        let conn = lock(&self.conn);
        store::get_store_global_context(&conn)
    }

    /// Set or clear the global context (`qmd context add /`).
    ///
    /// # Errors
    /// Propagates config I/O and database errors.
    pub fn set_global_context(&self, text: Option<&str>) -> Result<()> {
        self.mutate_config(|cfg| {
            cfg.global_context = text.map(str::to_owned);
            Ok(Some(()))
        })?;
        Ok(())
    }

    /// All contexts, global first (`getStoreContexts`).
    ///
    /// # Errors
    /// [`Error::Db`] on query failure.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "the connection guard must live until the query completes"
    )]
    pub fn contexts(&self) -> Result<Vec<ContextEntry>> {
        let conn = lock(&self.conn);
        Ok(store::get_store_contexts(&conn)?
            .into_iter()
            .map(|r| ContextEntry {
                collection: r.collection,
                path: r.path,
                context: r.context,
            })
            .collect())
    }

    /// Set a path-prefix context on a collection (`qmd context add`).
    /// The prefix is stored verbatim, like upstream.
    ///
    /// # Errors
    /// [`Error::CollectionNotFound`] when the collection is unknown.
    pub fn add_context(&self, collection: &str, path_prefix: &str, text: &str) -> Result<()> {
        self.mutate_config(|cfg| {
            let Some(coll) = cfg.collections.get_mut(collection) else {
                return Err(Error::CollectionNotFound {
                    name: collection.to_owned(),
                });
            };
            coll.context
                .get_or_insert_with(indexmap::IndexMap::new)
                .insert(path_prefix.to_owned(), text.to_owned());
            Ok(Some(()))
        })?;
        Ok(())
    }

    /// Remove a path-prefix context (`context rm`). Returns `Ok(false)`
    /// when the collection or prefix is unknown.
    ///
    /// # Errors
    /// Propagates config I/O and database errors.
    pub fn remove_context(&self, collection: &str, path_prefix: &str) -> Result<bool> {
        Ok(self
            .mutate_config(|cfg| {
                let Some(coll) = cfg.collections.get_mut(collection) else {
                    return Ok(None);
                };
                let Some(map) = coll.context.as_mut() else {
                    return Ok(None);
                };
                if map.shift_remove(path_prefix).is_none() {
                    return Ok(None);
                }
                if map.is_empty() {
                    coll.context = None;
                }
                Ok(Some(()))
            })?
            .is_some())
    }

    /// Find the collection covering a filesystem path and return the
    /// relative path inside it (`getCollectionForPath`).
    ///
    /// # Errors
    /// Propagates config load errors.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "the config guard must live for the duration of the load"
    )]
    pub fn detect_collection(&self, fs_path: &Path) -> Result<Option<CollectionPath>> {
        let guard = lock(&self.config);
        let config = self.load_config(&guard)?;
        let real = fs_path
            .canonicalize()
            .unwrap_or_else(|_| fs_path.to_path_buf());
        let mut best: Option<CollectionPath> = None;
        let mut best_len = 0usize;
        for (name, coll) in &config.collections {
            // Canonicalize both sides so collections stored under a
            // symlinked path (e.g. macOS /var -> /private/var) still
            // match — upstream realpaths only the file side and misses.
            let root_path = Path::new(&coll.path)
                .canonicalize()
                .unwrap_or_else(|_| PathBuf::from(&coll.path));
            let root_owned = root_path.to_string_lossy().into_owned();
            let root = root_owned.trim_end_matches('/');
            if root.is_empty() {
                continue;
            }
            let prefix = format!("{root}/");
            let real_str = real.to_string_lossy();
            if !real_str.starts_with(&prefix) && real_str != root {
                continue;
            }
            if root.len() > best_len {
                best_len = root.len();
                let relative = real_str.strip_prefix(&prefix).unwrap_or("").to_owned();
                best = Some(CollectionPath {
                    name: name.clone(),
                    relative_path: relative,
                });
            }
        }
        Ok(best)
    }

    /// Re-index collections: upstream `update` (index.ts:536-561).
    ///
    /// Collections come from the `store_collections` mirror (like
    /// upstream, not the config file), `llm_cache` is cleared first, then
    /// each collection runs `reindexCollection`. `progress` fires once
    /// per file, skipped files included.
    ///
    /// # Errors
    /// Propagates glob, I/O and database errors.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "the connection guard is held for the whole update pass"
    )]
    pub fn update(
        &self,
        opts: &UpdateOptions,
        progress: &mut dyn FnMut(&UpdateProgress),
    ) -> Result<UpdateReport> {
        // Resolve before locking conn: `load_config` may itself lock it
        // (DbOnly mode) and `Mutex` is not reentrant.
        let model = {
            let guard = lock(&self.config);
            self.embed_model(&self.load_config(&guard)?)
        };
        let mut conn = lock(&self.conn);
        let cols = store::get_store_collections(&conn)?;
        let filtered: Vec<_> = match &opts.collections {
            Some(names) if !names.is_empty() => cols
                .into_iter()
                .filter(|c| names.contains(&c.name))
                .collect(),
            _ => cols,
        };

        store::maintenance::delete_llm_cache(&conn)?;

        let mut report = UpdateReport {
            collections: filtered.len(),
            indexed: 0,
            updated: 0,
            unchanged: 0,
            removed: 0,
            skipped: 0,
            needs_embedding: 0,
        };
        for col in &filtered {
            let ignore = col.ignore.clone().unwrap_or_default();
            let pattern = if col.pattern.is_empty() {
                crate::config::DEFAULT_PATTERN
            } else {
                &col.pattern
            };
            let name = col.name.clone();
            let mut on_file = |fp: &crate::collection::index::FileProgress<'_>| {
                progress(&UpdateProgress {
                    collection: name.clone(),
                    file: fp.file.to_owned(),
                    current: fp.current,
                    total: fp.total,
                });
            };
            let r = crate::collection::index::reindex_collection(
                &mut conn,
                Path::new(&col.path),
                pattern,
                &col.name,
                &ignore,
                &mut on_file,
            )?;
            report.indexed += r.indexed;
            report.updated += r.updated;
            report.unchanged += r.unchanged;
            report.removed += r.removed;
            report.skipped += r.skipped_files.len();
        }
        report.needs_embedding = store::status::hashes_needing_embedding(&conn, None, &model)?;
        Ok(report)
    }

    /// Fetch a single document by `qmd://` URI, filesystem path,
    /// `#docid`/`docid`, or partial path — upstream `findDocument`
    /// (store.ts:4851-4957) via the SDK `get` (index.ts:483).
    ///
    /// A trailing `:N` is stripped before lookup, as upstream.
    ///
    /// # Errors
    /// [`Error::DocumentNotFound`] (with `similar_files` suggestions) or
    /// [`Error::ExcludedByIgnore`] when the path matches an ignore rule;
    /// [`Error::Db`] on query failure.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "the connection guard must live until the lookup completes"
    )]
    pub fn get(&self, path_or_docid: &str, opts: GetOptions) -> Result<Document> {
        let conn = lock(&self.conn);
        store::document::find_document(&conn, &self.env, path_or_docid, opts.include_body)
    }

    /// Body of a document, optionally sliced to a 1-based line window —
    /// upstream SDK `getDocumentBody` (index.ts:484-488, store.ts:4963).
    /// Returns `Ok(None)` when the lookup fails, like upstream's `null`.
    ///
    /// # Errors
    /// [`Error::Db`] on query failure; lookup misses return `Ok(None)`.
    pub fn document_body(&self, path_or_docid: &str, lines: LineRange) -> Result<Option<String>> {
        let conn = lock(&self.conn);
        store::document::get_document_body(
            &conn,
            &self.env,
            path_or_docid,
            lines.from_line,
            lines.max_lines,
        )
    }

    /// Fetch multiple documents by comma-separated names or a glob
    /// pattern — upstream `findDocuments` / `multiGet`
    /// (store.ts:5135-5233, index.ts:489). Per-name failures land in
    /// [`MultiGet::errors`]; oversized bodies become
    /// [`crate::document::MultiGetEntry::Skipped`].
    ///
    /// # Errors
    /// [`Error::Db`] on query failure, [`Error::InvalidInput`] on a
    /// malformed glob pattern.
    pub fn multi_get(&self, pattern: &str, opts: &MultiGetOptions) -> Result<MultiGet> {
        let conn = lock(&self.conn);
        store::document::find_documents(&conn, pattern, opts.include_body, opts.max_bytes)
    }

    /// Documents of one collection under an optional path prefix — the
    /// `qmd ls` listing (cli/qmd.ts:1712-1738), ordered by path.
    ///
    /// # Errors
    /// [`Error::CollectionNotFound`] when `collection` is unknown;
    /// [`Error::Db`] on query failure.
    pub fn list_documents(
        &self,
        collection: &str,
        prefix: Option<&str>,
    ) -> Result<Vec<DocumentEntry>> {
        let conn = lock(&self.conn);
        store::document::list_documents(&conn, collection, prefix)
    }

    /// Resolve a `qmd://` URI to an absolute filesystem path inside the
    /// collection — upstream `resolveVirtualPath` (store.ts:791-801).
    /// `Ok(None)` for non-virtual input, unknown collections, or paths
    /// escaping the collection root.
    ///
    /// # Errors
    /// [`Error::Db`] on query failure.
    pub fn resolve_virtual_path(&self, uri: &str) -> Result<Option<PathBuf>> {
        let conn = lock(&self.conn);
        store::document::resolve_virtual_path(&conn, uri)
    }

    /// Index status — upstream `getStatus` (store.ts:5239-5284) plus the
    /// CLI `status` aggregates and `reindex_required`.
    ///
    /// # Errors
    /// Propagates config load and database errors.
    pub fn status(&self) -> Result<Status> {
        let config = self.config()?;
        let model = self.embed_model(&config);
        let conn = lock(&self.conn);
        let row = store::status::get_status(&conn, &model)?;
        drop(conn);
        Ok(Status {
            total_documents: row.total_documents,
            needs_embedding: row.needs_embedding,
            has_vector_index: row.has_vector_index,
            vector_count: row.vector_count,
            orphaned_vectors: row.orphaned_vectors,
            latest_modified: row.latest_modified,
            reindex_required: row.reindex_required,
            collections: row
                .collections
                .into_iter()
                .map(|c| StatusCollection {
                    name: c.name,
                    path: c.path,
                    pattern: c.pattern,
                    documents: c.documents,
                    last_updated: c.last_updated,
                })
                .collect(),
        })
    }

    /// Embedding staleness summary — upstream `getIndexHealth`
    /// (store.ts:2656-2668).
    ///
    /// # Errors
    /// Propagates config load and database errors.
    pub fn index_health(&self) -> Result<IndexHealth> {
        let config = self.config()?;
        let model = self.embed_model(&config);
        let conn = lock(&self.conn);
        let row = store::status::get_index_health(&conn, &model)?;
        drop(conn);
        Ok(IndexHealth {
            needs_embedding: row.needs_embedding,
            total_docs: row.total_docs,
            days_stale: row.days_stale,
        })
    }

    /// `qmd cleanup` operations — upstream `Maintenance`
    /// (maintenance.ts:22-76). The handle borrows `self`.
    #[must_use]
    pub const fn maintenance(&self) -> Maintenance<'_> {
        Maintenance { qmd: self }
    }

    /// Full-text keyword search: upstream `searchLex` → `searchFTS`
    /// (index.ts:474, store.ts:4081-4180). Returns BM25-ranked hits;
    /// an empty result is produced for queries with no positive terms.
    ///
    /// # Errors
    /// Propagates database errors (including malformed FTS5 syntax).
    pub fn search_lex(&self, query: &str, opts: &LexOptions) -> Result<Vec<SearchResult>> {
        let conn = lock(&self.conn);
        store::search::search_fts(
            &conn,
            query,
            opts.limit.unwrap_or(20),
            opts.collections.clone(),
        )
    }

    #[allow(
        clippy::significant_drop_tightening,
        reason = "the connection guard must live until the query completes"
    )]
    fn load_config(&self, source: &ConfigSource) -> Result<Config> {
        match source {
            ConfigSource::File(path) => crate::config::load(path),
            ConfigSource::Inline(cfg) => Ok((**cfg).clone()),
            ConfigSource::DbOnly => {
                // `editor_uri` and unknown top-level keys are only stored
                // in the YAML file, so a DB-only reopen cannot recover
                // them — same limitation as upstream.
                let conn = lock(&self.conn);
                let mut config = Config {
                    global_context: store::get_store_global_context(&conn)?,
                    ..Config::default()
                };
                for row in store::get_store_collections(&conn)? {
                    config.collections.insert(
                        row.name,
                        Collection {
                            path: row.path,
                            pattern: Some(row.pattern),
                            ignore: row.ignore,
                            context: row.context,
                            update: row.update,
                            include_by_default: (!row.include_by_default).then_some(false),
                            ..Collection::default()
                        },
                    );
                }
                Ok(config)
            }
        }
    }

    /// Apply `f` to the current config. `Ok(None)` aborts without
    /// persisting; `Ok(Some(v))` persists the config, syncs the database
    /// mirror, and returns the value plus the sync outcome.
    fn mutate_config<T>(
        &self,
        f: impl FnOnce(&mut Config) -> Result<Option<T>>,
    ) -> Result<Option<(T, store::SyncOutcome)>> {
        self.mutate_config_then(f, |_| Ok(()))
    }

    /// [`Qmd::mutate_config`] with an extra step that runs after the
    /// config is persisted but before the sync pass (used by rename to
    /// move `documents` rows first).
    ///
    /// Failure semantics: if persistence succeeded but `between` or the
    /// sync fails, the config file (or inline state) is already updated
    /// while the DB mirror is stale; the next open re-syncs it.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "guards must live until the sync transaction completes"
    )]
    fn mutate_config_then<T>(
        &self,
        f: impl FnOnce(&mut Config) -> Result<Option<T>>,
        between: impl FnOnce(&Connection) -> Result<()>,
    ) -> Result<Option<(T, store::SyncOutcome)>> {
        let mut guard = lock(&self.config);
        let mut config = self.load_config(&guard)?;
        let Some(value) = f(&mut config)? else {
            return Ok(None);
        };
        match &mut *guard {
            ConfigSource::File(path) => crate::config::save(path, &config)?,
            ConfigSource::Inline(slot) => (**slot).clone_from(&config),
            ConfigSource::DbOnly => {}
        }
        let mut conn = lock(&self.conn);
        between(&conn)?;
        let outcome = store::sync_config_to_db(&mut conn, &config, true)?;
        Ok(Some((value, outcome)))
    }
}
