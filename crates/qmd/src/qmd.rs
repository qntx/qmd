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
use crate::env::Environment;
use crate::error::{Error, Result};
use crate::store;

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

/// Open handle to a qmd index.
///
/// `Send + Sync`: the connection and config are each guarded by a
/// [`Mutex`]. Dropping the handle closes the database; use
/// [`Qmd::close`] to observe shutdown errors.
pub struct Qmd {
    conn: Mutex<Connection>,
    config: Mutex<ConfigSource>,
    db_path: PathBuf,
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
