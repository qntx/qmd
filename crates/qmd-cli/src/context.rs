//! Index and config-source resolution, plus store opening.
//!
//! Mirrors the tail of upstream `parseCLI` (qmd.ts:3158-3174) and
//! `getStore` (qmd.ts:154-177):
//!
//! - `--index <name>` → index name normalized like `setIndexName` /
//!   `setConfigIndexName` ([`qmd::paths::normalize_index_name`]), config
//!   `<config-dir>/<name>.yml`, database
//!   [`qmd::paths::default_db_path`] (`$INDEX_PATH` always wins).
//! - Otherwise a project-local `.qmd/index.y{a,}ml` found by
//!   [`qmd::paths::find_local_config`] keeps both the config and the
//!   `.qmd/index-rs.sqlite` database inside the project.
//! - Otherwise the global `index` index.
//!
//! The handle opens lazily — only after the command's own validation —
//! so usage errors never touch the database. Opening runs
//! [`Qmd::ensure_models_configured`] like upstream's `getStore`
//! (qmd.ts:159): the resolved model triple is pinned into the config
//! file, and any failure is swallowed upstream-style.

use std::path::{Path, PathBuf};

use qmd::{Environment, Qmd, Result, paths};

/// Resolved index context — index name, config source and db path.
/// `?index=` switching (upstream getDocument, qmd.ts:1284-1291) mutates
/// [`CliContext::set_index`], used by `get`/`multi-get` in stage B.
pub(crate) struct CliContext {
    /// `process.cwd()` — `renderFullPath`'s anchor and the
    /// `--index`/`?index=` name-normalization base.
    cwd: PathBuf,
    /// Normalized active index name — feeds `?index=` in `qmd://` links
    /// (upstream `buildVirtualPath` skips it for `"index"`).
    index_name: String,
    /// Database path (`$INDEX_PATH` already folded in).
    db_path: PathBuf,
    /// Explicit config file — `Some` only when the source is a
    /// project-local `.qmd/index.y{a,}ml` or a path the caller pinned.
    /// `None` → derived `<config-dir>/<index-name>.yml` at open time.
    config_file: Option<PathBuf>,
}

impl CliContext {
    /// Resolve like upstream's `parseCLI` tail (qmd.ts:3158-3174).
    pub(crate) fn resolve(env: &Environment, cwd: &Path, index: Option<&str>) -> Self {
        if let Some(name) = index {
            // setIndexName + setConfigIndexName + setConfigSource()
            let normalized = paths::normalize_index_name(name, cwd);
            let index_name = if normalized.is_empty() {
                "index".to_owned()
            } else {
                normalized
            };
            return Self {
                cwd: cwd.to_path_buf(),
                index_name: index_name.clone(),
                db_path: paths::default_db_path(env, &index_name),
                config_file: None,
            };
        }
        // A project-local config overrides the global index.
        if let Some(local) = paths::find_local_config(cwd) {
            return Self {
                cwd: cwd.to_path_buf(),
                index_name: "index".to_owned(),
                db_path: paths::local_db_path(&local),
                config_file: Some(local),
            };
        }
        Self {
            cwd: cwd.to_path_buf(),
            index_name: "index".to_owned(),
            db_path: paths::default_db_path(env, "index"),
            config_file: None,
        }
    }

    /// `process.cwd()` — used by `--full-path` rendering.
    pub(crate) fn cwd(&self) -> &Path {
        &self.cwd
    }

    /// Active index name (upstream `getActiveIndexName`).
    pub(crate) fn index_name(&self) -> &str {
        &self.index_name
    }

    /// Config file the store loads (upstream `getConfigFilePath` when
    /// the source is the default file, the explicit path otherwise).
    pub(crate) fn config_path(&self, env: &Environment) -> PathBuf {
        self.config_file
            .clone()
            .unwrap_or_else(|| paths::config_file(env, &self.index_name))
    }

    /// Database path (upstream `getDbPath`).
    pub(crate) fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Switch the active index — upstream `setIndexName` +
    /// `setConfigIndexName` (qmd.ts:1284-1291): the db path moves to
    /// the new name and a derived config path follows it, while an
    /// explicit (project-local) config file is left alone.
    pub(crate) fn set_index(&mut self, env: &Environment, name: &str) {
        let normalized = paths::normalize_index_name(name, &self.cwd);
        self.index_name = if normalized.is_empty() {
            "index".to_owned()
        } else {
            normalized
        };
        self.db_path = paths::default_db_path(env, &self.index_name);
    }

    /// Open the store — upstream `getStore` (qmd.ts:154-177): build the
    /// `Qmd` handle, then pin resolved models into the config; its
    /// failure is ignored like upstream's `catch`.
    ///
    /// # Errors
    /// [`qmd::Error`] from `Qmd::builder(...).build()`.
    pub(crate) fn open_store(&self, env: &Environment) -> Result<Qmd> {
        let qmd = Qmd::builder(&self.db_path)
            .environment(env.clone())
            .config_file(self.config_path(env))
            .build()?;
        let _ = qmd.ensure_models_configured();
        Ok(qmd)
    }
}
