//! Path resolution helpers.
//!
//! Mirrors the upstream helpers from `src/utils/paths.ts` and `src/db.ts`,
//! with environment values supplied through [`Environment`] instead of
//! `process.env`.

use std::path::{Path, PathBuf};

use crate::env::Environment;

/// Upstream busy-timeout default in milliseconds (`db.ts:20`).
pub const DEFAULT_BUSY_TIMEOUT_MS: u64 = 120_000;

const LOCAL_CONFIG_DIR: &str = ".qmd";
/// Upstream checks `index.yaml` first, then `index.yml`
/// (`collections.ts:141-156`).
const LOCAL_CONFIG_FILES: [&str; 2] = ["index.yaml", "index.yml"];
const LOCAL_DB_FILE: &str = "index-rs.sqlite";

/// Upstream `qmdHomedir()`: `HOME` -> `USERPROFILE` -> OS home -> `/tmp`.
#[must_use]
pub fn home_dir(env: &Environment) -> PathBuf {
    env.home
        .clone()
        .or_else(|| env.userprofile.clone())
        .or_else(std::env::home_dir)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

/// `PWD`, falling back to the process working directory.
///
/// Upstream uses `process.env.PWD ?? process.cwd()` for `--index` resolution.
#[must_use]
pub fn pwd(env: &Environment) -> PathBuf {
    env.pwd
        .clone()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("/"))
}

/// Upstream `getConfigDir()`: `QMD_CONFIG_DIR` -> `XDG_CONFIG_HOME/qmd` ->
/// `~/.config/qmd`.
#[must_use]
pub fn config_dir(env: &Environment) -> PathBuf {
    if let Some(dir) = &env.qmd_config_dir {
        return dir.clone();
    }
    if let Some(xdg) = &env.xdg_config_home {
        return xdg.join("qmd");
    }
    home_dir(env).join(".config").join("qmd")
}

/// Upstream `getConfigFile(indexName)`: `<config-dir>/index-<name>.yml`.
#[must_use]
pub fn config_file(env: &Environment, index_name: &str) -> PathBuf {
    config_dir(env).join(format!("index-{index_name}.yml"))
}

/// Upstream `getCacheDir()`: `XDG_CACHE_HOME/qmd` -> `~/.cache/qmd`.
#[must_use]
pub fn cache_dir(env: &Environment) -> PathBuf {
    if let Some(xdg) = &env.xdg_cache_home {
        return xdg.join("qmd");
    }
    home_dir(env).join(".cache").join("qmd")
}

/// Default index database path: `$INDEX_PATH` or
/// `<cache-dir>/<index>-rs.sqlite`.
///
/// The `-rs` suffix keeps Rust indexes separate from upstream `.sqlite`
/// files so the two never clobber each other.
#[must_use]
pub fn default_db_path(env: &Environment, index_name: &str) -> PathBuf {
    if let Some(path) = &env.index_path {
        return path.clone();
    }
    cache_dir(env).join(format!("{index_name}-rs.sqlite"))
}

/// Database path for a project-local config: `.qmd/index-rs.sqlite` next
/// to `.qmd/index.yaml` / `.qmd/index.yml`.
#[must_use]
pub fn local_db_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or_else(|| Path::new(LOCAL_CONFIG_DIR))
        .join(LOCAL_DB_FILE)
}

/// Search upward from `start` for `.qmd/index.yaml` or `.qmd/index.yml`
/// (yaml wins when both exist), like upstream `findLocalConfigPath()`.
#[must_use]
pub fn find_local_config(start: &Path) -> Option<PathBuf> {
    let mut dir = start.to_path_buf();
    loop {
        let qmd_dir = dir.join(LOCAL_CONFIG_DIR);
        for name in LOCAL_CONFIG_FILES {
            let candidate = qmd_dir.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Normalize an `--index` argument into an index name, like upstream
/// `normalizeIndexName()`.
///
/// Values containing `/` or `\`, or Windows-style absolute paths
/// (`C:\...`, `\\server\...`), are treated as filesystem paths: relative
/// paths are resolved against `cwd`, then every run of `:` `\` `/` becomes
/// `_` and leading underscores are stripped. Everything else is returned
/// unchanged.
#[must_use]
pub fn normalize_index_name(name: &str, cwd: &Path) -> String {
    let bytes = name.as_bytes();
    let is_windows_absolute = name.starts_with('\\')
        || matches!(
            (bytes.first(), bytes.get(1), bytes.get(2)),
            (Some(c), Some(b':'), Some(b'\\' | b'/'))
                if c.is_ascii_alphabetic()
        );
    if !is_windows_absolute && !name.contains('/') && !name.contains('\\') {
        return name.to_owned();
    }
    let absolute = if is_windows_absolute {
        name.to_owned()
    } else {
        lexical_normalize(&cwd.join(name))
            .to_string_lossy()
            .into_owned()
    };
    let mut normalized = String::with_capacity(absolute.len());
    let mut last_was_sep = false;
    for ch in absolute.chars() {
        if matches!(ch, ':' | '/' | '\\') {
            if !last_was_sep {
                normalized.push('_');
            }
            last_was_sep = true;
        } else {
            normalized.push(ch);
            last_was_sep = false;
        }
    }
    normalized.trim_start_matches('_').to_owned()
}

/// Lexical path normalization, like `path.resolve`: drops `.`, resolves
/// `..` by popping, and collapses duplicate separators — without touching
/// the filesystem (so it works on non-existent paths, unlike
/// [`Path::canonicalize`]).
fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Busy timeout in milliseconds, parsed like upstream `db.ts:15-26`.
///
/// `QMD_SQLITE_BUSY_TIMEOUT` is parsed as a number; empty, non-numeric or
/// negative values fall back to [`DEFAULT_BUSY_TIMEOUT_MS`]; `0` restores
/// fail-fast behavior.
#[must_use]
pub fn busy_timeout_ms(env: &Environment) -> u64 {
    let Some(raw) = &env.qmd_sqlite_busy_timeout else {
        return DEFAULT_BUSY_TIMEOUT_MS;
    };
    if raw.is_empty() {
        return DEFAULT_BUSY_TIMEOUT_MS;
    }
    match raw.parse::<f64>() {
        Ok(ms) if ms.is_finite() && ms >= 0.0 => {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "checked finite and non-negative above; \
                          truncation to u64 mirrors upstream floor"
            )]
            {
                ms.floor() as u64
            }
        }
        Ok(_) | Err(_) => DEFAULT_BUSY_TIMEOUT_MS,
    }
}

/// Expand a leading `~` or `~/` against [`home_dir`], like upstream
/// `expandHome()`.
#[must_use]
pub fn expand_home(path: &str, env: &Environment) -> PathBuf {
    if path == "~" {
        return home_dir(env);
    }
    if let Some(rest) = path.strip_prefix("~/") {
        return home_dir(env).join(rest);
    }
    PathBuf::from(path)
}

/// Upstream `isPathInsideDir` (store.ts:681-691): `true` when `target` is
/// `dir` or a descendant after resolving symlinks. When `target` cannot
/// be canonicalized (unreadable or dangling), upstream falls back to a
/// lexical comparison so a mode-0 file inside the collection is not
/// treated as an escape.
pub(crate) fn is_path_inside_dir(dir: &Path, target: &Path) -> bool {
    let real_dir = std::fs::canonicalize(dir).unwrap_or_else(|_| lexical_normalize(dir));
    std::fs::canonicalize(target).map_or_else(
        // Mirror upstream's catch branch: the dir side is *not*
        // canonicalized here — `resolve(dir)` is lexical only.
        |_| {
            let t = lexical_normalize(target);
            let d = lexical_normalize(dir);
            t == d || t.starts_with(&d)
        },
        |real| real == real_dir || real.starts_with(&real_dir),
    )
}
