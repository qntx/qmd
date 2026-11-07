//! Path resolution helpers.
//!
//! Mirrors the upstream helpers from `src/utils/paths.ts` and `src/db.ts`,
//! with environment values supplied through [`Environment`] instead of
//! `process.env`.

use std::path::{Path, PathBuf};

use crate::env::Environment;

/// Upstream busy-timeout default in milliseconds (`db.ts:20`).
pub const DEFAULT_BUSY_TIMEOUT_MS: u64 = 120_000;

const LOCAL_CONFIG_DIR: &str = ".meme";
/// Upstream checks `index.yaml` first, then `index.yml`
/// (`collections.ts:141-156`).
const LOCAL_CONFIG_FILES: [&str; 2] = ["index.yaml", "index.yml"];
const LOCAL_DB_FILE: &str = "index.sqlite";

/// Upstream `memeHomedir()`: `HOME` -> `USERPROFILE` -> OS home -> `/tmp`.
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

/// Node `fs.realpathSync` semantics for canonicalization.
///
/// `std::fs::canonicalize` returns verbatim `\\?\`-prefixed paths on
/// Windows while Node's realpath does not — those prefixes would end
/// up written into collection YAML and break comparisons against
/// non-canonical paths (`dunce::canonicalize` only strips the prefix
/// when the path is representable without it, so long/UNC paths keep
/// it). Identical to `std::fs::canonicalize` off Windows.
///
/// # Errors
/// Same as `std::fs::canonicalize`.
pub fn canonicalize(path: impl AsRef<Path>) -> std::io::Result<PathBuf> {
    dunce::canonicalize(path)
}

/// Upstream `getConfigDir()` chain: `MEME_CONFIG_DIR` -> `XDG_CONFIG_HOME/meme` ->
/// `~/.config/meme`.
#[must_use]
pub fn config_dir(env: &Environment) -> PathBuf {
    if let Some(dir) = &env.meme_config_dir {
        return dir.clone();
    }
    if let Some(xdg) = &env.xdg_config_home {
        return xdg.join("meme");
    }
    home_dir(env).join(".config").join("meme")
}

/// Upstream `getConfigFilePath` (collections.ts:131-133):
/// `<config-dir>/<index-name>.yml` — `index.yml` for the default index.
#[must_use]
pub fn config_file(env: &Environment, index_name: &str) -> PathBuf {
    config_dir(env).join(format!("{index_name}.yml"))
}

/// Upstream `getCacheDir()`: `XDG_CACHE_HOME/qmd` -> `~/.cache/qmd`.
#[must_use]
pub fn cache_dir(env: &Environment) -> PathBuf {
    if let Some(xdg) = &env.xdg_cache_home {
        return xdg.join("meme");
    }
    home_dir(env).join(".cache").join("meme")
}

/// Default index database path: `$INDEX_PATH` or
/// `<cache-dir>/<index>.sqlite`.
///
/// The `meme` cache namespace already keeps this product's indexes
/// separate from upstream `qmd` files. Like upstream `getDefaultDbPath`
/// (store.ts:660-661), the cache directory is created best-effort so
/// opening a named index never fails on a missing parent.
#[must_use]
pub fn default_db_path(env: &Environment, index_name: &str) -> PathBuf {
    if let Some(path) = &env.index_path {
        return path.clone();
    }
    let dir = cache_dir(env);
    // Upstream swallows mkdir errors; a failure surfaces at DB open.
    drop(std::fs::create_dir_all(&dir));
    dir.join(format!("{index_name}.sqlite"))
}

/// Database path for a project-local config: `.meme/index.sqlite` next
/// to `.meme/index.yaml` / `.meme/index.yml`.
#[must_use]
pub fn local_db_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or_else(|| Path::new(LOCAL_CONFIG_DIR))
        .join(LOCAL_DB_FILE)
}

/// Search upward from `start` for `.meme/index.yaml` or `.meme/index.yml`
/// (yaml wins when both exist), like upstream `findLocalConfigPath()`.
#[must_use]
pub fn find_local_config(start: &Path) -> Option<PathBuf> {
    let mut dir = start.to_path_buf();
    loop {
        let meme_dir = dir.join(LOCAL_CONFIG_DIR);
        for name in LOCAL_CONFIG_FILES {
            let candidate = meme_dir.join(name);
            // Upstream `existsSync` (collections.ts:141-156): any existing
            // path counts — a `.meme/index.yaml` directory still wins and
            // fails later at read time instead of yielding a parent config.
            if candidate.try_exists().unwrap_or(false) {
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

/// Lexical `path.resolve(dir, rel)`: join and normalize without touching
/// the filesystem. Absolute `rel` wins over `dir`, like Node's resolve.
///
/// Public for the CLI crate — upstream's `resolve(getPwd(), arg)`
/// path handling.
#[must_use]
pub fn resolve_lexical(dir: &Path, rel: &str) -> PathBuf {
    lexical_normalize(&dir.join(rel))
}

/// [`lexical_normalize`] on a string path — for call sites that work in
/// upstream's stringly-typed path space (e.g. `trust.ts`).
pub(crate) fn lexical_normalize_str(path: &str) -> String {
    lexical_normalize(Path::new(path))
        .to_string_lossy()
        .into_owned()
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
/// `MEME_SQLITE_BUSY_TIMEOUT` is parsed as a number; empty, non-numeric or
/// negative values fall back to [`DEFAULT_BUSY_TIMEOUT_MS`]; `0` restores
/// fail-fast behavior.
#[must_use]
pub fn busy_timeout_ms(env: &Environment) -> u64 {
    let Some(raw) = &env.meme_sqlite_busy_timeout else {
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
    let real_dir = canonicalize(dir).unwrap_or_else(|_| lexical_normalize(dir));
    canonicalize(target).map_or_else(
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

/// `path`'s components joined with `/` — the separator-free relative
/// form stored for indexed files and matched against globs.
pub(crate) fn to_slash_path(path: &Path) -> String {
    path.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// `path` relative to `root`, `/`-joined — compared by path components,
/// so Windows `D:\proj\docs\a.md` under `D:\proj\docs` maps to `a.md`
/// (and `D:/proj/docs/a.md` too). `None` when `path` is not under
/// `root`, `Some("")` when both name the same directory.
///
/// Upstream compares `startsWith(root + "/")` strings, which never
/// matches Windows separators (deviation 86).
pub(crate) fn relative_slash_path(root: &Path, path: &Path) -> Option<String> {
    path.strip_prefix(root).ok().map(to_slash_path)
}

#[cfg(test)]
mod slash_path_tests {
    use std::path::Path;

    use super::relative_slash_path;

    #[test]
    #[cfg(unix)]
    fn relative_slash_path_unix() {
        assert_eq!(
            relative_slash_path(Path::new("/a/docs"), Path::new("/a/docs/x/y.md")),
            Some("x/y.md".to_owned())
        );
        assert_eq!(
            relative_slash_path(Path::new("/a/docs"), Path::new("/a/docs")),
            Some(String::new())
        );
        assert_eq!(
            relative_slash_path(Path::new("/a/docs"), Path::new("/a/docs2/x.md")),
            None
        );
        assert_eq!(
            relative_slash_path(Path::new("/a/docs/sub"), Path::new("/a/docs/x.md")),
            None
        );
    }

    #[test]
    #[cfg(windows)]
    fn relative_slash_path_windows() {
        assert_eq!(
            relative_slash_path(
                Path::new("D:\\proj\\docs"),
                Path::new("D:\\proj\\docs\\sub\\a.md")
            ),
            Some("sub/a.md".to_owned())
        );
        assert_eq!(
            relative_slash_path(Path::new("D:\\proj\\docs"), Path::new("D:/proj/docs/a.md")),
            Some("a.md".to_owned())
        );
        assert_eq!(
            relative_slash_path(
                Path::new("D:\\proj\\docs"),
                Path::new("D:\\proj\\docs2\\a.md")
            ),
            None
        );
    }
}
