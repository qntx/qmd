//! Approval gate for project-local `.qmd` config — port of upstream
//! `src/trust.ts`.
//!
//! A project-local `.qmd/index.yml` arrives with a `git clone`, and
//! `findLocalConfigPath` adopts it automatically for any command run
//! inside the tree. Three fields in that file can reach outside the
//! project:
//!
//! - `update:` — a shell command run by `qmd update` (upstream #886)
//! - `collections.*.path` — any directory the process can read (upstream
//!   #889)
//! - `models.embed` / `models.rerank` / `models.generate` — any `hf:` repo
//!   or local GGUF path (upstream #889)
//!
//! Global `~/.config/qmd` is never gated. In-project collection paths and
//! the built-in default model URIs are also allowed without approval:
//! those are what a local config is for. Approvals are per config file
//! and per gated set, recorded in `<config dir>/trusted.json`. Editing a
//! hook, pointing a collection outside the project, or changing a custom
//! model URI changes the digest and re-arms the gate.
//!
//! `trusted.json` is byte-compatible with upstream — files written by
//! either implementation are honored by the other (P3-D2). Deliberate
//! deviations:
//!
//! - `hookDigest` and `decideHookGate` are not ported: upstream has no
//!   caller for them (P3-D3), their tests are recorded N/A.
//! - [`list_trusted`] narrows upstream's `{path, ...record}` spread to a
//!   typed view: object records expose `hooks`/`trustedAt` (absent or
//!   wrong-typed fields become `None`); non-object records yield an
//!   entry with both `None` — upstream's spread of primitives produces
//!   exotic shapes no caller reads.
//! - Environment comes through [`Environment`] instead of `process.env`,
//!   and path helpers take `&str` — upstream's `resolve()`/`join()` are
//!   string operations on POSIX separators. Upstream `options.env` feeds
//!   only `isTruthyEnv` while path helpers read real process state; our
//!   single [`Environment`] supplies both (`from_process` is consistent).
//! - Where upstream `resolve()` uses `process.cwd()`, `node_resolve`
//!   uses [`pwd`] (`$PWD` falling back to `current_dir`). `$PWD` can
//!   differ from `getcwd` on symlinked directories, which only matters
//!   for *relative* config paths — real flows always pass absolute ones
//!   (`findLocalConfigPath`/`getConfigFile`), and a stale `PWD` fails
//!   closed (paths drift out of the root → gated).

use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::env::Environment;
use crate::error::{Error, Result};
use crate::paths::{config_dir, home_dir, pwd, resolve_lexical};

/// A collection's pre-update hook, as it will be executed — upstream
/// `UpdateHook` (trust.ts:28-31).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateHook {
    /// Collection name.
    pub collection: String,
    /// Shell command run via `bash -c` before re-indexing.
    pub command: String,
}

/// A declared collection path — upstream `CollectionPath`
/// (trust.ts:33-36).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionPath {
    /// Collection name.
    pub collection: String,
    /// Raw `path` value from the config file.
    pub path: String,
}

/// Model slot name — upstream `ModelSlot` (trust.ts:38).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSlot {
    /// `embed`
    Embed,
    /// `rerank`
    Rerank,
    /// `generate`
    Generate,
}

impl ModelSlot {
    /// Fixed slot order upstream iterates (trust.ts:140).
    const ALL: [Self; 3] = [Self::Embed, Self::Rerank, Self::Generate];

    /// Upstream string form.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::Rerank => "rerank",
            Self::Generate => "generate",
        }
    }
}

/// Model URIs declared by a config — upstream `ModelsSnapshot`
/// (trust.ts:40-44). `None` slots are equivalent to the built-in
/// defaults for gating.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelsSnapshot {
    /// `models.embed`
    pub embed: Option<String>,
    /// `models.rerank`
    pub rerank: Option<String>,
    /// `models.generate`
    pub generate: Option<String>,
}

impl ModelsSnapshot {
    const fn get(&self, slot: ModelSlot) -> Option<&String> {
        match slot {
            ModelSlot::Embed => self.embed.as_ref(),
            ModelSlot::Rerank => self.rerank.as_ref(),
            ModelSlot::Generate => self.generate.as_ref(),
        }
    }
}

/// The gated surface of a project-local config — upstream
/// `SensitiveSnapshot` (trust.ts:46-50).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SensitiveSnapshot {
    /// `update:` hooks declared per collection.
    pub hooks: Vec<UpdateHook>,
    /// `path` values declared per collection.
    pub paths: Vec<CollectionPath>,
    /// `models.*` URIs.
    pub models: ModelsSnapshot,
}

/// A model URI that needs approval — one element of
/// [`GatedItems::models`] (upstream `{slot, uri}`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct GatedModel {
    /// Which slot the URI was declared for.
    pub slot: ModelSlot,
    /// The non-default URI.
    pub uri: String,
}

/// The subset of a snapshot that actually needs approval — upstream
/// `GatedItems` (trust.ts:52-56).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct GatedItems {
    /// All declared hooks.
    pub hooks: Vec<UpdateHook>,
    /// Paths that resolve outside the project.
    pub paths: Vec<CollectionPath>,
    /// Non-default model URIs, in slot order.
    pub models: Vec<GatedModel>,
}

/// The built-in default model URIs — upstream `BuiltinModels`
/// (`Required<ModelsSnapshot>`, trust.ts:58).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltinModels {
    /// Default `embed` URI.
    pub embed: String,
    /// Default `rerank` URI.
    pub rerank: String,
    /// Default `generate` URI.
    pub generate: String,
}

impl BuiltinModels {
    const fn get(&self, slot: ModelSlot) -> &String {
        match slot {
            ModelSlot::Embed => &self.embed,
            ModelSlot::Rerank => &self.rerank,
            ModelSlot::Generate => &self.generate,
        }
    }
}

/// The trust database on disk — `<config dir>/trusted.json`, keyed on the
/// resolved config path.
///
/// Record shape is `{ "hooks": "<digest>", "trustedAt": "<ISO>" }` — the
/// field is named `hooks` but stores the *sensitive* digest (upstream quirk,
/// trust.ts:61-67). Raw JSON map so records written by upstream load
/// unchanged (P3-D2).
pub type TrustStore = serde_json::Map<String, serde_json::Value>;

/// Path of the trust database — upstream `getTrustFilePath`
/// (trust.ts:70-72).
#[must_use]
pub fn trust_file_path(env: &Environment) -> PathBuf {
    config_dir(env).join("trusted.json")
}

/// Node `path.resolve(path)` — lexical only, relative paths resolve
/// against the process working directory (our `pwd(env)`).
fn node_resolve(path: &str, env: &Environment) -> String {
    resolve_lexical(&pwd(env), path)
        .to_string_lossy()
        .into_owned()
}

/// `resolve(dir, rel)` where `dir` is already a string.
fn node_resolve_path(dir: &str, rel: &str) -> String {
    resolve_lexical(Path::new(dir), rel)
        .to_string_lossy()
        .into_owned()
}

/// Node `path.join(a, b)` — string concatenation plus normalization;
/// unlike `resolve`, an absolute `b` does not win.
fn node_join(dir: &str, rel: &str) -> String {
    crate::paths::lexical_normalize_str(&format!("{dir}/{rel}"))
}

/// Whether a config path is project-local — upstream `isLocalConfigPath`
/// (trust.ts:82-85).
///
/// The parent directory name is `.qmd`. `findLocalConfigPath`/`qmd init`
/// write `.qmd/index.y{a,}ml` while the global config lives in a directory
/// named `qmd`, so the parent name separates the two cases.
#[must_use]
pub fn is_local_config_path(config_path: &str, env: &Environment) -> bool {
    if config_path.is_empty() || config_path == "<inline>" {
        return false;
    }
    Path::new(&node_resolve(config_path, env))
        .parent()
        .and_then(Path::file_name)
        .is_some_and(|n| n == ".qmd")
}

/// Directory containing the `.qmd` folder of a project-local config —
/// upstream `projectRootFromConfig` (trust.ts:88-90).
#[must_use]
pub fn project_root_from_config(config_path: &str, env: &Environment) -> String {
    let resolved = PathBuf::from(node_resolve(config_path, env));
    resolved
        .parent()
        .map_or(resolved.as_path(), |d| d.parent().unwrap_or(d))
        .to_string_lossy()
        .into_owned()
}

/// Expand a leading `~` to the user home — upstream `expandUserPath`
/// (trust.ts:93-99). `~`, `~/` and `~\` expand; `~user` stays literal.
/// Distinct from `expandHome` — this one also accepts the Windows
/// separator.
#[must_use]
pub fn expand_user_path(raw: &str, env: &Environment) -> String {
    if raw == "~" {
        return home_dir(env).to_string_lossy().into_owned();
    }
    if raw.starts_with("~/") || raw.starts_with("~\\") {
        return node_join(&home_dir(env).to_string_lossy(), &raw[2..]);
    }
    raw.to_owned()
}

/// Resolve a collection `path` from a project-local config — upstream
/// `resolveConfigCollectionPath` (trust.ts:106-111).
///
/// `~` expands, then relative paths resolve against the project root (the
/// directory containing `.qmd`), not the process cwd.
#[must_use]
pub fn resolve_config_collection_path(
    raw_path: &str,
    config_path: &str,
    env: &Environment,
) -> String {
    // `String(rawPath ?? "").trim()` — JS trim, then `~` expansion.
    let expanded = expand_user_path(crate::chunk::js_trim(raw_path), env);
    if expanded.is_empty() {
        return project_root_from_config(config_path, env);
    }
    if Path::new(&expanded).is_absolute() {
        return node_resolve(&expanded, env);
    }
    node_resolve_path(&project_root_from_config(config_path, env), &expanded)
}

/// `realpathSync(path)` with `resolve(path)` fallback — upstream
/// `realOrResolve` (trust.ts:113-119).
fn real_or_resolve(path: &str, env: &Environment) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| resolve_lexical(&pwd(env), path))
}

/// Node `path.relative(from, to)` as path segments — both inputs are
/// already absolute and normalized. `[]` when the paths are equal.
fn relative_segments(from: &Path, to: &Path) -> Vec<String> {
    let from_c: Vec<_> = from.components().collect();
    let to_c: Vec<_> = to.components().collect();
    let common = from_c.iter().zip(&to_c).take_while(|(a, b)| a == b).count();
    let mut segs = vec!["..".to_owned(); from_c.len() - common];
    segs.extend(
        to_c.iter()
            .skip(common)
            .map(|c| c.as_os_str().to_string_lossy().into_owned()),
    );
    segs
}

/// True when `rawPath` from a project-local config stays inside the project
/// — upstream `isCollectionPathInsideProject` (trust.ts:126-133).
///
/// `~` expansion, relative resolution against the project root, symlink
/// `realpath`, then a `relative()` result that must not contain `..` when
/// split on `/` or `\`. On POSIX `relative` can never be absolute; the
/// Windows drive-mismatch case it guards collapses to `..` segments here —
/// same verdict.
#[must_use]
pub fn is_collection_path_inside_project(
    config_path: &str,
    raw_path: &str,
    env: &Environment,
) -> bool {
    let root = real_or_resolve(&project_root_from_config(config_path, env), env);
    let target = real_or_resolve(
        &resolve_config_collection_path(raw_path, config_path, env),
        env,
    );
    // `rel === ""` → inside; `isAbsolute(rel)` can't happen on POSIX;
    // upstream splits on both separators, so a filename like `a\..`
    // inside the project counts as an escape — mirrored verbatim.
    let rel = relative_segments(&root, &target).join("/");
    if rel.is_empty() {
        return true;
    }
    if Path::new(&rel).is_absolute() {
        return false;
    }
    !rel.split(['/', '\\']).any(|s| s == "..")
}

/// Model URIs that differ from the built-in defaults — upstream
/// `gatedModels` (trust.ts:135-147): fixed slot order, missing or empty
/// URIs and builtin-equal URIs are not gated.
#[must_use]
pub fn gated_models(models: &ModelsSnapshot, builtins: &BuiltinModels) -> Vec<GatedModel> {
    let mut out = Vec::new();
    for slot in ModelSlot::ALL {
        let Some(uri) = models.get(slot) else {
            continue;
        };
        if uri.is_empty() || uri == builtins.get(slot) {
            continue;
        }
        out.push(GatedModel {
            slot,
            uri: uri.clone(),
        });
    }
    out
}

/// The subset of the snapshot needing approval — upstream `gatedItems`
/// (trust.ts:149-159): every hook, paths that leave the project, and
/// non-default models.
#[must_use]
pub fn gated_items(
    config_path: &str,
    snapshot: &SensitiveSnapshot,
    builtins: &BuiltinModels,
    env: &Environment,
) -> GatedItems {
    GatedItems {
        hooks: snapshot.hooks.clone(),
        paths: snapshot
            .paths
            .iter()
            .filter(|p| !is_collection_path_inside_project(config_path, &p.path, env))
            .cloned()
            .collect(),
        models: gated_models(&snapshot.models, builtins),
    }
}

/// Upstream `hasGatedItems` (trust.ts:161-163).
#[must_use]
pub const fn has_gated_items(gated: &GatedItems) -> bool {
    !gated.hooks.is_empty() || !gated.paths.is_empty() || !gated.models.is_empty()
}

/// Upstream `byFirst` (trust.ts:165-169): compare `a[0]`/`b[0]` with JS
/// `<` — **UTF-16 code-unit order**, which differs from `str::cmp`
/// between `U+E000`–`U+FFFF` and supplementary-plane characters. Ties
/// keep input order (upstream `sort` is stable, like ours).
fn by_first(a: &[String; 2], b: &[String; 2]) -> std::cmp::Ordering {
    a[0].encode_utf16().cmp(b[0].encode_utf16())
}

/// Canonical sections as `JSON.stringify({hooks,paths,models})` —
/// fixed key order, `["k","v"]` pairs (metadata-filter.ts-free port of
/// trust.ts:197-205).
#[derive(Serialize)]
struct CanonicalSections {
    hooks: Vec<[String; 2]>,
    paths: Vec<[String; 2]>,
    models: Vec<[String; 2]>,
}

/// Digest of the gated surface of a project-local config — upstream
/// `sensitiveDigest` (trust.ts:191-207).
///
/// Hooks, resolved collection paths, and non-default model URIs are each
/// `[key, value]`-paired and sorted by key, serialized with
/// `JSON.stringify`, then sha256-hex. Order-independent so reordering
/// collections does not re-arm the gate; missing model keys and built-in
/// defaults are equivalent so `qmd init` filling defaults does not
/// invalidate an approval.
#[must_use]
pub fn sensitive_digest(
    snapshot: &SensitiveSnapshot,
    config_path: &str,
    builtins: &BuiltinModels,
    env: &Environment,
) -> String {
    use std::fmt::Write as _;
    let gated = gated_items(config_path, snapshot, builtins, env);
    let mut hooks: Vec<[String; 2]> = gated
        .hooks
        .iter()
        .map(|h| [h.collection.clone(), h.command.clone()])
        .collect();
    hooks.sort_by(by_first);
    let mut paths: Vec<[String; 2]> = gated
        .paths
        .iter()
        .map(|p| {
            [
                p.collection.clone(),
                resolve_config_collection_path(&p.path, config_path, env),
            ]
        })
        .collect();
    paths.sort_by(by_first);
    let mut models: Vec<[String; 2]> = gated
        .models
        .iter()
        .map(|m| [m.slot.as_str().to_owned(), m.uri.clone()])
        .collect();
    models.sort_by(by_first);
    // serde_json compact output is byte-identical to `JSON.stringify`
    // for every character these inputs can carry (short escapes for
    // \n \t \r \b \f, \uXXXX for other C0 controls, literal non-ASCII).
    let canonical = serde_json::to_string(&CanonicalSections {
        hooks,
        paths,
        models,
    })
    .unwrap_or_default();
    let mut hex = String::with_capacity(64);
    for b in Sha256::digest(canonical.as_bytes()) {
        write!(hex, "{b:02x}").ok();
    }
    hex
}

/// Upstream `loadTrustStore` (trust.ts:209-220): missing, unreadable or
/// non-object JSON loads as an empty store — a corrupt file must never
/// read as blanket trust.
#[must_use]
pub fn load_trust_store(env: &Environment) -> TrustStore {
    let path = trust_file_path(env);
    if !path.exists() {
        return TrustStore::new();
    }
    let Ok(content) = std::fs::read_to_string(&path) else {
        return TrustStore::new();
    };
    match serde_json::from_str::<serde_json::Value>(&content) {
        Ok(serde_json::Value::Object(map)) => map,
        // `!parsed || typeof !== "object" || isArray` — and corrupt JSON.
        _ => TrustStore::new(),
    }
}

/// `saveTrustStore` (trust.ts:222-227): `JSON.stringify(store, null, 2)`
/// plus a trailing newline, parent directory created as needed.
fn save_trust_store(env: &Environment, store: &TrustStore) -> Result<()> {
    let path = trust_file_path(env);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|source| Error::Io {
            path: dir.to_path_buf(),
            source,
        })?;
    }
    let content = serde_json::to_string_pretty(store).unwrap_or_default() + "\n";
    std::fs::write(&path, content).map_err(|source| Error::Io {
        path: path.clone(),
        source,
    })
}

/// Trust records are keyed on the resolved config path — upstream
/// `trustKey` (trust.ts:230-232).
fn trust_key(config_path: &str, env: &Environment) -> String {
    node_resolve(config_path, env)
}

/// Upstream `isTrusted` (trust.ts:234-237): the stored `hooks` digest
/// must equal the current one — wrong-typed records do not match.
#[must_use]
pub fn is_trusted(env: &Environment, config_path: &str, digest: &str) -> bool {
    load_trust_store(env)
        .get(&trust_key(config_path, env))
        .and_then(|r| r.get("hooks"))
        .and_then(serde_json::Value::as_str)
        == Some(digest)
}

/// Upstream `recordTrust` (trust.ts:239-243): overwrite or append the
/// record for `config_path` (insertion order preserved, like JS object
/// assignment).
///
/// # Errors
/// [`Error::Io`] on filesystem failures.
pub fn record_trust(env: &Environment, config_path: &str, digest: &str) -> Result<()> {
    let mut store = load_trust_store(env);
    store.insert(
        trust_key(config_path, env),
        serde_json::json!({
            "hooks": digest,
            "trustedAt": crate::store::documents::now_iso(),
        }),
    );
    save_trust_store(env, &store)
}

/// Drop the record for a config path — upstream `revokeTrust`
/// (trust.ts:246-253). Returns `false` when there was none.
/// `shift_remove` preserves entry order like JS `delete`.
///
/// # Errors
/// [`Error::Io`] on filesystem failures.
pub fn revoke_trust(env: &Environment, config_path: &str) -> Result<bool> {
    let mut store = load_trust_store(env);
    let key = trust_key(config_path, env);
    if store.shift_remove(&key).is_none() {
        return Ok(false);
    }
    save_trust_store(env, &store)?;
    Ok(true)
}

/// One entry of [`list_trusted`] — `{path, ...record}` narrowed to the
/// fields callers read (see module docs for the spread deviation).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TrustedEntry {
    /// Resolved config path (the map key).
    pub path: String,
    /// Approved digest (`record.hooks`), `None` when absent/non-string.
    pub hooks: Option<String>,
    /// ISO approval timestamp, `None` when absent/non-string.
    pub trusted_at: Option<String>,
}

/// Upstream `listTrusted` (trust.ts:255-257) — entries in insertion
/// order, like `Object.entries`.
#[must_use]
pub fn list_trusted(env: &Environment) -> Vec<TrustedEntry> {
    load_trust_store(env)
        .into_iter()
        .map(|(path, record)| TrustedEntry {
            path,
            hooks: record
                .get("hooks")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            trusted_at: record
                .get("trustedAt")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
        })
        .collect()
}

/// Upstream `isTruthyEnv` (trust.ts:266-269): unset/empty is falsy;
/// `0`/`false`/`off`/`no`/`none` (trimmed, lowercased) are falsy;
/// anything else is truthy.
fn is_truthy_env(value: Option<&str>) -> bool {
    let Some(v) = value else { return false };
    if v.is_empty() {
        return false;
    }
    !matches!(
        crate::chunk::js_trim(v).to_lowercase().as_str(),
        "0" | "false" | "off" | "no" | "none"
    )
}

/// True when the process has opted in to trusting project-local config
/// unattended — upstream `isLocalConfigTrustOptedIn` (trust.ts:272-274).
#[must_use]
pub fn is_local_config_trust_opted_in(env: &Environment) -> bool {
    is_truthy_env(env.qmd_trust_local_config.as_deref())
        || is_truthy_env(env.qmd_trust_update_hooks.as_deref())
}

/// What the gate decided — upstream `HookGateDecision`/`LocalConfigGateDecision`
/// action field (trust.ts:259-264).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateAction {
    /// Run the gated surface.
    Run,
    /// Ask the user first (interactive callers only).
    Prompt,
    /// Skip the gated surface (non-interactive fallback).
    Skip,
}

/// Gate outcome — the action plus the digest a later approval would
/// record (upstream `{action, digest}`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct GateDecision {
    /// Decision.
    pub action: GateAction,
    /// Digest of the gated surface that was evaluated.
    pub digest: String,
}

/// Inputs to [`decide_local_config_gate`] — upstream `options`
/// (trust.ts:313-320).
#[derive(Debug)]
pub struct LocalConfigGateInput<'a> {
    /// Config file path (resolved for the trust record key).
    pub config_path: &'a str,
    /// Declared gated surface.
    pub snapshot: &'a SensitiveSnapshot,
    /// Built-in default model URIs.
    pub builtins: &'a BuiltinModels,
    /// Whether a human can answer a prompt.
    pub is_interactive: bool,
}

/// Whether a project-local config may use its gated fields — upstream
/// `decideLocalConfigGate` (trust.ts:313-333).
///
/// In-project paths and built-in default models need no decision;
/// non-interactive callers get `skip`: in-project indexing still proceeds,
/// hooks/outside paths/custom models do not.
///
/// Injected `(configPath, digest) -> trusted` lookup for
/// [`decide_local_config_gate`] — mirrors upstream's `trustedCheck` so tests
/// can stub the store like upstream does.
pub type TrustedCheck<'a> = dyn Fn(&str, &str) -> bool + 'a;

/// `trusted_check` defaults to [`is_trusted`] against the real store;
/// tests inject a closure like upstream's `trustedCheck`.
#[must_use]
pub fn decide_local_config_gate(
    input: &LocalConfigGateInput<'_>,
    env: &Environment,
    trusted_check: Option<&TrustedCheck<'_>>,
) -> GateDecision {
    let digest = sensitive_digest(input.snapshot, input.config_path, input.builtins, env);
    let gated = gated_items(input.config_path, input.snapshot, input.builtins, env);

    if !has_gated_items(&gated) {
        return GateDecision {
            action: GateAction::Run,
            digest,
        };
    }
    if is_local_config_trust_opted_in(env) {
        return GateDecision {
            action: GateAction::Run,
            digest,
        };
    }
    if !is_local_config_path(input.config_path, env) {
        return GateDecision {
            action: GateAction::Run,
            digest,
        };
    }
    let trusted =
        |p: &str, d: &str| trusted_check.map_or_else(|| is_trusted(env, p, d), |f| f(p, d));
    if trusted(input.config_path, &digest) {
        return GateDecision {
            action: GateAction::Run,
            digest,
        };
    }
    GateDecision {
        action: if input.is_interactive {
            GateAction::Prompt
        } else {
            GateAction::Skip
        },
        digest,
    }
}

// =============================================================================
// The `cli/qmd.ts` composition layer — the trust helpers above combined
// the way upstream's `update`/`trust`/`getStore` paths combine them.
// =============================================================================

/// Upstream `builtinModels` (qmd.ts:715-719): the default model URIs a
/// declared `models:` entry is compared against.
#[must_use]
pub fn builtin_models() -> BuiltinModels {
    BuiltinModels {
        embed: crate::llm::DEFAULT_EMBED_MODEL.to_owned(),
        rerank: crate::llm::DEFAULT_RERANK_MODEL.to_owned(),
        generate: crate::llm::DEFAULT_GENERATE_MODEL.to_owned(),
    }
}

/// Upstream `collectSensitiveSnapshot` (qmd.ts:726-745): the gated
/// surface of the parsed config.
///
/// Every non-empty `update:` hook, every declared collection path, and
/// the `models:` URIs, in config order (`IndexMap` iteration matches
/// `Object.entries` on the YAML map). An empty-string `update:` does
/// not gate: upstream's `!!col.update` filter treats `""` as absent.
#[must_use]
pub fn collect_sensitive_snapshot(config: &crate::config::Config) -> SensitiveSnapshot {
    SensitiveSnapshot {
        hooks: config
            .collections
            .iter()
            .filter_map(|(name, col)| {
                let command = col.update.as_deref().filter(|c| !c.is_empty())?;
                Some(UpdateHook {
                    collection: name.clone(),
                    command: command.to_owned(),
                })
            })
            .collect(),
        paths: config
            .collections
            .iter()
            .map(|(name, col)| CollectionPath {
                collection: name.clone(),
                path: col.path.clone(),
            })
            .collect(),
        models: config
            .models
            .as_ref()
            .map_or_else(ModelsSnapshot::default, |m| ModelsSnapshot {
                embed: m.embed.clone(),
                rerank: m.rerank.clone(),
                generate: m.generate.clone(),
            }),
    }
}

/// Upstream `localConfigDigest` (qmd.ts:747-749): the sensitive digest
/// of `snapshot` under the built-in defaults.
///
/// `config_path` is stringified lossily — upstream keys trust records on
/// the resolved path string; non-UTF-8 paths degrade to U+FFFD there.
#[must_use]
pub fn local_config_digest(
    config_path: &Path,
    snapshot: &SensitiveSnapshot,
    env: &Environment,
) -> String {
    sensitive_digest(
        snapshot,
        &config_path.to_string_lossy(),
        &builtin_models(),
        env,
    )
}

/// Upstream `localConfigGated` (qmd.ts:751-753): the gated subset of
/// `snapshot` under the built-in defaults.
#[must_use]
pub fn local_config_gated(
    config_path: &Path,
    snapshot: &SensitiveSnapshot,
    env: &Environment,
) -> GatedItems {
    gated_items(
        &config_path.to_string_lossy(),
        snapshot,
        &builtin_models(),
        env,
    )
}

/// Whether this process may use the gated fields of the config —
/// upstream `localConfigIsFullyTrusted` (qmd.ts:759-767).
///
/// Non-local config paths and the `QMD_TRUST_LOCAL_CONFIG` /
/// `QMD_TRUST_UPDATE_HOOKS` opt-ins are always trusted; otherwise the
/// gated set must be empty or recorded in `trusted.json` under the
/// current digest.
#[must_use]
pub fn local_config_fully_trusted(
    config_path: &Path,
    config: &crate::config::Config,
    env: &Environment,
) -> bool {
    if !is_local_config_path(&config_path.to_string_lossy(), env) {
        return true;
    }
    if is_local_config_trust_opted_in(env) {
        return true;
    }
    let snapshot = collect_sensitive_snapshot(config);
    let gated = local_config_gated(config_path, &snapshot, env);
    if !has_gated_items(&gated) {
        return true;
    }
    is_trusted(
        env,
        &config_path.to_string_lossy(),
        &local_config_digest(config_path, &snapshot, env),
    )
}

/// Record the config's current gated set as approved in `trusted.json`
/// — upstream `trustCurrentConfig` (qmd.ts:770-776). A no-op for
/// non-local config paths.
///
/// # Errors
/// [`Error::Io`] when the trust store cannot be written.
pub fn trust_current_config(
    config_path: &Path,
    config: &crate::config::Config,
    env: &Environment,
) -> Result<()> {
    if !is_local_config_path(&config_path.to_string_lossy(), env) {
        return Ok(());
    }
    let snapshot = collect_sensitive_snapshot(config);
    let digest = local_config_digest(config_path, &snapshot, env);
    record_trust(env, &config_path.to_string_lossy(), &digest)
}
