//! Model download, HF cache naming, and GGUF validation.
//!
//! Ports upstream `pullModels` / `inspectGgufFile` / `validateGgufFile`
//! (`src/llm.ts:323-580`) plus node-llama-cpp's `hf:` URI resolution and
//! `hf_*` cache naming (`parseModelUri.js`, `buildHuggingFaceFilePrefix`),
//! npm `filenamify` defaults, and the endpoint resolution of
//! `modelDownloadEndpoints.js`.
//!
//! Downloads stream through `ureq` into `{name}.part` files that are promoted
//! by rename only after the body completes, so a partial file never poses as
//! a model. Upstream's ipull layer downloads split parts in parallel; this
//! port downloads them sequentially — same bytes, same filenames. `.part`
//! files are not resumed (upstream ipull resumes them); a stale `.part` is
//! truncated and overwritten on the next attempt.
//!
//! Deviation: the remote-etag HEAD request uses the resolved endpoint and
//! the URI's branch. Upstream `getRemoteEtag` hardcodes `huggingface.co` and
//! `main`, which silently defeats `HF_ENDPOINT` mirrors.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::env::Environment;
use crate::error::{Error, Result};
use crate::paths::{cache_dir, resolve_lexical};

/// Upstream `GGUF_MAGIC` (`llm.ts:361`).
pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";

const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co/";
const DEFAULT_HF_BRANCH: &str = "main";
const MAX_FILENAME_LENGTH: usize = 100;

// ============================================================================
// GGUF inspection (llm.ts:361-455)
// ============================================================================

/// Upstream `GgufFileInspection.kind`.
///
/// `Html` and `Invalid` mean the header was read and is confirmed bad;
/// `Unreadable` means the read itself failed, so nothing is known — only
/// the first two justify deleting the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufKind {
    /// File does not exist.
    Missing,
    /// First four bytes are `GGUF`.
    Gguf,
    /// Sniffed `<!doctype` or `<html` — a proxy/firewall/captive-portal page.
    Html,
    /// Readable but not GGUF and not HTML.
    Invalid,
    /// stat/open/read failed; content unknown.
    Unreadable,
}

/// Upstream `GgufFileInspection` (`llm.ts:365-378`).
#[derive(Debug, Clone)]
pub struct GgufInspection {
    /// The file exists.
    pub exists: bool,
    /// The file is a GGUF model.
    pub valid: bool,
    /// Classification.
    pub kind: GgufKind,
    /// File size in bytes (`None` when stat failed or file is missing).
    pub size_bytes: Option<u64>,
    /// Printable form of the first four bytes (`None` unless read).
    pub magic: Option<String>,
    /// Human-readable verdict, upstream wording.
    pub details: String,
}

/// Upstream `formatModelFileSize`: `${(size/1024).toFixed(0)} KB`.
/// `toFixed(0)` rounds half away from zero.
#[must_use]
pub fn format_model_file_size(size_bytes: u64) -> String {
    #[allow(
        clippy::cast_precision_loss,
        reason = "upstream divides by 1024 in f64"
    )]
    let kb = (size_bytes as f64 / 1024.0).round();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "kb is finite and non-negative"
    )]
    {
        format!("{} KB", kb as u64)
    }
}

/// Upstream `printableMagic`: utf8 when `^[\x20-\x7e]{1,4}$` (the whole
/// 4-byte header printable), else `0x{hex}` of all four bytes.
fn printable_magic(header: [u8; 4]) -> String {
    use std::fmt::Write as _;
    if header.iter().all(|b| (0x20..=0x7e).contains(b)) {
        return String::from_utf8_lossy(&header).into_owned();
    }
    let mut hex = String::from("0x");
    for b in header {
        write!(hex, "{b:02x}").ok();
    }
    hex
}

/// Upstream `inspectGgufFile` (`llm.ts:393-455`): read 512 bytes, check the
/// four-byte `GGUF` magic, then sniff for an HTML error page.
#[must_use]
pub fn inspect_gguf_file(path: &Path) -> GgufInspection {
    if !path.exists() {
        return GgufInspection {
            exists: false,
            valid: false,
            kind: GgufKind::Missing,
            size_bytes: None,
            magic: None,
            details: "file does not exist".to_owned(),
        };
    }
    let inspected = (|| -> std::io::Result<GgufInspection> {
        let size_bytes = std::fs::metadata(path)?.len();
        let mut sniff = [0u8; 512];
        let mut file = std::fs::File::open(path)?;
        let n = file.read(&mut sniff)?;
        let sniff = sniff.get(..n).unwrap_or_default();
        let header: [u8; 4] = sniff
            .get(..4)
            .and_then(|h| <&[u8] as TryInto<[u8; 4]>>::try_into(h).ok())
            .unwrap_or_default();
        if &header == GGUF_MAGIC {
            return Ok(GgufInspection {
                exists: true,
                valid: true,
                kind: GgufKind::Gguf,
                size_bytes: Some(size_bytes),
                magic: Some("GGUF".to_owned()),
                details: format!("valid GGUF ({})", format_model_file_size(size_bytes)),
            });
        }
        let magic = printable_magic(header);
        let text = String::from_utf8_lossy(sniff).to_lowercase();
        let kind = if text.contains("<!doctype") || text.contains("<html") {
            GgufKind::Html
        } else {
            GgufKind::Invalid
        };
        let details = match kind {
            GgufKind::Html => format!(
                "HTML page, not a GGUF model ({}); likely proxy/firewall/captive portal response",
                format_model_file_size(size_bytes)
            ),
            _ => format!(
                "not valid GGUF (expected magic \"GGUF\", got \"{magic}\", {})",
                format_model_file_size(size_bytes)
            ),
        };
        Ok(GgufInspection {
            exists: true,
            valid: false,
            kind,
            size_bytes: Some(size_bytes),
            magic: Some(magic),
            details,
        })
    })();
    inspected.unwrap_or_else(|e| GgufInspection {
        exists: true,
        valid: false,
        kind: GgufKind::Unreadable,
        size_bytes: None,
        magic: None,
        details: format!("cannot read model file: {e}"),
    })
}

// ============================================================================
// filenamify (npm package, default options: replacement '!', maxLength 100)
// ============================================================================

const FILENAMIFY_RESERVED: &[char] = &['<', '>', ':', '"', '/', '\\', '|', '?', '*'];

const fn is_filenamify_control(c: char) -> bool {
    matches!(c, '\u{0}'..='\u{1f}' | '\u{80}'..='\u{9f}')
}

fn is_windows_reserved_name(s: &str) -> bool {
    let stem = s.split('.').next().unwrap_or(s);
    let lower = stem.to_lowercase();
    matches!(
        lower.as_str(),
        "con"
            | "prn"
            | "aux"
            | "nul"
            | "com0"
            | "com1"
            | "com2"
            | "com3"
            | "com4"
            | "com5"
            | "com6"
            | "com7"
            | "com8"
            | "com9"
            | "lpt0"
            | "lpt1"
            | "lpt2"
            | "lpt3"
            | "lpt4"
            | "lpt5"
            | "lpt6"
            | "lpt7"
            | "lpt8"
            | "lpt9"
    )
}

/// Port of npm `filenamify` with default options (replacement `!`,
/// `maxLength` 100, NFD normalization, repeated-reserved collapse).
///
/// Rust has no NFD normalization in std; on real HF path components the
/// inputs are already ASCII, so the NFD step is an effective no-op for cache
/// naming. Control-character replacement covers `\u{0}-\u{1f}` and
/// `\u{80}-\u{9f}` like the original.
#[must_use]
pub fn filenamify(input: &str) -> String {
    // ([<>:"/\\|?*\x00-\x1f]){2,} -> $1 : collapse runs of reserved chars.
    let mut collapsed = String::with_capacity(input.len());
    let mut prev_reserved = false;
    for c in input.chars() {
        let reserved = FILENAMIFY_RESERVED.contains(&c) || ('\u{0}'..='\u{1f}').contains(&c);
        if reserved && prev_reserved {
            continue;
        }
        prev_reserved = reserved;
        collapsed.push(c);
    }
    // ^\.+(\\|\/)|^\.+$ -> '!'
    let mut s = collapsed;
    let bytes = s.as_bytes();
    let leading_dots = bytes.iter().take_while(|b| **b == b'.').count();
    if leading_dots > 0
        && (leading_dots == bytes.len() || matches!(bytes.get(leading_dots), Some(b'\\' | b'/')))
    {
        s = format!("!{}", s.get(leading_dots..).unwrap_or_default());
    }
    // filenameReservedRegex + control chars -> '!'
    s = s
        .chars()
        .map(|c| {
            if FILENAMIFY_RESERVED.contains(&c) || is_filenamify_control(c) {
                '!'
            } else {
                c
            }
        })
        .collect();
    // trailing periods stripped
    while s.ends_with('.') {
        s.pop();
    }
    // Upstream's dot-pad block is dead code (startedWithDot is measured on
    // the same string it gates), so leading dots are preserved verbatim.
    if is_windows_reserved_name(&s) {
        s.push('!');
    }
    // maxLength preserving extension
    if s.chars().count() > MAX_FILENAME_LENGTH {
        match s.rfind('.') {
            None => s = s.chars().take(MAX_FILENAME_LENGTH).collect(),
            Some(dot) => {
                let ext_len = s.chars().count() - dot;
                let keep = MAX_FILENAME_LENGTH.saturating_sub(ext_len).max(1);
                let head: String = s.chars().take(keep).collect();
                let ext: String = s.chars().skip(dot).collect();
                s = format!("{head}{ext}");
            }
        }
    }
    s
}

// ============================================================================
// hf: URI parsing and cache naming (parseModelUri.js)
// ============================================================================

/// A parsed `hf:user/model/path[...][#branch]` model reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfRef {
    /// `HuggingFace` user/org.
    pub user: String,
    /// Repository name.
    pub model: String,
    /// File path inside the repo (may contain `/`).
    pub file: String,
    /// Revision/branch; `main` when unspecified.
    pub branch: String,
}

impl HfRef {
    /// `{user}/{model}` — upstream's `repo` field.
    #[must_use]
    pub fn repo(&self) -> String {
        format!("{}/{}", self.user, self.model)
    }
}

/// Parse `hf:user/model/path[#branch]` (node-llama-cpp `parseModelUri`).
///
/// Returns `None` for non-`hf:` URIs and for `hf:` strings with fewer than
/// three `/`-separated parts — matching upstream `parseHfUri`, which treats
/// those as non-HF model specs.
#[must_use]
pub fn parse_hf_uri(model: &str) -> Option<HfRef> {
    let rest = model.strip_prefix("hf:")?;
    let (rest, branch) = rest.find('#').map_or_else(
        || (rest, DEFAULT_HF_BRANCH.to_owned()),
        |hash| {
            let b = rest.get(hash.saturating_add(1)..).unwrap_or("");
            (
                rest.get(..hash).unwrap_or(rest),
                if b.is_empty() {
                    DEFAULT_HF_BRANCH.to_owned()
                } else {
                    b.to_owned()
                },
            )
        },
    );
    let parts: Vec<&str> = rest.split('/').collect();
    if parts.len() < 3 {
        return None;
    }
    let user = (*parts.first()?).to_owned();
    let model_name = (*parts.get(1)?).to_owned();
    let file = parts.get(2..)?.join("/");
    if user.is_empty() || model_name.is_empty() || file.is_empty() {
        return None;
    }
    Some(HfRef {
        user,
        model: model_name,
        file,
        branch,
    })
}

/// Upstream `resolveHuggingFaceEndpoint`: explicit `MODEL_ENDPOINT`, then
/// `HF_ENDPOINT`, then `https://huggingface.co/` — always slash-terminated.
#[must_use]
pub fn resolve_hf_endpoint(env: &Environment) -> String {
    let candidate = env
        .model_endpoint
        .as_deref()
        .filter(|u| is_url(u))
        .or_else(|| env.hf_endpoint.as_deref().filter(|u| is_url(u)));
    match candidate {
        Some(u) if u.ends_with('/') => u.to_owned(),
        Some(u) => format!("{u}/"),
        None => DEFAULT_HF_ENDPOINT.to_owned(),
    }
}

fn is_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

/// Upstream `modelCacheDir` (`llm.ts:323-326`):
/// `XDG_CACHE_HOME/qmd/models` or `~/.cache/qmd/models`.
#[must_use]
pub fn model_cache_dir(env: &Environment) -> PathBuf {
    cache_dir(env).join("models")
}

/// encodeURIComponent per segment: encode everything except
/// `A-Z a-z 0-9 - _ . ! ~ * ' ( )`.
fn encode_uri_component(part: &str) -> String {
    use std::fmt::Write as _;
    const UNRESERVED: &[u8] = b"-_.!~*'()";
    let mut out = String::with_capacity(part.len());
    for &b in part.as_bytes() {
        if b.is_ascii_alphanumeric() || UNRESERVED.contains(&b) {
            out.push(char::from(b));
        } else {
            write!(out, "%{b:02X}").ok();
        }
    }
    out
}

/// The download URL for an [`HfRef`]:
/// `{endpoint}{user}/{model}/resolve/{branch}/{file}?download=true`.
#[must_use]
pub fn hf_download_url(hf: &HfRef, endpoint: &str) -> String {
    let file = hf
        .file
        .split('/')
        .map(encode_uri_component)
        .collect::<Vec<_>>()
        .join("/");
    format!(
        "{}{}/{}/resolve/{}/{}?download=true",
        endpoint,
        encode_uri_component(&hf.user),
        encode_uri_component(&hf.model),
        encode_uri_component(&hf.branch),
        file
    )
}

// ---- split GGUF parts (`resolveSplitGgufParts.js`) ----

/// `-(?<part>\d{5})-of-(?<parts>\d{5})\.gguf$` on a filename.
/// Returns `(part, parts, match_len)` like upstream `getGgufSplitPartsInfo`.
#[must_use]
pub fn get_gguf_split_parts_info(name: &str) -> Option<(u32, u32, usize)> {
    let stem = name
        .strip_suffix(".gguf")
        .or_else(|| name.strip_suffix(".GGUF"))?;
    let (stem, parts_str) = stem.rsplit_once("-of-")?;
    let (_base, part_str) = stem.rsplit_once('-')?;
    if part_str.len() != 5 || parts_str.len() != 5 {
        return None;
    }
    let part: u32 = part_str.parse().ok()?;
    let parts: u32 = parts_str.parse().ok()?;
    if part == 0 || parts == 0 || part > parts {
        return None;
    }
    // matched text: "-{part}-of-{parts}.gguf"
    let match_len = 1 + 5 + 4 + 5 + 5;
    Some((part, parts, match_len))
}

/// Upstream `createSplitPartFilename`.
#[must_use]
pub fn create_split_part_filename(filename: &str, part: u32, parts: u32) -> String {
    let mut base = filename.to_owned();
    if let Some((_, _, match_len)) = get_gguf_split_parts_info(&base) {
        base.truncate(base.len() - match_len);
        base.push_str(".gguf");
    }
    if base.to_lowercase().ends_with(".gguf") {
        base.truncate(base.len() - ".gguf".len());
    }
    format!("{base}-{part:05}-of-{parts:05}.gguf")
}

/// Expand `name-0000N-of-0000M.gguf` into all part names (upstream
/// `resolveSplitGgufParts` on a plain path). Returns `[name]` when the
/// input is not a named split part.
#[must_use]
pub fn resolve_split_gguf_parts(name: &str) -> Vec<String> {
    let Some((_, parts, match_len)) = get_gguf_split_parts_info(name) else {
        return vec![name.to_owned()];
    };
    let common = &name[..name.len() - match_len];
    (1..=parts)
        .map(|i| format!("{common}-{i:05}-of-{parts:05}.gguf"))
        .collect()
}

fn strip_gguf_suffix(name: &str) -> &str {
    name.strip_suffix(".gguf")
        .or_else(|| name.strip_suffix(".GGUF"))
        .unwrap_or(name)
}

fn strip_split_parts_suffix(name: &str) -> &str {
    match get_gguf_split_parts_info(name) {
        Some((_, _, match_len)) => name.get(..name.len() - match_len).unwrap_or(name),
        None => name,
    }
}

/// Upstream `doesFilenameMatchExactModelName`: model must end `-gguf` and
/// the filename (sans `.gguf`, and if that fails, sans the split-part
/// suffix too) must start with the model name sans `-gguf`.
fn filename_matches_model(filename: &str, model: &str) -> bool {
    let model_stem = model
        .to_lowercase()
        .strip_suffix("-gguf")
        .map(str::to_owned);
    let Some(model_stem) = model_stem else {
        return false;
    };
    if !filename.to_lowercase().ends_with(".gguf") {
        return false;
    }
    let lower = filename.to_lowercase();
    if strip_gguf_suffix(&lower).starts_with(&model_stem) {
        return true;
    }
    strip_gguf_suffix(strip_split_parts_suffix(&lower)).starts_with(&model_stem)
}

/// Upstream `doesFilenameMatchExactFolderName`: the directory name equals
/// the filename sans `.gguf` (or sans split-part suffix).
fn filename_matches_folder(filename: &str, folder: &str) -> bool {
    if !filename.to_lowercase().ends_with(".gguf") {
        return false;
    }
    let lower = filename.to_lowercase();
    if folder.to_lowercase() == strip_gguf_suffix(&lower) {
        return true;
    }
    folder.to_lowercase() == strip_gguf_suffix(strip_split_parts_suffix(&lower))
}

/// `resolveModelFilenameFromUrl` on an already-decoded basename: truncate
/// at the first `.gguf` (case-insensitive) and filenamify.
fn filename_from_file_part(part: &str) -> String {
    let lower = part.to_lowercase();
    let end = lower
        .find(".gguf")
        .map_or(part.len(), |i| i + ".gguf".len());
    filenamify(part.get(..end).unwrap_or(part))
}

/// `true` when the file is a binary split (`name.gguf.partNofM`): those
/// combine into one downloaded file rather than separate cache entries.
#[must_use]
pub fn is_binary_split_filename(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.find(".gguf.part").is_some_and(|i| {
        let tail = &lower[i + ".gguf.part".len()..];
        tail.split_once("of").is_some_and(|(n, m)| {
            !n.is_empty()
                && !m.is_empty()
                && n.bytes().all(|b| b.is_ascii_digit())
                && m.bytes().all(|b| b.is_ascii_digit())
        })
    })
}

/// Upstream `buildHuggingFaceFilePrefix` + `fullFilename`.
fn hf_file_prefix(hf: &HfRef, filename: &str) -> String {
    let mut res = vec!["hf".to_owned(), filenamify(&hf.user)];
    let is_main = hf.branch == DEFAULT_HF_BRANCH;
    if !filename_matches_model(filename, &hf.model) || !is_main {
        res.push(filenamify(&hf.model));
    }
    if !is_main {
        res.push(filenamify(&hf.branch));
    }
    let mut dirs: Vec<&str> = hf.file.split('/').collect();
    dirs.pop(); // the filename itself
    if dirs
        .last()
        .is_some_and(|last| filename_matches_folder(filename, last))
    {
        dirs.pop();
    }
    if !dirs.is_empty() {
        res.push(filenamify(&dirs.join("__")));
    }
    let mut out = res.join("_");
    out.push('_');
    out
}

/// The cache filename for an [`HfRef`] — node-llama-cpp's `fullFilename`
/// (`hf_{user}[_model][_branch][_dirs__]{file}`).
#[must_use]
pub fn hf_cache_filename(hf: &HfRef) -> String {
    let basename = hf.file.rsplit('/').next().unwrap_or(&hf.file);
    let filename = filename_from_file_part(basename);
    format!("{}{}", hf_file_prefix(hf, &filename), filename)
}

/// Every cache filename a model URI produces. A `-NNNNN-of-NNNNN` URI
/// names all its parts (prefix computed per part name); a `.partNofM`
/// binary-split URI combines to one.
#[must_use]
pub fn hf_cache_filenames(hf: &HfRef) -> Vec<String> {
    let basename = hf.file.rsplit('/').next().unwrap_or(&hf.file);
    let filename = filename_from_file_part(basename);
    resolve_split_gguf_parts(&filename)
        .into_iter()
        .map(|part| format!("{}{}", hf_file_prefix(hf, &part), part))
        .collect()
}

/// All download URLs for an [`HfRef`].
///
/// Named split parts expand to one URL per part (each
/// `/resolve/{branch}/{dir}/{part}`), binary splits expand to
/// `.../{stem}.part{i}of{N}`, and ordinary files give a single URL.
#[must_use]
pub fn hf_download_urls(hf: &HfRef, endpoint: &str) -> Vec<String> {
    let basename = hf.file.rsplit('/').next().unwrap_or(&hf.file);
    // binary split: file.gguf.partNofM -> N part URLs
    let lower = basename.to_lowercase();
    if let Some(i) = lower.find(".gguf.part")
        && let Some((n_str, m_str)) = lower
            .get(i + ".gguf.part".len()..)
            .and_then(|tail| tail.split_once("of"))
        && let (Ok(_n), Ok(parts)) = (n_str.parse::<u32>(), m_str.parse::<u32>())
    {
        let dir = hf
            .file
            .rsplit_once('/')
            .map_or("", |(d, _)| hf.file.get(..d.len()).unwrap_or(""));
        let stem = basename.get(..i).unwrap_or(basename);
        let width = n_str.len();
        return (1..=parts)
            .map(|p| binary_part_url(hf, dir, stem, p, width, m_str, endpoint))
            .collect();
    }
    // named split: file-0000N-of-0000M.gguf -> N part URLs
    if get_gguf_split_parts_info(basename).is_some() {
        return resolve_split_gguf_parts(basename)
            .into_iter()
            .map(|part_name| {
                let mut r = hf.clone();
                r.file = hf
                    .file
                    .rsplit_once('/')
                    .map_or_else(|| part_name.clone(), |(d, _)| format!("{d}/{part_name}"));
                hf_download_url(&r, endpoint)
            })
            .collect();
    }
    vec![hf_download_url(hf, endpoint)]
}

/// URL of one binary-split part: `/resolve/{branch}/{dir}/{part}`.
fn binary_part_url(
    hf: &HfRef,
    dir: &str,
    stem: &str,
    part: u32,
    width: usize,
    m_str: &str,
    endpoint: &str,
) -> String {
    let part_name = format!("{stem}.gguf.part{part:0>width$}of{m_str}");
    let mut r = hf.clone();
    r.file = if dir.is_empty() {
        part_name
    } else {
        format!("{dir}/{part_name}")
    };
    hf_download_url(&r, endpoint)
}

// ============================================================================
// pull_models (llm.ts:523-580 + resolveModelFile.js)
// ============================================================================

/// Upstream `PullResult`.
#[derive(Debug, Clone)]
pub struct PullResult {
    /// Model spec as given.
    pub model: String,
    /// Path of the (entrypoint) model file.
    pub path: PathBuf,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Whether a stale cached file was replaced this call.
    pub refreshed: bool,
}

/// `getRemoteEtag`: HEAD the resolve URL; `None` on any failure.
fn remote_etag(agent: &ureq::Agent, hf: &HfRef, endpoint: &str) -> Option<String> {
    let url = format!(
        "{}{}/resolve/{}/{}",
        endpoint,
        hf.repo(),
        hf.branch,
        hf.file
    );
    let resp = agent.head(&url).call().ok()?;
    resp.headers()
        .get("etag")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
}

fn etag_path(cache_dir: &Path, filename: &str) -> PathBuf {
    cache_dir.join(format!("{filename}.etag"))
}

/// Delete a confirmed-bad model file and produce the upstream error.
/// `unreadable`/`missing` are handled by the caller.
fn validate_gguf_file(path: &Path, model_uri: &str) -> Result<()> {
    let inspection = inspect_gguf_file(path);
    if !inspection.exists || inspection.valid {
        return Ok(());
    }
    if inspection.kind == GgufKind::Unreadable {
        return Err(Error::Download {
            reason: format!(
                "Model file could not be read, so it could not be validated ({}).\n\
                 Model: {model_uri}\nPath:  {}\n\n\
                 The file has been left in place. If this repeats, check open file limits, \
                 permissions, and whether the volume holding the model cache is still mounted.",
                inspection.details,
                path.display()
            ),
        });
    }
    let removed = std::fs::remove_file(path).is_ok();
    if inspection.kind == GgufKind::Html {
        return Err(Error::Download {
            reason: format!(
                "Downloaded model file is an HTML page, not a GGUF model ({}).\n\
                 Something is intercepting the download from huggingface.co \
                 (a proxy, firewall, or captive portal).\n\n\
                 Model: {model_uri}\nPath:  {}\n\n\
                 To fix this, either:\n  \
                 1. Try a HuggingFace mirror:  HF_ENDPOINT=https://hf-mirror.com qmd embed\n  \
                 2. Download the model manually and set the env var, e.g.:\n       \
                 QMD_EMBED_MODEL=/path/to/model.gguf qmd embed\n\n\
                 Note: 'qmd search' works without any model downloads.",
                format_model_file_size(inspection.size_bytes.unwrap_or(0)),
                path.display()
            ),
        });
    }
    Err(Error::Download {
        reason: format!(
            "Model file is not valid GGUF (expected magic \"GGUF\", got \"{}\", file is {}).\n\
             Model: {model_uri}\nPath:  {}\n\n{}",
            inspection.magic.as_deref().unwrap_or("unknown"),
            format_model_file_size(inspection.size_bytes.unwrap_or(0)),
            path.display(),
            if removed {
                "The file has been removed. Run the command again to re-download."
            } else {
                "The file could NOT be removed. Delete it manually, then run the command again."
            }
        ),
    })
}

/// `Content-Length` of the response, when the server sends one.
fn content_length(resp: &ureq::http::Response<ureq::Body>) -> Option<u64> {
    resp.headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
}

/// Stream `reader` into `out`, reporting `(bytes_done, total)` after
/// every chunk — `total` is the response `Content-Length` (`None` when
/// the server does not send one, e.g. chunked encoding).
fn copy_with_progress(
    reader: &mut impl Read,
    out: &mut std::fs::File,
    path: &Path,
    total: Option<u64>,
    on_progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<()> {
    let mut done = 0u64;
    let mut buf = vec![0u8; 64 * 1024].into_boxed_slice();
    loop {
        let n = reader.read(&mut buf).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        if n == 0 {
            break;
        }
        out.write_all(buf.get(..n).unwrap_or_default())
            .map_err(|e| Error::Io {
                path: path.to_path_buf(),
                source: e,
            })?;
        done = done.saturating_add(u64::try_from(n).unwrap_or(u64::MAX));
        on_progress(done, total);
    }
    Ok(())
}

/// Download `url` to `dest` via `{dest}.part`, promoting by rename.
fn download_to(
    agent: &ureq::Agent,
    url: &str,
    dest: &Path,
    on_progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<()> {
    let part = dest.with_file_name(format!("{}.part", dest_display_name(dest)));
    let mut resp = agent.get(url).call().map_err(|e| Error::Download {
        reason: format!("download failed for {url}: {e}"),
    })?;
    let total = content_length(&resp);
    let mut out = std::fs::File::create(&part).map_err(|e| Error::Io {
        path: part.clone(),
        source: e,
    })?;
    let mut reader = resp.body_mut().as_reader();
    copy_with_progress(&mut reader, &mut out, &part, total, on_progress)?;
    out.flush().map_err(|e| Error::Io {
        path: part.clone(),
        source: e,
    })?;
    drop(out);
    std::fs::rename(&part, dest).map_err(|e| Error::Io {
        path: dest.to_path_buf(),
        source: e,
    })
}

fn dest_display_name(dest: &Path) -> String {
    dest.file_name()
        .map_or_else(|| "model".to_owned(), |n| n.to_string_lossy().into_owned())
}

/// Upstream `pullModels` (`llm.ts:523-580`).
///
/// `cache_dir` overrides [`model_cache_dir`]; `refresh` forces
/// re-download of every model regardless of etags.
///
/// # Errors
/// [`Error::Download`] on HTTP failure or a confirmed-bad (HTML/invalid)
/// download; [`Error::Io`] on filesystem failure; [`Error::InvalidInput`]
/// on an `hf:` URI with fewer than three parts.
pub fn pull_models(
    models: &[String],
    env: &Environment,
    cache_dir: Option<&Path>,
    refresh: bool,
) -> Result<Vec<PullResult>> {
    pull_models_with_progress(models, env, cache_dir, refresh, &mut |_, _, _| {})
}

/// [`pull_models`] with download progress — upstream's `cli` option
/// (`pullModels({cli: …})`, llm.ts:523-568).
///
/// `on_progress(model, done, total)` fires after every written chunk:
/// `done`/`total` are the in-flight response's byte counters and
/// `total` is `None` when the server omits `Content-Length`.
///
/// # Errors
/// [`Error::Download`] on HTTP failure or a confirmed-bad (HTML/invalid)
/// download; [`Error::Io`] on filesystem failure; [`Error::InvalidInput`]
/// on an `hf:` URI with fewer than three parts.
pub fn pull_models_with_progress(
    models: &[String],
    env: &Environment,
    cache_dir: Option<&Path>,
    refresh: bool,
    on_progress: &mut dyn FnMut(&str, u64, Option<u64>),
) -> Result<Vec<PullResult>> {
    let dir = cache_dir.map_or_else(|| model_cache_dir(env), Path::to_path_buf);
    std::fs::create_dir_all(&dir).map_err(|e| Error::Io {
        path: dir.clone(),
        source: e,
    })?;
    let endpoint = resolve_hf_endpoint(env);
    let agent = ureq::Agent::new_with_defaults();

    let mut results = Vec::with_capacity(models.len());
    for model in models {
        let mut refreshed = false;
        let filename = model.rsplit('/').next().unwrap_or(model);
        let hf = parse_hf_uri(model);

        if model.starts_with("hf:") && hf.is_none() {
            return Err(Error::InvalidInput {
                reason: format!("Invalid Hugging Face URI: {model}"),
            });
        }

        // Etag-driven refresh (upstream lines 537-565).
        if let Some(hf_ref) = &hf {
            let etag = etag_path(&dir, filename);
            let remote = remote_etag(&agent, hf_ref, &endpoint);
            let local = std::fs::read_to_string(&etag)
                .ok()
                .map(|s| s.trim().to_owned());
            let cached = cached_files_containing(&dir, filename);
            let should_refresh =
                refresh || remote.is_none() || remote != local || cached.is_empty();
            if should_refresh {
                remove_files(&cached);
                drop(std::fs::remove_file(&etag));
                refreshed = !cached.is_empty();
            }

            let path = resolve_or_download(&agent, hf_ref, &endpoint, &dir, &mut |done, total| {
                on_progress(model, done, total);
            })?;
            validate_gguf_file(&path, model)?;
            let size_bytes = std::fs::metadata(&path).map_or(0, |m| m.len());
            // Upstream HEADs again after the download (llm.ts:572) so the
            // stored tag reflects the remote at completion, not the possibly
            // stale pre-download value.
            if let Some(etag_value) = remote_etag(&agent, hf_ref, &endpoint) {
                drop(std::fs::write(&etag, format!("{etag_value}\n")));
            }
            results.push(PullResult {
                model: model.clone(),
                path,
                size_bytes,
                refreshed,
            });
        } else {
            // Local file: path.resolve(cacheDir, spec) (resolveModelFile.js).
            if refresh {
                let cached = cached_files_containing(&dir, filename);
                remove_files(&cached);
                refreshed = !cached.is_empty();
            }
            let path = resolve_lexical(&dir, model);
            if !path.exists() {
                return Err(Error::Download {
                    reason: format!("No model file found at \"{}\"", path.display()),
                });
            }
            validate_gguf_file(&path, model)?;
            let size_bytes = std::fs::metadata(&path).map_or(0, |m| m.len());
            results.push(PullResult {
                model: model.clone(),
                path,
                size_bytes,
                refreshed,
            });
        }
    }
    Ok(results)
}

/// Delete each file (best-effort — upstream ignores unlink failures).
fn remove_files(files: &[PathBuf]) {
    for candidate in files {
        drop(std::fs::remove_file(candidate));
    }
}

/// Regular files in `dir` whose names contain `filename` — the refresh
/// sweep upstream performs before re-downloading.
fn cached_files_containing(dir: &Path, filename: &str) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(std::result::Result::ok)
                .filter(|e| {
                    e.file_type().is_ok_and(|t| t.is_file())
                        && e.file_name().to_string_lossy().contains(filename)
                })
                .map(|e| e.path())
                .collect()
        })
        .unwrap_or_default()
}

/// `resolveModelFile` for `hf:` URIs: reuse the cache entry when present,
/// else stream every part to disk and return the entrypoint name.
/// `on_progress` fires per response chunk with that response's byte
/// counters (the bar resets between split parts, like upstream's
/// per-file progress).
fn resolve_or_download(
    agent: &ureq::Agent,
    hf: &HfRef,
    endpoint: &str,
    dir: &Path,
    on_progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<PathBuf> {
    let names = hf_cache_filenames(hf);
    let entrypoint = dir.join(names.first().cloned().unwrap_or_default());
    if entrypoint.exists() && names.iter().all(|n| dir.join(n).exists()) {
        return Ok(entrypoint);
    }
    let urls = hf_download_urls(hf, endpoint);
    if is_binary_split_filename(hf.file.rsplit('/').next().unwrap_or(&hf.file)) {
        // binary split: concatenate parts into the single combined file
        let part = entrypoint.with_file_name(format!(
            "{}.part",
            names.first().cloned().unwrap_or_default()
        ));
        let mut out = std::fs::File::create(&part).map_err(|e| Error::Io {
            path: part.clone(),
            source: e,
        })?;
        for url in &urls {
            let mut resp = agent.get(url).call().map_err(|e| Error::Download {
                reason: format!("download failed for {url}: {e}"),
            })?;
            let total = content_length(&resp);
            let mut reader = resp.body_mut().as_reader();
            copy_with_progress(&mut reader, &mut out, &part, total, on_progress)?;
        }
        out.flush().map_err(|e| Error::Io {
            path: part.clone(),
            source: e,
        })?;
        drop(out);
        std::fs::rename(&part, &entrypoint).map_err(|e| Error::Io {
            path: entrypoint.clone(),
            source: e,
        })?;
        return Ok(entrypoint);
    }
    for (name, url) in names.iter().zip(urls.iter()) {
        download_to(agent, url, &dir.join(name), on_progress)?;
    }
    Ok(entrypoint)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hf(uri: &str) -> HfRef {
        parse_hf_uri(uri).expect("hf uri should parse")
    }

    // ---- parse_hf_uri ----

    #[test]
    fn parse_hf_uri_defaults() {
        let r = hf("hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf");
        assert_eq!(r.user, "ggml-org");
        assert_eq!(r.model, "embeddinggemma-300M-GGUF");
        assert_eq!(r.file, "embeddinggemma-300M-Q8_0.gguf");
        assert_eq!(r.branch, "main");
        assert_eq!(r.repo(), "ggml-org/embeddinggemma-300M-GGUF");
    }

    #[test]
    fn parse_hf_uri_branch_and_subdirs() {
        let r = hf("hf:u/m/dir/sub/model.gguf#dev");
        assert_eq!(r.file, "dir/sub/model.gguf");
        assert_eq!(r.branch, "dev");
        // Empty branch falls back to main (node-llama-cpp behavior).
        assert_eq!(hf("hf:u/m/model.gguf#").branch, "main");
    }

    #[test]
    fn parse_hf_uri_rejects() {
        assert_eq!(parse_hf_uri("hf:u/m"), None);
        assert_eq!(parse_hf_uri("hf:u/m/"), None);
        assert_eq!(parse_hf_uri("hf:/m/f.gguf"), None);
        assert_eq!(parse_hf_uri("/abs/path.gguf"), None);
        assert_eq!(parse_hf_uri("https://x/y.gguf"), None);
    }

    // ---- hf_cache_filename: real on-disk names ----

    #[test]
    fn cache_name_matches_observed_files() {
        // The model cache on disk: repo dropped because the filename starts
        // with the model name (sans -gguf).
        assert_eq!(
            hf_cache_filename(&hf(
                "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf"
            )),
            "hf_ggml-org_embeddinggemma-300M-Q8_0.gguf"
        );
        assert_eq!(
            hf_cache_filename(&hf(
                "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf"
            )),
            "hf_ggml-org_qwen3-reranker-0.6b-q8_0.gguf"
        );
        assert_eq!(
            hf_cache_filename(&hf(
                "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf"
            )),
            "hf_tobil_qmd-query-expansion-1.7B-q4_k_m.gguf"
        );
    }

    #[test]
    fn cache_name_model_kept_when_filename_differs() {
        // Model stem (sans -gguf, case-insensitive) prefix of the filename
        // means the repo is dropped from the prefix.
        assert_eq!(
            hf_cache_filename(&hf("hf:unsloth/Qwen3-GGUF/qwen3-q4_k_m.gguf")),
            "hf_unsloth_qwen3-q4_k_m.gguf"
        );
        // Filename that does NOT start with the model stem keeps the repo.
        assert_eq!(
            hf_cache_filename(&hf("hf:o/m-gguf/other-q8_0.gguf")),
            "hf_o_m-gguf_other-q8_0.gguf"
        );
    }

    #[test]
    fn cache_name_branch() {
        // Non-main branch forces model + branch into the prefix.
        assert_eq!(
            hf_cache_filename(&hf("hf:o/m-gguf/m-q8_0.gguf#v2")),
            "hf_o_m-gguf_v2_m-q8_0.gguf"
        );
    }

    #[test]
    fn cache_name_subdirs() {
        // Plain subdirectories fold in with `__` separators.
        assert_eq!(
            hf_cache_filename(&hf("hf:o/m-gguf/a/b/m-q8_0.gguf")),
            "hf_o_a__b_m-q8_0.gguf"
        );
        // A directory whose name equals the filename stem is dropped.
        assert_eq!(
            hf_cache_filename(&hf("hf:o/m-gguf/m-q8_0/m-q8_0.gguf")),
            "hf_o_m-q8_0.gguf"
        );
    }

    #[test]
    fn cache_name_named_split_parts() {
        let names = hf_cache_filenames(&hf("hf:o/m-gguf/m-00002-of-00003.gguf"));
        assert_eq!(
            names,
            vec![
                "hf_o_m-00001-of-00003.gguf",
                "hf_o_m-00002-of-00003.gguf",
                "hf_o_m-00003-of-00003.gguf"
            ]
        );
    }

    #[test]
    fn cache_name_binary_split_combines() {
        assert!(is_binary_split_filename("m.gguf.part1of9"));
        assert!(!is_binary_split_filename("m-00001-of-00009.gguf"));
        let names = hf_cache_filenames(&hf("hf:o/m-gguf/m.gguf.part1of9"));
        assert_eq!(names, vec!["hf_o_m.gguf".to_owned()]);
    }

    // ---- download urls ----

    #[test]
    fn download_url_main() {
        assert_eq!(
            hf_download_url(&hf("hf:o/m/f.gguf"), "https://huggingface.co/"),
            "https://huggingface.co/o/m/resolve/main/f.gguf?download=true"
        );
        assert_eq!(
            hf_download_url(&hf("hf:o/m/d/f.gguf#b2"), "https://hf-mirror.com/"),
            "https://hf-mirror.com/o/m/resolve/b2/d/f.gguf?download=true"
        );
    }

    #[test]
    fn download_urls_named_split() {
        let urls = hf_download_urls(
            &hf("hf:o/m/m-00002-of-00002.gguf"),
            "https://huggingface.co/",
        );
        assert_eq!(
            urls,
            vec![
                "https://huggingface.co/o/m/resolve/main/m-00001-of-00002.gguf?download=true",
                "https://huggingface.co/o/m/resolve/main/m-00002-of-00002.gguf?download=true"
            ]
        );
    }

    #[test]
    fn download_urls_binary_split() {
        let urls = hf_download_urls(&hf("hf:o/m/m.gguf.part2of3"), "https://huggingface.co/");
        assert_eq!(
            urls,
            vec![
                "https://huggingface.co/o/m/resolve/main/m.gguf.part1of3?download=true",
                "https://huggingface.co/o/m/resolve/main/m.gguf.part2of3?download=true",
                "https://huggingface.co/o/m/resolve/main/m.gguf.part3of3?download=true"
            ]
        );
    }

    // ---- endpoints ----

    #[test]
    fn endpoint_precedence() {
        let default = Environment::default();
        assert_eq!(resolve_hf_endpoint(&default), "https://huggingface.co/");
        let hf_only = Environment {
            hf_endpoint: Some("https://hf-mirror.com".to_owned()),
            ..Default::default()
        };
        assert_eq!(resolve_hf_endpoint(&hf_only), "https://hf-mirror.com/");
        let both = Environment {
            hf_endpoint: Some("https://hf-mirror.com".to_owned()),
            model_endpoint: Some("https://models.internal/".to_owned()),
            ..Default::default()
        };
        assert_eq!(resolve_hf_endpoint(&both), "https://models.internal/");
        // Non-URL endpoints are ignored like upstream isUrl checks.
        let invalid = Environment {
            model_endpoint: Some("not-a-url".to_owned()),
            ..Default::default()
        };
        assert_eq!(resolve_hf_endpoint(&invalid), "https://huggingface.co/");
    }

    // ---- filenamify ----

    #[test]
    fn filenamify_cases() {
        assert_eq!(
            filenamify("embeddinggemma-300M-Q8_0.gguf"),
            "embeddinggemma-300M-Q8_0.gguf"
        );
        assert_eq!(filenamify("foo/bar"), "foo!bar");
        assert_eq!(filenamify("a<b>c:d"), "a!b!c!d");
        // npm filenamify preserves leading dots (the pad block is dead code).
        assert_eq!(filenamify("...dots"), "...dots");
        assert_eq!(filenamify("trailing."), "trailing");
        assert_eq!(filenamify("con"), "con!");
        assert!(filenamify(&"x".repeat(150)).chars().count() <= 100);
    }

    // ---- split parts ----

    #[test]
    fn split_parts_info() {
        assert_eq!(
            get_gguf_split_parts_info("m-00001-of-00003.gguf"),
            Some((1, 3, 20))
        );
        assert_eq!(get_gguf_split_parts_info("m-00000-of-00003.gguf"), None);
        assert_eq!(get_gguf_split_parts_info("m-00004-of-00003.gguf"), None);
        assert_eq!(get_gguf_split_parts_info("m.gguf"), None);
        assert_eq!(
            create_split_part_filename("hf_u_m-00001-of-00003.gguf", 2, 3),
            "hf_u_m-00002-of-00003.gguf"
        );
        assert_eq!(
            resolve_split_gguf_parts("m-00001-of-00002.gguf"),
            vec!["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]
        );
    }

    // ---- inspect_gguf_file ----

    #[test]
    fn inspect_kinds() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("nope.gguf");
        assert_eq!(inspect_gguf_file(&missing).kind, GgufKind::Missing);

        let good = dir.path().join("good.gguf");
        std::fs::write(&good, b"GGUF\x03\0\0\0rest").expect("write");
        let i = inspect_gguf_file(&good);
        assert_eq!(i.kind, GgufKind::Gguf);
        assert!(i.valid);
        assert_eq!(i.magic.as_deref(), Some("GGUF"));

        let html = dir.path().join("page.gguf");
        std::fs::write(&html, b"<!DOCTYPE html><html>proxy</html>").expect("write");
        let html_info = inspect_gguf_file(&html);
        assert_eq!(html_info.kind, GgufKind::Html);
        assert!(!html_info.valid);

        let bad = dir.path().join("bad.gguf");
        std::fs::write(&bad, b"NOPE binary").expect("write");
        let bad_info = inspect_gguf_file(&bad);
        assert_eq!(bad_info.kind, GgufKind::Invalid);
        assert_eq!(bad_info.magic.as_deref(), Some("NOPE"));

        let nonprint = dir.path().join("np.gguf");
        std::fs::write(&nonprint, [0x01u8, 0x02, 0x03, 0x04]).expect("write");
        let np_info = inspect_gguf_file(&nonprint);
        assert_eq!(np_info.kind, GgufKind::Invalid);
        assert_eq!(np_info.magic.as_deref(), Some("0x01020304"));
    }

    // Upstream `llm.test.ts:100-112` — a path that exists but cannot be
    // read as a file is `unreadable`, not `invalid`.
    #[test]
    fn inspect_unreadable_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let not_a_file = dir.path().join("actually-a-directory.gguf");
        std::fs::create_dir(&not_a_file).expect("mkdir");
        let i = inspect_gguf_file(&not_a_file);
        assert!(i.exists);
        assert!(!i.valid);
        assert_eq!(i.kind, GgufKind::Unreadable);
        assert_eq!(i.magic, None);
        assert!(
            i.details.contains("cannot read model file"),
            "details: {}",
            i.details
        );
    }

    #[test]
    fn file_size_format() {
        assert_eq!(format_model_file_size(0), "0 KB");
        assert_eq!(format_model_file_size(1024), "1 KB");
        // JS toFixed(0) rounds half away from zero.
        assert_eq!(format_model_file_size(2560), "3 KB");
        assert_eq!(format_model_file_size(1_282_438_912), "1252382 KB");
    }
}
