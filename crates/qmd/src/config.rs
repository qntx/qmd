//! YAML configuration model, matching upstream `src/collections.ts`.
//!
//! Unknown keys, key order and all `editor_uri` spelling variants are
//! preserved across load/save so a round-trip stays readable for humans and
//! remains compatible with upstream `qmd`.

use std::fmt::Write;
use std::path::Path;

use indexmap::IndexMap;
use serde::{Deserialize, Deserializer, Serialize};

use crate::error::{ConfigError, Error, Result};

/// Default glob pattern upstream applies when none is configured.
pub const DEFAULT_PATTERN: &str = "**/*.md";

/// YAML value type used for unknown keys.
pub type YamlValue = serde_yaml_bw::Value;

/// YAML `null` (or missing) maps to `None`, not a parse error — upstream
/// treats `context:` with no value as absent.
fn de_nullable_map<'de, D, T>(de: D) -> Result<Option<IndexMap<String, T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<IndexMap<String, T>>::deserialize(de)
}

/// Same for a required map field: `collections:` empty or missing means
/// an empty map.
fn de_map<'de, D, T>(de: D) -> Result<IndexMap<String, T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Ok(de_nullable_map(de)?.unwrap_or_default())
}

/// Embedding pooling strategy (Rust-only extension key `models.embed_pooling`).
///
/// Upstream always uses `upstream_tail` (last position of the embedding
/// output). `full_sequence` is a repair path for models whose tail position
/// is not meaningful; it is never emitted unless explicitly configured.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum EmbedPooling {
    /// Upstream-compatible tail-only embedding (default, never emitted).
    #[default]
    UpstreamTail,
    /// Pool over the full sequence (Rust extension, opt-in repair path).
    FullSequence,
}

impl EmbedPooling {
    /// The extra line this mode contributes to
    /// [`crate::llm::embedding_fingerprint`] (architecture §8.3):
    /// `UpstreamTail` is upstream-identical (`None`); `FullSequence`
    /// switches the fingerprint so toggling it triggers re-embedding.
    #[must_use]
    pub const fn fingerprint_extra(self) -> Option<&'static str> {
        match self {
            Self::UpstreamTail => None,
            Self::FullSequence => Some("pooling:full"),
        }
    }
}

/// Model selection overrides (`models:` block).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelsConfig {
    /// Embedding model override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed: Option<String>,
    /// Generation model override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generate: Option<String>,
    /// Reranker model override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank: Option<String>,
    /// Rust-only extension, see [`EmbedPooling`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed_pooling: Option<EmbedPooling>,
    /// Unknown keys preserved verbatim.
    #[serde(flatten)]
    pub extra: IndexMap<String, YamlValue>,
}

/// A single collection entry (`collections.<name>`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Collection {
    /// Filesystem path to index (verbatim string, resolved by the CLI).
    pub path: String,
    /// Glob pattern, defaults to [`DEFAULT_PATTERN`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
    /// Additional ignore patterns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,
    /// Per-path context strings keyed by path prefix.
    #[serde(
        default,
        deserialize_with = "de_nullable_map",
        skip_serializing_if = "Option::is_none"
    )]
    pub context: Option<IndexMap<String, String>>,
    /// Shell command to run before re-indexing (upstream `update`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub update: Option<String>,
    /// Whether the collection is included in unscoped queries (default true).
    #[serde(
        rename = "includeByDefault",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub include_by_default: Option<bool>,
    /// Unknown keys preserved verbatim.
    #[serde(flatten)]
    pub extra: IndexMap<String, YamlValue>,
}

impl Collection {
    /// Effective glob pattern (upstream `coll.pattern ?? "**/*.md"`).
    #[must_use]
    pub fn pattern_or_default(&self) -> &str {
        self.pattern.as_deref().unwrap_or(DEFAULT_PATTERN)
    }

    /// Effective include-by-default flag (upstream default is `true`).
    #[must_use]
    pub fn include_by_default(&self) -> bool {
        self.include_by_default.unwrap_or(true)
    }

    /// Ignore patterns or empty slice.
    #[must_use]
    pub fn ignore_patterns(&self) -> &[String] {
        self.ignore.as_deref().unwrap_or(&[])
    }
}

/// Root configuration (`.qmd/index.yaml` / `<config-dir>/<name>.yml`).
///
/// Upstream accepts all four `editor_uri` spellings; each is stored in its
/// own field and written back under its original spelling.
///
/// Serialization order is `collections, models, global_context, editor_*`:
/// upstream's `saveConfig` emits JS-object insertion order, so keys a
/// mutation adds later (`models:` pinned by `getStore`, `global_context`
/// set by `context add /`) land at the end of the file. A config that
/// *starts* with `global_context` keeps it first upstream; our fixed order
/// moves it — accepted deviation: the reorder only shows up on the first
/// mutation save of such a file, and key order carries no semantics.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    /// Named collections, in file order.
    #[serde(default, deserialize_with = "de_map")]
    pub collections: IndexMap<String, Collection>,
    /// `models:` block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<ModelsConfig>,
    /// Fallback context applied to every collection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_context: Option<String>,
    /// `editor_uri` spelling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub editor_uri: Option<String>,
    /// `editor_uri_template` spelling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub editor_uri_template: Option<String>,
    /// `editorUri` spelling.
    #[serde(rename = "editorUri", default, skip_serializing_if = "Option::is_none")]
    pub editor_uri_camel: Option<String>,
    /// `editor-uri` spelling.
    #[serde(
        rename = "editor-uri",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub editor_uri_kebab: Option<String>,
    /// Unknown keys preserved verbatim.
    #[serde(flatten)]
    pub extra: IndexMap<String, YamlValue>,
}

impl Config {
    /// Effective editor URI: upstream `loadConfig` checks
    /// `editor_uri`, `editor_uri_template`, `editorUri`, `editor-uri` in
    /// that order and the first present wins.
    #[must_use]
    pub fn editor_uri_value(&self) -> Option<&str> {
        self.editor_uri
            .as_deref()
            .or(self.editor_uri_template.as_deref())
            .or(self.editor_uri_camel.as_deref())
            .or(self.editor_uri_kebab.as_deref())
    }
}

/// Load a YAML config file. A missing or empty file yields an empty
/// [`Config`], matching upstream `loadConfig` behavior.
///
/// # Errors
/// [`Error::Io`] on unreadable files, [`Error::Config`] on malformed YAML.
pub fn load(path: &Path) -> Result<Config> {
    let content = match std::fs::read_to_string(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Config::default());
        }
        Err(source) => {
            return Err(Error::Io {
                path: path.to_path_buf(),
                source,
            });
        }
        Ok(c) => c,
    };
    if content.trim().is_empty() {
        return Ok(Config::default());
    }
    // A document that is literally `null` (or `~`) also maps to an empty
    // config, like upstream `parsed ?? { collections: {} }`.
    let parsed: Option<Config> =
        serde_yaml_bw::from_str(&content).map_err(|e| config_err(path, e))?;
    Ok(parsed.unwrap_or_default())
}

/// Serialize and write a config file, creating parent directories.
///
/// # Errors
/// [`Error::Config`] on serialization failure, [`Error::Io`] on write
/// failure.
pub fn save(path: &Path, config: &Config) -> Result<()> {
    let content = stringify(config).map_err(|e| config_err(path, e))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| Error::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    std::fs::write(path, content).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Serialize to the byte style upstream's `YAML.stringify(config,
/// {indent: 2, lineWidth: 0})` produces (collections.ts `saveConfig`,
/// yaml@2.x): block sequences indent one level under their key, empty
/// collections flow as `{}`/`[]`, and scalars that cannot stay plain
/// are double-quoted. `serde_yaml_bw`'s emitter differs on both —
/// indentless sequences and single-quoted scalars — so the tree walks
/// through `Value` here instead.
fn stringify(config: &Config) -> std::result::Result<String, serde_yaml_bw::Error> {
    let value = serde_yaml_bw::to_value(config)?;
    let mut out = String::new();
    emit_node(&mut out, &value, 0);
    Ok(out)
}

/// Block-context dispatch: a root (or nested) mapping/sequence opens
/// its lines at `indent`; scalars occupy one line.
fn emit_node(out: &mut String, value: &YamlValue, indent: usize) {
    match value {
        YamlValue::Mapping(map) => emit_mapping(out, map, indent),
        YamlValue::Sequence(seq) => emit_sequence(out, seq, indent),
        scalar => {
            push_indent(out, indent);
            out.push_str(&scalar_text(scalar));
            out.push('\n');
        }
    }
}

fn emit_mapping(out: &mut String, map: &serde_yaml_bw::Mapping, indent: usize) {
    if map.is_empty() {
        push_indent(out, indent);
        out.push_str("{}\n");
        return;
    }
    for (key, value) in map {
        push_indent(out, indent);
        out.push_str(&scalar_text(key));
        out.push(':');
        emit_value_after_key(out, value, indent);
    }
}

fn emit_sequence(out: &mut String, seq: &serde_yaml_bw::Sequence, indent: usize) {
    if seq.elements.is_empty() {
        push_indent(out, indent);
        out.push_str("[]\n");
        return;
    }
    for item in &seq.elements {
        push_indent(out, indent);
        match item {
            YamlValue::Mapping(map) if map.is_empty() => out.push_str("- {}\n"),
            YamlValue::Sequence(inner) if inner.elements.is_empty() => {
                out.push_str("- []\n");
            }
            // `- key: v` — the first entry of a map item shares the dash
            // line, like yaml@2's emitter.
            YamlValue::Mapping(map) if !map.is_empty() => {
                let mut entries = map.iter();
                if let Some((key, value)) = entries.next() {
                    out.push_str("- ");
                    out.push_str(&scalar_text(key));
                    out.push(':');
                    emit_value_after_key(out, value, indent + 1);
                }
                for (key, value) in entries {
                    push_indent(out, indent + 1);
                    out.push_str(&scalar_text(key));
                    out.push(':');
                    emit_value_after_key(out, value, indent + 1);
                }
            }
            YamlValue::Sequence(inner) if !inner.elements.is_empty() => {
                // `-\n  - item` — nested block one level deeper.
                out.push('-');
                out.push('\n');
                emit_sequence(out, inner, indent + 1);
            }
            scalar => {
                out.push_str("- ");
                out.push_str(&scalar_text(scalar));
                out.push('\n');
            }
        }
    }
}

/// The part after `key:` on a line — flow-style empties stay inline;
/// block collections open on the next line one level deeper.
fn emit_value_after_key(out: &mut String, value: &YamlValue, indent: usize) {
    match value {
        YamlValue::Mapping(map) if map.is_empty() => out.push_str(" {}\n"),
        YamlValue::Sequence(seq) if seq.elements.is_empty() => out.push_str(" []\n"),
        YamlValue::Mapping(map) => {
            out.push('\n');
            emit_mapping(out, map, indent + 1);
        }
        YamlValue::Sequence(seq) => {
            out.push('\n');
            emit_sequence(out, seq, indent + 1);
        }
        scalar => {
            out.push(' ');
            out.push_str(&scalar_text(scalar));
            out.push('\n');
        }
    }
}

fn push_indent(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push_str("  ");
    }
}

/// Scalar text for keys and values — plain when the YAML core schema
/// would read it back as the same string, double-quoted otherwise.
fn scalar_text(value: &YamlValue) -> String {
    match value {
        YamlValue::Null(_) => "null".to_owned(),
        YamlValue::Bool(b, _) => if *b { "true" } else { "false" }.to_owned(),
        YamlValue::Number(n, _) => n.to_string(),
        YamlValue::String(s, _) => {
            if plain_safe(s) {
                s.clone()
            } else {
                double_quoted(s)
            }
        }
        // `*alias` is dropped — config files never carry anchors. A
        // `!tag` (serde_yaml_bw's enum serialization) keeps its marker.
        YamlValue::Tagged(tagged) => {
            format!("!{} {}", tagged.tag, scalar_text(&tagged.value))
        }
        YamlValue::Alias(_) | YamlValue::Sequence(_) | YamlValue::Mapping(_) => String::new(),
    }
}

/// Whether `s` may be emitted unquoted and still parse back as the same
/// string under the YAML core schema — the conservative half of
/// yaml@2's `isPlainSafe` plus its resolve-as-string check.
fn plain_safe(s: &str) -> bool {
    if s.is_empty() || s != s.trim() {
        return false;
    }
    let bytes = s.as_bytes();
    let Some(&first) = bytes.first() else {
        return false;
    };
    if matches!(
        first,
        b'&' | b'*'
            | b'!'
            | b'|'
            | b'>'
            | b'\''
            | b'"'
            | b'%'
            | b'@'
            | b'`'
            | b'#'
            | b'{'
            | b'}'
            | b'['
            | b']'
            | b','
    ) {
        return false;
    }
    if matches!(first, b'-' | b'?' | b':') && (bytes.len() == 1 || bytes.get(1) == Some(&b' ')) {
        return false;
    }
    if s.starts_with("---") || s.starts_with("...") || s.ends_with(':') {
        return false;
    }
    if s.contains(": ") || s.contains(" #") || s.contains(['\n', '\r', '\t']) {
        return false;
    }
    !resolves_non_string(s)
}

/// Core-schema resolution: a plain `null`/`true`/`false`/number-looking
/// string must be quoted or it parses back as a non-string.
fn resolves_non_string(s: &str) -> bool {
    if matches!(
        s,
        "null" | "Null" | "NULL" | "~" | "true" | "false" | "True" | "False" | "TRUE" | "FALSE"
    ) {
        return true;
    }
    let body = s.strip_prefix(['-', '+']).unwrap_or(s);
    if body.is_empty() {
        return false;
    }
    // ints (decimal, hex, octal, binary) and floats, per core schema.
    if body.bytes().all(|b| b.is_ascii_digit()) {
        return true;
    }
    if let Some(hex) = body.strip_prefix("0x") {
        return !hex.is_empty() && hex.bytes().all(|b| b.is_ascii_hexdigit());
    }
    if let Some(oct) = body.strip_prefix("0o") {
        return !oct.is_empty() && oct.bytes().all(|b| matches!(b, b'0'..=b'7'));
    }
    if let Some(bin) = body.strip_prefix("0b") {
        return !bin.is_empty() && bin.bytes().all(|b| matches!(b, b'0' | b'1'));
    }
    if matches!(body, ".inf" | ".Inf" | ".INF" | ".nan" | ".NaN" | ".NAN") {
        return true;
    }
    // `1.`, `1.5`, `.5`, `1e3`, `1.5e-3` — digits with an optional dot
    // and exponent.
    let mut seen_digit = false;
    let mut seen_dot = false;
    let mut seen_exp = false;
    let mut chars = body.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '0'..='9' => seen_digit = true,
            '.' if !seen_dot && !seen_exp => seen_dot = true,
            'e' | 'E' if seen_digit && !seen_exp => {
                seen_exp = true;
                if matches!(chars.peek(), Some('-' | '+')) {
                    chars.next();
                }
                if !matches!(chars.peek(), Some('0'..='9')) {
                    return false;
                }
            }
            _ => return false,
        }
    }
    seen_digit
}

/// yaml@2 double-quoted style: `"`/`\`/control characters escaped.
fn double_quoted(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 || u32::from(c) == 0x7f => {
                write!(out, "\\x{:02x}", u32::from(c)).ok();
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn config_err(path: &Path, e: serde_yaml_bw::Error) -> Error {
    Error::Config {
        path: path.to_path_buf(),
        source: ConfigError::from(e),
    }
}
