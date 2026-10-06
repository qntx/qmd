//! YAML configuration model, matching upstream `src/collections.ts`.
//!
//! Unknown keys, key order and all `editor_uri` spelling variants are
//! preserved across load/save so a round-trip stays readable for humans and
//! remains compatible with upstream `qmd`.

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

/// Model selection overrides (`models:` block).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelsConfig {
    /// Embedding model override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed: Option<String>,
    /// Reranker model override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank: Option<String>,
    /// Generation model override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generate: Option<String>,
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

/// Root configuration (`qmd.yml` / `index-<name>.yml`).
///
/// Upstream accepts all four `editor_uri` spellings; each is stored in its
/// own field and written back under its original spelling.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
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
    /// Named collections, in file order.
    #[serde(default, deserialize_with = "de_map")]
    pub collections: IndexMap<String, Collection>,
    /// `models:` block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<ModelsConfig>,
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
    let content = serde_yaml_bw::to_string(config).map_err(|e| config_err(path, e))?;
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

fn config_err(path: &Path, e: serde_yaml_bw::Error) -> Error {
    Error::Config {
        path: path.to_path_buf(),
        source: ConfigError::from(e),
    }
}
