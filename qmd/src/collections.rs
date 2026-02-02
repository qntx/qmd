//! Collections configuration management.
//!
//! This module manages the YAML-based collection configuration at `~/.config/qmd/index.yml`.
//! Collections define which directories to index and their associated contexts.

use crate::config::{get_config_dir, get_config_path};
use crate::error::{QmdError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;

/// Context definitions for a collection.
/// Key is path prefix (e.g., "/", "/2024", "/Board of Directors").
/// Value is the context description.
pub type ContextMap = BTreeMap<String, String>;

/// A single collection configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    /// Absolute path to index.
    pub path: String,
    /// Glob pattern (e.g., "**/*.md").
    pub pattern: String,
    /// Optional context definitions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<ContextMap>,
    /// Optional bash command to run during qmd update.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub update: Option<String>,
}

/// The complete configuration file structure.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Context applied to all collections.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub global_context: Option<String>,
    /// Collection name -> config.
    #[serde(default)]
    pub collections: BTreeMap<String, Collection>,
}

/// Collection with its name (for return values).
#[derive(Debug, Clone)]
pub struct NamedCollection {
    /// Collection name.
    pub name: String,
    /// Absolute path to index.
    pub path: String,
    /// Glob pattern.
    pub pattern: String,
    /// Optional context definitions.
    pub context: Option<ContextMap>,
    /// Optional update command.
    pub update: Option<String>,
}

impl From<(String, Collection)> for NamedCollection {
    fn from((name, coll): (String, Collection)) -> Self {
        Self {
            name,
            path: coll.path,
            pattern: coll.pattern,
            context: coll.context,
            update: coll.update,
        }
    }
}

/// Current index name (default: "index").
static INDEX_NAME: std::sync::RwLock<String> = std::sync::RwLock::new(String::new());

/// Set the current index name for config file lookup.
pub fn set_config_index_name(name: &str) {
    if let Ok(mut index_name) = INDEX_NAME.write() {
        *index_name = name.to_string();
    }
}

/// Get current index name.
fn get_index_name() -> String {
    INDEX_NAME.read().map_or_else(
        |_| "index".to_string(),
        |s| {
            if s.is_empty() {
                "index".to_string()
            } else {
                s.clone()
            }
        },
    )
}

/// Ensure config directory exists.
fn ensure_config_dir() -> Result<()> {
    if let Some(config_dir) = get_config_dir() {
        fs::create_dir_all(&config_dir)?;
    }
    Ok(())
}

/// Load configuration from ~/.config/qmd/index.yml.
/// Returns empty config if file doesn't exist.
pub fn load_config() -> Result<CollectionConfig> {
    let index_name = get_index_name();
    let config_path = get_config_path(&index_name)
        .ok_or_else(|| QmdError::Config("Could not determine config path".to_string()))?;

    if !config_path.exists() {
        return Ok(CollectionConfig::default());
    }

    let content = fs::read_to_string(&config_path)?;
    let config: CollectionConfig = serde_yaml::from_str(&content)?;
    Ok(config)
}

/// Save configuration to ~/.config/qmd/index.yml.
pub fn save_config(config: &CollectionConfig) -> Result<()> {
    ensure_config_dir()?;
    let index_name = get_index_name();
    let config_path = get_config_path(&index_name)
        .ok_or_else(|| QmdError::Config("Could not determine config path".to_string()))?;

    let yaml = serde_yaml::to_string(config)?;
    fs::write(&config_path, yaml)?;
    Ok(())
}

/// Get a specific collection by name.
pub fn get_collection(name: &str) -> Result<Option<NamedCollection>> {
    let config = load_config()?;
    Ok(config
        .collections
        .get(name)
        .cloned()
        .map(|c| NamedCollection::from((name.to_string(), c))))
}

/// List all collections.
pub fn list_collections() -> Result<Vec<NamedCollection>> {
    let config = load_config()?;
    Ok(config
        .collections
        .into_iter()
        .map(NamedCollection::from)
        .collect())
}

/// Add or update a collection.
pub fn add_collection(name: &str, path: &str, pattern: &str) -> Result<()> {
    let mut config = load_config()?;

    let existing_context = config.collections.get(name).and_then(|c| c.context.clone());

    config.collections.insert(
        name.to_string(),
        Collection {
            path: path.to_string(),
            pattern: pattern.to_string(),
            context: existing_context,
            update: None,
        },
    );

    save_config(&config)
}

/// Remove a collection.
pub fn remove_collection(name: &str) -> Result<bool> {
    let mut config = load_config()?;
    let removed = config.collections.remove(name).is_some();
    if removed {
        save_config(&config)?;
    }
    Ok(removed)
}

/// Rename a collection.
pub fn rename_collection(old_name: &str, new_name: &str) -> Result<bool> {
    let mut config = load_config()?;

    let Some(collection) = config.collections.remove(old_name) else {
        return Ok(false);
    };

    if config.collections.contains_key(new_name) {
        return Err(QmdError::Config(format!(
            "Collection '{new_name}' already exists"
        )));
    }

    config.collections.insert(new_name.to_string(), collection);
    save_config(&config)?;
    Ok(true)
}

/// Get global context.
pub fn get_global_context() -> Result<Option<String>> {
    let config = load_config()?;
    Ok(config.global_context)
}

/// Set global context.
pub fn set_global_context(context: Option<&str>) -> Result<()> {
    let mut config = load_config()?;
    config.global_context = context.map(str::to_string);
    save_config(&config)
}

/// Add or update a context for a specific path in a collection.
pub fn add_context(collection_name: &str, path_prefix: &str, context_text: &str) -> Result<bool> {
    let mut config = load_config()?;

    let Some(collection) = config.collections.get_mut(collection_name) else {
        return Ok(false);
    };

    let context_map = collection.context.get_or_insert_with(BTreeMap::new);
    context_map.insert(path_prefix.to_string(), context_text.to_string());

    save_config(&config)?;
    Ok(true)
}

/// Remove a context from a collection.
pub fn remove_context(collection_name: &str, path_prefix: &str) -> Result<bool> {
    let mut config = load_config()?;

    let Some(collection) = config.collections.get_mut(collection_name) else {
        return Ok(false);
    };

    let Some(ref mut context_map) = collection.context else {
        return Ok(false);
    };

    let removed = context_map.remove(path_prefix).is_some();

    if context_map.is_empty() {
        collection.context = None;
    }

    if removed {
        save_config(&config)?;
    }
    Ok(removed)
}

/// Context entry for listing.
#[derive(Debug, Clone)]
pub struct ContextEntry {
    /// Collection name ("*" for global).
    pub collection: String,
    /// Path prefix.
    pub path: String,
    /// Context description.
    pub context: String,
}

/// List all contexts across all collections.
pub fn list_all_contexts() -> Result<Vec<ContextEntry>> {
    let config = load_config()?;
    let mut results = Vec::new();

    // Add global context if present.
    if let Some(ref global) = config.global_context {
        results.push(ContextEntry {
            collection: "*".to_string(),
            path: "/".to_string(),
            context: global.clone(),
        });
    }

    // Add collection contexts.
    for (name, collection) in &config.collections {
        if let Some(ref context_map) = collection.context {
            for (path, context) in context_map {
                results.push(ContextEntry {
                    collection: name.clone(),
                    path: path.clone(),
                    context: context.clone(),
                });
            }
        }
    }

    Ok(results)
}

/// Find best matching context for a given collection and path.
/// Returns the most specific matching context (longest path prefix match).
pub fn find_context_for_path(collection_name: &str, file_path: &str) -> Result<Option<String>> {
    let config = load_config()?;

    let Some(collection) = config.collections.get(collection_name) else {
        return Ok(config.global_context);
    };

    let Some(ref context_map) = collection.context else {
        return Ok(config.global_context);
    };

    // Find all matching prefixes.
    let mut matches: Vec<(&str, &str)> = Vec::new();

    for (prefix, context) in context_map {
        let normalized_path = if file_path.starts_with('/') {
            file_path.to_string()
        } else {
            format!("/{file_path}")
        };
        let normalized_prefix = if prefix.starts_with('/') {
            prefix.clone()
        } else {
            format!("/{prefix}")
        };

        if normalized_path.starts_with(&normalized_prefix) {
            matches.push((prefix.as_str(), context.as_str()));
        }
    }

    // Return most specific match (longest prefix).
    if !matches.is_empty() {
        matches.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        return Ok(Some(matches[0].1.to_string()));
    }

    // Fallback to global context.
    Ok(config.global_context)
}

/// Get the config file path (useful for error messages).
#[must_use]
pub fn get_config_file_path() -> Option<std::path::PathBuf> {
    let index_name = get_index_name();
    get_config_path(&index_name)
}

/// Check if config file exists.
#[must_use]
pub fn config_exists() -> bool {
    get_config_file_path().is_some_and(|p| p.exists())
}

/// Validate a collection name.
/// Collection names must be valid and not contain special characters.
#[must_use]
pub fn is_valid_collection_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}
