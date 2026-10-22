//! Frontmatter `qmd.metadata` extraction — port of upstream
//! `src/metadata.ts`.
//!
//! Covers `extractDocumentMetadata`, `normalizeMetadata`,
//! `truncateErrorMessage`, `METADATA_LIMITS` and
//! `METADATA_EXTRACTION_VERSION`. Documents opt into metadata through a
//! namespaced Markdown frontmatter block (`qmd: > metadata:`). The raw
//! document is never modified — frontmatter stays part of the stored,
//! indexed, chunked and embedded content. YAML composition lives in
//! the private `yaml` submodule, which mirrors the npm `yaml` 2.9 core
//! schema used upstream. The `filter` submodule ports upstream
//! `src/metadata-filter.ts` (P3.2).

pub mod filter;

mod yaml;

use indexmap::IndexMap;

use crate::chunk::{is_js_ws, js_trim, utf16_len};

/// Upstream `METADATA_EXTRACTION_VERSION` (metadata.ts:61). Bump when
/// extraction semantics change so stored rows are re-extracted.
pub const METADATA_EXTRACTION_VERSION: i64 = 1;

/// Upstream `METADATA_LIMITS` (metadata.ts:64-72) — defensive limits for
/// metadata from untrusted repositories.
#[derive(Debug, Clone, Copy)]
pub struct MetadataLimits {
    /// UTF-8 byte cap on the frontmatter slice.
    pub max_frontmatter_bytes: usize,
    /// Maximum keys in `qmd.metadata`.
    pub max_keys: usize,
    /// UTF-8 byte cap per key.
    pub max_key_bytes: usize,
    /// UTF-16 code-unit cap per string (JS `length` semantics).
    pub max_string_length: usize,
    /// Maximum elements per array.
    pub max_array_length: usize,
    /// npm `yaml` `maxAliasCount` — billion-laughs guard.
    pub max_yaml_alias_count: usize,
    /// UTF-16 code-unit cap on extraction error messages.
    pub max_error_length: usize,
}

/// Upstream `METADATA_LIMITS` values (metadata.ts:64-72).
pub const METADATA_LIMITS: MetadataLimits = MetadataLimits {
    max_frontmatter_bytes: 64 * 1024,
    max_keys: 64,
    max_key_bytes: 128,
    max_string_length: 1024,
    max_array_length: 128,
    max_yaml_alias_count: 100,
    max_error_length: 200,
};

/// File extensions parsed for frontmatter metadata
/// (`FRONTMATTER_FILE_EXTENSIONS`, metadata.ts:75).
const FRONTMATTER_FILE_EXTENSIONS: [&str; 3] = [".md", ".markdown", ".mdx"];

/// One metadata value — upstream `MetadataValue` (metadata.ts:29-36):
/// a scalar or a homogeneous array of one scalar type.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// `string`
    String(String),
    /// `number` — finite only.
    Number(f64),
    /// `boolean`
    Bool(bool),
    /// `string[]`
    Strings(Vec<String>),
    /// `number[]`
    Numbers(Vec<f64>),
    /// `boolean[]`
    Bools(Vec<bool>),
}

/// Canonical document metadata — upstream `DocumentMetadata`.
///
/// Entries keep JavaScript object key order: canonical array-index keys
/// first in ascending order, then the rest in document insertion order
/// (metadata.ts:38).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DocumentMetadata {
    entries: IndexMap<String, MetadataValue>,
}

impl DocumentMetadata {
    /// Upstream `metadata[key]`.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.entries.get(key)
    }

    /// Iterate entries in JavaScript object key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &MetadataValue)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Number of keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the map is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `JSON.stringify(metadata)` — byte-identical to upstream,
    /// including JS number formatting and object key order.
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut out = String::from("{");
        for (i, (key, value)) in self.entries.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&json_string(key));
            out.push(':');
            write_json_value(&mut out, value);
        }
        out.push('}');
        out
    }

    /// Build from JS-ordered entries.
    pub(crate) fn from_ordered(entries: Vec<(String, MetadataValue)>) -> Self {
        Self {
            entries: entries.into_iter().collect(),
        }
    }
}

/// Result of [`extract_document_metadata`] — upstream
/// `MetadataExtractionResult` (metadata.ts:47-51).
///
/// `error` is set when the document opted into `qmd.metadata` but the
/// value was invalid; the document still indexes normally.
#[derive(Debug, Clone)]
pub struct MetadataExtraction {
    /// Extracted metadata (empty on failure — all-or-nothing).
    pub metadata: DocumentMetadata,
    /// Bounded extraction error, if the frontmatter was invalid.
    pub error: Option<String>,
    /// Always [`METADATA_EXTRACTION_VERSION`].
    pub extraction_version: i64,
}

/// Extract `qmd.metadata` from a document's leading YAML frontmatter —
/// upstream `extractDocumentMetadata` (metadata.ts:89-134).
///
/// Never fails: no frontmatter, no `qmd` namespace, or a
/// non-frontmatter extension yields empty metadata with no error;
/// invalid input yields empty metadata plus a bounded error.
#[must_use]
pub fn extract_document_metadata(content: &str, path: &str) -> MetadataExtraction {
    let success = |metadata: DocumentMetadata| MetadataExtraction {
        metadata,
        error: None,
        extraction_version: METADATA_EXTRACTION_VERSION,
    };
    let failure = |message: String| MetadataExtraction {
        metadata: DocumentMetadata::default(),
        error: Some(truncate_error_message(&message)),
        extraction_version: METADATA_EXTRACTION_VERSION,
    };

    if !has_frontmatter_extension(path) {
        return success(DocumentMetadata::default());
    }
    let Some(frontmatter_yaml) = frontmatter_yaml(content) else {
        return success(DocumentMetadata::default());
    };
    if frontmatter_yaml.len() > METADATA_LIMITS.max_frontmatter_bytes {
        return failure(format!(
            "frontmatter exceeds {} bytes",
            METADATA_LIMITS.max_frontmatter_bytes
        ));
    }
    let frontmatter = match yaml::parse_frontmatter(frontmatter_yaml) {
        Ok(v) => v,
        Err(message) => return failure(format!("invalid frontmatter YAML: {message}")),
    };
    let yaml::Js::Map(frontmatter_map) = frontmatter else {
        return success(DocumentMetadata::default());
    };
    let Some(qmd_namespace) = frontmatter_map.get("qmd") else {
        return success(DocumentMetadata::default());
    };
    let yaml::Js::Map(qmd_map) = qmd_namespace else {
        return failure("frontmatter 'qmd' must be a mapping".to_owned());
    };
    let Some(raw_metadata) = qmd_map.get("metadata") else {
        return success(DocumentMetadata::default());
    };
    let yaml::Js::Map(raw_map) = raw_metadata else {
        return failure("frontmatter 'qmd.metadata' must be a mapping".to_owned());
    };
    match normalize_metadata(raw_map) {
        Ok(metadata) => success(metadata),
        Err(message) => failure(message),
    }
}

/// Upstream `hasFrontmatterFileExtension` (metadata.ts:136-140):
/// `lastIndexOf('.')`, slice, lowercase, membership check.
fn has_frontmatter_extension(path: &str) -> bool {
    let Some(dot) = path.rfind('.') else {
        return false;
    };
    let ext = path.get(dot..).unwrap_or_default().to_lowercase();
    FRONTMATTER_FILE_EXTENSIONS.contains(&ext.as_str())
}

/// Slice the YAML between a leading `---` line and a closing `---` or
/// `...` line — upstream `getFrontmatterYaml` (metadata.ts:147-159),
/// with JS regex semantics implemented by hand: BOM only at offset 0,
/// opener `^---[ \t]*\r?\n`, closer `^(?:---|\.\.\.)[ \t]*(?:\r?\n|$)`
/// where JS multiline `^`/`$` treat `\n`, `\r`, U+2028 and U+2029 as
/// line terminators.
fn frontmatter_yaml(content: &str) -> Option<&str> {
    let body = content.strip_prefix('\u{feff}').unwrap_or(content);

    let after_dashes = body.strip_prefix("---")?;
    let after_ws = after_dashes.trim_start_matches([' ', '\t']);
    // `\r?\n`: an optional CR must be followed by LF.
    let yaml_start = if let Some(rest) = after_ws.strip_prefix('\n') {
        body.len() - rest.len()
    } else {
        body.len() - after_ws.strip_prefix("\r\n")?.len()
    };

    let slice = &body[yaml_start..];
    let mut pos = 0usize;
    loop {
        if is_closer(&slice[pos..]) {
            return Some(&slice[..pos]);
        }
        // Advance to the first position after the next line terminator —
        // where JS multiline `^` can match.
        let mut next = None;
        for (off, c) in slice[pos..].char_indices() {
            if matches!(c, '\n' | '\r' | '\u{2028}' | '\u{2029}') {
                next = Some(pos + off + c.len_utf8());
                break;
            }
        }
        pos = next?;
    }
}

/// `^(?:---|\.\.\.)[ \t]*(?:\r?\n|$)` at the start of `s`, with JS
/// multiline `$` matching before any JS line terminator.
fn is_closer(s: &str) -> bool {
    let Some(rest) = s.strip_prefix("---").or_else(|| s.strip_prefix("...")) else {
        return false;
    };
    rest.trim_start_matches([' ', '\t'])
        .chars()
        .next()
        .is_none_or(|c| matches!(c, '\n' | '\r' | '\u{2028}' | '\u{2029}'))
}

/// Upstream `normalizeMetadata` (metadata.ts:170-185): key count,
/// key validation, per-value normalization — all-or-nothing.
fn normalize_metadata(raw_map: &yaml::JsMap) -> Result<DocumentMetadata, String> {
    let keys = raw_map.js_ordered_keys();
    if keys.len() > METADATA_LIMITS.max_keys {
        return Err(format!(
            "metadata has {} keys (max {})",
            keys.len(),
            METADATA_LIMITS.max_keys
        ));
    }
    let mut entries = Vec::with_capacity(keys.len());
    for key in keys {
        validate_metadata_key(key)?;
        // `keys` came from `raw_map`, so the lookup cannot miss.
        let Some(value) = raw_map.get(key) else {
            continue;
        };
        entries.push((key.to_owned(), normalize_metadata_value(key, value)?));
    }
    Ok(DocumentMetadata::from_ordered(entries))
}

/// Upstream `validateMetadataKey` (metadata.ts:187-198): non-empty,
/// ≤ 128 UTF-8 bytes, no control characters (U+0000–U+001F, U+007F).
fn validate_metadata_key(key: &str) -> Result<(), String> {
    if key.is_empty() {
        return Err("metadata keys must be non-empty strings".to_owned());
    }
    if key.len() > METADATA_LIMITS.max_key_bytes {
        return Err(format!(
            "metadata key exceeds {} bytes",
            METADATA_LIMITS.max_key_bytes
        ));
    }
    if key.chars().any(|c| c <= '\u{1f}' || c == '\u{7f}') {
        return Err("metadata keys must not contain control characters".to_owned());
    }
    Ok(())
}

/// Upstream `normalizeMetadataValue` (metadata.ts:200-205).
fn normalize_metadata_value(key: &str, value: &yaml::Js) -> Result<MetadataValue, String> {
    match value {
        yaml::Js::Seq(items) => normalize_metadata_array(key, items),
        _ => normalize_metadata_scalar(key, value).map(|s| match s {
            MetadataScalar::String(v) => MetadataValue::String(v),
            MetadataScalar::Number(v) => MetadataValue::Number(v),
            MetadataScalar::Bool(v) => MetadataValue::Bool(v),
        }),
    }
}

/// Resolved scalar — upstream `MetadataScalar` (metadata.ts:26).
/// Public because [`filter::MetadataFilter`] condition operands use it.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataScalar {
    /// `string`
    String(String),
    /// `number`
    Number(f64),
    /// `boolean`
    Bool(bool),
}

/// Upstream `normalizeMetadataScalar` (metadata.ts:207-225): string
/// ≤ 1024 UTF-16 units, finite number, boolean; null and everything
/// else rejected.
fn normalize_metadata_scalar(key: &str, value: &yaml::Js) -> Result<MetadataScalar, String> {
    match value {
        yaml::Js::Str(s) => {
            if utf16_len(s) > METADATA_LIMITS.max_string_length {
                return Err(format!(
                    "metadata key \"{key}\": string exceeds {} characters",
                    METADATA_LIMITS.max_string_length
                ));
            }
            Ok(MetadataScalar::String(s.clone()))
        }
        yaml::Js::Num(n) => {
            if !n.is_finite() {
                return Err(format!("metadata key \"{key}\": numbers must be finite"));
            }
            Ok(MetadataScalar::Number(*n))
        }
        yaml::Js::Bool(b) => Ok(MetadataScalar::Bool(*b)),
        yaml::Js::Null => Err(format!(
            "metadata key \"{key}\": null is not supported — omit the key instead"
        )),
        _ => Err(format!(
            "metadata key \"{key}\": unsupported value type — use strings, numbers, booleans, or flat arrays of one of those"
        )),
    }
}

/// Upstream `normalizeMetadataArray` (metadata.ts:227-254): non-empty,
/// ≤ 128 elements, no nesting, homogeneous, first-seen dedup with JS
/// `SameValueZero` semantics (`0` and `-0` equal).
fn normalize_metadata_array(key: &str, items: &[yaml::Js]) -> Result<MetadataValue, String> {
    if items.is_empty() {
        return Err(format!(
            "metadata key \"{key}\": empty arrays are not supported — omit the key instead"
        ));
    }
    if items.len() > METADATA_LIMITS.max_array_length {
        return Err(format!(
            "metadata key \"{key}\": array exceeds {} values",
            METADATA_LIMITS.max_array_length
        ));
    }
    let mut scalars = Vec::with_capacity(items.len());
    for item in items {
        if matches!(item, yaml::Js::Seq(_)) {
            return Err(format!(
                "metadata key \"{key}\": nested arrays are not supported"
            ));
        }
        scalars.push(normalize_metadata_scalar(key, item)?);
    }
    if scalars
        .iter()
        .all(|s| matches!(s, MetadataScalar::String(_)))
    {
        let mut out: Vec<String> = Vec::with_capacity(scalars.len());
        for s in scalars {
            let MetadataScalar::String(v) = s else {
                continue;
            };
            if !out.contains(&v) {
                out.push(v);
            }
        }
        return Ok(MetadataValue::Strings(out));
    }
    if scalars
        .iter()
        .all(|s| matches!(s, MetadataScalar::Number(_)))
    {
        let mut out: Vec<f64> = Vec::with_capacity(scalars.len());
        for s in scalars {
            let MetadataScalar::Number(v) = s else {
                continue;
            };
            if !out.contains(&v) {
                out.push(v);
            }
        }
        return Ok(MetadataValue::Numbers(out));
    }
    if scalars.iter().all(|s| matches!(s, MetadataScalar::Bool(_))) {
        let mut out: Vec<bool> = Vec::with_capacity(scalars.len());
        for s in scalars {
            let MetadataScalar::Bool(v) = s else { continue };
            if !out.contains(&v) {
                out.push(v);
            }
        }
        return Ok(MetadataValue::Bools(out));
    }
    Err(format!(
        "metadata key \"{key}\": mixed-type arrays are not supported"
    ))
}

/// Upstream `truncateErrorMessage` (metadata.ts:260-264): collapse JS
/// `\s+` to one space and trim, then cap at 200 UTF-16 units as
/// `first 197 + "..."`. Upstream `slice` can split a surrogate pair
/// (emitting a lone surrogate); Rust strings cannot hold one, so we
/// stop before the boundary pair instead — same family as deviation
/// 49.
fn truncate_error_message(message: &str) -> String {
    let mut collapsed = String::with_capacity(message.len());
    let mut in_ws = false;
    for c in message.chars() {
        if is_js_ws(c) {
            if !in_ws {
                collapsed.push(' ');
            }
            in_ws = true;
        } else {
            collapsed.push(c);
            in_ws = false;
        }
    }
    let single_line = js_trim(&collapsed);
    if utf16_len(single_line) <= METADATA_LIMITS.max_error_length {
        return single_line.to_owned();
    }
    let mut units = 0usize;
    let mut end = 0usize;
    for (i, c) in single_line.char_indices() {
        let u = c.len_utf16();
        if units + u > METADATA_LIMITS.max_error_length - 3 {
            break;
        }
        units += u;
        end = i + c.len_utf8();
    }
    let mut out = String::with_capacity(end + 3);
    out.push_str(single_line.get(..end).unwrap_or_default());
    out.push_str("...");
    out
}

/// JS `String()` for a string — `JSON.stringify` escaping is identical,
/// so reuse `serde_json` (returns the quoted literal).
pub(crate) fn json_string(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_owned())
}

/// One [`MetadataValue`] as `JSON.stringify` output.
fn write_json_value(out: &mut String, value: &MetadataValue) {
    match value {
        MetadataValue::String(s) => out.push_str(&json_string(s)),
        MetadataValue::Number(n) => out.push_str(&crate::js::number_to_string(*n)),
        MetadataValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        MetadataValue::Strings(items) => {
            out.push('[');
            for (i, s) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&json_string(s));
            }
            out.push(']');
        }
        MetadataValue::Numbers(items) => {
            out.push('[');
            for (i, n) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&crate::js::number_to_string(*n));
            }
            out.push(']');
        }
        MetadataValue::Bools(items) => {
            out.push('[');
            for (i, b) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(if *b { "true" } else { "false" });
            }
            out.push(']');
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::missing_assert_message,
    clippy::shadow_unrelated,
    reason = "idiomatic test-code patterns"
)]
mod tests {
    use super::*;

    #[test]
    fn truncate_collapses_and_bounds() {
        assert_eq!(truncate_error_message("a\nb\t\tc"), "a b c");
        assert_eq!(truncate_error_message("  pad  "), "pad");
        let long = "x".repeat(300);
        let out = truncate_error_message(&long);
        assert_eq!(out.len(), 200);
        assert!(out.ends_with("..."));
        // Never split a surrogate pair: 196 units then an emoji.
        let s = format!("{}😀tail", "x".repeat(196));
        let out = truncate_error_message(&s);
        assert_eq!(out, format!("{}...", "x".repeat(196)));
    }
}
