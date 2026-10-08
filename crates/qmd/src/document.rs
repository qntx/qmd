//! Public types for document retrieval: [`crate::Qmd::get`],
//! [`crate::Qmd::document_body`], [`crate::Qmd::multi_get`] and
//! [`crate::Qmd::list_documents`].

/// Upstream `DEFAULT_MULTI_GET_MAX_BYTES` (store.ts:103): 64 KiB.
pub const DEFAULT_MULTI_GET_MAX_BYTES: usize = 64 * 1024;

/// A single document — upstream `DocumentResult` (store.ts:2348-2359).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Document {
    /// `qmd://collection/path` URI.
    pub virtual_path: String,
    /// `collection/path` for display.
    pub display_path: String,
    /// Document title.
    pub title: String,
    /// Folder context (global + longest-prefix path contexts).
    pub context: Option<String>,
    /// Content hash.
    pub hash: String,
    /// Short docid (first 6 hash chars).
    pub docid: String,
    /// Parent collection name.
    pub collection: String,
    /// Last modification timestamp (ISO-8601).
    pub modified_at: String,
    /// Body length in characters (SQLite `LENGTH()`), as upstream.
    pub body_length: usize,
    /// Body content when `include_body` was requested.
    pub body: Option<String>,
}

/// Options for [`Qmd::get`](crate::Qmd::get) — upstream
/// `{ includeBody?: boolean }`.
#[derive(Debug, Clone, Copy, Default)]
pub struct GetOptions {
    /// Load the body into [`Document::body`] (default false).
    pub include_body: bool,
}

/// 1-based line window for [`Qmd::document_body`](crate::Qmd::document_body).
#[derive(Debug, Clone, Copy, Default)]
pub struct LineRange {
    /// First line to return (1-based; `None` or 0 starts at the top).
    pub from_line: Option<u64>,
    /// Maximum number of lines; `None` reads to the end.
    pub max_lines: Option<u64>,
}

impl LineRange {
    /// No slicing — the whole body.
    pub const ALL: Self = Self {
        from_line: None,
        max_lines: None,
    };
}

/// Options for [`Qmd::multi_get`](crate::Qmd::multi_get).
#[derive(Debug, Clone, Copy)]
pub struct MultiGetOptions {
    /// Load document bodies (default false).
    pub include_body: bool,
    /// Per-file body cap; files above it are returned as
    /// [`MultiGetEntry::Skipped`]. Default
    /// [`DEFAULT_MULTI_GET_MAX_BYTES`] (64 KiB).
    pub max_bytes: usize,
}

impl Default for MultiGetOptions {
    fn default() -> Self {
        Self {
            include_body: false,
            max_bytes: DEFAULT_MULTI_GET_MAX_BYTES,
        }
    }
}

/// One entry of [`MultiGet::docs`] — upstream `MultiGetResult`
/// (store.ts:2516-2523): either the full document or a skip record.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MultiGetEntry {
    /// Document within `max_bytes`.
    Hit(Document),
    /// Document skipped because it exceeded `max_bytes`.
    Skipped {
        /// `qmd://collection/path` URI.
        virtual_path: String,
        /// `collection/path` for display.
        display_path: String,
        /// Human-readable reason, e.g. `File too large (20KB > 10KB)`.
        reason: String,
    },
}

/// Result of [`Qmd::multi_get`](crate::Qmd::multi_get): resolved entries
/// plus per-name error strings (not-found and ambiguity messages, in
/// upstream's wording).
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct MultiGet {
    /// Resolved documents, in pattern order for comma lists.
    pub docs: Vec<MultiGetEntry>,
    /// Names that could not be resolved, as upstream error strings.
    pub errors: Vec<String>,
}

/// One row of [`Qmd::list_documents`](crate::Qmd::list_documents) —
/// the `qmd ls` listing fields.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DocumentEntry {
    /// Collection-relative path.
    pub path: String,
    /// Document title.
    pub title: String,
    /// Last modification timestamp (ISO-8601).
    pub modified_at: String,
    /// Body length in characters (SQLite `LENGTH()`), as upstream `ls`.
    pub size: usize,
}
