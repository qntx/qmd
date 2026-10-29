//! `qmd://{+path}` resource read — upstream handler
//! (mcp/server.ts:204-231). Errors are returned as plain text contents
//! (200-shaped), never as protocol errors — except malformed escape
//! sequences: JS `decodeURIComponent` throws `URIError`, which upstream's
//! SDK wraps as `-32603 "URI error"` (recorded in the golden fixture).

use qmd::{Error as QmdError, GetOptions, Qmd};
use serde_json::json;

/// Resource-read failure — the dispatch layer renders it as the
/// JSON-RPC error payload via `Display`.
#[derive(Debug)]
pub(crate) enum ReadError {
    /// Store failure — the qmd error text.
    Store(QmdError),
    /// `decodeURIComponent` threw upstream → `-32603 "URI error"`
    /// (Bun's `URIError` message, recorded verbatim).
    Uri,
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Store(e) => e.fmt(f),
            Self::Uri => f.write_str("URI error"),
        }
    }
}

impl From<QmdError> for ReadError {
    fn from(e: QmdError) -> Self {
        Self::Store(e)
    }
}

/// `decodeURIComponent` for the `{+path}` segment — JS semantics: every
/// `%` must be followed by exactly two hex digits, and the decoded
/// bytes must be valid UTF-8. Either violation throws `URIError`
/// upstream (`%ZZ`, `%`, `%FF`, `%E2%82`), so they fail here too.
fn decode_uri_component(s: &str) -> Result<String, ReadError> {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while let Some(&b) = bytes.get(i) {
        if b == b'%' {
            let (Some(hi), Some(lo)) = (
                bytes.get(i + 1).and_then(|x| char::from(*x).to_digit(16)),
                bytes.get(i + 2).and_then(|x| char::from(*x).to_digit(16)),
            ) else {
                return Err(ReadError::Uri);
            };
            // hi/lo are hex nibbles — the pair is at most 0xFF.
            out.push(u8::try_from((hi << 4) | lo).unwrap_or(0));
            i += 3;
            continue;
        }
        out.push(b);
        i += 1;
    }
    String::from_utf8(out).map_err(|_| ReadError::Uri)
}

/// Read `qmd://<path>` through [`Qmd::get`] — raw result JSON (the
/// `name`/`title` keys upstream emits cannot live in rmcp's typed
/// `ResourceContents`, so callers wrap this in `CustomResult`). Body
/// gets line numbers and the `<!-- Context: … -->` prefix like
/// upstream.
///
/// # Errors
/// [`ReadError::Store`] on store failures, [`ReadError::Uri`] on a
/// malformed `decodeURIComponent` input (document-not-found and
/// excluded-by-ignore are rendered as contents, not errors).
pub(crate) fn read(qmd: &Qmd, uri: &str) -> Result<serde_json::Value, ReadError> {
    let raw = uri.strip_prefix("qmd://").unwrap_or(uri);
    let decoded = decode_uri_component(raw)?;
    let contents = match qmd.get(&decoded, GetOptions { include_body: true }) {
        Ok(doc) => {
            let mut text = qmd::add_line_numbers(doc.body.as_deref().unwrap_or(""), 1);
            if let Some(ctx) = doc.context.as_deref().filter(|c| !c.is_empty()) {
                text = format!("<!-- Context: {ctx} -->\n\n{text}");
            }
            let title = if doc.title.is_empty() {
                doc.display_path.clone()
            } else {
                doc.title.clone()
            };
            json!([{
                "uri": uri,
                "name": doc.display_path,
                "title": title,
                "mimeType": "text/markdown",
                "text": text,
            }])
        }
        Err(QmdError::ExcludedByIgnore {
            collection,
            path,
            rule,
            ..
        }) => json!([{
            "uri": uri,
            "text": format!("Document excluded by ignore rule: {decoded}\nCollection: {collection}\nMatched path: {path}\nIgnore rule: {rule}"),
        }]),
        Err(QmdError::DocumentNotFound { .. }) => json!([{
            "uri": uri,
            "text": format!("Document not found: {decoded}"),
        }]),
        Err(e) => return Err(e.into()),
    };
    Ok(json!({ "contents": contents }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `decodeURIComponent` round-trips the encodings upstream's tests
    /// exercise (mcp.test.ts:719-742).
    #[test]
    fn decodes_valid_escapes() {
        assert_eq!(decode_uri_component("readme.md").unwrap(), "readme.md");
        assert_eq!(
            decode_uri_component("notes%2Fmeeting.md").unwrap(),
            "notes/meeting.md"
        );
        assert_eq!(
            decode_uri_component("External%20Podcast%2F2023%20April%20-%20Interview.md").unwrap(),
            "External Podcast/2023 April - Interview.md"
        );
        // Multi-byte UTF-8 sequence — € is E2 82 AC.
        assert_eq!(decode_uri_component("%E2%82%AC").unwrap(), "\u{20ac}");
        // Colon escapes — upstream's `api.md%3A10` line suffix.
        assert_eq!(decode_uri_component("api.md%3A10").unwrap(), "api.md:10");
    }

    /// Decoding is single-pass: `%252F` → `%2F`, never `/`
    /// (mcp.test.ts:732 — upstream never double-decodes).
    #[test]
    fn double_encoded_decodes_once() {
        assert_eq!(
            decode_uri_component("meetings%252Fmeeting-2024-01.md").unwrap(),
            "meetings%2Fmeeting-2024-01.md"
        );
    }

    /// Each malformed form throws upstream (`URIError` → `-32603
    /// "URI error"` in the golden fixture).
    #[test]
    fn malformed_escapes_are_uri_errors() {
        for bad in ["docs%ZZ.md", "docs%2.md", "docs%.md", "%FF.md", "%E2%82.md"] {
            assert!(
                matches!(decode_uri_component(bad), Err(ReadError::Uri)),
                "{bad} should fail"
            );
        }
        // `%` inside a valid sequence still decodes the earlier parts.
        assert_eq!(decode_uri_component("a%20b").unwrap(), "a b");
    }

    /// `ReadError` renders the recorded wire text.
    #[test]
    fn uri_error_display_is_upstream_text() {
        assert_eq!(ReadError::Uri.to_string(), "URI error");
    }
}
