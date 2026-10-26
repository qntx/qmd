//! `qmd://{+path}` resource read — upstream handler
//! (mcp/server.ts:204-231). Errors are returned as plain text contents
//! (200-shaped), never as protocol errors.

use qmd::{Error as QmdError, GetOptions, Qmd};
use serde_json::json;

/// `decodeURIComponent` for the `{+path}` segment — percent-decodes
/// each UTF-8 `%XX` byte; a malformed escape passes through literally.
fn decode_uri_component(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while let Some(&b) = bytes.get(i) {
        if b == b'%'
            && let (Some(hi), Some(lo)) = (
                bytes.get(i + 1).and_then(|x| char::from(*x).to_digit(16)),
                bytes.get(i + 2).and_then(|x| char::from(*x).to_digit(16)),
            )
        {
            // hi/lo are hex nibbles — the pair is at most 0xFF.
            out.push(u8::try_from((hi << 4) | lo).unwrap_or(0));
            i += 3;
            continue;
        }
        out.push(b);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Read `qmd://<path>` through [`Qmd::get`] — raw result JSON (the
/// `name`/`title` keys upstream emits cannot live in rmcp's typed
/// `ResourceContents`, so callers wrap this in `CustomResult`). Body
/// gets line numbers and the `<!-- Context: … -->` prefix like
/// upstream.
///
/// # Errors
/// [`qmd::Error`] on store failures (document-not-found and
/// excluded-by-ignore are rendered as contents, not errors).
pub(crate) fn read(qmd: &Qmd, uri: &str) -> Result<serde_json::Value, QmdError> {
    let raw = uri.strip_prefix("qmd://").unwrap_or(uri);
    let decoded = decode_uri_component(raw);
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
        Err(e) => return Err(e),
    };
    Ok(json!({ "contents": contents }))
}
