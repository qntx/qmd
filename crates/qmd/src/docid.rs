//! Document identifiers: the first 6 characters of the content hash,
//! written `#abc123` in the CLI (upstream `src/store.ts`).

/// Upstream `getDocid` (store.ts:2364-2366): first 6 characters of the
/// content hash, without the `#` prefix.
#[must_use]
pub fn get_docid(hash: &str) -> String {
    hash.get(..6).unwrap_or(hash).to_owned()
}

/// Upstream `normalizeDocid` (store.ts:3401-3416): trim, strip
/// surrounding matching quotes (`"…"` or `'…'`), strip a leading `#`.
/// Returns the bare candidate hex string.
#[must_use]
pub fn normalize_docid(docid: &str) -> &str {
    let mut normalized = docid.trim();
    let quoted = match (normalized.as_bytes().first(), normalized.as_bytes().last()) {
        (Some(b'"'), Some(b'"')) | (Some(b'\''), Some(b'\'')) => normalized.len() >= 2,
        _ => false,
    };
    if quoted {
        normalized = normalized
            .get(1..normalized.len() - 1)
            .unwrap_or(normalized);
    }
    normalized.strip_prefix('#').unwrap_or(normalized)
}

/// Upstream `isDocid` (store.ts:3423-3427): at least 6 hexadecimal
/// characters after [`normalize_docid`].
#[must_use]
pub fn is_docid(input: &str) -> bool {
    let normalized = normalize_docid(input);
    normalized.len() >= 6 && normalized.bytes().all(|b| b.is_ascii_hexdigit())
}
