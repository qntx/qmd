//! `qmd://collection/path` virtual URIs and the `path:line` suffix
//! convention, ported from upstream `src/store.ts` (715-801) and
//! `src/cli/qmd.ts` (1261-1282).

/// Components of a `qmd://collection/path` URI — upstream
/// `VirtualPath` (store.ts:698-702). `index` is the `?index=` override.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct VirtualPath {
    /// Collection name.
    pub collection_name: String,
    /// Path relative to the collection root (empty for collection root).
    pub path: String,
    /// `?index=` override, when present.
    pub index_name: Option<String>,
}

/// Upstream `normalizeVirtualPath` (store.ts:715-735): normalize
/// `qmd:`, `qmd://` and `//collection/path` spellings to `qmd://`.
///
/// Bare `collection/path` input is returned unchanged — it could be a
/// relative filesystem path.
#[must_use]
pub fn normalize_virtual_path(input: &str) -> String {
    let path = input.trim();
    if let Some(rest) = path.strip_prefix("qmd:") {
        return format!("qmd://{}", rest.trim_start_matches('/'));
    }
    if let Some(rest) = path.strip_prefix("//") {
        return format!("qmd://{}", rest.trim_start_matches('/'));
    }
    path.to_owned()
}

/// Upstream `isVirtualPath` (store.ts:776-786): only explicit
/// `qmd:` / `//` spellings count; bare `collection/path` does not.
#[must_use]
pub fn is_virtual_path(path: &str) -> bool {
    let trimmed = path.trim();
    trimmed.starts_with("qmd:") || trimmed.starts_with("//")
}

/// Percent-decode like `URLSearchParams.get`: `%XX` sequences and `+`
/// as space (application/x-www-form-urlencoded).
fn url_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while let Some(&b) = bytes.get(i) {
        if b == b'+' {
            out.push(b' ');
            i += 1;
            continue;
        }
        // `%XX` with two valid hex digits decodes; anything else is
        // copied through verbatim, like `URLSearchParams`.
        let decoded = if b == b'%' {
            bytes
                .get(i + 1..i + 3)
                .and_then(|h| std::str::from_utf8(h).ok())
                .and_then(|h| u8::from_str_radix(h, 16).ok())
        } else {
            None
        };
        if let Some(v) = decoded {
            out.push(v);
            i += 3;
        } else {
            out.push(b);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Upstream `parseVirtualPath` (store.ts:742-757): `qmd://name[/path]`
/// into components, with an optional `?index=` parameter decoded like
/// `URLSearchParams.get("index")`.
#[must_use]
pub fn parse_virtual_path(virtual_path: &str) -> Option<VirtualPath> {
    let normalized = normalize_virtual_path(virtual_path);
    // `normalized.split("?")` — upstream takes elements 0 and 1, so a
    // second `?` truncates the query string.
    let mut segments = normalized.split('?');
    let path_part = segments.next().unwrap_or("");
    let query_string = segments.next().unwrap_or("");
    let rest = path_part.strip_prefix("qmd://")?;
    // `^qmd:\/\/([^\/]+)\/?(.*)$` — one optional slash between the
    // collection name and the path.
    let (collection_name, path) = rest
        .find('/')
        .map_or((rest, ""), |i| (&rest[..i], &rest[i + 1..]));
    if collection_name.is_empty() {
        return None;
    }
    let index_name = query_string
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .find(|(k, _)| *k == "index")
        .map(|(_, v)| url_decode(v).trim().to_owned())
        .filter(|v| !v.is_empty());
    Some(VirtualPath {
        collection_name: collection_name.to_owned(),
        path: path.to_owned(),
        index_name,
    })
}

/// Upstream `buildVirtualPath` (store.ts:762-765): assemble
/// `qmd://collection/path`, appending `?index=` percent-encoded when
/// given.
#[must_use]
pub fn build_virtual_path(collection_name: &str, path: &str, index_name: Option<&str>) -> String {
    let base = format!("qmd://{collection_name}/{path}");
    match index_name {
        Some(index) => {
            // `encodeURIComponent` leaves unescaped:
            // `A-Z a-z 0-9 - _ . ! ~ * ' ( )`
            const HEX: &[u8; 16] = b"0123456789ABCDEF";
            let mut encoded = String::new();
            for b in index.bytes() {
                if b.is_ascii_alphanumeric()
                    || matches!(
                        b,
                        b'-' | b'_' | b'.' | b'!' | b'~' | b'*' | b'\'' | b'(' | b')'
                    )
                {
                    encoded.push(char::from(b));
                } else {
                    let hi = HEX.get(usize::from(b >> 4)).copied().unwrap_or(b'0');
                    let lo = HEX.get(usize::from(b & 0x0F)).copied().unwrap_or(b'0');
                    encoded.push('%');
                    encoded.push(char::from(hi));
                    encoded.push(char::from(lo));
                }
            }
            format!("{base}?index={encoded}")
        }
        None => base,
    }
}

/// The path part of a `file.md:10:20` argument plus the parsed line
/// window (upstream `qmd.ts:1261-1282`):
///
/// - `file.md:100` — start at line 100
/// - `file.md:100:40` — start at line 100, read 40 lines
///
/// The `://` inside `qmd://` URIs is never matched because both forms
/// anchor digits at end of string.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct LineSuffix {
    /// Path with the line suffix removed.
    pub path: String,
    /// 1-based first line (`:N` or the first of `:N:M`), if any.
    pub from_line: Option<u64>,
    /// Line count (the `M` of `:N:M`), if any.
    pub max_lines: Option<u64>,
}

/// Split `name[:from[:count]]` into path and line window, mirroring the
/// upstream CLI (qmd.ts:1261-1282): the `:N:M` form wins over `:N`, and
/// a `from` of 0 is later clamped to 1 by the body slicer.
#[must_use]
pub fn parse_line_suffix(input: &str) -> LineSuffix {
    // `/:(\d+):(\d+)$/` first: two digit runs at the end.
    if let Some((head, last)) = input.rsplit_once(':')
        && !last.is_empty()
        && last.bytes().all(|b| b.is_ascii_digit())
        && let Some((head2, mid)) = head.rsplit_once(':')
        && !mid.is_empty()
        && mid.bytes().all(|b| b.is_ascii_digit())
    {
        return LineSuffix {
            path: head2.to_owned(),
            from_line: mid.parse().ok(),
            max_lines: last.parse().ok(),
        };
    }
    // `/:(\d+)$/`: one trailing digit run.
    if let Some((head, last)) = input.rsplit_once(':')
        && !last.is_empty()
        && last.bytes().all(|b| b.is_ascii_digit())
    {
        return LineSuffix {
            path: head.to_owned(),
            from_line: last.parse().ok(),
            max_lines: None,
        };
    }
    LineSuffix {
        path: input.to_owned(),
        from_line: None,
        max_lines: None,
    }
}
