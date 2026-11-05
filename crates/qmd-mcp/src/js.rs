//! `String.prototype.slice(0, n)` for the REST/log paths — the shared
//! JS `String(v)`/truthiness coercions live in [`qmd::js`].

/// `String.prototype.slice(0, n)` — chars, not bytes.
pub(crate) fn str_prefix(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}
