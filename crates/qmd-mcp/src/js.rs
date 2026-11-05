//! `String.prototype.slice(0, n)` for the REST/log paths — the shared
//! JS `String(v)`/truthiness coercions live in [`qmd::js`].
//!
//! Re-exports so call sites keep their `crate::js::*` paths.

pub(crate) use qmd::js::{is_truthy, js_string, js_string_or_empty};

/// `String.prototype.slice(0, n)` — chars, not bytes.
pub(crate) fn str_prefix(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}
