//! `Array.prototype.slice` on a split line list (ECMA-262): negative
//! indices count from the end, everything clamps into `[0, len]`, and
//! a `NaN` endpoint becomes 0.
//!
//! Callers encode `parseInt` NaN as `i64::MIN`: `ToInteger(NaN)` is 0
//! and `len + i64::MIN` clamps to 0 as well, so the one sentinel
//! reproduces both behaviors at every call site.

/// `lines.slice(start, end)` — returns the borrowed sub-slice.
pub(crate) fn js_slice<'a>(lines: &'a [&'a str], start: i64, end: i64) -> &'a [&'a str] {
    let len = i64::try_from(lines.len()).unwrap_or(i64::MAX);
    let norm = |v: i64| -> usize {
        let v = if v < 0 {
            len.saturating_add(v)
        } else {
            v.min(len)
        };
        usize::try_from(v).unwrap_or(0)
    };
    let s = norm(start);
    let e = norm(end);
    if e <= s {
        return &[];
    }
    lines.get(s..e).unwrap_or(&[])
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "idiomatic test-code pattern")]
mod tests {
    use super::js_slice;

    #[test]
    fn js_slice_clamps_and_counts_from_end() {
        let lines = ["a", "b", "c", "d"];
        assert_eq!(js_slice(&lines, 0, 2), ["a", "b"]);
        assert_eq!(js_slice(&lines, 2, 4), ["c", "d"]);
        // Negative end counts from the end — `slice(0, -2)`.
        assert_eq!(js_slice(&lines, 0, -2), ["a", "b"]);
        // NaN (`i64::MIN`) end behaves like 0 — empty slice.
        assert!(js_slice(&lines, 0, i64::MIN).is_empty());
        // Past-the-end starts and ends both clamp.
        assert!(js_slice(&lines, 99, 4).is_empty());
        assert_eq!(js_slice(&lines, 1, 99), ["b", "c", "d"]);
        // start > end yields an empty slice.
        assert!(js_slice(&lines, 3, 1).is_empty());
    }
}
