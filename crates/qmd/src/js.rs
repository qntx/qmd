//! ECMAScript number, whitespace and option-parsing semantics.
//!
//! Upstream `tobi/qmd` runs on JavaScript, so every number it prints
//! and every option value it parses follows ECMA-262 rules:
//! `Number::toString`, `Number.prototype.toFixed`, `Math.round`,
//! `parseInt`, `parseFloat` and the `\s` character class. Rust's native
//! formatting and parsing differ (ties-to-even rounding, `1.0`
//! suffixes, `e` notation, whole-string parsing), so the rules live
//! here for the library, the CLI and the MCP server to share.

pub use crate::chunk::is_js_ws;

/// ECMAScript `Number::toString` (radix 10).
///
/// Used for `JSON.stringify` numbers and `String(key)` on numeric keys:
/// shortest round-trip digits (Rust `{:e}` supplies them), decimal when
/// `1e-6 <= |x| < 1e21`, otherwise `d.ddde±N`; integral values carry no
/// `.0`; `-0` → `"0"`.
#[must_use]
pub fn number_to_string(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_owned();
    }
    if v.is_infinite() {
        return if v < 0.0 {
            "-Infinity".to_owned()
        } else {
            "Infinity".to_owned()
        };
    }
    if v == 0.0 {
        return "0".to_owned();
    }
    let neg = v.is_sign_negative();
    // `{:e}` gives the shortest round-trip mantissa: `d[.dd]e<exp>`.
    let sci = format!("{:e}", v.abs());
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    // `n` is the position of the decimal point: value = digits × 10^(n-k).
    let n = exponent.parse::<i64>().unwrap_or(0) + 1;
    let k = i64::try_from(digits.len()).unwrap_or(0);
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if k <= n && n <= 21 {
        // Integer: digits then `n - k` zeros.
        out.push_str(&digits);
        for _ in k..n {
            out.push('0');
        }
    } else if n > 0 && n <= 21 {
        // Decimal point inside the digit string.
        let cut = usize::try_from(n).unwrap_or(0);
        out.push_str(digits.get(..cut).unwrap_or_default());
        out.push('.');
        out.push_str(digits.get(cut..).unwrap_or_default());
    } else if n > -6 && n <= 0 {
        // `0.` + `-n` zeros + digits.
        out.push_str("0.");
        for _ in n..0 {
            out.push('0');
        }
        out.push_str(&digits);
    } else {
        // Exponential: `d[.dd]e<sign><exp>` (JS always signs the exponent).
        let e = n - 1;
        out.push_str(digits.get(..1).unwrap_or("0"));
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if e >= 0 { '+' } else { '-' });
        out.push_str(&e.abs().to_string());
    }
    out
}

/// ECMA-262 `Number.prototype.toFixed(digits)`.
///
/// `|x| >= 1e21` defers to `Number::toString`; negatives print `-` and
/// round the magnitude (so `(-0.001).toFixed(2)` is `"-0.00"` while
/// `(-0).toFixed(2)` is `"0.00"`). Ties on the exact binary value round
/// up — `format!("{:.N}")` would give ties-to-even and is wrong here
/// (upstream `formatScore`/`toFixed` call sites, qmd.ts:2364).
#[must_use]
pub fn to_fixed(v: f64, digits: u32) -> String {
    if v.is_nan() {
        return "NaN".to_owned();
    }
    if v.abs() >= 1e21 {
        return number_to_string(v);
    }
    if v < 0.0 {
        return format!("-{}", to_fixed_magnitude(-v, digits));
    }
    // `-0` reaches here (`x < 0` is false in JS too) and must not gain
    // a sign from Rust's formatter.
    to_fixed_magnitude(v.abs(), digits)
}

/// `toFixed` on a non-negative finite value below `1e21`.
fn to_fixed_magnitude(v: f64, digits: u32) -> String {
    // An f64 is a dyadic fraction, so a wide-precision decimal expansion
    // is exact (1074 fraction digits suffice for any f64). Rounding
    // half-up over that expansion reproduces JS, whose toFixed picks the
    // closer decimal and the larger one on an exact tie.
    let expanded = format!("{v:.1100}");
    let (int_part, frac_part) = expanded.split_once('.').unwrap_or((expanded.as_str(), ""));
    let digits = usize::try_from(digits)
        .unwrap_or(usize::MAX)
        .min(frac_part.len());
    let mut int_digits = int_part.as_bytes().to_vec();
    let mut keep = frac_part
        .as_bytes()
        .get(..digits)
        .unwrap_or_default()
        .to_vec();
    // The dropped tail sits on a decimal boundary: any remainder above
    // "4000…" (including the exact "5000…" tie) rounds up.
    let round_up = frac_part.as_bytes().get(digits).is_some_and(|&d| d > b'4');
    if round_up {
        let mut carry = bump_ascii_digits(&mut keep);
        if carry {
            carry = bump_ascii_digits(&mut int_digits);
            if carry {
                int_digits.insert(0, b'1');
            }
        }
    }
    let mut out = String::from_utf8_lossy(&int_digits).into_owned();
    if digits > 0 {
        out.push('.');
        out.push_str(&String::from_utf8_lossy(&keep));
    }
    out
}

/// Increment an ASCII digit string in place; `true` on carry-out.
fn bump_ascii_digits(digits: &mut [u8]) -> bool {
    for d in digits.iter_mut().rev() {
        if *d == b'9' {
            *d = b'0';
        } else {
            *d += 1;
            return false;
        }
    }
    true
}

/// JS `Math.round`: ties toward +∞ (`Math.round(-2.5)` is `-2`,
/// `Math.round(0.49999999999999994)` is `0`).
#[must_use]
#[allow(
    clippy::float_cmp,
    reason = "the fraction of an f64 is exact; the 0.5 tie test is an \
              exact-value check, not an approximation"
)]
pub fn math_round(v: f64) -> f64 {
    if !v.is_finite() || v == 0.0 {
        return v;
    }
    if v.fract().abs() == 0.5 {
        v.ceil()
    } else {
        v.round()
    }
}

/// JS `parseInt(s, 10)`.
///
/// Leading `\s` skipped, optional sign, then the longest run of decimal
/// digits; `None` (NaN upstream) when no digit follows. Overflows
/// saturate (upstream would return `Infinity`-scale numbers, which only
/// clamp further downstream anyway).
#[must_use]
pub fn parse_int(s: &str) -> Option<i64> {
    let t = s.trim_start_matches(is_js_ws);
    let (negative, digits) = t.strip_prefix('-').map_or_else(
        || (false, t.strip_prefix('+').unwrap_or(t)),
        |rest| (true, rest),
    );
    let mut value: i64 = 0;
    let mut seen = false;
    for b in digits.bytes() {
        if !b.is_ascii_digit() {
            break;
        }
        seen = true;
        value = value.saturating_mul(10).saturating_add(i64::from(b - b'0'));
    }
    if !seen {
        return None;
    }
    Some(if negative { -value } else { value })
}

/// JS `parseFloat(s)`: leading `\s`, optional sign, then the longest
/// valid float prefix — `digits[.digits][e[±]digits]`, `.digits`, or
/// `Infinity`; `None` (NaN upstream) when no mantissa digit is present.
#[must_use]
pub fn parse_float(s: &str) -> Option<f64> {
    let t = s.trim_start_matches(is_js_ws);
    let (negative, rest) = t.strip_prefix('-').map_or_else(
        || (false, t.strip_prefix('+').unwrap_or(t)),
        |rest| (true, rest),
    );
    if rest.starts_with("Infinity") {
        return Some(if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        });
    }
    let bytes = rest.as_bytes();
    let mut i = 0usize;
    let mut mantissa = false;
    while bytes.get(i).is_some_and(u8::is_ascii_digit) {
        i += 1;
        mantissa = true;
    }
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
            mantissa = true;
        }
    }
    if !mantissa {
        return None;
    }
    // An `e`/`E` joins the token only when digits follow (upstream drops
    // a trailing bare exponent: `parseFloat("1e") === 1`).
    let mut end = i;
    if bytes.get(i).is_some_and(|b| matches!(b, b'e' | b'E')) {
        let mut j = i + 1;
        if bytes.get(j).is_some_and(|b| matches!(b, b'+' | b'-')) {
            j += 1;
        }
        let exp_start = j;
        while bytes.get(j).is_some_and(u8::is_ascii_digit) {
            j += 1;
        }
        if j > exp_start {
            end = j;
        }
    }
    let v: f64 = rest.get(..end)?.parse().ok()?;
    Some(if negative { -v } else { v })
}

/// JS `Number(s)` — strict string→number conversion (not [`parse_float`]).
///
/// Used by upstream `parseEmbedBatchOption`/`parseEmbedTimeoutOption`
/// (cli/qmd.ts:2116-2141): trims `\s`, treats empty/whitespace as `0`,
/// accepts `Infinity` and decimal/exponent literals plus `0x`/`0o`/`0b`
/// radix prefixes; `None` (NaN upstream) for anything else, including
/// `parseFloat`-style truncated parses such as `"3abc"`.
#[must_use]
pub fn number(s: &str) -> Option<f64> {
    let trimmed = s.trim_matches(is_js_ws);
    if trimmed.is_empty() {
        return Some(0.0);
    }
    for (prefix, radix) in [
        ("0x", 16),
        ("0X", 16),
        ("0b", 2),
        ("0B", 2),
        ("0o", 8),
        ("0O", 8),
    ] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            return u64::from_str_radix(rest, radix).ok().map(|v| v as f64);
        }
    }
    match trimmed {
        "Infinity" | "+Infinity" => return Some(f64::INFINITY),
        "-Infinity" => return Some(f64::NEG_INFINITY),
        _ => {}
    }
    // Decimal grammar `[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?` — Rust's
    // `f64::from_str` also accepts `inf`/`nan` spellings JS rejects,
    // so the grammar is validated before delegating.
    let bytes = trimmed.as_bytes();
    let mut idx = 0usize;
    if matches!(bytes.get(idx), Some(b'+' | b'-')) {
        idx += 1;
    }
    let mut mantissa = false;
    while bytes.get(idx).is_some_and(u8::is_ascii_digit) {
        idx += 1;
        mantissa = true;
    }
    if bytes.get(idx) == Some(&b'.') {
        idx += 1;
        while bytes.get(idx).is_some_and(u8::is_ascii_digit) {
            idx += 1;
            mantissa = true;
        }
    }
    if !mantissa {
        return None;
    }
    if matches!(bytes.get(idx), Some(b'e' | b'E')) {
        let mut exp_end = idx + 1;
        if matches!(bytes.get(exp_end), Some(b'+' | b'-')) {
            exp_end += 1;
        }
        let exp_start = exp_end;
        while bytes.get(exp_end).is_some_and(u8::is_ascii_digit) {
            exp_end += 1;
        }
        if exp_end == exp_start {
            return None;
        }
        idx = exp_end;
    }
    if idx != bytes.len() {
        return None;
    }
    trimmed.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_to_string_cases() {
        let cases = [
            (0.0, "0"),
            (-0.0, "0"),
            (1.0, "1"),
            (-7.0, "-7"),
            (0.5, "0.5"),
            (-0.5, "-0.5"),
            (0.1, "0.1"),
            (2.5, "2.5"),
            (1000.0, "1000"),
            (0.0015, "0.0015"),
            (1e-6, "0.000001"),
            (1e-7, "1e-7"),
            (1e20, "100000000000000000000"),
            (1e21, "1e+21"),
            (6.02e23, "6.02e+23"),
            (12_345_678_901_234_567_890.0, "12345678901234567000"),
            (9_007_199_254_740_992.0, "9007199254740992"),
            (1.234_567_890_123_456_9e23, "1.2345678901234569e+23"),
            (f64::INFINITY, "Infinity"),
            (f64::NEG_INFINITY, "-Infinity"),
            (f64::NAN, "NaN"),
            (-1e-400, "0"),
        ];
        for (v, want) in cases {
            assert_eq!(number_to_string(v), want, "value {v}");
        }
    }

    #[test]
    fn to_fixed_matches_measured_js() {
        // Values measured against V8/JSC (upstream runs on both).
        let cases = [
            (0.125, 2, "0.13"),
            (1.25, 1, "1.3"),
            (2.5, 0, "3"),
            (12.5, 0, "13"),
            (0.375, 2, "0.38"),
            (1.005, 2, "1.00"),
            (0.00005, 4, "0.0001"),
            (0.12345, 4, "0.1235"),
            (0.85, 2, "0.85"),
            (3.25, 1, "3.3"),
            (1.15, 1, "1.1"),
            (0.5, 0, "1"),
            (-0.5, 0, "-1"),
            (-0.4, 0, "-0"),
            (-2.5, 0, "-3"),
            (-0.001, 2, "-0.00"),
            (0.0, 2, "0.00"),
            (-0.0, 2, "0.00"),
            (0.499_999_999_999_999_94, 0, "0"),
            (1e21, 2, "1e+21"),
            (f64::NAN, 2, "NaN"),
        ];
        for (v, digits, want) in cases {
            assert_eq!(to_fixed(v, digits), want, "({v}).toFixed({digits})");
        }
    }

    #[test]
    fn math_round_ties_toward_positive_infinity() {
        assert_eq!(math_round(-2.5), -2.0);
        assert_eq!(math_round(2.5), 3.0);
        assert_eq!(math_round(0.499_999_999_999_999_94), 0.0);
        assert_eq!(math_round(0.5), 1.0);
        assert_eq!(math_round(-0.5), -0.0);
        assert!(math_round(f64::NAN).is_nan());
        assert_eq!(math_round(-0.4), -0.0);
    }

    #[test]
    fn parse_int_lenient_prefix() {
        assert_eq!(parse_int("5abc"), Some(5));
        assert_eq!(parse_int("abc"), None);
        assert_eq!(parse_int("1e3"), Some(1));
        assert_eq!(parse_int("  -5"), Some(-5));
        assert_eq!(parse_int("+7"), Some(7));
        assert_eq!(parse_int("- 5"), None);
        assert_eq!(parse_int("0"), Some(0));
        assert_eq!(parse_int(""), None);
        assert_eq!(parse_int("\u{feff}12"), Some(12));
        assert_eq!(parse_int("0x10"), Some(0));
    }

    #[test]
    fn parse_float_lenient_prefix() {
        assert_eq!(parse_float("1e3"), Some(1000.0));
        assert_eq!(parse_float("5abc"), Some(5.0));
        assert_eq!(parse_float("abc"), None);
        assert_eq!(parse_float(" .5x"), Some(0.5));
        assert_eq!(parse_float("1."), Some(1.0));
        assert_eq!(parse_float("1e"), Some(1.0));
        assert_eq!(parse_float("-1.5e-2x"), Some(-0.015));
        assert_eq!(parse_float("Infinity"), Some(f64::INFINITY));
        assert_eq!(parse_float("-Infinity"), Some(f64::NEG_INFINITY));
        assert_eq!(parse_float("infinity"), None);
    }

    #[test]
    fn number_strict_grammar() {
        // `Number()` cases where it differs from `parseFloat`.
        assert_eq!(number(""), Some(0.0));
        assert_eq!(number("   "), Some(0.0));
        assert_eq!(number("3abc"), None);
        assert_eq!(number("0x10"), Some(16.0));
        assert_eq!(number("0b11"), Some(3.0));
        assert_eq!(number("0o17"), Some(15.0));
        assert_eq!(number("inf"), None);
        assert_eq!(number("NaN"), None);
        // Shared ground.
        assert_eq!(number("1e3"), Some(1000.0));
        assert_eq!(number("  -2.5 "), Some(-2.5));
        assert_eq!(number(".5"), Some(0.5));
        assert_eq!(number("1."), Some(1.0));
        assert_eq!(number("Infinity"), Some(f64::INFINITY));
        assert_eq!(number("+Infinity"), Some(f64::INFINITY));
        assert_eq!(number("-Infinity"), Some(f64::NEG_INFINITY));
        assert_eq!(number("1e"), None);
        assert_eq!(number("-0x10"), None);
        assert_eq!(number("1_000"), None);
        assert_eq!(number("."), None);
    }
}
