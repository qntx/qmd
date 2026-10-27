//! JavaScript coercions the REST/log paths need — `String(v)`, truthiness
//! and `slice(0, n)` semantics over JSON values (upstream server.ts).

use serde_json::Value;

/// JS truthiness for a JSON value (falsy: null, false, 0, NaN, "").
pub(crate) fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

/// `String(v || "")` — falsy values give `""`, else [`js_string`].
pub(crate) fn js_string_or_empty(v: &Value) -> String {
    if is_truthy(v) {
        js_string(v)
    } else {
        String::new()
    }
}

/// JS `String(v)`: number/bool/array (`,`-joined, null→empty)/object
/// (`"[object Object]"`)/string itself.
pub(crate) fn js_string(v: &Value) -> String {
    match v {
        Value::Null => "null".to_owned(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => qmd::js::number_to_string(n.as_f64().unwrap_or(f64::NAN)),
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .map(|i| match i {
                Value::Null => String::new(),
                other => js_string(other),
            })
            .collect::<Vec<_>>()
            .join(","),
        Value::Object(_) => "[object Object]".to_owned(),
    }
}

/// `String.prototype.slice(0, n)` — chars, not bytes.
pub(crate) fn str_prefix(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}
