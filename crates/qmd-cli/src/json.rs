//! `JSON.stringify(value, null, 2)` over [`serde_json::Value`].
//!
//! `serde_json`'s own `to_string_pretty` differs from JS on `f64`
//! (integral floats print `1.0`, large/small values take Rust-style
//! exponents), so every number goes through
//! [`qmd::js::number_to_string`] (T8). Object member order is insertion
//! order — the callers build [`serde_json::Map`] in upstream field
//! order, and `serde_json`'s `preserve_order` feature keeps it.
//! Non-finite numbers print `null`, like `JSON.stringify`.

use serde_json::Value;

/// `JSON.stringify(value, null, 2)`.
pub(crate) fn stringify_pretty(value: &Value) -> String {
    let mut out = String::new();
    write_value(value, 0, &mut out);
    out
}

fn indent(level: usize, out: &mut String) {
    for _ in 0..level {
        out.push_str("  ");
    }
}

fn write_value(value: &Value, level: usize, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            // Every number is a JS double: integers serde_json keeps at
            // u64 precision still round through f64 so `JSON.parse`'s
            // behavior on >2^53 literals is preserved.
            out.push_str(&n.as_f64().map_or_else(
                || "null".to_owned(),
                |f| {
                    if f.is_finite() {
                        qmd::js::number_to_string(f)
                    } else {
                        "null".to_owned()
                    }
                },
            ));
        }
        Value::String(s) => {
            // serde_json escapes exactly like JSON.stringify (P3.1
            // verified byte-for-byte).
            if let Ok(encoded) = serde_json::to_string(s) {
                out.push_str(&encoded);
            }
        }
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('\n');
                indent(level + 1, out);
                write_value(item, level + 1, out);
            }
            out.push('\n');
            indent(level, out);
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (key, val)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('\n');
                indent(level + 1, out);
                if let Ok(encoded) = serde_json::to_string(key) {
                    out.push_str(&encoded);
                }
                out.push_str(": ");
                write_value(val, level + 1, out);
            }
            out.push('\n');
            indent(level, out);
            out.push('}');
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn pretty_prints_like_json_stringify() {
        let v = json!({
            "a": 1,
            "b": 0.4,
            "c": "x\ny",
            "d": [],
            "e": {},
            "f": [1, {"g": true}],
        });
        let want = "{\n  \"a\": 1,\n  \"b\": 0.4,\n  \"c\": \"x\\ny\",\n  \"d\": [],\n  \"e\": {},\n  \"f\": [\n    1,\n    {\n      \"g\": true\n    }\n  ]\n}";
        assert_eq!(stringify_pretty(&v), want);
    }

    #[test]
    fn formats_numbers_with_js_rules() {
        let v = serde_json::from_str::<Value>("[0.40, 6.02e23, 12345678901234568000]").unwrap();
        assert_eq!(
            stringify_pretty(&v),
            "[\n  0.4,\n  6.02e+23,\n  12345678901234567000\n]"
        );
    }
}
