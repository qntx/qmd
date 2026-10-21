//! Metadata filter validation, SQL compilation, and end-to-end predicate
//! semantics — a port of upstream `test/metadata-filter.test.ts` — plus
//! the `metadata_filter_golden.json` compile fixture generated from
//! upstream by `scripts/gen_metadata_filter_golden.ts`.
//!
//! Upstream drives `matchPaths` on a hand-built `documents` +
//! `document_metadata*` schema; here the same SQL shape runs on a real
//! `Qmd` store so stored metadata arrives through the actual extraction
//! pipeline.

#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_assert_message,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::shadow_unrelated,
    reason = "idiomatic test-code patterns"
)]

use std::path::{Path, PathBuf};

use qmd::{
    METADATA_EXTRACTION_VERSION, METADATA_FILTER_LIMITS, MetadataFilter, MetadataScalar,
    parse_metadata_filter,
};
use rusqlite::params_from_iter;

// =============================================================================
// Validation — upstream `describe("parseMetadataFilter")`
// =============================================================================

/// `toEqual` for filters: rebuild the upstream JSON shape from the Rust
/// AST so structural assertions read like upstream's.
fn filter_to_json(f: &MetadataFilter) -> serde_json::Value {
    use serde_json::Value;
    let cond = |key: &str, op: &str, value: Value| serde_json::json!({"key": key, "operator": op, "value": value});
    match f {
        MetadataFilter::And { operands } => serde_json::json!({
            "operator": "and",
            "operands": operands.iter().map(filter_to_json).collect::<Vec<_>>(),
        }),
        MetadataFilter::Or { operands } => serde_json::json!({
            "operator": "or",
            "operands": operands.iter().map(filter_to_json).collect::<Vec<_>>(),
        }),
        MetadataFilter::Not { operand } => serde_json::json!({
            "operator": "not",
            "operand": filter_to_json(operand),
        }),
        MetadataFilter::Eq { key, value } => cond(key, "eq", scalar_to_json(value)),
        MetadataFilter::Ne { key, value } => cond(key, "ne", scalar_to_json(value)),
        MetadataFilter::Gt { key, value } => cond(key, "gt", scalar_to_json(value)),
        MetadataFilter::Gte { key, value } => cond(key, "gte", scalar_to_json(value)),
        MetadataFilter::Lt { key, value } => cond(key, "lt", scalar_to_json(value)),
        MetadataFilter::Lte { key, value } => cond(key, "lte", scalar_to_json(value)),
        MetadataFilter::In { key, values } => cond(
            key,
            "in",
            serde_json::json!(values.iter().map(scalar_to_json).collect::<Vec<_>>()),
        ),
        MetadataFilter::Nin { key, values } => cond(
            key,
            "nin",
            serde_json::json!(values.iter().map(scalar_to_json).collect::<Vec<_>>()),
        ),
        MetadataFilter::All { key, values } => cond(
            key,
            "all",
            serde_json::json!(values.iter().map(scalar_to_json).collect::<Vec<_>>()),
        ),
        MetadataFilter::Exists { key, value } => cond(key, "exists", serde_json::json!(*value)),
    }
}

fn scalar_to_json(s: &MetadataScalar) -> serde_json::Value {
    match s {
        MetadataScalar::String(v) => serde_json::json!(v),
        MetadataScalar::Number(v) => serde_json::json!(v),
        MetadataScalar::Bool(v) => serde_json::json!(v),
    }
}

/// `toEqual` for `serde_json::Value` with JS number semantics —
/// `1`/`1.0`/`-0` compare equal, strings and booleans strictly.
fn json_eq(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use serde_json::Value;
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(v, w)| json_eq(v, w))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.get(k).is_some_and(|w| json_eq(v, w)))
        }
        _ => a == b,
    }
}

fn parse(json: &str) -> MetadataFilter {
    parse_metadata_filter(&serde_json::from_str(json).expect("json literal parses"))
        .expect("filter parses")
}

fn parse_err(json: &str) -> qmd::MetadataFilterError {
    let value: serde_json::Value = serde_json::from_str(json).expect("json literal parses");
    parse_metadata_filter(&value).expect_err("filter must not parse")
}

#[test]
fn accepts_every_condition_operator_shape() {
    let conditions = [
        r#"{"key": "status", "operator": "eq", "value": "published"}"#,
        r#"{"key": "status", "operator": "ne", "value": "draft"}"#,
        r#"{"key": "priority", "operator": "gt", "value": 3}"#,
        r#"{"key": "priority", "operator": "gte", "value": 3}"#,
        r#"{"key": "priority", "operator": "lt", "value": 10}"#,
        r#"{"key": "name", "operator": "lte", "value": "m"}"#,
        r#"{"key": "topics", "operator": "in", "value": ["a", "b"]}"#,
        r#"{"key": "topics", "operator": "nin", "value": [1, 2]}"#,
        r#"{"key": "flags", "operator": "all", "value": [true, false]}"#,
        r#"{"key": "status", "operator": "exists", "value": false}"#,
    ];
    for condition in conditions {
        let input: serde_json::Value = serde_json::from_str(condition).unwrap();
        let parsed = parse_metadata_filter(&input).unwrap();
        assert!(json_eq(&filter_to_json(&parsed), &input), "for {condition}");
    }
}

#[test]
fn accepts_nested_groups_and_negation() {
    let json = r#"{
      "operator": "and",
      "operands": [
        {"key": "topics", "operator": "all", "value": ["typescript", "programming"]},
        {"operator": "or", "operands": [
          {"key": "status", "operator": "eq", "value": "published"},
          {"operator": "not", "operand": {"key": "audience", "operator": "eq", "value": "internal"}}
        ]}
      ]
    }"#;
    let input: serde_json::Value = serde_json::from_str(json).unwrap();
    let parsed = parse_metadata_filter(&input).unwrap();
    assert!(json_eq(&filter_to_json(&parsed), &input));
}

#[test]
fn canonicalizes_membership_arrays_by_deduplicating() {
    let parsed = parse(r#"{"key": "topics", "operator": "in", "value": ["a", "b", "a"]}"#);
    let expected: serde_json::Value =
        serde_json::from_str(r#"{"key": "topics", "operator": "in", "value": ["a", "b"]}"#)
            .unwrap();
    assert!(json_eq(&filter_to_json(&parsed), &expected));
}

/// `-0` cannot arrive through JSON text (`JSON.stringify(-0)` → `"0"`
/// before our boundary, `serde_json` parses `-0` as `+0`) but a
/// programmatic `Value` can still carry it — `SameValueZero` folds `-0`/`+0`
/// into one element while keeping the first-seen value (upstream `Set`,
/// metadata-filter.ts:232-238).
#[test]
fn deduplicates_neg_zero_as_zero_keeping_first_seen() {
    let parsed = parse_metadata_filter(&serde_json::json!({
        "key": "n",
        "operator": "in",
        "value": [-0.0_f64, 0, 1],
    }))
    .unwrap();
    let MetadataFilter::In { values, .. } = parsed else {
        panic!("expected in, got {parsed:?}")
    };
    let [MetadataScalar::Number(a), MetadataScalar::Number(b)] = values.as_slice() else {
        panic!("expected [Number, Number], got {values:?}")
    };
    assert!(a.is_sign_negative() && *a == 0.0);
    assert_eq!(*b, 1.0);
}

#[test]
fn rejects_invalid_node_shapes_with_failing_json_path() {
    let cases: &[(&str, &str)] = &[
        // (JSON input, substring expected in the error message)
        ("\"not-an-object\"", "at $:"),
        ("\"not-an-object\"", "must be an object"),
        (r#"{"key": "a", "value": 1}"#, "missing 'operator'"),
        (
            r#"{"operator": "equal", "key": "a", "value": 1}"#,
            "unknown operator 'equal'",
        ),
        (
            r#"{"operator": "and", "operands": []}"#,
            "non-empty 'operands'",
        ),
        (
            r#"{"operator": "and", "operands": "nope"}"#,
            "'operands' array",
        ),
        (
            r#"{"operator": "not", "operands": [{"key": "a", "operator": "eq", "value": 1}]}"#,
            "unknown property 'operands'",
        ),
        (r#"{"operator": "not"}"#, "exactly one 'operand'"),
        (
            r#"{"operator": "and", "operands": [{"operator": "eq"}]}"#,
            "at $.operands[0]",
        ),
        (
            r#"{"operator": "eq", "value": 1}"#,
            "non-empty string 'key'",
        ),
        (r#"{"operator": "eq", "key": "a"}"#, "requires a 'value'"),
        (
            r#"{"operator": "eq", "key": "a", "value": 1, "extra": true}"#,
            "unknown property 'extra'",
        ),
        (
            r#"{"operator": "and", "operands": [{"key": "a", "operator": "eq", "value": 1}], "key": "a"}"#,
            "unknown property 'key'",
        ),
        (
            r#"{"operator": "gt", "key": "a", "value": true}"#,
            "string or number value",
        ),
        // NaN cannot be written as JSON; a non-finite serde Value is
        // exercised below instead.
        (
            r#"{"operator": "eq", "key": "a", "value": {"nested": 1}}"#,
            "string, number, or boolean",
        ),
        (
            r#"{"operator": "in", "key": "a", "value": "x"}"#,
            "array value",
        ),
        (
            r#"{"operator": "in", "key": "a", "value": []}"#,
            "non-empty array",
        ),
        (
            r#"{"operator": "in", "key": "a", "value": [1, "two"]}"#,
            "homogeneous array",
        ),
        (
            r#"{"operator": "exists", "key": "a", "value": "yes"}"#,
            "boolean value",
        ),
    ];
    for (input, expected) in cases {
        let err = parse_err(input);
        let message = err.to_string();
        assert!(
            message.contains(expected),
            "expected {expected:?} in {message:?} (input {input})"
        );
    }

    // Upstream's `{value: NaN}` case is N/A by construction: upstream
    // accepts `unknown` input, so a programmatic `NaN` reaches
    // `Number.isFinite`; our input is `serde_json::Value`, which cannot
    // represent a non-finite number at all (`1e999`/`NaN` fail at JSON
    // parse; `Value::from(f64::INFINITY)` becomes `Null`, which the
    // scalar-type check above already rejects). The
    // `numbers must be finite` branch in `parse_scalar_value` is kept
    // for contract parity but is unreachable through `serde_json::Value`.
}

#[test]
fn rejects_excessive_depth_node_count_and_operand_count() {
    // depth: `maxDepth` `not` wrappers around a leaf = depth violation.
    let mut deep = serde_json::json!({"key": "a", "operator": "eq", "value": 1});
    for _ in 0..=METADATA_FILTER_LIMITS.max_depth {
        deep = serde_json::json!({"operator": "not", "operand": deep});
    }
    let err = parse_metadata_filter(&deep).expect_err("deep filter rejected");
    assert!(err.to_string().contains("nesting depth"));

    let condition = serde_json::json!({"key": "a", "operator": "eq", "value": 1});
    let wide = serde_json::json!({
        "operator": "or",
        "operands": vec![condition.clone(); METADATA_FILTER_LIMITS.max_group_operands + 1],
    });
    let err = parse_metadata_filter(&wide).expect_err("wide group rejected");
    assert!(err.to_string().contains("operands"));

    let many_nodes = serde_json::json!({
        "operator": "and",
        "operands": vec![
            serde_json::json!({
                "operator": "and",
                "operands": vec![condition; METADATA_FILTER_LIMITS.max_group_operands],
            });
            METADATA_FILTER_LIMITS.max_group_operands
        ],
    });
    let err = parse_metadata_filter(&many_nodes).expect_err("node flood rejected");
    assert!(err.to_string().contains("nodes"));

    let many_values = serde_json::json!({
        "key": "a",
        "operator": "in",
        "value": (0..=METADATA_FILTER_LIMITS.max_membership_values).collect::<Vec<_>>(),
    });
    let err = parse_metadata_filter(&many_values).expect_err("membership flood rejected");
    assert!(err.to_string().contains("values"));
}

// =============================================================================
// Semantics — upstream `describe("compileMetadataFilter semantics")`
// =============================================================================

fn write(dir: &Path, rel: &str, content: &str) {
    let path = dir.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}

/// A markdown document whose `qmd.metadata` frontmatter holds `yaml`
/// (already indented for the `metadata:` mapping) and a body containing
/// the common `qmdtesttoken` term.
fn doc(yaml: &str) -> String {
    format!("---\nqmd:\n  metadata:\n{yaml}---\nqmdtesttoken body\n")
}

/// `qmd` handle + `docs` collection over `tmp/docs` holding `docs`,
/// indexed. Returns the handle and the SQLite path for raw reads.
fn indexed_store(tmp: &Path, docs: &[(&str, String)]) -> (qmd::Qmd, PathBuf) {
    let docs_dir = tmp.join("docs");
    for (rel, content) in docs {
        write(&docs_dir, rel, content);
    }
    let db_path = tmp.join("i.sqlite");
    let qmd = qmd::Qmd::builder(&db_path)
        .environment(qmd::Environment::default())
        .build()
        .unwrap();
    qmd.add_collection(
        "docs",
        &qmd::CollectionSpec {
            path: docs_dir.to_string_lossy().into_owned(),
            ..qmd::CollectionSpec::default()
        },
    )
    .unwrap();
    qmd.update(&qmd::UpdateOptions::default(), &mut |_| {})
        .unwrap();
    (qmd, db_path)
}

/// Upstream `matchPaths` (metadata-filter.test.ts:155-166): compile the
/// filter, run the same `JOIN document_metadata` + extraction-gate
/// shape, return matching document paths sorted.
fn match_paths(conn: &rusqlite::Connection, filter: &serde_json::Value) -> Vec<String> {
    let parsed = parse_metadata_filter(filter).expect("filter parses");
    let compiled = qmd::compile_metadata_filter(&parsed, "d");
    let sql = format!(
        "SELECT d.path FROM documents d \
         JOIN document_metadata dm ON dm.document_id = d.id \
           AND dm.extraction_version = {METADATA_EXTRACTION_VERSION} \
           AND dm.extraction_error IS NULL \
         WHERE d.active = 1 AND {} \
         ORDER BY d.path",
        compiled.sql
    );
    conn.prepare(&sql)
        .unwrap()
        .query_map(params_from_iter(compiled.params), |r| r.get(0))
        .unwrap()
        .collect::<Result<Vec<String>, _>>()
        .unwrap()
}

fn raw_conn(path: &Path) -> rusqlite::Connection {
    rusqlite::Connection::open(path).expect("open raw connection")
}

#[test]
fn eq_matches_each_scalar_type_exactly_without_coercion() {
    let tmp = tempfile::tempdir().unwrap();
    let (_qmd, db) = indexed_store(
        tmp.path(),
        &[
            ("str.md", doc("    status: published\n")),
            ("num.md", doc("    status: 1\n")),
            ("bool.md", doc("    status: true\n")),
        ],
    );
    let conn = raw_conn(&db);

    let paths = |f: &str| match_paths(&conn, &serde_json::from_str(f).unwrap());
    assert_eq!(
        paths(r#"{"key": "status", "operator": "eq", "value": "published"}"#),
        ["str.md"]
    );
    assert_eq!(
        paths(r#"{"key": "status", "operator": "eq", "value": 1}"#),
        ["num.md"]
    );
    assert_eq!(
        paths(r#"{"key": "status", "operator": "eq", "value": true}"#),
        ["bool.md"]
    );
    assert_eq!(
        paths(r#"{"key": "status", "operator": "eq", "value": "1"}"#),
        Vec::<String>::new()
    );
}

#[test]
fn ne_requires_key_presence_and_matching_type() {
    let tmp = tempfile::tempdir().unwrap();
    let (_qmd, db) = indexed_store(
        tmp.path(),
        &[
            ("draft.md", doc("    status: draft\n")),
            ("published.md", doc("    status: published\n")),
            ("missing.md", doc("    other: x\n")),
            ("typed.md", doc("    status: 1\n")),
        ],
    );
    let conn = raw_conn(&db);
    assert_eq!(
        match_paths(
            &conn,
            &serde_json::from_str(r#"{"key": "status", "operator": "ne", "value": "draft"}"#)
                .unwrap()
        ),
        ["published.md"]
    );
}

#[test]
fn ordered_comparisons_match_numbers_and_binary_ordered_strings() {
    let tmp = tempfile::tempdir().unwrap();
    let (_qmd, db) = indexed_store(
        tmp.path(),
        &[
            ("low.md", doc("    priority: 1\n")),
            ("mid.md", doc("    priority: 3\n")),
            ("high.md", doc("    priority: 7\n")),
            ("alpha.md", doc("    name: alpha\n")),
            ("zulu.md", doc("    name: zulu\n")),
        ],
    );
    let conn = raw_conn(&db);
    let paths = |f: &str| match_paths(&conn, &serde_json::from_str(f).unwrap());
    assert_eq!(
        paths(r#"{"key": "priority", "operator": "gte", "value": 3}"#),
        ["high.md", "mid.md"]
    );
    assert_eq!(
        paths(r#"{"key": "priority", "operator": "lt", "value": 3}"#),
        ["low.md"]
    );
    assert_eq!(
        paths(r#"{"key": "name", "operator": "gt", "value": "alpha"}"#),
        ["zulu.md"]
    );
    // Type mismatch: no string 'priority' values exist.
    assert_eq!(
        paths(r#"{"key": "priority", "operator": "gte", "value": "3"}"#),
        Vec::<String>::new()
    );
}

#[test]
fn in_nin_and_all_evaluate_membership_over_value_sets() {
    let tmp = tempfile::tempdir().unwrap();
    let (_qmd, db) = indexed_store(
        tmp.path(),
        &[
            ("ts.md", doc("    topics: [typescript, programming]\n")),
            ("go.md", doc("    topics: [go]\n")),
            ("none.md", doc("    other: x\n")),
        ],
    );
    let conn = raw_conn(&db);
    let paths = |f: &str| match_paths(&conn, &serde_json::from_str(f).unwrap());
    assert_eq!(
        paths(r#"{"key": "topics", "operator": "in", "value": ["typescript", "rust"]}"#),
        ["ts.md"]
    );
    assert_eq!(
        paths(r#"{"key": "topics", "operator": "nin", "value": ["typescript", "rust"]}"#),
        ["go.md"]
    );
    assert_eq!(
        paths(r#"{"key": "topics", "operator": "all", "value": ["typescript", "programming"]}"#),
        ["ts.md"]
    );
    assert_eq!(
        paths(r#"{"key": "topics", "operator": "all", "value": ["typescript", "rust"]}"#),
        Vec::<String>::new()
    );
}

#[test]
fn exists_matches_presence_and_absence() {
    let tmp = tempfile::tempdir().unwrap();
    let (_qmd, db) = indexed_store(
        tmp.path(),
        &[
            ("has.md", doc("    status: ok\n")),
            ("hasnt.md", doc("    other: x\n")),
        ],
    );
    let conn = raw_conn(&db);
    let paths = |f: &str| match_paths(&conn, &serde_json::from_str(f).unwrap());
    assert_eq!(
        paths(r#"{"key": "status", "operator": "exists", "value": true}"#),
        ["has.md"]
    );
    assert_eq!(
        paths(r#"{"key": "status", "operator": "exists", "value": false}"#),
        ["hasnt.md"]
    );
}

#[test]
fn and_or_and_not_compose_recursively() {
    let tmp = tempfile::tempdir().unwrap();
    let (_qmd, db) = indexed_store(
        tmp.path(),
        &[
            ("a.md", doc("    status: published\n    priority: 5\n")),
            ("b.md", doc("    status: published\n    priority: 1\n")),
            ("c.md", doc("    status: draft\n    priority: 9\n")),
        ],
    );
    let conn = raw_conn(&db);
    let paths = |f: &str| match_paths(&conn, &serde_json::from_str(f).unwrap());
    assert_eq!(
        paths(
            r#"{"operator": "and", "operands": [
              {"key": "status", "operator": "eq", "value": "published"},
              {"key": "priority", "operator": "gte", "value": 3}
            ]}"#
        ),
        ["a.md"]
    );
    assert_eq!(
        paths(
            r#"{"operator": "or", "operands": [
              {"key": "priority", "operator": "gte", "value": 9},
              {"key": "priority", "operator": "lte", "value": 1}
            ]}"#
        ),
        ["b.md", "c.md"]
    );
    assert_eq!(
        paths(
            r#"{"operator": "not", "operand": {"key": "status", "operator": "eq", "value": "draft"}}"#
        ),
        ["a.md", "b.md"]
    );
}

#[test]
fn independent_conditions_on_one_array_key_match_different_elements() {
    let tmp = tempfile::tempdir().unwrap();
    let (_qmd, db) = indexed_store(
        tmp.path(),
        &[
            ("wide.md", doc("    priority: [1, 20]\n")),
            ("narrow.md", doc("    priority: [5]\n")),
        ],
    );
    let conn = raw_conn(&db);
    // Document-level semantics: [1, 20] satisfies both conditions via
    // different elements — no implicit same-element fusion.
    assert_eq!(
        match_paths(
            &conn,
            &serde_json::from_str(
                r#"{"operator": "and", "operands": [
                  {"key": "priority", "operator": "gte", "value": 3},
                  {"key": "priority", "operator": "lt", "value": 10}
                ]}"#
            )
            .unwrap()
        ),
        ["narrow.md", "wide.md"]
    );
}

#[test]
fn boolean_values_round_trip_through_membership_operators() {
    let tmp = tempfile::tempdir().unwrap();
    let (_qmd, db) = indexed_store(
        tmp.path(),
        &[
            ("flagged.md", doc("    reviewed: true\n")),
            ("unflagged.md", doc("    reviewed: false\n")),
        ],
    );
    let conn = raw_conn(&db);
    let paths = |f: &str| match_paths(&conn, &serde_json::from_str(f).unwrap());
    assert_eq!(
        paths(r#"{"key": "reviewed", "operator": "in", "value": [true]}"#),
        ["flagged.md"]
    );
    assert_eq!(
        paths(r#"{"key": "reviewed", "operator": "nin", "value": [true]}"#),
        ["unflagged.md"]
    );
}

#[test]
fn sql_injection_payloads_in_keys_and_values_stay_data() {
    let tmp = tempfile::tempdir().unwrap();
    let payload_key = "key'; DROP TABLE documents; --";
    let payload_value = "v'; DROP TABLE documents; --";
    let (_qmd, db) = indexed_store(
        tmp.path(),
        &[(
            "safe.md",
            doc(&format!("    \"{payload_key}\": \"{payload_value}\"\n")),
        )],
    );
    let conn = raw_conn(&db);
    let filter = serde_json::json!({
        "key": payload_key, "operator": "eq", "value": payload_value,
    });
    assert_eq!(match_paths(&conn, &filter), ["safe.md"]);

    // Table survived.
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn compiled_sql_never_interpolates_user_keys_or_values() {
    let parsed = parse(
        r#"{"operator": "and", "operands": [
          {"key": "key'; --", "operator": "eq", "value": "value'; --"},
          {"key": "topics", "operator": "in", "value": ["a'; --"]}
        ]}"#,
    );
    let compiled = qmd::compile_metadata_filter(&parsed, "d");
    assert!(!compiled.sql.contains("'; --"));
    let texts: Vec<&str> = compiled
        .params
        .iter()
        .filter_map(|p| match p {
            rusqlite::types::Value::Text(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    assert!(texts.contains(&"key'; --"));
    assert!(texts.contains(&"value'; --"));
    assert!(texts.contains(&"a'; --"));
}

// =============================================================================
// Compile golden — fixture generated by scripts/gen_metadata_filter_golden.ts
// =============================================================================

#[derive(serde::Deserialize)]
struct GoldenCase {
    name: String,
    input: String,
    sql: Option<String>,
    params: Option<String>,
    error: Option<String>,
}

#[test]
fn metadata_filter_golden_matches_upstream() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/metadata_filter_golden.json")).unwrap();
    let cases: Vec<GoldenCase> = serde_json::from_value(fixture["cases"].clone()).unwrap();
    assert_eq!(cases.len(), 79);
    for case in &cases {
        let input: serde_json::Value = serde_json::from_str(&case.input)
            .unwrap_or_else(|_| panic!("case {} has invalid input json", case.name));
        match (&case.sql, &case.error) {
            (Some(sql), None) => {
                let parsed = parse_metadata_filter(&input)
                    .unwrap_or_else(|e| panic!("case {} failed to parse: {e}", case.name));
                let compiled = qmd::compile_metadata_filter(&parsed, "d");
                assert_eq!(&compiled.sql, sql, "case {} sql", case.name);
                let expected: Vec<serde_json::Value> =
                    serde_json::from_str(case.params.as_deref().unwrap()).unwrap();
                let actual: Vec<serde_json::Value> = compiled
                    .params
                    .iter()
                    .map(|p| match p {
                        rusqlite::types::Value::Text(s) => serde_json::Value::String(s.clone()),
                        rusqlite::types::Value::Integer(i) => {
                            serde_json::Value::Number((*i).into())
                        }
                        rusqlite::types::Value::Real(f) => serde_json::Number::from_f64(*f)
                            .map_or(serde_json::Value::Null, serde_json::Value::Number),
                        _ => serde_json::Value::Null,
                    })
                    .collect();
                assert_eq!(
                    actual.len(),
                    expected.len(),
                    "case {} param count",
                    case.name
                );
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!(json_eq(a, e), "case {} param {a} != {e}", case.name);
                }
            }
            (None, Some(error)) => {
                let err = parse_metadata_filter(&input)
                    .expect_err(&format!("case {} must fail to parse", case.name));
                assert_eq!(&err.to_string(), error, "case {} error text", case.name);
            }
            _ => panic!("case {} has malformed fixture entry", case.name),
        }
    }
}
