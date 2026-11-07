//! Recursive metadata-filter AST, strict runtime validation, and
//! parameterized SQL compilation — a port of upstream
//! `src/metadata-filter.ts`.
//!
//! The filter has one canonical, `operator`-discriminated recursive shape
//! shared by every public search surface (CLI, SDK, MCP, HTTP):
//!
//! ```json
//! { "operator": "and", "operands": [ { "key": "status", "operator": "eq", "value": "published" } ] }
//! ```
//!
//! [`parse_metadata_filter`] is the boundary where untrusted JSON enters
//! (upstream `parseMetadataFilter(input: unknown)`); it produces a
//! validated [`MetadataFilter`]. [`compile_metadata_filter`] turns that
//! AST into correlated `EXISTS`/`NOT EXISTS` subqueries over
//! `document_metadata_values` with every user value bound as a parameter —
//! metadata keys and values are data, never SQL.
//!
//! Deviation: upstream's `MetadataFilter` is a structurally-typed union, so
//! nothing stops a caller from combining an operator with an incompatible
//! value shape. The Rust enum makes operator/value pairing part of the
//! type; filters built through [`parse_metadata_filter`] always satisfy
//! the invariants [`compile_metadata_filter`] relies on (non-empty groups
//! and membership arrays, homogeneous members, string/number operands for
//! ordered comparisons).

use std::collections::HashSet;
use std::fmt;

use rusqlite::types::Value as SqlValue;
use serde_json::Value as JsonValue;

use super::{METADATA_LIMITS, MetadataScalar};
use crate::chunk::utf16_len;

// =============================================================================
// Public types
// =============================================================================

/// A validated metadata filter — upstream `MetadataFilter`
/// (metadata-filter.ts:24-43), expressed as one Rust enum per
/// operator/value combination so incompatible pairs cannot be built.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataFilter {
    /// `{ "operator": "and", "operands": [...] }` — every operand must match.
    And {
        /// Sub-filters (non-empty, ≤ `max_group_operands` after parse).
        operands: Vec<Self>,
    },
    /// `{ "operator": "or", "operands": [...] }` — any operand may match.
    Or {
        /// Sub-filters (non-empty, ≤ `max_group_operands` after parse).
        operands: Vec<Self>,
    },
    /// `{ "operator": "not", "operand": {...} }`.
    Not {
        /// The negated sub-filter.
        operand: Box<Self>,
    },
    /// `{ key, "operator": "eq", value }` — any value row equals `value`.
    Eq {
        /// Metadata key.
        key: String,
        /// Scalar operand.
        value: MetadataScalar,
    },
    /// `{ key, "operator": "ne", value }` — a same-type value row exists
    /// AND no same-type row equals `value` (missing keys never match —
    /// deliberate upstream semantic, metadata-filter.ts:314-325).
    Ne {
        /// Metadata key.
        key: String,
        /// Scalar operand.
        value: MetadataScalar,
    },
    /// `{ key, "operator": "gt", value }` — `value` is string or number.
    Gt {
        /// Metadata key.
        key: String,
        /// Scalar operand (never [`MetadataScalar::Bool`] after parse).
        value: MetadataScalar,
    },
    /// `{ key, "operator": "gte", value }`.
    Gte {
        /// Metadata key.
        key: String,
        /// Scalar operand (never [`MetadataScalar::Bool`] after parse).
        value: MetadataScalar,
    },
    /// `{ key, "operator": "lt", value }`.
    Lt {
        /// Metadata key.
        key: String,
        /// Scalar operand (never [`MetadataScalar::Bool`] after parse).
        value: MetadataScalar,
    },
    /// `{ key, "operator": "lte", value }`.
    Lte {
        /// Metadata key.
        key: String,
        /// Scalar operand (never [`MetadataScalar::Bool`] after parse).
        value: MetadataScalar,
    },
    /// `{ key, "operator": "in", value: [...] }` — any value row is a member.
    In {
        /// Metadata key.
        key: String,
        /// Homogeneous members (non-empty, deduplicated after parse).
        values: Vec<MetadataScalar>,
    },
    /// `{ key, "operator": "nin", value: [...] }` — a same-type value row
    /// exists AND no row is a member (same missing-key semantics as
    /// [`MetadataFilter::Ne`]).
    Nin {
        /// Metadata key.
        key: String,
        /// Homogeneous members (non-empty, deduplicated after parse).
        values: Vec<MetadataScalar>,
    },
    /// `{ key, "operator": "all", value: [...] }` — every member exists as
    /// a value row (different rows may satisfy different members).
    All {
        /// Metadata key.
        key: String,
        /// Homogeneous members (non-empty, deduplicated after parse).
        values: Vec<MetadataScalar>,
    },
    /// `{ key, "operator": "exists", value: bool }`.
    Exists {
        /// Metadata key.
        key: String,
        /// `true` = key present, `false` = key absent.
        value: bool,
    },
}

/// The compiled predicate — upstream `CompiledMetadataFilter`
/// (metadata-filter.ts:45-48). `sql` slots into a `WHERE` clause of a
/// query that already aliases `documents`; `params` bind in order.
#[derive(Debug)]
#[non_exhaustive]
pub struct CompiledMetadataFilter {
    /// SQL predicate text.
    pub sql: String,
    /// Bind parameters for the `?` placeholders, in order.
    pub params: Vec<SqlValue>,
}

/// Raised by [`parse_metadata_filter`] with the JSON path of the failing
/// node — upstream `MetadataFilterError` (metadata-filter.ts:51-59).
#[derive(Debug)]
#[non_exhaustive]
pub struct MetadataFilterError {
    /// JSON path of the offending node (`$`, `$.operands[1]`, …).
    pub path: String,
    /// Failure detail (without the `Invalid metadata filter at …` prefix).
    pub message: String,
}

impl fmt::Display for MetadataFilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Invalid metadata filter at {}: {}",
            self.path, self.message
        )
    }
}

impl std::error::Error for MetadataFilterError {}

impl From<MetadataFilterError> for crate::Error {
    fn from(e: MetadataFilterError) -> Self {
        Self::InvalidInput {
            reason: e.to_string(),
        }
    }
}

// =============================================================================
// Limits
// =============================================================================

/// Defensive limits for recursive filters from untrusted callers —
/// upstream `METADATA_FILTER_LIMITS` (metadata-filter.ts:66-73).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct MetadataFilterLimits {
    /// Maximum nesting depth of `and`/`or`/`not` nodes.
    pub max_depth: usize,
    /// Maximum total AST nodes.
    pub max_nodes: usize,
    /// Maximum operands in one `and`/`or` group.
    pub max_group_operands: usize,
    /// Maximum values in one `in`/`nin`/`all` array.
    pub max_membership_values: usize,
    /// UTF-8 byte cap per metadata key (inherited from
    /// [`METADATA_LIMITS`]).
    pub max_key_bytes: usize,
    /// UTF-16 code-unit cap per string value (inherited from
    /// [`METADATA_LIMITS`]).
    pub max_string_length: usize,
}

/// Upstream `METADATA_FILTER_LIMITS` values (metadata-filter.ts:66-73).
pub const METADATA_FILTER_LIMITS: MetadataFilterLimits = MetadataFilterLimits {
    max_depth: 16,
    max_nodes: 256,
    max_group_operands: 32,
    max_membership_values: 64,
    max_key_bytes: METADATA_LIMITS.max_key_bytes,
    max_string_length: METADATA_LIMITS.max_string_length,
};

const ORDERED_OPERATORS: [&str; 4] = ["gt", "gte", "lt", "lte"];
const MEMBERSHIP_OPERATORS: [&str; 3] = ["in", "nin", "all"];
const CONDITION_OPERATORS: [&str; 10] = [
    "eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "all", "exists",
];
const ALL_OPERATORS: [&str; 13] = [
    "and", "or", "not", "eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "all", "exists",
];

// =============================================================================
// Validation — upstream parseMetadataFilter (metadata-filter.ts:94-266)
// =============================================================================

/// Strictly validate an untrusted JSON value as a [`MetadataFilter`] —
/// upstream `parseMetadataFilter` (metadata-filter.ts:94-97).
///
/// Rejects unknown operators, unknown properties, operator-incompatible
/// values, and inputs exceeding [`METADATA_FILTER_LIMITS`]. Membership
/// value arrays are canonicalized by de-duplicating while preserving
/// order (upstream `Array.from(new Set(…))`).
///
/// With `serde_json`'s `preserve_order` feature, object iteration order
/// matches JS `Object.keys`, so when several unknown properties are
/// present the error names the same one upstream would.
///
/// # Errors
/// [`MetadataFilterError`] with the failing node's JSON path.
pub fn parse_metadata_filter(input: &JsonValue) -> Result<MetadataFilter, MetadataFilterError> {
    let mut state = ParseState { nodes: 0 };
    parse_filter_node(input, "$", 1, &mut state)
}

/// `FilterParseState` (metadata-filter.ts:86).
struct ParseState {
    nodes: usize,
}

fn fail<T>(path: &str, message: impl Into<String>) -> Result<T, MetadataFilterError> {
    Err(MetadataFilterError {
        path: path.to_owned(),
        message: message.into(),
    })
}

/// Upstream `parseFilterNode` (metadata-filter.ts:99-130). Check order is
/// load-bearing for error parity: depth → node count → object shape →
/// `operator` type.
fn parse_filter_node(
    input: &JsonValue,
    path: &str,
    depth: usize,
    state: &mut ParseState,
) -> Result<MetadataFilter, MetadataFilterError> {
    if depth > METADATA_FILTER_LIMITS.max_depth {
        return fail(
            path,
            format!(
                "exceeds maximum nesting depth of {}",
                METADATA_FILTER_LIMITS.max_depth
            ),
        );
    }

    state.nodes += 1;
    if state.nodes > METADATA_FILTER_LIMITS.max_nodes {
        return fail(
            path,
            format!(
                "exceeds maximum of {} nodes",
                METADATA_FILTER_LIMITS.max_nodes
            ),
        );
    }

    let JsonValue::Object(node) = input else {
        return fail(path, "each filter node must be an object");
    };

    let operator = node.get("operator").and_then(JsonValue::as_str);
    let Some(operator) = operator else {
        return fail(path, "missing 'operator' property");
    };

    match operator {
        "and" | "or" => parse_filter_group(node, operator, path, depth, state),
        "not" => parse_filter_negation(node, path, depth, state),
        op if CONDITION_OPERATORS.contains(&op) => parse_filter_condition(node, op, path),
        _ => fail(
            path,
            format!(
                "unknown operator '{operator}' — expected one of: {}",
                ALL_OPERATORS.join(", ")
            ),
        ),
    }
}

/// Upstream `parseFilterGroup` (metadata-filter.ts:132-157).
fn parse_filter_group(
    node: &serde_json::Map<String, JsonValue>,
    operator: &str,
    path: &str,
    depth: usize,
    state: &mut ParseState,
) -> Result<MetadataFilter, MetadataFilterError> {
    reject_unknown_properties(node, &["operator", "operands"], path)?;

    let operands = node.get("operands");
    let Some(JsonValue::Array(operands)) = operands else {
        return fail(path, format!("'{operator}' requires an 'operands' array"));
    };
    if operands.is_empty() {
        return fail(
            path,
            format!("'{operator}' requires a non-empty 'operands' array"),
        );
    }
    if operands.len() > METADATA_FILTER_LIMITS.max_group_operands {
        return fail(
            path,
            format!(
                "'{operator}' exceeds maximum of {} operands",
                METADATA_FILTER_LIMITS.max_group_operands
            ),
        );
    }

    let parsed: Result<Vec<MetadataFilter>, MetadataFilterError> = operands
        .iter()
        .enumerate()
        .map(|(i, operand)| {
            parse_filter_node(operand, &format!("{path}.operands[{i}]"), depth + 1, state)
        })
        .collect();
    let filters = parsed?;
    Ok(if operator == "and" {
        MetadataFilter::And { operands: filters }
    } else {
        MetadataFilter::Or { operands: filters }
    })
}

/// Upstream `parseFilterNegation` (metadata-filter.ts:159-175).
fn parse_filter_negation(
    node: &serde_json::Map<String, JsonValue>,
    path: &str,
    depth: usize,
    state: &mut ParseState,
) -> Result<MetadataFilter, MetadataFilterError> {
    reject_unknown_properties(node, &["operator", "operand"], path)?;

    // `"operand" in node` — presence, not truthiness; a null operand
    // fails the object check one level down (metadata-filter.ts:167).
    if !node.contains_key("operand") {
        return fail(path, "'not' requires exactly one 'operand'");
    }

    let operand = parse_filter_node(
        &node["operand"],
        &format!("{path}.operand"),
        depth + 1,
        state,
    )?;
    Ok(MetadataFilter::Not {
        operand: Box::new(operand),
    })
}

/// Upstream `parseFilterCondition` (metadata-filter.ts:177-214).
fn parse_filter_condition(
    node: &serde_json::Map<String, JsonValue>,
    operator: &str,
    path: &str,
) -> Result<MetadataFilter, MetadataFilterError> {
    reject_unknown_properties(node, &["key", "operator", "value"], path)?;

    let key = node.get("key").and_then(JsonValue::as_str);
    let Some(key) = key.filter(|k| !k.is_empty()) else {
        return fail(
            path,
            format!("'{operator}' requires a non-empty string 'key'"),
        );
    };
    // `Buffer.byteLength(key, "utf-8")` — Rust `str::len` is already
    // the UTF-8 byte count.
    if key.len() > METADATA_FILTER_LIMITS.max_key_bytes {
        return fail(
            path,
            format!(
                "'key' exceeds {} bytes",
                METADATA_FILTER_LIMITS.max_key_bytes
            ),
        );
    }

    if !node.contains_key("value") {
        return fail(path, format!("'{operator}' requires a 'value'"));
    }
    let value = &node["value"];
    let key = key.to_owned();

    if operator == "exists" {
        let Some(value) = value.as_bool() else {
            return fail(
                &format!("{path}.value"),
                "'exists' requires a boolean value",
            );
        };
        return Ok(MetadataFilter::Exists { key, value });
    }

    if MEMBERSHIP_OPERATORS.contains(&operator) {
        let values = parse_membership_values(value, operator, path)?;
        return Ok(match operator {
            "in" => MetadataFilter::In { key, values },
            "nin" => MetadataFilter::Nin { key, values },
            _ => MetadataFilter::All { key, values },
        });
    }

    // Comparison operators: eq, ne, gt, gte, lt, lte.
    let scalar = parse_scalar_value(value, &format!("{path}.value"))?;
    if ORDERED_OPERATORS.contains(&operator) && matches!(scalar, MetadataScalar::Bool(_)) {
        return fail(
            &format!("{path}.value"),
            format!("'{operator}' requires a string or number value"),
        );
    }
    Ok(match operator {
        "eq" => MetadataFilter::Eq { key, value: scalar },
        "ne" => MetadataFilter::Ne { key, value: scalar },
        "gt" => MetadataFilter::Gt { key, value: scalar },
        "gte" => MetadataFilter::Gte { key, value: scalar },
        "lt" => MetadataFilter::Lt { key, value: scalar },
        _ => MetadataFilter::Lte { key, value: scalar },
    })
}

/// Upstream `parseMembershipValues` (metadata-filter.ts:216-241):
/// non-empty homogeneous scalar array, `Set`-deduplicated preserving
/// first-seen order.
fn parse_membership_values(
    value: &JsonValue,
    operator: &str,
    path: &str,
) -> Result<Vec<MetadataScalar>, MetadataFilterError> {
    let JsonValue::Array(value) = value else {
        return fail(
            &format!("{path}.value"),
            format!("'{operator}' requires an array value"),
        );
    };
    if value.is_empty() {
        return fail(
            &format!("{path}.value"),
            format!("'{operator}' requires a non-empty array value"),
        );
    }
    if value.len() > METADATA_FILTER_LIMITS.max_membership_values {
        return fail(
            &format!("{path}.value"),
            format!(
                "'{operator}' exceeds maximum of {} values",
                METADATA_FILTER_LIMITS.max_membership_values
            ),
        );
    }

    let scalars: Result<Vec<MetadataScalar>, MetadataFilterError> = value
        .iter()
        .enumerate()
        .map(|(i, element)| parse_scalar_value(element, &format!("{path}.value[{i}]")))
        .collect();
    let scalars = scalars?;

    let all = |f: fn(&MetadataScalar) -> bool| scalars.iter().all(f);
    if all(|s| matches!(s, MetadataScalar::String(_)))
        || all(|s| matches!(s, MetadataScalar::Number(_)))
        || all(|s| matches!(s, MetadataScalar::Bool(_)))
    {
        // JS `Set` uses SameValueZero: `0` and `-0` dedupe (first-seen
        // wins); NaN never reaches here (finite check).
        return Ok(dedup_scalars(scalars));
    }
    fail(
        &format!("{path}.value"),
        format!("'{operator}' requires a homogeneous array of one scalar type"),
    )
}

/// Upstream `parseScalarValue` (metadata-filter.ts:243-258): string
/// (≤ `max_string_length` UTF-16 units), finite number, or boolean.
fn parse_scalar_value(
    value: &JsonValue,
    path: &str,
) -> Result<MetadataScalar, MetadataFilterError> {
    match value {
        JsonValue::String(s) => {
            if utf16_len(s) > METADATA_FILTER_LIMITS.max_string_length {
                return fail(
                    path,
                    format!(
                        "string exceeds {} characters",
                        METADATA_FILTER_LIMITS.max_string_length
                    ),
                );
            }
            Ok(MetadataScalar::String(s.clone()))
        }
        JsonValue::Number(n) => {
            // serde_json numbers are always f64-representable, so a
            // None here is unreachable; the finite check still fires
            // for programmatically built non-finite Values.
            let v = n.as_f64().unwrap_or(f64::INFINITY);
            if !v.is_finite() {
                return fail(path, "numbers must be finite");
            }
            Ok(MetadataScalar::Number(v))
        }
        JsonValue::Bool(b) => Ok(MetadataScalar::Bool(*b)),
        _ => fail(path, "expected a string, number, or boolean"),
    }
}

/// Upstream `rejectUnknownProperties` (metadata-filter.ts:260-266).
/// Iteration order is insertion order (JS `Object.keys`), so the first
/// unknown property reported matches upstream's.
fn reject_unknown_properties(
    node: &serde_json::Map<String, JsonValue>,
    allowed: &[&str],
    path: &str,
) -> Result<(), MetadataFilterError> {
    for property in node.keys() {
        if !allowed.contains(&property.as_str()) {
            return fail(
                path,
                format!(
                    "unknown property '{property}' — allowed: {}",
                    allowed.join(", ")
                ),
            );
        }
    }
    Ok(())
}

/// `Set`-dedup over scalars under JS `SameValueZero` semantics
/// (metadata-filter.ts:232-238). Strings and booleans compare by value;
/// numbers compare so that `0`/`-0` are the same element — the
/// canonicalization `v + 0.0` maps `-0.0` to `+0.0` bits.
fn dedup_scalars(scalars: Vec<MetadataScalar>) -> Vec<MetadataScalar> {
    #[derive(Hash, PartialEq, Eq)]
    enum Key {
        Str(String),
        Num(u64),
        Bool(bool),
    }
    let key = |s: &MetadataScalar| match s {
        MetadataScalar::String(s) => Key::Str(s.clone()),
        MetadataScalar::Number(n) => Key::Num((n + 0.0).to_bits()),
        MetadataScalar::Bool(b) => Key::Bool(*b),
    };
    let mut seen = HashSet::new();
    scalars
        .into_iter()
        .filter(|s| seen.insert(key(s)))
        .collect()
}

// =============================================================================
// SQL compilation — upstream compileMetadataFilter (metadata-filter.ts:278-356)
// =============================================================================

/// Compile a validated filter into one parameterized SQL predicate
/// correlated against a documents-table alias (e.g. `d`) — upstream
/// `compileMetadataFilter` (metadata-filter.ts:278-282).
///
/// All keys and values are bound parameters. `documents_alias` is
/// interpolated verbatim into the SQL (upstream `documentsAlias`), so it
/// must be a trusted identifier — never user input. The caller is
/// responsible for restricting the surrounding query to active documents
/// with current, error-free metadata extraction.
///
/// Filters constructed through [`parse_metadata_filter`] satisfy every
/// invariant this relies on; manually built filters with empty `operands`
/// or `values` emit degenerate SQL (`(…)` over zero clauses) rather than
/// panicking — states upstream cannot produce either.
#[must_use]
pub fn compile_metadata_filter(
    filter: &MetadataFilter,
    documents_alias: &str,
) -> CompiledMetadataFilter {
    let mut params = Vec::new();
    let sql = compile_filter_node(filter, documents_alias, &mut params);
    CompiledMetadataFilter { sql, params }
}

/// Upstream `compileFilterNode` (metadata-filter.ts:284-356).
fn compile_filter_node(filter: &MetadataFilter, alias: &str, params: &mut Vec<SqlValue>) -> String {
    match filter {
        MetadataFilter::And { operands } | MetadataFilter::Or { operands } => {
            let joiner = if matches!(filter, MetadataFilter::And { .. }) {
                " AND "
            } else {
                " OR "
            };
            let parts: Vec<String> = operands
                .iter()
                .map(|operand| compile_filter_node(operand, alias, params))
                .collect();
            format!("({})", parts.join(joiner))
        }

        MetadataFilter::Not { operand } => {
            format!("NOT {}", compile_filter_node(operand, alias, params))
        }

        MetadataFilter::Exists { key, value } => {
            params.push(key.clone().into());
            let exists = value_exists_sql(alias, "mv.key = ?");
            if *value {
                exists
            } else {
                format!("NOT {exists}")
            }
        }

        MetadataFilter::Eq { key, value }
        | MetadataFilter::Gt { key, value }
        | MetadataFilter::Gte { key, value }
        | MetadataFilter::Lt { key, value }
        | MetadataFilter::Lte { key, value } => {
            let sql_operator = match filter {
                MetadataFilter::Eq { .. } => "=",
                MetadataFilter::Gt { .. } => ">",
                MetadataFilter::Gte { .. } => ">=",
                MetadataFilter::Lt { .. } => "<",
                _ => "<=",
            };
            params.push(key.clone().into());
            params.push(bind_scalar(value));
            value_exists_sql(
                alias,
                &format!(
                    "mv.key = ? AND mv.value_type = '{}' AND mv.{} {sql_operator} ?",
                    value_type_of(value),
                    value_column_of(value),
                ),
            )
        }

        MetadataFilter::Ne { key, value } => {
            // Key must have at least one same-type value, and no same-type
            // value may equal the operand. Missing keys and type
            // mismatches do not match (metadata-filter.ts:314-325).
            let value_type = value_type_of(value);
            params.push(key.clone().into());
            let present_sql = value_exists_sql(
                alias,
                &format!("mv.key = ? AND mv.value_type = '{value_type}'"),
            );
            params.push(key.clone().into());
            params.push(bind_scalar(value));
            let equal_sql = value_exists_sql(
                alias,
                &format!(
                    "mv.key = ? AND mv.value_type = '{value_type}' AND mv.{} = ?",
                    value_column_of(value),
                ),
            );
            format!("({present_sql} AND NOT {equal_sql})")
        }

        MetadataFilter::In { key, values } | MetadataFilter::Nin { key, values } => {
            // Parse guarantees non-empty; an empty manual build degrades
            // to `IN ()`, which SQLite evaluates as false.
            let (value_type, column) = values.first().map_or(("string", "text_value"), |v| {
                (value_type_of(v), value_column_of(v))
            });
            let placeholders = values.iter().map(|_| "?").collect::<Vec<_>>().join(", ");

            if matches!(filter, MetadataFilter::In { .. }) {
                params.push(key.clone().into());
                params.extend(values.iter().map(bind_scalar));
                return value_exists_sql(
                    alias,
                    &format!(
                        "mv.key = ? AND mv.value_type = '{value_type}' AND mv.{column} IN ({placeholders})"
                    ),
                );
            }

            params.push(key.clone().into());
            let present_sql = value_exists_sql(
                alias,
                &format!("mv.key = ? AND mv.value_type = '{value_type}'"),
            );
            params.push(key.clone().into());
            params.extend(values.iter().map(bind_scalar));
            let member_sql = value_exists_sql(
                alias,
                &format!(
                    "mv.key = ? AND mv.value_type = '{value_type}' AND mv.{column} IN ({placeholders})"
                ),
            );
            format!("({present_sql} AND NOT {member_sql})")
        }

        MetadataFilter::All { key, values } => {
            let (value_type, column) = values.first().map_or(("string", "text_value"), |v| {
                (value_type_of(v), value_column_of(v))
            });
            let member_sqls: Vec<String> = values
                .iter()
                .map(|element| {
                    params.push(key.clone().into());
                    params.push(bind_scalar(element));
                    value_exists_sql(
                        alias,
                        &format!(
                            "mv.key = ? AND mv.value_type = '{value_type}' AND mv.{column} = ?"
                        ),
                    )
                })
                .collect();
            format!("({})", member_sqls.join(" AND "))
        }
    }
}

/// Upstream `buildValueExistsSql` (metadata-filter.ts:358-360).
fn value_exists_sql(alias: &str, condition_sql: &str) -> String {
    format!(
        "EXISTS (SELECT 1 FROM document_metadata_values mv WHERE mv.document_id = {alias}.id AND {condition_sql})"
    )
}

/// Upstream `valueTypeOf` (metadata-filter.ts:362-364).
const fn value_type_of(scalar: &MetadataScalar) -> &'static str {
    match scalar {
        MetadataScalar::String(_) => "string",
        MetadataScalar::Number(_) => "number",
        MetadataScalar::Bool(_) => "boolean",
    }
}

/// Upstream `valueColumnOf` (metadata-filter.ts:366-370).
const fn value_column_of(scalar: &MetadataScalar) -> &'static str {
    match scalar {
        MetadataScalar::String(_) => "text_value",
        MetadataScalar::Number(_) => "number_value",
        MetadataScalar::Bool(_) => "boolean_value",
    }
}

/// Upstream `bindScalar` (metadata-filter.ts:372-375): booleans bind as
/// `1`/`0` (matching `boolean_value` column storage), strings and
/// numbers bind as-is.
fn bind_scalar(scalar: &MetadataScalar) -> SqlValue {
    match scalar {
        MetadataScalar::Bool(b) => SqlValue::Integer(i64::from(*b)),
        MetadataScalar::Number(n) => SqlValue::Real(*n),
        MetadataScalar::String(s) => SqlValue::Text(s.clone()),
    }
}
