//! REST `POST /query` (alias `/search`) — structured search without MCP
//! framing. Ports upstream `src/mcp/server.ts` REST block
//! (server.ts:1047-1102) 1:1, including its exact 400 bodies.
//!
//! Unlike the `query` tool this endpoint never expands a plain query:
//! `searches` is required, and each entry's `type` is passed through
//! unvalidated — anything that is not `lex`/`vec`/`hyde` is dropped by the
//! retrieval pipeline, so a garbage `type` means "no query" here too.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use qmd::metadata::filter::parse_metadata_filter;
use qmd::query::{ExpandedQuery, QueryKind};
use qmd::{Qmd, SearchInput, SearchOptions};
use serde_json::{Value, json};

use crate::http::{AppState, RequestLabel};
use crate::js::{is_truthy, js_string, js_string_or_empty};
use crate::tools;

/// Shared state for the REST handler.
pub(crate) struct RestState {
    pub(crate) qmd: Arc<Qmd>,
    /// `defaultCollectionNames` resolved at startup (server.ts:904).
    pub(crate) default_collections: Vec<String>,
}

/// Axum handler for `POST /query` and `POST /search`.
pub(crate) async fn search(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let rest = &state.rest;
    // server.ts:1057-1060 — `JSON.parse(rawBody)` failure.
    let params: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return error_400("Invalid JSON body"),
    };
    if !params.is_object() {
        return error_400("JSON body must be an object");
    }

    // server.ts:1071-1074.
    let Some(searches) = params.get("searches").and_then(Value::as_array) else {
        return error_400("Missing required field: searches (array)");
    };

    // `s.type` on null/undefined throws a TypeError upstream → the outer
    // catch answers 500. Other primitives yield `undefined`.
    let mut queries = Vec::with_capacity(searches.len());
    for s in searches {
        if s.is_null() {
            // Bun's TypeError wording, not V8's (recorded stderr).
            return internal_error("TypeError: null is not an object (evaluating 's.type')");
        }
        let query = s.get("query").map_or_else(String::new, js_string_or_empty);
        // The newline check runs for *every* type upstream
        // (store.ts:5981-5983) — a multi-line query on a garbage `type`
        // still throws → 500 here.
        if query.contains(['\r', '\n']) {
            // `search.line ? "Line …" : "Structured search"` — falsy
            // values (0, null, false) are "Structured search" too.
            let location = s.get("line").filter(|l| is_truthy(l)).map_or_else(
                || "Structured search".to_owned(),
                |l| format!("Line {}", js_string(l)),
            );
            let ty = s
                .get("type")
                .map_or_else(|| "undefined".to_owned(), js_string);
            return internal_error(format!(
                "{location} ({ty}): queries must be single-line. Remove newline characters."
            ));
        }
        // Unknown types reach `store.search` upstream and fall out of both
        // the lex and vec buckets — i.e. the entry is silently dropped.
        let Some(kind) = s.get("type").and_then(Value::as_str).and_then(|t| match t {
            "lex" => Some(QueryKind::Lex),
            "vec" => Some(QueryKind::Vec),
            "hyde" => Some(QueryKind::Hyde),
            _ => None,
        }) else {
            continue;
        };
        queries.push(ExpandedQuery::new(kind, query));
    }

    // server.ts:1084-1097 — `filter` must be an object, then a valid
    // AST. `params.filter !== undefined` is the only exemption: JSON
    // `null` still reaches the `=== null || !isObject` check → 400.
    let filter = match params.get("filter") {
        Some(f) => {
            if !f.is_object() {
                return error_400("Invalid field: filter (must be an object)");
            }
            match parse_metadata_filter(f) {
                Ok(f) => Some(f),
                Err(e) => return error_400(&e.to_string()),
            }
        }
        None => None,
    };

    // `Array.isArray(params.collections) ? map(String) : defaults`
    // (server.ts:1100-1101); empty → `undefined`.
    let collections = params
        .get("collections")
        .and_then(Value::as_array)
        .map_or_else(
            || rest.default_collections.clone(),
            |c| c.iter().map(js_string).collect::<Vec<_>>(),
        );
    let collections = if collections.is_empty() {
        None
    } else {
        Some(collections)
    };

    let intent = params
        .get("intent")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let intent_for_items = intent.clone();

    let mut opts = SearchOptions::default();
    opts.input = SearchInput::Expanded(queries);
    opts.collections = collections;
    opts.filter = filter;
    opts.intent = intent;
    // `typeof params.rerank === 'boolean' ? params.rerank : undefined` —
    // undefined means "store default", which is `true`.
    opts.rerank = params
        .get("rerank")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    opts.min_score = params
        .get("minScore")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    if let Some(limit) = params.get("limit").and_then(Value::as_f64).map(rest_limit) {
        opts.limit = limit;
    }
    if let Some(cand) = params
        .get("candidateLimit")
        .and_then(Value::as_f64)
        .map(rest_limit)
    {
        opts.candidate_limit = cand;
    }

    let rest = Arc::clone(rest);
    let results = match tokio::task::spawn_blocking(move || rest.qmd.search(&opts, &())).await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return internal_error(e),
        Err(e) => return internal_error(e),
    };

    let primary = primary_query(searches);

    let items = tools::rest_result_items(&results, &primary, intent_for_items.as_deref());
    // server.ts:1103 — `'/query ' + searches.length + ' queries'`,
    // `/query` even on the /search alias.
    let mut resp = Json(json!({ "results": items })).into_response();
    resp.extensions_mut()
        .insert(RequestLabel(format!("/query {} queries", searches.len())));
    resp
}

/// `s?.query` when truthy — the operand of upstream's `||` chain.
fn truthy_query(s: &Value) -> Option<&Value> {
    s.get("query").filter(|q| is_truthy(q))
}

/// `lex?.query || vec?.query || searches[0]?.query || ""` then
/// `String()` — the `||` chain runs on the query *values*, so a falsy
/// query falls through to the next candidate (server.ts:1105-1107).
fn primary_query(searches: &[Value]) -> String {
    let typed = |ty: &str| {
        searches
            .iter()
            .find(|s| s.get("type").and_then(Value::as_str) == Some(ty))
            .and_then(truthy_query)
    };
    typed("lex")
        .or_else(|| typed("vec"))
        .or_else(|| searches.first().and_then(truthy_query))
        .map_or_else(String::new, js_string)
}

/// 400 `{"error": "<msg>"}` (server.ts `{"Content-Type":"application/json"}`).
fn error_400(msg: &str) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({ "error": msg }))).into_response()
}

/// Upstream's catch-all `500 "Internal Server Error"` — plain text, no
/// `Content-Type` (node `res.end` without `writeHead` headers). The
/// handler's `console.error("HTTP handler error:", err)` is kept as a
/// single stderr line (deviation 77 — no Bun source frame).
#[expect(clippy::expect_used, reason = "a static body cannot fail to build")]
fn internal_error(err: impl std::fmt::Display) -> Response {
    use std::io::Write as _;
    drop(writeln!(std::io::stderr(), "HTTP handler error: {err}"));
    Response::builder()
        .status(StatusCode::INTERNAL_SERVER_ERROR)
        .body(axum::body::Body::from("Internal Server Error"))
        .expect("static response")
}

/// `typeof params.limit === 'number' ? params.limit : 10` without a range
/// check: a negative number reaches SQLite's `LIMIT`, where a negative
/// bound means "unlimited" — so negative maps to `usize::MAX` and floats
/// truncate like the SQL cast.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "JS number → SQL LIMIT; negative = unlimited, non-finite = 0"
)]
fn rest_limit(n: f64) -> usize {
    if !n.is_finite() {
        return 0;
    }
    if n < 0.0 {
        return usize::MAX;
    }
    n as usize
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "idiomatic test-code patterns")]
mod tests {
    //! `primaryQuery`'s `||`-on-values chain (server.ts:1105-1107) —
    //! the HTTP-level cases are covered by the golden replay and the
    //! subprocess suite in `qmd-cli`.

    use serde_json::json;

    use super::primary_query;

    #[test]
    fn primary_prefers_lex_then_vec_then_first() {
        assert_eq!(
            primary_query(&[
                json!({"type":"lex","query":"a"}),
                json!({"type":"vec","query":"b"})
            ]),
            "a"
        );
        assert_eq!(
            primary_query(&[
                json!({"type":"vec","query":"b"}),
                json!({"type":"bogus","query":"c"})
            ]),
            "b"
        );
        assert_eq!(primary_query(&[json!({"type":"bogus","query":"c"})]), "c");
    }

    #[test]
    fn primary_falsy_query_falls_through_to_next_candidate() {
        // `lex?.query || vec?.query || searches[0]?.query` — a lex entry
        // with an empty query yields the vec entry's query, not "".
        assert_eq!(
            primary_query(&[
                json!({"type":"lex","query":""}),
                json!({"type":"vec","query":"v"})
            ]),
            "v"
        );
        assert_eq!(
            primary_query(&[
                json!({"type":"lex","query":null}),
                json!({"type":"vec","query":"v"})
            ]),
            "v"
        );
        // All falsy → `|| ""` on searches[0].query.
        assert_eq!(
            primary_query(&[
                json!({"type":"lex","query":""}),
                json!({"type":"vec","query":""})
            ]),
            ""
        );
    }

    #[test]
    fn primary_stringifies_non_string_queries() {
        // `String(primaryQuery)` — a truthy number is stringified.
        assert_eq!(primary_query(&[json!({"type":"lex","query":42})]), "42");
        assert_eq!(primary_query(&[]), "");
    }
}
