//! Tool handlers — ports upstream `tobi/qmd` `src/mcp/server.ts`
//! (v2.8.3) `query`, `get`, `multi_get`, `status` handlers. Results are
//! raw `serde_json::Value` so the `Service` interception layer can emit
//! them as `CustomResult` (upstream's `resource` content blocks carry
//! `name`/`title` beyond rmcp's typed `ResourceContents`).

use qmd::metadata::filter::parse_metadata_filter;
use qmd::query::{ExpandedQuery, QueryKind};
use qmd::{
    GetOptions, LineRange, MultiGetEntry, MultiGetOptions, Qmd, SearchInput, SearchOptions,
    add_line_numbers, encode_qmd_path, extract_snippet,
};
use rmcp::ErrorData;
use rmcp::model::{CallToolResult, ContentBlock};
use serde_json::{Value, json};

const MULTI_GET_MAX_BYTES: usize = 65_536; // DEFAULT_MULTI_GET_MAX_BYTES

/// JS number args arrive as `f64`; these saturate like JS semantics
/// need. The `as` casts are the only place truncation/sign-loss is
/// intentional.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "JS number → usize; negative/non-finite saturate to 0"
)]
const fn f64_usize(n: f64) -> usize {
    if n.is_finite() {
        n.max(0.0) as usize
    } else {
        0
    }
}

/// Same, for `u64`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "JS number → u64; negative/non-finite saturate to 0"
)]
const fn f64_u64(n: f64) -> u64 {
    if n.is_finite() { n.max(0.0) as u64 } else { 0 }
}

/// `MetadataValue` → JSON (no `Serialize` impl on the enum).
fn metadata_value_json(v: &qmd::metadata::MetadataValue) -> Value {
    use qmd::metadata::MetadataValue as M;
    match v {
        M::String(s) => json!(s),
        M::Number(n) => json!(n),
        M::Bool(b) => json!(b),
        M::Strings(ss) => json!(ss),
        M::Numbers(nn) => json!(nn),
        M::Bools(bb) => json!(bb),
    }
}

/// Upstream `isError` tool result (server.ts `isError: true` paths).
fn error_result(text: &str) -> Value {
    json!({ "content": [{ "type": "text", "text": text }], "isError": true })
}

fn store_err(e: &qmd::Error) -> ErrorData {
    ErrorData::internal_error(e.to_string(), None)
}

/// `tools/call` dispatch for all four tools (upstream `server.ts:225-620`).
/// Unknown names keep rmcp's dispatch error shape (`invalid_params,
/// "tool not found"`).
///
/// # Errors
/// [`ErrorData`] on store failures; validation and lookup failures are
/// returned inside the result like upstream, not as protocol errors.
pub(crate) fn call(qmd: &Qmd, name: &str, args: &Value) -> Result<Value, ErrorData> {
    match name {
        "status" => status(qmd),
        "query" => query(qmd, args),
        "get" => get(qmd, args),
        "multi_get" => multi_get(qmd, args),
        _ => Err(ErrorData::invalid_params("tool not found", None)),
    }
}

/// `status` — text summary + `structuredContent` (server.ts:590-620).
fn status(qmd: &Qmd) -> Result<Value, ErrorData> {
    let r = status_result(qmd).map_err(|e| store_err(&e))?;
    let mut v =
        serde_json::to_value(r).map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
    // rmcp serializes its 2026 envelope fields (`resultType` etc.) onto
    // the typed result — the caller stamps them per-era instead.
    if let Some(o) = v.as_object_mut() {
        for k in ["resultType", "ttlMs", "cacheScope", "_meta"] {
            o.remove(k);
        }
    }
    Ok(v)
}

/// `status` — text summary plus `structuredContent` equal to upstream's
/// `getStatus()` shape (mcp/server.ts:553-580).
fn status_result(qmd: &Qmd) -> Result<CallToolResult, qmd::Error> {
    let s = qmd.status()?;
    let mut summary = vec![
        "QMD Index Status:".to_owned(),
        format!("  Total documents: {}", s.total_documents),
        format!("  Needs embedding: {}", s.needs_embedding),
        format!(
            "  Vector index: {}",
            if s.has_vector_index { "yes" } else { "no" }
        ),
        format!("  Collections: {}", s.collections.len()),
    ];
    for col in &s.collections {
        // JS prints `null` for a missing path (String(null)).
        let path = col.path.as_deref().unwrap_or("null");
        summary.push(format!(
            "    - {}: {} ({} docs)",
            col.name, path, col.documents
        ));
    }
    let mut r = CallToolResult::success(vec![ContentBlock::text(summary.join("\n"))]);
    r.structured_content = Some(json!({
        "totalDocuments": s.total_documents,
        "needsEmbedding": s.needs_embedding,
        "hasVectorIndex": s.has_vector_index,
        "pendingMetadata": s.pending_metadata,
        "collections": s.collections.iter().map(|c| json!({
            "name": c.name,
            "path": c.path,
            "pattern": c.pattern,
            "documents": c.documents,
            "lastUpdated": c.last_updated,
        })).collect::<Vec<_>>(),
    }));
    // Upstream emits no `isError` field on success.
    r.is_error = None;
    Ok(r)
}

// ---------------------------------------------------------------------------
// query (server.ts:336-430)
// ---------------------------------------------------------------------------

fn query(qmd: &Qmd, args: &Value) -> Result<Value, ErrorData> {
    let v = match validate::query(args) {
        Ok(v) => v,
        Err(msg) => return Ok(error_result(&msg)),
    };
    if let Some(msg) = query_exclusion_error(&v) {
        return Ok(error_result(msg));
    }

    let filter = match &v.filter {
        Some(f) => match parse_metadata_filter(f) {
            Ok(f) => Some(f),
            Err(e) => {
                return Ok(error_result(&format!("Error: {e}")));
            }
        },
        None => None,
    };

    let collections = effective_collections(qmd, v.collections)?;

    let searches = v.searches.unwrap_or_default();
    let input = v.query.as_ref().map_or_else(
        || {
            SearchInput::Expanded(
                searches
                    .iter()
                    .map(|s| ExpandedQuery::new(s.kind, s.query.clone()))
                    .collect(),
            )
        },
        |q| SearchInput::Query(q.clone()),
    );
    let mut opts = SearchOptions::default();
    opts.input = input;
    opts.filter = filter;
    opts.intent = v.intent;
    opts.rerank = v.rerank;
    opts.collections = collections;
    opts.limit = v.limit;
    opts.candidate_limit = v.candidate_limit;
    opts.min_score = v.min_score;
    let results = qmd.search(&opts, &()).map_err(|e| store_err(&e))?;

    // Plain query, else first lex/vec sub-query, for snippet extraction
    // (server.ts:404-408).
    let primary = v.query.unwrap_or_else(|| {
        searches
            .iter()
            .find(|s| s.kind == QueryKind::Lex)
            .or_else(|| searches.iter().find(|s| s.kind == QueryKind::Vec))
            .or_else(|| searches.first())
            .map_or_else(String::new, |s| s.query.clone())
    });

    let items = result_items(&results, &primary, opts.intent.as_deref());
    // formatSearchSummary (server.ts:93-102): one `{docid} {pct}% {file}
    // - {title}` line per result, joined by newlines.
    let summary = if items.is_empty() {
        format!("No results found for \"{primary}\"")
    } else {
        let mut lines = vec![format!(
            "Found {} result{} for \"{primary}\":\n",
            items.len(),
            if items.len() == 1 { "" } else { "s" }
        )];
        for i in &items {
            #[expect(clippy::cast_possible_truncation, reason = "score% within i64 range")]
            let pct = qmd::js::math_round(i["score"].as_f64().unwrap_or_default() * 100.0) as i64;
            lines.push(format!(
                "{} {}% {} - {}",
                i["docid"].as_str().unwrap_or_default(),
                pct,
                i["file"].as_str().unwrap_or_default(),
                i["title"].as_str().unwrap_or_default()
            ));
        }
        lines.join("\n")
    };
    Ok(json!({
        "content": [{ "type": "text", "text": summary }],
        "structuredContent": { "results": items },
    }))
}

// ---------------------------------------------------------------------------
// get (server.ts:433-500)
// ---------------------------------------------------------------------------

fn get(qmd: &Qmd, args: &Value) -> Result<Value, ErrorData> {
    let v = match validate::get(args) {
        Ok(v) => v,
        Err(msg) => return Ok(error_result(&msg)),
    };
    // `:N` / `:N:M` line-range suffixes; explicit args win (server.ts:449-465).
    let mut from = v.from_line;
    let mut max = v.max_lines;
    let mut lookup = v.file.clone();
    if let Some(cap) = suffix_range(&lookup) {
        if from.is_none() {
            from = Some(cap.0);
        }
        if max.is_none() {
            max = Some(cap.1);
        }
        lookup = lookup[..lookup.len() - cap.2].to_owned();
    } else if let Some(cap) = suffix_line(&lookup)
        && from.is_none()
    {
        from = Some(cap.0);
        lookup = lookup[..lookup.len() - cap.1].to_owned();
    }
    let from = from.map(|f| f.max(1));

    let doc = match qmd.get(
        &lookup,
        GetOptions {
            include_body: false,
        },
    ) {
        Err(qmd::Error::DocumentNotFound { similar_files, .. }) => {
            let mut msg = format!("Document not found: {}", v.file);
            if !similar_files.is_empty() {
                let suggestions: Vec<String> =
                    similar_files.iter().map(|f| format!("  - {f}")).collect();
                msg.push_str("\n\nDid you mean one of these?\n");
                msg.push_str(&suggestions.join("\n"));
            }
            return Ok(error_result(&msg));
        }
        Err(qmd::Error::ExcludedByIgnore {
            collection,
            path,
            rule,
            ..
        }) => {
            return Ok(error_result(&format!(
                "Document excluded by ignore rule: {}\nCollection: {collection}\nMatched path: {path}\nIgnore rule: {rule}",
                v.file
            )));
        }
        Err(e) => return Err(store_err(&e)),
        Ok(doc) => doc,
    };

    let body = qmd
        .document_body(
            &doc.virtual_path,
            LineRange {
                from_line: from,
                max_lines: max,
            },
        )
        .map_err(|e| store_err(&e))?
        .unwrap_or_default();
    let mut text = if v.line_numbers {
        add_line_numbers(
            &body,
            usize::try_from(from.unwrap_or(1)).unwrap_or(usize::MAX),
        )
    } else {
        body
    };
    if let Some(ctx) = &doc.context {
        text = format!("<!-- Context: {ctx} -->\n\n{text}");
    }
    Ok(json!({
        "content": [{
            "type": "resource",
            "resource": {
                "uri": format!("qmd://{}", encode_qmd_path(&doc.display_path)),
                "mimeType": "text/markdown",
                "text": text,
            },
        }]
    }))
}

/// `lines.slice(0, maxLines).join("\n")` + the upstream truncation
/// marker (server.ts:552-558).
fn truncate_lines(text: &str, max: u64) -> String {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "line counts and maxLines fit usize on 64-bit targets"
    )]
    let m = max as usize;
    let kept: Vec<&str> = text.split('\n').take(m).collect();
    let remaining = text.split('\n').count() - kept.len();
    let mut out = kept.join("\n");
    if remaining > 0 {
        out.push_str(
            &[
                "\n\n[... truncated ",
                &remaining.to_string(),
                " more lines]",
            ]
            .concat(),
        );
    }
    out
}

/// `:(\d+):(\d+)$` suffix — `(from, count, consumed_len)`.
fn suffix_range(s: &str) -> Option<(u64, u64, usize)> {
    let (head, tail) = s.rsplit_once(':')?;
    let (_, mid) = head.rsplit_once(':')?;
    let from: u64 = mid.parse().ok()?;
    let count: u64 = tail.parse().ok()?;
    Some((from, count, mid.len() + tail.len() + 2))
}

/// `:(\d+)$` suffix — `(line, consumed_len)`.
fn suffix_line(s: &str) -> Option<(u64, usize)> {
    let tail = s.rsplit(':').next()?;
    let line: u64 = tail.parse().ok()?;
    if tail.len() + 1 > s.len() {
        return None;
    }
    Some((line, tail.len() + 1))
}

// ---------------------------------------------------------------------------
// multi_get (server.ts:503-580)
// ---------------------------------------------------------------------------

fn multi_get(qmd: &Qmd, args: &Value) -> Result<Value, ErrorData> {
    let v = match validate::multi_get(args) {
        Ok(v) => v,
        Err(msg) => return Ok(error_result(&msg)),
    };
    let mg = qmd
        .multi_get(
            &v.pattern,
            &MultiGetOptions {
                include_body: true,
                max_bytes: v.max_bytes.unwrap_or(MULTI_GET_MAX_BYTES),
            },
        )
        .map_err(|e| store_err(&e))?;
    if mg.docs.is_empty() && mg.errors.is_empty() {
        return Ok(error_result(&format!(
            "No files matched pattern: {}",
            v.pattern
        )));
    }
    let mut content = Vec::new();
    if !mg.errors.is_empty() {
        content.push(json!({
            "type": "text",
            "text": format!("Errors:\n{}", mg.errors.join("\n")),
        }));
    }
    for entry in &mg.docs {
        match entry {
            MultiGetEntry::Skipped {
                display_path,
                reason,
                ..
            } => {
                content.push(json!({
                    "type": "text",
                    "text": format!("[SKIPPED: {display_path} - {reason}. Use 'qmd_get' with file=\"{display_path}\" to retrieve.]"),
                }));
            }
            MultiGetEntry::Hit(doc) => {
                let mut text = doc.body.clone().unwrap_or_default();
                if let Some(max) = v.max_lines {
                    text = truncate_lines(&text, max);
                }
                if v.line_numbers {
                    text = add_line_numbers(&text, 1);
                }
                if let Some(ctx) = &doc.context {
                    text = format!("<!-- Context: {ctx} -->\n\n{text}");
                }
                content.push(json!({
                    "type": "resource",
                    "resource": {
                        "uri": format!("qmd://{}", encode_qmd_path(&doc.display_path)),
                        "mimeType": "text/markdown",
                        "text": text,
                    },
                }));
            }
            _ => {}
        }
    }
    Ok(json!({ "content": content }))
}

// ---------------------------------------------------------------------------
// Argument validation — upstream zod texts verbatim
// ---------------------------------------------------------------------------

pub(crate) struct QueryArgs {
    pub query: Option<String>,
    pub searches: Option<Vec<SubSearch>>,
    pub limit: usize,
    pub min_score: f64,
    pub candidate_limit: usize,
    pub collections: Option<Vec<String>>,
    pub filter: Option<Value>,
    pub intent: Option<String>,
    pub rerank: bool,
}

pub(crate) struct SubSearch {
    pub kind: QueryKind,
    pub query: String,
}

pub(crate) struct GetArgs {
    pub file: String,
    pub from_line: Option<u64>,
    pub max_lines: Option<u64>,
    pub line_numbers: bool,
}

pub(crate) struct MultiGetArgs {
    pub pattern: String,
    pub max_lines: Option<u64>,
    pub max_bytes: Option<usize>,
    pub line_numbers: bool,
}

/// Mutual-exclusion gate — the upstream isError text or `None`
/// (server.ts:373-383).
fn query_exclusion_error(v: &QueryArgs) -> Option<&'static str> {
    if v.query.is_none() && v.searches.as_ref().is_none_or(Vec::is_empty) {
        return Some(
            "Error: provide either 'query' (plain text) or 'searches' (typed sub-queries)",
        );
    }
    if v.query.is_some() && v.searches.as_ref().is_some_and(|s| !s.is_empty()) {
        return Some("Error: 'query' and 'searches' are mutually exclusive; provide only one");
    }
    None
}

/// `collections ?? defaultCollectionNames`; an explicit empty list
/// becomes `undefined` downstream (server.ts:390-392).
fn effective_collections(
    qmd: &Qmd,
    collections: Option<Vec<String>>,
) -> Result<Option<Vec<String>>, ErrorData> {
    match collections {
        Some(c) if !c.is_empty() => Ok(Some(c)),
        Some(_) => Ok(None),
        None => {
            let names: Vec<String> = qmd
                .collections()
                .map_err(|e| store_err(&e))?
                .into_iter()
                .filter(|c| c.include_by_default)
                .map(|c| c.name)
                .collect();
            Ok(if names.is_empty() { None } else { Some(names) })
        }
    }
}

/// Upstream's per-result mapping (server.ts:410-421): `docid`/`file`/
/// `title`/`score`/`context` + optional `metadata` + `line`/`snippet`.
fn result_items(results: &[qmd::HybridResult], primary: &str, intent: Option<&str>) -> Vec<Value> {
    let mut items = Vec::with_capacity(results.len());
    for r in results {
        // `chunkPos: r.bestChunkPos, chunkLen: r.bestChunk.length`
        // (server.ts:411) — `length` is UTF-16 units.
        let chunk = Some(qmd::ChunkSpan::new(
            r.best_chunk_pos,
            qmd::utf16_len(&r.best_chunk),
        ));
        let snip = extract_snippet(&r.body, primary, 300, chunk, intent);
        // JS emits `1` not `1.0` — integral scores serialize as ints.
        let raw = qmd::js::math_round(r.score * 100.0) / 100.0;
        #[expect(clippy::cast_possible_truncation, reason = "score within i64 range")]
        let score = if raw.fract() == 0.0 {
            json!(raw as i64)
        } else {
            json!(raw)
        };
        let mut item = json!({
            "docid": format!("#{}", r.docid),
            "file": r.display_path,
            "title": r.title,
            "score": score,
            "context": r.context,
        });
        if let Some(obj) = item.as_object_mut() {
            if !r.metadata.is_empty() {
                let md: serde_json::Map<String, Value> = r
                    .metadata
                    .iter()
                    .map(|(k, mv)| (k.to_owned(), metadata_value_json(mv)))
                    .collect();
                obj.insert("metadata".to_owned(), Value::Object(md));
            }
            obj.insert("line".to_owned(), json!(snip.line));
            obj.insert(
                "snippet".to_owned(),
                json!(add_line_numbers(&snip.snippet, snip.line)),
            );
        }
        items.push(item);
    }
    items
}

mod validate {
    use super::{
        GetArgs, MultiGetArgs, QueryArgs, QueryKind, SubSearch, Value, f64_u64, f64_usize,
    };

    /// zod's `received <type>` naming for a present value.
    const fn received(v: &Value) -> &'static str {
        match v {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    }

    /// Upstream SDK text — all zod issues joined by `", "` in schema
    /// order (mcp-DXXb3Vv3.mjs:1432 + recorded transcripts).
    fn invalid(name: &str, issues: &[String]) -> String {
        format!(
            "Input validation error: Invalid arguments for tool {name}: {}",
            issues.join(", ")
        )
    }

    /// One `path: message` issue.
    fn issue(path: &str, msg: &str) -> String {
        format!("{path}: {msg}")
    }

    type Issues = Vec<String>;

    fn obj<'a>(
        name: &str,
        args: &'a Value,
        issues: &mut Issues,
    ) -> Option<&'a serde_json::Map<String, Value>> {
        args.as_object().or_else(|| {
            issues.push(format!(
                "Invalid input: expected object, received {}",
                received(args)
            ));
            let _ = name;
            None
        })
    }

    fn str_field(
        name: &str,
        o: &serde_json::Map<String, Value>,
        key: &str,
        required: bool,
        issues: &mut Issues,
    ) -> Option<String> {
        let _ = name;
        match o.get(key) {
            None => {
                if required {
                    issues.push(issue(
                        key,
                        "Invalid input: expected string, received undefined",
                    ));
                }
                None
            }
            Some(v @ Value::String(_)) => v.as_str().map(str::to_owned),
            Some(v) => {
                issues.push(issue(
                    key,
                    &format!("Invalid input: expected string, received {}", received(v)),
                ));
                None
            }
        }
    }

    fn num_field(
        o: &serde_json::Map<String, Value>,
        key: &str,
        issues: &mut Issues,
    ) -> Option<f64> {
        match o.get(key) {
            None => None,
            Some(v @ Value::Number(_)) => v.as_f64(),
            Some(v) => {
                issues.push(issue(
                    key,
                    &format!("Invalid input: expected number, received {}", received(v)),
                ));
                None
            }
        }
    }

    fn bool_field(
        o: &serde_json::Map<String, Value>,
        key: &str,
        issues: &mut Issues,
    ) -> Option<bool> {
        match o.get(key) {
            None => None,
            Some(Value::Bool(b)) => Some(*b),
            Some(v) => {
                issues.push(issue(
                    key,
                    &format!("Invalid input: expected boolean, received {}", received(v)),
                ));
                None
            }
        }
    }

    fn str_array(
        name: &str,
        o: &serde_json::Map<String, Value>,
        key: &str,
        issues: &mut Issues,
    ) -> Option<Vec<String>> {
        match o.get(key) {
            None => None,
            Some(Value::Array(items)) => {
                let mut out = Vec::with_capacity(items.len());
                for (i, it) in items.iter().enumerate() {
                    match it.as_str() {
                        Some(t) => out.push(t.to_owned()),
                        None => issues.push(issue(
                            &format!("{key}.{i}"),
                            &format!("Invalid input: expected string, received {}", received(it)),
                        )),
                    }
                }
                let _ = name;
                Some(out)
            }
            Some(v) => {
                issues.push(issue(
                    key,
                    &format!("Invalid input: expected array, received {}", received(v)),
                ));
                None
            }
        }
    }

    /// One `searches[i]` element — `{type: lex|vec|hyde, query: string}`.
    /// Pushes every issue on the element (zod reports each field).
    fn sub_search(name: &str, i: usize, it: &Value, issues: &mut Issues) -> Option<SubSearch> {
        const ENUM_ERR: &str = "Invalid option: expected one of \"lex\"|\"vec\"|\"hyde\"";
        let Some(so) = it.as_object() else {
            issues.push(issue(
                &format!("searches.{i}"),
                &format!("Invalid input: expected object, received {}", received(it)),
            ));
            return None;
        };
        let _ = name;
        let kind = if let Some(Value::String(t)) = so.get("type") {
            match t.as_str() {
                "lex" => Some(QueryKind::Lex),
                "vec" => Some(QueryKind::Vec),
                "hyde" => Some(QueryKind::Hyde),
                _ => {
                    issues.push(issue(&format!("searches.{i}.type"), ENUM_ERR));
                    None
                }
            }
        } else {
            issues.push(issue(&format!("searches.{i}.type"), ENUM_ERR));
            None
        };
        let query = match so.get("query") {
            Some(Value::String(q)) => Some(q.clone()),
            Some(v) => {
                issues.push(issue(
                    &format!("searches.{i}.query"),
                    &format!("Invalid input: expected string, received {}", received(v)),
                ));
                None
            }
            None => {
                issues.push(issue(
                    &format!("searches.{i}.query"),
                    "Invalid input: expected string, received undefined",
                ));
                None
            }
        };
        kind.zip(query)
            .map(|(kind, query)| SubSearch { kind, query })
    }

    pub(super) fn query(args: &Value) -> Result<QueryArgs, String> {
        let name = "query";
        let mut issues: Issues = Vec::new();
        let Some(o) = obj(name, args, &mut issues) else {
            return Err(invalid(name, &issues));
        };
        let query = str_field(name, o, "query", false, &mut issues);
        let searches = match o.get("searches") {
            None => None,
            Some(Value::Array(items)) => {
                let mut out = Vec::with_capacity(items.len());
                for (i, it) in items.iter().enumerate() {
                    out.extend(sub_search(name, i, it, &mut issues));
                }
                if items.len() > 10 {
                    issues.push(issue(
                        "searches",
                        "Too big: expected array to have <=10 items",
                    ));
                }
                Some(out)
            }
            Some(v) => {
                issues.push(issue(
                    "searches",
                    &format!("Invalid input: expected array, received {}", received(v)),
                ));
                None
            }
        };
        let limit = num_field(o, "limit", &mut issues);
        let min_score = num_field(o, "minScore", &mut issues);
        let candidate_limit = num_field(o, "candidateLimit", &mut issues);
        let collections = str_array(name, o, "collections", &mut issues);
        let filter = match o.get("filter") {
            None => None,
            Some(v @ Value::Object(_)) => Some(v.clone()),
            Some(v) => {
                issues.push(issue(
                    "filter",
                    &format!("Invalid input: expected object, received {}", received(v)),
                ));
                None
            }
        };
        let intent = str_field(name, o, "intent", false, &mut issues);
        let rerank = bool_field(o, "rerank", &mut issues);
        if !issues.is_empty() {
            return Err(invalid(name, &issues));
        }
        Ok(QueryArgs {
            query,
            searches,
            limit: f64_usize(limit.unwrap_or(10.0)),
            min_score: min_score.unwrap_or(0.0),
            candidate_limit: f64_usize(candidate_limit.unwrap_or(40.0)),
            collections,
            filter,
            intent,
            rerank: rerank.unwrap_or(true),
        })
    }

    pub(super) fn get(args: &Value) -> Result<GetArgs, String> {
        let name = "get";
        let mut issues: Issues = Vec::new();
        let Some(o) = obj(name, args, &mut issues) else {
            return Err(invalid(name, &issues));
        };
        let file = str_field(name, o, "file", true, &mut issues);
        let from_line = num_field(o, "fromLine", &mut issues);
        let max_lines = num_field(o, "maxLines", &mut issues);
        let line_numbers = bool_field(o, "lineNumbers", &mut issues);
        if !issues.is_empty() {
            return Err(invalid(name, &issues));
        }
        Ok(GetArgs {
            file: file.unwrap_or_default(),
            from_line: from_line.map(f64_u64),
            max_lines: max_lines.map(f64_u64),
            line_numbers: line_numbers.unwrap_or(true),
        })
    }

    pub(super) fn multi_get(args: &Value) -> Result<MultiGetArgs, String> {
        let name = "multi_get";
        let mut issues: Issues = Vec::new();
        let Some(o) = obj(name, args, &mut issues) else {
            return Err(invalid(name, &issues));
        };
        let pattern = str_field(name, o, "pattern", true, &mut issues);
        let max_lines = num_field(o, "maxLines", &mut issues);
        let max_bytes = num_field(o, "maxBytes", &mut issues);
        let line_numbers = bool_field(o, "lineNumbers", &mut issues);
        if !issues.is_empty() {
            return Err(invalid(name, &issues));
        }
        Ok(MultiGetArgs {
            pattern: pattern.unwrap_or_default(),
            max_lines: max_lines.map(f64_u64),
            max_bytes: max_bytes.map(f64_usize),
            line_numbers: line_numbers.unwrap_or(true),
        })
    }
}
