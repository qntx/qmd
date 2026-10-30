//! `qmd bench` — the search-quality benchmark harness, porting
//! upstream `src/bench/bench.ts` (fixture parsing, structured-query
//! parser, the four backends, and the run loop) plus the CLI wrapper
//! (upstream `qmd.ts:4788-4810`). Scoring lives in `score.rs`, output
//! shaping in `report.rs` — both byte-exact with upstream.

pub(crate) mod report;
pub(crate) mod score;

use std::io::Write;
use std::time::Instant;

use qmd::query::{ExpandedQuery, QueryKind};
use qmd::{LexOptions, Qmd, SearchInput, SearchOptions, VectorOptions, js};
use report::{BackendResult, QueryResult, Summary};

use crate::Failure;

/// One benchmark fixture query — upstream `BenchmarkQuery` field
/// order; `kind` is JSON `"type"` (omitted in output when absent).
#[derive(Debug, serde::Deserialize)]
pub(crate) struct FixtureQuery {
    pub(crate) id: String,
    pub(crate) query: String,
    #[serde(rename = "type")]
    pub(crate) kind: Option<String>,
    pub(crate) expected_files: Vec<String>,
    pub(crate) expected_in_top_k: usize,
}

/// The fixture file — upstream `BenchmarkFixture`; unknown keys
/// (`description`, `version`) ignored.
#[derive(Debug, serde::Deserialize)]
pub(crate) struct Fixture {
    pub(crate) collection: Option<String>,
    pub(crate) queries: Vec<FixtureQuery>,
}

/// Load + parse the fixture BEFORE opening the store (bench.ts:325-343
/// order): read file → JSON parse → `queries` must be an array →
/// deserialize each element with upstream's error text.
fn load_fixture(path: &str) -> Result<Fixture, Failure> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| Failure::Message(format!("Failed to read fixture {path}: {e}")))?;
    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| Failure::Message(format!("Invalid fixture JSON: {e}")))?;
    let Some(queries) = value.get("queries").and_then(|q| q.as_array()) else {
        return Err(Failure::Message(
            "Invalid fixture: missing 'queries' array".to_owned(),
        ));
    };
    let mut out = Vec::with_capacity(queries.len());
    for (i, q) in queries.iter().enumerate() {
        let fq: FixtureQuery = serde_json::from_value(q.clone())
            .map_err(|e| Failure::Message(format!("Invalid fixture query {i}: {e}")))?;
        out.push(fq);
    }
    // Upstream leaves `collection` loose-typed: `options.collection ??
    // fixture.collection` flows raw into the guard, so a non-string
    // value hits `c.name === collection` (never true) and is printed
    // with JS `String()` coercion — `123` → "Collection not found:
    // 123", `{}` → "[object Object]". Null/missing stay absent.
    let collection = value
        .get("collection")
        .filter(|c| !c.is_null())
        .map(js_string);
    Ok(Fixture {
        collection,
        queries: out,
    })
}

/// JS `String(value)` for fixture scalars (bench.ts:344 `??` keeps
/// the raw value; the guard's template coerces it): numbers and bools
/// stringify, arrays join with `,`, objects are `[object Object]`.
fn js_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_owned(),
        serde_json::Value::Array(a) => a.iter().map(js_string).collect::<Vec<_>>().join(","),
        serde_json::Value::Object(_) => "[object Object]".to_owned(),
    }
}

/// A parsed structured query — the `{searches, intent}` pair upstream
/// returns (bench.ts:37-40).
pub(crate) struct StructuredQuery {
    pub(crate) searches: Vec<ExpandedQuery>,
    pub(crate) intent: Option<String>,
}

/// `parseStructuredQuery` (bench.ts:46-94) — bench's own parser, NOT
/// the library `parse_query_document` (upstream-quirks.md Q34: no
/// `expand:`, different errors, swallowed into a zero row). Returns
/// `None` for a plain query; `Err` carries upstream's message verbatim.
fn parse_structured_query(query: &str) -> Result<Option<StructuredQuery>, String> {
    let lines: Vec<(String, usize)> = query
        .split('\n')
        .enumerate()
        .map(|(i, l)| (js::js_trim(l).to_owned(), i + 1))
        .filter(|(t, _)| !t.is_empty())
        .collect();
    if lines.is_empty() {
        return Ok(None);
    }
    let mut searches: Vec<ExpandedQuery> = Vec::new();
    let mut intent: Option<String> = None;
    for (line, number) in &lines {
        if let Some(rest) = strip_ci_prefix(line, "intent:") {
            if intent.is_some() {
                return Err(format!(
                    "Line {number}: only one intent: line is allowed per benchmark query."
                ));
            }
            let text = js::js_trim(rest);
            if text.is_empty() {
                return Err(format!("Line {number}: intent: must include text."));
            }
            intent = Some(text.to_owned());
            continue;
        }
        let mut matched = false;
        for (prefix, kind) in [
            ("lex:", QueryKind::Lex),
            ("vec:", QueryKind::Vec),
            ("hyde:", QueryKind::Hyde),
        ] {
            if let Some(rest) = strip_ci_prefix(line, prefix) {
                searches.push(typed_query(kind, rest, *number)?);
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }
        if lines.len() == 1 {
            return Ok(None);
        }
        return Err(format!(
            "Line {number} is missing a lex:/vec:/hyde:/intent: prefix."
        ));
    }
    if intent.is_some() && searches.is_empty() {
        return Err(
            "intent: cannot appear alone. Add at least one lex:, vec:, or hyde: line.".to_owned(),
        );
    }
    Ok((!searches.is_empty()).then_some(StructuredQuery { searches, intent }))
}

/// One typed line → `ExpandedQuery` — the `(<type>:) must include
/// text` error carries the JS `trim()`ed text check (bench.ts:78-79).
fn typed_query(kind: QueryKind, rest: &str, number: usize) -> Result<ExpandedQuery, String> {
    let text = js::js_trim(rest);
    if text.is_empty() {
        return Err(format!(
            "Line {number} ({}:) must include text.",
            kind.as_str()
        ));
    }
    Ok(ExpandedQuery::new(kind, text).with_line(number))
}

/// `/^(lex|vec|hyde|intent):\s*/i` — case-insensitive prefix then
/// JS-whitespace (NOT Rust `\s`, whose Unicode set differs from JS).
/// Returns the remainder *after* the prefix only; the JS `\s*` skip is
/// folded into the caller's `js_trim`.
fn strip_ci_prefix<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    line.get(..prefix.len()).and_then(|head| {
        head.eq_ignore_ascii_case(prefix)
            .then(|| &line[prefix.len()..])
    })
}

/// `SearchHooks` no-op — upstream's bench `store.search` fires
/// expansion/embed/rerank callbacks, which upstream ignores.
struct NoopHooks;
impl qmd::SearchHooks for NoopHooks {}

/// `uniqueFiles` (bench.ts:96-105) — dedupe preserving order, ≤limit.
fn unique_files(files: &[String], limit: usize) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for f in files {
        if seen.insert(f.clone()) {
            out.push(f.clone());
            if out.len() >= limit {
                break;
            }
        }
    }
    out
}

/// Search backends — upstream `BACKENDS` order and names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Backend {
    Bm25,
    Vector,
    Hybrid,
    Full,
}

impl Backend {
    /// Upstream's fixed order `["bm25","vector","hybrid","full"]`.
    pub(crate) const ALL: [Self; 4] = [Self::Bm25, Self::Vector, Self::Hybrid, Self::Full];

    const fn name(self) -> &'static str {
        match self {
            Self::Bm25 => "bm25",
            Self::Vector => "vector",
            Self::Hybrid => "hybrid",
            Self::Full => "full",
        }
    }

    /// `backend.run` (bench.ts:107-160). Any parse error propagates —
    /// `run_query` maps it to the zero result like upstream's catch.
    fn run(
        self,
        qmd: &Qmd,
        query: &FixtureQuery,
        limit: usize,
        collection: Option<&str>,
    ) -> Result<Vec<String>, String> {
        let structured = parse_structured_query(&query.query)?;
        let collections = collection.map(|c| vec![c.to_owned()]);
        match self {
            Self::Bm25 => {
                if let Some(sq) = structured {
                    return run_search_lines(
                        qmd,
                        &sq.searches,
                        Kind::Lex,
                        limit,
                        collections.as_ref(),
                    );
                }
                let results = qmd
                    .search_lex(
                        &query.query,
                        &LexOptions {
                            limit: Some(limit),
                            collections,
                            ..LexOptions::default()
                        },
                    )
                    .map_err(|e| e.to_string())?;
                Ok(results.into_iter().map(|r| r.filepath).collect())
            }
            Self::Vector => {
                if let Some(sq) = structured {
                    return run_search_lines(
                        qmd,
                        &sq.searches,
                        Kind::VecHyde,
                        limit,
                        collections.as_ref(),
                    );
                }
                let results = qmd
                    .search_vector(
                        &query.query,
                        &VectorOptions {
                            limit: Some(limit),
                            collections,
                            ..VectorOptions::default()
                        },
                    )
                    .map_err(|e| e.to_string())?;
                Ok(results.into_iter().map(|r| r.filepath).collect())
            }
            Self::Hybrid | Self::Full => {
                let rerank = self == Self::Full;
                let mut opts = SearchOptions::default();
                opts.rerank = rerank;
                opts.limit = limit;
                opts.collections = collections;
                match structured {
                    Some(sq) => {
                        opts.input = SearchInput::Expanded(sq.searches);
                        opts.intent = sq.intent;
                    }
                    None => opts.input = SearchInput::Query(query.query.clone()),
                }
                let results = qmd.search(&opts, &NoopHooks).map_err(|e| e.to_string())?;
                Ok(results.into_iter().map(|r| r.filepath).collect())
            }
        }
    }
}

/// The typed-line filter — `lex`-only (bm25) or `vec|hyde` (vector).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Lex,
    VecHyde,
}

/// The shared `for lex of queries` loop upstream inlines per backend
/// (bench.ts:112-119, 124-131): each typed line searches, results'
/// filepaths concatenate, then `uniqueFiles` dedupes.
fn run_search_lines(
    qmd: &Qmd,
    searches: &[ExpandedQuery],
    kind: Kind,
    limit: usize,
    collections: Option<&Vec<String>>,
) -> Result<Vec<String>, String> {
    let mut files = Vec::new();
    for q in searches {
        let take = match kind {
            Kind::Lex => q.kind == QueryKind::Lex,
            Kind::VecHyde => matches!(q.kind, QueryKind::Vec | QueryKind::Hyde),
        };
        if !take {
            continue;
        }
        let results = match kind {
            Kind::Lex => qmd.search_lex(
                &q.query,
                &LexOptions {
                    limit: Some(limit),
                    collections: collections.cloned(),
                    ..LexOptions::default()
                },
            ),
            Kind::VecHyde => qmd.search_vector(
                &q.query,
                &VectorOptions {
                    limit: Some(limit),
                    collections: collections.cloned(),
                    ..VectorOptions::default()
                },
            ),
        }
        .map_err(|e| e.to_string())?;
        files.extend(results.into_iter().map(|r| r.filepath));
    }
    Ok(unique_files(&files, limit))
}

/// `runQuery` (bench.ts:163-196) — `limit = max(expected_in_top_k,
/// 10)`, elapsed truncated to whole ms; any backend or parser error
/// becomes the zero result.
fn run_query(
    qmd: &Qmd,
    backend: Backend,
    query: &FixtureQuery,
    collection: Option<&str>,
) -> BackendResult {
    let limit = query.expected_in_top_k.max(10);
    let start = Instant::now();
    backend.run(qmd, query, limit, collection).map_or_else(
        |_| {
            BackendResult::error(
                &query.expected_files,
                u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
            )
        },
        |files| {
            BackendResult::new(
                &files,
                &query.expected_files,
                query.expected_in_top_k,
                u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
            )
        },
    )
}

/// `runBenchmark` (bench.ts:325-419) — fixture already parsed by the
/// caller; the store is opened before the guard; progress goes to
/// `err` in text mode; `fixture_path` is printed verbatim in JSON.
pub(crate) fn run_benchmark(
    qmd: &Qmd,
    fixture: &Fixture,
    collection: Option<&str>,
    backends: &[Backend],
    json_mode: bool,
    err: &mut dyn Write,
) -> Result<(Vec<QueryResult>, Vec<Summary>), Failure> {
    let collections: Vec<(String, usize)> = qmd
        .collections()?
        .into_iter()
        .map(|c| (c.name, c.active_count))
        .collect();
    report::assert_collection_ready(&collections, collection)?;

    let mut results = Vec::new();
    for query in &fixture.queries {
        let mut backends_out = Vec::with_capacity(backends.len());
        for backend in backends {
            if !json_mode {
                drop(write!(err, "  {} / {}...", query.id, backend.name()));
            }
            let r = run_query(qmd, *backend, query, collection);
            if !json_mode {
                drop(writeln!(
                    err,
                    " {}ms",
                    report::round_ms(r.latency_ms as f64)
                ));
            }
            backends_out.push((backend.name().to_owned(), r));
        }
        results.push(QueryResult {
            id: query.id.clone(),
            query: query.query.clone(),
            kind: query.kind.clone(),
            backends: backends_out,
        });
    }
    let summary = report::compute_summary(&results);
    Ok((results, summary))
}

/// `qmd bench` dispatch — upstream `qmd.ts:4788-4810`.
pub(crate) fn cmd_bench(
    args: &crate::args::BenchArgs,
    ctx: &crate::context::CliContext,
    env: &qmd::Environment,
    io: &mut crate::Io<'_>,
) -> Result<(), Failure> {
    let Some(fixture_path) = args.args.first() else {
        // 4 stderr lines, exit 1, NO doctor hint — upstream `Usage:` +
        // blank + description + pointer (qmd.ts:4790-4795); the pointer
        // names upstream's source path verbatim (upstream-quirks.md Q36).
        let _ = writeln!(
            io.err,
            "Usage: qmd bench <fixture.json> [--json] [-c collection]"
        );
        let _ = writeln!(io.err);
        let _ = writeln!(
            io.err,
            "Run search quality benchmarks against a fixture file."
        );
        let _ = writeln!(
            io.err,
            "See src/bench/fixtures/example.json for the fixture format."
        );
        return Err(Failure::Reported(1));
    };

    // Parse the fixture before opening the store (bench.ts:325-343).
    let fixture = load_fixture(fixture_path)?;
    // First `-c` value, unvalidated (upstream-quirks.md Q35).
    let collection = args
        .collection
        .first()
        .map(String::as_str)
        .or(fixture.collection.as_deref());
    let store = ctx.open_store_no_models_pin(env)?;
    let (results, summary) = run_benchmark(
        &store,
        &fixture,
        collection,
        &Backend::ALL,
        args.json,
        io.err,
    )?;
    drop(store);

    report::emit(
        &results,
        &summary,
        fixture_path,
        collection,
        &bench_timestamp(time::OffsetDateTime::now_utc()),
        args.json,
        (io.out, io.err),
    );
    Ok(())
}

/// `new Date().toISOString().replace(/[:.]/g,"").slice(0,15)` —
/// `YYYY-MM-DDTHHMM` in UTC — truncated to the minute
/// (upstream-quirks.md Q32). `now` is injected so tests assert a fixed
/// instant.
fn bench_timestamp(now: time::OffsetDateTime) -> String {
    use time::format_description::well_known::Iso8601;
    let iso = now.format(&Iso8601::DEFAULT).unwrap_or_default();
    iso.chars()
        .filter(|c| *c != ':' && *c != '.')
        .take(15)
        .collect()
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::missing_assert_message,
    reason = "test-only module in a feature-gated crate + idiomatic test patterns"
)]
mod tests {
    //! `parseStructuredQuery` branches (bench.ts:46-94) and
    //! `assertBenchCollectionReady` (bench.ts:283-310, porting
    //! `test/bench-guard.test.ts`); `load_fixture` error texts.
    use super::*;

    fn parse_err(query: &str) -> String {
        match parse_structured_query(query) {
            Err(e) => e,
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn parser_plain_single_line() {
        let ok = |q: &str| {
            parse_structured_query(q)
                .unwrap_or_else(|_| panic!("err"))
                .is_none()
        };
        assert!(ok("hello world"));
        assert!(ok(""));
        // A single `expand:` line is not a bench prefix → plain.
        assert!(ok("expand: foo"));
    }

    #[test]
    fn parser_lex_vec_hyde_intent() {
        let sq = parse_structured_query("lex: alpha\nVec: beta\nhyde: gamma\nintent: find X")
            .unwrap_or_else(|_| panic!("err"))
            .expect("structured");
        let s = &sq.searches;
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].kind, QueryKind::Lex);
        assert_eq!(s[1].kind, QueryKind::Vec);
        assert_eq!(s[2].kind, QueryKind::Hyde);
        assert_eq!(s[0].line, Some(1));
        assert_eq!(sq.intent.as_deref(), Some("find X"));
    }

    #[test]
    fn parser_errors() {
        assert_eq!(
            parse_err("intent: a\nintent: b\nlex: x"),
            "Line 2: only one intent: line is allowed per benchmark query."
        );
        assert_eq!(
            parse_err("intent:\nlex: x"),
            "Line 1: intent: must include text."
        );
        assert_eq!(
            parse_err("lex:\nlex: x"),
            "Line 1 (lex:) must include text."
        );
        assert_eq!(
            parse_err("lex: x\nbare"),
            "Line 2 is missing a lex:/vec:/hyde:/intent: prefix."
        );
        assert_eq!(
            parse_err("intent: only"),
            "intent: cannot appear alone. Add at least one lex:, vec:, or hyde: line."
        );
    }

    #[test]
    fn load_fixture_errors() {
        use std::io::Write as _;
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, body: &str| -> String {
            let p = dir.path().join(name);
            let mut f = std::fs::File::create(&p).unwrap();
            f.write_all(body.as_bytes()).unwrap();
            p.to_string_lossy().into_owned()
        };
        match load_fixture(&write("bad.json", "not json")) {
            Err(Failure::Message(reason)) => {
                assert!(reason.starts_with("Invalid fixture JSON:"), "{reason}");
            }
            Ok(_) => panic!("parsed ok"),
            Err(Failure::Reported(c)) => panic!("reported {c}"),
            Err(Failure::Error(_)) => panic!("expected Message"),
        }
        match load_fixture(&write("nq.json", r#"{"version":1}"#)) {
            Err(Failure::Message(reason)) => {
                assert_eq!(reason, "Invalid fixture: missing 'queries' array");
            }
            Ok(_) => panic!("parsed ok"),
            Err(Failure::Reported(c)) => panic!("reported {c}"),
            Err(Failure::Error(_)) => panic!("expected Message"),
        }
        assert!(
            load_fixture(&write("missing.json", "")).is_err(),
            "empty file parsed"
        );
        match load_fixture("/nonexistent-xyz.json") {
            Err(Failure::Message(reason)) => {
                assert!(reason.starts_with("Failed to read fixture /nonexistent-xyz.json:"));
            }
            Ok(_) => panic!("parsed ok"),
            Err(Failure::Reported(c)) => panic!("reported {c}"),
            Err(Failure::Error(_)) => panic!("expected Message"),
        }
    }

    #[test]
    fn fixture_collection_js_loose_typing() {
        use std::io::Write as _;
        let dir = tempfile::tempdir().unwrap();
        let parse = |body: &str| -> Option<String> {
            let p = dir.path().join("f.json");
            std::fs::File::create(&p)
                .unwrap()
                .write_all(body.as_bytes())
                .unwrap();
            load_fixture(&p.to_string_lossy())
                .unwrap_or_else(|_| panic!("fixture"))
                .collection
        };
        let q = r#"[{"id":"q","query":"x","expected_files":[],"expected_in_top_k":1}]"#;
        // JS `String()` coercion in the guard's error template.
        assert_eq!(
            parse(&format!(r#"{{"collection":123,"queries":{q}}}"#)),
            Some("123".to_owned())
        );
        assert_eq!(
            parse(&format!(r#"{{"collection":true,"queries":{q}}}"#)),
            Some("true".to_owned())
        );
        // Absent/null stay absent.
        assert_eq!(parse(&format!(r#"{{"queries":{q}}}"#)), None);
        assert_eq!(
            parse(&format!(r#"{{"collection":null,"queries":{q}}}"#)),
            None
        );
    }

    #[test]
    fn empty_collection_name_is_falsy() {
        // JS `if (collection)` truthiness — "" is the absent path.
        let cols = vec![("eval-docs".to_owned(), 3usize)];
        assert!(report::assert_collection_ready(&cols, Some("")).is_ok());
        assert!(report::assert_collection_ready(&[], Some("")).is_err());
        assert_eq!(
            report::all_zero_warning(Some("")),
            report::all_zero_warning(None)
        );
    }

    #[test]
    fn guard_messages() {
        use report::assert_collection_ready as g;
        fn msg(r: Result<(), Failure>) -> String {
            match r {
                Err(Failure::Message(reason)) => reason,
                Ok(()) => panic!("expected error"),
                Err(Failure::Reported(c)) => panic!("reported {c}"),
                Err(Failure::Error(_)) => panic!("expected Message"),
            }
        }
        // bench-guard.test.ts — missing named collection.
        let cols = vec![("other".to_owned(), 3usize)];
        assert_eq!(
            msg(g(&cols, Some("eval-docs"))),
            "Collection not found: eval-docs\nAvailable: other. Run 'qmd ls' to inspect."
        );
        assert_eq!(
            msg(g(&[], Some("eval-docs"))),
            "Collection not found: eval-docs\nRun 'qmd ls' to see available collections."
        );
        // Empty named collection.
        let empty = vec![("eval-docs".to_owned(), 0usize)];
        assert_eq!(
            msg(g(&empty, Some("eval-docs"))),
            "Collection 'eval-docs' has no indexed documents.\nRun 'qmd update', then 'qmd ls eval-docs' to confirm files are indexed before bench."
        );
        // Nothing indexed, no name.
        assert_eq!(
            msg(g(&[], None)),
            "No indexed documents found.\nIndex a collection with 'qmd collection add' / 'qmd update' before running bench."
        );
        let zero = vec![("c".to_owned(), 0usize)];
        assert_eq!(
            msg(g(&zero, None)),
            "No indexed documents found.\nIndex a collection with 'qmd collection add' / 'qmd update' before running bench."
        );
        // Ok paths.
        assert!(g(&cols, None).is_ok());
        assert!(g(&cols, Some("other")).is_ok());
    }

    #[test]
    fn bench_timestamp_fixed_instant() {
        // `new Date().toISOString()` strips `:` and `.` — non-zero
        // millis prove the `.` is removed, not just the `:`.
        let now = time::OffsetDateTime::parse(
            "2026-09-24T08:25:43.123Z",
            &time::format_description::well_known::Iso8601::DEFAULT,
        )
        .unwrap();
        assert_eq!(bench_timestamp(now), "2026-09-24T0825");
    }

    #[test]
    fn summary_all_zero_branches() {
        use report::summary_all_zero as z;
        let s = |p: f64, r: f64, m: f64| Summary {
            name: "x".into(),
            avg_precision: p,
            avg_recall: r,
            avg_recall_at_1: 0.0,
            avg_recall_at_3: 0.0,
            avg_recall_at_5: 0.0,
            avg_mrr: m,
            avg_f1: 0.0,
            avg_latency_ms: 0.0,
        };
        assert!(!z(&[]));
        assert!(z(&[s(0.0, 0.0, 0.0)]));
        assert!(!z(&[s(0.5, 0.0, 0.0)]));
        assert!(z(&[s(0.0, 0.0, 0.0), s(0.0, 0.0, 0.0)]));
        // Warning text (bench.ts:320-323).
        assert_eq!(
            report::all_zero_warning(Some("eval-docs")),
            "All benchmark scores were 0.00 — the collection is likely unindexed or the fixture's expected files are missing. Check with 'qmd ls eval-docs'."
        );
        assert_eq!(
            report::all_zero_warning(None),
            "All benchmark scores were 0.00 — the collection is likely unindexed or the fixture's expected files are missing. Check with 'qmd ls'."
        );
    }
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::missing_assert_message,
    clippy::cast_possible_truncation,
    clippy::cognitive_complexity,
    reason = "test-only module in a feature-gated crate + idiomatic test patterns"
)]
mod store_tests {
    //! Store-backed `run_benchmark` tests — the `bench-guard.test.ts`
    //! `runBenchmark collection guard` cases (:105-156), the bm25
    //! golden replay of `bench_golden.json`'s recorded runs, and a
    //! fake-backend pass over all four backends.
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use qmd::{CollectionSpec, Qmd, UpdateOptions};
    use tempfile::TempDir;

    use super::*;

    fn corpus() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../qmd/tests/fixtures/eval-docs")
    }

    /// Temp index with `corpus_dir` registered as `name` and updated.
    fn bench_store(name: &str, corpus_dir: &Path) -> (TempDir, Qmd) {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.sqlite");
        let qmd = Qmd::builder(&db).build().unwrap();
        qmd.add_collection(
            name,
            &CollectionSpec {
                path: corpus_dir.to_string_lossy().into_owned(),
                pattern: Some("**/*.md".to_owned()),
                ignore: vec![],
            },
        )
        .unwrap();
        if !corpus_dir.as_os_str().is_empty() {
            qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();
        }
        (dir, qmd)
    }

    fn fixture(queries: &str) -> (TempDir, Fixture, String) {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f.json");
        std::fs::write(&p, queries).unwrap();
        let path = p.to_string_lossy().into_owned();
        (
            dir,
            load_fixture(&path).unwrap_or_else(|_| panic!("fixture")),
            path,
        )
    }

    fn err_of(r: Result<(Vec<QueryResult>, Vec<Summary>), Failure>) -> String {
        match r {
            Err(Failure::Message(reason)) => reason,
            Ok(_) => panic!("expected error"),
            Err(Failure::Reported(c)) => panic!("reported {c}"),
            Err(Failure::Error(_)) => panic!("expected Message"),
        }
    }

    #[test]
    fn guard_missing_collection() {
        // bench-guard.test.ts:123 — fixture collection does not exist.
        let (_d, store) = bench_store("other", &corpus());
        let (_fd, fix, _path) = fixture(
            r#"{"collection":"eval-docs","queries":[{"id":"q","query":"x","expected_files":[],"expected_in_top_k":1}]}"#,
        );
        let mut err = Vec::new();
        let reason = err_of(run_benchmark(
            &store,
            &fix,
            fix.collection.as_deref(),
            &[Backend::Bm25],
            true,
            &mut err,
        ));
        assert!(
            reason.starts_with("Collection not found: eval-docs"),
            "{reason}"
        );
    }

    #[test]
    fn guard_empty_collection() {
        // bench-guard.test.ts:134 — collection exists, no documents.
        let (_d, store) = bench_store("eval-docs", Path::new(""));
        let (_fd, fix, _path) = fixture(
            r#"{"collection":"eval-docs","queries":[{"id":"q","query":"x","expected_files":[],"expected_in_top_k":1}]}"#,
        );
        let mut err = Vec::new();
        let reason = err_of(run_benchmark(
            &store,
            &fix,
            fix.collection.as_deref(),
            &[Backend::Bm25],
            true,
            &mut err,
        ));
        assert!(
            reason.starts_with("Collection 'eval-docs' has no indexed documents"),
            "{reason}"
        );
    }

    #[test]
    fn bm25_recall_positive_no_warning() {
        // bench-guard.test.ts:147 — bm25 against an indexed collection.
        let (_d, store) = bench_store("eval-docs", &corpus());
        let (_fd, fix, _path) = fixture(
            r#"{"collection":"eval-docs","queries":[{"id":"q","query":"API versioning","type":"exact","expected_files":["api-design-principles.md"],"expected_in_top_k":1}]}"#,
        );
        let mut err = Vec::new();
        let (results, summary) = run_benchmark(
            &store,
            &fix,
            fix.collection.as_deref(),
            &[Backend::Bm25],
            true,
            &mut err,
        )
        .unwrap_or_else(|_| panic!("bench"));
        let bm25 = &results[0].backends[0].1;
        assert!(bm25.m.recall > 0.0);
        assert!(!report::summary_all_zero(&summary));
        let stderr = String::from_utf8(err).unwrap();
        assert!(
            !stderr.contains("All benchmark scores were 0.00"),
            "{stderr}"
        );
    }

    #[test]
    fn all_zero_warns_with_ls_hint() {
        // All-zero summary → warning containing `qmd ls <collection>`.
        let (_d, store) = bench_store("eval-docs", &corpus());
        let (_fd, fix, _path) = fixture(
            r#"{"collection":"eval-docs","queries":[{"id":"q","query":"zzz-no-match-qqq","expected_files":["nope.md"],"expected_in_top_k":1}]}"#,
        );
        let mut err = Vec::new();
        let (_r, summary) = run_benchmark(
            &store,
            &fix,
            fix.collection.as_deref(),
            &[Backend::Bm25],
            true,
            &mut err,
        )
        .unwrap_or_else(|_| panic!("bench"));
        assert!(report::summary_all_zero(&summary));
        // cmd_bench prints the warning; assert the text builder.
        assert!(report::all_zero_warning(fix.collection.as_deref()).contains("qmd ls eval-docs"));
    }

    /// Byte-exact replay of the recorded bm25 runs — index the
    /// vendored eval-docs, run `run_benchmark` (progress → `err`),
    /// then `report::emit`, fold both streams with the recorder's
    /// regexes, and `assert_eq!` the full strings. This exercises
    /// JSON key order, the text table + summary, stderr progress,
    /// the all-zero warning, and the `0.125` toFixed(2) tie in
    /// `tie-125`.
    #[test]
    fn golden_replay_streams() {
        let (_d, store) = bench_store("eval-docs", &corpus());
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/bench_golden.json")).unwrap();
        for run in f["runs"].as_array().unwrap() {
            let name = run["name"].as_str().unwrap();
            let json_mode = run["json"].as_bool().unwrap();
            let fixture_path = run["fixture"].as_str().unwrap();
            // Recorded paths are repo-root-relative.
            let ws = Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap();
            let abs = ws.join(fixture_path).to_string_lossy().into_owned();
            let fix = load_fixture(&abs).unwrap_or_else(|_| panic!("fixture"));
            let collection = fix.collection.as_deref().or(Some("eval-docs"));
            let mut err = Vec::new();
            let (results, summary) = run_benchmark(
                &store,
                &fix,
                collection,
                &[Backend::Bm25],
                json_mode,
                &mut err,
            )
            .unwrap_or_else(|_| panic!("bench"));
            let mut out = Vec::new();
            report::emit(
                &results,
                &summary,
                fixture_path,
                collection,
                "<TS>",
                json_mode,
                (&mut out, &mut err),
            );
            let stdout = fold_stream(&String::from_utf8(out).unwrap());
            let stderr = fold_stream(&String::from_utf8(err).unwrap());
            assert_eq!(stderr, run["stderr"].as_str().unwrap(), "{name} stderr");
            assert_eq!(stdout, run["stdout"].as_str().unwrap(), "{name} stdout");
        }
    }

    /// Rust port of the recorder's `fold()` — collapses every
    /// clock-dependent value to a fixed token on both sides. Same
    /// patterns, same order as `gen_bench_golden.ts` `fold`.
    fn fold_stream(s: &str) -> String {
        let re = |p: &str| regex::Regex::new(p).unwrap();
        let s = re(r#""timestamp":\s*"[^"]*""#).replace_all(s, "\"timestamp\": \"<TS>\"");
        let s = re(r"\d+ms\n").replace_all(&s, " NNNms\n");
        let s = re(r"Avg=\d+ms").replace_all(&s, "Avg=NNms");
        let s = re(r#""latency_ms":\s*\d+"#).replace_all(&s, "\"latency_ms\": 0");
        let s = re(r#""avg_latency_ms":\s*[\d.]+"#).replace_all(&s, "\"avg_latency_ms\": 0");
        // JS `/\s\d+ms$/gm` — multiline, JS `\s`; the stream is ASCII
        // around these tokens, so Rust `\s` matches the same bytes.
        re(r"(?m)\s\d+ms$").replace_all(&s, " NNms").into_owned()
    }

    /// Fake backends over all four backends — every result carries the
    /// 4 backend keys in upstream order and exact queries score >0 in
    /// hybrid/full.
    #[test]
    fn fake_backends_all_four() {
        use qmd::testing::{FakeEmbedder, FakeGenerator, FakeReranker};
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.sqlite");
        let store = Qmd::builder(&db)
            .embedder(Arc::new(FakeEmbedder::new("fake-embed")))
            .reranker(Arc::new(FakeReranker::new("fake-rerank")))
            .generator(Arc::new(FakeGenerator::new("fake-gen")))
            .build()
            .unwrap();
        store
            .add_collection(
                "eval-docs",
                &CollectionSpec {
                    path: corpus().to_string_lossy().into_owned(),
                    pattern: Some("**/*.md".to_owned()),
                    ignore: vec![],
                },
            )
            .unwrap();
        store
            .update(&UpdateOptions::default(), &mut |_| {})
            .unwrap();
        store
            .embed(&qmd::EmbedOptions::default(), &mut |_| {})
            .unwrap();

        let fix =
            load_fixture("tests/fixtures/bench/example.json").unwrap_or_else(|_| panic!("fixture"));
        let mut err = Vec::new();
        let (results, _summary) = run_benchmark(
            &store,
            &fix,
            fix.collection.as_deref().or(Some("eval-docs")),
            &Backend::ALL,
            true,
            &mut err,
        )
        .unwrap_or_else(|_| panic!("bench"));
        for r in &results {
            let names: Vec<&str> = r.backends.iter().map(|(n, _)| n.as_str()).collect();
            assert_eq!(names, ["bm25", "vector", "hybrid", "full"], "{}", r.id);
        }
        // An exact query scores >0 in hybrid and full.
        let q = &results[0];
        for (n, b) in &q.backends {
            if n == "hybrid" || n == "full" {
                assert!(b.m.recall > 0.0, "{n} on {}", q.id);
            }
        }
    }
}
