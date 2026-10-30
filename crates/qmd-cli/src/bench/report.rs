//! Output shaping — upstream `bench.ts` output functions:
//! `assertBenchCollectionReady` (:283-310), `benchSummaryAllZero` /
//! `allZeroBenchWarning` (:312-323), `computeSummary` (:223-273),
//! `formatTable` (:198-221) and the summary block (:401-417), plus
//! the JSON builders that give each `serde_json` map the exact
//! upstream key order (result / query / top-level / summary).

use std::fmt::Write as _;

use qmd::js;
use serde_json::{Map, Value, json};

use super::score::{ScoreMetrics, score_results};
use crate::Failure;
use crate::output::utf16_prefix;

/// One backend's result row — upstream `BackendResult`; `errored`
/// selects upstream's error-branch key order (bench.ts:179-196).
#[derive(Debug, Clone)]
pub(crate) struct BackendResult {
    pub(crate) m: ScoreMetrics,
    pub(crate) total_expected: usize,
    pub(crate) latency_ms: u64,
    pub(crate) top_files: Vec<String>,
    pub(crate) errored: bool,
}

impl BackendResult {
    /// `scoreResults` + the three extra fields in upstream order.
    pub(crate) fn new(
        result_files: &[String],
        expected_files: &[String],
        top_k: usize,
        latency_ms: u64,
    ) -> Self {
        Self {
            m: score_results(result_files, expected_files, top_k),
            total_expected: expected_files.len(),
            latency_ms,
            top_files: result_files.iter().take(10).cloned().collect(),
            errored: false,
        }
    }

    /// The catch-branch zero result (bench.ts:179-196) — upstream
    /// puts `total_expected`, `latency_ms`, `top_files` before
    /// `matched_files` in this branch's object literal.
    pub(crate) fn error(expected_files: &[String], latency_ms: u64) -> Self {
        Self {
            m: ScoreMetrics {
                precision_at_k: 0.0,
                recall: 0.0,
                recall_at_1: 0.0,
                recall_at_3: 0.0,
                recall_at_5: 0.0,
                mrr: 0.0,
                f1: 0.0,
                hits_at_k: 0,
                matched_files: Vec::new(),
                unmatched_expected_files: expected_files.to_vec(),
            },
            total_expected: expected_files.len(),
            latency_ms,
            top_files: Vec::new(),
            errored: true,
        }
    }

    /// `BackendResult` → JSON with the exact upstream key order —
    /// normal: `precision_at_k … hits_at_k, matched_files,
    /// unmatched_expected_files, total_expected, latency_ms,
    /// top_files`; error: `… total_expected, latency_ms, top_files,
    /// matched_files, unmatched_expected_files` (the spread order of
    /// `{...scores, …}` upstream; upstream-quirks.md Q33).
    pub(crate) fn to_json(&self) -> Value {
        let m = &self.m;
        let mut o = Map::new();
        o.insert("precision_at_k".into(), json!(m.precision_at_k));
        o.insert("recall".into(), json!(m.recall));
        o.insert("recall_at_1".into(), json!(m.recall_at_1));
        o.insert("recall_at_3".into(), json!(m.recall_at_3));
        o.insert("recall_at_5".into(), json!(m.recall_at_5));
        o.insert("mrr".into(), json!(m.mrr));
        o.insert("f1".into(), json!(m.f1));
        o.insert("hits_at_k".into(), json!(m.hits_at_k));
        if self.errored {
            o.insert("total_expected".into(), json!(self.total_expected));
            o.insert("latency_ms".into(), json!(self.latency_ms));
            o.insert("top_files".into(), json!(self.top_files));
        }
        o.insert("matched_files".into(), json!(m.matched_files));
        o.insert(
            "unmatched_expected_files".into(),
            json!(m.unmatched_expected_files),
        );
        if !self.errored {
            o.insert("total_expected".into(), json!(self.total_expected));
            o.insert("latency_ms".into(), json!(self.latency_ms));
            o.insert("top_files".into(), json!(self.top_files));
        }
        Value::Object(o)
    }
}

/// One fixture query's results — upstream `QueryResult`.
#[derive(Debug)]
pub(crate) struct QueryResult {
    pub(crate) id: String,
    pub(crate) query: String,
    pub(crate) kind: Option<String>,
    pub(crate) backends: Vec<(String, BackendResult)>,
}

impl QueryResult {
    /// `id, query, type?, backends` — `type` omitted when undefined
    /// (JS `undefined` is dropped by JSON.stringify).
    pub(crate) fn to_json(&self) -> Value {
        let mut o = Map::new();
        o.insert("id".into(), json!(self.id));
        o.insert("query".into(), json!(self.query));
        if let Some(t) = &self.kind {
            o.insert("type".into(), json!(t));
        }
        let mut backends = Map::new();
        for (name, r) in &self.backends {
            backends.insert(name.clone(), r.to_json());
        }
        o.insert("backends".into(), Value::Object(backends));
        Value::Object(o)
    }
}

/// `computeSummary` (bench.ts:223-273) — backend names in first-seen
/// order; per-backend sums accumulated in results order, then /count.
#[derive(Debug)]
pub(crate) struct Summary {
    pub(crate) name: String,
    pub(crate) avg_precision: f64,
    pub(crate) avg_recall: f64,
    pub(crate) avg_recall_at_1: f64,
    pub(crate) avg_recall_at_3: f64,
    pub(crate) avg_recall_at_5: f64,
    pub(crate) avg_mrr: f64,
    pub(crate) avg_f1: f64,
    pub(crate) avg_latency_ms: f64,
}

pub(crate) fn compute_summary(results: &[QueryResult]) -> Vec<Summary> {
    let mut order: Vec<String> = Vec::new();
    for r in results {
        for (name, _) in &r.backends {
            if !order.contains(name) {
                order.push(name.clone());
            }
        }
    }
    let mut out = Vec::new();
    for name in order {
        let mut p = 0.0f64;
        let mut r_ = 0.0f64;
        let mut r1 = 0.0f64;
        let mut r3 = 0.0f64;
        let mut r5 = 0.0f64;
        let mut mrr = 0.0f64;
        let mut f1 = 0.0f64;
        let mut lat = 0.0f64;
        let mut count = 0usize;
        for r in results {
            let Some((_, br)) = r.backends.iter().find(|(n, _)| *n == name) else {
                continue;
            };
            p += br.m.precision_at_k;
            r_ += br.m.recall;
            r1 += br.m.recall_at_1;
            r3 += br.m.recall_at_3;
            r5 += br.m.recall_at_5;
            mrr += br.m.mrr;
            f1 += br.m.f1;
            lat += br.latency_ms as f64;
            count += 1;
        }
        if count > 0 {
            let c = count as f64;
            out.push(Summary {
                name,
                avg_precision: p / c,
                avg_recall: r_ / c,
                avg_recall_at_1: r1 / c,
                avg_recall_at_3: r3 / c,
                avg_recall_at_5: r5 / c,
                avg_mrr: mrr / c,
                avg_f1: f1 / c,
                avg_latency_ms: lat / c,
            });
        }
    }
    out
}

/// `assertBenchCollectionReady` (bench.ts:283-310) — guard text
/// verbatim; `None` collection only checks "anything indexed".
/// `Some("")` is falsy upstream (`if (collection)`) → same path.
pub(crate) fn assert_collection_ready(
    collections: &[(String, usize)],
    collection: Option<&str>,
) -> Result<(), Failure> {
    if let Some(collection) = collection.filter(|c| !c.is_empty()) {
        let m = collections.iter().find(|(n, _)| n == collection);
        match m {
            None => {
                let names: Vec<&str> = collections.iter().map(|(n, _)| n.as_str()).collect();
                let hint = if names.is_empty() {
                    "Run 'qmd ls' to see available collections.".to_owned()
                } else {
                    format!("Available: {}. Run 'qmd ls' to inspect.", names.join(", "))
                };
                return Err(Failure::Message(format!(
                    "Collection not found: {collection}\n{hint}"
                )));
            }
            Some((_, 0)) => {
                return Err(Failure::Message(format!(
                    "Collection '{collection}' has no indexed documents.\nRun 'qmd update', then 'qmd ls {collection}' to confirm files are indexed before bench."
                )));
            }
            Some(_) => return Ok(()),
        }
    }
    let total: usize = collections.iter().map(|(_, n)| n).sum();
    if collections.is_empty() || total == 0 {
        return Err(Failure::Message(
            "No indexed documents found.\nIndex a collection with 'qmd collection add' / 'qmd update' before running bench.".to_owned(),
        ));
    }
    Ok(())
}

/// `benchSummaryAllZero` (bench.ts:312-318) — empty summary is false.
pub(crate) fn summary_all_zero(summary: &[Summary]) -> bool {
    !summary.is_empty()
        && summary
            .iter()
            .all(|s| s.avg_precision == 0.0 && s.avg_recall == 0.0 && s.avg_mrr == 0.0)
}

/// `allZeroBenchWarning` (bench.ts:320-323) — `collection ?` is a JS
/// truthiness check: `Some("")` takes the bare `qmd ls` hint.
pub(crate) fn all_zero_warning(collection: Option<&str>) -> String {
    let hint = collection
        .filter(|c| !c.is_empty())
        .map_or_else(|| "qmd ls".to_owned(), |c| format!("qmd ls {c}"));
    format!(
        "All benchmark scores were 0.00 — the collection is likely unindexed or the fixture's expected files are missing. Check with '{hint}'."
    )
}

/// `Math.round(latency)` → the integer digits upstream prints; the
/// `as i64` truncation is a no-op on an already-rounded non-negative
/// value (clippy-safe path through `round_ms`).
#[allow(
    clippy::cast_possible_truncation,
    reason = "math_round output is integral and a latency never exceeds i64"
)]
pub(crate) fn round_ms(v: f64) -> i64 {
    js::math_round(v) as i64
}

/// `formatTable` (bench.ts:198-221) — UTF-16 `pad`, `num` =
/// `toFixed(2).padStart(5)`, 88-dash rule, trailing blank line per query.
pub(crate) fn format_table(results: &[QueryResult]) -> String {
    let pad = |s: &str, n: usize| {
        let t = utf16_prefix(s, n);
        let len = qmd::utf16_len(&t);
        if len >= n {
            t
        } else {
            format!("{t}{}", " ".repeat(n - len))
        }
    };
    let num = |n: f64| format!("{:>5}", js::to_fixed(n, 2));
    let mut lines = vec![
        format!(
            "{} {} {} {} {} {} {} {} {}",
            pad("Query", 25),
            pad("Backend", 8),
            pad("P@k", 6),
            pad("R@1", 6),
            pad("R@3", 6),
            pad("R@5", 6),
            pad("MRR", 6),
            pad("F1", 6),
            pad("ms", 8)
        ),
        "-".repeat(88),
    ];
    for r in results {
        for (name, br) in &r.backends {
            lines.push(format!(
                "{} {} {} {} {} {} {} {} {:>7}ms",
                pad(&r.id, 25),
                pad(name, 8),
                num(br.m.precision_at_k),
                num(br.m.recall_at_1),
                num(br.m.recall_at_3),
                num(br.m.recall_at_5),
                num(br.m.mrr),
                num(br.m.f1),
                round_ms(br.latency_ms as f64),
            ));
        }
        lines.push(String::new());
    }
    lines.join("\n")
}

/// The `Summary:` + 70-dash + per-backend `Avg=` block (bench.ts:402-417).
/// `num` = `toFixed(3).padStart(6)`.
pub(crate) fn format_summary_text(summary: &[Summary]) -> String {
    let pad = |s: &str, n: usize| {
        let t = utf16_prefix(s, n);
        let len = qmd::utf16_len(&t);
        if len >= n {
            t
        } else {
            format!("{t}{}", " ".repeat(n - len))
        }
    };
    let num = |n: f64| format!("{:>6}", js::to_fixed(n, 3));
    let mut out = String::from("Summary:\n");
    out.push_str(&"-".repeat(70));
    for s in summary {
        let _ = write!(
            out,
            "\n  {} P@k={} R@1={} R@3={} R@5={} MRR={} F1={} Avg={}ms",
            pad(&s.name, 8),
            num(s.avg_precision),
            num(s.avg_recall_at_1),
            num(s.avg_recall_at_3),
            num(s.avg_recall_at_5),
            num(s.avg_mrr),
            num(s.avg_f1),
            round_ms(s.avg_latency_ms),
        );
    }
    out
}

/// The JSON summary object for one backend — upstream key order
/// `avg_precision … avg_latency_ms`.
fn summary_json(s: &Summary) -> Value {
    let mut o = Map::new();
    o.insert("avg_precision".into(), json!(s.avg_precision));
    o.insert("avg_recall".into(), json!(s.avg_recall));
    o.insert("avg_recall_at_1".into(), json!(s.avg_recall_at_1));
    o.insert("avg_recall_at_3".into(), json!(s.avg_recall_at_3));
    o.insert("avg_recall_at_5".into(), json!(s.avg_recall_at_5));
    o.insert("avg_mrr".into(), json!(s.avg_mrr));
    o.insert("avg_f1".into(), json!(s.avg_f1));
    o.insert("avg_latency_ms".into(), json!(s.avg_latency_ms));
    Value::Object(o)
}

/// The top-level envelope — `timestamp, fixture, results, summary`.
fn envelope_json(
    timestamp: &str,
    fixture_path: &str,
    results: &[QueryResult],
    summary: &[Summary],
) -> Value {
    let mut root = Map::new();
    root.insert("timestamp".into(), json!(timestamp));
    root.insert("fixture".into(), json!(fixture_path));
    root.insert(
        "results".into(),
        Value::Array(results.iter().map(QueryResult::to_json).collect()),
    );
    let mut sum = Map::new();
    for s in summary {
        sum.insert(s.name.clone(), summary_json(s));
    }
    root.insert("summary".into(), Value::Object(sum));
    Value::Object(root)
}

/// The output tail of `runBenchmark` (bench.ts:393-417) — the all-zero
/// warning to `err`, then the JSON envelope or the text table +
/// `Summary:` block to `out`. Pure: `timestamp` is passed in so tests
/// replay recorded output deterministically.
pub(crate) fn emit(
    results: &[QueryResult],
    summary: &[Summary],
    fixture_path: &str,
    collection: Option<&str>,
    timestamp: &str,
    json: bool,
    io: (&mut dyn std::io::Write, &mut dyn std::io::Write),
) {
    let (out, err) = io;
    if summary_all_zero(summary) {
        let _ = writeln!(err, "\n{}", all_zero_warning(collection));
    }
    if json {
        let _ = writeln!(
            out,
            "{}",
            crate::json::stringify_pretty(&envelope_json(
                timestamp,
                fixture_path,
                results,
                summary
            ))
        );
    } else {
        let _ = writeln!(out, "\n{}", format_table(results));
        let _ = writeln!(out, "{}", format_summary_text(summary));
    }
}
