#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::shadow_unrelated,
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_assert_message,
    clippy::panic,
    reason = "idiomatic test-code patterns"
)]

//! `extract_snippet` chunk-window and intent-weight coverage — ports of
//! the `Snippet Extraction` `chunkPos` cases in `store.test.ts`
//! (2548-2683) and the `extractSnippet with intent` / `intent weight
//! behavior` describes in `intent.test.ts` (179-300).

use qmd::{ChunkSpan, extract_snippet};

#[allow(
    clippy::unnecessary_wraps,
    reason = "shorthand so call sites read `at(pos)` where an Option is expected"
)]
const fn at(pos: usize) -> Option<ChunkSpan> {
    Some(ChunkSpan::new(pos, 0))
}

// ── window-free cases (store.test.ts Snippet Extraction) ──

#[test]
fn finds_query_terms() {
    let body = "First line.\nSecond line with keyword.\nThird line.\nFourth line.";
    let r = extract_snippet(body, "keyword", 500, None, None);
    assert_eq!(r.line, 2);
    assert!(r.snippet.contains("keyword"));
}

#[test]
fn includes_context_lines() {
    let body = "Line 1\nLine 2\nLine 3 has keyword\nLine 4\nLine 5";
    let r = extract_snippet(body, "keyword", 500, None, None);
    assert!(r.snippet.contains("Line 2")); // context before
    assert!(r.snippet.contains("Line 3 has keyword"));
    assert!(r.snippet.contains("Line 4")); // context after
}

#[test]
fn respects_max_len_for_content() {
    let body = "A".repeat(1000);
    let r = extract_snippet(&body, "query", 100, None, None);
    assert!(r.snippet.contains("@@")); // has diff header
    assert!(r.snippet.contains("...")); // content truncated
}

#[test]
fn returns_beginning_when_no_match() {
    let body = "First line\nSecond line\nThird line";
    let r = extract_snippet(body, "nonexistent", 500, None, None);
    assert_eq!(r.line, 1);
    assert!(r.snippet.contains("First line"));
}

#[test]
fn includes_diff_style_header() {
    let body = "Line 1\nLine 2\nLine 3 has keyword\nLine 4\nLine 5";
    let r = extract_snippet(body, "keyword", 500, None, None);
    assert!(
        r.snippet.starts_with("@@ -2,4 @@ (1 before, 0 after)"),
        "{}",
        r.snippet
    );
    assert_eq!(r.lines_before, 1);
    assert_eq!(r.lines_after, 0);
    assert_eq!(r.snippet_lines, 4);
}

#[test]
fn mid_document_lines_before_and_after() {
    let body = "L1\nL2\nL3\nL4 match\nL5\nL6\nL7\nL8\nL9\nL10";
    let r = extract_snippet(body, "match", 500, None, None);
    assert_eq!(r.line, 4);
    assert_eq!(r.lines_before, 2); // L1, L2 before the L3-start window
    assert_eq!(r.snippet_lines, 4); // L3, L4, L5, L6
    assert_eq!(r.lines_after, 4); // L7-L10
}

#[test]
fn header_format_matches_diff_style() {
    let body = "A\nB\nC keyword\nD\nE\nF\nG\nH";
    let r = extract_snippet(body, "keyword", 500, None, None);
    assert!(
        r.snippet.starts_with("@@ -2,4 @@ (1 before, 3 after)"),
        "{}",
        r.snippet
    );
}

#[test]
fn document_start_has_zero_lines_before() {
    let body = "First line keyword\nSecond\nThird\nFourth\nFifth";
    let r = extract_snippet(body, "keyword", 500, None, None);
    assert_eq!(r.line, 1);
    assert_eq!(r.lines_before, 0);
    assert_eq!(r.snippet_lines, 3); // best-1..best+3 clamped
    assert_eq!(r.lines_after, 2); // Fourth, Fifth
}

#[test]
fn leading_blank_lines_report_one_before() {
    // Upstream regression lock: "1 before" counts absolute file lines,
    // not visible snippet lines — the snippet starts at line 2.
    let rest = (6..=77)
        .map(|i| format!("body line {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    let body = format!(
        "---\ntitle: Notes\n# Heading with keyword\nIntro paragraph.\nMore intro lines.\n{rest}"
    );
    let r = extract_snippet(&body, "keyword", 500, None, None);
    assert_eq!(r.line, 3);
    assert_eq!(r.lines_before, 1);
    assert_eq!(r.snippet_lines, 4);
    assert_eq!(r.lines_after, 72);
    assert!(r.snippet.contains("@@ -2,4 @@ (1 before, 72 after)"));
}

#[test]
fn document_end_has_zero_lines_after() {
    let body = "First\nSecond\nThird\nFourth\nFifth keyword";
    let r = extract_snippet(body, "keyword", 500, None, None);
    assert_eq!(r.line, 5);
    assert_eq!(r.lines_before, 3);
    assert_eq!(r.snippet_lines, 2); // Fourth, Fifth keyword
    assert_eq!(r.lines_after, 0);
}

#[test]
fn single_line_document() {
    let body = "Single line with keyword";
    let r = extract_snippet(body, "keyword", 500, None, None);
    assert_eq!(r.lines_before, 0);
    assert_eq!(r.lines_after, 0);
    assert_eq!(r.snippet_lines, 1);
    assert!(r.snippet.contains("@@ -1,1 @@ (0 before, 0 after)"));
    assert!(r.snippet.contains("Single line with keyword"));
}

// ── chunkPos (store.test.ts Snippet Extraction) ──

#[test]
fn chunk_pos_hint_focuses_window() {
    let body = format!(
        "{}{}{}",
        "First section...\n".repeat(50),
        "Target keyword here\n",
        "More content...".repeat(50)
    );
    let pos = body.find("Target keyword").unwrap();
    let r = extract_snippet(&body, "Target", 200, at(pos), None);
    assert!(r.snippet.contains("Target keyword"));
}

#[test]
fn chunk_pos_adjusts_line_numbers() {
    let padding = "Padding line\n".repeat(50);
    let body = format!("{padding}Target keyword here\nMore content\nEven more");
    let pos = padding.len();
    let r = extract_snippet(&body, "keyword", 200, at(pos), None);
    assert_eq!(r.line, 51);
    assert!(r.lines_before > 40);
}

#[test]
fn chunk_pos_anchors_when_no_lexical_match() {
    // FTS5-syntax queries tokenise into terms that never appear in the
    // body; bestScore stays 0 and the fallback anchors on chunkPos
    // rather than line 1 (store.test.ts:2663-2677).
    let pad_line = "Lorem ipsum dolor sit amet\n";
    let padding = pad_line.repeat(100);
    let body = format!("{padding}chunk content here\nmore chunk content\n{padding}");
    let pos = padding.len();
    let r = extract_snippet(&body, "\"unrelated quoted phrase\"", 200, at(pos), None);
    assert!(r.line > 50 && r.line < 110, "line={}", r.line);
}

#[test]
fn chunk_pos_zero_falls_back_to_full_body() {
    // pos=0 may be the chunk selector's bestIdx=0 default rather than a
    // real first-chunk hit — retry on the whole body (2679-2683).
    let padding = "Lorem ipsum dolor sit amet\n".repeat(200);
    let body = format!("{padding}TARGET_KEYWORD line content\ntail line\n");
    let r = extract_snippet(&body, "TARGET_KEYWORD", 200, at(0), None);
    assert_eq!(r.line, 201);
}

// ── JS whitespace semantics (A5) ──

#[test]
fn query_terms_split_on_js_whitespace() {
    // U+FEFF is JS `\s` — "alpha\u{FEFF}bravo" splits into two terms
    // and both hit line 2. U+0085 (NEL) is NOT JS `\s` — the query
    // stays one term, matches nothing, and scores 0 → line 1.
    let body = "intro line\nalpha bravo end\nthird\nfourth\nfifth";
    let r = extract_snippet(body, "alpha\u{feff}bravo", 500, None, None);
    assert_eq!(r.line, 2);
    let r = extract_snippet(body, "alpha\u{0085}bravo", 500, None, None);
    assert_eq!(r.line, 1);
}

#[test]
fn whitespace_only_window_fallback_uses_js_trim() {
    // A focused window whose snippet is whitespace-only retries on the
    // full body (store.ts:5397 `.trim()`). U+FEFF is JS whitespace →
    // the FEFF window trims to empty and the retry finds the match on
    // line 1. U+0085 (NEL) is NOT JS whitespace → the NEL window
    // survives and the match outside the window stays out.
    let mk = |sep: char| format!("alpha target\n{}\ntail line\n", sep.to_string().repeat(300));
    // Separator run at utf16 offset 13; the pos±100/len=50 window
    // [63,313) covers only separators.
    let window = Some(ChunkSpan::new(163, 50));

    let r = extract_snippet(&mk('\u{feff}'), "alpha", 3000, window, None);
    assert_eq!(r.line, 1, "{:?}", r.snippet);
    assert!(r.snippet.contains("alpha target"));

    let r = extract_snippet(&mk('\u{0085}'), "alpha", 3000, window, None);
    assert_eq!(r.line, 2, "{:?}", r.snippet);
    assert!(!r.snippet.contains("alpha target"));
}

// ── extractSnippet with intent (intent.test.ts:179-271) ──

/// Each section contains "performance" so query scores tie at 1.0;
/// intent terms (weight 0.3) then break the tie.
fn sectioned_body() -> String {
    [
        "# Notes on Various Topics",
        "",
        "## Web Performance Section",
        "Web performance means optimizing page load times and Core Web Vitals.",
        "Reduce latency, improve rendering speed, and measure performance budgets.",
        "",
        "## Team Performance Section",
        "Team performance depends on trust, psychological safety, and feedback.",
        "Build culture where performance reviews drive growth not fear.",
        "",
        "## Health Performance Section",
        "Health performance comes from consistent exercise, sleep, and endurance.",
        "Track fitness metrics, optimize recovery, and monitor healthspan.",
    ]
    .join("\n")
}

#[test]
fn no_intent_anchors_on_query_terms() {
    let r = extract_snippet(&sectioned_body(), "performance", 500, None, None);
    assert!(r.snippet.contains("Performance"));
}

#[test]
fn web_perf_intent_prefers_web_section() {
    let r = extract_snippet(
        &sectioned_body(),
        "performance",
        500,
        None,
        Some("Looking for notes about web performance, latency, and page load times"),
    );
    let lower = r.snippet.to_lowercase();
    assert!(
        lower.contains("latency")
            || lower.contains("page load")
            || lower.contains("core web vitals"),
        "snippet={lower}"
    );
}

#[test]
fn health_intent_prefers_health_section() {
    let r = extract_snippet(
        &sectioned_body(),
        "performance",
        500,
        None,
        Some("Looking for notes about personal health, fitness, and endurance"),
    );
    let lower = r.snippet.to_lowercase();
    assert!(
        lower.contains("health") || lower.contains("fitness") || lower.contains("endurance"),
        "snippet={lower}"
    );
}

#[test]
fn team_intent_prefers_team_section() {
    let r = extract_snippet(
        &sectioned_body(),
        "performance",
        500,
        None,
        Some("Looking for notes about building high-performing teams and culture"),
    );
    let lower = r.snippet.to_lowercase();
    assert!(
        lower.contains("team") || lower.contains("culture") || lower.contains("trust"),
        "snippet={lower}"
    );
}

#[test]
fn intent_does_not_override_strong_query_match() {
    let r = extract_snippet(
        &sectioned_body(),
        "Core Web Vitals",
        500,
        None,
        Some("Looking for notes about health and fitness"),
    );
    assert!(r.snippet.contains("Core Web Vitals"));
}

#[test]
fn absent_intent_same_as_none() {
    let body = sectioned_body();
    let a = extract_snippet(&body, "performance", 500, None, None);
    let b = extract_snippet(&body, "performance", 500, None, None);
    assert_eq!(a.line, b.line);
    assert_eq!(a.snippet, b.snippet);
}

#[test]
fn intent_with_no_matching_terms_uses_query_only() {
    let r = extract_snippet(
        &sectioned_body(),
        "performance",
        500,
        None,
        Some("quantum computing and entanglement"),
    );
    assert!(r.snippet.contains("Performance"));
    assert!(!r.snippet.is_empty());
}

#[test]
fn intent_works_with_chunk_position() {
    let body = sectioned_body();
    let pos = body.find("## Web Performance").unwrap();
    let r = extract_snippet(
        &body,
        "performance",
        500,
        Some(ChunkSpan::new(pos, 200)),
        Some("web page load times"),
    );
    let lower = r.snippet.to_lowercase();
    assert!(
        lower.contains("web performance")
            || lower.contains("core web vitals")
            || lower.contains("page load"),
        "snippet={lower}"
    );
}

// ── intent weight behavior (intent.test.ts:273-300) ──

#[test]
fn intent_breaks_tie_between_equal_query_matches() {
    // Query term appears on every line; intent decides.
    let body = "performance metrics for team velocity\nperformance metrics for web latency\nperformance metrics for athletic endurance";
    let no_intent = extract_snippet(body, "performance metrics", 500, None, None);
    assert_eq!(no_intent.line, 1); // first line wins ties

    let with_intent = extract_snippet(
        body,
        "performance metrics",
        500,
        None,
        Some("web latency and page speed"),
    );
    assert!(with_intent.snippet.contains("web latency"));
}
