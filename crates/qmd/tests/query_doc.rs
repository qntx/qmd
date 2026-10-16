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

//! Query-document parser and intent-term extraction — ports of
//! `structured-search.test.ts`'s `parseStructuredQuery` describes and
//! `intent.test.ts`'s `extractIntentTerms` / `parseStructuredQuery with
//! intent` describes. The upstream reference is the CLI copy
//! (`cli/qmd.ts:2755-2823`), which the library ports verbatim (P2-D2).

use qmd::{
    ExpandedQuery, ParsedQueryDocument, QueryKind, extract_intent_terms, parse_query_document,
};

fn parse(input: &str) -> Option<ParsedQueryDocument> {
    parse_query_document(input).unwrap()
}

fn err(input: &str) -> String {
    parse_query_document(input).unwrap_err().to_string()
}

fn lex(query: &str, line: usize) -> ExpandedQuery {
    ExpandedQuery::new(QueryKind::Lex, query).with_line(line)
}

fn vec_q(query: &str, line: usize) -> ExpandedQuery {
    ExpandedQuery::new(QueryKind::Vec, query).with_line(line)
}

fn hyde(query: &str, line: usize) -> ExpandedQuery {
    ExpandedQuery::new(QueryKind::Hyde, query).with_line(line)
}

// ── plain queries (returns None for normal expansion) ──

#[test]
fn plain_single_line_without_prefix_is_none() {
    assert_eq!(parse("CAP theorem"), None);
    assert_eq!(parse("distributed systems"), None);
}

#[test]
fn explicit_expand_line_is_none() {
    assert_eq!(parse("expand: error handling best practices"), None);
}

#[test]
fn empty_queries_are_none() {
    assert_eq!(parse(""), None);
    assert_eq!(parse("   "), None);
    assert_eq!(parse("\n\n"), None);
    assert_eq!(parse("  \n  \n  "), None);
}

// ── single prefixed queries ──

#[test]
fn single_prefixed_lines() {
    assert_eq!(
        parse("lex: CAP theorem").unwrap().searches,
        vec![lex("CAP theorem", 1)]
    );
    assert_eq!(
        parse("vec: what is the CAP theorem").unwrap().searches,
        vec![vec_q("what is the CAP theorem", 1)]
    );
    assert_eq!(
        parse("hyde: The CAP theorem states that...")
            .unwrap()
            .searches,
        vec![hyde("The CAP theorem states that...", 1)]
    );
}

#[test]
fn case_insensitive_prefixes() {
    assert_eq!(
        parse("LEX: keywords").unwrap().searches,
        vec![lex("keywords", 1)]
    );
    assert_eq!(
        parse("VEC: question").unwrap().searches,
        vec![vec_q("question", 1)]
    );
    assert_eq!(
        parse("HYDE: passage").unwrap().searches,
        vec![hyde("passage", 1)]
    );
    assert_eq!(parse("Lex: test").unwrap().searches, vec![lex("test", 1)]);
    assert_eq!(parse("VeC: test").unwrap().searches, vec![vec_q("test", 1)]);
}

// ── multiple prefixed queries ──

#[test]
fn multiple_prefixed_queries() {
    let p = parse("lex: keywords\nvec: natural language").unwrap();
    assert_eq!(
        p.searches,
        vec![lex("keywords", 1), vec_q("natural language", 2)]
    );

    let p = parse("lex: keywords\nvec: question\nhyde: hypothetical doc").unwrap();
    assert_eq!(
        p.searches,
        vec![
            lex("keywords", 1),
            vec_q("question", 2),
            hyde("hypothetical doc", 3)
        ]
    );

    let p = parse("lex: term1\nlex: term2\nlex: term3").unwrap();
    assert_eq!(
        p.searches,
        vec![lex("term1", 1), lex("term2", 2), lex("term3", 3)]
    );

    // Order preserved.
    let p = parse("hyde: passage\nvec: question\nlex: keywords").unwrap();
    assert_eq!(
        p.searches,
        vec![hyde("passage", 1), vec_q("question", 2), lex("keywords", 3)]
    );
}

// ── mixed plain and prefixed ──

#[test]
fn mixed_plain_and_prefixed_errors() {
    assert!(
        err("plain keywords\nvec: semantic question").contains("missing a lex:/vec:/hyde:/intent:")
    );
    assert!(
        err("keywords\nhyde: passage\nvec: question").contains("missing a lex:/vec:/hyde:/intent:")
    );
}

// ── error cases ──

#[test]
fn multiple_plain_lines_error() {
    assert!(err("line one\nline two").contains("missing a lex:/vec:/hyde:/intent:"));
    assert!(err("a\nb\nc").contains("missing a lex:/vec:/hyde:/intent:"));
}

#[test]
fn expand_mixed_with_other_lines_errors() {
    assert!(err("expand: question\nlex: keywords").contains("cannot mix expand with typed lines"));
}

#[test]
fn expand_without_text_errors() {
    assert!(err("expand:   ").contains("must include text"));
}

#[test]
fn typed_line_without_text_errors() {
    assert!(err("lex:   \nvec: real").contains("must include text"));
}

// ── whitespace handling ──

#[test]
fn empty_and_whitespace_lines_ignored_with_line_numbers() {
    let p = parse("lex: keywords\n\nvec: question\n").unwrap();
    assert_eq!(p.searches, vec![lex("keywords", 1), vec_q("question", 3)]);

    let p = parse("lex: keywords\n   \nvec: question").unwrap();
    assert_eq!(p.searches, vec![lex("keywords", 1), vec_q("question", 3)]);
}

#[test]
fn leading_trailing_whitespace_trimmed() {
    let p = parse("  lex: keywords  \n  vec: question  ").unwrap();
    assert_eq!(p.searches, vec![lex("keywords", 1), vec_q("question", 2)]);
}

#[test]
fn internal_whitespace_preserved() {
    assert_eq!(
        parse("lex:   multiple   spaces   ").unwrap().searches,
        vec![lex("multiple   spaces", 1)]
    );
}

#[test]
fn empty_prefix_values_error() {
    assert!(err("lex: \nvec: actual query").contains("must include text"));
    assert!(err("lex: \nvec: \nhyde: ").contains("must include text"));
}

// ── edge cases ──

#[test]
fn colon_and_prefix_like_text_preserved() {
    assert_eq!(
        parse("lex: time: 12:30 PM").unwrap().searches,
        vec![lex("time: 12:30 PM", 1)]
    );
    assert_eq!(
        parse("vec: what does lex: mean").unwrap().searches,
        vec![vec_q("what does lex: mean", 1)]
    );
    assert_eq!(
        parse("hyde: The answer is X. It means Y.")
            .unwrap()
            .searches,
        vec![hyde("The answer is X. It means Y.", 1)]
    );
}

#[test]
fn carriage_return_inside_typed_text_errors() {
    assert!(err("lex: foo\rbar").contains("contains a newline"));
}

// ── parseStructuredQuery with intent (intent.test.ts) ──

#[test]
fn intent_plus_lex() {
    let p = parse("intent: api design\nlex: rest endpoints").unwrap();
    assert_eq!(p.intent.as_deref(), Some("api design"));
    assert_eq!(p.searches, vec![lex("rest endpoints", 2)]);
}

#[test]
fn intent_plus_multiple_typed_lines() {
    let p = parse("intent: perf\nlex: latency\nvec: web vitals\nhyde: hypothetical").unwrap();
    assert_eq!(p.intent.as_deref(), Some("perf"));
    assert_eq!(
        p.searches,
        vec![
            lex("latency", 2),
            vec_q("web vitals", 3),
            hyde("hypothetical", 4)
        ]
    );
}

#[test]
fn intent_after_typed_lines() {
    let p = parse("lex: q\nintent: ctx").unwrap();
    assert_eq!(p.intent.as_deref(), Some("ctx"));
    assert_eq!(p.searches, vec![lex("q", 1)]);
}

#[test]
fn intent_prefix_case_insensitive() {
    let p = parse("INTENT: api\nlex: q").unwrap();
    assert_eq!(p.intent.as_deref(), Some("api"));
    let p = parse("Intent: api\nlex: q").unwrap();
    assert_eq!(p.intent.as_deref(), Some("api"));
}

#[test]
fn no_intent_returns_none_field() {
    let p = parse("lex: q").unwrap();
    assert!(p.intent.is_none());
}

#[test]
fn intent_alone_errors() {
    assert!(err("intent: api design").contains("cannot appear alone"));
}

#[test]
fn duplicate_intent_errors() {
    assert!(err("intent: a\nintent: b\nlex: q").contains("only one intent: line is allowed"));
}

#[test]
fn empty_intent_text_errors() {
    assert!(err("intent:\nlex: q").contains("intent: must include text"));
    assert!(err("intent:   \nlex: q").contains("intent: must include text"));
}

#[test]
fn intent_with_expand_errors() {
    assert!(err("intent: web\nexpand: performance").contains("cannot mix expand"));
}

#[test]
fn intent_with_blank_lines_parses() {
    let p = parse("intent: web perf\n\nlex: performance\n\nvec: speed").unwrap();
    assert_eq!(p.intent.as_deref(), Some("web perf"));
    assert_eq!(p.searches, vec![lex("performance", 3), vec_q("speed", 5)]);
}

#[test]
fn intent_preserves_colons_in_text() {
    let p = parse("intent: web performance: LCP, FID, CLS\nlex: performance").unwrap();
    assert_eq!(p.intent.as_deref(), Some("web performance: LCP, FID, CLS"));
    assert_eq!(p.searches, vec![lex("performance", 2)]);
}

// ── extractIntentTerms (intent.test.ts) ──

#[test]
fn intent_filters_stop_words() {
    // "looking", "for", "notes", "about" are stop words.
    assert_eq!(
        extract_intent_terms("looking for notes about latency optimization"),
        vec!["latency", "optimization"]
    );
}

#[test]
fn intent_filters_common_function_words() {
    // "what", "is", "the", "to", "find" are stop words; "best", "way" survive.
    assert_eq!(
        extract_intent_terms("what is the best way to find"),
        vec!["best", "way"]
    );
}

#[test]
fn intent_preserves_domain_terms() {
    assert_eq!(
        extract_intent_terms("web performance latency page load times"),
        vec!["web", "performance", "latency", "page", "load", "times"]
    );
}

#[test]
fn intent_strips_surrounding_punctuation() {
    assert_eq!(
        extract_intent_terms("personal health, fitness, and endurance"),
        vec!["personal", "health", "fitness", "endurance"]
    );
}

#[test]
fn intent_preserves_internal_hyphens() {
    assert_eq!(
        extract_intent_terms("self-hosted real-time (decision-making)"),
        vec!["self-hosted", "real-time", "decision-making"]
    );
}

#[test]
fn intent_short_domain_terms_survive() {
    assert_eq!(
        extract_intent_terms("API design for LLM agents"),
        vec!["api", "design", "llm", "agents"]
    );
}

#[test]
fn intent_empty_input() {
    assert_eq!(extract_intent_terms(""), Vec::<String>::new());
    assert_eq!(extract_intent_terms("  "), Vec::<String>::new());
}

#[test]
fn intent_filters_single_char_terms() {
    assert_eq!(extract_intent_terms("a b c web"), vec!["web"]);
}

#[test]
fn intent_all_stop_words_returns_empty() {
    assert_eq!(
        extract_intent_terms("the and or but in on at to for of with by"),
        Vec::<String>::new()
    );
}

#[test]
fn intent_two_char_domain_terms_survive() {
    let terms = extract_intent_terms("SQL CI CD DB");
    for t in ["sql", "ci", "cd", "db"] {
        assert!(terms.iter().any(|x| x == t));
    }
}

#[test]
fn intent_lowercases_terms() {
    let terms = extract_intent_terms("WebSocket HTTP REST");
    for t in ["websocket", "http", "rest"] {
        assert!(terms.iter().any(|x| x == t));
    }
}

#[test]
fn intent_handles_cpp_punctuation() {
    // "c++" strips to "c" (length 1) and is filtered; the words survive.
    let terms = extract_intent_terms("C++, performance! optimization.");
    for t in ["performance", "optimization"] {
        assert!(terms.iter().any(|x| x == t));
    }
    assert!(!terms.iter().any(|x| x == "c"));
}
