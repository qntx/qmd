#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_assert_message,
    clippy::float_cmp,
    reason = "idiomatic test-code patterns"
)]

//! AC5 golden test — every P2.4 pure function compared field-for-field
//! against upstream output recorded in `fixtures/hybrid_golden.json`
//! (regenerate with `scripts/gen_hybrid_golden.ts`).
//!
//! `getCacheKey` and the `pub(crate)` RRF helpers are covered inside
//! `store::cache` / `store::rrf` unit tests since they are not public.

use qmd::{ChunkSpan, QueryKind, extract_intent_terms, extract_snippet, parse_query_document};

fn golden() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/hybrid_golden.json"))
        .expect("hybrid_golden.json parses")
}

fn kind_str(k: QueryKind) -> &'static str {
    match k {
        QueryKind::Original => "original",
        QueryKind::Lex => "lex",
        QueryKind::Vec => "vec",
        QueryKind::Hyde => "hyde",
        _ => panic!("unknown QueryKind"),
    }
}

#[test]
fn parse_query_document_matches_upstream() {
    let g = golden();
    for case in g["cases"]["parse_query_doc"].as_array().unwrap() {
        let input = case["input"].as_str().unwrap();
        let expected_error = case["error"].as_str();
        match parse_query_document(input) {
            Err(e) => {
                assert_eq!(
                    Some(e.to_string().as_str()),
                    expected_error,
                    "input {input:?} should error with upstream message"
                );
            }
            Ok(None) => {
                assert!(
                    expected_error.is_none() && case["result"].is_null(),
                    "input {input:?} should parse to null upstream, got None result"
                );
            }
            Ok(Some(doc)) => {
                let result = &case["result"];
                assert!(
                    !result.is_null(),
                    "input {input:?} unexpectedly parsed: {doc:?}"
                );
                let searches = result["searches"].as_array().unwrap();
                assert_eq!(doc.searches.len(), searches.len(), "input {input:?}");
                for (got, want) in doc.searches.iter().zip(searches) {
                    assert_eq!(
                        kind_str(got.kind),
                        want["type"].as_str().unwrap(),
                        "{input:?}"
                    );
                    assert_eq!(got.query, want["query"].as_str().unwrap(), "{input:?}");
                    assert_eq!(
                        got.line.map(|l| l as u64),
                        want["line"].as_u64(),
                        "{input:?}"
                    );
                }
                match result["intent"].as_str() {
                    Some(want) => assert_eq!(doc.intent.as_deref(), Some(want), "{input:?}"),
                    None => assert_eq!(doc.intent, None, "{input:?}"),
                }
            }
        }
    }
}

#[test]
fn extract_intent_terms_matches_upstream() {
    let g = golden();
    for case in g["cases"]["intent_terms"].as_array().unwrap() {
        let intent = case["intent"].as_str().unwrap();
        let want: Vec<&str> = case["terms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap())
            .collect();
        let got = extract_intent_terms(intent);
        assert_eq!(
            got.iter().map(String::as_str).collect::<Vec<_>>(),
            want,
            "intent {intent:?}"
        );
    }
}

#[test]
fn extract_snippet_matches_upstream() {
    let g = golden();
    for (i, case) in g["cases"]["snippet"].as_array().unwrap().iter().enumerate() {
        let body = case["body"].as_str().unwrap();
        let query = case["query"].as_str().unwrap();
        let max_len = usize::try_from(case["maxLen"].as_u64().unwrap()).unwrap();
        let span = case["chunkPos"].as_u64().map(|pos| {
            ChunkSpan::new(
                usize::try_from(pos).unwrap(),
                usize::try_from(case["chunkLen"].as_u64().unwrap_or(0)).unwrap(),
            )
        });
        let intent = case["intent"].as_str();
        let want = &case["result"];
        let got = extract_snippet(body, query, max_len, span, intent);
        assert_eq!(
            got.snippet,
            want["snippet"].as_str().unwrap(),
            "case {i} snippet"
        );
        assert_eq!(
            got.line as u64,
            want["line"].as_u64().unwrap(),
            "case {i} line"
        );
        assert_eq!(
            got.lines_before as u64,
            want["linesBefore"].as_u64().unwrap(),
            "case {i} linesBefore"
        );
        assert_eq!(
            got.lines_after as u64,
            want["linesAfter"].as_u64().unwrap(),
            "case {i} linesAfter"
        );
        assert_eq!(
            got.snippet_lines as u64,
            want["snippetLines"].as_u64().unwrap(),
            "case {i} snippetLines"
        );
    }
}
