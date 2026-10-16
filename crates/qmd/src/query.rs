//! Query documents, expansion types and intent-term extraction.
//!
//! Ported from upstream `src/cli/qmd.ts` `parseStructuredQuery`
//! (2755-2823), `src/llm.ts` `expandQuery` line parsing (1686-1715) and
//! `src/store.ts` `extractIntentTerms` (5330-5334).
//!
//! The parse function lives here rather than in the CLI (P2-D2): upstream
//! keeps two near-copies in `cli/qmd.ts` and `bench/bench.ts`; the library
//! carries the CLI variant, which is the superset (it handles `expand:`
//! lines and a `\r` guard on typed text).

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;

#[allow(
    clippy::expect_used,
    reason = "the pattern is a fixed literal; a failure is a compile-time bug"
)]
fn compile_regex(pattern: &str) -> Regex {
    Regex::new(pattern).expect("static pattern must compile")
}

use crate::chunk::{is_js_ws, js_trim, utf16_len};
use crate::error::{Error, Result};

/// Expanded sub-query kind — upstream `ExpandedQuery["type"]`
/// (store.ts:440-445), plus `Original` used by ranked-list metadata to
/// mark the unexpanded caller query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueryKind {
    /// The original caller query (ranked-list metadata only).
    Original,
    /// Keyword variant — routes to FTS.
    Lex,
    /// Semantic variant — routes to vector search.
    Vec,
    /// Hypothetical document — routes to vector search.
    Hyde,
}

impl QueryKind {
    /// Upstream spelling (`lex`/`vec`/`hyde`/`original`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::Lex => "lex",
            Self::Vec => "vec",
            Self::Hyde => "hyde",
        }
    }

    /// Parse an upstream type string; `None` for anything else (upstream
    /// `as`-casts unknown strings, which then never route — for us they
    /// are dropped at parse time).
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "lex" => Some(Self::Lex),
            "vec" => Some(Self::Vec),
            "hyde" => Some(Self::Hyde),
            _ => None,
        }
    }
}

/// One expanded or structured sub-query — upstream `ExpandedQuery`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ExpandedQuery {
    /// Routing kind.
    pub kind: QueryKind,
    /// Query text.
    pub query: String,
    /// 1-indexed source line (structured query documents only).
    pub line: Option<usize>,
}

impl ExpandedQuery {
    /// A sub-query without a source line — the expansion-produced shape.
    #[must_use]
    pub fn new(kind: QueryKind, query: impl Into<String>) -> Self {
        Self {
            kind,
            query: query.into(),
            line: None,
        }
    }

    /// Attach a 1-indexed source line (structured query documents).
    #[must_use]
    pub const fn with_line(mut self, line: usize) -> Self {
        self.line = Some(line);
        self
    }
}

/// A parsed structured query document — upstream `ParsedStructuredQuery`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ParsedQueryDocument {
    /// Typed sub-queries in document order.
    pub searches: Vec<ExpandedQuery>,
    /// Optional `intent:` line text.
    pub intent: Option<String>,
}

/// Upstream `parseStructuredQuery` (cli/qmd.ts:2755-2823).
///
/// `None` means the input is a plain (single, unprefixed) query or a
/// standalone `expand:` line — both route to expansion-based hybrid
/// search. Errors carry the upstream message verbatim via
/// [`Error::InvalidQuery`].
///
/// # Errors
/// [`Error::InvalidQuery`] on any upstream `throw`.
pub fn parse_query_document(query: &str) -> Result<Option<ParsedQueryDocument>> {
    let raw_lines: Vec<(&str, usize)> = query
        .split('\n')
        .enumerate()
        .map(|(idx, line)| (js_trim(line), idx + 1))
        .filter(|(trimmed, _)| !trimmed.is_empty())
        .collect();

    if raw_lines.is_empty() {
        return Ok(None);
    }

    let mut typed: Vec<ExpandedQuery> = Vec::new();
    let mut intent: Option<String> = None;

    for &(line, number) in &raw_lines {
        let lower = line.to_lowercase();
        if lower.starts_with("expand:") {
            if raw_lines.len() > 1 {
                return Err(Error::InvalidQuery {
                    reason: format!(
                        "Line {number} starts with expand:, but query documents cannot mix expand with typed lines. Submit a single expand query instead."
                    ),
                });
            }
            let text = js_trim(&line["expand:".len()..]);
            if text.is_empty() {
                return Err(Error::InvalidQuery {
                    reason: "expand: query must include text.".to_owned(),
                });
            }
            return Ok(None); // treat as standalone expand query
        }

        if lower.starts_with("intent:") {
            if intent.is_some() {
                return Err(Error::InvalidQuery {
                    reason: format!(
                        "Line {number}: only one intent: line is allowed per query document."
                    ),
                });
            }
            let text = js_trim(&line["intent:".len()..]);
            if text.is_empty() {
                return Err(Error::InvalidQuery {
                    reason: format!("Line {number}: intent: must include text."),
                });
            }
            intent = Some(text.to_owned());
            continue;
        }

        let kind = ["lex:", "vec:", "hyde:"]
            .iter()
            .find(|p| lower.starts_with(**p))
            .and_then(|p| QueryKind::from_str(&p[..p.len() - 1]));
        if let Some(kind) = kind {
            let text = js_trim(&line[kind.as_str().len() + 1..]);
            if text.is_empty() {
                return Err(Error::InvalidQuery {
                    reason: format!("Line {number} ({}:) must include text.", kind.as_str()),
                });
            }
            if text.contains('\r') || text.contains('\n') {
                return Err(Error::InvalidQuery {
                    reason: format!(
                        "Line {number} ({}:) contains a newline. Keep each query on a single line.",
                        kind.as_str()
                    ),
                });
            }
            typed.push(ExpandedQuery {
                kind,
                query: text.to_owned(),
                line: Some(number),
            });
            continue;
        }

        if raw_lines.len() == 1 {
            // Single plain line -> implicit expand.
            return Ok(None);
        }

        return Err(Error::InvalidQuery {
            reason: format!(
                "Line {number} is missing a lex:/vec:/hyde:/intent: prefix. Each line in a query document must start with one."
            ),
        });
    }

    // intent: alone is not a valid query — must have at least one search.
    if intent.is_some() && typed.is_empty() {
        return Err(Error::InvalidQuery {
            reason: "intent: cannot appear alone. Add at least one lex:, vec:, or hyde: line."
                .to_owned(),
        });
    }

    Ok(if typed.is_empty() {
        None
    } else {
        Some(ParsedQueryDocument {
            searches: typed,
            intent,
        })
    })
}

/// Weight for intent terms relative to query terms (1.0) in snippet
/// scoring — upstream `INTENT_WEIGHT_SNIPPET` (store.ts:5299).
pub const INTENT_WEIGHT_SNIPPET: f64 = 0.3;

/// Weight for intent terms relative to query terms (1.0) in chunk
/// selection — upstream `INTENT_WEIGHT_CHUNK` (store.ts:5302).
pub const INTENT_WEIGHT_CHUNK: f64 = 0.5;

/// Upstream `INTENT_STOP_WORDS` (store.ts:5308-5323).
static INTENT_STOP_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        // 2-char function words
        "am", "an", "as", "at", "be", "by", "do", "he", "if", "in", "is", "it", "me", "my", "no",
        "of", "on", "or", "so", "to", "up", "us", "we", // 3-char function words
        "all", "and", "any", "are", "but", "can", "did", "for", "get", "has", "her", "him", "his",
        "how", "its", "let", "may", "not", "our", "out", "the", "too", "was", "who", "why", "you",
        // 4+ char common words
        "also", "does", "find", "from", "have", "into", "more", "need", "show", "some", "tell",
        "that", "them", "this", "want", "what", "when", "will", "with", "your",
        // Search-context noise
        "about", "looking", "notes", "search", "where", "which",
    ]
    .into_iter()
    .collect()
});

/// Upstream `extractIntentTerms` (store.ts:5330-5334).
///
/// Lowercase, split on JS whitespace, strip leading/trailing
/// non-letter/non-number chars (`[^\p{L}\p{N}]`), keep terms longer than
/// one UTF-16 unit that are not stop words.
#[must_use]
pub fn extract_intent_terms(intent: &str) -> Vec<String> {
    static EDGE_PUNCT: LazyLock<Regex> =
        LazyLock::new(|| compile_regex(r"(?u)^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$"));
    intent
        .to_lowercase()
        .split(is_js_ws)
        .map(|t| EDGE_PUNCT.replace_all(t, "").into_owned())
        .filter(|t| utf16_len(t) > 1 && !INTENT_STOP_WORDS.contains(t.as_str()))
        .collect()
}

/// Parse raw expansion output into expanded queries — the parse/filter
/// half of upstream `LlamaCpp.expandQuery` (llm.ts:1686-1704).
///
/// `lex:`/`vec:`/`hyde:` lines are recognized case-sensitively at the
/// first colon; each text must contain a normalized query term (empty
/// term list passes everything, upstream `hasQueryTerm`).
#[must_use]
pub(crate) fn parse_expansion_output(query: &str, raw: &str) -> Vec<ExpandedQuery> {
    let query_terms: Vec<String> = query
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_lowercase() || c.is_ascii_digit() || is_js_ws(c) {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split(is_js_ws)
        .filter(|t| !t.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    let has_query_term = |text: &str| -> bool {
        if query_terms.is_empty() {
            return true;
        }
        let lower = text.to_lowercase();
        query_terms.iter().any(|t| lower.contains(t.as_str()))
    };

    js_trim(raw)
        .split('\n')
        .filter_map(|line| {
            let colon = line.find(':')?;
            let kind = QueryKind::from_str(js_trim(&line[..colon]))?;
            let text = js_trim(&line[colon + 1..]);
            has_query_term(text).then(|| ExpandedQuery {
                kind,
                query: text.to_owned(),
                line: None,
            })
        })
        .collect()
}

/// Expansion fallback when parsing yields nothing — upstream
/// llm.ts:1710-1714 (`hyde` first, then `lex`, then `vec`).
#[must_use]
pub(crate) fn expansion_parse_fallback(query: &str) -> Vec<ExpandedQuery> {
    [
        (QueryKind::Hyde, format!("Information about {query}")),
        (QueryKind::Lex, query.to_owned()),
        (QueryKind::Vec, query.to_owned()),
    ]
    .into_iter()
    .map(|(kind, query)| ExpandedQuery {
        kind,
        query,
        line: None,
    })
    .collect()
}

/// Expansion fallback on backend error — upstream llm.ts:1716-1721
/// (`lex` then `vec`, no `hyde`).
#[must_use]
pub(crate) fn expansion_error_fallback(query: &str) -> Vec<ExpandedQuery> {
    [QueryKind::Lex, QueryKind::Vec]
        .into_iter()
        .map(|kind| ExpandedQuery {
            kind,
            query: query.to_owned(),
            line: None,
        })
        .collect()
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::shadow_unrelated,
    reason = "idiomatic test-code patterns"
)]
mod tests {
    use super::*;

    #[test]
    fn parse_query_document_plain_line_is_none() {
        assert_eq!(parse_query_document("CAP theorem").unwrap(), None);
        assert_eq!(parse_query_document("  CAP theorem  ").unwrap(), None);
    }

    #[test]
    fn parse_query_document_typed_lines() {
        let parsed = parse_query_document("lex: CAP theorem\nvec: consistency models")
            .unwrap()
            .unwrap();
        assert_eq!(parsed.searches.len(), 2);
        assert_eq!(parsed.searches[0].kind, QueryKind::Lex);
        assert_eq!(parsed.searches[0].query, "CAP theorem");
        assert_eq!(parsed.searches[0].line, Some(1));
        assert_eq!(parsed.searches[1].kind, QueryKind::Vec);
        assert!(parsed.intent.is_none());
    }

    #[test]
    fn parse_query_document_intent_line() {
        let parsed = parse_query_document("lex: a b\nintent: api design")
            .unwrap()
            .unwrap();
        assert_eq!(parsed.intent.as_deref(), Some("api design"));
    }

    #[test]
    fn parse_query_document_errors() {
        let err = parse_query_document("CAP\nconsistency").unwrap_err();
        assert!(err.to_string().contains("Line 1 is missing"));
        let err = parse_query_document("lex: a\nexpand: b").unwrap_err();
        assert!(err.to_string().contains("cannot mix expand"));
        let err = parse_query_document("intent: only").unwrap_err();
        assert!(err.to_string().contains("cannot appear alone"));
        let err = parse_query_document("lex:   \nvec: ok").unwrap_err();
        assert!(err.to_string().contains("Line 1 (lex:) must include text"));
    }

    #[test]
    fn extract_intent_terms_filters() {
        assert_eq!(
            extract_intent_terms("Find the API versioning notes!"),
            vec!["api", "versioning"] // "find"/"the"/"notes" are stop words; "!" stripped
        );
        // Single UTF-16-unit terms are dropped; two-unit terms survive
        // (emoji are stripped by the edge-punctuation rule, like upstream).
        assert_eq!(extract_intent_terms("a 🚀 42 launch"), vec!["42", "launch"]);
    }

    #[test]
    fn parse_expansion_output_filters() {
        let rows = parse_expansion_output(
            "cap theorem",
            "lex: CAP consistency\nvec: unrelated filler\nhyde: about the CAP theorem\njunk",
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].kind, QueryKind::Lex);
        assert_eq!(rows[0].query, "CAP consistency");
        assert_eq!(rows[1].kind, QueryKind::Hyde);
    }

    #[test]
    fn expansion_fallbacks() {
        let p = expansion_parse_fallback("q");
        assert_eq!(p.len(), 3);
        assert_eq!(p[0].kind, QueryKind::Hyde);
        let e = expansion_error_fallback("q");
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].kind, QueryKind::Lex);
    }
}
