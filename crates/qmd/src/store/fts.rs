//! FTS5 helpers, ported from upstream `src/store.ts`:
//! `normalizeCjkForFTS` (883), `sanitizeFTS5Term` (3888),
//! `FTS5_SEPARATOR_RUN` (3902), `splitFTS5CompoundTerm` (3915),
//! `containsCjk`/`sanitizeFTS5Phrase` (887-901), `buildFTS5Query`
//! (3946-4055), `validateLexQuery`/`validateSemanticQuery` (4064-4080),
//! `rebuildDocumentFTS` (2954-2976).
//!
//! Deviation: upstream additionally keeps FTS in sync through triggers
//! installed by `installFtsSyncTriggers`; this implementation updates the FTS
//! row explicitly inside each document mutation instead (P1 T3).

use std::sync::LazyLock;

use regex::Regex;
use rusqlite::{Connection, OptionalExtension, params};

use crate::error::{DbError, Result};

/// Upstream `CJK_RUN_PATTERN` (store.ts:871): runs of Han, Hiragana,
/// Katakana or Hangul characters.
static CJK_RUN: LazyLock<Regex> = LazyLock::new(|| {
    #[allow(clippy::expect_used, reason = "the pattern is a compile-time constant")]
    Regex::new(r"[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]+")
        .expect("CJK run pattern is valid")
});

/// Upstream `normalizeCjkForFTS` (store.ts:883-885).
///
/// FTS5's unicode61 tokenizer does not segment CJK text into searchable
/// words, so CJK runs are spaced character by character; the matching
/// query-side rewrite lives in P1.3.
pub(crate) fn normalize_cjk_for_fts(text: &str) -> String {
    CJK_RUN
        .replace_all(text, |caps: &regex::Captures<'_>| {
            let spaced = caps[0]
                .chars()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            format!(" {spaced} ")
        })
        .into_owned()
}

/// A run of characters the FTS5 tokenizer treats as a separator —
/// upstream `FTS5_SEPARATOR_RUN` (store.ts:3902). Letters, digits,
/// apostrophes and underscores are kept; everything else splits.
static FTS5_SEPARATOR_RUN: LazyLock<Regex> = LazyLock::new(|| {
    #[allow(clippy::expect_used, reason = "the pattern is a compile-time constant")]
    Regex::new(r"[^\p{L}\p{N}'_]+").expect("FTS5 separator pattern is valid")
});

/// Upstream `sanitizeFTS5Term` (store.ts:3888): strip separator
/// characters and lowercase.
#[must_use]
pub fn sanitize_fts5_term(term: &str) -> String {
    FTS5_SEPARATOR_RUN.replace_all(term, "").to_lowercase()
}

/// Upstream `splitFTS5CompoundTerm` (store.ts:3915-3917): split one
/// query term the way the tokenizer split the document text.
pub(crate) fn split_fts5_compound_term(term: &str) -> Vec<String> {
    FTS5_SEPARATOR_RUN
        .split(term)
        .map(sanitize_fts5_term)
        .filter(|p| !p.is_empty())
        .collect()
}

/// Upstream `containsCjk` (store.ts:887-889).
fn contains_cjk(text: &str) -> bool {
    CJK_RUN.is_match(text)
}

/// Upstream `sanitizeFTS5Phrase` (store.ts:891-902): CJK-normalize,
/// split on whitespace, split each token on tokenizer separators,
/// join back.
fn sanitize_fts5_phrase(phrase: &str) -> String {
    let normalized = normalize_cjk_for_fts(phrase);
    let mut out: Vec<String> = Vec::new();
    for token in normalized.split_whitespace() {
        out.extend(split_fts5_compound_term(token));
    }
    out.join(" ")
}

/// One scanned term of [`build_fts5_query`]: the sanitized FTS fragment
/// and whether it is negated.
fn lex_fragment(term: &str, is_phrase: bool) -> Option<String> {
    if is_phrase || contains_cjk(term) {
        let sanitized = sanitize_fts5_phrase(term.trim());
        return (!sanitized.is_empty()).then(|| format!("\"{sanitized}\""));
    }
    match split_fts5_compound_term(term).as_slice() {
        [] => None,
        [one] => Some(format!("\"{one}\"*")),
        parts => Some(format!("\"{}\"", parts.join(" "))),
    }
}

/// Upstream `buildFTS5Query` (store.ts:3946-4055).
///
/// Lex query syntax: quoted phrases (`"exact phrase"`), negation
/// (`-term` / `-"phrase"` via FTS5 binary `NOT`), compound terms
/// (`multi-agent`, `DEC-0054`, `src/lib/i18n.ts`) become adjacent-token
/// phrases, single terms keep `"term"*` prefix match, CJK terms become
/// phrases over per-character tokens. Returns `None` when no positive
/// term survives (FTS5 `NOT` is binary — negation alone cannot match).
#[must_use]
#[allow(
    clippy::indexing_slicing,
    reason = "every index is bounds-checked against `s.len()` in the same statement; the lint cannot see it"
)]
pub fn build_fts5_query(query: &str) -> Option<String> {
    let s: Vec<char> = query.trim().chars().collect();
    let mut positive: Vec<String> = Vec::new();
    let mut negative: Vec<String> = Vec::new();
    let mut i = 0;

    while i < s.len() {
        while i < s.len() && s[i].is_whitespace() {
            i += 1;
        }
        if i >= s.len() {
            break;
        }

        let negated = s[i] == '-';
        if negated {
            i += 1;
        }

        let fragment = if i < s.len() && s[i] == '"' {
            i += 1;
            let start = i;
            while i < s.len() && s[i] != '"' {
                i += 1;
            }
            let phrase: String = s[start..i].iter().collect();
            i += 1;
            lex_fragment(&phrase, true)
        } else {
            let start = i;
            while i < s.len() && !s[i].is_whitespace() && s[i] != '"' {
                i += 1;
            }
            let term: String = s[start..i].iter().collect();
            lex_fragment(&term, false)
        };

        if let Some(fragment) = fragment {
            if negated {
                negative.push(fragment);
            } else {
                positive.push(fragment);
            }
        }
    }

    if positive.is_empty() {
        return None;
    }

    let mut result = positive.join(" AND ");
    for neg in &negative {
        result = format!("{result} NOT {neg}");
    }
    Some(result)
}

/// Upstream `validateLexQuery` (store.ts:4073-4079): lex queries are
/// single-line with balanced double quotes.
#[must_use]
pub fn validate_lex_query(query: &str) -> Option<&'static str> {
    if query.contains(['\r', '\n']) {
        return Some(
            "Lex queries must be a single line. Remove newline characters or split into separate lex: lines.",
        );
    }
    if query.matches('"').count() % 2 == 1 {
        return Some(
            "Lex query has an unmatched double quote (\"). Add the closing quote or remove it.",
        );
    }
    None
}

/// Upstream `validateSemanticQuery` (store.ts:4064-4071): vec/hyde
/// queries reject lex-style `-term` negation at token boundaries.
/// `[A-Za-z0-9_]` mirrors JS `\w` (ASCII only), not the Unicode `\w`.
#[must_use]
pub fn validate_semantic_query(query: &str) -> Option<&'static str> {
    static NEGATION: LazyLock<Regex> = LazyLock::new(|| {
        #[allow(clippy::expect_used, reason = "the pattern is a compile-time constant")]
        Regex::new(r#"(^|\s)-[A-Za-z0-9_"]"#).expect("semantic-query validation pattern is valid")
    });
    if NEGATION.is_match(query) {
        return Some(
            "Negation (-term) is not supported in vec/hyde queries. Use lex for exclusions.",
        );
    }
    None
}

/// Upstream `rebuildDocumentFTS` (store.ts:2954-2976): delete the FTS row
/// and reinsert it for an active document; inactive or missing rows leave
/// `documents_fts` empty.
pub(crate) fn rebuild_document_fts(conn: &Connection, document_id: i64) -> Result<()> {
    let row = conn
        .query_row(
            "SELECT d.id, d.collection, d.path, d.title, content.doc \
             FROM documents d \
             JOIN content ON content.hash = d.hash \
             WHERE d.id = ? AND d.active = 1",
            [document_id],
            |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                ))
            },
        )
        .optional()
        .map_err(DbError::from)?;

    conn.execute("DELETE FROM documents_fts WHERE rowid = ?", [document_id])
        .map_err(DbError::from)?;

    if let Some((id, collection, path, title, body)) = row {
        conn.execute(
            "INSERT INTO documents_fts(rowid, filepath, title, body) \
             VALUES (?, ?, ?, ?)",
            params![
                id,
                normalize_cjk_for_fts(&format!("{collection}/{path}")),
                normalize_cjk_for_fts(&title),
                normalize_cjk_for_fts(&body),
            ],
        )
        .map_err(DbError::from)?;
    }
    Ok(())
}
