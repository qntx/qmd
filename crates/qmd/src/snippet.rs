//! Snippet extraction and line numbering, ported from upstream
//! `src/store.ts`: `extractSnippet` (5336-5418) and `addLineNumbers`
//! (5424-5430).
//!
//! Deviation: `max_len` truncation counts UTF-16 units like upstream,
//! but a cut landing inside a surrogate pair rounds down to the whole
//! character — upstream could emit a lone surrogate, we always emit
//! valid UTF-8 (identical output except at astral boundaries).

use crate::chunk::{CHUNK_SIZE_CHARS, is_js_ws, js_trim, utf16_len, utf16_to_byte_offset};
use crate::query::{INTENT_WEIGHT_SNIPPET, extract_intent_terms};

/// Upstream `SnippetResult` (store.ts:5290-5296).
#[derive(Debug)]
#[non_exhaustive]
pub struct SnippetResult {
    /// 1-indexed line number of the best match.
    pub line: usize,
    /// Snippet text with the `@@ -start,count @@` diff-style header.
    pub snippet: String,
    /// Lines in the document before the snippet.
    pub lines_before: usize,
    /// Lines in the document after the snippet.
    pub lines_after: usize,
    /// Number of lines in the snippet.
    pub snippet_lines: usize,
}

/// A UTF-16-unit span inside the body, used to focus snippet extraction
/// on the reranked chunk — upstream `chunkPos`/`chunkLen`
/// (store.ts:5336, 5341-5351).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ChunkSpan {
    /// UTF-16 offset where the chunk starts.
    pub pos: usize,
    /// Chunk length in UTF-16 units; `0` falls back to
    /// [`CHUNK_SIZE_CHARS`] (upstream `chunkLen || CHUNK_SIZE_CHARS`).
    pub len: usize,
}

impl ChunkSpan {
    /// A window at UTF-16 offset `pos`; `len == 0` uses the default
    /// chunk size as the window length (upstream `chunkLen || N`).
    #[must_use]
    pub const fn new(pos: usize, len: usize) -> Self {
        Self { pos, len }
    }
}

/// Upstream `extractSnippet` (store.ts:5336-5418).
///
/// When `chunk` is given, only the `±100`-unit window around it is
/// scored; on zero score a `pos == 0` window retries on the full body
/// while a `pos > 0` window anchors on the chunk start. Every line
/// scores by the number of lowercase query terms it contains plus
/// [`INTENT_WEIGHT_SNIPPET`] per intent term; the snippet is a window
/// of one line before and two after the best line, prefixed by a
/// `@@ -start,count @@ (N before, M after)` header.
#[must_use]
pub fn extract_snippet(
    body: &str,
    query: &str,
    max_len: usize,
    chunk: Option<ChunkSpan>,
    intent: Option<&str>,
) -> SnippetResult {
    // Split on '\n' (not `.lines()`) so a trailing newline produces a
    // final empty line, matching JS `split('\n')` line counts.
    let total_lines = body.split('\n').count();
    let mut search_body = body;
    let mut line_offset = 0usize;
    let mut context_start = 0usize; // UTF-16 units

    if let Some(span) = chunk {
        let search_len = if span.len == 0 {
            CHUNK_SIZE_CHARS
        } else {
            span.len
        };
        context_start = span.pos.saturating_sub(100);
        let context_end =
            (span.pos.saturating_add(search_len).saturating_add(100)).min(utf16_len(body));
        let start_b = utf16_to_byte_offset(body, context_start);
        let end_b = utf16_to_byte_offset(body, context_end);
        search_body = &body[start_b..end_b];
        if context_start > 0 {
            // `body.slice(0, contextStart).split('\n').length - 1` —
            // the number of newlines before the window.
            line_offset = body[..start_b].bytes().filter(|&b| b == b'\n').count();
        }
    }

    let lines: Vec<&str> = search_body.split('\n').collect();
    // JS `split(/\s+/)` (store.ts:5354).
    let query_terms: Vec<String> = query
        .to_lowercase()
        .split(is_js_ws)
        .filter(|t| !t.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    let intent_terms = intent.map_or_else(Vec::new, extract_intent_terms);

    let mut best_line = 0usize;
    let mut best_score = -1.0f64;
    for (i, line) in lines.iter().enumerate() {
        let lower = line.to_lowercase();
        let mut score = 0.0f64;
        for term in &query_terms {
            if lower.contains(term.as_str()) {
                score += 1.0;
            }
        }
        for term in &intent_terms {
            if lower.contains(term.as_str()) {
                score += INTENT_WEIGHT_SNIPPET;
            }
        }
        if score > best_score {
            best_score = score;
            best_line = i;
        }
    }

    if let Some(span) = chunk.filter(|_| best_score <= 0.0) {
        if span.pos == 0 {
            // pos=0 may be the chunk selector's initialization default
            // — retry on the full body (upstream comment at 5375-5379).
            return extract_snippet(body, query, max_len, None, intent);
        }
        // The reranker actively picked this chunk; anchor on its start
        // (upstream comment at 5380-5387).
        if span.pos > context_start {
            let rel_end = utf16_to_byte_offset(search_body, span.pos - context_start);
            best_line = search_body[..rel_end]
                .bytes()
                .filter(|&b| b == b'\n')
                .count();
        } else {
            best_line = 0;
        }
    }

    let start = best_line.saturating_sub(1);
    let end = (best_line + 3).min(lines.len());
    let snippet_lines = lines.get(start..end).unwrap_or_default();
    let mut snippet_text = snippet_lines.join("\n");

    // A focused window that produced a whitespace-only snippet falls
    // back to a full-document snippet.
    if chunk.is_some_and(|s| s.pos > 0) && js_trim(&snippet_text).is_empty() {
        return extract_snippet(body, query, max_len, None, intent);
    }

    if utf16_len(&snippet_text) > max_len {
        // `substring(0, maxLen - 3)` upstream — UTF-16 units; the
        // byte-offset helper rounds a mid-pair cut down to the whole
        // character.
        let cut = utf16_to_byte_offset(&snippet_text, max_len.saturating_sub(3));
        snippet_text = format!("{}...", &snippet_text[..cut]);
    }

    let absolute_start = line_offset + start + 1; // 1-indexed
    let snippet_line_count = snippet_lines.len();
    let lines_before = absolute_start - 1;
    let lines_after = total_lines - (absolute_start + snippet_line_count - 1);

    let header = format!(
        "@@ -{absolute_start},{snippet_line_count} @@ ({lines_before} before, {lines_after} after)"
    );
    SnippetResult {
        line: line_offset + best_line + 1,
        snippet: format!("{header}\n{snippet_text}"),
        lines_before,
        lines_after,
        snippet_lines: snippet_line_count,
    }
}

/// Upstream `addLineNumbers` (store.ts:5424-5430): `{n}: {content}` per
/// line, starting at `start_line` (1 by default).
#[must_use]
pub fn add_line_numbers(text: &str, start_line: usize) -> String {
    text.split('\n')
        .enumerate()
        .map(|(i, l)| format!("{}: {l}", start_line + i))
        .collect::<Vec<_>>()
        .join("\n")
}
