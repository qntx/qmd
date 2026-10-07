//! Snippet extraction and line numbering, ported from upstream
//! `src/store.ts`: `extractSnippet` (5336-5418) and `addLineNumbers`
//! (5424-5430).
//!
//! Deviations: the `chunkPos`/`chunkLen` window and `intent` weighting
//! arrive with chunking (P2) and query intent (P3); this version always
//! searches the full body. `max_len` truncation counts Rust `char`s
//! while upstream counts UTF-16 units — equal except for astral
//! characters, where the Rust version keeps whole code points.

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

/// Upstream `extractSnippet` (store.ts:5336-5418) without the chunk
/// window and intent parameters.
///
/// Every line scores by the number of lowercase query terms it
/// contains; the snippet is a window of one line before and two after
/// the best line, prefixed by a `@@ -start,count @@ (N before, M
/// after)` header.
#[must_use]
pub fn extract_snippet(body: &str, query: &str, max_len: usize) -> SnippetResult {
    // Split on '\n' (not `.lines()`) so a trailing newline produces a
    // final empty line, matching JS `split('\n')` line counts.
    let lines: Vec<&str> = body.split('\n').collect();
    let total_lines = lines.len();

    let query_terms: Vec<String> = query
        .to_lowercase()
        .split_whitespace()
        .filter(|t| !t.is_empty())
        .map(ToOwned::to_owned)
        .collect();

    let mut best_line = 0usize;
    let mut best_score = -1.0f64;
    for (i, line) in lines.iter().enumerate() {
        let lower = line.to_lowercase();
        let score = query_terms
            .iter()
            .filter(|term| lower.contains(term.as_str()))
            .count() as f64;
        if score > best_score {
            best_score = score;
            best_line = i;
        }
    }

    let start = best_line.saturating_sub(1);
    let end = (best_line + 3).min(lines.len());
    let snippet_lines = lines.get(start..end).unwrap_or_default();
    let mut snippet_text = snippet_lines.join("\n");
    if snippet_text.chars().count() > max_len {
        let truncated: String = snippet_text
            .chars()
            .take(max_len.saturating_sub(3))
            .collect();
        snippet_text = format!("{truncated}...");
    }

    let absolute_start = start + 1; // 1-indexed
    let snippet_line_count = snippet_lines.len();
    let lines_before = absolute_start - 1;
    let lines_after = total_lines - (absolute_start + snippet_line_count - 1);

    let header = format!(
        "@@ -{absolute_start},{snippet_line_count} @@ ({lines_before} before, {lines_after} after)"
    );
    SnippetResult {
        line: best_line + 1,
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
