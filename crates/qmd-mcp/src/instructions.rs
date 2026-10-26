//! Dynamic server instructions — upstream `buildInstructions`
//! (mcp/server.ts:124-174).

use qmd::Qmd;

/// Build the `instructions` string from live index state — injected
/// via initialize (2025) and `server/discover` (2026). Byte-identical
/// to upstream for the same store state.
///
/// # Errors
/// [`qmd::Error`] on status/context read failure.
pub(crate) fn build_instructions(qmd: &Qmd) -> Result<String, qmd::Error> {
    let status = qmd.status()?;
    let global_ctx = qmd.global_context()?;
    let mut lines: Vec<String> = Vec::new();

    lines.push(format!(
        "QMD is your local search engine over {} markdown documents.",
        status.total_documents
    ));
    if let Some(ctx) = global_ctx {
        lines.push(format!("Context: {ctx}"));
    }

    // Names only — per-collection detail is on demand via `status`.
    if !status.collections.is_empty() {
        lines.push(String::new());
        let names = status
            .collections
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        lines.push(format!(
            "Collections (scope with `collections` parameter): {names}"
        ));
        lines.push(
            "Call the `status` tool for collection descriptions, paths, and per-collection doc counts."
                .to_owned(),
        );
    }

    if !status.has_vector_index {
        lines.push(String::new());
        lines.push(
            "Note: No vector embeddings yet. Run `qmd embed` to enable semantic search (vec/hyde)."
                .to_owned(),
        );
    } else if status.needs_embedding > 0 {
        lines.push(String::new());
        lines.push(format!(
            "Note: {} documents need embedding. Run `qmd embed` to update.",
            status.needs_embedding
        ));
    }

    lines.push(String::new());
    lines.push("Search: Use `query` with sub-queries (lex/vec/hyde):".to_owned());
    lines.push("  - type:'lex' — BM25 keyword search (exact terms, fast)".to_owned());
    lines.push("  - type:'vec' — semantic vector search (meaning-based)".to_owned());
    lines.push(
        "  - type:'hyde' — hypothetical document (write what the answer looks like)".to_owned(),
    );
    lines.push(String::new());
    lines.push(
        "  Always provide `intent` on every search call to disambiguate and improve snippets."
            .to_owned(),
    );
    lines.push(String::new());
    lines.push("Examples:".to_owned());
    lines.push("  Quick keyword lookup: [{type:'lex', query:'error handling'}]".to_owned());
    lines.push(
        "  Semantic search: [{type:'vec', query:'how to handle errors gracefully'}]".to_owned(),
    );
    lines.push(
        "  Best results: [{type:'lex', query:'error'}, {type:'vec', query:'error handling best practices'}]".to_owned(),
    );
    lines.push(
        "  With intent: searches=[{type:'lex', query:'performance'}], intent='web page load times'"
            .to_owned(),
    );

    lines.push(String::new());
    lines.push("Retrieval:".to_owned());
    lines.push(
        "  - `get` — single document by path or docid (#abc123). Supports a line-range suffix: `file.md:100` (from line 100) or `file.md:100:40` (40 lines from line 100).".to_owned(),
    );
    lines.push(
        "  - `multi_get` — batch retrieve by glob (`journals/2025-05*.md`), comma-separated list, or docids (#abc123).".to_owned(),
    );

    lines.push(String::new());
    lines.push("Tips:".to_owned());
    lines.push("  - File paths in results are relative to their collection.".to_owned());
    lines.push("  - Use `minScore: 0.5` to filter low-confidence results.".to_owned());
    lines.push("  - Results include a `context` field describing the content type.".to_owned());

    Ok(lines.join("\n"))
}
