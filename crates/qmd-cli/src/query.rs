//! `qmd vsearch`/`vector-search` and `qmd query`/`deep-search` —
//! upstream `vectorSearch` (qmd.ts:2905-2950) and `querySearch`
//! (qmd.ts:2952-3081), dispatched from `case "vsearch"`/`case "query"`
//! (qmd.ts:4762-4786).
//!
//! Both commands share upstream's validation order: `--format`
//! resolution (done in `parseCLI`), missing-query usage error,
//! `--filter` parsing, the local-config trust gate, `getStore()`,
//! `resolveCollectionFilter`, `checkIndexHealth`, then
//! `warnPendingMetadata` before the search itself runs.
//!
//! Upstream wraps the pipeline in `withLLMSession` (a 10-minute
//! cooperative session cap, qmd.ts:2948, 3079-3080); this port's lazy
//! backends are stateless per call, so no session cap exists — the
//! deviation only bounds how long a stuck model load could hold GPU
//! memory, not result content.

use std::cell::RefCell;
use std::time::Duration;

use qmd::{
    Environment, ExpandedQuery, QueryKind, SearchHooks, SearchInput, SearchOptions,
    VectorQueryOptions, js,
};

use crate::args::QueryArgs;
use crate::context::CliContext;
use crate::output::{
    Format, Io, OutputOptions, OutputRow, Palette, Render, editor_uri_template, output_results,
    print_empty_search_results,
};
use crate::search::{
    check_index_health, log_expansion_tree, parse_cli_metadata_filter, resolve_collection_filter,
    resolve_format, warn_pending_metadata,
};
use crate::{Failure, gate, progress};

/// `qmd vsearch` — upstream `case "vsearch"` (qmd.ts:4762-4775) →
/// `vectorSearch`.
///
/// # Errors
/// [`Failure::Reported`] after the upstream usage/collection messages;
/// [`Failure::Error`] wraps store/search failures.
pub(crate) fn vsearch(
    env: &Environment,
    ctx: &CliContext,
    args: &QueryArgs,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    // Upstream resolves --format in parseCLI, before the command case.
    let Some(format) = resolve_format(&args.search.format, io) else {
        return Err(Failure::Reported(1));
    };
    let query = args.search.query.join(" ");
    if query.is_empty() {
        let _ = writeln!(io.err, "Usage: qmd vsearch [options] <query>");
        return Err(Failure::Reported(1));
    }
    // `if (!cli.values["min-score"]) cli.opts.minScore = 0.3` — then
    // `vectorSearch` applies `opts.minScore || 0.3`, so an explicit `0`
    // or a NaN parse also lands on 0.3 (qmd.ts:4768-4771, 2916).
    let min_score = args
        .search
        .min_score
        .as_deref()
        .and_then(js::parse_float)
        .filter(|v| *v != 0.0)
        .unwrap_or(0.3);
    let Ok(filter) = parse_cli_metadata_filter(args.search.filter.as_deref(), io) else {
        return Err(Failure::Reported(1));
    };
    gate::resolve_local_config_trust(env, ctx, io)?;

    let qmd = ctx.open_store(env)?;
    let c = Palette::new(Palette::enabled(env, io));
    let collection_names = resolve_collection_filter(&qmd, &args.search.collection, true, io)?;
    check_index_health(env, &qmd, io, c)?;
    if filter.is_some() {
        warn_pending_metadata(&qmd, io, c);
    }

    let limit = pipeline_limit(&args.search, format);
    let hooks = VsearchHooks {
        io: RefCell::new(&mut *io),
        c,
    };
    // `#[non_exhaustive]` — field assignment on `default()`.
    let mut vec_opts = VectorQueryOptions::default();
    vec_opts.collections = if collection_names.is_empty() {
        None
    } else {
        Some(collection_names)
    };
    vec_opts.filter = filter;
    vec_opts.limit = limit;
    vec_opts.min_score = min_score;
    let results = qmd.vector_query(&query, &vec_opts, &hooks)?;

    if results.is_empty() {
        print_empty_search_results(format, false, io);
        return Ok(());
    }
    let rows: Vec<OutputRow> = results
        .iter()
        .map(|r| OutputRow {
            display_path: r.display_path.clone(),
            title: r.title.clone(),
            body: r.body.clone(),
            score: r.score,
            context: r.context.clone(),
            docid: r.docid.clone(),
            metadata: r.metadata.clone(),
            // `searchVec` rows never carry explain/chunk fields
            // (qmd.ts:2937-2947).
            explain: None,
            chunk: None,
        })
        .collect();
    output_rows(
        &qmd,
        ctx,
        env,
        &rows,
        &query,
        args,
        format,
        min_score,
        results.len(),
        io,
        c,
    );
    Ok(())
}

/// `qmd query` — upstream `case "query"` (qmd.ts:4777-4786) →
/// `querySearch`.
///
/// # Errors
/// [`Failure::Reported`] after the upstream usage/collection messages;
/// [`Failure::Error`] wraps store/search/structured-query failures.
pub(crate) fn query(
    env: &Environment,
    ctx: &CliContext,
    args: &QueryArgs,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    let Some(format) = resolve_format(&args.search.format, io) else {
        return Err(Failure::Reported(1));
    };
    let query = args.search.query.join(" ");
    if query.is_empty() {
        let _ = writeln!(io.err, "Usage: qmd query [options] <query>");
        return Err(Failure::Reported(1));
    }
    // `opts.minScore || 0` — query defaults to 0, but an explicit NaN
    // or `0` lands on 0 too (qmd.ts:3205, 2986).
    let min_score = args
        .search
        .min_score
        .as_deref()
        .and_then(js::parse_float)
        .unwrap_or(0.0);
    let Ok(filter) = parse_cli_metadata_filter(args.search.filter.as_deref(), io) else {
        return Err(Failure::Reported(1));
    };
    // `--chunk-strategy` is validated in parseCLI upstream
    // (qmd.ts:3213) — before the store opens.
    let chunk_strategy = parse_chunk_strategy(args.chunk_strategy.as_deref(), io)?;
    let candidate_limit = parse_candidate_limit(args.candidate_limit.as_deref());
    gate::resolve_local_config_trust(env, ctx, io)?;

    let qmd = ctx.open_store(env)?;
    let c = Palette::new(Palette::enabled(env, io));
    let collection_names = resolve_collection_filter(&qmd, &args.search.collection, true, io)?;
    check_index_health(env, &qmd, io, c)?;
    if filter.is_some() {
        warn_pending_metadata(&qmd, io, c);
    }

    // `parseStructuredQuery` — `None` routes to expansion-based hybrid
    // search; `Error` propagates like upstream's uncaught throw (the
    // message text is identical, the Bun stack dump is not portable).
    let parsed = qmd::parse_query_document(&query)?;
    // `const intent = opts.intent || parsed?.intent` — the CLI flag
    // wins over the document's `intent:` line (qmd.ts:2964).
    let intent = args
        .intent
        .clone()
        .or_else(|| parsed.as_ref().and_then(|doc| doc.intent.clone()));
    let limit = pipeline_limit(&args.search, format);

    let input = parsed.as_ref().map_or_else(
        || SearchInput::Query(query.clone()),
        |doc| {
            print_structured_header(doc, intent.as_deref(), io, c);
            SearchInput::Expanded(doc.searches.clone())
        },
    );
    let hooks = QueryHooks {
        io: RefCell::new(&mut *io),
        c,
        structured: parsed.is_some(),
    };
    // `#[non_exhaustive]` — field assignment on `default()`.
    let mut search_opts = SearchOptions::default();
    search_opts.input = input;
    search_opts.filter = filter;
    // `intent` is consumed here — `output_rows` takes `args.intent`
    // (the raw flag), matching upstream's `{...opts}` spread.
    search_opts.intent = intent;
    search_opts.rerank = !args.no_rerank;
    search_opts.collections = if collection_names.is_empty() {
        None
    } else {
        Some(collection_names)
    };
    search_opts.limit = limit;
    search_opts.candidate_limit = candidate_limit;
    search_opts.min_score = min_score;
    search_opts.explain = args.explain;
    search_opts.chunk_strategy = chunk_strategy;
    let results = qmd.search(&search_opts, &hooks)?;

    if results.is_empty() {
        print_empty_search_results(format, false, io);
        return Ok(());
    }
    // `displayQuery`: first lex → first vec → original (qmd.ts:3064-3068).
    let display_query = parsed.as_ref().map_or(query.as_str(), |doc| {
        doc.searches
            .iter()
            .find(|s| s.kind == QueryKind::Lex)
            .or_else(|| doc.searches.iter().find(|s| s.kind == QueryKind::Vec))
            .map_or(query.as_str(), |s| s.query.as_str())
    });
    let result_count = results.len();
    // `into_iter` moves each `Box<Explain>` out — `Explain` is
    // intentionally not `Clone`.
    let rows: Vec<OutputRow> = results
        .into_iter()
        .map(|r| {
            // `chunkPos: r.bestChunkPos, chunkLen: r.bestChunk.length`
            // (qmd.ts:3071-3072) — `length` is UTF-16 units.
            let chunk = Some(qmd::ChunkSpan::new(
                r.best_chunk_pos,
                qmd::utf16_len(&r.best_chunk),
            ));
            OutputRow {
                display_path: r.display_path,
                title: r.title,
                body: r.body,
                score: r.score,
                context: r.context,
                docid: r.docid,
                metadata: r.metadata,
                explain: r.explain,
                chunk,
            }
        })
        .collect();
    output_rows(
        &qmd,
        ctx,
        env,
        &rows,
        display_query,
        args,
        format,
        min_score,
        result_count,
        io,
        c,
    );
    Ok(())
}

/// `opts.limit` as upstream computes it inside `vectorSearch`/
/// `querySearch` (qmd.ts:2917, 2985): `--all` → 500, otherwise the
/// `parseCLI` value (`-n` `parseInt`-parsed against the
/// 20-for-files/json or 5 default; `0`/NaN fall back).
///
/// Negative `-n` values reach upstream as `slice(0, -k)` — "all but
/// the last k" — which the `usize` pipeline option cannot express; the
/// port clamps them to `usize::MAX` (returns everything, the same
/// result for any `-n < -len`). NaN already fell back to the default
/// upstream, so it never reaches here.
fn pipeline_limit(args: &crate::args::SearchArgs, format: Format) -> usize {
    if args.all {
        return 500;
    }
    let default_limit: i64 = if matches!(format, Format::Json | Format::Files) {
        20
    } else {
        5
    };
    let limit = args
        .n
        .as_deref()
        .and_then(js::parse_int)
        .filter(|v| *v != 0)
        .unwrap_or(default_limit);
    usize::try_from(limit).unwrap_or(usize::MAX)
}

/// `parseChunkStrategy` (qmd.ts:2125-2130): `auto`/`regex` or the
/// verbatim `exitWithError` message.
///
/// # Errors
/// [`Failure::Reported`] after writing the upstream error line.
fn parse_chunk_strategy(raw: Option<&str>, io: &mut Io<'_>) -> Result<qmd::ChunkStrategy, Failure> {
    match raw {
        // `None` and `"regex"` are the same arm upstream
        // (`chunkStrategy === "auto" ? "auto" : "regex"` is not — the
        // parse rejects other strings first).
        Some("auto") => Ok(qmd::ChunkStrategy::Auto),
        None | Some("regex") => Ok(qmd::ChunkStrategy::Regex),
        Some(other) => {
            let _ = writeln!(
                io.err,
                "--chunk-strategy must be \"auto\" or \"regex\" (got \"{other}\")"
            );
            Err(Failure::Reported(1))
        }
    }
}

/// `opts.candidateLimit` (qmd.ts:3209): `parseInt` or `undefined` —
/// upstream's `NaN ?? 40` keeps `NaN`, whose `slice(0, NaN)` is empty,
/// so an unparsable value selects *zero* candidates; a negative value
/// slices off tail candidates, which the port clamps to `usize::MAX`
/// (keep all) since `usize` cannot express the negative end.
fn parse_candidate_limit(raw: Option<&str>) -> usize {
    let Some(raw) = raw else {
        return SearchOptions::default().candidate_limit;
    };
    js::parse_int(raw).map_or(0, |v| usize::try_from(v).unwrap_or(usize::MAX))
}

/// The `Structured search:` header block (qmd.ts:2971-2984): count +
/// joined type labels, the resolved intent, one `├─ type: preview`
/// per typed line, then `└─ Searching...`.
fn print_structured_header(
    doc: &qmd::ParsedQueryDocument,
    intent: Option<&str>,
    io: &mut Io<'_>,
    c: Palette,
) {
    let labels = doc
        .searches
        .iter()
        .map(|s| s.kind.as_str())
        .collect::<Vec<_>>()
        .join("+");
    let _ = writeln!(
        io.err,
        "{}Structured search: {} queries ({labels}){}",
        c.dim,
        doc.searches.len(),
        c.reset
    );
    if let Some(intent) = intent {
        let _ = writeln!(io.err, "{}├─ intent: {intent}{}", c.dim, c.reset);
    }
    for s in &doc.searches {
        let mut preview = s.query.replace('\n', " ");
        if qmd::utf16_len(&preview) > 72 {
            preview = format!("{}...", crate::output::utf16_prefix(&preview, 69));
        }
        let _ = writeln!(
            io.err,
            "{}├─ {}: {preview}{}",
            c.dim,
            s.kind.as_str(),
            c.reset
        );
    }
    let _ = writeln!(io.err, "{}└─ Searching...{}", c.dim, c.reset);
}

/// Shared tail of `vectorSearch`/`querySearch`: build the render
/// context and hand rows to `outputResults` with
/// `{...opts, limit: results.length}` (qmd.ts:2947, 3078).
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors upstream's OutputOptions + render-context bundle"
)]
fn output_rows(
    qmd: &qmd::Qmd,
    ctx: &CliContext,
    env: &Environment,
    rows: &[OutputRow],
    query: &str,
    args: &QueryArgs,
    format: Format,
    min_score: f64,
    result_count: usize,
    io: &mut Io<'_>,
    c: Palette,
) {
    let editor_template = qmd.config().map_or_else(
        |_| editor_uri_template(env, &qmd::Config::default()),
        |config| editor_uri_template(env, &config),
    );
    let render = Render {
        qmd,
        index_name: ctx.index_name(),
        cwd: ctx.cwd(),
        c,
        editor_template: &editor_template,
        stdout_tty: io.stdout_tty,
    };
    let opts = OutputOptions {
        format,
        full: args.search.full,
        limit: i64::try_from(result_count).unwrap_or(i64::MAX),
        min_score,
        line_numbers: args.search.line_numbers,
        full_path: args.search.full_path,
        explain: args.explain,
        intent: args.intent.clone(),
    };
    output_results(rows, query, &opts, &render, io);
}

/// `formatMs` (qmd.ts:428-431): `<1000` → `{ms}ms`, else
/// `{(ms/1000).toFixed(1)}s`.
fn format_ms(elapsed: Duration) -> String {
    let ms = elapsed.as_millis();
    if ms < 1000 {
        return format!("{ms}ms");
    }
    #[allow(clippy::cast_precision_loss, reason = "elapsed fits f64")]
    let secs = ms as f64 / 1000.0;
    format!("{}s", js::to_fixed(secs, 1))
}

/// `vectorSearch`'s `onExpand` hook (qmd.ts:2919-2924): the expansion
/// tree, then `Searching {len+1} vector queries...`.
struct VsearchHooks<'a, 'i> {
    /// Borrowed stderr — `SearchHooks` callbacks are `&self`.
    io: RefCell<&'a mut Io<'i>>,
    /// Color palette.
    c: Palette,
}

impl SearchHooks for VsearchHooks<'_, '_> {
    fn on_expand(&self, original: &str, expanded: &[ExpandedQuery], _elapsed: Duration) {
        let mut io = self.io.borrow_mut();
        log_expansion_tree(original, expanded, &mut io, self.c);
        let _ = writeln!(
            io.err,
            "{}Searching {} vector queries...{}",
            self.c.dim,
            expanded.len() + 1,
            self.c.reset
        );
    }
}

/// `querySearch`'s hook bundle (qmd.ts:2990-3050): identical for the
/// structured and hybrid branches except `structured` skips
/// `onStrongSignal`/`onExpandStart`/`onExpand` (upstream simply leaves
/// those hooks unset on the structured call).
struct QueryHooks<'a, 'i> {
    /// Borrowed stderr — `SearchHooks` callbacks are `&self`.
    io: RefCell<&'a mut Io<'i>>,
    /// Color palette.
    c: Palette,
    /// `structuredSearch` vs `hybridQuery` hook set (qmd.ts:2990-3007
    /// vs 3026-3050).
    structured: bool,
}

impl SearchHooks for QueryHooks<'_, '_> {
    fn on_strong_signal(&self, top_score: f64) {
        if self.structured {
            return;
        }
        let mut io = self.io.borrow_mut();
        let _ = writeln!(
            io.err,
            "{}Strong BM25 signal ({}) — skipping expansion{}",
            self.c.dim,
            js::to_fixed(top_score, 2),
            self.c.reset
        );
    }

    fn on_expand_start(&self) {
        if self.structured {
            return;
        }
        let mut io = self.io.borrow_mut();
        let _ = write!(io.err, "{}Expanding query...{}", self.c.dim, self.c.reset);
    }

    fn on_expand(&self, original: &str, expanded: &[ExpandedQuery], elapsed: Duration) {
        if self.structured {
            return;
        }
        let mut io = self.io.borrow_mut();
        let _ = writeln!(
            io.err,
            "{} ({}){}",
            self.c.dim,
            format_ms(elapsed),
            self.c.reset
        );
        log_expansion_tree(original, expanded, &mut io, self.c);
        let _ = writeln!(
            io.err,
            "{}Searching {} queries...{}",
            self.c.dim,
            expanded.len() + 1,
            self.c.reset
        );
    }

    fn on_embed_start(&self, count: usize) {
        let mut io = self.io.borrow_mut();
        let _ = write!(
            io.err,
            "{}Embedding {count} {}...{}",
            self.c.dim,
            if count == 1 { "query" } else { "queries" },
            self.c.reset
        );
    }

    fn on_embed_done(&self, elapsed: Duration) {
        let mut io = self.io.borrow_mut();
        let _ = writeln!(
            io.err,
            "{} ({}){}",
            self.c.dim,
            format_ms(elapsed),
            self.c.reset
        );
    }

    fn on_rerank_start(&self, chunk_count: usize) {
        let mut io = self.io.borrow_mut();
        let _ = write!(
            io.err,
            "{}Reranking {chunk_count} chunks...{}",
            self.c.dim, self.c.reset
        );
        progress::indeterminate(&mut io);
    }

    fn on_rerank_done(&self, elapsed: Duration) {
        let mut io = self.io.borrow_mut();
        progress::clear(&mut io);
        let _ = writeln!(
            io.err,
            "{} ({}){}",
            self.c.dim,
            format_ms(elapsed),
            self.c.reset
        );
    }
}
