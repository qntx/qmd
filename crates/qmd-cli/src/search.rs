//! `qmd search` — upstream `case "search"` (qmd.ts:4753-4761) plus the
//! `search()` body (qmd.ts:2855-2888) and its option preprocessing in
//! `parseCLI` (qmd.ts:3176-3215).
//!
//! Validation order mirrors upstream: `--format` resolution happens in
//! `parseCLI` (before the command runs), the missing-query usage error
//! comes first inside the command case, then `--filter` parsing, and
//! only then does `getDb()` open the store — so usage failures never
//! touch the database.

use qmd::{Environment, LexOptions, MetadataFilter, js, parse_metadata_filter};

use crate::Failure;
use crate::args::{FormatArgs, SearchArgs};
use crate::context::CliContext;
use crate::output::{
    Format, Io, OutputOptions, OutputRow, Palette, Render, editor_uri_template, output_results,
    print_empty_search_results,
};

/// `qmd search` entry.
///
/// # Errors
/// [`Failure::Reported`] after writing the upstream message;
/// [`Failure::Error`] wraps store/search failures.
pub(crate) fn run(
    env: &Environment,
    ctx: &CliContext,
    args: &SearchArgs,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    // Upstream resolves --format in parseCLI, before the command case;
    // its error therefore beats even the missing-query usage error.
    let Some(format) = resolve_format(&args.format, io) else {
        return Err(Failure::Reported(1));
    };

    let query = args.query.join(" ");
    if query.is_empty() {
        let _ = writeln!(io.err, "Usage: qmd search [options] <query>");
        return Err(Failure::Reported(1));
    }
    let Ok(filter) = parse_cli_metadata_filter(args.filter.as_deref(), io) else {
        return Err(Failure::Reported(1));
    };

    let qmd = ctx.open_store(env)?;
    let palette = Palette::new(Palette::enabled(env, io));

    // resolveCollectionFilter(-c, useDefaults=true) (qmd.ts:2708-2726).
    let collection_names = resolve_collection_filter(&qmd, &args.collection, true, io)?;

    if filter.is_some() {
        warn_pending_metadata(&qmd, io, palette);
    }

    let default_limit: i64 = if matches!(format, Format::Json | Format::Files) {
        20
    } else {
        5
    };
    // parseInt(n) || defaultLimit — NaN and 0 both fall back
    // (qmd.ts:3196-3201).
    let limit = if args.all {
        100_000
    } else {
        args.n
            .as_deref()
            .and_then(js::parse_int)
            .filter(|v| *v != 0)
            .unwrap_or(default_limit)
    };
    let min_score = args
        .min_score
        .as_deref()
        .and_then(js::parse_float)
        .unwrap_or(0.0);

    let fetch_limit: usize = if args.all {
        100_000
    } else {
        usize::try_from(50.max(limit.saturating_mul(2))).unwrap_or(usize::MAX)
    };
    let results = qmd.search_lex(
        &query,
        &LexOptions {
            limit: Some(fetch_limit),
            collections: if collection_names.is_empty() {
                None
            } else {
                Some(collection_names)
            },
            filter,
        },
    )?;

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
            // FTS rows never carry an explain trace — upstream
            // `row.explain` is `undefined` here even under `--explain`.
            explain: None,
            // ...nor a best-chunk window (`chunkPos`/`chunkLen`).
            chunk: None,
        })
        .collect();

    let editor_template = qmd.config().map_or_else(
        |_| editor_uri_template(env, &qmd::Config::default()),
        |config| editor_uri_template(env, &config),
    );
    let render = Render {
        qmd: &qmd,
        index_name: ctx.index_name(),
        cwd: ctx.cwd(),
        c: palette,
        editor_template: &editor_template,
        stdout_tty: io.stdout_tty,
    };
    let opts = OutputOptions {
        format,
        full: args.full,
        limit,
        min_score,
        line_numbers: args.line_numbers,
        full_path: args.full_path,
        // `--explain` is parsed for `query` (stage B); on `search` the
        // rows never carry a trace, so upstream's
        // `opts.explain && row.explain` can never fire anyway.
        explain: false,
        intent: None,
    };
    output_results(&rows, &query, &opts, &render, io);
    Ok(())
}

/// Upstream's `--format`/`--csv`/`--md`/`--xml`/`--files`/`--json`
/// resolution (qmd.ts:3176-3194): `--format` wins and is validated
/// case-insensitively after trim; the legacy booleans fall back in
/// csv→md→xml→files→json order.
pub(crate) fn resolve_format(args: &FormatArgs, io: &mut Io<'_>) -> Option<Format> {
    if let Some(raw) = &args.format {
        let normalized = raw.to_lowercase();
        let format = match normalized.trim() {
            "cli" => Format::Cli,
            "json" => Format::Json,
            "csv" => Format::Csv,
            "md" => Format::Md,
            "xml" => Format::Xml,
            "files" => Format::Files,
            _ => {
                let _ = writeln!(io.err, "Unknown --format value: {raw}");
                let _ = writeln!(io.err, "Valid: cli, json, csv, md, xml, files");
                return None;
            }
        };
        return Some(format);
    }
    if args.csv {
        Some(Format::Csv)
    } else if args.md {
        Some(Format::Md)
    } else if args.xml {
        Some(Format::Xml)
    } else if args.files {
        Some(Format::Files)
    } else if args.json {
        Some(Format::Json)
    } else {
        Some(Format::Cli)
    }
}

/// `resolveCollectionFilter` (qmd.ts:2708-2726): no `-c` names →
/// default collections when `use_defaults` (else `[]`), every given
/// name validated against the YAML config — an unknown name prints
/// `Collection not found: {name}` and exits 1.
///
/// # Errors
/// [`Failure::Reported`] after the upstream message; [`qmd::Error`]
/// from config reads.
pub(crate) fn resolve_collection_filter(
    qmd: &qmd::Qmd,
    names: &[String],
    use_defaults: bool,
    io: &mut Io<'_>,
) -> Result<Vec<String>, Failure> {
    if names.is_empty() {
        return Ok(if use_defaults {
            qmd.default_collection_names()?
        } else {
            Vec::new()
        });
    }
    for name in names {
        if qmd.collection(name)?.is_none() {
            let _ = writeln!(io.err, "Collection not found: {name}");
            return Err(Failure::Reported(1));
        }
    }
    Ok(names.to_vec())
}

/// `parseCliMetadataFilter` (qmd.ts:2827-2845): JSON syntax errors and
/// filter-AST errors are both fatal with upstream's messages.
///
/// # Errors
/// `Err(())` once the message has been written to stderr.
pub(crate) fn parse_cli_metadata_filter(
    raw: Option<&str>,
    io: &mut Io<'_>,
) -> Result<Option<MetadataFilter>, ()> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let json: serde_json::Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(err) => {
            let _ = writeln!(io.err, "Invalid --filter JSON: {err}");
            let _ = writeln!(
                io.err,
                "Example: --filter '{{\"key\":\"status\",\"operator\":\"eq\",\"value\":\"published\"}}'"
            );
            return Err(());
        }
    };
    match parse_metadata_filter(&json) {
        Ok(filter) => Ok(Some(filter)),
        Err(err) => {
            let _ = writeln!(io.err, "{err}");
            Err(())
        }
    }
}

/// `warnPendingMetadata` (qmd.ts:2849-2853): filtered search skips
/// documents whose metadata extraction is pending; say so on stderr.
pub(crate) fn warn_pending_metadata(qmd: &qmd::Qmd, io: &mut Io<'_>, c: Palette) {
    let pending = qmd.status().map_or(0, |s| s.pending_metadata);
    if pending == 0 {
        return;
    }
    let _ = writeln!(
        io.err,
        "{}Warning: {pending} document(s) lack current metadata extraction \
         and are excluded from filtered results. Run 'qmd update'.{}",
        c.yellow, c.reset,
    );
}

/// Upstream `checkIndexHealth` (qmd.ts:361-378): stderr-only tips —
/// a needs-embedding share of ≥10% is a yellow `Warning`, otherwise a
/// dim `Tip`; a ≥14-day-stale index gets a dim `Tip`. Upstream calls
/// it in `vectorSearch`/`querySearch` right after
/// `resolveCollectionFilter` (qmd.ts:2912, 2957) — `search()` itself
/// never calls it.
///
/// Pending vectors count against the *declared* model —
/// `resolveEmbedModelForCli()` is upstream's default parameter.
///
/// # Errors
/// [`qmd::Error`] from `Qmd::index_health_for_model` — upstream does
/// not guard `getIndexHealth` either, so index failures surface rather
/// than silently skipping the check.
pub(crate) fn check_index_health(
    env: &Environment,
    qmd: &qmd::Qmd,
    io: &mut Io<'_>,
    c: Palette,
) -> Result<(), Failure> {
    let config = qmd.config()?;
    let declared_model = qmd::llm::resolve_embed_model(config.models.as_ref(), env);
    let health = qmd.index_health_for_model(&declared_model)?;
    if health.needs_embedding > 0 {
        #[allow(clippy::cast_precision_loss, reason = "document counts fit f64")]
        let pct = js::math_round(health.needs_embedding as f64 / health.total_docs as f64 * 100.0);
        if pct >= 10.0 {
            let _ = writeln!(
                io.err,
                "{}Warning: {} documents ({}%) need embeddings. Run 'qmd embed' \
                 for better results.{}",
                c.yellow,
                health.needs_embedding,
                js::number_to_string(pct),
                c.reset,
            );
        } else {
            let _ = writeln!(
                io.err,
                "{}Tip: {} documents need embeddings. Run 'qmd embed' to index \
                 them.{}",
                c.dim, health.needs_embedding, c.reset,
            );
        }
    }
    if let Some(days) = health.days_stale
        && days >= 14
    {
        let _ = writeln!(
            io.err,
            "{}Tip: Index last updated {days} days ago. Run 'qmd update' to \
             refresh.{}",
            c.dim, c.reset,
        );
    }
    Ok(())
}

/// Upstream `logExpansionTree` (qmd.ts:2889-2903): dim stderr tree —
/// `├─ <original>`, then `├─ <type>: <preview>` per expanded query
/// (newlines → spaces, previews over 72 units truncated to 69 +
/// `"..."`), the last line's `├─` swapped for `└─`.
pub(crate) fn log_expansion_tree(
    original: &str,
    expanded: &[qmd::ExpandedQuery],
    io: &mut Io<'_>,
    c: Palette,
) {
    let mut lines = Vec::with_capacity(expanded.len() + 1);
    lines.push(format!("{}├─ {original}{}", c.dim, c.reset));
    for q in expanded {
        let mut preview = q.query.replace('\n', " ");
        if qmd::utf16_len(&preview) > 72 {
            preview = format!("{}...", crate::output::utf16_prefix(&preview, 69));
        }
        lines.push(format!(
            "{}├─ {}: {preview}{}",
            c.dim,
            q.kind.as_str(),
            c.reset
        ));
    }
    // `lines` is never empty — `original` is always pushed.
    if let Some(last) = lines.last_mut() {
        // JS `.replace('├─', '└─')` — first occurrence only.
        *last = last.replacen("├─", "└─", 1);
    }
    for line in lines {
        let _ = writeln!(io.err, "{line}");
    }
}
