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

    // resolveCollectionFilter(-c, useDefaults=true) (qmd.ts:2708-2726):
    // no -c → default collections; unknown name → exit 1.
    let collection_names = if args.collection.is_empty() {
        qmd.default_collection_names()?
    } else {
        for name in &args.collection {
            if qmd.collection(name)?.is_none() {
                let _ = writeln!(io.err, "Collection not found: {name}");
                return Err(Failure::Reported(1));
            }
        }
        args.collection.clone()
    };

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

/// `parseCliMetadataFilter` (qmd.ts:2827-2845): JSON syntax errors and
/// filter-AST errors are both fatal with upstream's messages.
///
/// # Errors
/// `Err(())` once the message has been written to stderr.
fn parse_cli_metadata_filter(
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
fn warn_pending_metadata(qmd: &qmd::Qmd, io: &mut Io<'_>, c: Palette) {
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
