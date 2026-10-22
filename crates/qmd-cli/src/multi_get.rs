//! `qmd multi-get` — upstream `case "multi-get"` (qmd.ts:4494-4505) and
//! `multiGet` (qmd.ts:1362-1603).
//!
//! Selection comes from [`qmd::Qmd::select_multi_get`] (the CLI path,
//! not the SDK `findDocuments`); each resolved file's body, title and
//! context are then fetched through `qmd.get`, mirroring upstream's
//! per-file queries. Unlike `get`, upstream multi-get never switches
//! the index — a `?index=` in an item is parsed away by the
//! comma-list resolver but the database stays put.

use std::fmt::Write as _;
use std::path::Path;

use qmd::docid::is_docid;
use qmd::{Environment, GetOptions, MultiGetFile, Qmd, js};

use crate::Failure;
use crate::args::MultiGetArgs;
use crate::context::CliContext;
use crate::output::{
    Format, Io, Palette, escape_csv, render_full_path, warn_unresolved_full_paths,
};
use crate::search::resolve_format;
use crate::slice::js_slice;

/// `qmd multi-get` entry.
///
/// # Errors
/// [`Failure::Reported`] after writing the upstream message;
/// [`Failure::Error`] wraps store/open failures.
pub(crate) fn run(
    env: &Environment,
    ctx: &CliContext,
    args: &MultiGetArgs,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    // --format resolves in parseCLI upstream — before the command's own
    // usage validation (qmd.ts:3176-3194).
    let Some(format) = resolve_format(&args.format, io) else {
        return Err(Failure::Reported(1));
    };
    let Some(pattern) = args.pattern.first() else {
        let _ = writeln!(
            io.err,
            "Usage: qmd multi-get <pattern> [-l <lines>] [--max-bytes <bytes>] [--no-line-numbers] [--full-path] [--format json|csv|md|xml|files]"
        );
        let _ = writeln!(
            io.err,
            "  pattern: glob (e.g., 'journals/2025-05*.md') or comma-separated list"
        );
        return Err(Failure::Reported(1));
    };

    // `i64::MIN` stands in for NaN — `slice(0, NaN)` empties like
    // `slice(0, 0)`; the truncation marker's `len > NaN` comparison is
    // guarded explicitly (qmd.ts:1475-1481).
    let max_lines = args
        .l
        .as_deref()
        .map(|s| js::parse_int(s).unwrap_or(i64::MIN));
    // `--max-bytes` stays an f64: `bodyLength > NaN` is false upstream,
    // so a NaN value disables skipping entirely.
    let max_bytes = args.max_bytes.as_deref().map_or_else(
        || qmd::DEFAULT_MULTI_GET_MAX_BYTES as f64,
        |s| js::parse_int(s).map_or(f64::NAN, |v| v as f64),
    );

    let qmd = ctx.open_store(env)?;
    let selection = qmd.select_multi_get(pattern)?;

    // Per-name failures print immediately, in order (qmd.ts:1379-1384).
    for error in &selection.errors {
        let _ = writeln!(io.err, "{error}");
    }
    let is_comma_separated = pattern.contains(',')
        && !pattern.contains('*')
        && !pattern.contains('?')
        && !pattern.contains('{');
    if is_comma_separated || is_docid(pattern) {
        if is_docid(pattern) && selection.files.is_empty() {
            return Err(Failure::Reported(1));
        }
    } else if selection.files.is_empty() {
        let _ = writeln!(io.err, "No files matched pattern: {pattern}");
        return Err(Failure::Reported(1));
    }

    let mut rows: Vec<Row> = Vec::new();
    for file in &selection.files {
        if let Some(row) = collect_row(&qmd, file, args, max_bytes, max_lines, ctx.cwd())? {
            rows.push(row);
        }
    }

    let unresolved = if args.full_path {
        rows.iter().filter(|r| r.fs_path.is_none()).count()
    } else {
        0
    };
    output(&rows, format, args.full_path, io);
    warn_unresolved_full_paths(
        unresolved,
        rows.len(),
        io,
        Palette::new(Palette::enabled(env, io)),
    );
    Ok(())
}

/// Build one `Row` for a selected file (qmd.ts:1408-1496): `None` when
/// the document vanished between selection and fetch (`if (!doc)
/// continue`).
fn collect_row(
    qmd: &Qmd,
    file: &MultiGetFile,
    args: &MultiGetArgs,
    max_bytes: f64,
    max_lines: Option<i64>,
    cwd: &Path,
) -> Result<Option<Row>, Failure> {
    // --full-path: resolve the on-disk path when it exists
    // (qmd.ts:1437-1443).
    let fs_path = if args.full_path {
        match qmd.resolve_virtual_path(&file.virtual_path) {
            Ok(Some(abs)) if abs.exists() => Some(render_full_path(&abs, cwd)),
            Ok(_) => None,
            Err(e) => return Err(Failure::Error(e)),
        }
    } else {
        None
    };

    // Upstream resolves docid/context before the size check; a
    // `qmd.get` without the body covers both.
    let skip = (file.body_length as f64) > max_bytes;
    let doc = match qmd.get(
        &file.virtual_path,
        GetOptions {
            include_body: !skip,
        },
    ) {
        Ok(doc) => doc,
        Err(qmd::Error::DocumentNotFound { .. } | qmd::Error::ExcludedByIgnore { .. }) => {
            return Ok(None);
        }
        Err(e) => return Err(Failure::Error(e)),
    };
    let docid = (!doc.docid.is_empty()).then_some(doc.docid);
    let basename = || {
        file.display_path
            .rsplit('/')
            .next()
            .filter(|s| !s.is_empty())
            .unwrap_or(&file.display_path)
            .to_owned()
    };

    if skip {
        // `Math.round(len/1024)` in both halves (qmd.ts:1452).
        return Ok(Some(Row {
            display_path: file.display_path.clone(),
            fs_path,
            docid,
            title: basename(),
            body: String::new(),
            context: doc.context,
            skipped: true,
            reason: format!(
                "File too large ({}KB > {}KB). Use 'qmd get {}' to retrieve.",
                js::number_to_string(js::math_round(file.body_length as f64 / 1024.0)),
                js::number_to_string(js::math_round(max_bytes / 1024.0)),
                file.display_path,
            ),
        }));
    }

    let mut body = doc.body.unwrap_or_default();
    // `slice(0, maxLines)` + the truncation marker (qmd.ts:1474-1482).
    if let Some(max) = max_lines {
        let lines: Vec<&str> = body.split('\n').collect();
        let len = i64::try_from(lines.len()).unwrap_or(i64::MAX);
        body = js_slice(&lines, 0, max).join("\n");
        if max != i64::MIN && len > max {
            let more = len.saturating_sub(max);
            let _ = write!(body, "\n\n[... truncated {more} more lines]");
        }
    }
    if !args.no_line_numbers {
        body = qmd::add_line_numbers(&body, 1);
    }
    Ok(Some(Row {
        display_path: file.display_path.clone(),
        fs_path,
        docid,
        title: if doc.title.is_empty() {
            basename()
        } else {
            doc.title
        },
        body,
        context: doc.context,
        skipped: false,
        reason: String::new(),
    }))
}

/// One assembled result row — upstream's `results[]` shape
/// (qmd.ts:1408-1496).
struct Row {
    /// `displayPath` (see [`qmd::MultiGetFile`] for the two quirks).
    display_path: String,
    /// Rendered on-disk path under `--full-path`, when resolvable.
    fs_path: Option<String>,
    /// Short docid without `#`.
    docid: Option<String>,
    /// `doc.title || basename(displayPath)`.
    title: String,
    /// Sliced/numbered body (empty when skipped).
    body: String,
    /// Folder context.
    context: Option<String>,
    /// Skipped for exceeding `--max-bytes`.
    skipped: bool,
    /// Skip reason (upstream `skipReason`).
    reason: String,
}

impl Row {
    /// `identOf` (qmd.ts:1502): the on-disk path when `--full-path`
    /// resolved it, else `displayPath`.
    fn ident(&self, full_path: bool) -> &str {
        if full_path && let Some(fs) = &self.fs_path {
            return fs;
        }
        &self.display_path
    }

    /// `docidOf` (qmd.ts:1503): hidden once `--full-path` resolved.
    fn docid_shown(&self, full_path: bool) -> Option<&str> {
        if full_path && self.fs_path.is_some() {
            return None;
        }
        self.docid.as_deref()
    }
}

/// Dispatch to the per-format emitters (qmd.ts:1506-1600).
fn output(rows: &[Row], format: Format, full_path: bool, io: &mut Io<'_>) {
    match format {
        Format::Json => output_json(rows, full_path, io),
        Format::Csv => output_csv(rows, full_path, io),
        Format::Files => output_files(rows, full_path, io),
        Format::Md => output_md(rows, full_path, io),
        Format::Xml => output_xml(rows, full_path, io),
        Format::Cli => output_cli(rows, full_path, io),
    }
}

/// `JSON.stringify(output, null, 2)` (qmd.ts:1506-1521).
fn output_json(rows: &[Row], full_path: bool, io: &mut Io<'_>) {
    let mut items = Vec::with_capacity(rows.len());
    for row in rows {
        let mut obj = serde_json::Map::new();
        obj.insert(
            "file".to_owned(),
            serde_json::Value::from(row.ident(full_path)),
        );
        if let Some(docid) = row.docid_shown(full_path) {
            obj.insert("docid".to_owned(), format!("#{docid}").into());
        }
        obj.insert("title".to_owned(), row.title.clone().into());
        if let Some(ctx) = &row.context {
            obj.insert("context".to_owned(), ctx.clone().into());
        }
        if row.skipped {
            obj.insert("skipped".to_owned(), true.into());
            obj.insert("reason".to_owned(), row.reason.clone().into());
        } else {
            obj.insert("body".to_owned(), row.body.clone().into());
        }
        items.push(serde_json::Value::Object(obj));
    }
    let _ = writeln!(
        io.out,
        "{}",
        crate::json::stringify_pretty(&serde_json::Value::Array(items))
    );
}

/// Header plus `escapeField`ed rows (qmd.ts:1522-1535); a null context
/// becomes the empty field.
fn output_csv(rows: &[Row], full_path: bool, io: &mut Io<'_>) {
    let _ = writeln!(io.out, "docid,file,title,context,skipped,body");
    for row in rows {
        let docid = row
            .docid_shown(full_path)
            .map_or(String::new(), |d| format!("#{d}"));
        let fields = [
            docid,
            row.ident(full_path).to_owned(),
            row.title.clone(),
            row.context.clone().unwrap_or_default(),
            if row.skipped { "true" } else { "false" }.to_owned(),
            if row.skipped {
                row.reason.clone()
            } else {
                row.body.clone()
            },
        ];
        let _ = writeln!(
            io.out,
            "{}",
            fields
                .iter()
                .map(|f| escape_csv(f))
                .collect::<Vec<_>>()
                .join(",")
        );
    }
}

/// Headerless `#docid,file[,context][,status]` rows (qmd.ts:1536-1545).
fn output_files(rows: &[Row], full_path: bool, io: &mut Io<'_>) {
    for row in rows {
        let id = row
            .docid_shown(full_path)
            .map_or(String::new(), |d| format!("#{d},"));
        let ctx = row.context.as_ref().map_or_else(String::new, |ctx| {
            format!(",\"{}\"", ctx.replace('"', "\"\""))
        });
        let status = if row.skipped { ",[SKIPPED]" } else { "" };
        let _ = writeln!(io.out, "{id}{}{ctx}{status}", row.ident(full_path));
    }
}

/// Markdown document sections (qmd.ts:1546-1563).
fn output_md(rows: &[Row], full_path: bool, io: &mut Io<'_>) {
    for row in rows {
        let _ = writeln!(io.out, "## {}\n", row.ident(full_path));
        if let Some(docid) = row.docid_shown(full_path) {
            let _ = writeln!(io.out, "**docid:** `#{docid}`\n");
        }
        if !row.title.is_empty() && row.title != row.display_path {
            let _ = writeln!(io.out, "**Title:** {}\n", row.title);
        }
        if let Some(ctx) = &row.context {
            let _ = writeln!(io.out, "**Context:** {ctx}\n");
        }
        if row.skipped {
            let _ = writeln!(io.out, "> {}\n", row.reason);
        } else {
            let _ = writeln!(io.out, "```");
            let _ = writeln!(io.out, "{}", row.body);
            let _ = writeln!(io.out, "```\n");
        }
    }
}

/// `<documents>` XML (qmd.ts:1564-1581) with `escapeXml`
/// (formatter.ts:81-88).
fn output_xml(rows: &[Row], full_path: bool, io: &mut Io<'_>) {
    let _ = writeln!(io.out, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
    let _ = writeln!(io.out, "<documents>");
    for row in rows {
        let attr = row
            .docid_shown(full_path)
            .map_or(String::new(), |d| format!(" docid=\"#{d}\""));
        let _ = writeln!(io.out, "  <document{attr}>");
        let _ = writeln!(
            io.out,
            "    <file>{}</file>",
            escape_xml(row.ident(full_path))
        );
        let _ = writeln!(io.out, "    <title>{}</title>", escape_xml(&row.title));
        if let Some(ctx) = &row.context {
            let _ = writeln!(io.out, "    <context>{}</context>", escape_xml(ctx));
        }
        if row.skipped {
            let _ = writeln!(io.out, "    <skipped>true</skipped>");
            let _ = writeln!(io.out, "    <reason>{}</reason>", escape_xml(&row.reason));
        } else {
            let _ = writeln!(io.out, "    <body>{}</body>", escape_xml(&row.body));
        }
        let _ = writeln!(io.out, "  </document>");
    }
    let _ = writeln!(io.out, "</documents>");
}

/// `=` × 60 blocks (qmd.ts:1582-1600).
fn output_cli(rows: &[Row], full_path: bool, io: &mut Io<'_>) {
    let bar = "=".repeat(60);
    for row in rows {
        let id = row
            .docid_shown(full_path)
            .map_or(String::new(), |d| format!("  #{d}"));
        let _ = writeln!(io.out, "\n{bar}");
        let _ = writeln!(io.out, "File: {}{id}", row.ident(full_path));
        let _ = writeln!(io.out, "{bar}\n");
        if row.skipped {
            let _ = writeln!(io.out, "[SKIPPED: {}]", row.reason);
            continue;
        }
        if let Some(ctx) = &row.context {
            let _ = writeln!(io.out, "Folder Context: {ctx}\n---\n");
        }
        let _ = writeln!(io.out, "{}", row.body);
    }
}

/// `escapeXml` (formatter.ts:81-88).
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
