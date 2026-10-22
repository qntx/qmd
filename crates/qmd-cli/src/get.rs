//! `qmd get` — upstream `case "get"` (qmd.ts:4481-4492) and
//! `getDocument` (qmd.ts:1260-1359).
//!
//! The `?index=` switch happens before the store opens (upstream
//! `setIndexName`/`setConfigIndexName`, qmd.ts:1284-1291); suffix and
//! flag line-window parsing runs before [`CliContext::open_store`], so
//! usage errors never touch the database.

use qmd::vpath::{build_virtual_path, is_virtual_path, parse_line_suffix, parse_virtual_path};
use qmd::{Environment, GetOptions, js};

use crate::Failure;
use crate::args::GetArgs;
use crate::context::CliContext;
use crate::output::{Io, Palette, render_full_path, warn_unresolved_full_paths};
use crate::slice::js_slice;

/// `qmd get` entry.
///
/// # Errors
/// [`Failure::Reported`] after writing the upstream message;
/// [`Failure::Error`] wraps store/open failures.
pub(crate) fn run(
    env: &Environment,
    ctx: &mut CliContext,
    args: &GetArgs,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    let Some(filename) = args.file.first() else {
        let _ = writeln!(
            io.err,
            "Usage: qmd get <filepath>[:from[:count]] [--from <line>] [-l <lines>] [--no-line-numbers] [--full-path]"
        );
        return Err(Failure::Reported(1));
    };

    // Upstream parses `:(\d+):(\d+)$` then `:(\d+)$`, and explicit
    // --from/-l flags win over the suffix (qmd.ts:1266-1280).
    let suffix = parse_line_suffix(filename);
    let mut input = suffix.path;
    // `i64::MIN` stands in for NaN: every consumer (clamp, slice) lands
    // on the same result — see `crate::slice`.
    let flag = |raw: &Option<String>| raw.as_deref().map(js_int);
    let from_line = flag(&args.from).or_else(|| {
        suffix
            .from_line
            .map(|v| i64::try_from(v).unwrap_or(i64::MAX))
    });
    let max_lines = flag(&args.l).or_else(|| {
        suffix
            .max_lines
            .map(|v| i64::try_from(v).unwrap_or(i64::MAX))
    });
    let from_clamped = from_line.map(|v| v.max(1));

    // `qmd://…?index=X` switches the active index before opening;
    // rebuilding the virtual path canonicalizes extra slashes
    // (qmd.ts:1284-1291).
    if is_virtual_path(&input)
        && let Some(parsed) = parse_virtual_path(&input)
    {
        if let Some(index) = &parsed.index_name {
            ctx.set_index(env, index);
        }
        input = build_virtual_path(&parsed.collection_name, &parsed.path, None);
    }

    let qmd = ctx.open_store(env)?;
    let doc = match qmd.get(&input, GetOptions { include_body: true }) {
        Ok(doc) => doc,
        Err(err) => return Err(get_error(err, filename, io)),
    };

    let canonical_path = format!("qmd://{}", doc.display_path);
    let mut unresolved = false;
    let header = if args.full_path {
        match qmd.resolve_virtual_path(&canonical_path) {
            Ok(Some(abs)) if abs.exists() => render_full_path(&abs, ctx.cwd()),
            _ => {
                unresolved = true;
                header_with_docid(&canonical_path, &doc.docid)
            }
        }
    } else {
        header_with_docid(&canonical_path, &doc.docid)
    };

    let mut output = doc.body.unwrap_or_default();
    let start_line = from_clamped.unwrap_or(1);

    // `split('\n').slice(start, end).join('\n')` (qmd.ts:1337-1342).
    if from_line.is_some() || max_lines.is_some() {
        let lines: Vec<&str> = output.split('\n').collect();
        let start = start_line.saturating_sub(1);
        let len = i64::try_from(lines.len()).unwrap_or(i64::MAX);
        let end = max_lines.map_or(len, |m| start.saturating_add(m));
        output = js_slice(&lines, start, end).join("\n");
    }
    if !args.no_line_numbers {
        output = qmd::add_line_numbers(&output, usize::try_from(start_line).unwrap_or(usize::MAX));
    }

    let _ = writeln!(io.out, "{header}");
    if let Some(context) = &doc.context {
        let _ = writeln!(io.out, "Folder Context: {context}");
    }
    let _ = writeln!(io.out, "---\n");
    let _ = writeln!(io.out, "{output}");

    if unresolved {
        warn_unresolved_full_paths(1, 1, io, Palette::new(Palette::enabled(env, io)));
    }
    Ok(())
}

/// Upstream header: `qmd://<displayPath>  #<docid>`, or the bare path
/// when no docid was resolved (qmd.ts:1311-1324).
fn header_with_docid(canonical_path: &str, docid: &str) -> String {
    if docid.is_empty() {
        canonical_path.to_owned()
    } else {
        format!("{canonical_path}  #{docid}")
    }
}

/// Upstream `findDocument` error mapping (qmd.ts:1300-1316): excluded
/// paths print the four ignore-rule lines, misses print
/// `Document not found` plus a `Similar files:` block — both are plain
/// `process.exit(1)`, not `exitWithError`.
fn get_error(err: qmd::Error, filename: &str, io: &mut Io<'_>) -> Failure {
    match &err {
        qmd::Error::ExcludedByIgnore {
            collection,
            path,
            rule,
            ..
        } => {
            let _ = writeln!(io.err, "Document is excluded by ignore rule: {filename}");
            let _ = writeln!(io.err, "Collection: {collection}");
            let _ = writeln!(io.err, "Matched path: {path}");
            let _ = writeln!(io.err, "Ignore rule: {rule}");
            Failure::Reported(1)
        }
        qmd::Error::DocumentNotFound { similar_files, .. } => {
            let _ = writeln!(io.err, "Document not found: {filename}");
            if !similar_files.is_empty() {
                let _ = writeln!(io.err, "Similar files:");
                for file in similar_files {
                    let _ = writeln!(io.err, "  {file}");
                }
            }
            Failure::Reported(1)
        }
        _ => Failure::Error(err),
    }
}

/// JS `parseInt(s, 10)`; `i64::MIN` encodes NaN — see [`crate::slice`].
fn js_int(s: &str) -> i64 {
    js::parse_int(s).unwrap_or(i64::MIN)
}
