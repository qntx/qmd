//! Result rendering — upstream `outputResults` (qmd.ts:2469-2704) and
//! its helpers: `printEmptySearchResults` (2388-2410), `formatScore`
//! (2364-2370), `highlightTerms` (2352-2362), `renderFullPath`
//! (1228-1238), `warnUnresolvedFullPaths` (1248-1258), the OSC 8
//! editor-link helpers (2416-2466), and `escapeCSV`/`escapeXml`
//! (formatter.ts:72-87).
//!
//! Renderers write through [`Io`]; pure string helpers
//! ([`format_score`], [`term_link`], …) are unit-testable.

use std::io::{BufRead, Write};
use std::path::Path;

use qmd::metadata::DocumentMetadata;
use qmd::{Environment, Qmd, extract_snippet, js};

/// Process I/O injected into [`crate::run`] (P4-D1/T2): real stdio in
/// `main`, in-memory buffers in tests.
#[doc(hidden)]
pub struct Io<'a> {
    /// stdout sink.
    pub out: &'a mut dyn Write,
    /// stderr sink.
    pub err: &'a mut dyn Write,
    /// stdin source — only read by the interactive local-config trust
    /// prompt (`confirmOnTty`, P4.2); every other command leaves it alone.
    pub stdin: &'a mut dyn BufRead,
    /// `stdout.isTTY` — drives OSC 8 links and `--full-path` nothing
    /// else.
    pub stdout_tty: bool,
    /// `stderr.isTTY`.
    pub stderr_tty: bool,
    /// `stdin.isTTY`.
    pub stdin_tty: bool,
    /// Local UTC offset captured in `main` before any thread starts
    /// (P1 T5); used by `ls` rendering (P4.1 stage B).
    pub local_offset: time::UtcOffset,
}

impl std::fmt::Debug for Io<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Io")
            .field("stdout_tty", &self.stdout_tty)
            .field("stderr_tty", &self.stderr_tty)
            .field("stdin_tty", &self.stdin_tty)
            .field("local_offset", &self.local_offset)
            .finish_non_exhaustive()
    }
}

/// Upstream output formats (qmd.ts:3178 `VALID_FORMATS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Format {
    /// Human-oriented block output.
    Cli,
    /// `JSON.stringify(rows, null, 2)`.
    Json,
    /// Header + one row per result.
    Csv,
    /// `---` separated document blocks.
    Md,
    /// `<file …>` elements.
    Xml,
    /// `#docid,score,path[,context]` lines.
    Files,
}

/// Upstream `OutputOptions` fields used by `outputResults`
/// (qmd.ts:2332-2349).
#[derive(Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "the flag set mirrors upstream's OutputOptions 1:1"
)]
pub(crate) struct OutputOptions {
    /// Resolved `--format`/legacy-flag output format.
    pub(crate) format: Format,
    /// `--full` — body instead of snippet.
    pub(crate) full: bool,
    /// `-n`/`--all` limit (JS `slice(0, n)` semantics — may be negative).
    pub(crate) limit: i64,
    /// `--min-score` (`parseFloat` result).
    pub(crate) min_score: f64,
    /// `--line-numbers`.
    pub(crate) line_numbers: bool,
    /// `--full-path`.
    pub(crate) full_path: bool,
    /// `--explain` — retrieval score traces (rows still need
    /// [`OutputRow::explain`]; only `query` results carry one).
    pub(crate) explain: bool,
    /// `--intent` — forwarded to `extractSnippet` for window picking
    /// (upstream `opts.intent`, qmd.ts:2344).
    pub(crate) intent: Option<String>,
}

/// One scored row for `outputResults` — a [`qmd::SearchResult`]
/// flattened to what the renderers read.
#[derive(Debug)]
pub(crate) struct OutputRow {
    /// `collection/path` shown in identifiers (upstream `displayPath`).
    pub(crate) display_path: String,
    /// Document title.
    pub(crate) title: String,
    /// Full body.
    pub(crate) body: String,
    /// Score in `[0, 1)`.
    pub(crate) score: f64,
    /// Folder context (upstream `getContextForFile`).
    pub(crate) context: Option<String>,
    /// Short docid (no `#`).
    pub(crate) docid: String,
    /// Extracted frontmatter metadata (empty → field omitted).
    pub(crate) metadata: DocumentMetadata,
    /// Upstream `row.explain` — RRF/rerank trace; only hybrid `query`
    /// rows carry one (boxed: the row stays small for the common case).
    pub(crate) explain: Option<Box<qmd::Explain>>,
    /// Upstream `row.chunkPos`/`row.chunkLen` — the best-chunk window
    /// `query` results carry (UTF-16 offsets); `None` for
    /// `search`/`vsearch` rows.
    pub(crate) chunk: Option<qmd::ChunkSpan>,
}

/// Upstream's `c` palette (qmd.ts:249-257): ANSI codes when color is
/// enabled, empty strings otherwise so renderers stay unconditional.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Palette {
    /// `\x1b[0m`.
    pub(crate) reset: &'static str,
    /// `\x1b[2m`.
    pub(crate) dim: &'static str,
    /// `\x1b[1m`.
    pub(crate) bold: &'static str,
    /// `\x1b[36m`.
    pub(crate) cyan: &'static str,
    /// `\x1b[33m`.
    pub(crate) yellow: &'static str,
    /// `\x1b[32m`.
    pub(crate) green: &'static str,
    /// `\x1b[35m` — kept for palette parity; used by later stages.
    #[allow(dead_code, reason = "stage B/C renderers use it")]
    pub(crate) magenta: &'static str,
    /// `\x1b[34m` — kept for palette parity; used by later stages.
    #[allow(dead_code, reason = "stage B/C renderers use it")]
    pub(crate) blue: &'static str,
}

impl Palette {
    /// `useColor = !NO_COLOR && stdout.isTTY` (qmd.ts:248).
    pub(crate) const fn new(enabled: bool) -> Self {
        if enabled {
            Self {
                reset: "\x1b[0m",
                dim: "\x1b[2m",
                bold: "\x1b[1m",
                cyan: "\x1b[36m",
                yellow: "\x1b[33m",
                green: "\x1b[32m",
                magenta: "\x1b[35m",
                blue: "\x1b[34m",
            }
        } else {
            Self {
                reset: "",
                dim: "",
                bold: "",
                cyan: "",
                yellow: "",
                green: "",
                magenta: "",
                blue: "",
            }
        }
    }

    /// Color on iff `stdout_tty && env.no_color.is_none()` (qmd.ts:248).
    pub(crate) const fn enabled(env: &Environment, io: &Io<'_>) -> bool {
        io.stdout_tty && env.no_color.is_none()
    }
}

/// Shared render context: the open store, active index name, working
/// directory, color and editor-link settings.
pub(crate) struct Render<'a> {
    /// Open store (virtual-path resolution, contexts).
    pub(crate) qmd: &'a Qmd,
    /// Active index name for `?index=` on `qmd://` URIs.
    pub(crate) index_name: &'a str,
    /// `process.cwd()` — `renderFullPath`'s anchor.
    pub(crate) cwd: &'a Path,
    /// ANSI palette (`c` in upstream).
    pub(crate) c: Palette,
    /// Editor URI template for OSC 8 links.
    pub(crate) editor_template: &'a str,
    /// `stdout.isTTY`.
    pub(crate) stdout_tty: bool,
}

/// `qmd://` URI for a `displayPath` — upstream `toQmdPath` inside
/// `outputResults` (qmd.ts:2485-2495): `?index=` is added only for
/// non-default indexes.
pub(crate) fn to_qmd_path(display_path: &str, index_name: &str) -> String {
    let mut parts = display_path.splitn(2, '/');
    let collection = parts.next().unwrap_or_default();
    parts.next().map_or_else(
        || format!("qmd://{display_path}"),
        |path| {
            qmd::vpath::build_virtual_path(
                collection,
                path,
                (index_name != "index").then_some(index_name),
            )
        },
    )
}

/// Upstream `renderFullPath` (qmd.ts:1228-1238): `./`-relative under
/// cwd, absolute realpath otherwise; `None` never happens here — the
/// caller falls back to `qmd://`.
pub(crate) fn render_full_path(absolute: &Path, cwd: &Path) -> String {
    let real = qmd::paths::canonicalize(absolute).unwrap_or_else(|_| absolute.to_path_buf());
    let cwd_real = qmd::paths::canonicalize(cwd).unwrap_or_else(|_| cwd.to_path_buf());
    if real == cwd_real {
        return "./".to_owned();
    }
    // Upstream tests `real.startsWith(cwdReal + "/")` — for `cwd == "/"`
    // that requires a `//` prefix, which never occurs, so root-cwd
    // always renders absolute. `strip_prefix("/")` would wrongly match.
    if cwd_real != Path::new("/")
        && let Ok(rel) = real.strip_prefix(&cwd_real)
    {
        return format!("./{}", rel.display());
    }
    real.display().to_string()
}

/// Upstream `warnUnresolvedFullPaths` (qmd.ts:1248-1258).
pub(crate) fn warn_unresolved_full_paths(
    unresolved: usize,
    total: usize,
    io: &mut Io<'_>,
    c: Palette,
) {
    if unresolved == 0 {
        return;
    }
    let subject = if total == 1 {
        "the file".to_owned()
    } else {
        format!("{unresolved} of {total} results")
    };
    let _ = writeln!(
        io.err,
        "{}warning:{} --full-path could not resolve {subject} on disk \
         (moved or deleted since indexing); showing qmd:// + docid instead. \
         Run 'qmd update' to refresh the index.",
        c.yellow, c.reset,
    );
}

/// Upstream `formatScore` (qmd.ts:2364-2370): `(score*100).toFixed(0)`
/// padded to 3 columns, colored by threshold when color is on.
pub(crate) fn format_score(score: f64, c: Palette) -> String {
    let pct = js::to_fixed(score * 100.0, 0);
    let padded = format!("{pct:>3}");
    if score >= 0.7 {
        return format!("{}{padded}%{}", c.green, c.reset);
    }
    if score >= 0.4 {
        return format!("{}{padded}%{}", c.yellow, c.reset);
    }
    format!("{}{padded}%{}", c.dim, c.reset)
}

/// Upstream `highlightTerms` (qmd.ts:2352-2362): terms of ≥3 UTF-16
/// units get `{yellow}{bold}…{reset}`, all occurrences, case-insensitive.
/// Identity when color is off (upstream returns the text untouched).
pub(crate) fn highlight_terms(text: &str, query: &str, c: Palette) -> String {
    if c.yellow.is_empty() {
        return text.to_owned();
    }
    let terms: Vec<String> = query
        .to_lowercase()
        .split(js::is_js_ws)
        .filter(|t| qmd::utf16_len(t) >= 3)
        .map(ToOwned::to_owned)
        .collect();
    let mut result = text.to_owned();
    // Every term applies to the already-highlighted text (no dedup) —
    // a duplicated query term nests another highlight, as upstream.
    for term in terms {
        result = replace_case_insensitive(&result, &term, c);
    }
    result
}

/// Case-insensitive global replace for [`highlight_terms`].
fn replace_case_insensitive(text: &str, term: &str, c: Palette) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some((start, end)) = find_ci(rest, term) {
        out.push_str(&rest[..start]);
        out.push_str(c.yellow);
        out.push_str(c.bold);
        out.push_str(&rest[start..end]);
        out.push_str(c.reset);
        rest = &rest[end..];
    }
    out.push_str(rest);
    out
}

/// Byte range of the first case-insensitive `term` match in `text`.
fn find_ci(text: &str, term: &str) -> Option<(usize, usize)> {
    let term_chars = term.chars().count();
    for (start, _) in text.char_indices() {
        let mut end = start;
        let mut count = 0;
        while count < term_chars {
            let ch = text.get(end..)?.chars().next()?;
            end += ch.len_utf8();
            count += 1;
        }
        if text[start..end].eq_ignore_ascii_case(term) || text[start..end].to_lowercase() == term {
            return Some((start, end));
        }
    }
    None
}

/// `escapeCSV` (formatter.ts:72-79).
pub(crate) fn escape_csv(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        return format!("\"{}\"", value.replace('"', "\"\""));
    }
    value.to_owned()
}

/// `DEFAULT_EDITOR_URI_TEMPLATE` (qmd.ts:2415).
pub(crate) const DEFAULT_EDITOR_URI_TEMPLATE: &str = "vscode://file/{path}:{line}:{col}";

/// Upstream `getEditorUriTemplate` (qmd.ts:2424-2446):
/// `QMD_EDITOR_URI` > config `editor_uri`/`editor_uri_template`/
/// `editorUri`/`editor-uri` (first non-empty spelling, then trimmed) >
/// `vscode://file/{path}:{line}:{col}`.
#[must_use]
pub(crate) fn editor_uri_template(env: &Environment, config: &qmd::Config) -> String {
    if let Some(t) = env
        .qmd_editor_uri
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return t.to_owned();
    }
    let configured = [
        config.editor_uri.as_deref(),
        config.editor_uri_template.as_deref(),
        config.editor_uri_camel.as_deref(),
        config.editor_uri_kebab.as_deref(),
    ]
    .into_iter()
    .flatten()
    .find(|s| !s.is_empty())
    .map(str::trim);
    match configured {
        Some(t) if !t.is_empty() => t.to_owned(),
        _ => DEFAULT_EDITOR_URI_TEMPLATE.to_owned(),
    }
}

/// Upstream `encodePathForEditorUri` (qmd.ts:2417-2422): `encodeURI`
/// plus `?`→`%3F` and `#`→`%23`.
fn encode_path_for_editor_uri(path: &str) -> String {
    // JS encodeURI keeps RFC-2396 unreserved+reserved marks; upstream
    // then force-encodes `?` and `#`.
    const KEEP: &[u8] = b"-_.!~*'();,/:@&=+$";
    let mut out = String::with_capacity(path.len());
    for b in path.bytes() {
        if b.is_ascii_alphanumeric() || KEEP.contains(&b) {
            out.push(char::from(b));
        } else {
            use std::fmt::Write;
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

/// Upstream `buildEditorUri` (qmd.ts:2448-2457): substitutes `{path}`,
/// `{line}`, `{col}` and `{column}`; non-positive or non-finite
/// line/col clamp to 1 and are floored.
#[must_use]
#[allow(
    clippy::literal_string_with_formatting_args,
    reason = "the {line}/{col}/{column} literals are upstream template \
              placeholders, not format strings"
)]
pub(crate) fn build_editor_uri(template: &str, absolute_path: &str, line: f64, col: f64) -> String {
    let safe_line = if line.is_finite() && line > 0.0 {
        line.floor()
    } else {
        1.0
    };
    let safe_col = if col.is_finite() && col > 0.0 {
        col.floor()
    } else {
        1.0
    };
    let encoded = encode_path_for_editor_uri(absolute_path);
    template
        .replace("{path}", &encoded)
        .replace("{line}", &js::number_to_string(safe_line))
        .replace("{col}", &js::number_to_string(safe_col))
        .replace("{column}", &js::number_to_string(safe_col))
}

/// Upstream `termLink` (qmd.ts:2459-2464): OSC 8 hyperlink only on a
/// TTY.
pub(crate) fn term_link(text: &str, url: &str, is_tty: bool) -> String {
    if !is_tty {
        return text.to_owned();
    }
    format!("\x1b]8;;{url}\x07{text}\x1b]8;;\x07")
}

/// `printEmptySearchResults` (qmd.ts:2388-2410). `min_score` is the
/// "no results above threshold" variant.
pub(crate) fn print_empty_search_results(format: Format, min_score: bool, io: &mut Io<'_>) {
    match format {
        Format::Json => {
            let _ = writeln!(io.out, "[]");
        }
        Format::Csv => {
            let _ = writeln!(io.out, "docid,score,file,title,context,line,snippet");
        }
        Format::Xml => {
            let _ = writeln!(io.out, "<results></results>");
        }
        Format::Md | Format::Files => {}
        Format::Cli => {
            if min_score {
                let _ = writeln!(io.out, "No results found above minimum score threshold.");
            } else {
                let _ = writeln!(io.out, "No results found.");
            }
        }
    }
}

/// Resolved per-row identifier (`--full-path` swap), upstream
/// `resolutions` map (qmd.ts:2500-2520).
struct Resolution {
    /// `qmd://` URI or on-disk path.
    ident: String,
    /// Whether an on-disk path was found (only under `--full-path`).
    resolved: bool,
}

/// `outputResults` (qmd.ts:2469-2704): min-score filter, `slice(0,
/// limit)`, then per-format rendering. Rows must already carry the
/// fetch-limit ordering.
pub(crate) fn output_results(
    rows: &[OutputRow],
    query: &str,
    opts: &OutputOptions,
    render: &Render<'_>,
    io: &mut Io<'_>,
) {
    // JS `filter(score >= min).slice(0, limit)`; a negative limit drops
    // that many trailing rows.
    let kept: Vec<&OutputRow> = rows.iter().filter(|r| r.score >= opts.min_score).collect();
    let end = if opts.limit < 0 {
        i64::try_from(kept.len()).unwrap_or(i64::MAX) + opts.limit
    } else {
        opts.limit
    };
    let end = usize::try_from(end.max(0))
        .unwrap_or(usize::MAX)
        .min(kept.len());
    let filtered = kept.get(..end).unwrap_or(&[]);

    if filtered.is_empty() {
        print_empty_search_results(opts.format, true, io);
        return;
    }

    // Resolve identifiers up front so unresolved rows are counted
    // before anything prints (qmd.ts:2500-2520).
    let resolutions: Vec<Resolution> = filtered
        .iter()
        .map(|row| {
            let uri = to_qmd_path(&row.display_path, render.index_name);
            if opts.full_path
                && let Ok(Some(absolute)) = render.qmd.resolve_virtual_path(&uri)
                && absolute.exists()
            {
                return Resolution {
                    ident: render_full_path(&absolute, render.cwd),
                    resolved: true,
                };
            }
            Resolution {
                ident: uri,
                resolved: false,
            }
        })
        .collect();
    let unresolved = if opts.full_path {
        resolutions.iter().filter(|r| !r.resolved).count()
    } else {
        0
    };

    match opts.format {
        Format::Json => output_json(filtered, &resolutions, opts, query, io),
        Format::Files => output_files(filtered, &resolutions, opts, io),
        Format::Cli => output_cli(filtered, &resolutions, opts, query, render, io),
        Format::Md => output_md(filtered, &resolutions, opts, query, io),
        Format::Xml => output_xml(filtered, &resolutions, opts, query, io),
        Format::Csv => output_csv(filtered, &resolutions, opts, query, io),
    }

    warn_unresolved_full_paths(unresolved, filtered.len(), io, render.c);
}

/// The docid stays visible while it is still the row's identifier
/// (qmd.ts:2532): `--full-path` drops it only for resolved files.
const fn show_docid(opts: &OutputOptions, res: &Resolution) -> bool {
    !opts.full_path || !res.resolved
}

/// `files` format (qmd.ts:2558-2564): `#docid,score,path[,context]`;
/// the `,context` tail is `"…"`-quoted and the docid column disappears
/// under `--full-path`.
fn output_files(
    rows: &[&OutputRow],
    resolutions: &[Resolution],
    opts: &OutputOptions,
    io: &mut Io<'_>,
) {
    for (row, res) in rows.iter().zip(resolutions) {
        let ctx = row.context.as_ref().map_or_else(String::new, |ctx| {
            format!(",\"{}\"", ctx.replace('"', "\"\""))
        });
        if opts.full_path {
            let _ = writeln!(io.out, "{},{}{ctx}", js::to_fixed(row.score, 2), res.ident);
        } else {
            let _ = writeln!(
                io.out,
                "#{},{},{}{ctx}",
                row.docid,
                js::to_fixed(row.score, 2),
                res.ident
            );
        }
    }
}

/// Markdown blocks (qmd.ts:2623-2648).
fn output_md(
    rows: &[&OutputRow],
    resolutions: &[Resolution],
    opts: &OutputOptions,
    query: &str,
    io: &mut Io<'_>,
) {
    for (row, res) in rows.iter().zip(resolutions) {
        let heading = if row.title.is_empty() {
            res.ident.clone()
        } else {
            row.title.clone()
        };
        let mut content = if opts.full {
            row.body.clone()
        } else {
            extract_snippet(&row.body, query, 500, row.chunk, opts.intent.as_deref()).snippet
        };
        if opts.line_numbers {
            content = qmd::add_line_numbers(&content, 1);
        }
        let docid_line = if show_docid(opts, res) {
            format!("**docid:** `#{docid}`\n", docid = row.docid)
        } else {
            String::new()
        };
        let context_line = row
            .context
            .as_ref()
            .map_or_else(String::new, |c| format!("**context:** {c}\n"));
        let _ = writeln!(
            io.out,
            "---\n# {heading}\n**file:** `{file}`\n{docid_line}{context_line}\n{content}\n",
            file = res.ident
        );
    }
}

/// XML `<file>` elements (qmd.ts:2653-2701).
fn output_xml(
    rows: &[&OutputRow],
    resolutions: &[Resolution],
    opts: &OutputOptions,
    query: &str,
    io: &mut Io<'_>,
) {
    for (row, res) in rows.iter().zip(resolutions) {
        let title_attr = if row.title.is_empty() {
            String::new()
        } else {
            format!(" title=\"{}\"", row.title.replace('"', "&quot;"))
        };
        let context_attr = row.context.as_ref().map_or_else(String::new, |ctx| {
            format!(" context=\"{}\"", ctx.replace('"', "&quot;"))
        });
        let mut content = if opts.full {
            row.body.clone()
        } else {
            extract_snippet(&row.body, query, 500, row.chunk, opts.intent.as_deref()).snippet
        };
        if opts.line_numbers {
            content = qmd::add_line_numbers(&content, 1);
        }
        let docid_attr = if show_docid(opts, res) {
            format!(" docid=\"#{}\"", row.docid)
        } else {
            String::new()
        };
        let _ = writeln!(
            io.out,
            "<file{docid_attr} name=\"{}\"{title_attr}{context_attr}>\n{content}\n</file>\n",
            res.ident
        );
    }
}

/// CSV rows (qmd.ts:2664-2688): header plus `escapeCSV` fields and a
/// `toFixed(4)` score.
fn output_csv(
    rows: &[&OutputRow],
    resolutions: &[Resolution],
    opts: &OutputOptions,
    query: &str,
    io: &mut Io<'_>,
) {
    let _ = writeln!(io.out, "docid,score,file,title,context,line,snippet");
    for (row, res) in rows.iter().zip(resolutions) {
        let snippet = extract_snippet(&row.body, query, 500, row.chunk, opts.intent.as_deref());
        let line_start = if opts.full { 1 } else { snippet.line };
        let mut content = if opts.full {
            row.body.clone()
        } else {
            snippet.snippet.clone()
        };
        if opts.line_numbers {
            content = qmd::add_line_numbers(&content, line_start);
        }
        let docid_field = if show_docid(opts, res) {
            format!("#{}", row.docid)
        } else {
            String::new()
        };
        let _ = writeln!(
            io.out,
            "{docid},{score},{file},{title},{context},{line},{snippet}",
            docid = docid_field,
            score = js::to_fixed(row.score, 4),
            file = escape_csv(&res.ident),
            title = escape_csv(&row.title),
            context = escape_csv(row.context.as_deref().unwrap_or("")),
            line = snippet.line,
            snippet = escape_csv(&content),
        );
    }
}

/// JSON output — upstream's `{docid, score, file, line, title,
/// context?, metadata?, body?, snippet?}` field order (qmd.ts:2527-2554).
fn output_json(
    rows: &[&OutputRow],
    resolutions: &[Resolution],
    opts: &OutputOptions,
    query: &str,
    io: &mut Io<'_>,
) {
    let mut items = Vec::with_capacity(rows.len());
    for (row, res) in rows.iter().zip(resolutions) {
        let snippet = extract_snippet(&row.body, query, 300, row.chunk, opts.intent.as_deref());
        let mut obj = serde_json::Map::new();
        if !row.docid.is_empty() && show_docid(opts, res) {
            obj.insert("docid".to_owned(), format!("#{}", row.docid).into());
        }
        let score = js::math_round(row.score * 100.0) / 100.0;
        obj.insert("score".to_owned(), serde_json::Value::from(score));
        obj.insert("file".to_owned(), res.ident.clone().into());
        obj.insert(
            "line".to_owned(),
            serde_json::Value::from(u64::try_from(snippet.line).unwrap_or(u64::MAX)),
        );
        obj.insert("title".to_owned(), row.title.clone().into());
        if let Some(ctx) = &row.context {
            obj.insert("context".to_owned(), ctx.clone().into());
        }
        if !row.metadata.is_empty()
            && let Ok(meta) = serde_json::from_str::<serde_json::Value>(&row.metadata.to_json())
        {
            obj.insert("metadata".to_owned(), meta);
        }
        if opts.full {
            let body = if opts.line_numbers {
                qmd::add_line_numbers(&row.body, 1)
            } else {
                row.body.clone()
            };
            obj.insert("body".to_owned(), body.into());
        } else {
            let snippet_text = if opts.line_numbers {
                qmd::add_line_numbers(&snippet.snippet, 1)
            } else {
                snippet.snippet
            };
            obj.insert("snippet".to_owned(), snippet_text.into());
        }
        // Upstream's `...(opts.explain && row.explain && {explain})` —
        // the explain object is the row's last key (qmd.ts:2551).
        if opts.explain
            && let Some(explain) = &row.explain
        {
            obj.insert("explain".to_owned(), explain_json(explain));
        }
        items.push(serde_json::Value::Object(obj));
    }
    let _ = writeln!(
        io.out,
        "{}",
        crate::json::stringify_pretty(&serde_json::Value::Array(items))
    );
}

/// `row.explain` as a JSON value — upstream `HybridQueryExplain`
/// field names/order (store.ts:5486-5499, `RRFContributionTrace` at
/// 2459-2468: `rrfContribution`, camelCase throughout). Non-finite
/// numbers become `null` like `JSON.stringify`.
fn explain_json(explain: &qmd::Explain) -> serde_json::Value {
    use serde_json::{Map, Value};
    let contributions = explain
        .rrf
        .contributions
        .iter()
        .map(|c| {
            Value::Object(Map::from_iter([
                ("listIndex".to_owned(), Value::from(c.list_index)),
                ("source".to_owned(), c.source.as_str().into()),
                ("queryType".to_owned(), c.query_type.as_str().into()),
                ("query".to_owned(), c.query.clone().into()),
                ("rank".to_owned(), Value::from(c.rank)),
                ("weight".to_owned(), Value::from(c.weight)),
                ("backendScore".to_owned(), Value::from(c.backend_score)),
                ("rrfContribution".to_owned(), Value::from(c.contribution)),
            ]))
        })
        .collect::<Vec<_>>();
    Value::Object(Map::from_iter([
        (
            "ftsScores".to_owned(),
            Value::Array(explain.fts_scores.iter().map(|v| Value::from(*v)).collect()),
        ),
        (
            "vectorScores".to_owned(),
            Value::Array(
                explain
                    .vector_scores
                    .iter()
                    .map(|v| Value::from(*v))
                    .collect(),
            ),
        ),
        (
            "rrf".to_owned(),
            Value::Object(Map::from_iter([
                ("rank".to_owned(), Value::from(explain.rrf.rank)),
                (
                    "positionScore".to_owned(),
                    Value::from(explain.rrf.position_score),
                ),
                ("weight".to_owned(), Value::from(explain.rrf.weight)),
                ("baseScore".to_owned(), Value::from(explain.rrf.base_score)),
                (
                    "topRankBonus".to_owned(),
                    Value::from(explain.rrf.top_rank_bonus),
                ),
                (
                    "totalScore".to_owned(),
                    Value::from(explain.rrf.total_score),
                ),
                ("contributions".to_owned(), Value::Array(contributions)),
            ])),
        ),
        ("rerankScore".to_owned(), Value::from(explain.rerank_score)),
        (
            "blendedScore".to_owned(),
            Value::from(explain.blended_score),
        ),
    ]))
}

/// CLI block format (qmd.ts:2566-2650).
fn output_cli(
    rows: &[&OutputRow],
    resolutions: &[Resolution],
    opts: &OutputOptions,
    query: &str,
    render: &Render<'_>,
    io: &mut Io<'_>,
) {
    let c = render.c;
    for (i, (row, res)) in rows.iter().zip(resolutions).enumerate() {
        let snippet = extract_snippet(&row.body, query, 500, row.chunk, opts.intent.as_deref());
        let line = snippet.line;
        let visible = &res.ident;

        // `:line` only when a query term occurs in the snippet body
        // (header line excluded), upstream hasMatch (qmd.ts:2594-2597).
        let snippet_body = snippet
            .snippet
            .split('\n')
            .skip(1)
            .collect::<Vec<_>>()
            .join("\n")
            .to_lowercase();
        let has_match = query
            .to_lowercase()
            .split(js::is_js_ws)
            .any(|t| !t.is_empty() && snippet_body.contains(t));
        let line_info = if has_match {
            format!(":{line}")
        } else {
            String::new()
        };
        let docid_str = if !row.docid.is_empty() && show_docid(opts, res) {
            format!(" {}#{}{}", c.dim, row.docid, c.reset)
        } else {
            String::new()
        };

        let uri = to_qmd_path(&row.display_path, render.index_name);
        let absolute = render.qmd.resolve_virtual_path(&uri).ok().flatten();
        // `stdout.isTTY && absolutePath && parsed?.path` (qmd.ts:2597)
        // — a virtual path with an empty path part never links.
        let parsed_path =
            qmd::vpath::parse_virtual_path(&uri).is_some_and(|parsed| !parsed.path.is_empty());
        if render.stdout_tty
            && let Some(abs) = &absolute
            && parsed_path
        {
            let link_line = if has_match { line } else { 1 };
            let target = build_editor_uri(
                render.editor_template,
                &abs.to_string_lossy(),
                f64::from(u32::try_from(link_line).unwrap_or(u32::MAX)),
                1.0,
            );
            let clickable = term_link(&format!("{visible}{line_info}"), &target, true);
            let _ = writeln!(io.out, "{}{clickable}{}{docid_str}", c.cyan, c.reset);
        } else {
            let _ = writeln!(
                io.out,
                "{}{visible}{}{line_info}{}{docid_str}",
                c.cyan, c.dim, c.reset
            );
        }

        if !row.title.is_empty() {
            let _ = writeln!(io.out, "{}Title: {}{}", c.bold, row.title, c.reset);
        }
        if let Some(ctx) = &row.context {
            let _ = writeln!(io.out, "{}Context: {ctx}{}", c.dim, c.reset);
        }
        let _ = writeln!(
            io.out,
            "Score: {}{}{}",
            c.bold,
            format_score(row.score, c),
            c.reset
        );
        if opts.explain
            && let Some(explain) = &row.explain
        {
            output_explain(explain, c, io);
        }
        let _ = writeln!(io.out);

        let content = if opts.full {
            row.body.clone()
        } else {
            snippet.snippet
        };
        let content = if opts.line_numbers {
            qmd::add_line_numbers(&content, if opts.full { 1 } else { line })
        } else {
            content
        };
        let _ = writeln!(io.out, "{}", highlight_terms(&content, query, c));
        if i + 1 < rows.len() {
            let _ = io.out.write_all(b"\n\n");
        }
    }
}

/// The `--explain` block — upstream `outputResults` (qmd.ts:2619-2640):
/// `Explain:`/`RRF:`/`Blend:` plus the top-3 contributions by
/// `rrfContribution`, all numbers via `formatExplainNumber`
/// (`toFixed(4)`); `Math.round` percentages via [`js::math_round`].
fn output_explain(explain: &qmd::Explain, c: Palette, io: &mut Io<'_>) {
    // `formatExplainNumber` — toFixed(4).
    let f4 = |v: f64| js::to_fixed(v, 4);
    let fts = if explain.fts_scores.is_empty() {
        "none".to_owned()
    } else {
        explain
            .fts_scores
            .iter()
            .map(|s| f4(*s))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let vec = if explain.vector_scores.is_empty() {
        "none".to_owned()
    } else {
        explain
            .vector_scores
            .iter()
            .map(|s| f4(*s))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut contribs: Vec<&qmd::RrfContribution> = explain.rrf.contributions.iter().collect();
    // `.slice().sort(desc rrfContribution).slice(0, 3)` — stable sort.
    contribs.sort_by(|a, b| b.contribution.total_cmp(&a.contribution));
    let contrib_summary = contribs
        .iter()
        .take(3)
        .map(|contrib| {
            format!(
                "{}/{}#{}:{}",
                contrib.source.as_str(),
                contrib.query_type.as_str(),
                contrib.rank,
                f4(contrib.contribution)
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    let _ = writeln!(
        io.out,
        "{}Explain: fts=[{fts}] vec=[{vec}]{}",
        c.dim, c.reset
    );
    let _ = writeln!(
        io.out,
        "{}  RRF: total={} base={} bonus={} rank={}{}",
        c.dim,
        f4(explain.rrf.total_score),
        f4(explain.rrf.base_score),
        f4(explain.rrf.top_rank_bonus),
        explain.rrf.rank,
        c.reset
    );
    let _ = writeln!(
        io.out,
        "{}  Blend: {}%*{} + {}%*{} = {}{}",
        c.dim,
        js::number_to_string(js::math_round(explain.rrf.weight * 100.0)),
        f4(explain.rrf.position_score),
        js::number_to_string(js::math_round((1.0 - explain.rrf.weight) * 100.0)),
        f4(explain.rerank_score),
        f4(explain.blended_score),
        c.reset
    );
    if !contrib_summary.is_empty() {
        let _ = writeln!(
            io.out,
            "{}  Top RRF contributions: {contrib_summary}{}",
            c.dim, c.reset
        );
    }
}

// =============================================================================
// Shared CLI formatters (P4.3) — upstream helpers used by the stage-B
// commands (`query`/`vsearch`/`embed`/`pull`/`doctor`).
// =============================================================================

/// Upstream `formatMs` (qmd.ts:428-431): `{ms}ms` under a second, else
/// `toFixed(1)` seconds.
#[allow(dead_code, reason = "used by stage-B command timings")]
pub(crate) fn format_ms(ms: u64) -> String {
    if ms < 1000 {
        return format!("{ms}ms");
    }
    #[allow(clippy::cast_precision_loss, reason = "ms fits f64")]
    let seconds = ms as f64 / 1000.0;
    format!("{}s", js::to_fixed(seconds, 1))
}

/// Upstream `formatCount` (qmd.ts:3752-3754):
/// `n.toLocaleString("en-US")` — thousands-grouped digits.
#[allow(dead_code, reason = "used by stage-B doctor/embed output")]
pub(crate) fn format_count(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, d) in digits.bytes().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(char::from(d));
    }
    out
}

/// Upstream `formatBytes` (qmd.ts:433-438): `B` under 1 KiB, then
/// `KB`/`MB`/`GB` with one decimal — `toFixed(1)` rounds half-up, so
/// [`js::to_fixed`] rather than ties-to-even `{:.1}`.
pub(crate) fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;
    #[allow(clippy::cast_precision_loss, reason = "sizes fit f64")]
    let b = bytes as f64;
    if bytes < KB {
        format!("{bytes} B")
    } else if bytes < MB {
        format!("{} KB", js::to_fixed(b / 1024.0, 1))
    } else if bytes < GB {
        format!("{} MB", js::to_fixed(b / (1024.0 * 1024.0), 1))
    } else {
        format!("{} GB", js::to_fixed(b / (1024.0 * 1024.0 * 1024.0), 1))
    }
}

/// Upstream `renderProgressBar` (qmd.ts:2109-2114): `width` cells,
/// `Math.round(percent / 100 * width)` filled `█`, the rest `░`.
#[allow(dead_code, reason = "used by stage-B embed/pull progress")]
pub(crate) fn render_progress_bar(percent: f64, width: usize) -> String {
    // Upstream callers keep percent in [0, 100]; `repeat` on a
    // negative count throws upstream, so clamp rather than replicate
    // the RangeError.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to [0, width] on the next line"
    )]
    let filled = (js::math_round(percent / 100.0 * width as f64) as i64)
        .clamp(0, i64::try_from(width).unwrap_or(i64::MAX)) as usize;
    let mut bar = "█".repeat(filled);
    bar.push_str(&"░".repeat(width - filled));
    bar
}

/// Upstream `shortModelName` (qmd.ts:3756-3760): `hf:` URIs shorten to
/// the last path segment (empty tail falls back to the whole URI);
/// other strings truncate at 56 units to 53 + `"..."`.
#[allow(dead_code, reason = "used by stage-B doctor output")]
pub(crate) fn short_model_name(model: &str) -> String {
    if model.starts_with("hf:") {
        if let Some(last) = model.rsplit('/').next().filter(|s| !s.is_empty()) {
            return last.to_owned();
        }
        return model.to_owned();
    }
    if qmd::utf16_len(model) > 56 {
        return format!("{}...", utf16_prefix(model, 53));
    }
    model.to_owned()
}

/// Upstream `summarizeDeviceNames` (qmd.ts:492-499): duplicates render
/// `<count>× <name>`, first-seen order (JS `Map` semantics).
#[allow(dead_code, reason = "used by stage-B doctor device output")]
pub(crate) fn summarize_device_names(names: &[String]) -> String {
    let mut counts: Vec<(&str, usize)> = Vec::new();
    for name in names {
        if let Some((_, n)) = counts.iter_mut().find(|(n, _)| *n == name.as_str()) {
            *n += 1;
        } else {
            counts.push((name.as_str(), 1));
        }
    }
    counts
        .iter()
        .map(|(name, n)| {
            if *n > 1 {
                format!("{n}× {name}")
            } else {
                (*name).to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// First `units` UTF-16 code units of `text` — JS `slice(0, units)`.
/// A split surrogate pair ends before its char (Rust cannot represent
/// a lone surrogate).
#[allow(dead_code, reason = "shared UTF-16 slicing for stage-B helpers")]
pub(crate) fn utf16_prefix(text: &str, units: usize) -> String {
    let mut out = String::new();
    let mut used = 0usize;
    for ch in text.chars() {
        let w = ch.len_utf16();
        if used + w > units {
            break;
        }
        used += w;
        out.push(ch);
    }
    out
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::missing_assert_message,
    clippy::shadow_unrelated,
    clippy::field_reassign_with_default,
    reason = "idiomatic test-code patterns"
)]
mod tests {
    use super::*;

    fn io<'a>(
        out: &'a mut Vec<u8>,
        err: &'a mut Vec<u8>,
        stdout_tty: bool,
        stdin: &'a mut dyn BufRead,
    ) -> Io<'a> {
        Io {
            out,
            err,
            stdin,
            stdout_tty,
            stderr_tty: false,
            stdin_tty: false,
            local_offset: time::UtcOffset::UTC,
        }
    }

    /// Empty stdin — the prompt path answers "no" on EOF.
    fn empty_stdin() -> &'static mut dyn BufRead {
        Box::leak(Box::new(std::io::empty()))
    }

    // Ported from upstream `test/cli.test.ts` "editor URI templates".

    #[test]
    fn build_editor_uri_expands_path_line_col() {
        let uri = build_editor_uri(
            "vscode://file/{path}:{line}:{col}",
            "/tmp/my notes/readme.md",
            42.0,
            1.0,
        );
        assert_eq!(uri, "vscode://file//tmp/my%20notes/readme.md:42:1");
    }

    #[test]
    fn build_editor_uri_supports_column_alias() {
        let uri = build_editor_uri(
            "cursor://file/{path}:{line}:{column}",
            "/tmp/docs/api.md",
            7.0,
            3.0,
        );
        assert_eq!(uri, "cursor://file//tmp/docs/api.md:7:3");
    }

    #[test]
    fn build_editor_uri_encodes_query_and_fragment() {
        // Upstream adds `?`→%3F and `#`→%23 on top of encodeURI.
        let uri = build_editor_uri("{path}", "/tmp/a?b#c/d.md", 1.0, 1.0);
        assert_eq!(uri, "/tmp/a%3Fb%23c/d.md");
    }

    #[test]
    fn build_editor_uri_clamps_nonpositive_line_col() {
        let uri = build_editor_uri("{line}:{col}", "p", 0.0, -3.0);
        assert_eq!(uri, "1:1");
    }

    #[test]
    fn term_link_plain_when_not_tty() {
        let linked = term_link(
            "docs/api.md:12",
            "vscode://file//tmp/docs/api.md:12:1",
            false,
        );
        assert_eq!(linked, "docs/api.md:12");
    }

    #[test]
    fn term_link_osc8_when_tty() {
        let linked = term_link(
            "docs/api.md:12",
            "vscode://file//tmp/docs/api.md:12:1",
            true,
        );
        assert_eq!(
            linked,
            "\x1b]8;;vscode://file//tmp/docs/api.md:12:1\x07docs/api.md:12\x1b]8;;\x07"
        );
    }

    #[test]
    fn editor_uri_template_env_beats_config_and_default() {
        let mut env = Environment::default();
        env.qmd_editor_uri = Some("  zed://file/{path}:{line}  ".to_owned());
        let mut config = qmd::Config::default();
        config.editor_uri = Some("cursor://file/{path}".to_owned());
        assert_eq!(
            editor_uri_template(&env, &config),
            "zed://file/{path}:{line}"
        );

        env.qmd_editor_uri = None;
        assert_eq!(editor_uri_template(&env, &config), "cursor://file/{path}");

        config.editor_uri = None;
        config.editor_uri_kebab = Some("windsurf://file/{path}".to_owned());
        assert_eq!(editor_uri_template(&env, &config), "windsurf://file/{path}");

        config.editor_uri_kebab = None;
        assert_eq!(
            editor_uri_template(&env, &config),
            DEFAULT_EDITOR_URI_TEMPLATE
        );
    }

    #[test]
    fn format_score_thresholds_and_color() {
        let color = Palette::new(true);
        let plain = Palette::new(false);
        assert_eq!(format_score(0.85, color), "\x1b[32m 85%\x1b[0m");
        assert_eq!(format_score(0.5, color), "\x1b[33m 50%\x1b[0m");
        assert_eq!(format_score(0.1, color), "\x1b[2m 10%\x1b[0m");
        assert_eq!(format_score(0.85, plain), " 85%");
        // JS toFixed half-up: 0.745 -> " 75%".
        assert_eq!(format_score(0.745, plain), " 75%");
    }

    #[test]
    fn escape_csv_only_when_needed() {
        assert_eq!(escape_csv("plain"), "plain");
        assert_eq!(escape_csv("a,b"), "\"a,b\"");
        assert_eq!(escape_csv("a\"b"), "\"a\"\"b\"");
        assert_eq!(escape_csv("a\nb"), "\"a\nb\"");
    }

    #[test]
    fn to_qmd_path_adds_index_only_when_named() {
        assert_eq!(
            to_qmd_path("docs/readme.md", "index"),
            "qmd://docs/readme.md"
        );
        assert_eq!(
            to_qmd_path("docs/readme.md", "work"),
            "qmd://docs/readme.md?index=work"
        );
    }

    #[test]
    fn render_full_path_relative_under_cwd() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().canonicalize().unwrap();
        let file = cwd.join("docs/a.md");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, "x").unwrap();
        assert_eq!(render_full_path(&file, &cwd), "./docs/a.md");
        let outside = tempfile::tempdir().unwrap();
        let other = outside.path().join("b.md");
        std::fs::write(&other, "x").unwrap();
        let rendered = render_full_path(&other, &cwd);
        assert!(rendered.starts_with('/'), "{rendered}");
        assert!(!rendered.starts_with("./"), "{rendered}");
    }

    #[test]
    fn highlight_terms_identity_without_color() {
        let plain = Palette::new(false);
        assert_eq!(
            highlight_terms("hello world", "hello", plain),
            "hello world"
        );
        let color = Palette::new(true);
        assert_eq!(
            highlight_terms("say Hello again", "hello", color),
            "say \x1b[33m\x1b[1mHello\x1b[0m again"
        );
        // Terms shorter than 3 UTF-16 units are not highlighted.
        assert_eq!(highlight_terms("hi hi", "hi", color), "hi hi");
        // Regex metacharacters are literal (upstream escapes them).
        assert_eq!(
            highlight_terms("a.c axc", "a.c", color),
            "\x1b[33m\x1b[1ma.c\x1b[0m axc"
        );
        // A duplicated term nests a second highlight around the first
        // (upstream applies terms in order, no dedup).
        assert_eq!(
            highlight_terms("index here", "index index", color),
            "\x1b[33m\x1b[1m\x1b[33m\x1b[1mindex\x1b[0m\x1b[0m here"
        );
    }

    #[test]
    fn empty_results_match_oracle() {
        let cases = [
            (Format::Cli, false, "No results found.\n"),
            (
                Format::Cli,
                true,
                "No results found above minimum score threshold.\n",
            ),
            (Format::Json, false, "[]\n"),
            (
                Format::Csv,
                false,
                "docid,score,file,title,context,line,snippet\n",
            ),
            (Format::Xml, false, "<results></results>\n"),
            (Format::Md, false, ""),
            (Format::Files, false, ""),
        ];
        for (format, min_score, want) in cases {
            let mut out = Vec::new();
            let mut err = Vec::new();
            print_empty_search_results(
                format,
                min_score,
                &mut io(&mut out, &mut err, false, empty_stdin()),
            );
            assert_eq!(String::from_utf8(out).unwrap(), want, "{format:?}");
        }
    }

    #[test]
    fn warn_unresolved_full_paths_wording() {
        let mut out = Vec::new();
        let mut err = Vec::new();
        let plain = Palette::new(false);
        warn_unresolved_full_paths(
            1,
            1,
            &mut io(&mut out, &mut err, false, empty_stdin()),
            plain,
        );
        assert_eq!(
            String::from_utf8(err).unwrap(),
            "warning: --full-path could not resolve the file on disk \
             (moved or deleted since indexing); showing qmd:// + docid instead. \
             Run 'qmd update' to refresh the index.\n"
        );

        let mut out = Vec::new();
        let mut err = Vec::new();
        warn_unresolved_full_paths(
            2,
            5,
            &mut io(&mut out, &mut err, false, empty_stdin()),
            plain,
        );
        assert!(String::from_utf8(err).unwrap().contains("2 of 5 results"));

        let mut out = Vec::new();
        let mut err = Vec::new();
        warn_unresolved_full_paths(
            0,
            3,
            &mut io(&mut out, &mut err, false, empty_stdin()),
            plain,
        );
        assert!(err.is_empty());
    }
}
