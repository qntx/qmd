//! Top-level help — a subset of upstream `showHelp` (qmd.ts:3635-3745).
//!
//! Only commands implemented on this branch are listed (P4-D8: no
//! placeholders); the lines kept use upstream's wording and layout.
//! The trailing `Index:` line mirrors upstream's `getDbPath()` tail.

use crate::context::CliContext;
use crate::output::Io;

/// Print top-level help to stdout (upstream `showHelp`).
pub(crate) fn show(io: &mut Io<'_>, ctx: &CliContext) {
    for line in [
        "qmd — Quick Markdown Search",
        "",
        "Usage:",
        "  qmd <command> [options]",
        "",
        "Primary commands:",
        "  qmd search <query>            - Full-text BM25 keywords (no LLM)",
        "  qmd get <file>[:from[:count]] - Show a document (line-numbered; #docid in header)",
        "  qmd multi-get <pattern>       - Batch fetch via glob or comma-separated list",
        "",
        "Collections & context:",
        "  qmd collection add/list/remove/rename/show   - Manage indexed folders",
        "  qmd context add/list/rm                      - Attach human-written summaries",
        "  qmd ls [collection[/path]]                   - Inspect indexed files",
        "",
        "Maintenance:",
        "  qmd init                      - Create a project-local .qmd index",
        "  qmd status                    - View index + collection health",
        // Upstream keeps `[--pull]` in the help text even though the
        // flag was dropped from the update dispatch (qmd.ts:3661).
        "  qmd update [--pull]           - Re-index collections (optionally git pull first)",
        "  qmd trust [list|revoke]       - Approve a checked-in .qmd config's hooks/paths/models",
        "  qmd cleanup [--dry-run]       - Drop inactive docs/orphans, compact FTS, vacuum",
        "",
        "Global options:",
        "  --index <name>             - Use a named index (default: index)",
        "  QMD_EDITOR_URI             - Editor link template for clickable TTY search output",
        "",
        "Search options:",
        "  -n <num>                   - Max results (default 5, or 20 for --format files|json)",
        "  --all                      - Return all matches (pair with --min-score)",
        "  --min-score <num>          - Minimum similarity score",
        "  --full                     - Output full document instead of snippet",
        "  --no-gpu                   - Force CPU mode for llama.cpp operations (same as QMD_FORCE_CPU=1)",
        "  --line-numbers             - Include line numbers (search; get/multi-get are on by default)",
        "  --no-line-numbers          - Disable line numbers for get/multi-get",
        "  --full-path                - Show on-disk paths instead of qmd:// + docid (get/multi-get/search/query)",
        "                                Paths are ./-prefixed when under $PWD, absolute otherwise",
        "                                Results whose file is gone keep qmd:// + docid and warn on stderr",
        "  --format <kind>            - Output format: cli (default) | json | csv | md | xml | files",
        "  -c, --collection <name>    - Filter by one or more collections",
        "  --filter <json>            - Metadata filter (recursive JSON AST; search/vsearch/query)",
        "                                e.g. '{\"key\":\"status\",\"operator\":\"eq\",\"value\":\"published\"}'",
        "",
        "Get options:",
        "  -l <num>                   - Maximum lines per file",
        "  --from <line>              - Start at line N (overrides the :from suffix)",
        "",
        "Multi-get options:",
        "  -l <num>                   - Maximum lines per file",
        "  --max-bytes <num>          - Skip files larger than N bytes (default 65536)",
        "  --format <kind>            - Same formats as search",
        "",
    ] {
        let _ = writeln!(io.out, "{line}");
    }
    let _ = writeln!(io.out, "Index: {}", ctx.db_path().display());
}
