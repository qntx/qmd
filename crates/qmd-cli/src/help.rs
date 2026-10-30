//! Top-level help — a subset of upstream `showHelp` (qmd.ts:3635-3745).
//!
//! Only implemented commands are listed (P4-D8: no placeholders) — the
//! `mcp` lines follow the `mcp` feature; the lines kept use upstream's
//! wording and layout.
//! The trailing `Index:` line mirrors upstream's `getDbPath()` tail.

use crate::context::CliContext;
use crate::output::Io;

/// Print top-level help to stdout (upstream `showHelp`).
pub(crate) fn show(io: &mut Io<'_>, ctx: &CliContext) {
    print(io, HEAD);
    #[cfg(feature = "mcp")]
    print(
        io,
        &["  qmd mcp                       - Start the MCP server (stdio transport for AI agents)"],
    );
    // Upstream's `qmd bench` line (qmd.ts:3651) follows `mcp`
    // unconditionally — it exists with or without the mcp feature.
    print(
        io,
        &["  qmd bench <fixture.json>      - Run search quality benchmarks against a fixture file"],
    );
    print(io, MIDDLE);
    #[cfg(feature = "mcp")]
    print(
        io,
        &[
            "  - Run `qmd mcp` to expose the MCP server (stdio) to agents/IDEs.",
            // qmd.ts:3710 verbatim.
            "  - Advanced: `qmd mcp --http ...` and `qmd mcp --http --daemon` are optional for custom transports.",
        ],
    );
    print(io, TAIL);
    let _ = writeln!(io.out, "Index: {}", ctx.db_path().display());
}

/// Write each line to stdout.
fn print(io: &mut Io<'_>, lines: &[&str]) {
    for line in lines {
        let _ = writeln!(io.out, "{line}");
    }
}

/// Banner, usage and primary commands up to `skill`.
const HEAD: &[&str] = &[
    "qmd — Quick Markdown Search",
    "",
    "Usage:",
    "  qmd <command> [options]",
    "",
    "Primary commands:",
    "  qmd query <query>             - Hybrid search with auto expansion + reranking (recommended)",
    // Upstream's `\n` here is inside a JS template literal → a real
    // newline in the printed output (qmd.ts:3638). Keep it literal.
    "  qmd query 'lex:..\nvec:...'   - Structured query document (you provide lex/vec/hyde lines)",
    "  qmd search <query>            - Full-text BM25 keywords (no LLM)",
    "  qmd vsearch <query>           - Vector similarity only",
    "  qmd get <file>[:from[:count]] - Show a document (line-numbered; #docid in header)",
    "  qmd multi-get <pattern>       - Batch fetch via glob or comma-separated list",
    "  qmd skills list/get/path      - List and retrieve bundled runtime skills",
    "  qmd skill show/install        - Show or install the QMD skill",
];

/// Collections through query syntax, ending at the agents heading.
const MIDDLE: &[&str] = &[
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
    "  qmd embed [-f] [-c <name>]    - Generate/refresh vector embeddings",
    "    --max-docs-per-batch <n>    - Cap docs loaded into memory per embedding batch",
    "    --max-batch-mb <n>          - Cap UTF-8 MB loaded into memory per embedding batch",
    "    --timeout <minutes>         - Embed session cap in minutes (0 = no limit; default 30)",
    "  qmd pull [--refresh] [--progress] - Download embedding/generation/rerank models",
    "  qmd cleanup [--dry-run]       - Drop inactive docs/orphans, compact FTS, vacuum",
    "",
    "Query syntax (qmd query):",
    "  QMD queries are either a single expand query (no prefix) or a multi-line",
    "  document where every line is typed with lex:, vec:, or hyde:. This grammar",
    "  matches the docs in docs/SYNTAX.md and is enforced in the CLI.",
    "",
    "  Grammar:",
    "    query          = expand_query | query_document ;",
    "    expand_query   = text | explicit_expand ;",
    "    explicit_expand= \"expand:\" text ;",
    "    query_document = [ intent_line ] { typed_line } ;",
    "    intent_line    = \"intent:\" text newline ;",
    "    typed_line     = type \":\" text newline ;",
    "    type           = \"lex\" | \"vec\" | \"hyde\" ;",
    "    text           = quoted_phrase | plain_text ;",
    "    quoted_phrase  = '\"' { character } '\"' ;",
    "    plain_text     = { character } ;",
    "    newline        = \"\\n\" ;",
    "",
    "  Examples:",
    "    qmd query \"how does auth work\"                # single-line → implicit expand",
    "    qmd query $'lex: CAP theorem\\nvec: consistency'  # typed query document",
    "    qmd query $'lex: \"exact matches\" sports -baseball'  # phrase + negation lex search",
    "    qmd query $'hyde: Hypothetical answer text'       # hyde-only document",
    "",
    "  Constraints:",
    "    - Standalone expand queries cannot mix with typed lines.",
    "    - Query documents allow only lex:, vec:, or hyde: prefixes.",
    "    - Each typed line must be single-line text with balanced quotes.",
    "",
    "AI agents & integrations:",
];

/// Remaining agents lines, then global/search/get/embed/multi-get options.
const TAIL: &[&str] = &[
    "  - Run `qmd skills get qmd --full` for version-matched agent instructions.",
    "  - `qmd skill install` installs the QMD skill into ./.agents/skills/qmd.",
    "  - Use `qmd skill install --global` for ~/.agents/skills/qmd.",
    "  - `qmd --skill` is kept as an alias for `qmd skill show`.",
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
    "  -C, --candidate-limit <n>  - Max candidates to rerank (default 40, lower = faster)",
    "  --no-rerank                - Skip LLM reranking (use RRF scores only, much faster on CPU)",
    "  --no-gpu                   - Force CPU mode for llama.cpp operations (same as QMD_FORCE_CPU=1)",
    "  --line-numbers             - Include line numbers (search; get/multi-get are on by default)",
    "  --no-line-numbers          - Disable line numbers for get/multi-get",
    "  --full-path                - Show on-disk paths instead of qmd:// + docid (get/multi-get/search/query)",
    "                                Paths are ./-prefixed when under $PWD, absolute otherwise",
    "                                Results whose file is gone keep qmd:// + docid and warn on stderr",
    "  --explain                  - Include retrieval score traces (query, CLI/--format json)",
    "  --format <kind>            - Output format: cli (default) | json | csv | md | xml | files",
    "  -c, --collection <name>    - Filter by one or more collections",
    "  --filter <json>            - Metadata filter (recursive JSON AST; search/vsearch/query)",
    "                                e.g. '{\"key\":\"status\",\"operator\":\"eq\",\"value\":\"published\"}'",
    "",
    "Get options:",
    "  -l <num>                   - Maximum lines per file",
    "  --from <line>              - Start at line N (overrides the :from suffix)",
    "",
    "Embed/query options:",
    "  --chunk-strategy <auto|regex> - Chunking mode (default: regex; auto uses AST for code files)",
    "  --timeout <minutes>          - Embed session cap in minutes (0 = no limit; default 30)",
    "",
    "Multi-get options:",
    "  -l <num>                   - Maximum lines per file",
    "  --max-bytes <num>          - Skip files larger than N bytes (default 65536)",
    "  --format <kind>            - Same formats as search",
    "",
];
