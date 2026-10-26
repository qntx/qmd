//! Command-line parsing — upstream `parseCLI` (qmd.ts:3084-3225).
//!
//! Upstream uses `util.parseArgs` with `strict: false`, so every option
//! is accepted everywhere and typos are silently dropped. P4-D2 decided
//! strict clap parsing instead: unknown options error (exit 1, not
//! clap's default 2), and options belong to the subcommand that
//! consumes them — so `qmd -n 5 search x`, which upstream accepts, is
//! rejected here. `--index`, `--no-gpu`, `-h/--help` and `-v/--version`
//! are true globals, matching upstream's global-option bucket.

use clap::{Parser, Subcommand};

/// Parsed command line, reduced to what the dispatcher consumes.
#[derive(Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "global upstream boolean flags mirror the parseArgs bucket 1:1"
)]
pub(crate) struct ParsedCli {
    /// Global `--index <name>` (normalized by [`crate::context`]).
    pub(crate) index: Option<String>,
    /// Global `--no-gpu`.
    pub(crate) no_gpu: bool,
    /// Global `-h`/`--help`.
    pub(crate) help: bool,
    /// Global `-v`/`--version`.
    pub(crate) version: bool,
    /// Global `--skill` — `qmd skill show` alias (upstream
    /// `cli.values.skill`, qmd.ts:3097 + 4379-4382).
    pub(crate) skill: bool,
    /// Subcommand, or `None` (bare `qmd` → help + exit 1).
    pub(crate) command: Option<Command>,
}

/// A parsed subcommand.
#[derive(Debug)]
pub(crate) enum Command {
    /// `search` with its options.
    Search(SearchArgs),
    /// `get` with its options.
    Get(GetArgs),
    /// `multi-get` with its options.
    MultiGet(MultiGetArgs),
    /// `ls` with an optional path argument.
    Ls(LsArgs),
    /// `collection` — state-mutating collection management (P4.2).
    Collection(CollectionArgs),
    /// `context` — collection/global context management (P4.2).
    Context(ContextArgs),
    /// `init` — create a project-local `.qmd` index.
    Init,
    /// `status` — index/collection health report.
    Status,
    /// `update` — re-index collections (runs `update:` hooks).
    Update,
    /// `cleanup` — maintenance vacuum/compaction (`--dry-run` previews).
    Cleanup(CleanupArgs),
    /// `trust` — approve/inspect/revoke local-config gating.
    Trust(TrustArgs),
    /// `embed` — generate/refresh vector embeddings.
    Embed(EmbedArgs),
    /// `pull` — download the active embed/generate/rerank models.
    Pull(PullArgs),
    /// `vsearch` (alias `vector-search`) — vector similarity search.
    Vsearch(QueryArgs),
    /// `query` (alias `deep-search`) — hybrid search with expansion +
    /// reranking; also parses structured query documents.
    Query(QueryArgs),
    /// `doctor` — config/index/model/device diagnostics.
    Doctor,
    /// `skills` — list/print bundled runtime skills (P4.4).
    Skills(SkillsArgs),
    /// `skill` — print or install the QMD skill (P4.4).
    Skill(SkillArgs),
    /// `mcp` — stdio MCP server (P5.2).
    #[cfg(feature = "mcp")]
    Mcp(McpArgs),
    /// Any unregistered first positional (`help`, `version`, …) —
    /// upstream's `default:` branch.
    Unknown(UnknownArgs),
}

/// The words of an unrecognized command; `command()` is the first one.
#[derive(Debug)]
pub(crate) struct UnknownArgs(pub(crate) Vec<String>);

/// `qmd mcp [args]` — `stop`/other positionals are rejected
/// (`Unknown subcommand:`) until P5.4 adds the daemon lifecycle.
#[derive(Debug, clap::Args)]
#[cfg(feature = "mcp")]
pub(crate) struct McpArgs {
    /// Positional words after `mcp` (upstream `cli.args[0]`).
    pub(crate) args: Vec<String>,
    /// `--http` — serve streamable HTTP instead of stdio
    /// (qmd.ts:4839 `cli.values.http`).
    #[arg(long)]
    pub(crate) http: bool,
    /// `--port <N>` — HTTP port; resolved with `Number(v) || 8181`
    /// semantics (qmd.ts:4840), so kept as a string here.
    #[arg(long)]
    pub(crate) port: Option<String>,
    /// `--host <H>` — bind host; falsy falls back to `QMD_HOST` then
    /// `localhost` (qmd.ts:4841-4844).
    #[arg(long)]
    pub(crate) host: Option<String>,
}

impl UnknownArgs {
    /// Command name as typed (first positional).
    pub(crate) fn command(&self) -> &str {
        self.0.first().map_or("", String::as_str)
    }
}

/// Options for `qmd search` (upstream `OutputOptions` subset it reads,
/// qmd.ts:2332-2348 and 3176-3221).
#[derive(Debug, Clone, clap::Args)]
#[command(allow_negative_numbers = true)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "the flag set mirrors upstream's search options 1:1"
)]
pub(crate) struct SearchArgs {
    /// Query terms — joined with `" "` into `cli.query` (qmd.ts:3220).
    #[arg(num_args = 0..)]
    pub(crate) query: Vec<String>,
    /// `-n <num>` — string parsed with `parseInt(…) || default`
    /// semantics (qmd.ts:3196-3201), so `-n 0` and `-n abc` fall back.
    #[arg(short = 'n')]
    pub(crate) n: Option<String>,
    /// `--min-score <num>` — `parseFloat(…) || 0` (qmd.ts:3201).
    #[arg(long = "min-score")]
    pub(crate) min_score: Option<String>,
    /// `--all` — return every match (limit 100000).
    #[arg(long)]
    pub(crate) all: bool,
    /// `--full` — full document body instead of a snippet.
    #[arg(long)]
    pub(crate) full: bool,
    /// `--full-path` — on-disk paths instead of `qmd://` URIs.
    #[arg(long = "full-path")]
    pub(crate) full_path: bool,
    /// `--line-numbers` — number lines in body/snippet output.
    #[arg(long = "line-numbers")]
    pub(crate) line_numbers: bool,
    /// `--format` and the five legacy aliases — upstream resolves them
    /// in `parseCLI` for every command (qmd.ts:3176-3194).
    #[command(flatten)]
    pub(crate) format: FormatArgs,
    /// `-c`/`--collection <name>`, repeatable (qmd.ts:3120).
    #[arg(short = 'c', long = "collection")]
    pub(crate) collection: Vec<String>,
    /// `--filter <json>` — metadata filter AST (qmd.ts:3122).
    #[arg(long)]
    pub(crate) filter: Option<String>,
    /// `--intent <text>` — rides `{...opts}` into `outputResults`'s
    /// snippet extractor on all three retrieval commands
    /// (qmd.ts:2436, 3126); `query` additionally merges it with the
    /// document's `intent:` line for the search pipeline
    /// (qmd.ts:2964).
    #[arg(long)]
    pub(crate) intent: Option<String>,
}

/// The format option bundle — shared by `search` and `multi-get`
/// (upstream's `cli.opts.format`, resolved in `parseCLI`).
#[derive(Debug, Clone, Default, clap::Args)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "upstream keeps five legacy format flags alongside --format"
)]
pub(crate) struct FormatArgs {
    /// `--format <kind>` (qmd.ts:3176-3194).
    #[arg(long)]
    pub(crate) format: Option<String>,
    /// Legacy `--json` alias (qmd.ts:3114).
    #[arg(long)]
    pub(crate) json: bool,
    /// Legacy `--csv` alias.
    #[arg(long)]
    pub(crate) csv: bool,
    /// Legacy `--md` alias.
    #[arg(long)]
    pub(crate) md: bool,
    /// Legacy `--xml` alias.
    #[arg(long)]
    pub(crate) xml: bool,
    /// Legacy `--files` alias.
    #[arg(long)]
    pub(crate) files: bool,
}

/// Options for `qmd vsearch`/`qmd query` (upstream `case "vsearch"`/
/// `case "query"`, qmd.ts:4762-4786). Upstream resolves every flag in
/// the shared `parseArgs` bucket, so both commands accept the full
/// option surface even where a flag is unused — `vsearch` ignores
/// `--candidate-limit`/`--no-rerank`/`--explain`/`--chunk-strategy`
/// like upstream's `vectorSearch` does.
#[derive(Debug, Clone, clap::Args)]
#[command(allow_negative_numbers = true)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "the flag set mirrors upstream's shared query options 1:1"
)]
pub(crate) struct QueryArgs {
    /// Query terms plus the shared search-option bundle
    /// (`-n`/`--min-score`/`--all`/`--full`/`--full-path`/
    /// `--line-numbers`/`--format`/`-c`/`--filter`/`--intent`).
    #[command(flatten)]
    pub(crate) search: SearchArgs,
    /// `-C`/`--candidate-limit <n>` — `parseInt` semantics; `0`/NaN
    /// select zero rerank candidates upstream (qmd.ts:3209).
    #[arg(short = 'C', long = "candidate-limit")]
    pub(crate) candidate_limit: Option<String>,
    /// `--no-rerank` — RRF scores only, no reranker load.
    #[arg(long = "no-rerank")]
    pub(crate) no_rerank: bool,
    /// `--explain` — per-row retrieval score traces.
    #[arg(long)]
    pub(crate) explain: bool,
    /// `--chunk-strategy <auto|regex>` — best-chunk selection mode.
    #[arg(long = "chunk-strategy")]
    pub(crate) chunk_strategy: Option<String>,
}

/// Options for `qmd embed` (upstream `case "embed"`, qmd.ts:4705-4728).
/// Numeric options stay raw strings: validation mirrors upstream's
/// `Number()`-based `parseEmbedBatchOption`/`parseEmbedTimeoutOption`
/// errors (qmd.ts:2116-2141), including its `exitWithError` text.
#[derive(Debug, Clone, clap::Args)]
#[command(allow_negative_numbers = true)]
pub(crate) struct EmbedArgs {
    /// `-f`/`--force` — clear all vectors and re-embed.
    #[arg(short = 'f', long)]
    pub(crate) force: bool,
    /// `--max-docs-per-batch <n>` — `Number()` → integer ≥ 1.
    #[arg(long = "max-docs-per-batch")]
    pub(crate) max_docs_per_batch: Option<String>,
    /// `--max-batch-mb <n>` — `Number()` → integer ≥ 1 (mebibytes).
    #[arg(long = "max-batch-mb")]
    pub(crate) max_batch_mb: Option<String>,
    /// `--timeout <minutes>` — `Number()` → ≥ 0 (`0` = no limit).
    #[arg(long)]
    pub(crate) timeout: Option<String>,
    /// `--chunk-strategy <auto|regex>` (qmd.ts:2125-2130).
    #[arg(long = "chunk-strategy")]
    pub(crate) chunk_strategy: Option<String>,
    /// `-c`/`--collection <name>` — every value is validated against
    /// the config but only the first scopes the embed (qmd.ts:4716-4717).
    #[arg(short = 'c', long = "collection")]
    pub(crate) collection: Vec<String>,
}

/// Options for `qmd pull` (upstream `case "pull"`, qmd.ts:4730-4751).
#[derive(Debug, Clone, clap::Args)]
#[command(allow_negative_numbers = true)]
pub(crate) struct PullArgs {
    /// `--refresh` — revalidate/re-download cached models.
    #[arg(long)]
    pub(crate) refresh: bool,
    /// `--progress` — show a download progress indicator on stderr.
    #[arg(long)]
    pub(crate) progress: bool,
}

/// Options for `qmd get` (upstream `case "get"`, qmd.ts:4481-4492).
#[derive(Debug, Clone, clap::Args)]
#[command(allow_negative_numbers = true)]
pub(crate) struct GetArgs {
    /// `<filepath>[:from[:count]]` — upstream reads only `cli.args[0]`,
    /// so extra positionals are accepted and ignored.
    #[arg(num_args = 0..)]
    pub(crate) file: Vec<String>,
    /// `--from <line>` — `parseInt` semantics; overrides the `:from`
    /// suffix (qmd.ts:1266-1282).
    #[arg(long)]
    pub(crate) from: Option<String>,
    /// `-l <lines>` — `parseInt` semantics; overrides the `:count`
    /// suffix.
    #[arg(short = 'l')]
    pub(crate) l: Option<String>,
    /// `--no-line-numbers` — line numbers are on by default.
    #[arg(long = "no-line-numbers")]
    pub(crate) no_line_numbers: bool,
    /// `--full-path` — show the on-disk path instead of the `qmd://`
    /// header when the file still exists.
    #[arg(long = "full-path")]
    pub(crate) full_path: bool,
}

/// Options for `qmd multi-get` (upstream `case "multi-get"`,
/// qmd.ts:4494-4505).
#[derive(Debug, Clone, clap::Args)]
#[command(allow_negative_numbers = true)]
pub(crate) struct MultiGetArgs {
    /// Glob or comma-separated list — upstream reads only
    /// `cli.args[0]`, so extra positionals are ignored.
    #[arg(num_args = 0..)]
    pub(crate) pattern: Vec<String>,
    /// `-l <lines>` — `parseInt` semantics.
    #[arg(short = 'l')]
    pub(crate) l: Option<String>,
    /// `--max-bytes <num>` — `parseInt` semantics, default 65536.
    #[arg(long = "max-bytes")]
    pub(crate) max_bytes: Option<String>,
    /// `--no-line-numbers` — line numbers are on by default.
    #[arg(long = "no-line-numbers")]
    pub(crate) no_line_numbers: bool,
    /// `--full-path`.
    #[arg(long = "full-path")]
    pub(crate) full_path: bool,
    /// `--format` and legacy aliases.
    #[command(flatten)]
    pub(crate) format: FormatArgs,
}

/// Options for `qmd ls` (upstream `case "ls"`, qmd.ts:4508-4511).
#[derive(Debug, Clone, clap::Args)]
#[command(allow_negative_numbers = true)]
pub(crate) struct LsArgs {
    /// Optional `[collection[/path]]` — upstream reads only
    /// `cli.args[0]`, so extra positionals are ignored.
    #[arg(num_args = 0..)]
    pub(crate) path: Vec<String>,
}

/// Arguments for `qmd collection` (upstream `case "collection"`,
/// qmd.ts:4513-4679). Upstream dispatches on `cli.args[0]` with a
/// `switch`, so unknown subcommands fall through to a custom
/// `Unknown subcommand:` error rather than a clap diagnostic — the
/// positionals stay a flat `Vec` for the same reason.
#[derive(Debug, Clone, clap::Args)]
#[command(allow_negative_numbers = true)]
pub(crate) struct CollectionArgs {
    /// Subcommand plus its positionals — upstream `cli.args`
    /// (`args[0]` subcommand, `args[1..]` names/paths/hook words).
    #[arg(num_args = 0..)]
    pub(crate) args: Vec<String>,
    /// `--name <name>` for `add` (upstream `cli.values.name`).
    #[arg(long)]
    pub(crate) name: Option<String>,
    /// `--mask <glob>` for `add` (upstream `collectionGlobFromCli`,
    /// qmd.ts:1824-1827).
    #[arg(long)]
    pub(crate) mask: Option<String>,
    /// `--glob <glob>` — `--mask` alias (upstream #536).
    #[arg(long)]
    pub(crate) glob: Option<String>,
}

/// Arguments for `qmd context` (upstream `case "context"`,
/// qmd.ts:4404-4478). `cli.args[0]` is the subcommand; `add` splits the
/// rest into `[path] "text"`.
#[derive(Debug, Clone, clap::Args)]
#[command(allow_negative_numbers = true)]
pub(crate) struct ContextArgs {
    /// Subcommand plus positionals — upstream `cli.args`.
    #[arg(num_args = 0..)]
    pub(crate) args: Vec<String>,
}

/// Options for `qmd cleanup` (upstream `case "cleanup"`,
/// qmd.ts:4979-5019).
#[derive(Debug, Clone, clap::Args)]
#[command(allow_negative_numbers = true)]
pub(crate) struct CleanupArgs {
    /// `--dry-run` — report what a cleanup would remove.
    #[arg(long = "dry-run")]
    pub(crate) dry_run: bool,
}

/// Arguments for `qmd trust` (upstream `case "trust"`, qmd.ts:4701-4704
/// → `manageTrust`, qmd.ts:850-905). `cli.args[0]` is `list`, `revoke`
/// or empty.
#[derive(Debug, Clone, clap::Args)]
#[command(allow_negative_numbers = true)]
pub(crate) struct TrustArgs {
    /// Subcommand positionals — upstream `cli.args`.
    #[arg(num_args = 0..)]
    pub(crate) args: Vec<String>,
}

/// Arguments for `qmd skills` (upstream `case "skills"`,
/// qmd.ts:4920-4936 → `runSkillsCommand`, qmd.ts:3457-3546).
/// `cli.args[0]` is the subcommand (`list`/`get`/`path`/`help`); the
/// flags are also re-scanned in the positional args upstream
/// (`args.includes("--full")`), which our `Vec` preserves for
/// post-`--` spellings.
#[derive(Debug, Clone, clap::Args)]
#[command(allow_negative_numbers = true)]
pub(crate) struct SkillsArgs {
    /// Subcommand plus positionals — upstream `cli.args`.
    #[arg(num_args = 0..)]
    pub(crate) args: Vec<String>,
    /// `--json` — `cli.values.json`: structured output and the
    /// `{success:false,error}` envelope.
    #[arg(long)]
    pub(crate) json: bool,
    /// `--full` — `cli.values.full` for `get` (include
    /// references/templates/scripts).
    #[arg(long)]
    pub(crate) full: bool,
    /// `--all` — `cli.values.all` for `get` (every runtime skill).
    #[arg(long)]
    pub(crate) all: bool,
}

/// Arguments for `qmd skill` (upstream `case "skill"`,
/// qmd.ts:4938-4977 → `installSkill`, qmd.ts:3617-3633).
/// `cli.args[0]` is `show`/`install`/`help`; `--global`/`--yes`/`-f`
/// are `cli.values.global`/`yes`/`force`.
#[derive(Debug, Clone, clap::Args)]
#[command(allow_negative_numbers = true)]
pub(crate) struct SkillArgs {
    /// Subcommand plus positionals — upstream `cli.args`.
    #[arg(num_args = 0..)]
    pub(crate) args: Vec<String>,
    /// `--global` — install into `~/.agents/skills/qmd` instead of
    /// `./.agents/skills/qmd`.
    #[arg(long = "global")]
    pub(crate) global: bool,
    /// `--yes` — also create the `.claude/skills/qmd` symlink without
    /// prompting.
    #[arg(long)]
    pub(crate) yes: bool,
    /// `-f`/`--force` — replace an existing install or symlink.
    #[arg(short = 'f', long)]
    pub(crate) force: bool,
}

#[derive(Parser)]
#[command(
    name = "qmd",
    disable_help_flag = true,
    disable_help_subcommand = true,
    disable_version_flag = true
)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "global upstream boolean flags mirror the parseArgs bucket 1:1"
)]
struct RawCli {
    #[arg(long, global = true)]
    index: Option<String>,
    #[arg(long = "no-gpu", global = true)]
    no_gpu: bool,
    #[arg(short = 'h', long = "help", global = true)]
    help: bool,
    #[arg(short = 'v', long = "version", global = true)]
    version: bool,
    // Upstream's `skill` boolean (qmd.ts:3097) — `qmd skill show`
    // alias; global so `qmd <anything> --skill` prints the skill.
    #[arg(long, global = true)]
    skill: bool,
    #[command(subcommand)]
    command: Option<RawCommand>,
}

#[derive(Subcommand)]
enum RawCommand {
    Search(SearchArgs),
    Get(GetArgs),
    #[command(name = "multi-get")]
    MultiGet(MultiGetArgs),
    Ls(LsArgs),
    Collection(CollectionArgs),
    Context(ContextArgs),
    Init,
    Status,
    Update,
    Cleanup(CleanupArgs),
    Trust(TrustArgs),
    Embed(EmbedArgs),
    Pull(PullArgs),
    // Upstream keeps both aliases undocumented (qmd.ts:4763, 4778).
    #[command(name = "vsearch", alias = "vector-search")]
    Vsearch(QueryArgs),
    #[command(name = "query", alias = "deep-search")]
    Query(QueryArgs),
    Doctor,
    Skills(SkillsArgs),
    Skill(SkillArgs),
    /// `mcp` — MCP server (upstream qmd.ts:4811+ `case "mcp"`); the
    /// bare command serves stdio, positionals are rejected for now.
    #[cfg(feature = "mcp")]
    #[command(name = "mcp")]
    Mcp(McpArgs),
    #[command(external_subcommand)]
    Other(Vec<String>),
}

/// Parse `argv[1..]` into [`ParsedCli`]; `Err` is the clap diagnostic to
/// print on stderr (exit 1).
pub(crate) fn parse(argv: &[String]) -> Result<ParsedCli, clap::Error> {
    let raw =
        RawCli::try_parse_from(std::iter::once("qmd".to_owned()).chain(argv.iter().cloned()))?;
    let command = raw.command.map(|c| match c {
        RawCommand::Search(search) => Command::Search(search),
        RawCommand::Get(get) => Command::Get(get),
        RawCommand::MultiGet(multi) => Command::MultiGet(multi),
        RawCommand::Ls(ls) => Command::Ls(ls),
        RawCommand::Collection(collection) => Command::Collection(collection),
        RawCommand::Context(context) => Command::Context(context),
        RawCommand::Init => Command::Init,
        RawCommand::Status => Command::Status,
        RawCommand::Update => Command::Update,
        RawCommand::Cleanup(cleanup) => Command::Cleanup(cleanup),
        RawCommand::Trust(trust) => Command::Trust(trust),
        RawCommand::Embed(embed) => Command::Embed(embed),
        RawCommand::Pull(pull) => Command::Pull(pull),
        RawCommand::Vsearch(vsearch) => Command::Vsearch(vsearch),
        RawCommand::Query(query) => Command::Query(query),
        RawCommand::Doctor => Command::Doctor,
        RawCommand::Skills(skills) => Command::Skills(skills),
        RawCommand::Skill(skill) => Command::Skill(skill),
        #[cfg(feature = "mcp")]
        RawCommand::Mcp(mcp) => Command::Mcp(mcp),
        RawCommand::Other(words) => Command::Unknown(UnknownArgs(words)),
    });
    Ok(ParsedCli {
        index: raw.index,
        no_gpu: raw.no_gpu,
        help: raw.help,
        version: raw.version,
        skill: raw.skill,
        command,
    })
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::missing_assert_message,
    reason = "idiomatic test-code patterns"
)]
mod tests {
    //! Parse-level port of upstream `test/multi-collection-filter.test.ts`
    //! (tobi/qmd#191, tobi/qmd#775) — upstream asserts `parseArgs` with
    //! `multiple: true` semantics for `-c`/`--collection`; our strict
    //! clap derive must accept repeats, singles, and absence.

    use super::{Command, parse};

    fn search_collections(argv: &[&str]) -> Vec<String> {
        let argv: Vec<String> = argv.iter().map(|s| (*s).to_owned()).collect();
        let parsed = parse(&argv).unwrap();
        let Some(Command::Search(args)) = parsed.command else {
            panic!("expected search command, got {:?}", parsed.command);
        };
        args.collection
    }

    #[test]
    fn repeated_collection_flags_produce_an_array() {
        assert_eq!(
            search_collections(&["search", "-c", "docs", "-c", "notes", "q"]),
            ["docs", "notes"]
        );
    }

    #[test]
    fn single_collection_flag_produces_single_element_array() {
        assert_eq!(search_collections(&["search", "-c", "docs", "q"]), ["docs"]);
    }

    #[test]
    fn absent_collection_flag_is_empty() {
        // Upstream `cli.opts.collection === undefined` → `[]` in
        // `resolveCollectionFilter`; our `Vec` is empty directly.
        assert!(search_collections(&["search", "q"]).is_empty());
    }

    #[test]
    fn collection_long_form_and_equals_value() {
        assert_eq!(
            search_collections(&["search", "--collection", "docs", "--collection=notes", "q"]),
            ["docs", "notes"]
        );
    }
}
