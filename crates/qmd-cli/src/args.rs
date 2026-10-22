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
pub(crate) struct ParsedCli {
    /// Global `--index <name>` (normalized by [`crate::context`]).
    pub(crate) index: Option<String>,
    /// Global `--no-gpu`.
    pub(crate) no_gpu: bool,
    /// Global `-h`/`--help`.
    pub(crate) help: bool,
    /// Global `-v`/`--version`.
    pub(crate) version: bool,
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
    /// Any unregistered first positional (`help`, `version`, …) —
    /// upstream's `default:` branch.
    Unknown(UnknownArgs),
}

/// The words of an unrecognized command; `command()` is the first one.
#[derive(Debug)]
pub(crate) struct UnknownArgs(pub(crate) Vec<String>);

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

#[derive(Parser)]
#[command(
    name = "qmd",
    disable_help_flag = true,
    disable_help_subcommand = true,
    disable_version_flag = true
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
        RawCommand::Other(words) => Command::Unknown(UnknownArgs(words)),
    });
    Ok(ParsedCli {
        index: raw.index,
        no_gpu: raw.no_gpu,
        help: raw.help,
        version: raw.version,
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
