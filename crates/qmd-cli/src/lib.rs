//! `qmd` command-line interface (P4.1).
//!
//! Ports upstream `tobi/qmd` `src/cli/qmd.ts` dispatch (qmd.ts:4361-5032)
//! and the read-only commands `search` (with its `outputResults`
//! renderers, qmd.ts:2388-2704), `get`, `multi-get`, `ls` and the
//! `skills`/`skill`/`--skill` surface (P4.4). The crate
//! is a library plus a thin `qmd` binary: `main` only assembles the
//! real process environment and calls [`run`], so tests drive the CLI
//! in-process or through `CARGO_BIN_EXE_qmd` (P4-D1).
//!
//! All output goes through the [`Io`] writers (the workspace forbids
//! `print_stdout`/`print_stderr`), and every path returns an
//! [`ExitCode`] instead of `process.exit` (P4-D2/T3). The
//! [`qmd::Qmd`] handle must not outlive `main` — llama.cpp asserts in a static
//! destructor on macOS when Metal resources outlive the process — so
//! commands hold it in a local that drops before `run` returns
//! (parity-matrix §12 #47).

// Writes into `Io` go to stdout/stderr (or test buffers); upstream's
// console.log/console.error cannot surface short-write errors either,
// so `let _ = writeln!(…)` is the idiom throughout this crate.
#![allow(
    clippy::let_underscore_must_use,
    reason = "output errors are deliberately ignored, mirroring upstream \
              console.log/console.error"
)]

use std::ffi::OsString;
use std::path::Path;
use std::process::ExitCode;

// `regex` is a dev-dependency used by the `cli_golden` test binary; the
// lib itself never touches it.
#[cfg(test)]
use regex as _;
// `rusqlite` is a dev-dependency used by `cli_doctor` to seed
// `content_vectors` rows; the lib never touches it.
#[cfg(test)]
use rusqlite as _;
// `signal-hook` (unix) / `ctrlc` (windows) backs the cursor-restore
// watcher `run` installs for non-MCP commands once argv is parsed.
mod args;
mod bench;
mod collections;
#[cfg(test)]
mod commit_stamp;
mod context;
mod contexts;
mod doctor;
mod embed;
mod embed_lock;
mod gate;
mod get;
mod help;
mod json;
mod ls;
mod maintenance;
#[cfg(feature = "mcp")]
mod mcp;
mod multi_get;
mod output;
mod progress;
mod query;
mod search;
mod skills;
mod slice;

pub use output::Io;

/// Upstream `printDoctorHint` (qmd.ts:4325-4327).
const DOCTOR_HINT: &str = "If qmd still behaves unexpectedly, run 'qmd doctor' for diagnostics.";

/// Command outcome (P4-D2): either the message was already written to
/// stderr, or a [`qmd::Error`] that maps through upstream
/// `exitWithError` (qmd.ts:4329-4333: message + doctor hint, exit 1).
pub(crate) enum Failure {
    /// User-facing output already written; exit with this code.
    Reported(u8),
    /// Unhandled error — printed with the doctor hint, exit 1.
    Error(qmd::Error),
    /// Verbatim user-facing message — printed with the doctor hint,
    /// exit 1 (upstream `exitWithError` on a `throw`n `Error`).
    Message(String),
}

impl From<qmd::Error> for Failure {
    fn from(err: qmd::Error) -> Self {
        Self::Error(err)
    }
}

/// CLI entry point — mirrors the `isMain` block in upstream qmd.ts.
///
/// `args` is `argv[1..]`; `env` is the injected process environment
/// (`--no-gpu` rewrites `env.qmd_force_cpu`, upstream sets
/// `QMD_FORCE_CPU`); `cwd` fills in for `process.cwd()`
/// (`findLocalConfigPath`, `renderFullPath`); `env.pwd` for `getPwd()`.
#[doc(hidden)]
pub fn run(
    args: impl IntoIterator<Item = OsString>,
    env: qmd::Environment,
    cwd: &Path,
    io: &mut Io<'_>,
) -> ExitCode {
    let argv: Vec<String> = args
        .into_iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    let parsed = match args::parse(&argv) {
        Ok(parsed) => parsed,
        Err(err) => {
            // Strict parsing (P4-D2): clap diagnostics on stderr, exit 1.
            let _ = write!(io.err, "{}", err.render());
            return ExitCode::from(1);
        }
    };
    // Deliberate deviation (parity-matrix deviation 76): upstream keeps
    // its cursor-restore `process.exit` handler under `mcp` too. The
    // server is long-lived with models resident, and exiting with a live
    // llama.cpp context risks the Metal teardown abort (deviation 47);
    // the default signal action ends the process without exit handlers.
    // The check runs on the parsed command, so `qmd --index <n> mcp` is
    // covered — upstream's argv-position check misses it. Under `--http`
    // the server's own SIGTERM/SIGINT handlers own shutdown instead
    // (qmd.ts:4894-4896 removes the CLI's listeners there).
    #[cfg(feature = "mcp")]
    let is_mcp = matches!(&parsed.command, Some(args::Command::Mcp(_)));
    #[cfg(not(feature = "mcp"))]
    let is_mcp = false;
    if !is_mcp {
        install_signal_handlers();
    }
    let mut env = env;
    if parsed.no_gpu {
        // Upstream `QMD_FORCE_CPU = "1"` (qmd.ts:3155-3157); writing the
        // injected environment, not the real one.
        env.qmd_force_cpu = Some("1".to_owned());
    }
    let mut ctx = context::CliContext::resolve(&env, cwd, parsed.index.as_deref());

    // Upstream precedence: --version > --skill > --help
    // (qmd.ts:4374-4401).
    if parsed.version {
        let _ = writeln!(io.out, "{}", version_string());
        return ExitCode::SUCCESS;
    }
    // `cli.values.skill` (qmd.ts:4379-4382) — `qmd skill show` alias.
    // Upstream's unguarded `showSkill()` would crash on a missing
    // skill; `Failure::Error` renders the message + doctor hint
    // instead (deliberate clean-up).
    if parsed.skill {
        return exit_code(skills::show_skill(&env, &ctx, io), io);
    }
    // `cli.values.help && cli.command === "skill"` (qmd.ts:4384-4396)
    // — skill usage, exit 0; checked before the general help branch
    // like upstream.
    if parsed.help
        && let Some(args::Command::Skill(_)) = parsed.command
    {
        skills::print_skill_usage(io);
        return ExitCode::SUCCESS;
    }
    if parsed.help {
        help::show(io, &ctx);
        return ExitCode::SUCCESS;
    }
    let Some(command) = parsed.command else {
        help::show(io, &ctx);
        return ExitCode::from(1);
    };

    let result: Result<(), Failure> = match command {
        args::Command::Search(search) => search::run(&env, &ctx, &search, io),
        args::Command::Get(get) => get::run(&env, &mut ctx, &get, io),
        args::Command::MultiGet(multi) => multi_get::run(&env, &ctx, &multi, io),
        args::Command::Ls(ls) => ls::run(&env, &ctx, &ls, io),
        args::Command::Collection(collection) => collections::run(&env, &ctx, &collection, io),
        args::Command::Context(context_args) => contexts::run(&env, &ctx, &context_args, io),
        args::Command::Init => maintenance::init(&env, io),
        args::Command::Status => maintenance::status(&env, &ctx, io),
        args::Command::Update => maintenance::update(&env, &ctx, io),
        args::Command::Cleanup(cleanup) => maintenance::cleanup(&env, &ctx, cleanup.dry_run, io),
        args::Command::Trust(trust_args) => maintenance::trust(&env, &ctx, &trust_args, io),
        args::Command::Embed(embed_args) => embed::embed(&env, &ctx, &embed_args, io),
        args::Command::Pull(pull_args) => embed::pull(&env, &ctx, &pull_args, io),
        args::Command::Vsearch(query_args) => query::vsearch(&env, &ctx, &query_args, io),
        args::Command::Query(query_args) => query::query(&env, &ctx, &query_args, io),
        args::Command::Doctor => doctor::doctor(&env, &ctx, io),
        args::Command::Skills(skills_args) => skills::run_skills(&env, &ctx, &skills_args, io),
        args::Command::Skill(skill_args) => skills::run_skill(&env, &ctx, &skill_args, io),
        args::Command::Bench(bench_args) => bench::cmd_bench(&bench_args, &ctx, &env, io),
        #[cfg(feature = "mcp")]
        args::Command::Mcp(mcp_args) => mcp::run(&env, &ctx, &mcp_args, io),
        args::Command::Unknown(unknown) => {
            let _ = writeln!(io.err, "Unknown command: {}", unknown.command());
            let _ = writeln!(io.err, "Run 'qmd --help' for usage.");
            let _ = writeln!(io.err, "{DOCTOR_HINT}");
            return ExitCode::from(1);
        }
    };
    exit_code(result, io)
}

/// Map a command result onto the process exit code — shared by the
/// command dispatch and the `--skill` early return.
fn exit_code(result: Result<(), Failure>, io: &mut Io<'_>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Reported(code)) => ExitCode::from(code),
        // Upstream `exitWithError` (qmd.ts:4329-4333).
        Err(Failure::Error(err)) => {
            let _ = writeln!(io.err, "{err}");
            let _ = writeln!(io.err, "{DOCTOR_HINT}");
            ExitCode::from(1)
        }
        Err(Failure::Message(msg)) => {
            let _ = writeln!(io.err, "{msg}");
            let _ = writeln!(io.err, "{DOCTOR_HINT}");
            ExitCode::from(1)
        }
    }
}

/// `qmd <version>[ (<commit>)]` — upstream `showVersion` (qmd.ts:4347-4358).
/// The commit is the build-time stamp from build.rs (P4-D6); a foreign
/// or absent checkout stamps nothing, matching `resolveCommit`'s #787
/// guard.
fn version_string() -> String {
    match option_env!("QMD_CLI_COMMIT") {
        Some(commit) if !commit.is_empty() => {
            format!("qmd {} ({commit})", env!("CARGO_PKG_VERSION"))
        }
        _ => format!("qmd {}", env!("CARGO_PKG_VERSION")),
    }
}

/// Upstream `process.on('SIGINT'|'SIGTERM')` (qmd.ts:332-333): show the
/// cursor (`cursor.show()` — it can stay hidden after an interrupted
/// `embed` render, qmd.ts:2224) then `process.exit(130|143)`.
///
/// `signal-hook` runs real handlers on a watcher thread (the installed
/// OS handler only writes a self-pipe), so the cursor write below is
/// ordinary stderr output — a lock wait on an in-flight `writeln!` is a
/// bounded stall, matching upstream's unconditional
/// `process.stderr.write`. `std::process::exit` runs from that thread,
/// so destructors — including the llama.cpp Metal teardown upstream
/// avoids via `process.exit` (qmd.ts:285-303) — never run.
#[cfg(unix)]
#[allow(
    clippy::exit,
    clippy::disallowed_methods,
    clippy::let_underscore_must_use,
    reason = "signal termination cannot return an ExitCode and cannot surface \
              a write error; mirrors upstream process.exit(130|143) and an \
              unconditional process.stderr.write"
)]
fn install_signal_handlers() {
    use std::io::Write as _;

    use signal_hook::consts::{SIGINT, SIGTERM};
    let Ok(mut signals) = signal_hook::iterator::Signals::new([SIGINT, SIGTERM]) else {
        return;
    };
    // A detached watcher is fine: failure to spawn only loses the
    // cursor restore, and the default signal action still applies.
    drop(
        std::thread::Builder::new()
            .name("signal-cursor-restore".to_owned())
            .spawn(move || {
                // The process is gone after the first signal.
                if let Some(sig) = signals.forever().next() {
                    let _ = std::io::stderr().write_all(b"\x1b[?25h");
                    std::process::exit(if sig == SIGTERM { 143 } else { 130 });
                }
            }),
    );
}

/// Windows counterpart of the unix handler above: `ctrlc`'s handler
/// runs on its own thread, so the same cursor-show + `exit(130)`
/// applies (it also fires on Ctrl-Break — Node maps that to SIGBREAK
/// with a terminate default, so exiting 130 is the same outcome).
/// Windows raises no SIGTERM, so upstream's 143 branch is
/// unreachable.
#[cfg(windows)]
#[allow(
    clippy::exit,
    clippy::disallowed_methods,
    reason = "signal termination cannot return an ExitCode and cannot surface \
              a write error; mirrors upstream process.exit(130) and an \
              unconditional process.stderr.write"
)]
fn install_signal_handlers() {
    use std::io::Write as _;

    // Registration failure only loses the cursor restore; the console
    // default action still applies.
    drop(ctrlc::set_handler(move || {
        let _ = std::io::stderr().write_all(b"\x1b[?25h");
        std::process::exit(130);
    }));
}

/// Other platforms have no console signal path to mirror — the
/// handler is optional anyway (it only restores the cursor).
#[cfg(not(any(unix, windows)))]
fn install_signal_handlers() {}
