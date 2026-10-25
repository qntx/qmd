//! `qmd` binary — thin wrapper over [`qmd_cli::run`] (P4-D1).
//!
//! Assembles the real process environment: the local UTC offset is read
//! first, before anything could spawn a thread (P1 T5); tty detection
//! uses `IsTerminal`. SIGINT/SIGTERM restore the terminal cursor and
//! exit 130/143 like upstream (qmd.ts:332-333).

use std::io::{IsTerminal, Write};
use std::process::ExitCode;

// The bin uses only `qmd_cli::run`; these are lib-facing dependencies.
use clap as _;
#[cfg(feature = "mcp")]
use qmd_mcp as _;
#[cfg(test)]
use regex as _;
#[cfg(test)]
use rusqlite as _;
use serde_json as _;
#[cfg(test)]
use tempfile as _;
#[cfg(feature = "mcp")]
use tokio as _;

fn main() -> ExitCode {
    // `local-offset` queries the system zoneinfo and is not safe once
    // other threads exist; the offset is captured here for the whole
    // process (upstream reads the system timezone lazily everywhere).
    let local_offset = time::UtcOffset::current_local_offset().unwrap_or(time::UtcOffset::UTC);
    // MCP stdio gets no signal watcher: upstream bin/qmd:27 checks
    // `argv[2] === "mcp"` for the same reason — the process exits via
    // stdin EOF, and stdout must stay JSON-RPC only.
    if std::env::args().nth(1).as_deref() != Some("mcp") {
        install_signal_handlers();
    }
    let env = qmd::Environment::from_process();
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let stdout_tty = std::io::stdout().is_terminal();
    let stderr_tty = std::io::stderr().is_terminal();
    let stdin_tty = std::io::stdin().is_terminal();
    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();
    let mut stdin = std::io::BufReader::new(std::io::stdin());
    let mut io = qmd_cli::Io {
        out: &mut stdout,
        err: &mut stderr,
        stdin: &mut stdin,
        stdout_tty,
        stderr_tty,
        stdin_tty,
        local_offset,
    };
    qmd_cli::run(std::env::args_os().skip(1), env, &cwd, &mut io)
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
#[allow(
    clippy::exit,
    clippy::disallowed_methods,
    clippy::let_underscore_must_use,
    reason = "signal termination cannot return an ExitCode and cannot surface \
              a write error; mirrors upstream process.exit(130|143) and an \
              unconditional process.stderr.write"
)]
fn install_signal_handlers() {
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
