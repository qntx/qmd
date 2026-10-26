//! `qmd` binary — thin wrapper over [`qmd_cli::run`] (P4-D1).
//!
//! Assembles the real process environment: the local UTC offset is read
//! first, before anything could spawn a thread (P1 T5); tty detection
//! uses `IsTerminal`. SIGINT/SIGTERM restore the terminal cursor and
//! exit 130/143 like upstream (qmd.ts:332-333).

use std::io::IsTerminal;
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
// Used by the lib's cursor-restore watcher; the bin itself never
// references it, but the lint counts dependencies per target.
use signal_hook as _;
#[cfg(test)]
use tempfile as _;
#[cfg(feature = "mcp")]
use tokio as _;

fn main() -> ExitCode {
    // `local-offset` queries the system zoneinfo and is not safe once
    // other threads exist; the offset is captured here for the whole
    // process (upstream reads the system timezone lazily everywhere).
    let local_offset = time::UtcOffset::current_local_offset().unwrap_or(time::UtcOffset::UTC);
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
