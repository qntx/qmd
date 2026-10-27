//! `qmd mcp` — stdio MCP server (upstream qmd.ts `case "mcp"`,
//! qmd.ts:4811-4917; server.ts `startMcpServer`), `--http`
//! (server.ts `startMcpHttpServer`), and the `--daemon`/`stop` pidfile
//! lifecycle (`mcp-pid.ts`, qmd.ts:4814-4888).

use std::path::Path;
use std::sync::Arc;

use qmd::Environment;
use qmd_mcp::daemon::{self, DaemonPaths, is_qmd_mcp_pid, read_pidfile, sigterm};

use crate::context::CliContext;
use crate::{Failure, Io, args};

/// Run the MCP server. Positionals other than `stop` are rejected —
/// deliberate deviation (parity-matrix deviation 73): upstream only
/// branches on `stop`, so `qmd mcp status` or a typo silently starts a
/// stdio server; here they fail fast instead.
pub(crate) fn run(
    env: &Environment,
    ctx: &CliContext,
    args: &args::McpArgs,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    // A daemon-respawned child detaches its session before anything
    // else — the safe equivalent of upstream's `detached: true` spawn
    // option (P5-D8). A stray marker on a normal invocation is harmless:
    // `setsid` only moves the process to a new session.
    if std::env::var_os(daemon::DAEMON_CHILD_ENV).is_some() {
        drop(daemon::detach_daemon_child());
    }

    // qmd.ts:4814-4838 — the subcommand wins over every flag, so
    // `qmd mcp stop` works without `--http`.
    let daemon = daemon::daemon_paths(env, ctx.index_name());
    match args.args.first().map(String::as_str) {
        Some("stop") => return stop_daemon(&daemon, io),
        Some(sub) => {
            let _ = writeln!(io.err, "Unknown subcommand: {sub}");
            let _ = writeln!(io.err, "{}", crate::DOCTOR_HINT);
            return Err(Failure::Reported(1));
        }
        None => {}
    }

    // Port/host/`--daemon` are parsed inside `if (cli.values.http)`
    // upstream (qmd.ts:4839-4849) — stdio ignores all three flags.
    let (port, host) = if args.http {
        // `Number(cli.values.port) || 8181` (qmd.ts:4840).
        let port = match http_port(args.port.as_deref()) {
            Ok(p) => p,
            Err(msg) => {
                let _ = writeln!(io.err, "{msg}");
                return Err(Failure::Reported(1));
            }
        };
        // `cli.values.host ? String(host) : undefined` — a falsy host
        // falls through to `QMD_HOST`/`localhost` in serve_http.
        let host = args
            .host
            .as_deref()
            .filter(|h| !h.is_empty())
            .map(str::to_owned);
        (port, host)
    } else {
        (0, None)
    };

    // qmd.ts:4849-4888 — the launcher exits after printing; it never
    // opens the store (the spawned child does).
    if args.http && args.daemon {
        return spawn_daemon(ctx, &daemon, port, host.as_deref(), io);
    }

    // `open_store` applies the trust gate to config-declared runtime
    // models — stricter than upstream's ungated `createStore`, which is
    // the right call for a long-lived server process (parity-matrix deviation 75).
    let store = match ctx.open_store(env) {
        Ok(s) => s,
        Err(e) => {
            let _ = writeln!(io.err, "{e}");
            let _ = writeln!(io.err, "{}", crate::DOCTOR_HINT);
            return Err(Failure::Reported(1));
        }
    };
    // Multi-thread runtime so `spawn_blocking` tool calls can overlap —
    // upstream's store interleaves requests at each `await`.
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            let _ = writeln!(io.err, "{e}");
            let _ = writeln!(io.err, "{}", crate::DOCTOR_HINT);
            return Err(Failure::Reported(1));
        }
    };
    let store = Arc::new(store);
    let outcome = if args.http {
        let mut opts = qmd_mcp::HttpOptions::default();
        opts.port = port;
        opts.host = host;
        rt.block_on(qmd_mcp::serve_http(store, opts))
    } else {
        rt.block_on(qmd_mcp::serve_stdio(
            store,
            qmd_mcp::StdioOptions::default(),
        ))
    };
    // `Runtime`'s `Drop` waits forever for `spawn_blocking` work; after a
    // drain timeout a tool call may still be parked in inference, and
    // upstream abandons it and exits (server.ts:742-760).
    rt.shutdown_background();
    // `process.on("exit", unlinkOwnPidfile)` (qmd.ts:4892-4899) — runs on
    // every exit upstream; here it covers every path out of the server.
    if args.http {
        cleanup_own_pidfile(&daemon.pid_path);
    }
    match outcome {
        Ok(()) => Ok(()),
        // qmd.ts:4908-4911 — the only bind error upstream translates.
        Err(qmd_mcp::Error::Io(e)) if args.http && e.kind() == std::io::ErrorKind::AddrInUse => {
            let _ = writeln!(
                io.err,
                "Port {port} already in use. Try a different port with --port."
            );
            Err(Failure::Reported(1))
        }
        Err(e) => {
            let _ = writeln!(io.err, "{e}");
            let _ = writeln!(io.err, "{}", crate::DOCTOR_HINT);
            Err(Failure::Reported(1))
        }
    }
}

/// `qmd mcp stop` (qmd.ts:4821-4838): no pidfile, stale file, or a
/// confirmed qmd pid — all three outcomes exit 0.
fn stop_daemon(daemon: &DaemonPaths, io: &mut Io<'_>) -> Result<(), Failure> {
    if !daemon.pid_path.exists() {
        let _ = writeln!(io.out, "Not running (no PID file).");
        return Ok(());
    }
    let mut stale = || -> Result<(), Failure> {
        drop(std::fs::remove_file(&daemon.pid_path));
        let _ = writeln!(
            io.out,
            "Cleaned up stale PID file (server was not running)."
        );
        Ok(())
    };
    match read_pidfile(&daemon.pid_path) {
        // `isQmdMcpPid` — alive AND qmd-looking cmdline; a foreign or
        // dead pid makes the file stale.
        Some(pid) if is_qmd_mcp_pid(pid) => {
            // Upstream wraps kill AND unlinkSync in one try — a failure
            // of either lands on the catch's stale-file message.
            match sigterm(pid).and_then(|()| std::fs::remove_file(&daemon.pid_path)) {
                Ok(()) => {
                    let _ = writeln!(io.out, "Stopped QMD MCP server (PID {pid}).");
                    Ok(())
                }
                Err(_) => stale(),
            }
        }
        _ => stale(),
    }
}

/// `cli.values.daemon` (qmd.ts:4849-4888): refuse a live daemon, clean
/// a stale pidfile, respawn the CLI detached with a truncated log,
/// write the pidfile, then report on stdout.
fn spawn_daemon(
    ctx: &CliContext,
    daemon: &DaemonPaths,
    port: u16,
    host: Option<&str>,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    // qmd.ts:4851-4860 — a live qmd pid refuses; a stale or recycled
    // file is removed and the spawn continues.
    if let Some(pid) = read_pidfile(&daemon.pid_path)
        && is_qmd_mcp_pid(pid)
    {
        let _ = writeln!(
            io.err,
            "Already running (PID {pid}). Run 'qmd mcp stop' first."
        );
        return Err(Failure::Reported(1));
    }
    if daemon.pid_path.exists() {
        drop(std::fs::remove_file(&daemon.pid_path));
    }

    let spawn = || -> std::io::Result<u32> {
        std::fs::create_dir_all(&daemon.cache_dir)?;
        // `openSync(logPath, "w")` — a fresh log per daemon run; the two
        // child output streams share the file description like Node's
        // `stdio: ["ignore", logFd, logFd]`.
        let log = std::fs::File::create(&daemon.log_path)?;
        let err_log = log.try_clone()?;
        // `spawnArgs` (qmd.ts:4863-4872): our single binary replaces
        // upstream's execPath+script pair. `INDEX_PATH` pins the
        // resolved db so the child cannot inherit a stale one; the
        // marker env has the child detach its own session.
        let mut argv: Vec<std::ffi::OsString> = Vec::new();
        if ctx.index_name() != "index" {
            argv.extend(["--index".into(), ctx.index_name().into()]);
        }
        argv.extend([
            "mcp".into(),
            "--http".into(),
            "--port".into(),
            port.to_string().into(),
        ]);
        if let Some(h) = host {
            argv.extend(["--host".into(), h.into()]);
        }
        let child = std::process::Command::new(std::env::current_exe()?)
            .args(argv)
            .stdin(std::process::Stdio::null())
            .stdout(log)
            .stderr(err_log)
            .env("INDEX_PATH", ctx.db_path())
            .env(daemon::DAEMON_CHILD_ENV, "1")
            .spawn()?;
        // `child.unref()` — nothing waits on the daemon from here.
        std::fs::write(&daemon.pid_path, child.id().to_string())?;
        Ok(child.id())
    };
    match spawn() {
        // qmd.ts:4883-4885 — stdout lines, then `process.exit(0)`.
        Ok(pid) => {
            let _ = writeln!(
                io.out,
                "Started on http://{}:{port}/mcp (PID {pid})",
                host.unwrap_or("localhost")
            );
            let _ = writeln!(io.out, "Logs: {}", daemon.log_path.display());
            Ok(())
        }
        Err(e) => {
            let _ = writeln!(io.err, "{e}");
            let _ = writeln!(io.err, "{}", crate::DOCTOR_HINT);
            Err(Failure::Reported(1))
        }
    }
}

/// `process.on("exit", unlinkOwnPidfile)` (qmd.ts:4892-4899): remove the
/// daemon pidfile only when it names *this* process — foreground mode
/// never writes it, so this only fires when a recycled pidfile happens
/// to match.
fn cleanup_own_pidfile(pid_path: &Path) {
    if read_pidfile(pid_path) == Some(i64::from(std::process::id())) {
        drop(std::fs::remove_file(pid_path));
    }
}

/// `Number(v) || 8181` (qmd.ts:4840): parse failures (`NaN`) and `0`
/// fall back to the default; out-of-range is rejected rather than
/// crashed on like node's `ERR_SOCKET_BAD_PORT`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "JS Number→port semantics; range-checked to 1..=65535 first"
)]
fn http_port(arg: Option<&str>) -> Result<u16, String> {
    let Some(raw) = arg else {
        return Ok(8181);
    };
    match raw.trim().parse::<f64>() {
        Ok(n) if n.is_nan() || n == 0.0 => Ok(8181),
        Ok(n) if (1.0..=65535.0).contains(&n) => Ok(n as u16),
        Ok(_) => Err(format!("Invalid port: {raw}")),
        Err(_) => Ok(8181),
    }
}
