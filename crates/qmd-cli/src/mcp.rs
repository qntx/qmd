//! `qmd mcp` — stdio MCP server (upstream qmd.ts `case "mcp"`,
//! qmd.ts:4811-4917; server.ts `startMcpServer`) and `--http`
//! (server.ts `startMcpHttpServer`). `--daemon`, `stop` and the pidfile
//! lifecycle stay with P5.4.

use std::sync::Arc;

use qmd::Environment;

use crate::context::CliContext;
use crate::{Failure, Io, args};

/// Run the MCP server. Any positional is rejected until P5.4 adds
/// `stop`. Deliberate deviation (parity-matrix deviation 73): upstream
/// only branches on `stop`, so `qmd mcp status` or a typo silently
/// starts a stdio server; here they fail fast instead.
pub(crate) fn run(
    env: &Environment,
    ctx: &CliContext,
    args: &args::McpArgs,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    if let Some(sub) = args.args.first() {
        let _ = writeln!(io.err, "Unknown subcommand: {sub}");
        let _ = writeln!(io.err, "{}", crate::DOCTOR_HINT);
        return Err(Failure::Reported(1));
    }
    // `Number(cli.values.port) || 8181` (qmd.ts:4840); resolved before
    // the store opens like upstream's flag handling.
    let port = match http_port(args.port.as_deref()) {
        Ok(p) => p,
        Err(msg) => {
            let _ = writeln!(io.err, "{msg}");
            return Err(Failure::Reported(1));
        }
    };
    // `cli.values.host ? String(host) : undefined` (qmd.ts:4841-4844) —
    // a falsy host falls through to `QMD_HOST`/`localhost` in serve_http.
    let host = args
        .host
        .as_deref()
        .filter(|h| !h.is_empty())
        .map(str::to_owned);
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
