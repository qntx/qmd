//! `qmd mcp` — stdio MCP server (upstream qmd.ts `case "mcp"`,
//! qmd.ts:4811-4917; server.ts `startMcpServer`).

use std::sync::Arc;

use qmd::Environment;

use crate::context::CliContext;
use crate::{Failure, Io, args};

/// Run the stdio MCP server. Any positional is rejected until P5.4
/// adds `stop`. Deliberate deviation (parity-matrix deviation 73):
/// upstream only branches on `stop`, so `qmd mcp status` or a typo
/// silently starts a stdio server; here they fail fast instead.
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
    let outcome = rt.block_on(qmd_mcp::serve_stdio(
        Arc::new(store),
        qmd_mcp::StdioOptions::default(),
    ));
    // `Runtime`'s `Drop` waits forever for `spawn_blocking` work; after a
    // drain timeout a tool call may still be parked in inference, and
    // upstream abandons it and exits (server.ts:742-760).
    rt.shutdown_background();
    match outcome {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = writeln!(io.err, "{e}");
            let _ = writeln!(io.err, "{}", crate::DOCTOR_HINT);
            Err(Failure::Reported(1))
        }
    }
}
