//! `qmd mcp` — stdio MCP server (upstream qmd.ts `case "mcp"`,
//! qmd.ts:4811-4917; server.ts `startMcpServer`).

use std::sync::Arc;

use qmd::Environment;

use crate::context::CliContext;
use crate::{Failure, Io, args};

/// Run the stdio MCP server. Any positional is rejected until P5.4
/// adds `stop` (upstream's `status` is likewise not implemented — it
/// falls into the same unknown-subcommand shape upstream has for
/// `qmd mcp <other>`).
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
    // the right call for a long-lived server process (deviation 76).
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
    match rt.block_on(qmd_mcp::serve_stdio(
        Arc::new(store),
        qmd_mcp::StdioOptions::default(),
    )) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = writeln!(io.err, "{e}");
            let _ = writeln!(io.err, "{}", crate::DOCTOR_HINT);
            Err(Failure::Reported(1))
        }
    }
}
