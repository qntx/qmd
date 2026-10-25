//! stdio transport + EOF lifecycle — upstream `startMcpServer` +
//! `registerStdioEofShutdown` (mcp/server.ts:642-818). On stdin EOF the
//! SDK transport closes and `waiting()` returns; we then breadcrumb to
//! stderr, drain in-flight handlers up to `drain_timeout`, and drop
//! the store. An EOF with nothing in flight exits promptly.

use std::io::Write as _;
use std::sync::Arc;
use std::time::Duration;

use qmd::Qmd;
use rmcp::ServiceExt;
use rmcp::service::{QuitReason, ServerInitializeError};
use tokio::io::{AsyncRead, AsyncWrite};

use crate::{Error, QmdMcpService};

/// Options for the stdio transport.
#[derive(Debug, Clone, Copy)]
pub struct StdioOptions {
    /// Deadline for the in-flight drain on stdin EOF — upstream's
    /// `idleTimeoutMs` default (server.ts:670, `5000`).
    pub drain_timeout: Duration,
}

impl Default for StdioOptions {
    fn default() -> Self {
        Self {
            drain_timeout: Duration::from_secs(5),
        }
    }
}

/// Serve MCP over injected byte streams — the testable form
/// (upstream's `stdin`/`stderr` injection, server.ts:700-720).
///
/// # Errors
/// [`Error::Store`] when building the server fails; [`Error::Serve`]
/// when the transport fails.
pub async fn serve<R, W>(
    qmd: Arc<Qmd>,
    reader: R,
    writer: W,
    opts: StdioOptions,
) -> Result<(), Error>
where
    R: AsyncRead + Send + Unpin + 'static,
    W: AsyncWrite + Send + Unpin + 'static,
{
    let service = QmdMcpService::new(qmd)?;
    let inflight = Arc::clone(&service.inflight);
    // EOF before initialize reaches us as the typed
    // `ServerInitializeError::ConnectionClosed` — upstream exits 0 on
    // any stdin close, so it maps onto the same clean path.
    let running = match service.serve((reader, writer)).await {
        Ok(r) => r,
        Err(ServerInitializeError::ConnectionClosed(_)) => return Ok(()),
        Err(e) => return Err(Error::Serve(e.to_string())),
    };
    // `waiting()` resolves when the transport closes (stdin EOF —
    // upstream treats it as the client being gone, server.ts:731-738);
    // a JoinError means the service task died — a real failure.
    match running.waiting().await {
        Ok(QuitReason::JoinError(e)) | Err(e) => {
            return Err(Error::Serve(e.to_string()));
        }
        // `Closed`/`Cancelled` (and any newer variant) — normal shutdown.
        Ok(_) => {}
    }

    // Breadcrumb writes are best-effort — the shutdown proceeds even
    // when stderr is already gone.
    drop(writeln!(
        std::io::stderr(),
        "Shutting down (stdin closed)..."
    ));
    if !inflight.wait_for_idle(opts.drain_timeout).await {
        drop(writeln!(
            std::io::stderr(),
            "QMD Warning: in-flight request did not settle before the shutdown deadline; continuing shutdown."
        ));
    }
    Ok(())
}

/// Serve MCP on process stdin/stdout — `qmd mcp` (upstream
/// `serveStdio`, server.ts:808-813).
///
/// # Errors
/// Same as [`serve`].
pub async fn serve_stdio(qmd: Arc<Qmd>, opts: StdioOptions) -> Result<(), Error> {
    serve(qmd, tokio::io::stdin(), tokio::io::stdout(), opts).await
}
