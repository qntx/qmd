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
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::sync::Notify;

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
    // The reader is wrapped so stdin's 0-byte read (EOF) is observable
    // the moment it happens — upstream prints its breadcrumb on stdin
    // 'end', before waiting out the drain deadline (server.ts:731-760),
    // while rmcp's `waiting()` only resolves after its internal
    // response drain (~5 s).
    let eof = Arc::new(Notify::new());
    let reader = EofWatch {
        inner: reader,
        eof: Arc::clone(&eof),
    };
    // EOF before initialize reaches us as the typed
    // `ServerInitializeError::ConnectionClosed` — upstream exits 0 on
    // any stdin close, so it maps onto the same clean path.
    let running = match service.serve((reader, writer)).await {
        Ok(r) => r,
        Err(ServerInitializeError::ConnectionClosed(_)) => return Ok(()),
        Err(e) => return Err(Error::Serve(e.to_string())),
    };
    // `Box::pin` keeps ownership so dropping it (drain-timeout path)
    // drops the future — and the `RunningService` inside — letting
    // its DropGuard cancel rmcp's task instead of draining for 5 s.
    let mut waiter = Box::pin(running.waiting());

    tokio::select! {
        () = eof.notified() => {
            // Breadcrumb first — upstream's order is stdin-'end' →
            // print → wait out `drain_timeout` for in-flight calls.
            // Breadcrumb writes are best-effort: shutdown proceeds even
            // when stderr is already gone.
            drop(writeln!(std::io::stderr(), "Shutting down (stdin closed)..."));
            if inflight.wait_for_idle(opts.drain_timeout).await {
                // Everything settled inside the deadline — let rmcp
                // finish flushing responses, then map its quit reason.
                match waiter.await {
                    Ok(QuitReason::JoinError(e)) | Err(e) => {
                        Err(Error::Serve(e.to_string()))
                    }
                    Ok(_) => Ok(()),
                }
            } else {
                drop(writeln!(
                    std::io::stderr(),
                    "QMD Warning: in-flight request did not settle before the shutdown deadline; continuing shutdown."
                ));
                // Drop the waiter → `RunningService` drops → its
                // DropGuard cancels the service task asynchronously, so
                // rmcp's internal response drain cannot extend
                // shutdown (upstream abandons the call and exits 0).
                drop(waiter);
                Ok(())
            }
        }
        res = waiter.as_mut() => {
            // The service ended without an observed 0-byte read —
            // transport close/error surfaced by rmcp itself.
            match res {
                Ok(QuitReason::JoinError(e)) | Err(e) => Err(Error::Serve(e.to_string())),
                Ok(_) => {
                    drop(writeln!(std::io::stderr(), "Shutting down (stdin closed)..."));
                    if !inflight.wait_for_idle(opts.drain_timeout).await {
                        drop(writeln!(
                            std::io::stderr(),
                            "QMD Warning: in-flight request did not settle before the shutdown deadline; continuing shutdown."
                        ));
                    }
                    Ok(())
                }
            }
        }
    }
}

/// `AsyncRead` wrapper that fires `eof` when the underlying read
/// returns 0 bytes or errors — upstream listens on stdin's `end`/`close`
/// events (server.ts:731-738). `notify_one` stores a permit, so an EOF
/// that lands before `serve` polls `notified()` is not lost.
struct EofWatch<R> {
    inner: R,
    eof: Arc<Notify>,
}

impl<R: AsyncRead + Unpin> AsyncRead for EofWatch<R> {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let filled = buf.filled().len();
        match std::pin::Pin::new(&mut self.inner).poll_read(cx, buf) {
            std::task::Poll::Ready(Ok(())) => {
                if buf.filled().len() == filled {
                    self.eof.notify_one();
                }
                std::task::Poll::Ready(Ok(()))
            }
            r @ std::task::Poll::Ready(Err(_)) => {
                self.eof.notify_one();
                r
            }
            r => r,
        }
    }
}

/// Serve MCP on process stdin/stdout — `qmd mcp` (upstream
/// `serveStdio`, server.ts:808-813).
///
/// # Errors
/// Same as [`serve`].
pub async fn serve_stdio(qmd: Arc<Qmd>, opts: StdioOptions) -> Result<(), Error> {
    serve(qmd, tokio::io::stdin(), tokio::io::stdout(), opts).await
}
