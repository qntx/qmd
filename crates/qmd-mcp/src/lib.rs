//! MCP server for qmd — ports `tobi/qmd` `src/mcp/server.ts` (v2.8.3).
//!
//! Dual-era protocol over stdio and HTTP (`serve_stdio`/`serve_http`):
//! 2025-era clients use the initialize handshake; 2026-07-28 clients
//! send namespaced `_meta` and call `server/discover` without one.
//! Known envelope deviations vs upstream are registered in
//! `design/parity-matrix.md` §12 (deviations 69+).
//!
//! Embedding the HTTP server — upstream's `HttpServerHandle` —
//! gives the bound port and a caller-controlled shutdown:
//!
//! ```no_run
//! # async fn doc() -> Result<(), Box<dyn std::error::Error>> {
//! use std::sync::Arc;
//!
//! use qmd_mcp::{HttpOptions, HttpServer};
//!
//! let qmd = Arc::new(qmd::Qmd::builder("index.sqlite").build()?);
//! let mut options = HttpOptions::default();
//! options.port = 0; // ask the OS for a free port
//! options.quiet = true;
//! let server = HttpServer::bind(qmd, options).await?;
//! let port = server.port(); // real port for `port: 0`
//!
//! let (tx, rx) = tokio::sync::oneshot::channel::<()>();
//! let task = tokio::spawn(async move { server.serve(async move { rx.await.ok(); }).await });
//! tx.send(()).ok(); // shut down when ready
//! task.await??;
//! # Ok(())
//! # }
//! ```

pub mod daemon;
mod http;
mod instructions;
mod js;
mod origin_guard;
mod resource;
mod rest;
mod schema;
mod server;
mod stdio;
mod tools;

pub use http::{HttpOptions, HttpServer, serve_http};
pub use server::QmdMcpService;
pub use stdio::{StdioOptions, serve, serve_stdio};

/// MCP server error — a shutdown-step failure surfaces as `Err` so the
/// caller exits nonzero (upstream sets `process.exitCode = 1`).
#[expect(
    clippy::error_impl_error,
    reason = "the crate's public error type is named Error by the settled P5.2 interface"
)]
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The qmd store failed.
    #[error(transparent)]
    Store(#[from] qmd::Error),
    /// The transport or a spawned handler failed.
    #[error("{0}")]
    Serve(String),
    /// The pinned upstream schema/config JSON failed to deserialize.
    #[error("pinned JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// Binding or serving the HTTP listener failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
use tempfile as _;
