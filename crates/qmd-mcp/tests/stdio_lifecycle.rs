//! Ports `mcp-stdio-lifecycle.test.ts`: stdin EOF shuts the server
//! down cleanly — `serve` returns `Ok` promptly, and the response
//! stream carries only JSON-RPC. The drain-timeout path is deferred to
//! P5.2-B: stage A has no tool call that can block deterministically.

#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_assert_message,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "idiomatic test-code patterns"
)]

use std::sync::Arc;
use std::time::Duration;

use qmd::Qmd;
use qmd_mcp::{StdioOptions, serve};
use tokio::io::{AsyncWriteExt, duplex};

/// A store on a throwaway index with no collections.
fn store() -> Arc<Qmd> {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.keep().join("index.sqlite");
    Arc::new(Qmd::builder(db).build().unwrap())
}

/// EOF before any request — upstream exits 0 promptly
/// (`registerStdioEofShutdown` serving an empty stream).
#[tokio::test(flavor = "multi_thread")]
async fn eof_without_requests_returns_ok() {
    let (client_r, _server_w) = duplex(64);
    let (server_r, client_w) = duplex(64);
    // Drop the client's writer end: stdin EOF with no in-flight call.
    drop(client_w);
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        serve(store(), server_r, client_r, StdioOptions::default()),
    )
    .await;
    assert!(result.is_ok(), "serve hung past EOF");
    result.unwrap().unwrap();
}

/// A request, its response, then EOF — `serve` returns `Ok` and the
/// byte stream the client received parses as JSON-RPC.
#[tokio::test(flavor = "multi_thread")]
async fn serve_answers_then_clean_eof() {
    let (mut to_server, server_read) = duplex(4096);
    let (server_write, mut from_server) = duplex(4096);
    let qmd = store();
    let task = tokio::spawn(async move {
        serve(qmd, server_read, server_write, StdioOptions::default()).await
    });
    // rmcp serves only after initialize (or 2026-era `_meta`) —
    // deviation 69 vs upstream's permissive no-handshake path.
    to_server
        .write_all(
            br#"{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}"#,
        )
        .await
        .unwrap();
    to_server.write_all(b"\n").await.unwrap();
    to_server
        .write_all(
            br#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"status","arguments":{}}}"#,
        )
        .await
        .unwrap();
    to_server.write_all(b"\n").await.unwrap();
    let mut buf = Vec::new();
    // Read until the id-1 response line arrives.
    while !String::from_utf8_lossy(&buf)
        .lines()
        .any(|l| l.contains("\"id\":1"))
    {
        let mut chunk = [0u8; 1024];
        let n = tokio::io::AsyncReadExt::read(&mut from_server, &mut chunk)
            .await
            .unwrap();
        assert!(n > 0, "server closed stream without answering");
        buf.extend_from_slice(&chunk[..n]);
    }
    for line in String::from_utf8_lossy(&buf).lines() {
        let _: serde_json::Value = serde_json::from_str(line).expect("stdout line not JSON-RPC");
    }
    // EOF — serve must finish promptly and successfully.
    drop(to_server);
    let result = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("serve hung past EOF")
        .unwrap();
    result.unwrap();
}
