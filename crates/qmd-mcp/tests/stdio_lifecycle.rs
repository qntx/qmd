//! Ports `mcp-stdio-lifecycle.test.ts`: stdin EOF shuts the server
//! down cleanly — `serve` returns `Ok` promptly, and the response
//! stream carries only JSON-RPC. The drain-timeout and settled-within-
//! drain paths need a gated blocking backend and live in `tools.rs`.

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
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter, duplex};

/// A store on a throwaway index with no collections; the `TempDir`
/// guard must outlive the store.
fn store() -> (tempfile::TempDir, Arc<Qmd>) {
    let dir = tempfile::tempdir().unwrap();
    let qmd = Qmd::builder(dir.path().join("index.sqlite"))
        .build()
        .unwrap();
    (dir, Arc::new(qmd))
}

/// EOF before any request — upstream exits 0 promptly
/// (`registerStdioEofShutdown` serving an empty stream).
#[tokio::test(flavor = "multi_thread")]
async fn eof_without_requests_returns_ok() {
    let (client_r, _server_w) = duplex(64);
    let (server_r, client_w) = duplex(64);
    // Drop the client's writer end: stdin EOF with no in-flight call.
    drop(client_w);
    let (_dir, qmd) = store();
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        serve(qmd, server_r, client_r, StdioOptions::default()),
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
    let (_dir, qmd) = store();
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

/// F2 parity lock — `validate_inline_era` replicates rmcp's
/// pre-dispatch validation for intercepted methods; it must produce
/// identical errors to the delegated path, so an rmcp upgrade that
/// changes validation fails here instead of silently diverging.
#[tokio::test(flavor = "multi_thread")]
async fn intercepted_and_delegated_methods_validate_identically() {
    let (_dir, qmd) = store();
    let (client_r, server_w) = duplex(65536);
    let (server_r, client_w) = duplex(65536);
    let server = tokio::spawn(serve(qmd, server_r, server_w, StdioOptions::default()));
    let mut writer = BufWriter::new(client_w);
    let mut reader = BufReader::new(client_r);
    let mut buf = String::new();

    // A 2026 request missing required `_meta` keys and one with an
    // unsupported `_meta.protocolVersion` — sent to both an intercepted
    // method (resources/read) and a delegated one
    // (resources/templates/list). Errors must match byte-for-byte.
    let cases = [
        // Missing `_meta` entirely on a no-initialize session.
        r#"{"jsonrpc":"2.0","id":1,"method":"{METHOD}","params":{PARAMS},"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28"}}"#,
        // Unsupported declared version.
        r#"{"jsonrpc":"2.0","id":2,"method":"{METHOD}","params":{PARAMS},"_meta":{"io.modelcontextprotocol/protocolVersion":"2099-01-01","io.modelcontextprotocol/clientCapabilities":{},"io.modelcontextprotocol/clientInfo":{"name":"t","version":"0"}}}"#,
    ];
    let intercepted = cases.map(|c| {
        c.replace("{METHOD}", "resources/read")
            .replace("{PARAMS}", r#"{"uri":"qmd://x.md"}"#)
    });
    let delegated = cases.map(|c| {
        c.replace("{METHOD}", "resources/templates/list")
            .replace("{PARAMS}", "{}")
    });

    for (i, (a, b)) in intercepted.iter().zip(delegated.iter()).enumerate() {
        let mut errs = Vec::new();
        let reqs = vec![a, b];
        for req in &reqs {
            writer.write_all(req.as_bytes()).await.unwrap();
            writer.write_all(b"\n").await.unwrap();
            writer.flush().await.unwrap();
            buf.clear();
            reader.read_line(&mut buf).await.unwrap();
            errs.push(serde_json::from_str::<serde_json::Value>(buf.trim()).unwrap());
        }
        assert_eq!(
            errs[0], errs[1],
            "case {i}: intercepted vs delegated error differs"
        );
        assert!(
            errs[0].get("error").is_some(),
            "case {i}: expected an error response, got {}",
            errs[0]
        );
    }

    drop(writer);
    server.await.unwrap().unwrap();
}
