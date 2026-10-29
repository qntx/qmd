//! Tool-call paths through `serve` over `tokio::io::duplex`, plus the
//! lifecycle cases that need a blocking call.
//!
//! Mapping to upstream `test/mcp.test.ts` "MCP Server":
//! - `searchFTS`/`searchVec`/`hybridQuery` — those describe blocks test
//!   `store.search` internals; the equivalent coverage lives in
//!   `crates/qmd/tests/hybrid.rs`, not here.
//! - `qmd_get`, `qmd_multi_get`, `qmd_status`, `qmd://` resource —
//!   response shapes are verified byte-for-byte by
//!   `crates/qmd-cli/tests/mcp_golden.rs`; this file adds the cases the
//!   golden corpus cannot express (docid lookup, gated-blocking
//!   lifecycle).
//! - `edge cases`/`MCP spec compliance` — covered by the golden replay
//!   and `stdio_lifecycle.rs`.
//! - `MCP HTTP Transport*` describes — P5.3 (HTTP transport).
//! - `mcp-stdio-lifecycle.test.ts` `InflightGate` — the drain-timeout
//!   and concurrent-dispatch tests below.

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

mod common;

use std::time::{Duration, Instant};

use common::gated_store;
use qmd_mcp::{StdioOptions, serve};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter, duplex};

async fn send<W: tokio::io::AsyncWrite + Unpin>(w: &mut W, v: &Value) {
    w.write_all(v.to_string().as_bytes()).await.unwrap();
    w.write_all(b"\n").await.unwrap();
    w.flush().await.unwrap();
}

async fn recv<R: AsyncBufReadExt + Unpin>(r: &mut R, buf: &mut String) -> Value {
    buf.clear();
    r.read_line(buf).await.unwrap();
    serde_json::from_str(buf.trim()).unwrap()
}

const QUERY_CALL: &str = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"query","arguments":{"searches":[{"type":"lex","query":"index"}],"rerank":true}}}"#;

/// 2025 initialize + initialized notification — required before bare
/// requests (rmcp rejects `_meta`-less requests on no-init sessions;
/// deviation D4).
async fn handshake<W: tokio::io::AsyncWrite + Unpin, R: AsyncBufReadExt + Unpin>(
    w: &mut W,
    r: &mut R,
    buf: &mut String,
) {
    send(w, &json!({"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}))
        .await;
    let _init_resp = recv(r, buf).await;
    send(
        w,
        &json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
    )
    .await;
}

/// In-flight `tools/call` parked in the reranker must not block a
/// second call — concurrent dispatch (upstream serves all requests
/// concurrently; the shared `Arc<Qmd>` unlocks at inference per
/// parity-matrix deviation 43).
#[tokio::test(flavor = "multi_thread")]
async fn second_call_answers_while_first_is_parked() {
    let (_dir, qmd, entered, release) = gated_store();
    let (client_r, server_w) = duplex(65536);
    let (server_r, client_w) = duplex(65536);
    let server = tokio::spawn(serve(qmd, server_r, server_w, StdioOptions::default()));
    let mut writer = BufWriter::new(client_w);
    let mut reader = BufReader::new(client_r);
    let mut buf = String::new();
    handshake(&mut writer, &mut reader, &mut buf).await;

    send(&mut writer, &serde_json::from_str(QUERY_CALL).unwrap()).await;
    match entered.recv_timeout(Duration::from_secs(10)) {
        Ok(()) => {}
        Err(e) => {
            buf.clear();
            let n = tokio::time::timeout(Duration::from_secs(1), reader.read_line(&mut buf)).await;
            panic!("query never reached the reranker ({e:?}); wire said: {n:?} {buf:?}");
        }
    }

    send(
        &mut writer,
        &json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"status","arguments":{}}}),
    )
    .await;
    let first = recv(&mut reader, &mut buf).await;
    assert_eq!(
        first["id"],
        json!(2),
        "status must answer while query is parked"
    );

    release.send(()).unwrap();
    let second = recv(&mut reader, &mut buf).await;
    assert_eq!(second["id"], json!(1));

    drop(writer);
    server.await.unwrap().unwrap();
}

/// EOF with a parked call — upstream's `InflightGate` warns after
/// `drain_timeout` (server.ts:642-818). rmcp additionally drains
/// in-flight handler responses for a hardcoded 5 s inside the service
/// task before `waiting()` resolves (service.rs:1762-1795), so the
/// observable wait is `max(5 s, drain_timeout)`; our gate then covers
/// blocking work still running after the handler task unwinds.
#[tokio::test(flavor = "multi_thread")]
async fn eof_waits_drain_timeout_for_in_flight_call() {
    let (_dir, qmd, entered, release) = gated_store();
    let (client_r, server_w) = duplex(65536);
    let (server_r, client_w) = duplex(65536);
    let server = tokio::spawn(serve(
        qmd,
        server_r,
        server_w,
        StdioOptions {
            drain_timeout: Duration::from_millis(300),
        },
    ));
    let mut writer = BufWriter::new(client_w);
    let mut reader = BufReader::new(client_r);
    let mut buf = String::new();
    handshake(&mut writer, &mut reader, &mut buf).await;
    let _ = (&mut reader, &mut buf);

    send(&mut writer, &serde_json::from_str(QUERY_CALL).unwrap()).await;
    entered
        .recv_timeout(Duration::from_secs(10))
        .expect("query never reached the reranker");

    drop(writer); // stdin EOF with a call parked in the reranker
    let t = Instant::now();
    let result = tokio::time::timeout(Duration::from_secs(10), server)
        .await
        .expect("serve did not finish after drain timeout")
        .unwrap();
    result.unwrap();
    let elapsed = t.elapsed();
    assert!(
        elapsed >= Duration::from_millis(250),
        "drain returned before drain_timeout: {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "rmcp's internal drain extended shutdown past 2 s: {elapsed:?}"
    );
    release.send(()).unwrap();
    let _ = reader; // keep the channel alive through the assertion
}

/// A call that settles *inside* the drain window still gets its
/// response on the wire — rmcp's response drain delivers it before the
/// task ends.
#[tokio::test(flavor = "multi_thread")]
async fn call_settled_within_drain_still_responds() {
    let (_dir, qmd, entered, release) = gated_store();
    let (client_r, server_w) = duplex(65536);
    let (server_r, client_w) = duplex(65536);
    let server = tokio::spawn(serve(
        qmd,
        server_r,
        server_w,
        StdioOptions {
            drain_timeout: Duration::from_secs(1),
        },
    ));
    let mut writer = BufWriter::new(client_w);
    let mut reader = BufReader::new(client_r);
    let mut buf = String::new();
    handshake(&mut writer, &mut reader, &mut buf).await;

    send(&mut writer, &serde_json::from_str(QUERY_CALL).unwrap()).await;
    entered
        .recv_timeout(Duration::from_secs(10))
        .expect("query never reached the reranker");

    // EOF, then release the parked call inside the 1 s window.
    drop(writer);
    tokio::time::sleep(Duration::from_millis(100)).await;
    release.send(()).unwrap();
    let resp = tokio::time::timeout(Duration::from_secs(5), recv(&mut reader, &mut buf))
        .await
        .expect("response for the settled call was not delivered");
    assert_eq!(resp["id"], json!(1));
    assert!(resp.get("error").is_none(), "{resp}");
    server.await.unwrap().unwrap();
}
