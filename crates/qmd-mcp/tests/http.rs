//! HTTP transport concurrency — ports upstream `test/mcp.test.ts`
//! HTTP-section coverage that a golden replay cannot express: real
//! concurrent requests against a live server while a call parks inside
//! inference (the P5.1/deviation-43 connection-lock release), plus the
//! embeddable `HttpServer` handle (upstream `startMcpHttpServer`'s
//! `HttpServerHandle` and `allowedOrigins`/`allowedHosts` options,
//! server.ts:889-1183).
//!
//! Skipped upstream cases: the request/response *content* of every
//! HTTP case is byte-verified by `mcp_golden.rs::mcp_http_golden_replay`;
//! here only the concurrency and handle-behavior assertions remain.

#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_assert_message,
    clippy::panic,
    reason = "idiomatic test-code patterns"
)]

mod common;

use std::io::{Read as _, Write as _};
use std::net::TcpStream;
use std::time::Duration;

use common::gated_store;
use qmd_mcp::{HttpOptions, HttpServer};

/// `Connection: close` request; the response is the full wire text —
/// `(status_line, body)` split on the header boundary.
fn http_request(
    port: u16,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&str>,
) -> (u16, String) {
    use std::fmt::Write as _;
    let mut stream = TcpStream::connect(format!("localhost:{port}")).unwrap();
    // A request that never answers must fail the test, not hang it —
    // the parked-rerank test relies on this bound.
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let mut req =
        format!("{method} {path} HTTP/1.1\r\nHost: localhost:{port}\r\nConnection: close\r\n");
    for (k, v) in headers {
        write!(req, "{k}: {v}\r\n").unwrap();
    }
    if let Some(b) = body {
        write!(req, "Content-Length: {}\r\n", b.len()).unwrap();
    }
    req.push_str("\r\n");
    stream.write_all(req.as_bytes()).unwrap();
    if let Some(b) = body {
        stream.write_all(b.as_bytes()).unwrap();
    }
    let mut text = String::new();
    stream.read_to_string(&mut text).unwrap();
    let status: u16 = text
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let resp_body = text.split("\r\n\r\n").nth(1).unwrap_or("").to_owned();
    (status, resp_body)
}

/// Quiet `HttpServer` on an OS-assigned port + oneshot shutdown.
async fn http_server(
    qmd: std::sync::Arc<qmd::Qmd>,
) -> (
    u16,
    tokio::task::JoinHandle<Result<(), qmd_mcp::Error>>,
    tokio::sync::oneshot::Sender<()>,
) {
    let mut opts = HttpOptions::default();
    opts.port = 0;
    opts.quiet = true;
    let server = HttpServer::bind(qmd, opts).await.unwrap();
    let port = server.port();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(server.serve(async move {
        drop(rx.await);
    }));
    (port, task, tx)
}

/// A call parked inside the reranker does not block other requests —
/// the store's connection lock is released during inference
/// (parity-matrix deviation 43; upstream's single-threaded store
/// interleaves at each `await`).
#[tokio::test(flavor = "multi_thread")]
async fn parked_rerank_does_not_block_other_requests() {
    let (_dir, qmd, entered, release) = gated_store();
    let (port, task, shutdown) = http_server(qmd).await;

    // POST /query with a lex search parks inside the gated reranker.
    let parked = std::thread::spawn(move || {
        http_request(
            port,
            "POST",
            "/query",
            &[("Content-Type", "application/json")],
            Some(r#"{"searches":[{"type":"lex","query":"index"}],"rerank":true}"#),
        )
    });
    entered
        .recv_timeout(Duration::from_secs(10))
        .expect("query never reached the reranker");

    // While parked, independent requests still answer.
    let (health_status, _) = http_request(port, "GET", "/health", &[], None);
    assert_eq!(health_status, 200);
    let (mcp_status, body) = http_request(
        port,
        "POST",
        "/mcp",
        &[
            ("Content-Type", "application/json"),
            ("Accept", "application/json, text/event-stream"),
        ],
        Some(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"status","arguments":{}}}"#,
        ),
    );
    assert_eq!(mcp_status, 200, "{body}");
    assert!(
        body.contains(r#""result""#) && !body.contains(r#""error""#),
        "{body}"
    );
    let (nr_status, nr_body) = http_request(
        port,
        "POST",
        "/query",
        &[("Content-Type", "application/json")],
        Some(r#"{"searches":[{"type":"lex","query":"index"}],"rerank":false}"#),
    );
    assert_eq!(nr_status, 200, "{nr_body}");

    release.send(()).unwrap();
    let (parked_status, parked_body) = parked.join().expect("parked request thread");
    assert_eq!(parked_status, 200, "{parked_body}");

    let _shutdown_sent = shutdown.send(());
    task.await.unwrap().unwrap();
    assert!(
        TcpStream::connect(format!("localhost:{port}")).is_err(),
        "listener still open after shutdown"
    );
}

/// `options.allowedOrigins` wins over env and is enforced by the guard
/// middleware (origin-guard.ts:98). Env is not touched.
#[tokio::test(flavor = "multi_thread")]
async fn options_allowlist_overrides_env() {
    let (_dir, qmd, _entered, _release) = gated_store();
    let mut opts = HttpOptions::default();
    opts.port = 0;
    opts.quiet = true;
    opts.allowed_origins = Some(vec!["https://app.example".to_owned()]);
    let server = HttpServer::bind(qmd, opts).await.unwrap();
    let port = server.port();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(server.serve(async move {
        drop(rx.await);
    }));

    let (ok_status, _) = http_request(
        port,
        "GET",
        "/health",
        &[("Origin", "https://app.example")],
        None,
    );
    assert_eq!(ok_status, 200);
    let (denied_status, body) = http_request(
        port,
        "GET",
        "/health",
        &[("Origin", "https://evil.example")],
        None,
    );
    assert_eq!(denied_status, 403, "{body}");
    assert!(body.contains("-32003"), "{body}");

    let _shutdown_sent = tx.send(());
    task.await.unwrap().unwrap();
}

/// `handle.port` — the bound port is observable with `port: 0`
/// (upstream `HttpServerHandle.port`).
#[tokio::test(flavor = "multi_thread")]
async fn port_zero_reports_actual_port() {
    let (_dir, qmd, _entered, _release) = gated_store();
    let (port, task, shutdown) = http_server(qmd).await;
    assert_ne!(port, 0);
    let (status, _) = http_request(port, "GET", "/health", &[], None);
    assert_eq!(status, 200);
    let _shutdown_sent = shutdown.send(());
    task.await.unwrap().unwrap();
}
