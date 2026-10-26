//! Process-level `qmd mcp --http` tests (P5.3, issue #90): bind
//! errors, signal lifecycle, `--index`-prefixed dispatch, host/port
//! flags, the wildcard-bind warning and parallel load — everything the
//! golden replay in `mcp_golden.rs` can't cover. Upstream references:
//! `qmd.ts:4841-4917` (flags, `EADDRINUSE`), `server.ts:1164-1181`
//! (signal handlers, startup warnings) and `mcp.test.ts`
//! "MCP HTTP Transport".

#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::missing_assert_message,
    clippy::panic,
    reason = "idiomatic test-code patterns"
)]

mod common;

use std::io::{BufRead as _, BufReader};
use std::process::{Child, Command, ExitStatus};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use common::{CliEnv, collection_yaml, fixtures_corpus};

/// Corpus + `index.yml` + populated `INDEX_PATH` index.
fn indexed_env() -> CliEnv {
    let env = CliEnv::new();
    fixtures_corpus(&env);
    env.write_config("index", &collection_yaml("docs", &env.fixtures, "**/*.md"));
    env.update();
    env
}

/// Header map builder for [`common::http_request`].
fn hdrs(pairs: &[(&str, &str)]) -> serde_json::Map<String, serde_json::Value> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), serde_json::Value::String((*v).to_owned())))
        .collect()
}

/// `kill(1)` by pid — same approach the golden replay uses for SIGTERM.
fn signal(pid: u32, sig: &str) {
    let out = Command::new("kill")
        .args([sig, &pid.to_string()])
        .output()
        .unwrap();
    assert!(out.status.success(), "kill {sig} failed");
}

/// An `mcp --http` subprocess: stderr streams over a channel, the port
/// is read off the `listening on` line, and the child is `SIGKILLed` on
/// drop if still running (graceful exits are what the tests assert).
struct Server {
    child: Option<Child>,
    rx: Receiver<String>,
    /// stderr up to and including the listening line.
    seen: String,
    /// Port parsed out of that line.
    port: u16,
}

impl Server {
    /// Spawn `qmd <args>` and block until the listening line or a
    /// 30-second deadline.
    fn spawn(env: &CliEnv, args: &[&str], index_path: bool, extra: &[(&str, &str)]) -> Self {
        let mut child = env.qmd_command(args, index_path, extra).spawn().unwrap();
        let (tx, rx) = channel::<String>();
        let pipe = child.stderr.take().unwrap();
        std::thread::spawn(move || {
            let mut r = BufReader::new(pipe);
            let mut line = String::new();
            while r.read_line(&mut line).unwrap_or(0) > 0 {
                drop(tx.send(std::mem::take(&mut line)));
            }
        });
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut seen = String::new();
        let port = loop {
            let line = rx
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .unwrap_or_else(|e| {
                    panic!("server never printed its listening line ({e}); stderr:\n{seen}")
                });
            seen.push_str(&line);
            if let Some(p) = listening_port(&line) {
                break p;
            }
        };
        Self {
            child: Some(child),
            rx,
            seen,
            port,
        }
    }

    fn id(&self) -> u32 {
        self.child.as_ref().unwrap().id()
    }

    /// stderr lines buffered so far (non-blocking).
    fn drain(&self) -> String {
        let mut s = self.seen.clone();
        while let Ok(l) = self.rx.try_recv() {
            s.push_str(&l);
        }
        s
    }

    /// Wait for exit and collect the rest of stderr.
    fn wait(mut self) -> (ExitStatus, String) {
        let status = self.child.take().unwrap().wait().unwrap();
        let mut err = self.seen.clone();
        while let Ok(l) = self.rx.recv() {
            err.push_str(&l);
        }
        (status, err)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(c) = &mut self.child {
            drop(
                Command::new("kill")
                    .args(["-KILL", &c.id().to_string()])
                    .output(),
            );
            drop(c.wait());
        }
    }
}

/// Port out of `QMD MCP server listening on http://<host>:<port>/mcp`.
fn listening_port(line: &str) -> Option<u16> {
    let rest = line
        .trim_end()
        .strip_prefix("QMD MCP server listening on http://")?;
    rest.strip_suffix("/mcp")?.rsplit(':').next()?.parse().ok()
}

/// `Port <n> already in use` — qmd.ts:4908-4911 translates EADDRINUSE;
/// a held `127.0.0.1` listener collides deterministically now that the
/// bind uses the first resolved address like Node's `listen`.
#[test]
fn http_port_in_use_is_a_clean_error() {
    let env = indexed_env();
    let blocker = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = blocker.local_addr().unwrap().port().to_string();
    let out = env.run(&[
        "mcp",
        "--http",
        "--host",
        "127.0.0.1",
        "--port",
        port.as_str(),
    ]);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr
            .contains("already in use. Try a different port with --port."),
        "stderr: {}",
        out.stderr
    );
}

/// `Number(v) || 8181` — out-of-range values are rejected before the
/// store opens (`Invalid port` text is ours: upstream's node throws
/// `ERR_SOCKET_BAD_PORT`; the rejection itself is the parity point).
#[test]
fn http_out_of_range_port_rejected() {
    let env = CliEnv::new();
    let out = env.run(&["mcp", "--http", "--port", "70000"]);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr.contains("Invalid port: 70000"),
        "stderr: {}",
        out.stderr
    );
}

/// Positionals are rejected until P5.4 lands `stop` — `qmd mcp stop`
/// must fail instead of silently starting stdio like upstream.
#[test]
fn http_unknown_subcommand_rejected() {
    let env = CliEnv::new();
    let out = env.run(&["mcp", "stop"]);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stderr.contains("Unknown subcommand: stop"),
        "stderr: {}",
        out.stderr
    );
}

/// `Shutting down (SIGINT)...` + exit 0 (server.ts:1169-1173).
#[test]
fn http_sigint_shuts_down_cleanly() {
    let env = indexed_env();
    let port = common::free_port().to_string();
    let server = Server::spawn(&env, &["mcp", "--http", "--port", &port], true, &[]);
    signal(server.id(), "-INT");
    let (status, err) = server.wait();
    assert_eq!(status.code(), Some(0), "exit: {status}");
    assert!(err.contains("Shutting down (SIGINT)..."), "stderr: {err}");
}

/// The CLI signal watcher is skipped based on the PARSED command, not
/// argv[1]: `qmd --index <n> mcp --http` is still MCP, so SIGTERM must
/// hit the server's own handler (exit 0), not the generic watcher.
#[test]
fn http_index_prefixed_mcp_stays_http() {
    let env = CliEnv::new();
    fixtures_corpus(&env);
    env.write_config(
        "testidx",
        &collection_yaml("docs", &env.fixtures, "**/*.md"),
    );
    env.reindex("testidx", "testidx", false);
    let port = common::free_port().to_string();
    let server = Server::spawn(
        &env,
        &["--index", "testidx", "mcp", "--http", "--port", &port],
        false,
        &[],
    );
    assert_eq!(
        common::http_request(server.port, "GET", "/health", &hdrs(&[]), None).status,
        200
    );
    signal(server.id(), "-TERM");
    let (status, err) = server.wait();
    assert_eq!(status.code(), Some(0), "exit: {status}");
    assert!(err.contains("Shutting down (SIGTERM)..."), "stderr: {err}");
}

/// `--host 127.0.0.1 --port <n>` lands in the listening line; a
/// loopback bind enforces Host so neither guard warning prints
/// (server.ts:1175-1180).
#[allow(
    clippy::disallowed_methods,
    reason = "sync test asserting a warning is absent — it needs a fixed real-time window for the line to have had a chance to arrive"
)]
#[test]
fn http_custom_host_and_port() {
    let env = indexed_env();
    let port = common::free_port().to_string();
    let server = Server::spawn(
        &env,
        &["mcp", "--http", "--host", "127.0.0.1", "--port", &port],
        true,
        &[],
    );
    assert!(
        server.seen.contains(&format!(
            "QMD MCP server listening on http://127.0.0.1:{}/mcp",
            server.port
        )),
        "stderr: {}",
        server.seen
    );
    // Give the warning lines (which would print right after the
    // listening line) a brief window to arrive, then assert absence.
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        !server.drain().contains("Warning:"),
        "stderr: {}",
        server.seen
    );
    assert_eq!(
        common::http_request(server.port, "GET", "/health", &hdrs(&[]), None).status,
        200
    );
}

/// `QMD_HOST` supplies the bind host when `--host` is absent
/// (server.ts:977).
#[test]
fn http_qmd_host_env_var() {
    let env = indexed_env();
    let port = common::free_port().to_string();
    let server = Server::spawn(
        &env,
        &["mcp", "--http", "--port", &port],
        true,
        &[("QMD_HOST", "127.0.0.1")],
    );
    assert!(
        server.seen.contains("http://127.0.0.1:"),
        "stderr: {}",
        server.seen
    );
}

/// `0.0.0.0` without `QMD_ALLOWED_HOSTS` prints the upstream warning
/// and Host validation is off — a foreign `Host` reaches the handler
/// (upstream's documented tradeoff, server.ts:1178-1180).
#[allow(
    clippy::disallowed_methods,
    reason = "sync test waiting for the post-listen warning line — a brief real-time window, no runtime in the test process"
)]
#[test]
fn http_wildcard_bind_warns_and_skips_host_check() {
    let env = indexed_env();
    let port = common::free_port().to_string();
    let server = Server::spawn(
        &env,
        &["mcp", "--http", "--host", "0.0.0.0", "--port", &port],
        true,
        &[],
    );
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        server.drain().contains(
            "Warning: bound to 0.0.0.0 with no QMD_ALLOWED_HOSTS — Host validation is off"
        ),
        "stderr: {}",
        server.seen
    );
    let reply = common::http_request(
        server.port,
        "GET",
        "/health",
        &hdrs(&[("Host", "evil.example")]),
        None,
    );
    assert_eq!(reply.status, 200, "foreign Host should reach /health");
}

/// Parallel load: eight threads each hit /health, /mcp (initialize)
/// and /query on the shared `Arc<Qmd>` — the multi-thread runtime and
/// `spawn_blocking` searches must not deadlock or error.
#[test]
fn http_concurrent_requests_all_succeed() {
    let env = indexed_env();
    let port = common::free_port().to_string();
    let server = Server::spawn(&env, &["mcp", "--http", "--port", &port], true, &[]);
    let bound = server.port;
    let mut handles = Vec::new();
    for i in 0..8 {
        handles.push(std::thread::spawn(move || {
            let health = common::http_request(bound, "GET", "/health", &hdrs(&[]), None);
            assert_eq!(health.status, 200, "health #{i}");
            let init = common::http_request(
                bound,
                "POST",
                "/mcp",
                &hdrs(&[(
                    "Accept",
                    "application/json, text/event-stream",
                )]),
                Some(
                    r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}"#,
                ),
            );
            assert_eq!(init.status, 200, "initialize #{i}");
            let query = common::http_request(
                bound,
                "POST",
                "/query",
                &hdrs(&[]),
                Some(r#"{"searches":[{"type":"lex","query":"test"}],"rerank":false}"#),
            );
            assert_eq!(query.status, 200, "query #{i}");
            assert!(query.body.contains("\"results\""), "query #{i} body");
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

/// `params.filter !== undefined` is the only exemption — a JSON `null`
/// filter still fails the object check → 400 (server.ts:1084-1087).
#[test]
fn http_rest_filter_null_is_a_400() {
    let env = indexed_env();
    let port = common::free_port().to_string();
    let server = Server::spawn(&env, &["mcp", "--http", "--port", &port], true, &[]);
    let reply = common::http_request(
        server.port,
        "POST",
        "/query",
        &hdrs(&[]),
        Some(r#"{"searches":[{"type":"lex","query":"test"}],"filter":null}"#),
    );
    assert_eq!(reply.status, 400, "body: {}", reply.body);
    assert!(
        reply.body.contains("Invalid field: filter"),
        "body: {}",
        reply.body
    );
}

/// `search.line ? "Line n" : "Structured search"` — a falsy `line`
/// (0 / null) must not appear in the error location
/// (store.ts:5981-5983, JSC `TypeError` wording on `[null]`).
#[allow(
    clippy::disallowed_methods,
    reason = "sync test waiting for the handler-error stderr line — a brief real-time window, no runtime in the test process"
)]
#[test]
fn http_rest_error_locations_match_upstream() {
    let env = indexed_env();
    let port = common::free_port().to_string();
    let server = Server::spawn(&env, &["mcp", "--http", "--port", &port], true, &[]);
    let newline = common::http_request(
        server.port,
        "POST",
        "/query",
        &hdrs(&[]),
        Some(r#"{"searches":[{"type":"lex","query":"a\nb","line":0}]}"#),
    );
    assert_eq!(newline.status, 500, "body: {}", newline.body);
    let null_elem = common::http_request(
        server.port,
        "POST",
        "/query",
        &hdrs(&[]),
        Some(r#"{"searches":[null]}"#),
    );
    assert_eq!(null_elem.status, 500, "body: {}", null_elem.body);
    std::thread::sleep(Duration::from_millis(300));
    let err = server.drain();
    assert!(
        err.contains("HTTP handler error: Structured search (lex): queries must be single-line."),
        "stderr: {err}"
    );
    assert!(
        err.contains("HTTP handler error: TypeError: null is not an object (evaluating 's.type')"),
        "stderr: {err}"
    );
}

/// GET/HEAD skip the body entirely upstream (server.ts:1106-1108) — a
/// GET carrying a payload over the cap is still a 405, not a 413.
#[test]
fn http_get_mcp_ignores_body() {
    let env = indexed_env();
    let port = common::free_port().to_string();
    let server = Server::spawn(&env, &["mcp", "--http", "--port", &port], true, &[]);
    let big = "x".repeat(5 * 1024 * 1024);
    let reply = common::http_request(server.port, "GET", "/mcp", &hdrs(&[]), Some(big.as_str()));
    assert_eq!(reply.status, 405, "body: {}", reply.body);
    assert!(
        reply.body.contains("Method not allowed."),
        "body: {}",
        reply.body
    );
}

/// The 4 MiB deviation cap applies to `/query` with the same shape as
/// `/mcp` — not axum's smaller plain-text extractor limit.
#[test]
fn http_rest_body_cap_matches_mcp() {
    let env = indexed_env();
    let port = common::free_port().to_string();
    let server = Server::spawn(&env, &["mcp", "--http", "--port", &port], true, &[]);
    let mut body = String::from(r#"{"searches":[{"type":"lex","query":""#);
    body.push_str(&"x".repeat(5 * 1024 * 1024));
    body.push_str(r#""}]}"#);
    let reply = common::http_request(
        server.port,
        "POST",
        "/query",
        &hdrs(&[]),
        Some(body.as_str()),
    );
    assert_eq!(reply.status, 413, "body: {}", reply.body);
    assert_eq!(reply.body, "Payload Too Large");
}

/// rmcp picks the response era with a `>= "2026-07-28"` string compare
/// on the peer version — an unrecognized `MCP-Protocol-Version` must be
/// clamped to a known legacy version or the 2026 envelope leaks into
/// the legacy SSE frame.
#[test]
fn http_bogus_version_header_stays_legacy() {
    let env = indexed_env();
    let port = common::free_port().to_string();
    let server = Server::spawn(&env, &["mcp", "--http", "--port", &port], true, &[]);
    let reply = common::http_request(
        server.port,
        "POST",
        "/mcp",
        &hdrs(&[
            ("Accept", "application/json, text/event-stream"),
            ("MCP-Protocol-Version", "bogus"),
        ]),
        Some(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#),
    );
    assert_eq!(reply.status, 200, "body: {}", reply.body);
    assert!(
        reply.body.starts_with("event: message\ndata: "),
        "legacy SSE frame expected: {}",
        reply.body
    );
    assert!(
        !reply.body.contains("resultType"),
        "2026 envelope leaked into a legacy frame: {}",
        reply.body
    );
}
