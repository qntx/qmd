#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "idiomatic test-code patterns"
)]

//! Subprocess stdio lifecycle — upstream `mcp-stdio-lifecycle.test.ts`
//! cases that need a real `qmd mcp` binary: (:86) a second stdin EOF
//! must not re-trigger shutdown logging, and (:188) stderr writes that
//! fail (EPIPE — the parent dropped the read end) do not block the
//! JSON-RPC response or clean shutdown.

mod common;

use std::io::{Read as _, Write as _};
use std::process::Stdio;
use std::time::Duration;

use common::{CliEnv, collection_yaml, fixtures_corpus};

fn indexed_env() -> CliEnv {
    let env = CliEnv::new();
    fixtures_corpus(&env);
    env.write_config("index", &collection_yaml("docs", &env.fixtures, "**/*.md"));
    env.update();
    env
}

/// Read a pipe with a hard deadline so a hung child can't wedge the
/// suite (upstream relies on vitest's test timeout instead).
#[allow(
    clippy::disallowed_methods,
    reason = "wait polling needs real-time retries; no runtime in the test process"
)]
fn read_with_deadline(
    mut child: std::process::Child,
    timeout: Duration,
) -> (Option<i32>, String, String) {
    let mut stdout_pipe = child.stdout.take().unwrap();
    let mut stderr_pipe = child.stderr.take().unwrap();
    let out_thread = std::thread::spawn(move || {
        let mut s = String::new();
        drop(stdout_pipe.read_to_string(&mut s));
        s
    });
    let err_thread = std::thread::spawn(move || {
        let mut s = String::new();
        drop(stderr_pipe.read_to_string(&mut s));
        s
    });
    let deadline = std::time::Instant::now() + timeout;
    let status = loop {
        match child.try_wait().unwrap() {
            Some(s) => break Some(s),
            None if std::time::Instant::now() > deadline => {
                drop(child.kill());
                break None;
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    };
    (
        status.map(|s| s.code().unwrap_or(-1)),
        out_thread.join().unwrap(),
        err_thread.join().unwrap(),
    )
}

/// EOF → exactly one `Shutting down (stdin closed)...` on stderr,
/// exit 0 (mcp-stdio-lifecycle.test.ts:86 — the EOF path is idempotent;
/// upstream asserts a double-EOF does not double-log).
#[test]
fn stdio_eof_breadcrumb_prints_once() {
    let env = indexed_env();
    let mut child = env.qmd_command(&["mcp"], false, &[]);
    let mut spawned = child.spawn().unwrap();
    // One answered call so the server is mid-session, then EOF.
    spawned
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},\"clientInfo\":{\"name\":\"t\",\"version\":\"1\"}}}\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"status\",\"arguments\":{}}}\n")
        .unwrap();
    drop(spawned.stdin.take());
    let (code, _stdout, stderr) = read_with_deadline(spawned, Duration::from_secs(15));
    assert_eq!(code, Some(0), "stderr: {stderr}");
    let count = stderr.matches("Shutting down (stdin closed)...").count();
    assert_eq!(count, 1, "stderr: {stderr}");
}

/// stderr EPIPE (parent dropped the read end) — the server still
/// answers on stdout and exits 0 on EOF
/// (mcp-stdio-lifecycle.test.ts:188 — a throwing stderr must not
/// block shutdown).
#[test]
#[allow(
    clippy::disallowed_methods,
    reason = "wait polling needs real-time retries; no runtime in the test process"
)]
fn stdio_stderr_epipe_still_answers_and_exits() {
    let env = indexed_env();
    let mut child = env.qmd_command(&["mcp"], false, &[]);
    child.stderr(Stdio::piped());
    let mut spawned = child.spawn().unwrap();
    // Dropping the parent's read end makes every stderr write hit EPIPE.
    drop(spawned.stderr.take());
    spawned
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},\"clientInfo\":{\"name\":\"t\",\"version\":\"1\"}}}\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"status\",\"arguments\":{}}}\n")
        .unwrap();
    drop(spawned.stdin.take());
    let mut stdout_pipe = spawned.stdout.take().unwrap();
    let mut stdout = String::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let code = loop {
        match spawned.try_wait().unwrap() {
            Some(s) => break s.code().unwrap_or(-1),
            None if std::time::Instant::now() > deadline => {
                drop(spawned.kill());
                panic!("child hung after EPIPE+EOF");
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    };
    drop(stdout_pipe.read_to_string(&mut stdout));
    assert_eq!(code, 0);
    assert!(
        stdout.contains("\"id\":1") && stdout.contains("QMD Index Status"),
        "stdout: {stdout}"
    );
}
