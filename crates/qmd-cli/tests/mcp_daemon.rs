//! Process-level `qmd mcp --http --daemon` / `qmd mcp stop` tests
//! (P5.4, issue #95): daemon launch/stop/stale-pidfile/already-running,
//! `--index` scoping, the `qmd status` MCP line and the detached log —
//! ported from upstream `cli.test.ts` "mcp http daemon" plus the
//! foreign-PID guard from `mcp-pid.ts`. Upstream references:
//! `qmd.ts:4814-4888`, `mcp-pid.ts`.

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

use std::path::PathBuf;
use std::process::Command;
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

/// `<XDG_CACHE_HOME>/qmd/mcp[-<index>].{pid,log}` — `daemon_paths`.
fn daemon_paths(env: &CliEnv, index_name: &str) -> (PathBuf, PathBuf) {
    let suffix = if index_name == "index" {
        String::new()
    } else {
        format!("-{index_name}")
    };
    let dir = env.root.join("xdg-cache").join("qmd");
    (
        dir.join(format!("mcp{suffix}.pid")),
        dir.join(format!("mcp{suffix}.log")),
    )
}

/// Poll `GET /health` until 200 or the deadline — the daemon writes its
/// listening line to the log file, not the launcher's stderr, so the
/// port is the only readiness signal. A refused connection is
/// "not ready yet", not a failure — `http_request` panics on it.
#[allow(
    clippy::disallowed_methods,
    reason = "readiness polling needs real-time retries; no runtime in the test process"
)]
fn wait_health(port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        // `localhost` can bind ::1 (first resolved address, like Node's
        // `listen`) — let the resolver try every address.
        if std::net::TcpStream::connect(("localhost", port)).is_ok() {
            let reply = common::http_request(port, "GET", "/health", &serde_json::Map::new(), None);
            if reply.status == 200 {
                return true;
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}

/// The pid recorded in the pidfile, read with the same lenient parse
/// the implementation uses.
fn recorded_pid(path: &std::path::Path) -> i64 {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.split_whitespace().next()?.parse().ok())
        .unwrap_or(-1)
}

/// `ps -p <pid> -o ppid=` — the parent pid as a trimmed string.
fn ppid_of(pid: i64) -> String {
    let out = Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "ppid="])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).trim().to_owned()
}

/// `kill -0` liveness via `ps` (no signal sent — this only *checks*).
fn alive(pid: i64) -> bool {
    Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "pid="])
        .output()
        .is_ok_and(|o| o.status.success() && !String::from_utf8_lossy(&o.stdout).trim().is_empty())
}

/// `ps -p <pid> -o args=` — the process's full command line.
fn args_of(pid: i64) -> String {
    Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "args="])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default()
}

/// Best-effort SIGTERM for daemons a test launched — a failed assert
/// must not leak a setsid'd server. Drop only signals when the
/// recorded pid still looks like our `qmd` binary (a foreign process
/// that reused the pid is never touched); the guard is a no-op once
/// `mcp stop` removed the pidfile.
struct DaemonGuard {
    pidfiles: Vec<PathBuf>,
}

impl DaemonGuard {
    fn new(pidfiles: impl IntoIterator<Item = PathBuf>) -> Self {
        Self {
            pidfiles: pidfiles.into_iter().collect(),
        }
    }
}

impl Drop for DaemonGuard {
    fn drop(&mut self) {
        for path in &self.pidfiles {
            let pid = recorded_pid(path);
            if pid > 0 && args_of(pid).contains("qmd") {
                drop(Command::new("kill").arg(pid.to_string()).output());
            }
        }
    }
}

/// Full lifecycle (cli.test.ts "mcp http daemon" foreground+daemon
/// flow): launch → Started/Logs lines + exit 0 → pidfile + truncated
/// log → detached session (ppid 1) → serves traffic → `mcp stop` →
/// Stopped line + pidfile gone → port dead.
#[allow(
    clippy::disallowed_methods,
    reason = "readiness polling needs real-time retries; no runtime in the test process"
)]
#[test]
fn daemon_lifecycle_start_serve_stop() {
    let env = indexed_env();
    let port = common::free_port().to_string();
    let (pid_path, log_path) = daemon_paths(&env, "index");

    let out = env.run(&["mcp", "--http", "--daemon", "--port", &port]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let _guard = DaemonGuard::new([pid_path.clone()]);
    assert!(
        out.stdout
            .contains(&format!("Started on http://localhost:{port}/mcp (PID ")),
        "stdout: {}",
        out.stdout
    );
    assert!(
        out.stdout
            .contains(&format!("Logs: {}", log_path.display())),
        "stdout: {}",
        out.stdout
    );
    let pid = recorded_pid(&pid_path);
    assert!(pid > 0, "pidfile: {}", pid_path.display());
    assert!(wait_health(port.parse().unwrap(), Duration::from_secs(10)));
    // setsid detached it — init adopted the daemon once the launcher
    // exited (Node's `detached: true` equivalent).
    assert_eq!(ppid_of(pid), "1", "daemon should be reparented");
    // The child's request log went to the log file, not the launcher.
    let log = std::fs::read_to_string(&log_path).unwrap();
    assert!(log.contains("QMD MCP server listening"), "log: {log}");

    let stop = env.run(&["mcp", "stop"]);
    assert_eq!(stop.exit_code, 0, "stderr: {}", stop.stderr);
    assert!(
        stop.stdout
            .contains(&format!("Stopped QMD MCP server (PID {pid}).")),
        "stdout: {}",
        stop.stdout
    );
    assert!(!pid_path.exists(), "pidfile should be removed");
    // Give the graceful shutdown a beat, then the port must refuse.
    std::thread::sleep(Duration::from_millis(500));
    assert!(
        std::net::TcpStream::connect(format!("localhost:{port}")).is_err(),
        "daemon still serving after stop"
    );
}

/// A live pidfile refuses a second daemon (qmd.ts:4851-4856).
#[allow(
    clippy::disallowed_methods,
    reason = "readiness polling needs real-time retries; no runtime in the test process"
)]
#[test]
fn daemon_already_running_refuses() {
    let env = indexed_env();
    let port = common::free_port().to_string();
    let (pid_path, _) = daemon_paths(&env, "index");
    let out = env.run(&["mcp", "--http", "--daemon", "--port", &port]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let pid = recorded_pid(&pid_path);
    let _guard = DaemonGuard::new([pid_path]);
    assert!(wait_health(port.parse().unwrap(), Duration::from_secs(10)));

    let again = env.run(&["mcp", "--http", "--daemon", "--port", &port]);
    assert_eq!(again.exit_code, 1, "stdout: {}", again.stdout);
    assert!(
        again.stderr.contains(&format!(
            "Already running (PID {pid}). Run 'qmd mcp stop' first."
        )),
        "stderr: {}",
        again.stderr
    );

    let stop = env.run(&["mcp", "stop"]);
    assert_eq!(stop.exit_code, 0, "cleanup failed: {}", stop.stderr);
}

/// `--daemon` without `--http` is ignored like upstream — the flag is
/// only read inside `if (cli.values.http)` (qmd.ts:4839-4849), so a
/// plain `qmd mcp --daemon` starts stdio, writes no pidfile, and exits
/// on stdin EOF.
#[test]
fn daemon_flag_without_http_is_ignored() {
    let env = CliEnv::new();
    let out = env.run(&["mcp", "--daemon"]); // stdin is null → EOF
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let (pid_path, _) = daemon_paths(&env, "index");
    assert!(!pid_path.exists(), "no daemon should have spawned");
}

/// `mcp stop` wins over `--http` — upstream checks `cli.args[0]` before
/// the flag (qmd.ts:4819).
#[test]
fn mcp_stop_beats_http_flag() {
    let env = CliEnv::new();
    let out = env.run(&["mcp", "stop", "--http"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("Not running (no PID file)."),
        "stdout: {}",
        out.stdout
    );
}

/// `mcp stop` without a pidfile (qmd.ts:4821-4824).
#[test]
fn mcp_stop_without_pidfile() {
    let env = CliEnv::new();
    let out = env.run(&["mcp", "stop"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("Not running (no PID file)."),
        "stdout: {}",
        out.stdout
    );
}

/// A pidfile whose pid is dead or unparseable is stale — removed, exit
/// 0, upstream's cleanup line (qmd.ts:4826-4831).
#[test]
fn mcp_stop_cleans_stale_pidfile() {
    let env = CliEnv::new();
    let (pid_path, _) = daemon_paths(&env, "index");
    std::fs::create_dir_all(pid_path.parent().unwrap()).unwrap();
    for contents in ["999999999", "not-a-pid"] {
        std::fs::write(&pid_path, contents).unwrap();
        let out = env.run(&["mcp", "stop"]);
        assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
        assert!(
            out.stdout
                .contains("Cleaned up stale PID file (server was not running)."),
            "stdout: {} (contents {contents:?})",
            out.stdout
        );
        assert!(!pid_path.exists());
    }
}

/// The cmdline identity check must protect a *live* foreign PID —
/// a recycled pidfile never gets signalled (mcp-pid.ts).
#[test]
fn mcp_stop_never_signals_foreign_process() {
    let env = CliEnv::new();
    let (pid_path, _) = daemon_paths(&env, "index");
    std::fs::create_dir_all(pid_path.parent().unwrap()).unwrap();
    let mut sleeper = Command::new("/bin/sleep").arg("60").spawn().unwrap();
    std::fs::write(&pid_path, sleeper.id().to_string()).unwrap();

    let out = env.run(&["mcp", "stop"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout
            .contains("Cleaned up stale PID file (server was not running)."),
        "stdout: {}",
        out.stdout
    );
    assert!(!pid_path.exists());
    assert!(
        alive(i64::from(sleeper.id())),
        "foreign process was signalled"
    );
    drop(sleeper.kill());
    drop(sleeper.wait());
}

/// `--index <name>` scopes the state files (upstream #772): a named
/// daemon writes `mcp-<name>.pid`, leaves `mcp.pid` absent, serves the
/// named index, and `mcp stop` under the same `--index` kills it.
#[allow(
    clippy::disallowed_methods,
    reason = "readiness polling needs real-time retries; no runtime in the test process"
)]
#[test]
fn daemon_named_index_scopes_state_files() {
    let env = CliEnv::new();
    fixtures_corpus(&env);
    env.write_config("mcpalt", &collection_yaml("docs", &env.fixtures, "**/*.md"));
    env.reindex("mcpalt", "mcpalt", false);
    let port = common::free_port().to_string();
    let (named_pid, _) = daemon_paths(&env, "mcpalt");
    let (default_pid, _) = daemon_paths(&env, "index");

    let mut launch = env.qmd_command(
        &[
            "--index", "mcpalt", "mcp", "--http", "--daemon", "--port", &port,
        ],
        false,
        &[],
    );
    let out = launch.output().unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(&format!("http://localhost:{port}/mcp")),
        "stdout: {stdout}"
    );
    assert!(named_pid.exists(), "named pidfile missing");
    assert!(!default_pid.exists(), "default pidfile must not appear");
    let _guard = DaemonGuard::new([named_pid.clone()]);
    let pid = recorded_pid(&named_pid);
    assert!(wait_health(port.parse().unwrap(), Duration::from_secs(10)));

    let mut stop = env.qmd_command(&["--index", "mcpalt", "mcp", "stop"], false, &[]);
    let stopped = stop.output().unwrap();
    assert!(
        stopped.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&stopped.stderr)
    );
    let stopped_out = String::from_utf8_lossy(&stopped.stdout);
    assert!(
        stopped_out.contains(&format!("Stopped QMD MCP server (PID {pid}).")),
        "stdout: {stopped_out}"
    );
    assert!(!named_pid.exists());
}

/// `qmd status` silently removes a pidfile whose contents do not name
/// a live qmd process — including unparseable garbage (qmd.ts:554-560:
/// `existsSync` gates the check, `parseInt` → NaN → `unlinkSync`).
#[test]
fn status_cleans_stale_pidfile() {
    let env = indexed_env();
    let (pid_path, _) = daemon_paths(&env, "index");
    std::fs::create_dir_all(pid_path.parent().unwrap()).unwrap();
    for contents in ["not-a-pid", "999999999"] {
        std::fs::write(&pid_path, contents).unwrap();
        let status = env.run(&["status"]);
        assert_eq!(status.exit_code, 0, "stderr: {}", status.stderr);
        assert!(!status.stdout.contains("MCP:"), "stdout: {}", status.stdout);
        assert!(
            !pid_path.exists(),
            "stale pidfile ({contents:?}) left behind"
        );
    }
}

/// `qmd status` prints `MCP:   running (PID <n>)` while the daemon is
/// alive (qmd.ts:552-560).
#[allow(
    clippy::disallowed_methods,
    reason = "readiness polling needs real-time retries; no runtime in the test process"
)]
#[test]
fn status_reports_running_daemon() {
    let env = indexed_env();
    let port = common::free_port().to_string();
    let (pid_path, _) = daemon_paths(&env, "index");
    let out = env.run(&["mcp", "--http", "--daemon", "--port", &port]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let pid = recorded_pid(&pid_path);
    let _guard = DaemonGuard::new([pid_path]);
    assert!(wait_health(port.parse().unwrap(), Duration::from_secs(10)));

    let status = env.run(&["status"]);
    assert_eq!(status.exit_code, 0, "stderr: {}", status.stderr);
    assert!(
        status
            .stdout
            .contains(&format!("MCP:   running (PID {pid})")),
        "stdout: {}",
        status.stdout
    );

    let stop = env.run(&["mcp", "stop"]);
    assert_eq!(stop.exit_code, 0, "cleanup failed: {}", stop.stderr);

    // After stop the line is gone entirely (no "stopped" state).
    let after = env.run(&["status"]);
    assert_eq!(after.exit_code, 0, "stderr: {}", after.stderr);
    assert!(!after.stdout.contains("MCP:"), "stdout: {}", after.stdout);
}

/// `--index <name>` daemon serves that named index (cli.test.ts
/// "mcp http daemon" ~:2630): the named corpus carries a term absent
/// from the default corpus, and the daemon's REST /query finds it.
#[test]
fn daemon_serves_named_index_data() {
    let env = CliEnv::new();
    // Default index — the shared corpus (no unique term).
    fixtures_corpus(&env);
    env.write_config("index", &collection_yaml("docs", &env.fixtures, "**/*.md"));
    env.update();
    // Named index — a separate corpus with a unique searchable term.
    let alt = env.root.join("alt-corpus");
    std::fs::create_dir_all(&alt).unwrap();
    std::fs::write(alt.join("u.md"), "# U\n\nqwixzilonic term\n").unwrap();
    env.write_config("alt", &collection_yaml("docs", &alt, "**/*.md"));
    env.reindex("alt", "alt", false);

    let port = common::free_port().to_string();
    let out = env
        .qmd_command(
            &[
                "--index", "alt", "mcp", "--http", "--daemon", "--port", &port,
            ],
            false,
            &[],
        )
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let _guard = DaemonGuard::new([daemon_paths(&env, "alt").0]);
    assert!(wait_health(port.parse().unwrap(), Duration::from_secs(10)));

    // The named daemon answers the named index's term.
    let reply = common::http_request(
        port.parse().unwrap(),
        "POST",
        "/query",
        &serde_json::Map::new(),
        Some(r#"{"searches":[{"type":"lex","query":"qwixzilonic"}],"rerank":false}"#),
    );
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert!(
        reply.body.contains("qwixzilonic") || reply.body.contains("u.md"),
        "{}",
        reply.body
    );

    let stopped = env
        .qmd_command(&["--index", "alt", "mcp", "stop"], false, &[])
        .output()
        .unwrap();
    let stop_out = String::from_utf8_lossy(&stopped.stdout);
    assert!(
        stopped.status.success() && stop_out.contains("Stopped QMD MCP server (PID "),
        "named daemon not stopped: {stop_out}"
    );
}

/// Named and default daemons run side by side on different ports
/// (cli.test.ts ~:2785): both serve /health, each `mcp stop` only
/// kills its own pidfile and port.
#[test]
fn named_and_default_daemons_coexist() {
    let env = CliEnv::new();
    fixtures_corpus(&env);
    env.write_config("index", &collection_yaml("docs", &env.fixtures, "**/*.md"));
    env.update();
    let alt = env.root.join("alt-corpus");
    std::fs::create_dir_all(&alt).unwrap();
    std::fs::write(alt.join("u.md"), "# U\n\nalt corpus doc\n").unwrap();
    env.write_config("alt", &collection_yaml("docs", &alt, "**/*.md"));
    env.reindex("alt", "alt", false);

    let port_a = common::free_port().to_string();
    let port_b = common::free_port().to_string();
    let (pid_a, _) = daemon_paths(&env, "index");
    let (pid_b, _) = daemon_paths(&env, "alt");

    let out_a = env
        .qmd_command(
            &["mcp", "--http", "--daemon", "--port", &port_a],
            false,
            &[],
        )
        .output()
        .unwrap();
    assert!(out_a.status.success());
    let out_b = env
        .qmd_command(
            &[
                "--index", "alt", "mcp", "--http", "--daemon", "--port", &port_b,
            ],
            false,
            &[],
        )
        .output()
        .unwrap();
    assert!(out_b.status.success());
    let _guard = DaemonGuard::new([pid_a.clone(), pid_b.clone()]);
    assert!(wait_health(
        port_a.parse().unwrap(),
        Duration::from_secs(10)
    ));
    assert!(wait_health(
        port_b.parse().unwrap(),
        Duration::from_secs(10)
    ));
    assert!(pid_a.exists() && pid_b.exists());
    assert_ne!(recorded_pid(&pid_a), recorded_pid(&pid_b));

    // Stopping the named daemon leaves the default one serving.
    let stopped_b = env
        .qmd_command(&["--index", "alt", "mcp", "stop"], false, &[])
        .output()
        .unwrap();
    assert!(
        stopped_b.status.success()
            && String::from_utf8_lossy(&stopped_b.stdout).contains("Stopped QMD MCP server (PID "),
        "stopped_b: {}",
        String::from_utf8_lossy(&stopped_b.stdout)
    );
    assert!(!pid_b.exists());
    let health_a = common::http_request(
        port_a.parse().unwrap(),
        "GET",
        "/health",
        &serde_json::Map::new(),
        None,
    );
    assert_eq!(health_a.status, 200);

    let stopped_a = env
        .qmd_command(&["mcp", "stop"], false, &[])
        .output()
        .unwrap();
    assert!(
        stopped_a.status.success()
            && String::from_utf8_lossy(&stopped_a.stdout).contains("Stopped QMD MCP server (PID "),
        "stopped_a: {}",
        String::from_utf8_lossy(&stopped_a.stdout)
    );
    assert!(!pid_a.exists());
}

/// `--daemon` with a stale pidfile (dead pid) starts and rewrites the
/// pidfile (cli.test.ts ~:2920).
#[test]
fn daemon_starts_over_stale_pidfile() {
    let env = indexed_env();
    let (pid_path, _) = daemon_paths(&env, "index");
    std::fs::create_dir_all(pid_path.parent().unwrap()).unwrap();
    std::fs::write(&pid_path, "999999999\n").unwrap(); // dead-pid contents; writes the same text upstream records

    let port = common::free_port().to_string();
    let out = env
        .qmd_command(&["mcp", "--http", "--daemon", "--port", &port], false, &[])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let _guard = DaemonGuard::new([pid_path.clone()]);
    assert!(wait_health(port.parse().unwrap(), Duration::from_secs(10)));
    let pid = recorded_pid(&pid_path);
    assert!(
        pid > 0 && pid != 999_999_999,
        "pidfile not rewritten: {pid}"
    );

    let stopped = env
        .qmd_command(&["mcp", "stop"], false, &[])
        .output()
        .unwrap();
    assert!(stopped.status.success());
}

/// `--daemon` with a *live foreign* pid in the pidfile starts anyway
/// (the identity check refuses to signal a non-qmd process → treated
/// like a stale file, cli.test.ts ~:2964) — and the foreign process
/// is still alive afterwards.
#[test]
fn daemon_starts_over_foreign_pidfile() {
    let env = indexed_env();
    let (pid_path, _) = daemon_paths(&env, "index");
    std::fs::create_dir_all(pid_path.parent().unwrap()).unwrap();
    let mut sleeper = Command::new("sleep").arg("300").spawn().unwrap();
    let sleeper_pid = i64::from(sleeper.id().cast_signed());
    std::fs::write(&pid_path, format!("{sleeper_pid}\n")).unwrap();

    let port = common::free_port().to_string();
    let out = env
        .qmd_command(&["mcp", "--http", "--daemon", "--port", &port], false, &[])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(wait_health(port.parse().unwrap(), Duration::from_secs(10)));
    let _guard = DaemonGuard::new([pid_path.clone()]);
    let daemon_pid = recorded_pid(&pid_path);
    assert!(
        daemon_pid > 0 && daemon_pid != sleeper_pid,
        "pidfile not rewritten: {daemon_pid}"
    );

    // The foreign process was never signalled.
    assert!(alive(sleeper_pid), "foreign process was killed");
    drop(sleeper.kill());
    drop(sleeper.wait());

    let stopped = env
        .qmd_command(&["mcp", "stop"], false, &[])
        .output()
        .unwrap();
    assert!(stopped.status.success());
    let stopped_out = String::from_utf8_lossy(&stopped.stdout);
    assert!(
        stopped_out.contains(&format!("Stopped QMD MCP server (PID {daemon_pid}).")),
        "stdout: {stopped_out}"
    );
}
