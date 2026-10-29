//! Golden replay of upstream `qmd mcp` stdio sessions (P5.2).
//! Plain auto-expand `query` needs the generator model and is covered
//! only by the `#[ignore]` real-model test at the bottom of this file.
//! `fixtures/mcp_golden.json` is recorded by `scripts/gen_mcp_golden.ts`
//! against `tobi/qmd` v2.8.3; responses are matched to requests by JSON
//!-RPC id (dispatch is concurrent, order may differ) and compared as
//! `serde_json::Value` — key-order-insensitive (deviation: serde emits
//! fields in declaration order). Also covers upstream
//! `cli.test.ts` `mcp stdio launcher` — every stdout line must parse as
//! JSON-RPC — and `mcp-stdio-lifecycle.test.ts`'s real-process EOF
//! (exit 0, `Shutting down (stdin closed)...` on stderr).

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
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    reason = "idiomatic test-code patterns"
)]

mod common;

use std::collections::BTreeMap;

use common::CliEnv;
use serde_json::Value;

/// Build the recorded corpus + config under a fresh `CliEnv` and index
/// it through the library.
fn golden_env(fixture: &Value) -> CliEnv {
    let env = CliEnv::new();
    let corpus = &fixture["corpus"];
    for f in corpus["files"].as_array().unwrap() {
        let path = env.write_file(f["path"].as_str().unwrap(), f["content"].as_str().unwrap());
        // The recorded mtimes pin `lastUpdated` ordering.
        let mtime =
            std::time::UNIX_EPOCH + std::time::Duration::from_secs(f["mtime"].as_u64().unwrap());
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(mtime)
            .unwrap();
    }
    env.write_config(
        "index",
        &corpus["config_yaml"]
            .as_str()
            .unwrap()
            .replace("<ROOT>", &env.fixtures.to_string_lossy()),
    );
    env.update();
    env
}

/// Normalize the only fields that cannot be byte-stable across
/// processes/versions: this run's root path → `<ROOT>` (matching the
/// record side), `serverInfo.version` → `<V>` (ours is the crate
/// version; upstream's is its package version), `/health` `uptime` →
/// `<U>` and integral floats → ints (JSON.stringify writes `1` for
/// `1.0`, `serde_json` writes `1.0`).
fn normalize(v: &mut Value, our_root: &str) {
    match v {
        Value::Object(map) => {
            let is_server_info =
                map.get("name").is_some_and(|n| n == "qmd") && map.contains_key("version");
            for (k, x) in map.iter_mut() {
                if is_server_info && k == "version" {
                    *x = Value::from("<V>");
                } else if k == "uptime" {
                    *x = Value::from("<U>");
                } else if let Value::String(s) = x {
                    *x = Value::from(s.replace(our_root, "<ROOT>/corpus"));
                } else {
                    normalize(x, our_root);
                }
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|x| normalize(x, our_root)),
        Value::Number(n) => {
            if let Some(f) = n.as_f64()
                && f.fract() == 0.0
                && f.abs() < 9e15
            {
                *v = Value::from(f as i64);
            }
        }
        _ => {}
    }
}

/// Deviation 74: on a 2025-negotiated session a request with 2026
/// `_meta` gets the 2026 envelope from rmcp (it honors the request's
/// declared version; its resultType fill cannot be vetoed) — upstream
/// keys off the session version and emits a legacy result. Strip our
/// envelope keys.
fn strip_envelope(v: &mut Value) {
    if let Some(r) = v.pointer_mut("/result").and_then(Value::as_object_mut) {
        for k in ["resultType", "ttlMs", "cacheScope", "_meta"] {
            r.remove(k);
        }
    }
}

#[test]
fn mcp_stdio_golden_replay() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/mcp_golden.json")).unwrap();
    let env = golden_env(&fixture);
    let root = env.fixtures.to_string_lossy().into_owned();
    // The recorded root was the generator's corpus dir = our fixtures dir.
    for session in fixture["sessions"].as_array().unwrap() {
        let name = session["name"].as_str().unwrap();
        let mut input = String::new();
        for req in session["requests"].as_array().unwrap() {
            input.push_str(&serde_json::to_string(req).unwrap());
            input.push('\n');
        }
        let out = env.run_mcp(&input);

        // Launcher assertion: every stdout line parses as JSON-RPC —
        // nothing else may reach stdout (AC6).
        let mut got: BTreeMap<String, Value> = BTreeMap::new();
        for line in out.stdout.lines() {
            assert!(!line.trim().is_empty(), "session {name}: blank stdout line");
            let parsed: Value = serde_json::from_str(line).unwrap_or_else(|e| {
                panic!("session {name}: non-JSON-RPC stdout line {line:?}: {e}")
            });
            if let Some(id) = parsed.get("id") {
                got.insert(id.to_string(), parsed);
            }
        }

        let expected = session["responses"].as_object().unwrap();
        assert_eq!(
            got.len(),
            expected.len(),
            "session {name}: response count — got ids {:?}",
            got.keys().collect::<Vec<_>>()
        );
        for (id, want) in expected {
            let mut want = want.clone();
            // Deviation (registered, D6): upstream's -32601 message is
            // the literal "Method not found"; rmcp's is the method name.
            if want.pointer("/error/code") == Some(&Value::from(-32601))
                && let Some(m) = want.pointer_mut("/error/message")
            {
                *m = Value::from("<M>");
            }
            let mut have = got
                .get(id)
                .unwrap_or_else(|| panic!("session {name}: no response for id {id}"))
                .clone();
            if name == "mixed-era" {
                strip_envelope(&mut have);
            }
            if have.pointer("/error/code") == Some(&Value::from(-32601))
                && let Some(m) = have.pointer_mut("/error/message")
            {
                *m = Value::from("<M>");
            }
            normalize(&mut want, "__NONE__");
            normalize(&mut have, &root);
            assert_eq!(have, want, "session {name} id {id} differs");
        }

        // EOF lifecycle: clean exit + the upstream shutdown breadcrumb.
        assert_eq!(
            i64::from(out.exit_code),
            session["exit_code"].as_i64().unwrap(),
            "session {name} exit code"
        );
        assert_eq!(
            out.stderr,
            session["stderr"].as_str().unwrap(),
            "session {name} stderr"
        );
    }
}

/// `qmd mcp <positional>` other than `stop` is rejected — upstream has
/// no `status` here.
#[test]
fn mcp_positional_is_unknown_subcommand() {
    let env = CliEnv::new();
    let out = env.run(&["mcp", "status"]);
    assert_eq!(out.exit_code, 1);
    assert!(out.stderr.contains("Unknown subcommand: status"));
    assert!(out.stderr.contains("qmd doctor"));
}

/// Real-model smoke test: auto-expand `query` (generator + embedder +
/// reranker) through `qmd mcp` on an isolated, embedded copy of the
/// upstream `cli.test.ts` corpus. The models are the user's cached
/// default GGUF files, passed as local paths through `QMD_*_MODEL` so
/// no network is touched. Stdin stays open until the query answers —
/// closing it early would hand the cold model load to the 5 s EOF
/// drain. Every stdout line must be JSON-RPC: native llama.cpp logging
/// must never reach stdout.
#[test]
#[ignore = "requires cached GGUF models"]
fn mcp_query_real_models_emits_jsonrpc_only() {
    use std::io::{BufRead as _, Read as _, Write as _};

    let env = CliEnv::new();
    common::fixtures_corpus(&env);
    env.write_config(
        "index",
        &common::collection_yaml("fixtures", &env.fixtures, "**/*.md"),
    );
    env.update();
    let cache = qmd::llm::download::model_cache_dir(&qmd::Environment::from_process());
    let models: Vec<(&str, String)> = [
        (
            "QMD_EMBED_MODEL",
            "hf_ggml-org_embeddinggemma-300M-Q8_0.gguf",
        ),
        (
            "QMD_GENERATE_MODEL",
            "hf_tobil_qmd-query-expansion-1.7B-q4_k_m.gguf",
        ),
        (
            "QMD_RERANK_MODEL",
            "hf_ggml-org_qwen3-reranker-0.6b-q8_0.gguf",
        ),
    ]
    .into_iter()
    .map(|(key, file)| (key, cache.join(file).to_string_lossy().into_owned()))
    .collect();
    let model_env: Vec<(&str, &str)> = models.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let embed = env.run_opts(&["embed"], &env.fixtures, true, &model_env);
    assert_eq!(embed.exit_code, 0, "embed failed: {}", embed.stderr);

    let mut cmd = env.mcp_command();
    cmd.envs(model_env.iter().copied());
    let mut child = cmd.spawn().expect("spawn qmd mcp");
    let mut stderr = child.stderr.take().unwrap();
    let stderr_reader = std::thread::spawn(move || {
        let mut s = String::new();
        stderr.read_to_string(&mut s).unwrap();
        s
    });
    let mut stdin = child.stdin.take().unwrap();
    for req in [
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}"#,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"query","arguments":{"query":"who fixes the authentication bug"}}}"#,
    ] {
        writeln!(stdin, "{req}").unwrap();
    }
    let mut stdout = std::io::BufReader::new(child.stdout.take().unwrap());
    let mut lines = Vec::new();
    let query = loop {
        let mut line = String::new();
        assert_ne!(
            stdout.read_line(&mut line).unwrap(),
            0,
            "stdout closed before the query answered: {lines:?}"
        );
        let v: Value =
            serde_json::from_str(&line).unwrap_or_else(|_| panic!("not JSON-RPC: {line:?}"));
        assert_eq!(v["jsonrpc"], "2.0", "bad line: {line}");
        lines.push(line);
        if v["id"] == 2 {
            break v;
        }
    };
    drop(stdin);
    let mut rest = String::new();
    stdout.read_to_string(&mut rest).unwrap();
    assert!(rest.is_empty(), "stdout after the last response: {rest:?}");
    let status = child.wait().unwrap();
    let err_text = stderr_reader.join().unwrap();
    assert!(
        status.success(),
        "mcp exited {status:?}; stderr: {err_text}"
    );
    assert!(
        err_text.contains("Shutting down (stdin closed)..."),
        "{err_text}"
    );

    assert!(query.pointer("/result/isError").is_none(), "{query}");
    let results = query
        .pointer("/result/structuredContent/results")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("no results array: {query}"));
    assert!(
        results
            .first()
            .is_some_and(|r| r["file"] == "fixtures/notes/meeting.md"),
        "meeting notes should rank first: {query}"
    );
}

// ---------------------------------------------------------------------------
// HTTP transport (P5.3) — replay the recorded `qmd mcp --http` session:
// the 56 exchanges in `fixture["http"]["cases"]` then the SIGTERM
// shutdown. Requests go over a raw socket so the recorded header set
// (including a deliberately absent `Accept`) is sent verbatim — the
// socket client lives in `common`.
// ---------------------------------------------------------------------------

/// `data:` lines of an SSE response, parsed as JSON.
fn sse_events(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|d| serde_json::from_str(d).unwrap_or_else(|e| panic!("bad SSE data {d:?}: {e}")))
        .collect()
}

/// stderr → comparable lines: clock prefix, port and ms stripped;
/// upstream's `responseMode:` SDK warning and its `HTTP handler error`
/// JS stack are folded (ours logs a single line instead of a stack —
/// the stderr *shape* deviation is recorded in the parity matrix).
fn norm_stderr(s: &str) -> Vec<String> {
    let ts = regex::Regex::new(r"^\d{2}:\d{2}:\d{2}\.\d{3} ").unwrap();
    let ts_line = regex::Regex::new(r"^\d{2}:\d{2}:\d{2}").unwrap();
    let port = regex::Regex::new(r"localhost:\d+").unwrap();
    let ms = regex::Regex::new(r"\(\d+ms\)").unwrap();
    let mut out = Vec::new();
    // After `HTTP handler error` Bun prints the error's source frame
    // and `at` stack until the next real log line — fold the block.
    let mut skip_error_block = false;
    for line in s.lines() {
        let is_log_line = ts_line.is_match(line)
            || line.starts_with("Shutting down")
            || line.starts_with("QMD MCP server");
        if skip_error_block && !is_log_line {
            continue;
        }
        skip_error_block = false;
        // Upstream's SDK prints this warning at startup; our transport
        // does not (recorded deviation).
        if line.starts_with("responseMode:") {
            continue;
        }
        if line.contains("HTTP handler error:") {
            out.push("HTTP handler error".to_owned());
            skip_error_block = true;
            continue;
        }
        let l = ts.replace(line, "<TS> ");
        let l = port.replace_all(&l, "localhost:<PORT>");
        let l = ms.replace_all(&l, "(<MS>)");
        out.push(l.into_owned());
    }
    out
}

/// Golden replay of the recorded HTTP session (P5.3).
#[test]
fn mcp_http_golden_replay() {
    use std::io::{BufRead as _, BufReader};
    use std::sync::mpsc::channel;
    use std::time::{Duration, Instant};

    let fixture: Value = serde_json::from_str(include_str!("fixtures/mcp_golden.json")).unwrap();
    let env = golden_env(&fixture);
    let root = env.fixtures.to_string_lossy().into_owned();

    let port = common::free_port();
    let mut child = env
        .mcp_command_with(&["--http".to_owned(), "--port".to_owned(), port.to_string()])
        .spawn()
        .unwrap();

    // stderr lines stream over a channel — wait for the startup line,
    // keep everything for the final comparison.
    let (tx, rx) = channel::<String>();
    let stderr_pipe = child.stderr.take().unwrap();
    let reader = std::thread::spawn(move || {
        let mut r = BufReader::new(stderr_pipe);
        let mut line = String::new();
        while r.read_line(&mut line).unwrap_or(0) > 0 {
            drop(tx.send(std::mem::take(&mut line)));
        }
    });
    let mut stderr = String::new();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(line) => {
                stderr.push_str(&line);
                if line.contains("listening on") {
                    break;
                }
            }
            Err(e) => {
                panic!("server never printed its listening line ({e}); stderr so far:\n{stderr}");
            }
        }
    }

    let cases = fixture["http"]["cases"].as_array().unwrap();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let method = case["method"].as_str().unwrap();
        let path = case["path"].as_str().unwrap();
        let headers = case["headers"].as_object().unwrap();
        let body = match &case["body"] {
            Value::Null => None,
            Value::String(s) => Some(s.clone()),
            other => Some(serde_json::to_string(other).unwrap()),
        };
        let reply = common::http_request(port, method, path, headers, body.as_deref());

        let want = &case["response"];
        assert_eq!(
            reply.status,
            want["status"].as_u64().unwrap() as u16,
            "case {name}: status"
        );
        for (k, v) in want["headers"].as_object().unwrap() {
            let have = reply.headers.get(k.as_str());
            if v.is_null() {
                assert!(
                    have.is_none(),
                    "case {name}: header {k} should be absent, got {have:?}"
                );
            } else {
                assert_eq!(
                    have.map(String::as_str),
                    Some(v.as_str().unwrap()),
                    "case {name}: header {k}"
                );
            }
        }

        let want_body = want["body"].as_str().unwrap();
        let want_sse = want["sse"].as_array().unwrap();
        if !want_sse.is_empty() {
            // SSE: compare the parsed `data:` events, not the frame
            // bytes (JSON key order differs by implementation).
            let mut have: Vec<Value> = sse_events(&reply.body);
            let mut want_events: Vec<Value> = want_sse.clone();
            for v in &mut want_events {
                normalize(v, "__NONE__");
            }
            for v in &mut have {
                normalize(v, &root);
            }
            assert_eq!(have, want_events, "case {name}: SSE events differ");
        } else if let Ok(mut want_v) = serde_json::from_str::<Value>(want_body) {
            let mut have_v: Value = serde_json::from_str(&reply.body)
                .unwrap_or_else(|e| panic!("case {name}: body not JSON {:?}: {e}", reply.body));
            normalize(&mut want_v, "__NONE__");
            normalize(&mut have_v, &root);
            assert_eq!(have_v, want_v, "case {name}: JSON body differs");
        } else {
            assert_eq!(reply.body, want_body, "case {name}: body differs");
        }
    }

    // SIGTERM shutdown — `Shutting down (SIGTERM)...` + exit 0.
    std::process::Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .output()
        .unwrap();
    let status = child.wait().unwrap();
    while let Ok(line) = rx.recv() {
        stderr.push_str(&line);
    }
    reader.join().unwrap();
    assert_eq!(status.code(), Some(0), "SIGTERM exit code: {status}");

    let want_lines = norm_stderr(fixture["http"]["stderr"].as_str().unwrap());
    let have_lines = norm_stderr(&stderr);
    assert_eq!(
        have_lines, want_lines,
        "stderr differs:\n--- got ---\n{stderr}"
    );
}
