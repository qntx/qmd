//! Shared harness for the CLI integration tests — temporary
//! environment, corpus files, index populated through the library (the
//! `update` CLI is P4.2), and subprocess runs under the same minimal
//! environment the upstream vitest helpers (`test/cli.test.ts` runQmd
//! and friends) and `cli_golden.rs` use.

#![allow(
    dead_code,
    reason = "each test binary only uses a subset of the helpers"
)]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test-code patterns; a failed setup should panic with context"
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use qmd::{Environment, Qmd, UpdateOptions};

const BIN: &str = env!("CARGO_BIN_EXE_qmd");

/// Captured subprocess output.
pub(crate) struct Out {
    /// `stdout` as text.
    pub(crate) stdout: String,
    /// `stderr` as text.
    pub(crate) stderr: String,
    /// Process exit code (`-1` on signal).
    pub(crate) exit_code: i32,
}

/// One isolated CLI environment: separate `$HOME`, `$QMD_CONFIG_DIR`,
/// `$INDEX_PATH` and XDG dirs under one canonicalized tempdir.
pub(crate) struct CliEnv {
    /// Keeps the tempdir alive.
    _top: tempfile::TempDir,
    /// Canonicalized tempdir root — macOS `/tmp` is `/private/tmp` and
    /// `--full-path` prints realpaths, so comparisons need this form.
    pub(crate) root: PathBuf,
    /// `<root>/config` — `$QMD_CONFIG_DIR`.
    pub(crate) config_dir: PathBuf,
    /// `<root>/index.sqlite` — `$INDEX_PATH` when `index_path` is used.
    pub(crate) db_path: PathBuf,
    /// `<root>/fixtures` — corpus dir and default subprocess cwd.
    pub(crate) fixtures: PathBuf,
    /// `<root>/home` — `$HOME`.
    pub(crate) home: PathBuf,
}

impl CliEnv {
    /// A fresh environment with `config/`, `fixtures/`, `home/` and the
    /// XDG/tmp dirs created.
    pub(crate) fn new() -> Self {
        let top = tempfile::tempdir().unwrap();
        // Same canonicalizer the CLI uses — verbatim `\\?\` prefixes
        // must not diverge from what gets written into configs.
        let root = qmd::paths::canonicalize(top.path()).unwrap();
        for dir in [
            "config",
            "fixtures",
            "home",
            "xdg-config",
            "xdg-cache",
            "tmp",
        ] {
            fs::create_dir_all(root.join(dir)).unwrap();
        }
        Self {
            _top: top,
            config_dir: root.join("config"),
            db_path: root.join("index.sqlite"),
            fixtures: root.join("fixtures"),
            home: root.join("home"),
            root,
        }
    }

    /// Write a corpus file under `fixtures/` (parents created).
    pub(crate) fn write_file(&self, rel: &str, content: &str) -> PathBuf {
        let path = self.fixtures.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, content).unwrap();
        path
    }

    /// Write `<name>.yml` into the config dir; `{root}` expands to the
    /// canonicalized root.
    pub(crate) fn write_config(&self, name: &str, yaml: &str) -> PathBuf {
        let path = self.config_dir.join(format!("{name}.yml"));
        fs::write(&path, yaml.replace("{root}", &self.root.to_string_lossy())).unwrap();
        path
    }

    /// `index.yml` + `INDEX_PATH` database — the common case.
    pub(crate) fn update(&self) {
        self.reindex("index", "index", true);
    }

    /// Populate an index through the library (`Qmd::update`); `config_name`
    /// selects `<config_dir>/<config_name>.yml`, `index_name` the db
    /// (`INDEX_PATH` when `index_path`, else the named db under
    /// `XDG_CACHE_HOME` like `--index <name>`).
    pub(crate) fn reindex(&self, config_name: &str, index_name: &str, index_path: bool) {
        let env = self.library_env(index_path);
        let qmd = Qmd::builder(qmd::paths::default_db_path(&env, index_name))
            .environment(env)
            .config_file(self.config_dir.join(format!("{config_name}.yml")))
            .build()
            .unwrap();
        qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();
    }

    /// Run the `qmd` binary with `INDEX_PATH` set, cwd = `fixtures/`.
    pub(crate) fn run(&self, args: &[&str]) -> Out {
        self.run_opts(args, &self.fixtures, true, &[])
    }

    /// `run` with an explicit cwd (`$PWD` follows, like upstream).
    pub(crate) fn run_at(&self, args: &[&str], cwd: &Path) -> Out {
        self.run_opts(args, cwd, true, &[])
    }

    /// Full control: cwd, whether `$INDEX_PATH` is set (unset it for
    /// `--index <name>` scenarios so the named db under
    /// `XDG_CACHE_HOME` is used), and extra env vars.
    pub(crate) fn run_opts(
        &self,
        args: &[&str],
        cwd: &Path,
        index_path: bool,
        extra: &[(&str, &str)],
    ) -> Out {
        let mut cmd = Command::new(BIN);
        cmd.args(args)
            .current_dir(cwd)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", &self.home)
            .env("XDG_CONFIG_HOME", self.root.join("xdg-config"))
            .env("XDG_CACHE_HOME", self.root.join("xdg-cache"))
            .env("QMD_CONFIG_DIR", &self.config_dir)
            .env("TZ", "UTC")
            .env("NO_COLOR", "1")
            .env("QMD_DOCTOR_DEVICE_PROBE", "0")
            .env("TMPDIR", self.root.join("tmp"))
            .env("PWD", cwd)
            .stdin(Stdio::null());
        if index_path {
            cmd.env("INDEX_PATH", &self.db_path);
        }
        for (key, value) in extra {
            cmd.env(key, value);
        }
        let output = cmd.output().unwrap();
        Out {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            exit_code: output.status.code().unwrap_or(-1),
        }
    }

    /// Run `qmd mcp` over stdio: `input` (raw JSON-RPC lines) is
    /// written to stdin and closed; output is captured after the
    /// process exits on stdin EOF.
    pub(crate) fn run_mcp(&self, input: &str) -> Out {
        use std::io::Write as _;
        let mut child = self.mcp_command().spawn().unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(input.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        Out {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            exit_code: output.status.code().unwrap_or(-1),
        }
    }

    /// `qmd mcp` with the isolated env, cwd = `fixtures/` and all three
    /// stdio streams piped — for tests that drive the session live.
    pub(crate) fn mcp_command(&self) -> Command {
        self.mcp_command_with(&[])
    }

    /// `qmd mcp [extra]` with the isolated env — used by both the stdio
    /// harness (`[]`) and the HTTP one (`--http --port <n>`).
    pub(crate) fn mcp_command_with(&self, extra: &[String]) -> Command {
        let mut args: Vec<&str> = vec!["mcp"];
        args.extend(extra.iter().map(String::as_str));
        self.qmd_command(&args, true, &[])
    }

    /// `qmd [args]` with the isolated env and piped stdio — the general
    /// form behind `mcp_command_with`, for tests that put flags ahead of
    /// the subcommand (`--index <name> mcp --http`), drop `INDEX_PATH`
    /// (named-index scenarios) or set extra env (`QMD_HOST`, …).
    pub(crate) fn qmd_command(
        &self,
        args: &[&str],
        index_path: bool,
        extra: &[(&str, &str)],
    ) -> Command {
        let mut cmd = Command::new(BIN);
        cmd.args(args)
            .current_dir(&self.fixtures)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", &self.home)
            .env("XDG_CONFIG_HOME", self.root.join("xdg-config"))
            .env("XDG_CACHE_HOME", self.root.join("xdg-cache"))
            .env("QMD_CONFIG_DIR", &self.config_dir)
            .env("TZ", "UTC")
            .env("NO_COLOR", "1")
            .env("TMPDIR", self.root.join("tmp"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if index_path {
            cmd.env("INDEX_PATH", &self.db_path);
        }
        for (key, value) in extra {
            cmd.env(key, value);
        }
        cmd
    }

    /// The `Environment` matching the subprocess env (same dirs).
    fn library_env(&self, index_path: bool) -> Environment {
        let mut env = Environment::default();
        env.home = Some(self.home.clone());
        env.xdg_config_home = Some(self.root.join("xdg-config"));
        env.xdg_cache_home = Some(self.root.join("xdg-cache"));
        env.qmd_config_dir = Some(self.config_dir.clone());
        if index_path {
            env.index_path = Some(self.db_path.clone());
        }
        env
    }
}

/// Write the upstream `cli.test.ts` `beforeAll` corpus under
/// `fixtures/` — README, notes/{meeting,ideas}, docs/api and
/// test{1,2}.md — with no config or index.
pub(crate) fn fixtures_corpus(env: &CliEnv) {
    env.write_file(
        "README.md",
        "# Test Project\n\nThis is a test project for QMD CLI testing.\n\n## Features\n\n- Full-text search with BM25\n- Vector similarity search\n- Hybrid search with reranking\n",
    );
    env.write_file(
        "notes/meeting.md",
        "# Team Meeting Notes\n\nDate: 2024-01-15\n\n## Attendees\n- Alice\n- Bob\n- Charlie\n\n## Discussion Topics\n- Project timeline review\n- Resource allocation\n- Technical debt prioritization\n\n## Action Items\n1. Alice to update documentation\n2. Bob to fix authentication bug\n3. Charlie to review pull requests\n",
    );
    env.write_file(
        "notes/ideas.md",
        "# Product Ideas\n\n## Feature Requests\n- Dark mode support\n- Keyboard shortcuts\n- Export to PDF\n\n## Technical Improvements\n- Improve search performance\n- Add caching layer\n- Optimize database queries\n",
    );
    env.write_file(
        "docs/api.md",
        "# API Documentation\n\n## Endpoints\n\n### GET /search\nSearch for documents.\n\nParameters:\n- q: Search query (required)\n- limit: Max results (default: 10)\n\n### GET /document/:id\nRetrieve a specific document.\n\n### POST /index\nIndex new documents.\n",
    );
    env.write_file(
        "test1.md",
        "# Test Document 1\n\nThis is the first test document.\n\nIt has multiple lines for testing line numbers.\nLine 6 is here.\nLine 7 is here.\n",
    );
    env.write_file(
        "test2.md",
        "# Test Document 2\n\nThis is the second test document.\n",
    );
}

/// `collections:` yaml for one `<fixtures>` collection (upstream
/// `collection add . --name <name>` writes the equivalent).
pub(crate) fn collection_yaml(name: &str, path: &Path, pattern: &str) -> String {
    format!(
        "collections:\n  {name}:\n    path: {}\n    pattern: \"{pattern}\"\n",
        path.display()
    )
}

/// A parsed HTTP/1.1 response.
pub(crate) struct HttpReply {
    /// Status code.
    pub(crate) status: u16,
    /// Lower-cased header map.
    pub(crate) headers: std::collections::BTreeMap<String, String>,
    /// Raw body text.
    pub(crate) body: String,
}

/// A free loopback port: bind :0, read it back, release. There is an
/// inherent race before the server claims it, but nothing else here
/// binds ports so in practice it does not trip.
pub(crate) fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Send one request on a fresh connection and read the full response
/// (every response here is `Content-Length`-sized). `Host` defaults to
/// `localhost:<port>` like `fetch`; POSTs default to
/// `Content-Type: application/json` like the fixture recorder.
pub(crate) fn http_request(
    port: u16,
    method: &str,
    path: &str,
    headers: &serde_json::Map<String, serde_json::Value>,
    body: Option<&str>,
) -> HttpReply {
    use std::fmt::Write as _;
    use std::io::{Read as _, Write as _};

    let mut stream = std::net::TcpStream::connect(format!("localhost:{port}")).unwrap();
    let mut req = format!("{method} {path} HTTP/1.1\r\n");
    let has = |name: &str| headers.keys().any(|k| k.eq_ignore_ascii_case(name));
    if !has("host") {
        write!(req, "Host: localhost:{port}\r\n").unwrap();
    }
    for (k, v) in headers {
        write!(req, "{k}: {}\r\n", v.as_str().unwrap()).unwrap();
    }
    if method == "POST" && body.is_some() && !has("content-type") {
        req.push_str("Content-Type: application/json\r\n");
    }
    if let Some(b) = body {
        write!(req, "Content-Length: {}\r\n", b.len()).unwrap();
    }
    req.push_str("\r\n");
    stream.write_all(req.as_bytes()).unwrap();
    if let Some(b) = body {
        // The server may legitimately answer (405/413) and close before
        // the body is fully sent — a failed write is not a test failure;
        // the response still has to arrive.
        drop(stream.write_all(b.as_bytes()));
    }

    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let head_end = loop {
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p + 4;
        }
        let n = stream.read(&mut chunk).unwrap();
        assert_ne!(n, 0, "connection closed mid-headers");
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut lines = head.lines();
    let status: u16 = lines
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    let mut resp_headers = std::collections::BTreeMap::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            resp_headers.insert(k.trim().to_lowercase(), v.trim().to_owned());
        }
    }
    let body_start = head_end;
    let want_len: usize = resp_headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    while buf.len() - body_start < want_len {
        let n = stream.read(&mut chunk).unwrap();
        assert_ne!(n, 0, "connection closed mid-body");
        buf.extend_from_slice(&chunk[..n]);
    }
    HttpReply {
        status,
        headers: resp_headers,
        body: String::from_utf8_lossy(&buf[body_start..body_start + want_len]).into_owned(),
    }
}

/// Strip ANSI SGR (`\x1b[…m`) and OSC 8 (`\x1b]8;;…\x07`) sequences —
/// upstream tests do the same before asserting on plain text.
pub(crate) fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('\x1b') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        if rest.starts_with("\x1b[") {
            rest = rest.find('m').map_or("", |e| &rest[e + 1..]);
        } else if rest.starts_with("\x1b]8;;") {
            rest = rest.find('\x07').map_or("", |e| &rest[e + 1..]);
        } else {
            rest = &rest[1..];
        }
    }
    out.push_str(rest);
    out
}
