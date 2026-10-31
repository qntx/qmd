//! MCP HTTP daemon state files and PID identity — ports upstream
//! `src/cli/mcp-pid.ts` (v2.8.3) 1:1, plus the session detach the
//! respawned child needs.
//!
//! Pidfiles alone are unsafe after PID reuse (e.g. post-reboot):
//! `stop`/`status`/`--daemon` confirm a recorded PID still belongs to a
//! qmd process before signalling it or treating it as "already running"
//! (upstream `isQmdMcpPid`).
//!
//! Upstream detaches the daemon with `spawn(detached: true)`; the Rust
//! equivalent, `CommandExt::pre_exec(setsid)`, is `unsafe` and the
//! workspace forbids `unsafe` outside the sqlite-vec shim — so the
//! parent marks the respawned child with [`DAEMON_CHILD_ENV`] and the
//! child calls [`detach_daemon_child`] itself (P5-D8). The observable
//! result is identical: new session, no controlling terminal, reparented
//! to init when the launcher exits.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use regex::Regex;

/// Env marker the daemon spawner sets on the respawned child; seeing
/// it, the CLI calls [`detach_daemon_child`] before serving.
pub const DAEMON_CHILD_ENV: &str = "QMD_MCP_DAEMON_CHILD";

/// `mcpDaemonStateFiles(indexName)` (mcp-pid.ts:15-21).
///
/// The default index keeps `mcp.pid`/`mcp.log` for compatibility;
/// named indexes are scoped `mcp-<name>.{pid,log}` so a named daemon
/// can run alongside the default (#772).
#[must_use]
pub fn daemon_state_files(index_name: &str) -> (String, String) {
    let suffix = if index_name.is_empty() || index_name == "index" {
        String::new()
    } else {
        format!("-{index_name}")
    };
    (format!("mcp{suffix}.pid"), format!("mcp{suffix}.log"))
}

/// Resolved daemon paths — `mcpDaemonPaths()` (qmd.ts:211-218):
/// `$XDG_CACHE_HOME/qmd` else `~/.cache/qmd`, scoped per index name.
#[derive(Debug)]
#[non_exhaustive]
pub struct DaemonPaths {
    /// Cache directory the pidfile and log live in.
    pub cache_dir: PathBuf,
    /// `mcp[-<index>].pid`.
    pub pid_path: PathBuf,
    /// `mcp[-<index>].log`.
    pub log_path: PathBuf,
}

/// `mcpDaemonPaths()` over [`qmd::paths::cache_dir`] — the same
/// XDG/`~/.cache` resolution upstream uses.
#[must_use]
pub fn daemon_paths(env: &qmd::Environment, index_name: &str) -> DaemonPaths {
    let cache_dir = qmd::paths::cache_dir(env);
    let (pid_file, log_file) = daemon_state_files(index_name);
    DaemonPaths {
        pid_path: cache_dir.join(pid_file),
        log_path: cache_dir.join(log_file),
        cache_dir,
    }
}

/// `parseInt(readFileSync(pidPath).trim())` — lenient JS parse; a
/// missing file or non-numeric content is `None` (upstream's `NaN`
/// takes the stale-file branch in every caller).
#[must_use]
pub fn read_pidfile(path: &Path) -> Option<i64> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| qmd::js::parse_int(&s))
}

/// `looksLikeQmdMcpCommand` (mcp-pid.ts:25-30).
///
/// Matches a bare `qmd`, `qmd.ts`/`qmd.js`, or a path ending in
/// `/qmd(.ts|.js)` — case-insensitive (`/i`), never inside another
/// token.
///
/// # Panics
/// Never — the static pattern cannot fail to compile.
#[must_use]
#[expect(clippy::expect_used, reason = "a static regex cannot fail to compile")]
pub fn looks_like_qmd_mcp_command(cmdline: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let s = cmdline.trim();
    if s.is_empty() {
        return false;
    }
    RE.get_or_init(|| {
        Regex::new(r"(?i)(?:^|[\s/\\])qmd(?:\.(?:ts|js))?(?:[\s]|$)").expect("static regex")
    })
    .is_match(s)
}

/// `readProcessCmdline` (mcp-pid.ts:33-59).
///
/// Linux `/proc/<pid>/cmdline` preferred; `ps -p <pid> -o args=` with
/// `-o command=` fallback. Upstream caps `ps` with a 2s timeout —
/// `std::process::Command` has no timeout knob; a wedged `ps` is
/// theoretical and this keeps the implementation dependency-free.
#[cfg(unix)]
#[must_use]
pub fn read_process_cmdline(pid: i64) -> Option<String> {
    if pid <= 0 {
        return None;
    }
    let proc_path = format!("/proc/{pid}/cmdline");
    if Path::new(&proc_path).exists()
        && let Ok(raw) = std::fs::read_to_string(&proc_path)
    {
        let cmdline = raw.replace('\0', " ");
        let cmdline = cmdline.trim();
        if !cmdline.is_empty() {
            return Some(cmdline.to_owned());
        }
    }

    let ps = |opt: &str| -> Option<String> {
        std::process::Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", opt])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
    };
    // `args=` first; only a *failed* ps retries with `command=` —
    // a successful empty `args=` answer means "no such process".
    let cmdline = ps("args=").or_else(|| ps("command="))?;
    let trimmed = cmdline.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

/// Non-unix fallback — upstream's `ps`/`/proc` probes would all fail,
/// landing on the same `null` → "treated as stale" verdict.
#[cfg(not(unix))]
#[must_use]
pub fn read_process_cmdline(_pid: i64) -> Option<String> {
    None
}

/// `isQmdMcpPid` (mcp-pid.ts:65-77).
///
/// Alive AND its command line looks like qmd; anything else is treated
/// as stale. The upstream `kill(pid, 0)` liveness gate is folded into
/// the cmdline read — a readable cmdline already proves the process
/// exists, and a zombie's empty `/proc` cmdline falls through to `ps`,
/// which reports `<defunct>` and fails the regex either way.
#[must_use]
pub fn is_qmd_mcp_pid(pid: i64) -> bool {
    if pid <= 0 {
        return false;
    }
    read_process_cmdline(pid).is_some_and(|c| looks_like_qmd_mcp_command(&c))
}

/// `process.kill(pid, "SIGTERM")` — `Err` mirrors the JS throw (the
/// `stop` caller then takes its stale-file branch).
///
/// # Errors
/// `ESRCH` (no such process), `EPERM` (foreign process), or
/// `EINVAL` (pid out of range).
#[cfg(unix)]
pub fn sigterm(pid: i64) -> std::io::Result<()> {
    let pid = i32::try_from(pid)
        .ok()
        .and_then(rustix::process::Pid::from_raw)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid pid"))?;
    rustix::process::kill_process(pid, rustix::process::Signal::TERM)
        .map_err(|e| std::io::Error::from_raw_os_error(e.raw_os_error()))
}

/// Non-unix fallback — no POSIX `kill`; reporting failure puts `stop`
/// on its stale-file branch like upstream's `catch`.
///
/// # Errors
/// Always `EINVAL` — signalling is unsupported on this platform.
#[cfg(not(unix))]
pub fn sigterm(_pid: i64) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        "unsupported platform",
    ))
}

/// `setsid()` in the respawned daemon child — the safe equivalent of
/// Node's `detached: true` (P5-D8).
///
/// Failures are caller-ignored: a child that somehow cannot detach
/// still serves; it just dies with its terminal.
///
/// # Errors
/// `EPERM` when the process is already a process-group leader.
#[cfg(unix)]
pub fn detach_daemon_child() -> std::io::Result<()> {
    rustix::process::setsid()
        .map(|_| ())
        .map_err(|e| std::io::Error::from_raw_os_error(e.raw_os_error()))
}

/// Non-unix fallback — there is no session to leave. On Windows the
/// detach already happened at spawn time (`DETACHED_PROCESS |
/// CREATE_NEW_PROCESS_GROUP` in `qmd-cli`'s `spawn_daemon`), mirroring
/// Node's `detached: true` console semantics.
///
/// # Errors
/// Never fails.
#[cfg(not(unix))]
pub fn detach_daemon_child() -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::missing_assert_message,
    clippy::indexing_slicing,
    clippy::disallowed_methods,
    reason = "idiomatic test-code patterns; sleeps are brief retry windows waiting for ps/proc to see a spawned pid"
)]
mod tests {
    //! Ports `test/mcp-pid.test.ts` (7 cases) plus a live-PID check the
    //! upstream suite could not do.

    use super::{
        daemon_state_files, is_qmd_mcp_pid, looks_like_qmd_mcp_command, read_pidfile,
        read_process_cmdline,
    };

    #[test]
    fn matches_bare_qmd_and_common_cli_script_paths() {
        // upstream test.each positive cases.
        for cmd in [
            "qmd mcp --http --port 8181",
            "/usr/local/bin/qmd mcp --http",
            "node /home/me/qmd/src/cli/qmd.ts mcp --http",
            "node /home/me/qmd/dist/cli/qmd.js mcp --http",
            "tsx src/cli/qmd.ts mcp --http --daemon",
        ] {
            assert!(looks_like_qmd_mcp_command(cmd), "{cmd}");
        }
    }

    #[test]
    fn rejects_empty_whitespace_and_unrelated_processes() {
        for cmd in [
            "",
            "   ",
            "/System/Library/PrivateFrameworks/GenerativeExperiencesRuntime.framework/Versions/A/generativeexperiencesd",
            "sleep 1000000",
            "node server.js",
        ] {
            assert!(!looks_like_qmd_mcp_command(cmd), "{cmd}");
        }
    }

    #[test]
    fn does_not_match_qmd_as_a_substring_of_another_token() {
        assert!(!looks_like_qmd_mcp_command("myqmdtool serve"));
        assert!(!looks_like_qmd_mcp_command("qmdfoo"));
    }

    #[test]
    fn default_index_keeps_unscoped_state_files() {
        assert_eq!(
            daemon_state_files("index"),
            ("mcp.pid".to_owned(), "mcp.log".to_owned())
        );
        assert_eq!(
            daemon_state_files(""),
            ("mcp.pid".to_owned(), "mcp.log".to_owned())
        );
    }

    #[test]
    fn named_indexes_get_scoped_state_files() {
        // #772.
        assert_eq!(
            daemon_state_files("hsm-public-repro"),
            (
                "mcp-hsm-public-repro.pid".to_owned(),
                "mcp-hsm-public-repro.log".to_owned()
            )
        );
    }

    #[test]
    fn invalid_or_dead_pids_are_not_qmd() {
        for pid in [0_i64, -1, 999_999_999] {
            assert!(!is_qmd_mcp_pid(pid), "{pid}");
        }
    }

    #[test]
    fn read_pidfile_parses_leniently() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mcp.pid");
        assert_eq!(read_pidfile(&path), None, "missing file");
        std::fs::write(&path, " 1234 \n").unwrap();
        assert_eq!(read_pidfile(&path), Some(1234));
        std::fs::write(&path, "55abc").unwrap();
        assert_eq!(read_pidfile(&path), Some(55), "parseInt prefix");
        std::fs::write(&path, "abc").unwrap();
        assert_eq!(read_pidfile(&path), None, "NaN upstream");
    }

    #[cfg(unix)]
    #[test]
    fn live_qmd_named_process_passes_identity_check() {
        // Upstream can't force a matching cmdline in vitest; we can —
        // symlink `sleep` as `qmd` so the cmdline matches the regex.
        let dir = tempfile::tempdir().unwrap();
        let qmd_link = dir.path().join("qmd");
        std::os::unix::fs::symlink("/bin/sleep", &qmd_link).unwrap();
        let mut child = std::process::Command::new(&qmd_link)
            .arg("30")
            .spawn()
            .unwrap();
        let pid = i64::from(child.id());
        // `ps`/proc visibility is not instantaneous — retry briefly.
        let mut ok = false;
        for _ in 0..50 {
            if is_qmd_mcp_pid(pid) {
                ok = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(ok, "cmdline: {:?}", read_process_cmdline(pid));
        assert_eq!(
            read_process_cmdline(pid).as_deref(),
            Some(format!("{} 30", qmd_link.display()).as_str())
        );
        drop(child.kill());
        drop(child.wait());
    }

    #[cfg(unix)]
    #[test]
    fn live_non_qmd_process_fails_identity_check() {
        let mut child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .unwrap();
        let pid = i64::from(child.id());
        // Give `ps` a moment to see the process.
        let mut verdict = false;
        for _ in 0..50 {
            if read_process_cmdline(pid).is_some() {
                verdict = is_qmd_mcp_pid(pid);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(!verdict);
        drop(child.kill());
        drop(child.wait());
    }

    #[cfg(unix)]
    #[test]
    fn read_cmdline_of_dead_pid_is_none() {
        assert_eq!(read_process_cmdline(999_999_999), None);
    }
}
