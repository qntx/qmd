//! Pre-update hook execution — the primitive behind a collection's
//! `update:` field, ported from upstream `cli/qmd.ts:947-979`.
//!
//! Upstream spawns `bash -c <cmd>` with the collection path as cwd,
//! stdin ignored, stdout/stderr captured, and `process.exit`s on a
//! non-zero code or spawn failure. This library component only runs the
//! command and reports the outcome — printing, exit-code propagation
//! and process exit are the caller's decision (P3-D1: orchestration
//! lives in the CLI).
//!
//! Micro-deviation: upstream appends `Buffer.toString()` per `data`
//! event, so a multi-byte UTF-8 sequence split across chunks yields two
//! U+FFFDs; we decode the whole buffer once, which yields the correct
//! character instead (parity deviation 56).

use std::path::Path;
use std::process::{Command, Stdio};

/// The outcome of [`run_update_hook`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UpdateHookOutput {
    /// Captured stdout (UTF-8 lossy, like `Buffer.toString()`).
    pub stdout: String,
    /// Captured stderr (UTF-8 lossy).
    pub stderr: String,
    /// Process exit code; `1` when the process was killed by a signal
    /// (upstream `code ?? 1` on the `close` event).
    pub code: i32,
}

/// Run `bash -c <command>` with `cwd` as the working directory —
/// upstream's per-collection `update:` invocation (qmd.ts:951-961):
/// stdin ignored, stdout and stderr piped and captured whole.
///
/// # Errors
/// Spawn failure (no `bash`, resource exhaustion) — upstream rejects and
/// exits `1`; the caller decides how to surface it.
pub fn run_update_hook(command: &str, cwd: &Path) -> Result<UpdateHookOutput, std::io::Error> {
    let out = Command::new("bash")
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .output()?;
    Ok(UpdateHookOutput {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        code: out.status.code().unwrap_or(1),
    })
}
