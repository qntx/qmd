//! Project-local `.qmd` config trust gate — upstream
//! `resolveLocalConfigTrust` / `printGatedItems` / `confirmOnTty` /
//! `trustCurrentConfig` (cli/qmd.ts:779-848) composed over
//! [`qmd::trust`].
//!
//! `update` (and later `embed`/`query`) consult this before honoring a
//! checked-in config's `update:` hooks, out-of-project collection paths
//! or custom model URIs; in-project indexing always proceeds.

use qmd::Environment;
use qmd::trust::{self, GateAction, GatedItems, LocalConfigGateInput};

use crate::Failure;
use crate::context::CliContext;
use crate::output::{Io, Palette};

/// Upstream `printGatedItems` (qmd.ts:790-811): the gated hooks,
/// out-of-project paths and custom models, in that fixed order.
pub(crate) fn print_gated_items(gated: &GatedItems, io: &mut Io<'_>, c: Palette) {
    if !gated.hooks.is_empty() {
        let _ = writeln!(
            io.out,
            "{}This project's config defines update commands:{}",
            c.yellow, c.reset
        );
        for hook in &gated.hooks {
            let _ = writeln!(
                io.out,
                "  {}{}{}: {}",
                c.bold, hook.collection, c.reset, hook.command
            );
        }
    }
    if !gated.paths.is_empty() {
        let _ = writeln!(
            io.out,
            "{}Collection paths outside this project:{}",
            c.yellow, c.reset
        );
        for item in &gated.paths {
            let _ = writeln!(
                io.out,
                "  {}{}{}: {}",
                c.bold, item.collection, c.reset, item.path
            );
        }
    }
    if !gated.models.is_empty() {
        let _ = writeln!(io.out, "{}Custom models:{}", c.yellow, c.reset);
        for item in &gated.models {
            let _ = writeln!(
                io.out,
                "  {}{}{}: {}",
                c.bold,
                item.slot.as_str(),
                c.reset,
                item.uri
            );
        }
    }
}

/// Upstream `confirmOnTty` (qmd.ts:779-788): print the question on
/// stdout, read one trimmed line from stdin; `y`/`yes` approves.
/// End-of-input answers "no", like upstream's closed readline.
pub(crate) fn confirm_on_tty(question: &str, io: &mut Io<'_>) -> bool {
    let _ = write!(io.out, "{question}");
    let _ = io.out.flush();
    let mut line = String::new();
    if io.stdin.read_line(&mut line).is_err() {
        return false;
    }
    matches!(line.trim().to_lowercase().as_str(), "y" | "yes")
}

/// Upstream `resolveLocalConfigTrust` (qmd.ts:817-848): decide whether
/// the active config's gated fields may be used. Non-interactive runs
/// print the skip notice; interactive runs prompt and record the
/// approval into `trusted.json`.
///
/// # Errors
/// [`Failure::Error`] when the config file or trust store cannot be
/// read/written.
pub(crate) fn resolve_local_config_trust(
    env: &Environment,
    ctx: &CliContext,
    io: &mut Io<'_>,
) -> Result<bool, Failure> {
    let c = Palette::new(Palette::enabled(env, io));
    let config_path = ctx.config_path(env);
    // Upstream `collectSensitiveSnapshot` loads `loadConfig()` — the
    // active file, not a possibly-stale DB mirror.
    let config = qmd::config::load(&config_path)?;
    let snapshot = trust::collect_sensitive_snapshot(&config);
    let gated = trust::local_config_gated(&config_path, &snapshot, env);
    if !trust::has_gated_items(&gated) {
        return Ok(true);
    }
    let decision = trust::decide_local_config_gate(
        &LocalConfigGateInput {
            config_path: &config_path.to_string_lossy(),
            snapshot: &snapshot,
            builtins: &trust::builtin_models(),
            is_interactive: io.stdin_tty && io.stdout_tty,
        },
        env,
        None,
    );

    match decision.action {
        GateAction::Run => Ok(true),
        action => {
            print_gated_items(&gated, io, c);
            let _ = writeln!(
                io.out,
                "{}Config that came with a checkout is not trusted by default.{}",
                c.dim, c.reset
            );
            if action == GateAction::Skip {
                let _ = writeln!(
                    io.out,
                    "{}Skipping them — no terminal to confirm on. Indexing of this project continues.{}",
                    c.yellow, c.reset
                );
                let _ = writeln!(
                    io.out,
                    "{}Approve with 'qmd trust', or set QMD_TRUST_LOCAL_CONFIG=1 for unattended runs.{}\n",
                    c.dim, c.reset
                );
                return Ok(false);
            }
            let approved = confirm_on_tty("Trust this project's .qmd config? [y/N] ", io);
            if !approved {
                let _ = writeln!(
                    io.out,
                    "{}Skipped. Indexing of this project continues.{}\n",
                    c.yellow, c.reset
                );
                return Ok(false);
            }
            trust::record_trust(env, &config_path.to_string_lossy(), &decision.digest)?;
            let _ = writeln!(
                io.out,
                "{}Trusted {}.{} {}Editing a hook, path, or model will ask again.{}\n",
                c.green,
                config_path.display(),
                c.reset,
                c.dim,
                c.reset
            );
            Ok(true)
        }
    }
}

/// Upstream `trustCurrentConfig` (qmd.ts:770-776): record the active
/// config's current gated set as approved — a no-op for non-local
/// configs. Called after the user explicitly mutates a gated field
/// (`collection add`, `collection update-cmd`), so the new state needs
/// no separate approval (#886/#889).
///
/// # Errors
/// [`qmd::Error`] when the config cannot be loaded or the trust store
/// cannot be written.
pub(crate) fn trust_current_config(env: &Environment, ctx: &CliContext) -> Result<(), qmd::Error> {
    let config_path = ctx.config_path(env);
    let config = qmd::config::load(&config_path)?;
    trust::trust_current_config(&config_path, &config, env)
}
