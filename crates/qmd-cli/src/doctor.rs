//! `qmd doctor` — upstream `showDoctor` (cli/qmd.ts:4205-4323) with its
//! check helpers: `doctorCheck` (3747-3749), `normalizedDoctorNextSteps`
//! (3763-3768), `findCachedModelInspection` (3810-3835),
//! `collectEnvironmentOverrides`/`checkEnvironmentOverrides`
//! (3848-3928), `checkDoctorIndexConfig` (3897-3915),
//! `checkModelDefaults` (3930-3955), `checkModelCache` (3957-3999),
//! `linuxCudaRuntimeDiagnostic` (4085-4125) and `runDoctorDeviceChecks`
//! (4127-4203). Store-level probes live in [`qmd::doctor`]; this module
//! only renders.
//!
//! Deliberate deviations:
//! - `Runtime:` reports `rusqlite` — upstream prints its driver
//!   (`bun:sqlite`/`better-sqlite3`), which has no Rust analogue.
//! - `better-sqlite3 package` always reports `not declared` — the port
//!   has no `package.json` dependency to display; the check stays to
//!   preserve the upstream diagnostic shape.
//! - The device probe calls [`llamacpp::probe_device`], a ggml device
//!   listing, rather than node-llama-cpp `getDeviceInfo`; GPU/VRAM
//!   semantics are equivalent, and `gpuOffloading` is rendered as
//!   enabled whenever a GPU device is usable under the resolved mode
//!   (upstream's flag is a build-capability bit that is always true
//!   here when a device exists).
//! - Freshness/fingerprint checks use the *declared* model triple like
//!   upstream `resolveModelsForCli`; the two re-embed probes (legacy
//!   adoption, vector samples) use the runtime embedder — upstream
//!   passes the declared URI to `session.embed`, a difference that only
//!   surfaces for an untrusted project-local config.
//! - No `withLLMSession` cap — this port's backends are stateless per
//!   call (same deviation as `query`/`vsearch`).
//! - `probe_device` cannot catch a native ggml *crash* — like upstream,
//!   the TTY crash hint is the mitigation; a clean `Err` renders the
//!   `probe failed:` check.

use std::path::{Path, PathBuf};

use qmd::llm::ResolvedModels;
use qmd::llm::download::{inspect_gguf_file, model_cache_dir};
use qmd::{Environment, ModelsConfig};

use crate::Failure;
use crate::context::CliContext;
use crate::output::{
    Io, Palette, format_bytes, format_count, short_model_name, summarize_device_names, utf16_prefix,
};

/// `qmd doctor` — upstream `case "doctor"` → `showDoctor`.
///
/// # Errors
/// [`Failure::Error`] when the database cannot be opened at all —
/// upstream's `createStore` failure also escapes `showDoctor`.
pub(crate) fn doctor(env: &Environment, ctx: &CliContext, io: &mut Io<'_>) -> Result<(), Failure> {
    // `getStore()`: upstream swallows config load/sync failures there;
    // `checkDoctorIndexConfig` then reports the invalid YAML itself, so
    // a `Config` build error falls back to a DB-only handle here.
    let qmd = match ctx.open_store(env) {
        Ok(qmd) => qmd,
        Err(qmd::Error::Config { .. }) => qmd::Qmd::builder(ctx.db_path())
            .environment(env.clone())
            .build()?,
        Err(err) => return Err(Failure::Error(err)),
    };
    let c = Palette::new(Palette::enabled(env, io));

    // `resolveModelsForCli()` — the declared triple (the
    // `ensureModelsConfiguredForCli` pin already ran inside
    // `open_store`); a broken config yields env/defaults like
    // upstream's `catch`.
    let config_path = ctx.config_path(env);
    let declared = qmd::config::load(&config_path).unwrap_or_default();
    let active = qmd::llm::resolve_models(declared.models.as_ref(), env);
    let embed_model = active.embed.clone();
    // Rust extension: the declared `embed_pooling` key joins the
    // fingerprint like `embed_fingerprint_extra` (D14); `None` is
    // upstream-identical.
    let extra = declared
        .models
        .as_ref()
        .and_then(|m| m.embed_pooling)
        .unwrap_or_default()
        .fingerprint_extra()
        .map(str::to_owned);
    let fingerprint = qmd::llm::embedding_fingerprint(&embed_model, extra.as_deref());
    let mut next_steps: Vec<String> = Vec::new();

    let _ = writeln!(io.out, "{}QMD Doctor{}\n", c.bold, c.reset);
    let _ = writeln!(io.out, "Index: {}", ctx.db_path().display());
    // Deviation: upstream prints `isBun ? "bun:sqlite" : "better-sqlite3"`.
    let _ = writeln!(io.out, "Runtime: rusqlite");

    match qmd::doctor::sqlite_version(&qmd) {
        Ok(version) => doctor_check(io, c, "SQLite runtime", true, &version),
        Err(err) => doctor_check(io, c, "SQLite runtime", false, &err.to_string()),
    }
    // Deviation: no `package.json` to read a driver version from.
    doctor_check(io, c, "better-sqlite3 package", true, "not declared");
    match qmd::doctor::vec_version(&qmd) {
        Ok(version) => doctor_check(io, c, "sqlite-vec", true, &version),
        Err(err) => doctor_check(io, c, "sqlite-vec", false, &err.to_string()),
    }

    let config_check = check_index_config(env, ctx, io, c, &mut next_steps);
    let config_models = config_check.and_then(|cfg| cfg.models).unwrap_or_default();
    check_environment_overrides(env, ctx, io, c, &active, &config_models);
    check_model_defaults(env, io, c, &active, &config_models);
    check_model_cache(env, ctx, io, c, &active, &mut next_steps);
    run_device_checks(env, ctx, io, c, &mut next_steps);

    run_embedding_checks(
        &qmd,
        env,
        ctx,
        &embed_model,
        extra.as_deref(),
        &fingerprint,
        io,
        c,
        &mut next_steps,
    );

    let steps = normalized_next_steps(next_steps);
    if !steps.is_empty() {
        let _ = writeln!(
            io.out,
            "\n{}Recommended next step{}{}",
            c.bold,
            if steps.len() == 1 { "" } else { "s" },
            c.reset
        );
        for step in steps {
            let _ = writeln!(io.out, "  - {step}");
        }
    }
    Ok(())
}

/// The embedding-related tail of `showDoctor` (cli/qmd.ts:4205-4323):
/// legacy fingerprint adoption, pending/freshness, the
/// `content_vectors` fingerprint groups, and the vector-sample
/// reproduction check — each feeding `next_steps`.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors upstream's shared showDoctor scope: store + declared model identity + io"
)]
fn run_embedding_checks(
    qmd: &qmd::Qmd,
    env: &Environment,
    ctx: &CliContext,
    embed_model: &str,
    fingerprint_extra: Option<&str>,
    fingerprint: &str,
    io: &mut Io<'_>,
    c: Palette,
    next_steps: &mut Vec<String>,
) {
    // `maybeAdoptLegacyEmbeddingFingerprint` — the embedder stands in
    // for upstream's `getDefaultLlamaCpp()` session; its failure lands
    // in the same `catch` arm.
    match qmd
        .embedder()
        .and_then(|e| qmd::doctor::maybe_adopt_legacy_embedding_fingerprint(qmd, e.as_ref()))
    {
        Ok(adoption) if adoption.checked || adoption.adopted > 0 => {
            let details = if adoption.adopted > 0 {
                format!(
                    "adopted {} legacy chunks; {}",
                    adoption.adopted, adoption.reason
                )
            } else {
                adoption.reason
            };
            doctor_check(
                io,
                c,
                "legacy fingerprint adoption",
                adoption.adopted > 0,
                &details,
            );
        }
        Ok(_) => {}
        Err(err) => doctor_check(
            io,
            c,
            "legacy fingerprint adoption",
            false,
            &err.to_string(),
        ),
    }

    match qmd::doctor::pending_embeddings_for(qmd, embed_model, fingerprint_extra) {
        Ok(0) => doctor_check(
            io,
            c,
            "embedding freshness",
            true,
            "all active documents match current fingerprint",
        ),
        Ok(pending) => {
            doctor_check(
                io,
                c,
                "embedding freshness",
                false,
                &format!(
                    "{} active documents need embeddings. Next: `qmd embed`",
                    format_count(pending as u64)
                ),
            );
            next_steps.push(format!(
                "Run `qmd embed` to generate {} missing/stale document embeddings.",
                format_count(pending as u64)
            ));
        }
        Err(err) => doctor_check(io, c, "embedding freshness", false, &err.to_string()),
    }

    check_embedding_fingerprints(qmd, embed_model, fingerprint, io, c, next_steps);

    match qmd
        .embedder()
        .and_then(|e| qmd::doctor::check_embedding_vector_samples(qmd, e.as_ref(), 3))
    {
        Ok(report) => {
            doctor_check(io, c, "embedding vector sample", report.ok, &report.details);
            if !report.ok {
                next_steps.push(
                    "Run `qmd embed --force` to rebuild existing vectors that no longer \
                     reproduce under the current embedding pipeline."
                        .to_owned(),
                );
            }
        }
        Err(err) => {
            doctor_check(
                io,
                c,
                "embedding vector sample",
                false,
                &format!(
                    "{}; rebuild with `qmd embed --force`",
                    sanitize_diagnostic(&err.to_string(), env, ctx.cwd())
                ),
            );
            next_steps.push(
                "Run `qmd embed --force` to rebuild existing vectors, then rerun `qmd doctor`."
                    .to_owned(),
            );
        }
    }
}

/// Upstream `doctorCheck` (cli/qmd.ts:3747-3749): `✓`/`⚠` + label +
/// details on stdout.
fn doctor_check(io: &mut Io<'_>, c: Palette, label: &str, ok: bool, details: &str) {
    let mark = if ok {
        format!("{}✓{}", c.green, c.reset)
    } else {
        format!("{}⚠{}", c.yellow, c.reset)
    };
    let _ = writeln!(io.out, "{mark} {label}: {details}");
}

/// Upstream `sanitizeDiagnosticMessage` (cli/qmd.ts:502-509): home →
/// `~`, cwd → `.`, then trimmed non-empty lines, first three, `"; "`.
fn sanitize_diagnostic(message: &str, env: &Environment, cwd: &Path) -> String {
    let home = qmd::paths::home_dir(env);
    message
        .replace(&*home.to_string_lossy(), "~")
        .replace(&*cwd.to_string_lossy(), ".")
        .split('\n')
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(3)
        .collect::<Vec<_>>()
        .join("; ")
}

/// Upstream `normalizedDoctorNextSteps` (cli/qmd.ts:3763-3768): dedupe,
/// and when a `--force` rebuild is recommended, drop every other
/// `qmd embed` step that doesn't lead with `--force`.
fn normalized_next_steps(steps: Vec<String>) -> Vec<String> {
    let mut unique: Vec<String> = Vec::new();
    for step in steps {
        if !unique.contains(&step) {
            unique.push(step);
        }
    }
    if !unique.iter().any(|s| s.contains("qmd embed --force")) {
        return unique;
    }
    unique
        .into_iter()
        .filter(|s| !s.contains("qmd embed") || s.starts_with("Run `qmd embed --force`"))
        .collect()
}

/// `checkDoctorIndexConfig` (cli/qmd.ts:3897-3915): the config check is
/// also the source of `configModels` for the following checks —
/// `Some(config)` when valid, `None` on an invalid file like upstream's
/// `{config: null}`.
fn check_index_config(
    env: &Environment,
    ctx: &CliContext,
    io: &mut Io<'_>,
    c: Palette,
    next_steps: &mut Vec<String>,
) -> Option<qmd::Config> {
    let config_path = ctx.config_path(env);
    match qmd::config::load(&config_path) {
        Ok(config) => {
            let count = config.collections.len();
            if count == 0 {
                doctor_check(
                    io,
                    c,
                    "index config",
                    false,
                    "no collections configured. Next: `qmd collection add .`",
                );
                next_steps.push(
                    "Run `qmd collection add . --name <name>` from the folder you want to \
                     index, or edit .qmd/index.yml manually."
                        .to_owned(),
                );
            } else {
                doctor_check(
                    io,
                    c,
                    "index config",
                    true,
                    &format!(
                        "{} {} configured",
                        format_count(count as u64),
                        if count == 1 {
                            "collection"
                        } else {
                            "collections"
                        }
                    ),
                );
            }
            Some(config)
        }
        Err(err) => {
            let message = sanitize_diagnostic(&err.to_string(), env, ctx.cwd());
            doctor_check(
                io,
                c,
                "index config",
                false,
                &format!(
                    "invalid index.yml at {}: {message}. Next: fix the YAML and rerun `qmd doctor`",
                    config_path.display()
                ),
            );
            next_steps.push(format!(
                "Fix invalid YAML in {}, then rerun `qmd doctor`.",
                config_path.display()
            ));
            None
        }
    }
}

/// `envValueForDisplay` (cli/qmd.ts:3843-3846): sanitized, then
/// truncated to 96 UTF-16 units as `…93…"..."`.
fn env_value_for_display(value: &str, env: &Environment, cwd: &Path) -> String {
    let sanitized = sanitize_diagnostic(value, env, cwd);
    if qmd::utf16_len(&sanitized) > 96 {
        format!("{}...", utf16_prefix(&sanitized, 93))
    } else {
        sanitized
    }
}

/// One `add` row of upstream `EnvOverride` (cli/qmd.ts:3837-3841).
struct EnvOverride {
    /// Variable name.
    name: &'static str,
    /// Sanitized display value.
    value: String,
    /// Upstream consequence text.
    consequence: String,
}

/// `collectEnvironmentOverrides` (cli/qmd.ts:3848-3890) — the verbatim
/// name/consequence table in upstream order.
#[allow(
    clippy::too_many_lines,
    reason = "one row per upstream add()/addModel() call; the table mirrors qmd.ts:3864-3890 verbatim"
)]
fn collect_environment_overrides(
    env: &Environment,
    cwd: &Path,
    active: &ResolvedModels,
    config_models: &ModelsConfig,
) -> Vec<EnvOverride> {
    let mut overrides = Vec::new();

    let path = |p: Option<&PathBuf>| p.map(|p| p.to_string_lossy().into_owned());
    let text = |s: &Option<String>| s.clone();

    push_override(
        &mut overrides,
        env,
        cwd,
        "INDEX_PATH",
        path(env.index_path.as_ref()),
        "overrides the SQLite index path; QMD reads/writes a different database",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "QMD_CONFIG_DIR",
        path(env.qmd_config_dir.as_ref()),
        "overrides the QMD config directory and takes precedence over XDG_CONFIG_HOME",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "XDG_CONFIG_HOME",
        path(env.xdg_config_home.as_ref()),
        "moves QMD config to $XDG_CONFIG_HOME/qmd when QMD_CONFIG_DIR is not set",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "XDG_CACHE_HOME",
        path(env.xdg_cache_home.as_ref()),
        "moves the default index cache, model cache, and MCP daemon PID files",
    );
    push_model_override(
        &mut overrides,
        env,
        cwd,
        "QMD_EMBED_MODEL",
        "embed",
        &active.embed,
        config_models,
    );
    push_model_override(
        &mut overrides,
        env,
        cwd,
        "QMD_GENERATE_MODEL",
        "generate",
        &active.generate,
        config_models,
    );
    push_model_override(
        &mut overrides,
        env,
        cwd,
        "QMD_RERANK_MODEL",
        "rerank",
        &active.rerank,
        config_models,
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "QMD_FORCE_CPU",
        text(&env.qmd_force_cpu),
        "forces llama.cpp to bypass GPU backends; embeddings/query will be slower but GPU \
         crashes are avoided",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "QMD_LLAMA_GPU",
        text(&env.qmd_llama_gpu),
        "selects llama.cpp GPU backend (metal/cuda/vulkan) or disables GPU when set to \
         false/off/0",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "QMD_DOCTOR_DEVICE_PROBE",
        text(&env.qmd_doctor_device_probe),
        "controls qmd doctor native device probing; 0/off skips GPU probing",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "QMD_EMBED_PARALLELISM",
        text(&env.qmd_embed_parallelism),
        "overrides embedding parallel context count; too high can exhaust RAM/VRAM",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "QMD_EXPAND_CONTEXT_SIZE",
        text(&env.qmd_expand_context_size),
        "overrides query expansion context size; larger values use more memory",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "QMD_RERANK_CONTEXT_SIZE",
        text(&env.qmd_rerank_context_size),
        "overrides reranker context size; larger values use more memory",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "QMD_EMBED_CONTEXT_SIZE",
        text(&env.qmd_embed_context_size),
        "overrides embed context size; larger values use more memory",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "QMD_EDITOR_URI",
        text(&env.qmd_editor_uri),
        "overrides clickable editor link template in terminal output",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "QMD_SKILLS_DIR",
        text(&env.qmd_skills_dir),
        "overrides where qmd skills are discovered from",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "QMD_METAL_KEEP_RESIDENCY",
        text(&env.qmd_metal_keep_residency),
        "opts back into libggml-metal residency sets on darwin; restores ~0ms perf wins for \
         long-lived processes but re-exposes the static-destructor backtrace dump at process \
         exit (ggml-org/llama.cpp#22593)",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "GGML_METAL_NO_RESIDENCY",
        text(&env.ggml_metal_no_residency),
        "set automatically by the launcher on darwin to disable Metal residency sets (avoids \
         ggml-org/llama.cpp#22593); override via QMD_METAL_KEEP_RESIDENCY=1",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "NO_COLOR",
        text(&env.no_color),
        "disables colored terminal output",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "CI",
        text(&env.ci),
        "disables real LLM operations inside QMD's LlamaCpp wrapper",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "HF_ENDPOINT",
        text(&env.hf_endpoint),
        "changes Hugging Face download endpoint used when pulling models",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "QMD_WRAPPER_CAPTURE",
        text(&env.qmd_wrapper_capture),
        "test/debug hook for the qmd shell wrapper; should not be set in normal use",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "WSL_DISTRO_NAME",
        text(&env.wsl_distro_name),
        "enables WSL path handling heuristics",
    );
    push_override(
        &mut overrides,
        env,
        cwd,
        "WSL_INTEROP",
        text(&env.wsl_interop),
        "enables WSL path handling heuristics",
    );
    overrides
}

/// One `add(...)` row of `collectEnvironmentOverrides` (cli/qmd.ts:3851-3861):
/// skip unset/blank values, sanitize + truncate the display value.
fn push_override(
    overrides: &mut Vec<EnvOverride>,
    env: &Environment,
    cwd: &Path,
    name: &'static str,
    raw: Option<String>,
    consequence: &str,
) {
    let Some(raw) = raw else { return };
    let raw = raw.trim();
    if raw.is_empty() {
        return;
    }
    overrides.push(EnvOverride {
        name,
        value: env_value_for_display(raw, env, cwd),
        consequence: consequence.to_owned(),
    });
}

/// The `addModel(name, key, active)` arm of
/// `collectEnvironmentOverrides` (cli/qmd.ts:3863-3872): env value is
/// ignored when the index config pins a different `models.<key>`.
fn push_model_override(
    overrides: &mut Vec<EnvOverride>,
    env: &Environment,
    cwd: &Path,
    name: &'static str,
    key: &str,
    active_model: &str,
    config_models: &ModelsConfig,
) {
    let raw = match name {
        "QMD_EMBED_MODEL" => env.qmd_embed_model.as_deref(),
        "QMD_GENERATE_MODEL" => env.qmd_generate_model.as_deref(),
        _ => env.qmd_rerank_model.as_deref(),
    };
    let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return;
    };
    let configured = match key {
        "embed" => config_models.embed.as_deref(),
        "generate" => config_models.generate.as_deref(),
        _ => config_models.rerank.as_deref(),
    };
    let consequence = if configured.is_some_and(|c| c != raw) {
        format!(
            "set but ignored because index models.{key} is configured as {}",
            configured.unwrap_or_default()
        )
    } else {
        format!(
            "sets the active {key} model to {active_model}; changes embedding/search \
             semantics and may require `qmd pull` plus `qmd embed`"
        )
    };
    push_override(
        overrides,
        env,
        cwd,
        name,
        Some(raw.to_owned()),
        &consequence,
    );
}

/// `checkEnvironmentOverrides` (cli/qmd.ts:3917-3928).
fn check_environment_overrides(
    env: &Environment,
    ctx: &CliContext,
    io: &mut Io<'_>,
    c: Palette,
    active: &ResolvedModels,
    config_models: &ModelsConfig,
) {
    let overrides = collect_environment_overrides(env, ctx.cwd(), active, config_models);
    if overrides.is_empty() {
        doctor_check(io, c, "environment overrides", true, "none");
        return;
    }
    doctor_check(
        io,
        c,
        "environment overrides",
        false,
        &format!("{} set", overrides.len()),
    );
    for o in overrides {
        let _ = writeln!(io.out, "  - {}={}: {}", o.name, o.value, o.consequence);
    }
}

/// `checkModelDefaults` (cli/qmd.ts:3930-3955): per-role notes about
/// non-default model selections, then a single check line.
fn check_model_defaults(
    env: &Environment,
    io: &mut Io<'_>,
    c: Palette,
    active: &ResolvedModels,
    config_models: &ModelsConfig,
) {
    let checks = [
        (
            "embedding",
            "embed",
            active.embed.as_str(),
            config_models.embed.as_deref(),
            qmd::llm::DEFAULT_EMBED_MODEL,
            "QMD_EMBED_MODEL",
            env.qmd_embed_model.as_deref(),
        ),
        (
            "generation",
            "generate",
            active.generate.as_str(),
            config_models.generate.as_deref(),
            qmd::llm::DEFAULT_GENERATE_MODEL,
            "QMD_GENERATE_MODEL",
            env.qmd_generate_model.as_deref(),
        ),
        (
            "reranking",
            "rerank",
            active.rerank.as_str(),
            config_models.rerank.as_deref(),
            qmd::llm::DEFAULT_RERANK_MODEL,
            "QMD_RERANK_MODEL",
            env.qmd_rerank_model.as_deref(),
        ),
    ];
    let mut notes: Vec<String> = Vec::new();
    for (role, _key, active_model, configured, default_model, env_name, env_value) in checks {
        let env_value = env_value.map(str::trim).filter(|v| !v.is_empty());
        if let Some(v) = env_value
            && active_model == v
        {
            notes.push(format!(
                "{role}: env {env_name}={active_model} (default {default_model}; might be ok)"
            ));
        } else if configured.is_some_and(|cfg| cfg != default_model) {
            notes.push(format!(
                "{role}: index {} (default {default_model}; might be ok)",
                configured.unwrap_or_default()
            ));
        } else if let Some(v) = env_value
            && active_model != v
        {
            notes.push(format!(
                "{role}: {env_name} is set to {v} but index config uses {active_model}"
            ));
        }
    }
    if notes.is_empty() {
        doctor_check(io, c, "model defaults", true, "using QMD codebase defaults");
        return;
    }
    doctor_check(
        io,
        c,
        "model defaults",
        false,
        &format!("non-default model configuration: {}", notes.join("; ")),
    );
}

/// Upstream `CachedModelInspection` (cli/qmd.ts:3801-3804).
struct CachedModelInspection {
    /// Verified GGUF path, when found.
    path: Option<PathBuf>,
    /// Verbatim `path: details` strings for invalid candidates.
    invalid: Vec<String>,
}

/// `findCachedModelInspection` (cli/qmd.ts:3810-3835): `hf:` URIs probe
/// the model cache for `.gguf` blobs containing the filename (the
/// `.etag` sidecar is excluded by the `.gguf` suffix, #812); other URIs
/// are inspected as plain files.
#[allow(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "upstream's name.endsWith(\".gguf\") is case-sensitive (cli/qmd.ts:3819)"
)]
fn find_cached_model_inspection(
    model: &str,
    env: &Environment,
    cwd: &Path,
) -> CachedModelInspection {
    let mut invalid = Vec::new();
    if model.starts_with("hf:") {
        let filename = model.rsplit('/').next().unwrap_or(model);
        let cache_dir = model_cache_dir(env);
        if filename.is_empty() || !cache_dir.is_dir() {
            return CachedModelInspection {
                path: None,
                invalid,
            };
        }
        let Ok(entries) = std::fs::read_dir(&cache_dir) else {
            return CachedModelInspection {
                path: None,
                invalid,
            };
        };
        for entry in entries.flatten() {
            let is_file = entry.file_type().is_ok_and(|t| t.is_file());
            let name = entry.file_name().to_string_lossy().into_owned();
            if !is_file || !name.ends_with(".gguf") || !name.contains(filename) {
                continue;
            }
            let candidate = entry.path();
            let inspection = inspect_gguf_file(&candidate);
            if inspection.valid {
                return CachedModelInspection {
                    path: Some(candidate),
                    invalid,
                };
            }
            invalid.push(format!(
                "{}: {}",
                sanitize_diagnostic(&candidate.to_string_lossy(), env, cwd),
                inspection.details
            ));
        }
        return CachedModelInspection {
            path: None,
            invalid,
        };
    }
    let inspection = inspect_gguf_file(Path::new(model));
    if inspection.valid {
        return CachedModelInspection {
            path: Some(PathBuf::from(model)),
            invalid,
        };
    }
    if inspection.exists {
        invalid.push(format!(
            "{}: {}",
            sanitize_diagnostic(model, env, cwd),
            inspection.details
        ));
    }
    CachedModelInspection {
        path: None,
        invalid,
    }
}

/// `checkModelCache` (cli/qmd.ts:3957-3999): role-grouped dedup, then
/// cache inspection per unique model.
fn check_model_cache(
    env: &Environment,
    ctx: &CliContext,
    io: &mut Io<'_>,
    c: Palette,
    active: &ResolvedModels,
    next_steps: &mut Vec<String>,
) {
    let models = [
        ("embedding", &active.embed),
        ("generation", &active.generate),
        ("reranking", &active.rerank),
    ];
    // Upstream `Map` dedup — first-seen insertion order.
    let mut unique: Vec<(&String, Vec<&str>)> = Vec::new();
    for (role, model) in &models {
        if let Some(entry) = unique.iter_mut().find(|(m, _)| *m == *model) {
            entry.1.push(role);
        } else {
            unique.push((model, vec![role]));
        }
    }

    let mut missing: Vec<String> = Vec::new();
    let mut cached: Vec<String> = Vec::new();
    let mut invalid: Vec<String> = Vec::new();
    for (model, roles) in &unique {
        let label = format!("{}: {model}", roles.join("+"));
        let inspection = find_cached_model_inspection(model, env, ctx.cwd());
        invalid.extend(
            inspection
                .invalid
                .into_iter()
                .map(|detail| format!("{label} ({detail})")),
        );
        if inspection.path.is_some() {
            cached.push(label);
        } else {
            missing.push(label);
        }
    }

    if missing.is_empty() && invalid.is_empty() {
        doctor_check(
            io,
            c,
            "model cache",
            true,
            &format!(
                "{} active {} downloaded and valid GGUF",
                cached.len(),
                if cached.len() == 1 {
                    "model is"
                } else {
                    "models are"
                }
            ),
        );
        return;
    }

    let mut parts: Vec<String> = Vec::new();
    if !invalid.is_empty() {
        parts.push(format!("invalid {}: {}", invalid.len(), invalid.join("; ")));
    }
    if !missing.is_empty() {
        parts.push(format!(
            "missing {}/{}: {}",
            missing.len(),
            unique.len(),
            missing.join("; ")
        ));
    }
    let next = if invalid.is_empty() {
        "Next: run `qmd pull`"
    } else {
        "Next: run `qmd pull --refresh` (or remove the bad cached file)"
    };
    doctor_check(
        io,
        c,
        "model cache",
        false,
        &format!("{}. {next}", parts.join("; ")),
    );
    if invalid.is_empty() {
        next_steps.push(
            "Run `qmd pull` to download missing embedding/generation/reranking models before \
             `qmd embed` or `qmd query`."
                .to_owned(),
        );
    } else {
        next_steps.push(
            "Run `qmd pull --refresh` to replace invalid cached model files, or delete the \
             listed file and rerun `qmd pull`."
                .to_owned(),
        );
    }
}

/// `hasLibraryInDirs` (cli/qmd.ts:4073-4083): `libfoo.so` or
/// `libfoo.so.*` in any readable directory.
#[cfg(target_os = "linux")]
fn has_library_in_dirs(library_base_name: &str, dirs: &[PathBuf]) -> bool {
    for dir in dirs {
        if !dir.is_dir() {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name == library_base_name || name.starts_with(&format!("{library_base_name}.")) {
                return true;
            }
        }
    }
    false
}

/// `linuxCudaRuntimeDiagnostic` (cli/qmd.ts:4085-4125): driver libraries
/// visible but CUDA user-space libraries missing from loader paths.
/// Upstream guards on `process.platform !== "linux"` — same shape here.
#[cfg(target_os = "linux")]
fn linux_cuda_runtime_diagnostic(env: &Environment) -> Option<String> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    for value in [&env.ld_library_path, &env.cuda_path].into_iter().flatten() {
        for part in value.split(':') {
            if !part.is_empty() {
                dirs.push(PathBuf::from(part));
            }
        }
    }
    if let Some(cuda_path) = &env.cuda_path {
        dirs.push(PathBuf::from(cuda_path).join("lib64"));
        dirs.push(
            PathBuf::from(cuda_path)
                .join("targets")
                .join("x86_64-linux")
                .join("lib"),
        );
    }
    for dir in [
        "/usr/lib",
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
    ] {
        dirs.push(PathBuf::from(dir));
    }
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_lowercase();
            if !name.starts_with("cuda-") {
                continue;
            }
            let root = entry.path();
            dirs.push(root.join("lib64"));
            dirs.push(root.join("targets").join("x86_64-linux").join("lib"));
        }
    }

    let has_driver =
        has_library_in_dirs("libcuda.so", &dirs) || has_library_in_dirs("libnvidia-ml.so", &dirs);
    if !has_driver {
        return None;
    }
    let missing: Vec<&str> = [
        ("libcudart.so", "CUDA runtime"),
        ("libcublas.so", "cuBLAS"),
        ("libcublasLt.so", "cuBLASLt"),
    ]
    .into_iter()
    .filter(|(library, _)| !has_library_in_dirs(library, &dirs))
    .map(|(_, label)| label)
    .collect();
    if missing.is_empty() {
        return None;
    }
    Some(format!(
        "NVIDIA driver libraries are visible, but CUDA user-space libraries are missing \
         from loader paths ({})",
        missing.join(", ")
    ))
}

/// Non-Linux stub — upstream's `process.platform !== "linux"` early
/// return (cli/qmd.ts:4086).
#[cfg(not(target_os = "linux"))]
const fn linux_cuda_runtime_diagnostic(_env: &Environment) -> Option<String> {
    None
}

/// `configuredGpuModeLabel` (cli/qmd.ts:486-490). Upstream's
/// `isForceCpuEnabled` checks the *raw* env value for truthiness, so
/// `QMD_FORCE_CPU=" "` labels `CPU forced` while
/// `resolveLlamaGpuMode`'s trimmed check resolves `auto` — the quirk
/// is deliberate here.
fn configured_gpu_mode_label(env: &Environment) -> String {
    let forced = env.qmd_force_cpu.as_deref().is_some_and(|v| {
        !v.is_empty()
            && !["false", "off", "none", "disable", "disabled", "0"]
                .contains(&v.trim().to_lowercase().as_str())
    });
    if forced {
        return "CPU forced (QMD_FORCE_CPU)".to_owned();
    }
    env.qmd_llama_gpu
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .unwrap_or("auto")
        .to_owned()
}

/// `isDarwinMetalMitigationActive` (llm.ts:2202-2206): only meaningful
/// on darwin — `GGML_METAL_NO_RESIDENCY=1` set unless the keep-residency
/// override is `1`.
fn darwin_metal_mitigation_active(env: &Environment) -> bool {
    if !cfg!(target_os = "macos") {
        return false;
    }
    if env.qmd_metal_keep_residency.as_deref() == Some("1") {
        return false;
    }
    env.ggml_metal_no_residency.as_deref() == Some("1")
}

/// `runDoctorDeviceChecks` (cli/qmd.ts:4127-4203): mode label, the
/// `QMD_DOCTOR_DEVICE_PROBE` skip, the TTY crash hint, then the probe.
fn run_device_checks(
    env: &Environment,
    ctx: &CliContext,
    io: &mut Io<'_>,
    c: Palette,
    next_steps: &mut Vec<String>,
) {
    const CRASH_HINT: &str = "Probing native llama backend now. If qmd crashes here, rerun \
                              with `QMD_FORCE_CPU=1 qmd doctor` (or \
                              `QMD_DOCTOR_DEVICE_PROBE=0 qmd doctor` to skip this probe).";
    doctor_check(io, c, "device mode", true, &configured_gpu_mode_label(env));

    let skip_probe = env.qmd_doctor_device_probe.as_deref().is_some_and(|v| {
        ["0", "false", "off", "no", "skip"].contains(&v.trim().to_lowercase().as_str())
    });
    if skip_probe {
        doctor_check(
            io,
            c,
            "device probe",
            false,
            "skipped by QMD_DOCTOR_DEVICE_PROBE=0. Next: unset it and rerun `qmd doctor` to \
             verify GPU/CPU acceleration",
        );
        next_steps.push(
            "Unset `QMD_DOCTOR_DEVICE_PROBE` and rerun `qmd doctor` when you want to verify \
             llama.cpp device acceleration."
                .to_owned(),
        );
        return;
    }

    if io.stdout_tty {
        let _ = write!(io.out, "{}{}{}", c.dim, CRASH_HINT, c.reset);
    }
    // ggml init happens inside `probe_device` — like upstream's
    // `ensureLlama(allowBuild:false)` it can abort the process on
    // broken drivers; a crash after the hint is the documented escape.
    let result = qmd::llm::llamacpp::probe_device(env);
    if io.stdout_tty {
        let _ = write!(io.out, "\r{}\r", " ".repeat(CRASH_HINT.len()));
    }
    match result {
        Ok(device) => report_device_probe(&device, env, io, c, next_steps),
        Err(err) => {
            doctor_check(
                io,
                c,
                "device probe",
                false,
                &format!(
                    "probe failed: {}. Next: run with QMD_FORCE_CPU=1 to bypass GPU probing, \
                     or set QMD_LLAMA_GPU=metal|cuda|vulkan and retry",
                    sanitize_diagnostic(&err.to_string(), env, ctx.cwd())
                ),
            );
            next_steps.push(
                "GPU probe failed; try `QMD_FORCE_CPU=1 qmd doctor` to confirm CPU fallback, \
                 then fix GPU drivers/backend if acceleration is expected."
                    .to_owned(),
            );
        }
    }
}

/// `runDoctorDeviceChecks`' probe-success arms (cli/qmd.ts:4147-4197):
/// the GPU detail line plus the darwin Metal-residency check, or the
/// CPU-mode diagnosis (with the Linux CUDA-runtime hint when present).
fn report_device_probe(
    device: &qmd::llm::llamacpp::DeviceProbe,
    env: &Environment,
    io: &mut Io<'_>,
    c: Palette,
    next_steps: &mut Vec<String>,
) {
    let Some(gpu) = &device.gpu else {
        let cuda_diagnostic = linux_cuda_runtime_diagnostic(env);
        let suffix = cuda_diagnostic
            .as_ref()
            .map_or(String::new(), |d| format!(" {d}."));
        doctor_check(
            io,
            c,
            "device probe",
            false,
            &format!(
                "running on CPU ({} math cores).{suffix} Next: install/configure \
                 Metal, CUDA, or Vulkan for faster embeddings, or set \
                 QMD_FORCE_CPU=1 to make CPU mode explicit",
                device.cpu_cores
            ),
        );
        if let Some(diagnostic) = cuda_diagnostic {
            next_steps.push(format!(
                "{diagnostic}; install CUDA runtime/cuBLAS libraries or add their \
                 directory to LD_LIBRARY_PATH, then rerun `qmd doctor`."
            ));
        } else {
            next_steps.push(
                "Vector operations are running on CPU; install/configure Metal, \
                 CUDA, or Vulkan if embedding/query performance is too slow."
                    .to_owned(),
            );
        }
        return;
    };

    let gpu_label = if gpu == "metal" && cfg!(target_os = "macos") {
        "metal (macOS Metal backend)".to_owned()
    } else {
        gpu.clone()
    };
    // A usable GPU under the resolved mode always offloads
    // — upstream's `gpuOffloading` build flag has no
    // disable path here (see module doc).
    let gpu_offloading = true;
    let mut parts = vec![
        format!("GPU {gpu_label}"),
        format!(
            "offloading {}",
            if gpu_offloading {
                "enabled"
            } else {
                "disabled"
            }
        ),
    ];
    if !device.gpu_devices.is_empty() {
        parts.push(format!(
            "devices: {}",
            summarize_device_names(&device.gpu_devices)
        ));
    }
    if let Some((free, total)) = device.vram {
        parts.push(format!(
            "VRAM {} free / {} total",
            format_bytes(free),
            format_bytes(total)
        ));
    }
    parts.push(format!("{} CPU math cores", device.cpu_cores));
    doctor_check(
        io,
        c,
        "device probe",
        gpu_offloading,
        &if gpu_offloading {
            parts.join("; ")
        } else {
            format!(
                "{}. Next: check QMD_LLAMA_GPU and llama.cpp backend support",
                parts.join("; ")
            )
        },
    );
    if !gpu_offloading {
        next_steps.push(
            "GPU was detected but offloading is disabled; check \
             `QMD_LLAMA_GPU=metal|cuda|vulkan` and rerun `qmd doctor`."
                .to_owned(),
        );
    }

    if gpu == "metal" && cfg!(target_os = "macos") {
        check_darwin_metal_residency(env, io, c, next_steps);
    }
}

/// The darwin-only Metal residency-set check of `runDoctorDeviceChecks`
/// (cli/qmd.ts:4164-4193).
fn check_darwin_metal_residency(
    env: &Environment,
    io: &mut Io<'_>,
    c: Palette,
    next_steps: &mut Vec<String>,
) {
    if darwin_metal_mitigation_active(env) {
        doctor_check(
            io,
            c,
            "darwin metal residency",
            true,
            "GGML_METAL_NO_RESIDENCY=1 set by launcher; clean process exit \
             (avoids ggml-org/llama.cpp#22593). Opt back in with \
             QMD_METAL_KEEP_RESIDENCY=1 if you run long-lived qmd processes.",
        );
        return;
    }
    doctor_check(
        io,
        c,
        "darwin metal residency",
        false,
        "residency sets active (QMD_METAL_KEEP_RESIDENCY=1 or launcher \
         bypassed); llama-using commands may dump a libggml-metal \
         backtrace at exit (ggml-org/llama.cpp#22593) even when output \
         succeeded.",
    );
    next_steps.push(
        "Unset `QMD_METAL_KEEP_RESIDENCY` so the launcher can disable \
         Metal residency sets; without this, query/vsearch/embed dump a \
         stack trace at exit even on success."
            .to_owned(),
    );
}

/// The `content_vectors` fingerprint summary + mixed-named-fingerprints
/// checks (cli/qmd.ts:4262-4300). `model`/`fingerprint` are the
/// *declared* identity — see module doc.
fn check_embedding_fingerprints(
    qmd: &qmd::Qmd,
    embed_model: &str,
    fingerprint: &str,
    io: &mut Io<'_>,
    c: Palette,
    next_steps: &mut Vec<String>,
) {
    let rows = match qmd::doctor::fingerprint_groups(qmd) {
        Ok(rows) => rows,
        Err(err) => {
            doctor_check(io, c, "embedding fingerprints", false, &err.to_string());
            return;
        }
    };

    let mut unique_fingerprints: Vec<&str> = Vec::new();
    for row in &rows {
        if !unique_fingerprints.contains(&row.fingerprint.as_str()) {
            unique_fingerprints.push(&row.fingerprint);
        }
    }
    let off_current = rows
        .iter()
        .filter(|r| r.model == embed_model && r.fingerprint != fingerprint)
        .count();
    let ok = rows.is_empty()
        || (unique_fingerprints.len() == 1
            && rows.first().is_some_and(|r| r.fingerprint == fingerprint)
            && off_current == 0);
    let current_docs: usize = rows
        .iter()
        .filter(|r| r.model == embed_model && r.fingerprint == fingerprint)
        .map(|r| r.docs)
        .sum();
    let other_docs: usize = rows.iter().map(|r| r.docs).sum::<usize>() - current_docs;
    let groups = rows
        .iter()
        .map(|r| {
            let label = if r.fingerprint == fingerprint {
                "current"
            } else if r.fingerprint.is_empty() {
                "legacy"
            } else {
                r.fingerprint.as_str()
            };
            format!(
                "{}:{label} {} docs/{} chunks",
                short_model_name(&r.model),
                format_count(r.docs as u64),
                format_count(r.chunks as u64)
            )
        })
        .collect::<Vec<_>>()
        .join("; ");

    let named_rows: Vec<&qmd::doctor::FingerprintGroup> =
        rows.iter().filter(|r| !r.fingerprint.is_empty()).collect();
    let mut named_fingerprints: Vec<&str> = Vec::new();
    for row in &named_rows {
        if !named_fingerprints.contains(&row.fingerprint.as_str()) {
            named_fingerprints.push(&row.fingerprint);
        }
    }
    if named_fingerprints.len() > 1 {
        let named_groups = named_rows
            .iter()
            .map(|r| {
                format!(
                    "{}{}: {} {} docs/{} chunks",
                    r.fingerprint,
                    if r.fingerprint == fingerprint {
                        " (current)"
                    } else {
                        ""
                    },
                    short_model_name(&r.model),
                    format_count(r.docs as u64),
                    format_count(r.chunks as u64)
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        doctor_check(
            io,
            c,
            "mixed named embedding fingerprints",
            false,
            &format!(
                "content_vectors contains {} named fingerprints: {named_groups}. Next: \
                 `qmd embed` or `qmd embed --force`",
                named_fingerprints.len()
            ),
        );
        next_steps.push(
            "Run `qmd embed` to converge mixed named embedding fingerprints; use `qmd embed \
             --force` if old named fingerprints or vector sample mismatches remain."
                .to_owned(),
        );
    }

    let details = if rows.is_empty() {
        format!("no vectors yet; current fingerprint {fingerprint}")
    } else if ok {
        format!(
            "{} docs on current fingerprint ({fingerprint})",
            format_count(current_docs as u64)
        )
    } else {
        format!(
            "{} docs current, {} docs legacy/stale. {groups}. Next: `qmd embed`",
            format_count(current_docs as u64),
            format_count(other_docs as u64)
        )
    };
    doctor_check(io, c, "embedding fingerprints", ok, &details);
    if !ok {
        next_steps.push(
            "Run `qmd embed` to migrate active documents to the current embedding \
             fingerprint; use `qmd embed --force` if vector samples still fail afterward."
                .to_owned(),
        );
    }
}
