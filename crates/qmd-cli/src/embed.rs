//! `qmd embed` and `qmd pull` — upstream `case "embed"`/`case "pull"`
//! (cli/qmd.ts:4705-4751) → `vectorIndex` (cli/qmd.ts:2189-2320) and
//! `pullModels` (llm.ts:523-580).
//!
//! `embed` mirrors upstream's dispatch order: `resolveLocalConfigTrust`,
//! the `Number()`-based option validators, `resolveCollectionFilter`
//! against the YAML config, then `vectorIndex` — store open, exclusive
//! embed lock, force banner, pending-hash pre-check, model/batch lines,
//! and the byte-progress `generateEmbeddings` run. Both batch options
//! and the collection filter are validated *before* the store opens,
//! like upstream.
//!
//! `pull` never opens the database (upstream doesn't either); it pins
//! resolved models into the config like `ensureModelsConfiguredForCli`,
//! picks `resolveModelsForRuntime`, and streams downloads into the
//! model cache. Upstream delegates the `--progress` bar to
//! node-llama-cpp's `resolveModelFile`; this port renders an
//! equivalent byte-progress line on stderr (TTY only) — the bar text
//! itself is deliberately not ported 1:1.

use std::time::{Duration, Instant};

use qmd::Environment;
use qmd::llm::ResolvedModels;

use crate::args::{EmbedArgs, PullArgs};
use crate::context::CliContext;
use crate::embed_lock::EmbedLock;
use crate::output::{
    Io, Palette, format_bytes, format_count, render_progress_bar, short_model_name,
};
use crate::{Failure, gate, progress};

/// `qmd embed` — upstream `case "embed"` (cli/qmd.ts:4705-4728).
///
/// # Errors
/// [`Failure::Reported`] for `Collection not found: <name>`;
/// [`Failure::Error`] carries option-validation and store/embed
/// failures through upstream's `exitWithError` path.
pub(crate) fn embed(
    env: &Environment,
    ctx: &CliContext,
    args: &EmbedArgs,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    gate::resolve_local_config_trust(env, ctx, io)?;
    // parseEmbedBatchOption / parseChunkStrategy / parseEmbedTimeoutOption
    // run before resolveCollectionFilter upstream (cli/qmd.ts:4708-4712).
    // Upstream actually validates `--chunk-strategy` earlier still —
    // `cli.opts` inside `parseCLI` (qmd.ts:3213) throws before dispatch,
    // so a bad value produces a Bun dump, not `exitWithError`. The
    // case-level call (qmd.ts:4710) is the intended handler and its
    // `exitWithError` shape is what this port follows (P4-D2 records
    // the dump instead).
    let max_docs_per_batch =
        parse_embed_batch_option("maxDocsPerBatch", args.max_docs_per_batch.as_deref(), io)?;
    let max_batch_mb = parse_embed_batch_option("maxBatchBytes", args.max_batch_mb.as_deref(), io)?;
    let chunk_strategy = parse_chunk_strategy(args.chunk_strategy.as_deref(), io)?;
    let max_duration = parse_embed_timeout(args.timeout.as_deref(), io)?;

    // `resolveCollectionFilter(collection, false)` (cli/qmd.ts:4716):
    // every `-c` value is checked against the YAML config
    // (`getCollectionFromYaml`), only the first scopes the embed.
    let config_path = ctx.config_path(env);
    let config = qmd::config::load(&config_path)?;
    for name in &args.collection {
        if !config.collections.contains_key(name) {
            let _ = writeln!(io.err, "Collection not found: {name}");
            return Err(Failure::Reported(1));
        }
    }
    let collection = args.collection.first().cloned();

    // `vectorIndex` body (cli/qmd.ts:2189-2320): store first, then the
    // exclusive lock; a busy lock is a quiet skip, not an error.
    let qmd = ctx.open_store(env)?;
    let Some(_lock) = EmbedLock::try_acquire(ctx.db_path()).map_err(|source| {
        Failure::Error(qmd::Error::Io {
            path: EmbedLock::path_for_db(ctx.db_path()),
            source,
        })
    })?
    else {
        let _ = writeln!(io.out, "{}", crate::embed_lock::BUSY_MESSAGE);
        return Ok(());
    };

    let c = Palette::new(Palette::enabled(env, io));
    if args.force {
        let _ = writeln!(
            io.out,
            "{}Force re-indexing: clearing all vectors...{}",
            c.yellow, c.reset
        );
    }

    // `getHashesNeedingEmbedding(db, collection, model)` — the runtime
    // model (trust-gated via `open_store`) inside `Qmd`.
    let pending = qmd.hashes_needing_embedding(collection.as_deref())?;
    if pending == 0 && !args.force {
        let _ = writeln!(
            io.out,
            "{}✓ All content hashes already have embeddings.{}",
            c.green, c.reset
        );
        return Ok(());
    }

    let model = qmd::doctor::resolved_embed_model(&qmd)?;
    let _ = writeln!(
        io.out,
        "{}Model: {}{}\n",
        c.dim,
        short_model_name(&model),
        c.reset
    );
    let defaults = qmd::EmbedOptions::default();
    if max_docs_per_batch.is_some() || max_batch_mb.is_some() {
        let docs = max_docs_per_batch.unwrap_or(defaults.max_docs_per_batch);
        let bytes = max_batch_mb.map_or(defaults.max_batch_bytes, |mb| {
            mb.saturating_mul(1024 * 1024)
        });
        let _ = writeln!(
            io.out,
            "{}Batch: {} docs / {}{}\n",
            c.dim,
            docs,
            format_bytes(u64::try_from(bytes).unwrap_or(u64::MAX)),
            c.reset
        );
    }

    progress::cursor_hide(io);
    progress::indeterminate(io);
    let start = Instant::now();
    let stderr_tty = io.stderr_tty;
    let report = qmd.embed(
        &qmd::EmbedOptions {
            force: args.force,
            collection,
            max_docs_per_batch: max_docs_per_batch.unwrap_or(defaults.max_docs_per_batch),
            max_batch_bytes: max_batch_mb.map_or(defaults.max_batch_bytes, |mb| {
                mb.saturating_mul(1024 * 1024)
            }),
            chunk_strategy,
            max_duration,
            ..defaults
        },
        &mut |p| embed_progress_tick(p, start, io, c, stderr_tty),
    )?;
    // On error upstream skips `progress.clear()`/`cursor.show()` — the
    // throw jumps to `finally { embedLock.release() }` (cli/qmd.ts:2264,
    // 2287), leaving the cursor hidden on the error path.
    progress::clear(io);
    progress::cursor_show(io);

    if report.chunks_embedded == 0 && report.docs_processed == 0 {
        let _ = writeln!(
            io.out,
            "{}✓ No non-empty documents to embed.{}",
            c.green, c.reset
        );
        return Ok(());
    }
    print_embed_summary(&report, io, c);
    Ok(())
}

/// `vectorIndex`'s post-run block (cli/qmd.ts:2267-2290): the 100% bar,
/// the `✓ Done!` totals line, then up to 8 failure rows plus an
/// `...and N more` tail.
fn print_embed_summary(report: &qmd::EmbedReport, io: &mut Io<'_>, c: Palette) {
    let _ = writeln!(
        io.out,
        "\r{}{}{} {}100%{}                                    ",
        c.green,
        render_progress_bar(100.0, 30),
        c.reset,
        c.bold,
        c.reset
    );
    let _ = writeln!(
        io.out,
        "\n{}✓ Done!{} Embedded {}{}{} chunks from {}{}{} documents in {}{}{}",
        c.green,
        c.reset,
        c.bold,
        format_count(report.chunks_embedded as u64),
        c.reset,
        c.bold,
        format_count(report.docs_processed as u64),
        c.reset,
        c.bold,
        progress::format_eta(report.duration.as_secs_f64()),
        c.reset
    );
    if report.errors == 0 {
        return;
    }
    let _ = writeln!(
        io.out,
        "{}⚠ {} chunks still failed after retries{}",
        c.yellow,
        format_count(report.errors as u64),
        c.reset
    );
    for failure in report.failures.iter().take(8) {
        let _ = writeln!(
            io.out,
            "  {}{}#{} ({} attempts): {}{}",
            c.dim, failure.path, failure.seq, failure.attempts, failure.reason, c.reset
        );
    }
    if report.failures.len() > 8 {
        let _ = writeln!(
            io.out,
            "  {}...and {} more{}",
            c.dim,
            format_count(report.failures.len() as u64 - 8),
            c.reset
        );
    }
}

/// The `onProgress` body of `vectorIndex` (cli/qmd.ts:2249-2265):
/// input-byte percentage with throughput + ETA, TTY stderr only. The
/// OSC 9;4 `progress.set` fires even though the `\r` line is TTY-gated
/// — both live behind the same `isTTY` flag upstream.
fn embed_progress_tick(
    p: &qmd::EmbedProgress,
    start: Instant,
    io: &mut Io<'_>,
    c: Palette,
    stderr_tty: bool,
) {
    if p.total_bytes == 0 {
        return;
    }
    #[allow(clippy::cast_precision_loss, reason = "byte counts fit f64")]
    let percent = (p.bytes_processed as f64 / p.total_bytes as f64 * 100.0).min(100.0);
    progress::set(io, percent);
    let elapsed = start.elapsed().as_secs_f64();
    let bytes_per_sec = if elapsed > 0.0 {
        p.bytes_processed as f64 / elapsed
    } else {
        0.0
    };
    let remaining = p.total_bytes.saturating_sub(p.bytes_processed);
    let eta_sec = if bytes_per_sec > 0.0 {
        remaining as f64 / bytes_per_sec
    } else {
        f64::INFINITY
    };
    let eta = if elapsed > 2.0 && eta_sec.is_finite() {
        progress::format_eta(eta_sec)
    } else {
        "...".to_owned()
    };
    let err_str = if p.errors > 0 {
        format!(
            " {}{} err{}",
            c.yellow,
            format_count(p.errors as u64),
            c.reset
        )
    } else {
        String::new()
    };
    if stderr_tty {
        // Precomputed so the `write!` args stay format-free.
        let pct_str = format!("{:>3}", qmd::js::to_fixed(percent, 0));
        let chunks_str = format_count(p.chunks_embedded as u64);
        let done_str = format_bytes(p.bytes_processed);
        let total_str = format_bytes(p.total_bytes);
        let rate_str = if bytes_per_sec > 0.0 {
            format_bytes_rate(bytes_per_sec)
        } else {
            // Upstream's `bytesPerSec > 0 ? … : ".../s"` arm.
            ".../s".to_owned()
        };
        let _ = write!(
            io.err,
            "\r{}{}{} {}{}% input{} {}{} chunks{} · {}/{} input · {} · ETA {}{}   ",
            c.cyan,
            render_progress_bar(percent, 30),
            c.reset,
            c.bold,
            pct_str,
            c.reset,
            c.dim,
            chunks_str,
            err_str,
            done_str,
            total_str,
            rate_str,
            eta,
            c.reset
        );
    }
}

/// `${formatBytes(bytesPerSec)}/s` (cli/qmd.ts:2258) — upstream's
/// `${bytes} B` arm interpolates the raw float (`123.7 B/s`); the
/// integer helper cannot.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the >1024 arm only casts finite positive rates"
)]
fn format_bytes_rate(bytes_per_sec: f64) -> String {
    if bytes_per_sec < 1024.0 {
        // `${bytes} B` — upstream prints the raw float.
        return format!("{} B/s", qmd::js::number_to_string(bytes_per_sec));
    }
    format!("{}/s", format_bytes(bytes_per_sec as u64))
}

/// `qmd pull` — upstream `case "pull"` (cli/qmd.ts:4730-4751).
///
/// # Errors
/// [`Failure::Error`] wraps `pull_models` download/validation failures
/// (upstream lets them reach `exitWithError` too).
pub(crate) fn pull(
    env: &Environment,
    ctx: &CliContext,
    args: &PullArgs,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    let trusted = gate::resolve_local_config_trust(env, ctx, io)?;
    let active = runtime_models(env, ctx, trusted);
    let c = Palette::new(Palette::enabled(env, io));
    let _ = writeln!(io.out, "{}Pulling models{}", c.bold, c.reset);
    let models = [active.embed, active.generate, active.rerank];
    let show_progress = args.progress && io.stderr_tty;
    let mut current_model = String::new();
    let results = qmd::llm::download::pull_models_with_progress(
        &models,
        env,
        None,
        args.refresh,
        &mut |model, done, total| {
            if !show_progress {
                return;
            }
            if current_model != model {
                // Fresh line per model so bars don't overwrite each other.
                if !current_model.is_empty() {
                    let _ = writeln!(io.err);
                }
                model.clone_into(&mut current_model);
            }
            #[allow(clippy::cast_precision_loss, reason = "byte counts fit f64")]
            let percent = total.map_or(0.0, |t| {
                if t == 0 {
                    100.0
                } else {
                    (done as f64 / t as f64 * 100.0).min(100.0)
                }
            });
            let total_str = total.map_or_else(|| "?".to_owned(), format_bytes);
            let _ = write!(
                io.err,
                "\r{} {}{}{} {}/{}",
                short_model_name(model),
                c.dim,
                render_progress_bar(percent, 30),
                c.reset,
                format_bytes(done),
                total_str,
            );
        },
    )?;
    if show_progress && !current_model.is_empty() {
        let _ = writeln!(io.err);
    }
    for result in results {
        let note = if result.refreshed {
            "refreshed"
        } else {
            "cached/checked"
        };
        let _ = writeln!(
            io.out,
            "- {} -> {} ({}, {})",
            result.model,
            result.path.display(),
            format_bytes(result.size_bytes),
            note
        );
    }
    Ok(())
}

/// Upstream `exitWithError(new Error(message))` (cli/qmd.ts:4329-4333):
/// the verbatim message plus `printDoctorHint` on stderr, exit 1.
fn exit_with_error(message: &str, io: &mut Io<'_>) -> Failure {
    let _ = writeln!(io.err, "{message}");
    let _ = writeln!(io.err, "{}", crate::DOCTOR_HINT);
    Failure::Reported(1)
}

/// `parseEmbedBatchOption` (cli/qmd.ts:2116-2123): `Number()` semantics —
/// a non-integer or `<1` value throws `"{name} must be a positive
/// integer"`, which upstream's `catch` maps to `exitWithError`.
///
/// # Errors
/// [`Failure::Reported`] after the upstream message + hint.
fn parse_embed_batch_option(
    name: &str,
    raw: Option<&str>,
    io: &mut Io<'_>,
) -> Result<Option<usize>, Failure> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let parsed = qmd::js::number(raw);
    let valid = parsed.is_some_and(|v| v.fract() == 0.0 && v >= 1.0);
    if !valid {
        return Err(exit_with_error(
            &format!("{name} must be a positive integer"),
            io,
        ));
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "validated ≥1 integer; huge values saturate like upstream"
    )]
    Ok(Some(parsed.unwrap_or_default() as usize))
}

/// `parseChunkStrategy` (cli/qmd.ts:2125-2130) — the `embed` call site
/// (qmd.ts:4710). Upstream's parse-time throw at qmd.ts:3213 actually
/// wins for bad values (Bun dump); we render the intended
/// `exitWithError` shape like the batch/timeout validators.
///
/// # Errors
/// [`Failure::Reported`] after the upstream message + hint.
fn parse_chunk_strategy(raw: Option<&str>, io: &mut Io<'_>) -> Result<qmd::ChunkStrategy, Failure> {
    match raw {
        Some("auto") => Ok(qmd::ChunkStrategy::Auto),
        None | Some("regex") => Ok(qmd::ChunkStrategy::Regex),
        Some(other) => Err(exit_with_error(
            &format!("--chunk-strategy must be \"auto\" or \"regex\" (got \"{other}\")"),
            io,
        )),
    }
}

/// `parseEmbedTimeoutOption` (cli/qmd.ts:2134-2141): `Number()` minutes,
/// `>= 0`; `0` disables the cap, `None` keeps the 30-minute default.
///
/// # Errors
/// [`Failure::Reported`] after the upstream message + hint.
fn parse_embed_timeout(raw: Option<&str>, io: &mut Io<'_>) -> Result<Option<Duration>, Failure> {
    let Some(raw) = raw else {
        return Ok(Some(
            qmd::EmbedOptions::default()
                .max_duration
                .unwrap_or_default(),
        ));
    };
    let minutes = qmd::js::number(raw);
    let valid = minutes.is_some_and(|m| m.is_finite() && m >= 0.0);
    if !valid {
        return Err(exit_with_error(
            "--timeout must be a non-negative number of minutes (0 = no limit)",
            io,
        ));
    }
    let minutes = minutes.unwrap_or_default();
    if minutes == 0.0 {
        // `maxDuration <= 0` skips the timer upstream.
        return Ok(None);
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "validated non-negative finite; huge values saturate"
    )]
    Ok(Some(Duration::from_secs_f64(minutes * 60.0)))
}

/// `resolveModelsForRuntime` (cli/qmd.ts:2183-2187): the pinned declared
/// triple while the local config is fully trusted, else env/defaults —
/// `pull` resolves without opening the store.
fn runtime_models(env: &Environment, ctx: &CliContext, trusted: bool) -> ResolvedModels {
    let configured = ensure_models_configured_cli(env, ctx);
    if trusted {
        configured
    } else {
        qmd::llm::resolve_models(None, env)
    }
}

/// `ensureModelsConfiguredForCli` (cli/qmd.ts:2143-2163) without the
/// store: resolve the declared triple, pin it into the config file when
/// it differs, and swallow the write failure like upstream's `catch` —
/// which falls back to `resolveModels()` (env/defaults).
fn ensure_models_configured_cli(env: &Environment, ctx: &CliContext) -> ResolvedModels {
    let config_path = ctx.config_path(env);
    let Ok(mut config) = qmd::config::load(&config_path) else {
        return qmd::llm::resolve_models(None, env);
    };
    let models = qmd::llm::resolve_models(config.models.as_ref(), env);
    let differs = config.models.as_ref().is_none_or(|m| {
        m.embed.as_deref() != Some(models.embed.as_str())
            || m.generate.as_deref() != Some(models.generate.as_str())
            || m.rerank.as_deref() != Some(models.rerank.as_str())
    });
    if differs {
        // Upstream spreads `...current` — `embed_pooling`/`extra` keys
        // survive the pin.
        let m = config.models.get_or_insert_default();
        m.embed = Some(models.embed.clone());
        m.generate = Some(models.generate.clone());
        m.rerank = Some(models.rerank.clone());
        let _ = qmd::config::save(&config_path, &config);
    }
    models
}
