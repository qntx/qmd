//! Maintenance commands — upstream `init` (qmd.ts:447-477),
//! `status`/`showStatus` (qmd.ts:521-713), `update`/`updateCollections`
//! (qmd.ts:907-1019), `cleanup` (qmd.ts:4979-5019), `trust`/
//! `manageTrust` (qmd.ts:850-905), plus the shared
//! `reportSkippedReads`/`reportMetadataErrors`/`formatBytes`/
//! `formatOrphanedVectorHint` helpers (qmd.ts:433-438, 515-519,
//! 2089-2098).
//!
//! `update` is the only command that consults the local-config trust
//! gate *before* opening the store (upstream qmd.ts:909-910: "Prompt
//! before opening the store so an approval is visible to getStore").

use std::path::Path;

use qmd::{Environment, Qmd, SkippedFile, trust};

use crate::args::TrustArgs;
use crate::collections::format_time_ago;
use crate::context::CliContext;
use crate::output::{Io, Palette};
use crate::{Failure, gate, progress};

/// `qmd init` — upstream `initLocalIndex` (qmd.ts:447-477) dispatched
/// through `try/catch → exitWithError`: [`qmd::Error::LocalInitRefused`]
/// carries the refusal text verbatim and propagates as
/// [`Failure::Error`], so the doctor hint prints like upstream.
pub(crate) fn init(env: &Environment, io: &mut Io<'_>) -> Result<(), Failure> {
    // `const cwd = getPwd()` — $PWD first, like upstream.
    let cwd = qmd::paths::pwd(env);
    let _config_path = Qmd::init_local(env, &cwd)?;
    let _ = writeln!(io.out, "ready to go with new local index");
    Ok(())
}

/// `qmd status` — upstream `showStatus` (qmd.ts:521-713).
///
/// Deviations: no MCP-daemon block (the Rust CLI has no daemon), and
/// the AST Chunking block prints upstream's import-failure branch —
/// `Status:   not available` — because this port has no tree-sitter
/// module (the fixture soft-normalizes the line anyway).
pub(crate) fn status(env: &Environment, ctx: &CliContext, io: &mut Io<'_>) -> Result<(), Failure> {
    let c = Palette::new(Palette::enabled(env, io));
    // Upstream `status` deliberately displays the declared YAML models
    // (`resolveModelsForCli`), not the trust-gated runtime set.
    let qmd = ctx.open_store_with_models(env, true)?;
    let status = qmd.status()?;
    let collections = qmd.collections()?;
    let config = qmd.config()?;
    let index_size = std::fs::metadata(ctx.db_path()).map_or(0, |m| m.len());

    let _ = writeln!(io.out, "{}QMD Status{}\n", c.bold, c.reset);
    let _ = writeln!(io.out, "Index: {}", ctx.db_path().display());
    let _ = writeln!(io.out, "Size:  {}", format_bytes(index_size));
    let _ = writeln!(io.out);

    let _ = writeln!(io.out, "{}Documents{}", c.bold, c.reset);
    let _ = writeln!(
        io.out,
        "  Total:    {} files indexed",
        status.total_documents
    );
    let _ = writeln!(io.out, "  Vectors:  {} embedded", status.vector_count);
    if status.orphaned_vectors > 0 {
        // `vectorCount > 0 ? Math.round((orphaned / total) * 100) : 0`
        let pct = if status.vector_count > 0 {
            js_round(status.orphaned_vectors as f64 / status.vector_count as f64 * 100.0)
        } else {
            0
        };
        let _ = writeln!(
            io.out,
            "  {}Orphaned: {} embedding chunks ({}%){} — run 'qmd cleanup'",
            c.yellow, status.orphaned_vectors, pct, c.reset
        );
    }
    if status.needs_embedding > 0 {
        let _ = writeln!(
            io.out,
            "  {}Pending:  {} need embedding{} (run 'qmd embed')",
            c.yellow, status.needs_embedding, c.reset
        );
    }
    if status.pending_metadata > 0 {
        let _ = writeln!(
            io.out,
            "  {}Metadata: {} need extraction{} \
             (run 'qmd update'; excluded from --filter searches)",
            c.yellow, status.pending_metadata, c.reset
        );
    }
    let now = time::OffsetDateTime::now_utc();
    if status.latest_modified.is_some() {
        let _ = writeln!(
            io.out,
            "  Updated:  {}",
            format_time_ago(status.latest_modified.as_deref(), now)
        );
    }

    // `getASTStatus` import branch — no tree-sitter in this port.
    let _ = writeln!(io.out, "\n{}AST Chunking{}", c.bold, c.reset);
    let _ = writeln!(io.out, "  Status:   {}not available{}", c.dim, c.reset);

    // Contexts grouped by collection — `listAllContexts` (YAML).
    let mut contexts_by_collection: Vec<(&str, Vec<(&str, &str)>)> = Vec::new();
    if let Some(global) = config.global_context.as_deref() {
        push_context(&mut contexts_by_collection, "*", "/", global);
    }
    for (name, coll) in &config.collections {
        if let Some(map) = &coll.context {
            for (prefix, text) in map {
                push_context(
                    &mut contexts_by_collection,
                    name.as_str(),
                    prefix.as_str(),
                    text.as_str(),
                );
            }
        }
    }

    if collections.is_empty() {
        let _ = writeln!(
            io.out,
            "\n{}No collections. Run 'qmd collection add .' to index markdown files.{}",
            c.dim, c.reset
        );
    } else {
        print_collections_section(&collections, &contexts_by_collection, now, io, c);
    }

    // `hf:org/repo/file.gguf` → `https://huggingface.co/org/repo`.
    let models = qmd.ensure_models_configured();
    let _ = writeln!(io.out, "\n{}Models{}", c.bold, c.reset);
    let _ = writeln!(io.out, "  Embedding:   {}", hf_link(&models.embed));
    let _ = writeln!(io.out, "  Reranking:   {}", hf_link(&models.rerank));
    let _ = writeln!(io.out, "  Generation:  {}", hf_link(&models.generate));

    // Tips (qmd.ts:684-713): collections without any context, then
    // collections without an `update:` hook when there is more than one.
    let tips = status_tips(&collections, &contexts_by_collection, &config, c);
    if !tips.is_empty() {
        let _ = writeln!(io.out, "\n{}Tips{}", c.bold, c.reset);
        for tip in &tips {
            let _ = writeln!(io.out, "  {tip}");
        }
    }
    Ok(())
}

/// The `Collections`/`Examples` section of `status` (qmd.ts:574-660).
fn print_collections_section(
    collections: &[qmd::CollectionInfo],
    contexts_by_collection: &[(&str, Vec<(&str, &str)>)],
    now: time::OffsetDateTime,
    io: &mut Io<'_>,
    c: Palette,
) {
    let _ = writeln!(io.out, "\n{}Collections{}", c.bold, c.reset);
    for col in collections {
        let last_mod = col
            .last_modified
            .as_deref()
            .map_or_else(|| "never".to_owned(), |iso| format_time_ago(Some(iso), now));
        let _ = writeln!(
            io.out,
            "  {}{}{} {}(qmd://{}/){}",
            c.cyan, col.name, c.reset, c.dim, col.name, c.reset
        );
        let _ = writeln!(io.out, "    {}Pattern:{}  {}", c.dim, c.reset, col.pattern);
        let _ = writeln!(
            io.out,
            "    {}Files:{}    {} (updated {})",
            c.dim, c.reset, col.active_count, last_mod
        );
        let contexts = contexts_by_collection
            .iter()
            .find(|(n, _)| *n == col.name)
            .map_or(&[][..], |(_, v)| v.as_slice());
        print_context_entries(contexts, io, c);
    }
    let _ = writeln!(io.out, "\n{}Examples{}", c.bold, c.reset);
    let _ = writeln!(io.out, "  {}# List files in a collection{}", c.dim, c.reset);
    let first = collections.first().map_or("", |col| col.name.as_str());
    let _ = writeln!(io.out, "  qmd ls {first}");
    let _ = writeln!(io.out, "  {}# Get a document{}", c.dim, c.reset);
    let _ = writeln!(io.out, "  qmd get qmd://{first}/path/to/file.md");
    let _ = writeln!(io.out, "  {}# Search within a collection{}", c.dim, c.reset);
    let _ = writeln!(io.out, "  qmd search \"query\" -c {first}");
}

/// One collection's `Contexts:` block in `status` (qmd.ts:592-607).
fn print_context_entries(contexts: &[(&str, &str)], io: &mut Io<'_>, c: Palette) {
    if contexts.is_empty() {
        return;
    }
    let _ = writeln!(
        io.out,
        "    {}Contexts:{} {}",
        c.dim,
        c.reset,
        contexts.len()
    );
    for (prefix, text) in contexts {
        let display = if prefix.is_empty() || *prefix == "/" {
            "/".to_owned()
        } else {
            format!("/{prefix}")
        };
        let preview = context_preview(text);
        let _ = writeln!(
            io.out,
            "      {dim}{display}:{reset} {preview}",
            dim = c.dim,
            reset = c.reset
        );
    }
}

/// The `Tips` lines of `status` (qmd.ts:684-713).
fn status_tips(
    collections: &[qmd::CollectionInfo],
    contexts_by_collection: &[(&str, Vec<(&str, &str)>)],
    config: &qmd::Config,
    c: Palette,
) -> Vec<String> {
    let mut tips: Vec<String> = Vec::new();
    let without_context: Vec<&str> = collections
        .iter()
        .filter(|col| {
            !contexts_by_collection
                .iter()
                .any(|(n, v)| *n == col.name && !v.is_empty())
        })
        .map(|col| col.name.as_str())
        .collect();
    if !without_context.is_empty() {
        tips.push(format!(
            "Add context to collections for better search results: {}",
            join_first_three(&without_context)
        ));
        tips.push(format!(
            "  {c_dim}qmd context add qmd://<name>/ \"What this collection contains\"{c_reset}",
            c_dim = c.dim,
            c_reset = c.reset
        ));
        tips.push(format!(
            "  {c_dim}qmd context add qmd://<name>/meeting-notes \
             \"Weekly team meeting notes\"{c_reset}",
            c_dim = c.dim,
            c_reset = c.reset
        ));
    }
    if collections.len() > 1 {
        // `!yamlCol?.update` (qmd.ts:702) — JS truthiness treats an
        // empty `update: ""` the same as an absent key.
        let without_update: Vec<&str> = collections
            .iter()
            .filter(|col| {
                config
                    .collections
                    .get(&col.name)
                    .is_none_or(|yc| yc.update.as_deref().is_none_or(str::is_empty))
            })
            .map(|col| col.name.as_str())
            .collect();
        if !without_update.is_empty() {
            tips.push(format!(
                "Add update commands to keep collections fresh: {}",
                join_first_three(&without_update)
            ));
            tips.push(format!(
                "  {c_dim}qmd collection update-cmd <name> \
                 'git stash && git pull --rebase --ff-only && git stash pop'{c_reset}",
                c_dim = c.dim,
                c_reset = c.reset
            ));
        }
    }
    tips
}

/// `qmd update` — upstream `updateCollections` (qmd.ts:907-1019): the
/// trust gate runs first, `llm_cache` clears once, then each
/// `store_collections` row reindexes under its per-collection
/// `update:` hook interleaved exactly like upstream.
pub(crate) fn update(env: &Environment, ctx: &CliContext, io: &mut Io<'_>) -> Result<(), Failure> {
    let c = Palette::new(Palette::enabled(env, io));
    // Prompt before opening the store so an approval is visible to
    // getStore (#889).
    let allowed = gate::resolve_local_config_trust(env, ctx, io)?;
    let qmd = ctx.open_store(env)?;
    let collections = qmd.collections()?;
    if collections.is_empty() {
        let _ = writeln!(
            io.out,
            "{}No collections found. Run 'qmd collection add .' to index markdown files.{}",
            c.dim, c.reset
        );
        return Ok(());
    }
    // `clearCache(db)` — once per run, before the loop.
    qmd.maintenance().clear_llm_cache()?;
    let config_path = ctx.config_path(env);
    let config_path = config_path.to_string_lossy();

    let _ = writeln!(
        io.out,
        "{}Updating {} collection(s)...{}\n",
        c.bold,
        collections.len(),
        c.reset
    );
    for (i, col) in collections.iter().enumerate() {
        let _ = writeln!(
            io.out,
            "{}[{}/{}]{} {}{}{} {}({}){}",
            c.cyan,
            i + 1,
            collections.len(),
            c.reset,
            c.bold,
            col.name,
            c.reset,
            c.dim,
            col.pattern,
            c.reset
        );
        update_collection_entry(env, &qmd, col, &config_path, allowed, io, c)?;
    }

    let tail = qmd.update_tail_counts()?;
    let _ = writeln!(io.out, "{}✓ All collections updated.{}", c.green, c.reset);
    if tail.needs_embedding > 0 {
        let _ = writeln!(
            io.out,
            "\nRun 'qmd embed' to update embeddings ({} unique hashes need vectors)",
            tail.needs_embedding
        );
    }
    if tail.vector_total > 0
        && tail.orphaned_vectors as f64 / tail.vector_total as f64 >= ORPHAN_VECTOR_HINT_RATIO
    {
        let _ = writeln!(
            io.out,
            "\n{}",
            orphan_vector_hint(tail.orphaned_vectors, tail.vector_total)
        );
    }
    Ok(())
}

/// One `update` loop iteration (qmd.ts:929-984): the gated-path skip,
/// the trusted `update:` hook, then `indexFiles`-style reporting.
///
/// # Errors
/// [`Failure::Reported`] carrying the hook's exit code, or
/// [`qmd::Error`] from the collection read/reindex.
fn update_collection_entry(
    env: &Environment,
    qmd: &Qmd,
    col: &qmd::CollectionInfo,
    config_path: &str,
    allowed: bool,
    io: &mut Io<'_>,
    c: Palette,
) -> Result<(), Failure> {
    let yaml_col = qmd.collection(&col.name)?;
    let raw_path = yaml_col
        .as_ref()
        .map_or(col.path.as_str(), |yc| yc.path.as_str());
    if !allowed
        && trust::is_local_config_path(config_path, env)
        && !trust::is_collection_path_inside_project(config_path, raw_path, env)
    {
        let _ = writeln!(
            io.out,
            "{}Skipping collection '{}' — path {} is outside this project \
             and this .qmd config is not trusted.{}",
            c.yellow, col.name, raw_path, c.reset
        );
        let _ = writeln!(io.out, "{}Approve with 'qmd trust'.{}\n", c.dim, c.reset);
        return Ok(());
    }
    if let Some(cmd) = yaml_col.as_ref().and_then(|yc| yc.update.as_deref())
        && !cmd.is_empty()
        && allowed
    {
        run_update_hook_entry(cmd, Path::new(&col.path), io, c)?;
    }
    // `startTime` anchors before the `Collection:` line (qmd.ts:981-983).
    let start = std::time::Instant::now();
    let _ = writeln!(io.out, "Collection: {} ({})", col.path, col.pattern);
    progress::indeterminate(io);
    let report = qmd.update_collection(&col.name, &mut |p| {
        progress::indexing_tick(io, p.current, p.total, start);
    })?;
    // `progress.clear()` then `\nIndexed:` (qmd.ts:998-999) — upstream
    // `reindexCollection` never prints "No files found matching
    // pattern."; only `indexFiles` does.
    progress::clear(io);
    let _ = writeln!(
        io.out,
        "\nIndexed: {} new, {} updated, {} unchanged, {} removed",
        report.indexed, report.updated, report.unchanged, report.removed
    );
    report_skipped_reads(&report.skipped_files, io);
    report_metadata_errors(report.metadata_errors, io);
    if report.orphaned_cleaned > 0 {
        let _ = writeln!(
            io.out,
            "Cleaned up {} orphaned content hash(es)",
            report.orphaned_cleaned
        );
    }
    let _ = writeln!(io.out);
    Ok(())
}

/// The `Running update command:` branch (qmd.ts:946-977): the hook's
/// exit code becomes the process code, like `process.exit(exitCode)`.
///
/// # Errors
/// [`Failure::Reported`] on a non-zero or failed-to-spawn hook.
fn run_update_hook_entry(
    cmd: &str,
    cwd: &Path,
    io: &mut Io<'_>,
    c: Palette,
) -> Result<(), Failure> {
    let _ = writeln!(
        io.out,
        "{}    Running update command: {cmd}{}",
        c.dim, c.reset
    );
    match qmd::update_hook::run_update_hook(cmd, cwd) {
        Ok(out) => {
            print_hook_output(&out.stdout, io);
            print_hook_output(&out.stderr, io);
            if out.code != 0 {
                let _ = writeln!(
                    io.out,
                    "{}✗ Update command failed with exit code {}{}",
                    c.yellow, out.code, c.reset
                );
                return Err(Failure::Reported(u8::try_from(out.code).unwrap_or(1)));
            }
            Ok(())
        }
        Err(err) => {
            let _ = writeln!(
                io.out,
                "{}✗ Update command failed: {err}{}",
                c.yellow, c.reset
            );
            Err(Failure::Reported(1))
        }
    }
}

/// `qmd cleanup [--dry-run]` — upstream `case "cleanup"`
/// (qmd.ts:4979-5019): `previewCleanup`/`runCleanup` rendered with the
/// same conditional lines.
pub(crate) fn cleanup(
    env: &Environment,
    ctx: &CliContext,
    dry_run: bool,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    let c = Palette::new(Palette::enabled(env, io));
    let qmd = ctx.open_store(env)?;
    let maintenance = qmd.maintenance();
    if dry_run {
        let stats = maintenance.preview()?;
        let _ = writeln!(io.out, "Dry run — no changes made.\n");
        let _ = writeln!(
            io.out,
            "Would clear {} cached API responses",
            stats.cache_count
        );
        if stats.orphaned_vectors > 0 {
            let _ = writeln!(
                io.out,
                "Would remove {} orphaned embedding chunks",
                stats.orphaned_vectors
            );
        } else {
            let _ = writeln!(
                io.out,
                "{}No orphaned embeddings to remove{}",
                c.dim, c.reset
            );
        }
        if stats.inactive_docs > 0 {
            let _ = writeln!(
                io.out,
                "Would remove {} inactive document records",
                stats.inactive_docs
            );
        }
        if stats.orphaned_content > 0 {
            let _ = writeln!(
                io.out,
                "Would remove {} orphaned content hashes",
                stats.orphaned_content
            );
        }
        let _ = writeln!(io.out, "Would compact FTS and vacuum the database");
        return Ok(());
    }
    let stats = maintenance.run()?;
    let _ = writeln!(
        io.out,
        "{}✓{} Cleared {} cached API responses",
        c.green, c.reset, stats.cache_count
    );
    if stats.orphaned_vectors > 0 {
        let _ = writeln!(
            io.out,
            "{}✓{} Removed {} orphaned embedding chunks",
            c.green, c.reset, stats.orphaned_vectors
        );
    } else {
        let _ = writeln!(
            io.out,
            "{}No orphaned embeddings to remove{}",
            c.dim, c.reset
        );
    }
    if stats.inactive_docs > 0 {
        let _ = writeln!(
            io.out,
            "{}✓{} Removed {} inactive document records",
            c.green, c.reset, stats.inactive_docs
        );
    }
    if stats.orphaned_content > 0 {
        let _ = writeln!(
            io.out,
            "{}✓{} Removed {} orphaned content hashes",
            c.green, c.reset, stats.orphaned_content
        );
    }
    let _ = writeln!(
        io.out,
        "{}✓{} FTS compacted, database vacuumed",
        c.green, c.reset
    );
    Ok(())
}

/// `qmd trust [list|revoke]` — upstream `manageTrust`
/// (qmd.ts:850-905): inspect, approve or drop the approval for the
/// project-local config in scope.
pub(crate) fn trust(
    env: &Environment,
    ctx: &CliContext,
    args: &TrustArgs,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    let c = Palette::new(Palette::enabled(env, io));
    let config_path = ctx.config_path(env);
    let config_path_str = config_path.to_string_lossy();
    match args.args.first().map(String::as_str) {
        Some("list") => {
            let records = trust::list_trusted(env);
            if records.is_empty() {
                let _ = writeln!(io.out, "{}No trusted project configs.{}", c.dim, c.reset);
            } else {
                for record in &records {
                    // A malformed record renders `undefined`, like JS.
                    let _ = writeln!(
                        io.out,
                        "{} {}(trusted {}){}",
                        record.path,
                        c.dim,
                        record.trusted_at.as_deref().unwrap_or("undefined"),
                        c.reset
                    );
                }
            }
            Ok(())
        }
        Some("revoke") => {
            if !trust::is_local_config_path(&config_path_str, env) {
                let _ = writeln!(
                    io.out,
                    "{}No project-local .qmd config in scope — {} is your own config \
                     and is never gated.{}",
                    c.dim, config_path_str, c.reset
                );
                let _ = writeln!(
                    io.out,
                    "{}Run this from inside the project, or see 'qmd trust list'.{}",
                    c.dim, c.reset
                );
                return Ok(());
            }
            if trust::revoke_trust(env, &config_path_str)? {
                let _ = writeln!(
                    io.out,
                    "{}✓ Revoked trust for {}{}",
                    c.green, config_path_str, c.reset
                );
            } else {
                let _ = writeln!(
                    io.out,
                    "{}{} was not trusted.{}",
                    c.dim, config_path_str, c.reset
                );
            }
            Ok(())
        }
        Some(_) => {
            let _ = writeln!(io.err, "Usage: qmd trust [list|revoke]");
            Err(Failure::Reported(1))
        }
        None => {
            if !trust::is_local_config_path(&config_path_str, env) {
                let _ = writeln!(
                    io.out,
                    "{}{} is your own config — it is never gated. Nothing to trust.{}",
                    c.dim, config_path_str, c.reset
                );
                return Ok(());
            }
            let config = qmd::config::load(&config_path)?;
            let snapshot = trust::collect_sensitive_snapshot(&config);
            let gated = trust::local_config_gated(&config_path, &snapshot, env);
            if !trust::has_gated_items(&gated) {
                let _ = writeln!(
                    io.out,
                    "{}{} defines no update commands, out-of-project collection paths, \
                     or custom models. Nothing to trust.{}",
                    c.dim, config_path_str, c.reset
                );
                return Ok(());
            }
            gate::print_gated_items(&gated, io, c);
            let digest = trust::local_config_digest(&config_path, &snapshot, env);
            trust::record_trust(env, &config_path_str, &digest)?;
            let _ = writeln!(
                io.out,
                "{}✓ Trusted {}{}",
                c.green, config_path_str, c.reset
            );
            let _ = writeln!(
                io.out,
                "{}Editing a hook, out-of-project path, or custom model will ask again. \
                 Revoke with 'qmd trust revoke'.{}",
                c.dim, c.reset
            );
            Ok(())
        }
    }
}

/// Upstream `reportSkippedReads` (qmd.ts:2094-2107) — `console.warn`
/// goes to stderr.
pub(crate) fn report_skipped_reads(skipped: &[SkippedFile], io: &mut Io<'_>) {
    if skipped.is_empty() {
        return;
    }
    for file in skipped {
        if file.code == "OUTSIDE_COLLECTION" {
            let _ = writeln!(io.err, "⚠ Skipped file outside collection: {}", file.file);
        } else {
            let _ = writeln!(
                io.err,
                "⚠ Skipped unreadable file: {} ({})",
                file.file, file.code
            );
        }
    }
    let escaped = skipped
        .iter()
        .filter(|f| f.code == "OUTSIDE_COLLECTION")
        .count();
    let unreadable = skipped.len() - escaped;
    if escaped > 0 {
        let _ = writeln!(
            io.err,
            "Skipped {escaped} file(s) outside the collection root"
        );
    }
    if unreadable > 0 {
        let _ = writeln!(io.err, "Skipped {unreadable} unreadable file(s)");
    }
}

/// Upstream `reportMetadataErrors` (qmd.ts:2089-2092).
pub(crate) fn report_metadata_errors(count: usize, io: &mut Io<'_>) {
    if count == 0 {
        return;
    }
    let _ = writeln!(
        io.err,
        "⚠ {count} file(s) have invalid qmd.metadata frontmatter and are excluded \
         from filtered search"
    );
}

/// Captured hook stdout/stderr — `output.trim()` then each line
/// re-indented four spaces (qmd.ts:963-968).
fn print_hook_output(output: &str, io: &mut Io<'_>) {
    let trimmed = output.trim();
    if trimmed.is_empty() {
        return;
    }
    for line in trimmed.split('\n') {
        let _ = writeln!(io.out, "    {line}");
    }
}

/// Upstream `formatBytes` (qmd.ts:433-438): `B` under 1 KiB, then
/// `KB`/`MB`/`GB` with one decimal — `toFixed(1)` rounds half-up, so
/// [`qmd::js::to_fixed`] rather than Rust's ties-to-even `{:.1}`
/// (1.25 KB prints `1.3 KB` upstream).
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;
    let b = bytes as f64;
    if bytes < KB {
        format!("{bytes} B")
    } else if bytes < MB {
        format!("{} KB", qmd::js::to_fixed(b / 1024.0, 1))
    } else if bytes < GB {
        format!("{} MB", qmd::js::to_fixed(b / (1024.0 * 1024.0), 1))
    } else {
        format!(
            "{} GB",
            qmd::js::to_fixed(b / (1024.0 * 1024.0 * 1024.0), 1)
        )
    }
}

/// `ORPHAN_VECTOR_HINT_RATIO` (qmd.ts:515).
const ORPHAN_VECTOR_HINT_RATIO: f64 = 0.1;

/// Upstream `formatOrphanedVectorHint` (qmd.ts:517-519).
fn orphan_vector_hint(orphaned: usize, total: usize) -> String {
    let pct = if total > 0 {
        js_round(orphaned as f64 / total as f64 * 100.0)
    } else {
        0
    };
    format!(
        "{orphaned} orphaned embedding chunks ({pct}% of vectors) — \
         run 'qmd cleanup' to reclaim space"
    )
}

/// `Math.round` — half away from zero.
#[allow(
    clippy::missing_const_for_fn,
    clippy::cast_possible_truncation,
    reason = "f64::round is not const; percentages are small ints"
)]
fn js_round(value: f64) -> i64 {
    value.round() as i64
}

/// `hfLink` (qmd.ts:666-670): `hf:org/repo/…` →
/// `https://huggingface.co/org/repo`, anything else verbatim.
fn hf_link(uri: &str) -> String {
    if let Some(rest) = uri.strip_prefix("hf:") {
        let mut parts = rest.splitn(3, '/');
        if let (Some(org), Some(repo), Some(_)) = (parts.next(), parts.next(), parts.next()) {
            return format!("https://huggingface.co/{org}/{repo}");
        }
    }
    uri.to_owned()
}

/// `ctx.context.substring(0, 57) + '...'` for contexts over 60 UTF-16
/// units — whole chars only (a split surrogate cannot be represented
/// in Rust; no fixture context approaches the limit anyway).
fn context_preview(text: &str) -> String {
    if qmd::utf16_len(text) <= 60 {
        return text.to_owned();
    }
    let mut out = String::new();
    let mut units = 0usize;
    for ch in text.chars() {
        let w = ch.len_utf16();
        if units + w > 57 {
            break;
        }
        units += w;
        out.push(ch);
    }
    format!("{out}...")
}

/// Upstream `listAllContexts` grouping (store.ts:1118-1132) — global
/// context first (`collection "*"` / `path "/"`), then each
/// collection's `context:` map in config order.
fn push_context<'a>(
    groups: &mut Vec<(&'a str, Vec<(&'a str, &'a str)>)>,
    name: &'a str,
    prefix: &'a str,
    text: &'a str,
) {
    match groups.iter_mut().find(|(n, _)| *n == name) {
        Some((_, entries)) => entries.push((prefix, text)),
        None => groups.push((name, vec![(prefix, text)])),
    }
}

/// `names.slice(0, 3).join(', ')` + ` +${len-3} more` (qmd.ts:691-692
/// and 706-707).
fn join_first_three(names: &[&str]) -> String {
    let shown = names.iter().take(3).copied().collect::<Vec<_>>().join(", ");
    if names.len() > 3 {
        format!("{shown} +{} more", names.len() - 3)
    } else {
        shown
    }
}
