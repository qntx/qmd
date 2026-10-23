//! `qmd collection` subcommands — upstream `case "collection"`
//! (cli/qmd.ts:4513-4679) plus `collectionList` (qmd.ts:1787-1821),
//! `collectionAdd` (qmd.ts:1828-1868), `collectionRemove`/`Rename`
//! (qmd.ts:1870-1920), the `show`/`update-cmd`/`include`/`exclude`
//! inline branches, and `formatTimeAgo` (qmd.ts:417-427).
//!
//! Store opening is lazy like upstream: `update-cmd`, `include`,
//! `exclude` and `show` are pure-YAML through `loadConfig`/
//! `updateCollectionSettings`/`getCollection` (collections.ts:183-309)
//! and never create `index.sqlite`; `add`/`remove`/`rename` run their
//! YAML existence checks before `open_store`, so validation errors do
//! not create the database either. Only `list`, a successful `add`
//! index pass, and `remove`/`rename` mutations actually open the store
//! — upstream's `getDb`/`resyncConfig` call sites.

use std::path::Path;

use qmd::config::DEFAULT_PATTERN;
use qmd::{Environment, Error, Qmd};

use crate::args::CollectionArgs;
use crate::context::CliContext;
use crate::output::{Io, Palette};
use crate::{DOCTOR_HINT, Failure, gate, progress};

/// Dispatch `cli.args[0]` — upstream's `switch (subcommand)`
/// (qmd.ts:4514-4679). Positionals arrive flat like `cli.args`, so
/// subcommand-specific arity/usage errors are produced here rather
/// than by clap, matching upstream's hand-rolled dispatch.
pub(crate) fn run(
    env: &Environment,
    ctx: &CliContext,
    args: &CollectionArgs,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    let c = Palette::new(Palette::enabled(env, io));
    let argv = &args.args;
    let sub = argv.first().map_or("help", String::as_str);
    match sub {
        "list" => list(env, ctx, io, c),
        "add" => add(env, ctx, argv, args, io, c),
        "remove" | "rm" => {
            let Some(name) = argv.get(1) else {
                let _ = writeln!(io.err, "Usage: qmd collection remove <name>");
                let _ = writeln!(
                    io.err,
                    "  Use 'qmd collection list' to see available collections"
                );
                return Err(Failure::Reported(1));
            };
            remove(env, ctx, name, io, c)
        }
        "rename" | "mv" => {
            let (Some(old), Some(new)) = (argv.get(1), argv.get(2)) else {
                let _ = writeln!(io.err, "Usage: qmd collection rename <old-name> <new-name>");
                let _ = writeln!(
                    io.err,
                    "  Use 'qmd collection list' to see available collections"
                );
                return Err(Failure::Reported(1));
            };
            rename(env, ctx, old, new, io, c)
        }
        "set-update" | "update-cmd" => {
            let Some(name) = argv.get(1) else {
                let _ = writeln!(io.err, "Usage: qmd collection update-cmd <name> [command]");
                let _ = writeln!(
                    io.err,
                    "  Set the command to run before indexing (e.g., 'git pull')"
                );
                let _ = writeln!(io.err, "  Omit command to clear it");
                return Err(Failure::Reported(1));
            };
            // `cli.args.slice(2).join(' ') || null` — the rest of the
            // words rejoined, empty meaning "clear".
            let cmd = argv.get(2..).map_or_else(String::new, |w| w.join(" "));
            update_cmd(env, ctx, name, (!cmd.is_empty()).then_some(cmd), io)
        }
        "include" | "exclude" => {
            let Some(name) = argv.get(1) else {
                let _ = writeln!(io.err, "Usage: qmd collection {sub} <name>");
                let _ = writeln!(
                    io.err,
                    "  {} collection in default queries",
                    if sub == "include" {
                        "Include"
                    } else {
                        "Exclude"
                    }
                );
                return Err(Failure::Reported(1));
            };
            set_include(env, ctx, name, sub == "include", io)
        }
        "show" | "info" => {
            let Some(name) = argv.get(1) else {
                let _ = writeln!(io.err, "Usage: qmd collection show <name>");
                return Err(Failure::Reported(1));
            };
            show(env, ctx, name, io)
        }
        "help" => {
            help(io);
            Err(Failure::Reported(0))
        }
        other => {
            let _ = writeln!(io.err, "Unknown subcommand: {other}");
            let _ = writeln!(io.err, "Run 'qmd collection help' for usage");
            let _ = writeln!(io.err, "{DOCTOR_HINT}");
            Err(Failure::Reported(1))
        }
    }
}

/// The `collection help` block — also the `case undefined` branch,
/// upstream qmd.ts:4652-4674 (`process.exit(0)`).
fn help(io: &mut Io<'_>) {
    for line in [
        "Usage: qmd collection <command> [options]",
        "",
        "Commands:",
        "  list                      List all collections",
        "  add <path> [--name NAME] [--mask|--glob GLOB]  Add a collection",
        "  remove <name>             Remove a collection",
        "  rename <old> <new>        Rename a collection",
        "  show <name>               Show collection details",
        "  update-cmd <name> [cmd]   Set pre-update command (e.g., 'git pull')",
        "  include <name>            Include in default queries",
        "  exclude <name>            Exclude from default queries",
        "",
        "Examples:",
        "  qmd collection add ~/notes --name notes",
        "  qmd collection add ~/notes --name notes --mask 'a.md,journals/*.md'",
        "  qmd collection update-cmd brain 'git pull'",
        "  qmd collection exclude archive",
    ] {
        let _ = writeln!(io.out, "{line}");
    }
}

/// Upstream `formatTimeAgo` (qmd.ts:417-427): `Math.floor` on
/// millisecond deltas. `None` mirrors `new Date()` ≈ now → `0s ago`;
/// a timestamp `new Date` cannot parse upstream yields `NaN` diffs —
/// every comparison false → `NaNd ago`; a future timestamp is not
/// clamped and prints `-Ns ago` (`div_euclid` matches `Math.floor`
/// under our positive divisors).
pub(crate) fn format_time_ago(iso: Option<&str>, now: time::OffsetDateTime) -> String {
    let Some(iso) = iso else {
        return "0s ago".to_owned();
    };
    let Some(then) = parse_iso(iso) else {
        return "NaNd ago".to_owned();
    };
    let seconds = (now - then).whole_milliseconds().div_euclid(1000);
    if seconds < 60 {
        return format!("{seconds}s ago");
    }
    let minutes = seconds.div_euclid(60);
    if minutes < 60 {
        return format!("{minutes}m ago");
    }
    let hours = minutes.div_euclid(60);
    if hours < 24 {
        return format!("{hours}h ago");
    }
    format!("{}d ago", hours.div_euclid(24))
}

/// Parse the `modified_at` ISO timestamp (`YYYY-MM-DDTHH:MM:SS.sssZ`)
/// the indexer stores — `new Date(s)` upstream.
fn parse_iso(s: &str) -> Option<time::OffsetDateTime> {
    time::OffsetDateTime::parse(s, &time::format_description::well_known::Iso8601::DEFAULT).ok()
}

/// `qmd collection list` — upstream `collectionList`
/// (qmd.ts:1787-1821). Rows come from `store_collections` joined to
/// document stats; `Ignore:`/`[excluded]` are read from the YAML
/// collection, like `getCollectionFromYaml`.
fn list(env: &Environment, ctx: &CliContext, io: &mut Io<'_>, c: Palette) -> Result<(), Failure> {
    let qmd = ctx.open_store(env)?;
    let collections = qmd.collections()?;
    if collections.is_empty() {
        let _ = writeln!(
            io.out,
            "No collections found. Run 'qmd collection add .' to create one."
        );
        return Ok(());
    }
    let _ = writeln!(
        io.out,
        "{}Collections ({}):{}\n",
        c.bold,
        collections.len(),
        c.reset
    );
    let now = time::OffsetDateTime::now_utc();
    for coll in &collections {
        let yaml = qmd.collection(&coll.name)?;
        // `yamlColl?.includeByDefault === false` — the YAML read, not
        // the store mirror (identical after sync, but the display key
        // is the YAML one upstream).
        let excluded = yaml
            .as_ref()
            .is_some_and(|y| y.include_by_default == Some(false));
        let tag = if excluded {
            format!(" {}[excluded]{}", c.yellow, c.reset)
        } else {
            String::new()
        };
        let _ = writeln!(
            io.out,
            "{}{}{} {}(qmd://{}/){}{tag}",
            c.cyan, coll.name, c.reset, c.dim, coll.name, c.reset
        );
        let _ = writeln!(io.out, "  {}Pattern:{}  {}", c.dim, c.reset, coll.pattern);
        if let Some(ignore) = yaml.as_ref().and_then(|y| y.ignore.as_ref())
            && !ignore.is_empty()
        {
            let _ = writeln!(
                io.out,
                "  {}Ignore:{}   {}",
                c.dim,
                c.reset,
                ignore.join(", ")
            );
        }
        let _ = writeln!(
            io.out,
            "  {}Files:{}    {}",
            c.dim, c.reset, coll.active_count
        );
        // `coll.last_modified ? new Date(coll.last_modified) : new Date()`
        let ago = format_time_ago(coll.last_modified.as_deref(), now);
        let _ = writeln!(io.out, "  {}Updated:{}  {}", c.dim, c.reset, ago);
        let _ = writeln!(io.out);
    }
    Ok(())
}

/// `qmd collection add <path>` — dispatch validation (qmd.ts:4521-4549)
/// plus `collectionAdd` (qmd.ts:1828-1868): name from `--name` or the
/// basename of the resolved path, duplicate checks, YAML write, trust
/// re-record, then `indexFiles`.
fn add(
    env: &Environment,
    ctx: &CliContext,
    argv: &[String],
    args: &CollectionArgs,
    io: &mut Io<'_>,
    c: Palette,
) -> Result<(), Failure> {
    let Some(pwd) = argv.get(1) else {
        let _ = writeln!(
            io.err,
            "Usage: qmd collection add <path> [--name NAME] [--mask GLOB]"
        );
        let _ = writeln!(io.err, "  Pass '.' to index the current directory.");
        let _ = writeln!(
            io.err,
            "  --mask / --glob: glob (default **/*.md), brace group, or comma-separated list"
        );
        return Err(Failure::Reported(1));
    };
    // `pwd === '.' ? getPwd() : getRealPath(resolve(pwd))`
    let lexical = qmd::paths::resolve_lexical(ctx.cwd(), pwd);
    let resolved = if pwd == "." {
        qmd::paths::pwd(env)
    } else {
        lexical.canonicalize().unwrap_or(lexical)
    };
    // `collectionGlobFromCli` (qmd.ts:1822-1827): --mask wins, then
    // --glob, each only when non-empty.
    let glob = [&args.mask, &args.glob]
        .into_iter()
        .flatten()
        .find(|s| !s.is_empty())
        .map_or(DEFAULT_PATTERN, String::as_str)
        .to_owned();
    if !resolved.exists() {
        let _ = writeln!(
            io.err,
            "{}Collection path does not exist.{}",
            c.yellow, c.reset
        );
        let _ = writeln!(io.err, "  Received: {pwd}");
        let _ = writeln!(io.err, "  Resolved: {}", resolved.display());
        let _ = writeln!(
            io.err,
            "Provide an existing directory and run 'qmd collection add <path>' again."
        );
        return Err(Failure::Reported(1));
    }
    if !resolved.is_dir() {
        let _ = writeln!(
            io.err,
            "{}Collection path is not a directory.{}",
            c.yellow, c.reset
        );
        let _ = writeln!(io.err, "  Received: {pwd}");
        let _ = writeln!(io.err, "  Resolved: {}", resolved.display());
        let _ = writeln!(
            io.err,
            "Choose a directory and run 'qmd collection add <path>' again."
        );
        return Err(Failure::Reported(1));
    }
    let spec = qmd::CollectionSpec {
        path: resolved.to_string_lossy().into_owned(),
        pattern: Some(glob.clone()),
        ..qmd::CollectionSpec::default()
    };
    let name = collection_name(&spec.path, args.name.as_deref());

    // `collectionAdd`'s duplicate checks read the YAML config directly
    // (getCollectionFromYaml/yamlListCollections, qmd.ts:1837-1854) —
    // they run before `resyncConfig`, so a rejected add never creates
    // the database.
    let cfg = qmd::config::load(&ctx.config_path(env))?;
    if cfg.collections.contains_key(&name) {
        let _ = writeln!(
            io.err,
            "{}Collection '{name}' already exists.{}",
            c.yellow, c.reset
        );
        let _ = writeln!(io.err, "Use a different name with --name <name>");
        return Err(Failure::Reported(1));
    }
    // `c.pattern === globPattern` is verbatim — a YAML entry without a
    // `pattern:` key does not collide with a CLI-provided glob.
    let dup = cfg
        .collections
        .iter()
        .find(|(_, coll)| coll.path == spec.path && coll.pattern.as_deref() == Some(glob.as_str()));
    if let Some((existing, _)) = dup {
        let _ = writeln!(
            io.err,
            "{}A collection already exists for this path and pattern:{}",
            c.yellow, c.reset
        );
        let _ = writeln!(io.err, "  Name: {existing} (qmd://{existing}/)");
        let _ = writeln!(io.err, "  Pattern: {glob}");
        let _ = writeln!(
            io.err,
            "\nUse 'qmd update' to re-index it, or remove it first with \
             'qmd collection remove {existing}'"
        );
        return Err(Failure::Reported(1));
    }

    let qmd = ctx.open_store(env)?;
    match qmd.add_collection(&name, &spec) {
        Ok(()) => {}
        // `getCollectionFromYaml(collName)` check (qmd.ts:1837-1841).
        Err(Error::CollectionExists { name }) => {
            let _ = writeln!(
                io.err,
                "{}Collection '{name}' already exists.{}",
                c.yellow, c.reset
            );
            let _ = writeln!(io.err, "Use a different name with --name <name>");
            return Err(Failure::Reported(1));
        }
        // `yamlListCollections().find(c.path === pwd && c.pattern === glob)`
        // (qmd.ts:1844-1854).
        Err(Error::DuplicateCollectionSource { existing }) => {
            let _ = writeln!(
                io.err,
                "{}A collection already exists for this path and pattern:{}",
                c.yellow, c.reset
            );
            let _ = writeln!(io.err, "  Name: {existing} (qmd://{existing}/)");
            let _ = writeln!(io.err, "  Pattern: {glob}");
            let _ = writeln!(
                io.err,
                "\nUse 'qmd update' to re-index it, or remove it first with \
                 'qmd collection remove {existing}'"
            );
            return Err(Failure::Reported(1));
        }
        Err(err) => return Err(err.into()),
    }
    // The user just typed this path — no separate approval (#889).
    gate::trust_current_config(env, ctx)?;

    let _ = writeln!(io.out, "Creating collection '{name}'...");
    index_and_report(&qmd, &resolved, &glob, &name, io)?;
    let _ = writeln!(
        io.out,
        "{}✓{} Collection '{name}' created successfully",
        c.green, c.reset
    );
    Ok(())
}

/// `collectionAdd`'s name fallback (qmd.ts:1830-1834): a `--name ""`
/// is falsy like `undefined`; the basename is the resolved path's last
/// `/`-separated non-empty segment, `root` when none remains (`/`).
fn collection_name(resolved: &str, name: Option<&str>) -> String {
    if let Some(name) = name.filter(|s| !s.is_empty()) {
        return name.to_owned();
    }
    resolved
        .split('/')
        .rfind(|s| !s.is_empty())
        .unwrap_or("root")
        .to_owned()
}

/// `indexFiles` (qmd.ts:1922-2068) as used by `collection add`:
/// `clearCache`, the `Collection:` line, `progress.indeterminate`, the
/// reindex under the shared `onProgress` renderer, then the
/// indexed/removed/skipped/metadata report and the embed hint
/// (`suppressEmbedNotice` is false for `collection add`).
fn index_and_report(
    qmd: &Qmd,
    pwd: &Path,
    glob: &str,
    name: &str,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    // `clearCache(db)` — llm_cache cleared on every indexFiles; failure
    // is immaterial (same `try { } catch { }` shape as getStore).
    let _ = qmd.maintenance().clear_llm_cache();
    let _ = writeln!(io.out, "Collection: {} ({glob})", pwd.display());
    progress::indeterminate(io);
    // Upstream anchors `startTime` after the glob scan (qmd.ts:1967);
    // our scan lives inside `update_collection`, so the anchor includes
    // it — the ETA merely reflects it in the first samples.
    let start = std::time::Instant::now();
    let report = qmd.update_collection(name, &mut |p| {
        progress::indexing_tick(io, p.current, p.total, start);
    })?;
    // Upstream clears inside `hasNoFiles` (qmd.ts:1955-1958) and again
    // before the report (qmd.ts:2066) — two clear writes when empty.
    if report.scanned == 0 {
        progress::clear(io);
        let _ = writeln!(io.out, "No files found matching pattern.");
    }
    progress::clear(io);
    let _ = writeln!(
        io.out,
        "\nIndexed: {} new, {} updated, {} unchanged, {} removed",
        report.indexed, report.updated, report.unchanged, report.removed
    );
    crate::maintenance::report_skipped_reads(&report.skipped_files, io);
    crate::maintenance::report_metadata_errors(report.metadata_errors, io);
    if report.orphaned_cleaned > 0 {
        let _ = writeln!(
            io.out,
            "Cleaned up {} orphaned content hash(es)",
            report.orphaned_cleaned
        );
    }
    let tail = qmd.update_tail_counts()?;
    if tail.needs_embedding > 0 {
        let _ = writeln!(
            io.out,
            "\nRun 'qmd embed' to update embeddings ({} unique hashes need vectors)",
            tail.needs_embedding
        );
    }
    Ok(())
}

/// `qmd collection remove|rm <name>` — upstream `collectionRemove`
/// (qmd.ts:1870-1889): the `getCollectionFromYaml` existence check runs
/// before `getDb`, so a missing collection never creates the database;
/// then the hard-delete and its `sqlite3_total_changes` report (see
/// [`qmd::Qmd::remove_collection`] for why the count is not a row count).
fn remove(
    env: &Environment,
    ctx: &CliContext,
    name: &str,
    io: &mut Io<'_>,
    c: Palette,
) -> Result<(), Failure> {
    let cfg = qmd::config::load(&ctx.config_path(env))?;
    if !cfg.collections.contains_key(name) {
        let _ = writeln!(
            io.err,
            "{}Collection not found: {name}{}",
            c.yellow, c.reset
        );
        let _ = writeln!(
            io.err,
            "Run 'qmd collection list' to see available collections."
        );
        return Err(Failure::Reported(1));
    }
    let qmd = ctx.open_store(env)?;
    let Some(removal) = qmd.remove_collection(name)? else {
        return Err(Failure::Reported(1));
    };
    let _ = writeln!(
        io.out,
        "{}✓{} Removed collection '{name}'",
        c.green, c.reset
    );
    let _ = writeln!(io.out, "  Deleted {} documents", removal.deleted_docs);
    if removal.cleaned_hashes > 0 {
        let _ = writeln!(
            io.out,
            "  Cleaned up {} orphaned content hashes",
            removal.cleaned_hashes
        );
    }
    Ok(())
}

/// `qmd collection rename|mv <old> <new>` — upstream `collectionRename`
/// (qmd.ts:1892-1920): `getCollectionFromYaml` existence checks on both
/// ends run before `getDb`, then the document `collection` column swap
/// plus the virtual-path note.
fn rename(
    env: &Environment,
    ctx: &CliContext,
    old: &str,
    new: &str,
    io: &mut Io<'_>,
    c: Palette,
) -> Result<(), Failure> {
    let cfg = qmd::config::load(&ctx.config_path(env))?;
    if !cfg.collections.contains_key(old) {
        let _ = writeln!(io.err, "{}Collection not found: {old}{}", c.yellow, c.reset);
        let _ = writeln!(
            io.err,
            "Run 'qmd collection list' to see available collections."
        );
        return Err(Failure::Reported(1));
    }
    if cfg.collections.contains_key(new) {
        let _ = writeln!(
            io.err,
            "{}Collection name already exists: {new}{}",
            c.yellow, c.reset
        );
        let _ = writeln!(
            io.err,
            "Choose a different name or remove the existing collection first."
        );
        return Err(Failure::Reported(1));
    }
    let qmd = ctx.open_store(env)?;
    match qmd.rename_collection(old, new) {
        Ok(true) => {}
        // Both YAML checks ran above; only a race reaches these.
        Ok(false) | Err(Error::CollectionNotFound { .. }) => {
            let _ = writeln!(io.err, "{}Collection not found: {old}{}", c.yellow, c.reset);
            let _ = writeln!(
                io.err,
                "Run 'qmd collection list' to see available collections."
            );
            return Err(Failure::Reported(1));
        }
        Err(Error::CollectionExists { .. }) => {
            let _ = writeln!(
                io.err,
                "{}Collection name already exists: {new}{}",
                c.yellow, c.reset
            );
            let _ = writeln!(
                io.err,
                "Choose a different name or remove the existing collection first."
            );
            return Err(Failure::Reported(1));
        }
        Err(err) => return Err(err.into()),
    }
    let _ = writeln!(
        io.out,
        "{}✓{} Renamed collection '{old}' to '{new}'",
        c.green, c.reset
    );
    let _ = writeln!(
        io.out,
        "  Virtual paths updated: {}qmd://{old}/{} → {}qmd://{new}/{}",
        c.cyan, c.reset, c.cyan, c.reset
    );
    Ok(())
}

/// `qmd collection update-cmd|set-update <name> [command]` — upstream
/// dispatch (qmd.ts:4575-4601): the `getCollection` check and
/// `updateCollectionSettings` write are pure YAML
/// (collections.ts:244-311), so this never opens the database.
/// `trustCurrentConfig` runs after the save because the user typed the
/// hook.
fn update_cmd(
    env: &Environment,
    ctx: &CliContext,
    name: &str,
    cmd: Option<String>,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    let config_path = ctx.config_path(env);
    let mut cfg = qmd::config::load(&config_path)?;
    let Some(coll) = cfg.collections.get_mut(name) else {
        let _ = writeln!(io.err, "Collection not found: {name}");
        return Err(Failure::Reported(1));
    };
    let message = cmd.as_ref().map_or_else(
        || format!("✓ Cleared update command for '{name}'"),
        |cmd| format!("✓ Set update command for '{name}': {cmd}"),
    );
    // `update === null` deletes the key upstream; `None` is skipped by
    // serde so clearing removes it rather than writing an empty value.
    coll.update = cmd;
    qmd::config::save(&config_path, &cfg)?;
    // Re-record so the digest covers the new hook set (#886).
    gate::trust_current_config(env, ctx)?;
    let _ = writeln!(io.out, "{message}");
    Ok(())
}

/// `qmd collection include|exclude <name>` — upstream dispatch
/// (qmd.ts:4603-4624): `true` deletes `includeByDefault`, `false`
/// persists it.
fn set_include(
    env: &Environment,
    ctx: &CliContext,
    name: &str,
    include: bool,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    // Pure YAML (collections.ts:283-311): `include` deletes
    // `includeByDefault`, `exclude` persists `false` — no store open.
    let config_path = ctx.config_path(env);
    let mut cfg = qmd::config::load(&config_path)?;
    let Some(coll) = cfg.collections.get_mut(name) else {
        let _ = writeln!(io.err, "Collection not found: {name}");
        return Err(Failure::Reported(1));
    };
    coll.include_by_default = (!include).then_some(false);
    qmd::config::save(&config_path, &cfg)?;
    let verb = if include {
        "included in"
    } else {
        "excluded from"
    };
    let _ = writeln!(io.out, "✓ Collection '{name}' {verb} default queries");
    Ok(())
}

/// `qmd collection show|info <name>` — upstream dispatch
/// (qmd.ts:4626-4650): `getCollection` is a pure `loadConfig` read
/// (collections.ts:244-253), so this never opens the database.
/// `Pattern:` prints the stored `pattern` verbatim; a collection
/// without one prints `undefined`, like JS.
fn show(env: &Environment, ctx: &CliContext, name: &str, io: &mut Io<'_>) -> Result<(), Failure> {
    let cfg = qmd::config::load(&ctx.config_path(env))?;
    let Some(col) = cfg.collections.get(name) else {
        let _ = writeln!(io.err, "Collection not found: {name}");
        return Err(Failure::Reported(1));
    };
    let _ = writeln!(io.out, "Collection: {name}");
    let _ = writeln!(io.out, "  Path:     {}", col.path);
    let _ = writeln!(
        io.out,
        "  Pattern:  {}",
        col.pattern.as_deref().unwrap_or("undefined")
    );
    let _ = writeln!(
        io.out,
        "  Include:  {}",
        if col.include_by_default == Some(false) {
            "no"
        } else {
            "yes (default)"
        }
    );
    // `if (col.update)` — JS truthiness skips an empty string.
    if let Some(update) = col.update.as_deref().filter(|u| !u.is_empty()) {
        let _ = writeln!(io.out, "  Update:   {update}");
    }
    if let Some(contexts) = &col.context {
        let _ = writeln!(io.out, "  Contexts: {}", contexts.len());
    }
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::missing_assert_message,
    reason = "idiomatic test-code patterns"
)]
mod tests {
    use super::{collection_name, format_time_ago};

    // `collName = parts[parts.length - 1] || 'root'` on the resolved
    // path's `/`-split (qmd.ts:1830-1834).
    #[test]
    fn collection_name_falls_back_to_basename() {
        assert_eq!(collection_name("/abs/dir", Some("n")), "n");
        // `""` is falsy upstream (`if (!collName)`) → basename.
        assert_eq!(collection_name("/abs/dir", Some("")), "dir");
        assert_eq!(collection_name("/a/b/", None), "b");
        // Resolved `/` has no segments left after the filter.
        assert_eq!(collection_name("/", None), "root");
    }

    // Upstream `formatTimeAgo` edges (qmd.ts:417-427) driven by
    // `new Date()`/`Math.floor` semantics.
    #[test]
    fn format_time_ago_matches_js_edges() {
        let epoch = time::OffsetDateTime::from_unix_timestamp(0).unwrap();
        // `coll.last_modified` absent → `new Date()` ≈ now.
        assert_eq!(format_time_ago(None, epoch), "0s ago");
        // `Invalid Date` → `NaN` diffs; every comparison is false and
        // upstream lands in the `d` branch.
        assert_eq!(format_time_ago(Some("not a date"), epoch), "NaNd ago");
        // `Math.floor(-5000/1000)` → -5 — a future timestamp is not
        // clamped upstream.
        assert_eq!(
            format_time_ago(Some("1970-01-01T00:00:05.000Z"), epoch),
            "-5s ago"
        );
        assert_eq!(
            format_time_ago(Some("1969-12-31T23:59:30.000Z"), epoch),
            "30s ago"
        );
        assert_eq!(
            format_time_ago(Some("1969-12-31T23:58:30.000Z"), epoch),
            "1m ago"
        );
        assert_eq!(
            format_time_ago(Some("1969-12-31T22:00:00.000Z"), epoch),
            "2h ago"
        );
        assert_eq!(
            format_time_ago(Some("1969-12-30T00:00:00.000Z"), epoch),
            "2d ago"
        );
    }
}
