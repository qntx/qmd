//! `qmd context` subcommands — upstream `case "context"`
//! (cli/qmd.ts:4404-4478) plus `contextAdd` (qmd.ts:1058-1122),
//! `contextList` (qmd.ts:1124-1150) and `contextRemove`
//! (qmd.ts:1152-1226).
//!
//! Context lives in the YAML config (`context:` maps per collection,
//! `global_context` at the root) and is mirrored into
//! `store_collections.context`/`store_config.global_context` by the
//! config sync, like upstream's `yamlAddContext` + `resyncConfig`.

use std::path::PathBuf;

use qmd::Environment;
use qmd::vpath::{is_virtual_path, parse_virtual_path};

use crate::Failure;
use crate::args::ContextArgs;
use crate::context::CliContext;
use crate::output::{Io, Palette};

/// Dispatch `cli.args[0]` — upstream `case "context"` (qmd.ts:4404).
pub(crate) fn run(
    env: &Environment,
    ctx: &CliContext,
    args: &ContextArgs,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    let c = Palette::new(Palette::enabled(env, io));
    let argv = &args.args;
    let Some(sub) = argv.first() else {
        // Bare `qmd context` — the 7-line usage block (qmd.ts:4407-4413).
        for line in [
            "Usage: qmd context <add|list|rm>",
            "",
            "Commands:",
            "  qmd context add [path] \"text\"  - Add context (defaults to current dir)",
            "  qmd context add / \"text\"       - Add global context to all collections",
            "  qmd context list                - List all contexts",
            "  qmd context rm <path>           - Remove context",
        ] {
            let _ = writeln!(io.err, "{line}");
        }
        return Err(Failure::Reported(1));
    };
    match sub.as_str() {
        "add" => {
            if argv.len() < 2 {
                // qmd.ts:4420-4431.
                for line in [
                    "Usage: qmd context add [path] \"text\"",
                    "",
                    "Examples:",
                    "  qmd context add \"Context for current directory\"",
                    "  qmd context add . \"Context for current directory\"",
                    "  qmd context add /subfolder \"Context for subfolder\"",
                    "  qmd context add / \"Global context for all collections\"",
                    "",
                    "  Using virtual paths:",
                    "  qmd context add qmd://journals/ \"Context for entire journals collection\"",
                    "  qmd context add qmd://journals/2024 \"Context for 2024 journals\"",
                ] {
                    let _ = writeln!(io.err, "{line}");
                }
                return Err(Failure::Reported(1));
            }
            // `secondArg ? (pathArg=args[1], text=args[2..].join(' '))
            // : (pathArg=undefined, text=args[1])` (qmd.ts:4435-4445).
            let (path_arg, text) = if argv.len() >= 3 {
                (
                    argv.get(1).map(String::as_str),
                    argv.get(2..).map_or_else(String::new, |w| w.join(" ")),
                )
            } else {
                (None, argv.get(1).cloned().unwrap_or_default())
            };
            add(env, ctx, path_arg, &text, io, c)
        }
        "list" => list(env, ctx, io, c),
        "rm" | "remove" => {
            let Some(path) = argv.get(1) else {
                for line in [
                    "Usage: qmd context rm <path>",
                    "Examples:",
                    "  qmd context rm /",
                    "  qmd context rm qmd://journals/2024",
                ] {
                    let _ = writeln!(io.err, "{line}");
                }
                return Err(Failure::Reported(1));
            };
            remove(env, ctx, path, io, c)
        }
        other => {
            let _ = writeln!(io.err, "Unknown subcommand: {other}");
            let _ = writeln!(io.err, "Available: add, list, rm");
            Err(Failure::Reported(1))
        }
    }
}

/// The `context add`/`context rm` filesystem-path resolution —
/// upstream's shared chain (qmd.ts:1071-1077 and 1195-1201): `.`/`./`
/// → `getPwd()`, `~/…` → `homedir() + rest`, non-absolute non-`qmd://`
/// → `resolve(getPwd(), arg)`. (`context add` defaults `pathArg` to
/// `'.'` upstream — the caller passes `"."` for that case.)
fn resolve_fs_path(env: &Environment, arg: &str) -> String {
    let mut fs_path = arg.to_owned();
    if fs_path == "." || fs_path == "./" {
        fs_path = qmd::paths::pwd(env).to_string_lossy().into_owned();
    } else if let Some(rest) = fs_path.strip_prefix("~/") {
        fs_path = format!("{}/{rest}", qmd::paths::home_dir(env).display());
    } else if !fs_path.starts_with('/') && !fs_path.starts_with("qmd://") {
        fs_path = qmd::paths::resolve_lexical(&qmd::paths::pwd(env), &fs_path)
            .to_string_lossy()
            .into_owned();
    }
    fs_path
}

/// `qmd context add [path] "text"` — upstream `contextAdd`
/// (qmd.ts:1058-1122): `/` is the global context, `qmd://` URIs take
/// the virtual branch, everything else resolves to a filesystem path
/// and lands on the deepest collection covering it.
fn add(
    env: &Environment,
    ctx: &CliContext,
    path_arg: Option<&str>,
    text: &str,
    io: &mut Io<'_>,
    c: Palette,
) -> Result<(), Failure> {
    let qmd = ctx.open_store(env)?;
    if path_arg == Some("/") {
        qmd.set_global_context(Some(text))?;
        let _ = writeln!(io.out, "{}✓{} Set global context", c.green, c.reset);
        let _ = writeln!(io.out, "{}Context: {text}{}", c.dim, c.reset);
        return Ok(());
    }
    let fs_path = resolve_fs_path(env, path_arg.unwrap_or("."));
    if is_virtual_path(&fs_path) {
        let Some(parsed) = parse_virtual_path(&fs_path) else {
            let _ = writeln!(
                io.err,
                "{}Invalid virtual path: {fs_path}{}",
                c.yellow, c.reset
            );
            return Err(Failure::Reported(1));
        };
        if qmd.collection(&parsed.collection_name)?.is_none() {
            let _ = writeln!(
                io.err,
                "{}Collection not found: {}{}",
                c.yellow, parsed.collection_name, c.reset
            );
            return Err(Failure::Reported(1));
        }
        qmd.add_context(&parsed.collection_name, &parsed.path, text)?;
        let display = if parsed.path.is_empty() {
            format!("qmd://{}/ (collection root)", parsed.collection_name)
        } else {
            format!("qmd://{}/{}", parsed.collection_name, parsed.path)
        };
        let _ = writeln!(
            io.out,
            "{}✓{} Added context for: {display}",
            c.green, c.reset
        );
        let _ = writeln!(io.out, "{}Context: {text}{}", c.dim, c.reset);
        return Ok(());
    }
    let Some(detected) = qmd.detect_collection(PathBuf::from(&fs_path).as_path())? else {
        let _ = writeln!(
            io.err,
            "{}Path is not in any indexed collection: {fs_path}{}",
            c.yellow, c.reset
        );
        let _ = writeln!(
            io.err,
            "{}Run 'qmd status' to see indexed collections{}",
            c.dim, c.reset
        );
        return Err(Failure::Reported(1));
    };
    qmd.add_context(&detected.name, &detected.relative_path, text)?;
    let display = if detected.relative_path.is_empty() {
        format!("qmd://{}/", detected.name)
    } else {
        format!("qmd://{}/{}", detected.name, detected.relative_path)
    };
    let _ = writeln!(
        io.out,
        "{}✓{} Added context for: {display}",
        c.green, c.reset
    );
    let _ = writeln!(io.out, "{}Context: {text}{}", c.dim, c.reset);
    Ok(())
}

/// `qmd context list` — upstream `contextList` (qmd.ts:1124-1150):
/// `listAllContexts()` — global first, then every collection's
/// `context:` map in YAML order — grouped under a repeating
/// collection header.
fn list(env: &Environment, ctx: &CliContext, io: &mut Io<'_>, c: Palette) -> Result<(), Failure> {
    let qmd = ctx.open_store(env)?;
    // `listAllContexts` reads `loadConfig()` — the YAML file, not the
    // store mirror (same content; the file is the ordering source).
    let config = qmd.config()?;
    let mut entries: Vec<(&str, &str, &str)> = Vec::new();
    if let Some(global) = config.global_context.as_deref() {
        entries.push(("*", "/", global));
    }
    for (name, coll) in &config.collections {
        if let Some(map) = &coll.context {
            for (path, text) in map {
                entries.push((name.as_str(), path.as_str(), text.as_str()));
            }
        }
    }
    if entries.is_empty() {
        let _ = writeln!(
            io.out,
            "{}No contexts configured. Use 'qmd context add' to add one.{}",
            c.dim, c.reset
        );
        return Ok(());
    }
    let _ = writeln!(io.out, "\n{}Configured Contexts{}\n", c.bold, c.reset);
    let mut last = "";
    for (collection, path, text) in entries {
        if collection != last {
            let _ = writeln!(io.out, "{}{collection}{}", c.cyan, c.reset);
            last = collection;
        }
        // `ctx.path ? `  ${ctx.path}` : '  / (root)'`
        if path.is_empty() {
            let _ = writeln!(io.out, "  / (root)");
        } else {
            let _ = writeln!(io.out, "  {path}");
        }
        let _ = writeln!(io.out, "    {}{text}{}", c.dim, c.reset);
    }
    Ok(())
}

/// `qmd context rm <path>` — upstream `contextRemove`
/// (qmd.ts:1152-1226): `/` clears the global context and resyncs
/// through the store, `qmd://` URIs are pure `yamlRemoveContext`
/// (collections.ts:422-441 — no `getDb`, no `resyncConfig`), and
/// filesystem paths open the store for `detectCollectionFromPath`
/// before the same YAML removal.
fn remove(
    env: &Environment,
    ctx: &CliContext,
    path_arg: &str,
    io: &mut Io<'_>,
    c: Palette,
) -> Result<(), Failure> {
    if path_arg == "/" {
        // Upstream calls `getStore()` here for the `resyncConfig` that
        // pushes the cleared global context into `store_config`.
        let qmd = ctx.open_store(env)?;
        qmd.set_global_context(None)?;
        let _ = writeln!(io.out, "{}✓{} Removed global context", c.green, c.reset);
        return Ok(());
    }
    if is_virtual_path(path_arg) {
        let Some(parsed) = parse_virtual_path(path_arg) else {
            let _ = writeln!(
                io.err,
                "{}Invalid virtual path: {path_arg}{}",
                c.yellow, c.reset
            );
            return Err(Failure::Reported(1));
        };
        // `getCollectionFromYaml(parsed.collectionName)` (qmd.ts:1172).
        if !qmd::config::load(&ctx.config_path(env))?
            .collections
            .contains_key(&parsed.collection_name)
        {
            let _ = writeln!(
                io.err,
                "{}Collection not found: {}{}",
                c.yellow, parsed.collection_name, c.reset
            );
            return Err(Failure::Reported(1));
        }
        if !yaml_remove_context(env, ctx, &parsed.collection_name, &parsed.path)? {
            let _ = writeln!(
                io.err,
                "{}No context found for: {path_arg}{}",
                c.yellow, c.reset
            );
            return Err(Failure::Reported(1));
        }
        let _ = writeln!(
            io.out,
            "{}✓{} Removed context for: {path_arg}",
            c.green, c.reset
        );
        return Ok(());
    }
    let fs_path = resolve_fs_path(env, path_arg);
    // `getDb()` + `detectCollectionFromPath` + `closeDb()`
    // (qmd.ts:1199-1201) — the upstream detection reads
    // `yamlListCollections` anyway, so the open handle is load-bearing
    // only for parity with `getDb()` creating the index.
    let detected = {
        let qmd = ctx.open_store(env)?;
        qmd.detect_collection(PathBuf::from(&fs_path).as_path())?
    };
    let Some(detected) = detected else {
        let _ = writeln!(
            io.err,
            "{}Path is not in any indexed collection: {fs_path}{}",
            c.yellow, c.reset
        );
        return Err(Failure::Reported(1));
    };
    let display = format!("qmd://{}/{}", detected.name, detected.relative_path);
    if !yaml_remove_context(env, ctx, &detected.name, &detected.relative_path)? {
        let _ = writeln!(
            io.err,
            "{}No context found for: {display}{}",
            c.yellow, c.reset
        );
        return Err(Failure::Reported(1));
    }
    let _ = writeln!(
        io.out,
        "{}✓{} Removed context for: {display}",
        c.green, c.reset
    );
    Ok(())
}

/// Upstream `removeContext` (collections.ts:422-441) — load, delete the
/// `context:` key, `delete collection.context` when the map empties,
/// save. `Ok(false)` when the collection or prefix is missing, like
/// upstream's `collection?.context?.[pathPrefix]` guard.
fn yaml_remove_context(
    env: &Environment,
    ctx: &CliContext,
    collection: &str,
    prefix: &str,
) -> Result<bool, qmd::Error> {
    let config_path = ctx.config_path(env);
    let mut cfg = qmd::config::load(&config_path)?;
    let Some(coll) = cfg.collections.get_mut(collection) else {
        return Ok(false);
    };
    let Some(map) = coll.context.as_mut() else {
        return Ok(false);
    };
    // `collection?.context?.[pathPrefix]` is JS-truthy — an
    // empty-string context reports missing and stays put.
    if map.get(prefix).is_none_or(String::is_empty) {
        return Ok(false);
    }
    map.shift_remove(prefix);
    if map.is_empty() {
        coll.context = None;
    }
    qmd::config::save(&config_path, &cfg)?;
    Ok(true)
}
