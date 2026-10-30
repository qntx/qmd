//! `qmd skills` / `qmd skill` / `--skill` — bundled-skill discovery,
//! printing and installation.
//!
//! Ports upstream `getSkillInstallDir`/`getClaudeSkillLinkPath`
//! (cli/qmd.ts:3226-3236), `pathExists`/`removePath` (3238-3254),
//! `getSkillSearchDirs`/`parseSkillFrontmatter`/`discoverSkills`/
//! `findSkill`/`readSkillContent`/`collectSkillFiles` (3283-3380),
//! `showSkill` (3382-3391), `copyDirectoryContents` (3393-3405),
//! `installedSkillStubContent`/`writeSkillInstall` (3407-3451),
//! `runSkillsCommand`/`showSkillsHelp` (3457-3560),
//! `ensureClaudeSymlink`/`shouldCreateClaudeSymlink`/`installSkill`
//! (3562-3633) and the `skills`/`skill` dispatch plus the
//! `skill --help` precedence quirk (4384-4396, 4920-4977).
//!
//! Where upstream discovers skills from the package's `skills/`
//! directory (`findPackageRoot`, qmd.ts:3265-3293), this port embeds
//! the tree in the binary via `include_str!` and materializes it under
//! `<cache-dir>/skills/<crate-version>` only when a real path is
//! needed (`skills path`, `skill install`). `skills/qmd/` is a
//! byte-verbatim copy of upstream's so `skills get qmd`/`skill show`
//! output matches byte-for-byte — including its `npm install -g
//! @tobilu/qmd` line, a known upstream-inherited inaccuracy (this port
//! is not installed through npm). Upstream's `skills/release`
//! maintainer skill is deliberately not ported (P4-D7: a dev-tool, not
//! a runtime skill).
//!
//! Error channels differ by command like upstream: `skills ...` errors
//! become a `{success:false,error}` envelope on stdout under `--json`
//! or the bare message on stderr — exit 1, no doctor hint; `skill
//! show`/`skill install` errors propagate as [`Failure::Error`]
//! (message + doctor hint, upstream `exitWithError`), which also
//! covers the crash upstream's unguarded `--skill`/`skill show` call
//! would produce — a deliberate clean-up.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use qmd::js::is_js_ws;
use qmd::{Environment, paths, utf16_len};
use serde_json::{Map, Value};

use crate::args::{SkillArgs, SkillsArgs};
use crate::context::CliContext;
use crate::gate::confirm_on_tty;
use crate::output::Io;
use crate::{DOCTOR_HINT, Failure};

/// Upstream `skills/<name>/` — one bundled skill embedded in the
/// binary. `QMD_SKILLS_DIR` bypasses this table entirely, like
/// upstream's `findPackageRoot` early return (qmd.ts:3266-3268).
struct EmbeddedSkill {
    /// Directory name under the skills dir (`skills/qmd`).
    dir: &'static str,
    /// `(relative_path, content)` — `SKILL.md` plus payload files under
    /// `references/`/`templates/`/`scripts/`.
    files: &'static [(&'static str, &'static str)],
}

/// The embedded skill tree — `crates/qmd-cli/skills/` holds
/// byte-verbatim copies of upstream `tobi/qmd/skills/qmd/` only.
const EMBEDDED_SKILLS: &[EmbeddedSkill] = &[EmbeddedSkill {
    dir: "qmd",
    files: &[
        ("SKILL.md", include_str!("../skills/qmd/SKILL.md")),
        (
            "references/mcp-setup.md",
            include_str!("../skills/qmd/references/mcp-setup.md"),
        ),
    ],
}];

/// Upstream `SkillInfo` (qmd.ts:3256-3261): parsed frontmatter plus the
/// backing store content is read from.
struct SkillInfo {
    /// `name:` frontmatter value.
    name: String,
    /// `description:` frontmatter value (continuations folded).
    description: String,
    /// `hidden:` frontmatter value.
    hidden: bool,
    /// Where the skill's files live.
    source: SkillSource,
}

/// Backing store for a discovered skill.
enum SkillSource {
    /// Index into [`EMBEDDED_SKILLS`] — content comes from the binary;
    /// a real path is materialized under the cache dir on demand.
    Embedded(usize),
    /// `<search-dir>/<entry>` resolved at discovery time (upstream
    /// `resolve(dir, entry)` — absolute already).
    Dir(PathBuf),
}

/// One `references/`/`templates/`/`scripts/` payload file.
struct SkillFile {
    /// `{subdir}/{basename}` — upstream `relativePath`.
    relative_path: String,
    /// File body.
    content: String,
}

/// Parsed frontmatter — upstream `parseSkillFrontmatter`'s return
/// (qmd.ts:3295-3326).
struct SkillFrontmatter {
    /// `name:` value.
    name: String,
    /// `description:` value, continuation lines joined with `" "`.
    description: String,
    /// `hidden:` value (`true`/`yes`, case-insensitive).
    hidden: bool,
}

/// `Skill not found` text — shared by `showSkill` and
/// `writeSkillInstall` (qmd.ts:3385, 3446).
const SKILL_NOT_FOUND: &str = "QMD skill not found. Reinstall qmd or set QMD_SKILLS_DIR.";

/// `installedSkillStubContent` (qmd.ts:3407-3434) — byte-verbatim
/// generated output, including the `` !`qmd skill show` `` line and
/// the trailing newline.
const INSTALLED_SKILL_STUB: &str = r"---
name: qmd
description: Bootstrap QMD search instructions from the installed qmd CLI. Use when users ask to find notes, retrieve documents, inspect a wiki, or answer from indexed local markdown.
license: MIT
compatibility: Requires qmd CLI. Run `qmd skill show` for version-matched instructions.
allowed-tools: Bash(qmd:*), mcp__qmd__*
---

# QMD - Query Markdown Documents

This installed skill is intentionally a small bootstrap so it does not go stale
when the qmd package updates.

Load the full, version-matched QMD instructions from the CLI:

!`qmd skill show`

If your agent does not support bang-command expansion, run:

```bash
qmd skill show
```

Then follow those instructions. In short: search first, fetch full sources with
`qmd get` or `qmd multi-get`, and answer from retrieved text rather than snippets.
";

/// `qmd skills [list|get|path|help]` — upstream `case "skills"`
/// (qmd.ts:4920-4936). Any error is rendered here: the
/// `{success:false,error}` envelope on **stdout** under `--json`, else
/// the bare message on stderr; exit 1 and no doctor hint either way.
pub(crate) fn run_skills(
    env: &Environment,
    ctx: &CliContext,
    args: &SkillsArgs,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    match run_skills_command(env, ctx, args, io) {
        Ok(()) => Ok(()),
        Err(err) => {
            let message = err.to_string();
            if args.json {
                let mut payload = Map::new();
                payload.insert("success".to_owned(), Value::Bool(false));
                payload.insert("error".to_owned(), Value::from(message));
                output_skills_json(io, &Value::Object(payload));
            } else {
                let _ = writeln!(io.err, "{message}");
            }
            Err(Failure::Reported(1))
        }
    }
}

/// `qmd skill [show|install|help]` — upstream `case "skill"`
/// (qmd.ts:4938-4977). `help`/bare print usage and exit **0** —
/// upstream's `process.exit(0)` quirk versus other commands' 1.
pub(crate) fn run_skill(
    env: &Environment,
    ctx: &CliContext,
    args: &SkillArgs,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    match args.args.first().map(String::as_str) {
        Some("show") => show_skill(env, ctx, io),
        Some("install") => install_skill(env, ctx, args, io),
        Some("help") | None => {
            print_skill_usage(io);
            Ok(())
        }
        Some(other) => {
            let _ = writeln!(io.err, "Unknown subcommand: {other}");
            let _ = writeln!(io.err, "Run 'qmd skill help' for usage");
            let _ = writeln!(io.err, "{DOCTOR_HINT}");
            Err(Failure::Reported(1))
        }
    }
}

/// `showSkill` (qmd.ts:3382-3391) — `QMD Skill\n\n` on stdout, then the
/// `SKILL.md` body via `process.stdout.write` semantics (trailing
/// newline ensured, no extra line). Shared by `skill show` and
/// `--skill`.
///
/// # Errors
/// [`Failure::Error`] when the `qmd` skill is missing or unreadable —
/// upstream's unguarded call crashes; this port renders the message
/// plus the doctor hint (deliberate clean-up).
pub(crate) fn show_skill(
    env: &Environment,
    ctx: &CliContext,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    let skill = find_skill("qmd", env, ctx.cwd()).ok_or_else(|| skill_error(SKILL_NOT_FOUND))?;
    let _ = writeln!(io.out, "QMD Skill\n");
    let content = read_skill_content(&skill)?;
    write_body(io.out, &content);
    Ok(())
}

/// The `qmd skill` usage block (qmd.ts:4385-4395, 4957-4967) —
/// identical for bare `skill`, `skill help` and `skill --help` (the
/// latter is dispatched in `lib.rs` before general help).
pub(crate) fn print_skill_usage(io: &mut Io<'_>) {
    for line in [
        "Usage: qmd skill <show|install> [options]",
        "",
        "Commands:",
        "  show                 Print the QMD skill",
        "  install              Install QMD skill into ./.agents/skills/qmd",
        "",
        "Options:",
        "  --global             Install into ~/.agents/skills/qmd",
        "  --yes                Also create the .claude/skills/qmd symlink",
        "  -f, --force          Replace existing install or symlink",
    ] {
        let _ = writeln!(io.out, "{line}");
    }
}

/// `runSkillsCommand` (qmd.ts:3457-3546). The `cli.values.help` branch
/// of the `skills` case (qmd.ts:4922) is unreachable — the general
/// `--help` check runs first — so only `skills help` lands here.
///
/// # Errors
/// [`qmd::Error`] carrying the upstream message for `run_skills`'s
/// envelope: `Skill not found: <name>`, the get-usage error, or
/// `Unknown skills subcommand: <sub>`; [`qmd::Error::Io`] on materialize
/// or skill-read failures (upstream's swallowed `readdirSync` errors
/// stay swallowed inside discovery).
fn run_skills_command(
    env: &Environment,
    ctx: &CliContext,
    args: &SkillsArgs,
    io: &mut Io<'_>,
) -> Result<(), qmd::Error> {
    let argv = &args.args;
    match argv.first().map_or("list", String::as_str) {
        "list" => {
            skills_list(env, ctx, args.json, io);
            Ok(())
        }
        "get" => skills_get(env, ctx, args, io),
        "path" => skills_path(env, ctx, args, io),
        "help" => {
            show_skills_help(io);
            Ok(())
        }
        other => Err(skill_error(format!("Unknown skills subcommand: {other}"))),
    }
}

/// `case "list"` (qmd.ts:3462-3477): runtime skills, JSON envelope or
/// padded `name  description` rows.
fn skills_list(env: &Environment, ctx: &CliContext, json_mode: bool, io: &mut Io<'_>) {
    let skills = runtime_skills(env, ctx.cwd());
    if json_mode {
        let data: Vec<Value> = skills
            .iter()
            .map(|skill| {
                let mut row = Map::new();
                row.insert("name".to_owned(), Value::from(skill.name.clone()));
                row.insert(
                    "description".to_owned(),
                    Value::from(skill.description.clone()),
                );
                Value::Object(row)
            })
            .collect();
        output_skills_json(io, &success_envelope(Value::Array(data)));
        return;
    }
    if skills.is_empty() {
        let _ = writeln!(io.out, "No skills found");
        return;
    }
    // `padEnd(maxName)` — JS `length` is UTF-16 units.
    let max_name = skills
        .iter()
        .map(|skill| utf16_len(&skill.name))
        .max()
        .unwrap_or(0);
    for skill in &skills {
        let _ = writeln!(
            io.out,
            "  {}  {}",
            pad_end(&skill.name, max_name),
            skill.description
        );
    }
}

/// `case "get"` (qmd.ts:3479-3519). `fullOption ||
/// args.includes("--full")` keeps the raw positional scan so a
/// post-`--` `--full`/`--all` still counts.
///
/// # Errors
/// [`qmd::Error::Skill`] for `Skill not found: <name>` and the
/// no-name usage error; [`qmd::Error::Io`] on a skill-read failure.
fn skills_get(
    env: &Environment,
    ctx: &CliContext,
    args: &SkillsArgs,
    io: &mut Io<'_>,
) -> Result<(), qmd::Error> {
    let argv = &args.args;
    let full = args.full || argv.iter().any(|arg| arg == "--full");
    let all = args.all || argv.iter().any(|arg| arg == "--all");
    let names = argv
        .get(1..)
        .unwrap_or(&[])
        .iter()
        .filter(|arg| arg.as_str() != "--full" && arg.as_str() != "--all");
    let targets: Vec<SkillInfo> = if all {
        runtime_skills(env, ctx.cwd())
    } else {
        let mut found = Vec::new();
        for name in names {
            match find_skill(name, env, ctx.cwd()) {
                Some(skill) => found.push(skill),
                None => return Err(skill_error(format!("Skill not found: {name}"))),
            }
        }
        found
    };
    if targets.is_empty() {
        return Err(skill_error(
            "No skill name provided. Usage: qmd skills get <name>",
        ));
    }
    if args.json {
        let mut data = Vec::with_capacity(targets.len());
        for skill in &targets {
            data.push(skill_get_row(skill, full)?);
        }
        output_skills_json(io, &success_envelope(Value::Array(data)));
        return Ok(());
    }
    for (index, skill) in targets.iter().enumerate() {
        // `console.log("\n---\n")` — the argument's own newline plus
        // the call's.
        if index > 0 {
            let _ = writeln!(io.out, "\n---\n");
        }
        let content = read_skill_content(skill)?;
        write_body(io.out, &content);
        if full {
            for file in collect_skill_files(skill) {
                let _ = writeln!(io.out, "\n--- {} ---\n", file.relative_path);
                write_body(io.out, &file.content);
            }
        }
    }
    Ok(())
}

/// One `data` row of `skills get --json` (qmd.ts:3496-3503):
/// `{name, content}` plus a `files` key only under `--full`.
///
/// # Errors
/// [`qmd::Error::Io`] on a skill-read failure.
fn skill_get_row(skill: &SkillInfo, full: bool) -> Result<Value, qmd::Error> {
    let mut row = Map::new();
    row.insert("name".to_owned(), Value::from(skill.name.clone()));
    row.insert(
        "content".to_owned(),
        Value::from(read_skill_content(skill)?.into_owned()),
    );
    if full {
        row.insert(
            "files".to_owned(),
            Value::Array(
                collect_skill_files(skill)
                    .iter()
                    .map(|file| {
                        let mut entry = Map::new();
                        entry.insert("path".to_owned(), Value::from(file.relative_path.clone()));
                        entry.insert("content".to_owned(), Value::from(file.content.clone()));
                        Value::Object(entry)
                    })
                    .collect(),
            ),
        );
    }
    Ok(Value::Object(row))
}

/// `case "path"` (qmd.ts:3521-3536): no name → every search dir; a
/// name → that skill's dir (materializing the embedded tree first).
///
/// # Errors
/// [`qmd::Error::Skill`] for `Skill not found: <name>`;
/// [`qmd::Error::Io`] on materialize failure.
fn skills_path(
    env: &Environment,
    ctx: &CliContext,
    args: &SkillsArgs,
    io: &mut Io<'_>,
) -> Result<(), qmd::Error> {
    let argv = &args.args;
    let Some(name) = argv.get(1) else {
        let paths = skill_search_dirs(env)?;
        if args.json {
            let mut data = Map::new();
            data.insert(
                "paths".to_owned(),
                Value::Array(paths.iter().map(|p| Value::from(p.clone())).collect()),
            );
            output_skills_json(io, &success_envelope(Value::Object(data)));
        } else {
            for path in paths {
                let _ = writeln!(io.out, "{path}");
            }
        }
        return Ok(());
    };
    let skill = find_skill(name, env, ctx.cwd())
        .ok_or_else(|| skill_error(format!("Skill not found: {name}")))?;
    let dir = skill.dir(env)?;
    if args.json {
        let mut data = Map::new();
        data.insert("name".to_owned(), Value::from(skill.name.as_str()));
        data.insert(
            "path".to_owned(),
            Value::from(dir.to_string_lossy().into_owned()),
        );
        output_skills_json(io, &success_envelope(Value::Object(data)));
    } else {
        let _ = writeln!(io.out, "{}", dir.display());
    }
    Ok(())
}

/// `showSkillsHelp` (qmd.ts:3548-3560) — verbatim.
fn show_skills_help(io: &mut Io<'_>) {
    for line in [
        "Usage: qmd skills <list|get|path> [options]",
        "",
        "Commands:",
        "  list                 List bundled runtime skills",
        "  get <name>           Print a bundled runtime skill",
        "  get <name> --full    Include references/templates/scripts",
        "  get --all            Print all bundled runtime skills",
        "  path [name]          Print runtime skill directory path(s)",
        "",
        "Options:",
        "  --json               Print structured JSON",
    ] {
        let _ = writeln!(io.out, "{line}");
    }
}

/// `installSkill` (qmd.ts:3617-3633) — install, report, optionally link
/// `.claude/skills/qmd`. Errors propagate as [`Failure::Error`] like
/// upstream's `exitWithError` (message + doctor hint).
///
/// # Errors
/// [`Failure::Error`] on an existing target without `--force`, a
/// missing `qmd` skill, an existing `.claude` link path without
/// `--force`, or any filesystem failure.
fn install_skill(
    env: &Environment,
    ctx: &CliContext,
    args: &SkillArgs,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    let install_dir = skill_install_dir(args.global, env);
    write_skill_install(&install_dir, args.force, env, ctx)?;
    let _ = writeln!(io.out, "✓ Installed QMD skill to {}", install_dir.display());

    let claude_link = claude_skill_link_path(args.global, env);
    if !should_create_claude_symlink(&claude_link, args.yes, io) {
        return Ok(());
    }
    if ensure_claude_symlink(&claude_link, &install_dir, args.force)? {
        let _ = writeln!(io.out, "✓ Linked Claude skill at {}", claude_link.display());
    } else {
        let parent = claude_link
            .parent()
            .map_or_else(|| claude_link.clone(), Path::to_path_buf);
        let _ = writeln!(
            io.out,
            "✓ Claude already sees the skill via {}",
            parent.display()
        );
    }
    Ok(())
}

/// `writeSkillInstall` (qmd.ts:3436-3451): refuse an existing target
/// without `--force`, copy the skill directory, then overwrite
/// `SKILL.md` with the bootstrap stub.
///
/// # Errors
/// [`qmd::Error::Skill`] on the upstream message errors;
/// [`qmd::Error::Io`] on filesystem failures.
fn write_skill_install(
    target_dir: &Path,
    force: bool,
    env: &Environment,
    ctx: &CliContext,
) -> Result<(), qmd::Error> {
    if path_exists(target_dir) {
        if !force {
            return Err(skill_error(format!(
                "Skill already exists: {} (use --force to replace it)",
                target_dir.display()
            )));
        }
        remove_path(target_dir)?;
    }
    let skill = find_skill("qmd", env, ctx.cwd()).ok_or_else(|| skill_error(SKILL_NOT_FOUND))?;
    let source_dir = skill.dir(env)?;
    copy_directory_contents(&source_dir, target_dir)?;
    let stub_path = target_dir.join("SKILL.md");
    fs::write(&stub_path, INSTALLED_SKILL_STUB).map_err(|err| io_error(&stub_path, err))
}

/// `ensureClaudeSymlink` (qmd.ts:3562-3592) — `true` when the link was
/// created or already points at the right target; `false` when
/// `.claude/skills` already resolves to the install dir's parent (the
/// skill is already visible and a `qmd -> qmd` link would loop).
///
/// # Errors
/// [`qmd::Error::Skill`] on the existing-path refusal;
/// [`qmd::Error::Io`] on filesystem failures.
fn ensure_claude_symlink(
    link_path: &Path,
    target_dir: &Path,
    force: bool,
) -> Result<bool, qmd::Error> {
    let parent_dir = link_path.parent().unwrap_or_else(|| Path::new("."));
    if path_exists(parent_dir) {
        // `realpathSync` throws when the path is missing — but the
        // install dir was just written, so only `parent_dir` could
        // legitimately fail and it is gated by `path_exists`.
        let target_parent = target_dir.parent().unwrap_or_else(|| Path::new("."));
        let resolved_target =
            fs::canonicalize(target_parent).map_err(|err| io_error(target_parent, err))?;
        let resolved_parent =
            fs::canonicalize(parent_dir).map_err(|err| io_error(parent_dir, err))?;
        if resolved_target == resolved_parent {
            return Ok(false);
        }
    }

    let relative = path_relative(parent_dir, target_dir);
    let link_target = if relative.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        relative
    };

    fs::create_dir_all(parent_dir).map_err(|err| io_error(parent_dir, err))?;

    if path_exists(link_path) {
        let meta = fs::symlink_metadata(link_path).map_err(|err| io_error(link_path, err))?;
        if meta.file_type().is_symlink()
            && fs::read_link(link_path).map_err(|err| io_error(link_path, err))? == link_target
        {
            return Ok(true);
        }
        if !force {
            return Err(skill_error(format!(
                "Claude skill path already exists: {} (use --force to replace it)",
                link_path.display()
            )));
        }
        remove_path(link_path)?;
    }

    symlink_dir(&link_target, link_path).map_err(|err| io_error(link_path, err))?;
    Ok(true)
}

/// `shouldCreateClaudeSymlink` (qmd.ts:3594-3615): `--yes` accepts;
/// non-interactive runs print the manual tip on stdout and decline;
/// interactive runs get the `[y/N]` prompt (`y`/`yes`,
/// case-insensitive — `confirmOnTty` semantics).
fn should_create_claude_symlink(link_path: &Path, auto_yes: bool, io: &mut Io<'_>) -> bool {
    if auto_yes {
        return true;
    }
    if !(io.stdin_tty && io.stdout_tty) {
        let _ = writeln!(
            io.out,
            "Tip: create a Claude symlink manually at {}",
            link_path.display()
        );
        return false;
    }
    confirm_on_tty(
        &format!("Create a symlink in {}? [y/N] ", link_path.display()),
        io,
    )
}

/// `getSkillInstallDir` (qmd.ts:3226-3230): `~/.agents/skills/qmd`
/// global, `$PWD/.agents/skills/qmd` local (upstream `getPwd()`).
#[must_use]
fn skill_install_dir(global_install: bool, env: &Environment) -> PathBuf {
    let base = if global_install {
        paths::home_dir(env)
    } else {
        paths::pwd(env)
    };
    base.join(".agents").join("skills").join("qmd")
}

/// `getClaudeSkillLinkPath` (qmd.ts:3232-3236): `.claude` instead of
/// `.agents`, same global/local split.
#[must_use]
fn claude_skill_link_path(global_install: bool, env: &Environment) -> PathBuf {
    let base = if global_install {
        paths::home_dir(env)
    } else {
        paths::pwd(env)
    };
    base.join(".claude").join("skills").join("qmd")
}

/// `getSkillSearchDirs` (qmd.ts:3283-3293): `QMD_SKILLS_DIR` verbatim —
/// upstream returns the env string unresolved and existence-unchecked —
/// else the embedded dir once it exists on disk. Upstream's
/// `_runtimeOnly` parameter is dead code and is not carried.
///
/// # Errors
/// [`qmd::Error::Io`] when the embedded tree cannot be written.
fn skill_search_dirs(env: &Environment) -> Result<Vec<String>, qmd::Error> {
    if let Some(dir) = env.qmd_skills_dir.as_deref().filter(|d| !d.is_empty()) {
        return Ok(vec![dir.to_owned()]);
    }
    // `existsSync(dir) ? [dir] : []` — materializing first makes the
    // embedded dir exist before it is returned.
    Ok(vec![
        materialize_embedded_skills(env)?
            .to_string_lossy()
            .into_owned(),
    ])
}

/// Write the embedded skill tree under
/// `<cache-dir>/skills/<crate-version>` and return that dir — the
/// embedded counterpart of upstream's package `skills/` dir. Lazy by
/// design: only `skills path*`/`skill install` need real paths;
/// `list`/`get`/`show` serve content from the binary.
///
/// # Errors
/// [`qmd::Error::Io`] when a directory or file cannot be written.
fn materialize_embedded_skills(env: &Environment) -> Result<PathBuf, qmd::Error> {
    let dir = paths::cache_dir(env)
        .join("skills")
        .join(env!("CARGO_PKG_VERSION"));
    for skill in EMBEDDED_SKILLS {
        for (relative, content) in skill.files {
            let path = dir.join(skill.dir).join(relative);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|err| io_error(parent, err))?;
            }
            fs::write(&path, content).map_err(|err| io_error(&path, err))?;
        }
    }
    Ok(dir)
}

/// `discoverSkills` (qmd.ts:3328-3357): for `QMD_SKILLS_DIR`, readdir
/// and keep entries with a readable, parseable `SKILL.md`; otherwise
/// parse every embedded skill's frontmatter. Upstream's `runtimeOnly`
/// is a no-op inside discovery (the `!hidden` filter lives in
/// `runtimeSkills()`), so it is dropped.
fn discover_skills(env: &Environment, cwd: &Path) -> Vec<SkillInfo> {
    let mut skills = env
        .qmd_skills_dir
        .as_deref()
        .filter(|dir| !dir.is_empty())
        .map_or_else(embedded_skills, |dir| discover_dir_skills(dir, cwd));
    // `localeCompare` — byte order differs from locale collation only
    // for non-ASCII names; skill dirs are ASCII in practice.
    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills
}

/// The embedded branch of `discoverSkills`: parse each bundled
/// `SKILL.md`'s frontmatter, skipping tables that lack or fail it.
fn embedded_skills() -> Vec<SkillInfo> {
    EMBEDDED_SKILLS
        .iter()
        .enumerate()
        .filter_map(|(index, skill)| {
            let frontmatter = parse_skill_frontmatter(embedded_main_file(skill)?)?;
            Some(SkillInfo {
                name: frontmatter.name,
                description: frontmatter.description,
                hidden: frontmatter.hidden,
                source: SkillSource::Embedded(index),
            })
        })
        .collect()
}

/// The `QMD_SKILLS_DIR` branch of `discoverSkills` (qmd.ts:3330-3351):
/// `readdirSync`, then keep entries whose `SKILL.md` exists, reads and
/// parses — every failure mode skips the entry.
fn discover_dir_skills(search_dir: &str, cwd: &Path) -> Vec<SkillInfo> {
    let mut skills = Vec::new();
    // `readdirSync(dir)` — an unreadable dir contributes nothing.
    if let Ok(entries) = fs::read_dir(search_dir) {
        // `resolve(dir, entry)` — upstream's resolve anchors at
        // `process.cwd()` and normalizes.
        let base = paths::resolve_lexical(cwd, search_dir);
        for entry in entries.flatten() {
            if let Some(skill) = read_dir_skill(base.join(entry.file_name())) {
                skills.push(skill);
            }
        }
    }
    skills
}

/// One `readdirSync` entry of `discoverSkills` (qmd.ts:3338-3350):
/// `existsSync` gates the read (a directory named `SKILL.md` still
/// counts and fails on the read, like upstream), `readFileSync` and
/// the frontmatter parse each skip on failure.
fn read_dir_skill(skill_dir: PathBuf) -> Option<SkillInfo> {
    let skill_path = skill_dir.join("SKILL.md");
    if !skill_path.exists() {
        return None;
    }
    let content = fs::read_to_string(&skill_path).ok()?;
    let frontmatter = parse_skill_frontmatter(&content)?;
    Some(SkillInfo {
        name: frontmatter.name,
        description: frontmatter.description,
        hidden: frontmatter.hidden,
        source: SkillSource::Dir(skill_dir),
    })
}

/// The embedded skill's `SKILL.md` body, or `None` if its table lacks
/// one (upstream: an entry without `SKILL.md` is skipped).
fn embedded_main_file(skill: &EmbeddedSkill) -> Option<&'static str> {
    skill
        .files
        .iter()
        .find_map(|(path, content)| (*path == "SKILL.md").then_some(*content))
}

/// `runtimeSkills()` inside `runSkillsCommand` (qmd.ts:3459):
/// discovery minus `hidden:` skills — the only place the flag is
/// honored (`list` and `get --all`; `findSkill` ignores it upstream).
fn runtime_skills(env: &Environment, cwd: &Path) -> Vec<SkillInfo> {
    discover_skills(env, cwd)
        .into_iter()
        .filter(|skill| !skill.hidden)
        .collect()
}

/// `findSkill` (qmd.ts:3359) — first name match over discovery; hidden
/// skills still match (upstream quirk: `get <name>`/`path <name>` see
/// them even though `list` does not).
fn find_skill(name: &str, env: &Environment, cwd: &Path) -> Option<SkillInfo> {
    discover_skills(env, cwd)
        .into_iter()
        .find(|skill| skill.name == name)
}

/// `readSkillContent` (qmd.ts:3363): embedded skills serve `SKILL.md`
/// from the binary; directory skills read it from disk each call.
///
/// # Errors
/// [`qmd::Error::Io`] when a directory skill's `SKILL.md` cannot be
/// read.
fn read_skill_content(skill: &SkillInfo) -> Result<Cow<'static, str>, qmd::Error> {
    match &skill.source {
        SkillSource::Embedded(index) => Ok(Cow::Borrowed(
            EMBEDDED_SKILLS
                .get(*index)
                .and_then(embedded_main_file)
                .unwrap_or(""),
        )),
        SkillSource::Dir(dir) => {
            let path = dir.join("SKILL.md");
            fs::read_to_string(&path)
                .map(Cow::Owned)
                .map_err(|err| io_error(&path, err))
        }
    }
}

/// `collectSkillFiles` (qmd.ts:3364-3380): only the
/// `references`/`templates`/`scripts` subdirs, entries sorted per
/// subdir, regular files only, unreadable files skipped silently.
/// Embedded skills play the same shape over their virtual file table.
fn collect_skill_files(skill: &SkillInfo) -> Vec<SkillFile> {
    let mut files = Vec::new();
    for subdir_name in ["references", "templates", "scripts"] {
        match &skill.source {
            SkillSource::Embedded(index) => {
                collect_embedded_files(*index, subdir_name, &mut files);
            }
            SkillSource::Dir(dir) => {
                collect_dir_files(dir, subdir_name, &mut files);
            }
        }
    }
    files
}

/// The embedded branch of `collectSkillFiles`: table entries under
/// `{subdir}/`, sorted like `readdirSync(subdir).sort()`.
fn collect_embedded_files(index: usize, subdir_name: &str, files: &mut Vec<SkillFile>) {
    let Some(skill) = EMBEDDED_SKILLS.get(index) else {
        return;
    };
    let mut entries: Vec<(&str, &str)> = skill
        .files
        .iter()
        .filter_map(|(path, content)| {
            let name = path
                .strip_prefix(subdir_name)
                .and_then(|rest| rest.strip_prefix('/'))?;
            // `readdirSync` returns one level: a nested `sub/x.md` is
            // reached only through `sub`, a non-file entry upstream
            // skips — mirror that by keeping flat names only.
            (!name.contains('/')).then_some((name, *content))
        })
        .collect();
    // `readdirSync(subdir).sort()` — UTF-16 code-unit order.
    entries.sort_by(|a, b| utf16_cmp(a.0, b.0));
    files.extend(entries.into_iter().map(|(name, content)| SkillFile {
        relative_path: format!("{subdir_name}/{name}"),
        content: content.to_owned(),
    }));
}

/// The `Dir` branch of `collectSkillFiles` (qmd.ts:3367-3378):
/// `existsSync` gates the subdir, `readdirSync(subdir).sort()` orders,
/// `statSync(...).isFile()` and the read each skip on failure.
fn collect_dir_files(dir: &Path, subdir_name: &str, files: &mut Vec<SkillFile>) {
    let subdir = dir.join(subdir_name);
    if !subdir.exists() {
        return;
    }
    let Ok(entries) = fs::read_dir(&subdir) else {
        return;
    };
    let mut names: Vec<String> = entries
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort_by(|a, b| utf16_cmp(a, b));
    for name in names {
        if let Some(file) = read_skill_file(&subdir, &name, subdir_name) {
            files.push(file);
        }
    }
}

/// One `readdirSync` entry of `collectSkillFiles`: a regular file
/// (`statSync` follows symlinks) that reads as UTF-8, else skipped
/// silently.
fn read_skill_file(subdir: &Path, name: &str, subdir_name: &str) -> Option<SkillFile> {
    let file_path = subdir.join(name);
    let meta = fs::metadata(&file_path).ok()?;
    if !meta.is_file() {
        return None;
    }
    let content = fs::read_to_string(&file_path).ok()?;
    Some(SkillFile {
        relative_path: format!("{subdir_name}/{name}"),
        content,
    })
}

/// `parseSkillFrontmatter` (qmd.ts:3295-3326): `---` fence, then
/// `name:`/`description:`/`hidden:` line prefixes; `description`
/// continuation lines (`^\s+\S`) are consumed and joined with `" "`.
/// No `name` → `None`.
fn parse_skill_frontmatter(content: &str) -> Option<SkillFrontmatter> {
    let trimmed = content.trim_start_matches(is_js_ws);
    if !trimmed.starts_with("---") {
        return None;
    }
    // `indexOf("\n---")` — first occurrence after the opening fence.
    let end = trimmed.get(3..)?.find("\n---")?;
    let frontmatter = trimmed.get(3..3 + end)?;

    let mut name = String::new();
    let mut description = String::new();
    let mut hidden = false;
    // `frontmatter.split(/\r?\n/)`.
    let lines: Vec<&str> = frontmatter
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .collect();

    let mut index = 0;
    while let Some(line) = lines.get(index).copied() {
        if let Some(rest) = line.strip_prefix("name:") {
            js_trim(rest).clone_into(&mut name);
        } else if let Some(rest) = line.strip_prefix("description:") {
            let mut parts = vec![js_trim(rest).to_owned()];
            // `while (i + 1 < lines.length && /^\s+\S/.test(lines[i+1])) { i++; parts.push(lines[i].trim()) }`
            while let Some(next) = lines.get(index + 1).copied()
                && is_continuation(next)
            {
                index += 1;
                parts.push(js_trim(next).to_owned());
            }
            description = parts.join(" ");
        } else if let Some(rest) = line.strip_prefix("hidden:") {
            let value = js_trim(rest).to_lowercase();
            hidden = value == "true" || value == "yes";
        }
        index += 1;
    }

    if name.is_empty() {
        return None;
    }
    Some(SkillFrontmatter {
        name,
        description,
        hidden,
    })
}

/// Upstream's `/^\s+\S/` continuation test: at least one whitespace
/// char, then a non-whitespace char somewhere after it.
fn is_continuation(line: &str) -> bool {
    line.chars().next().is_some_and(is_js_ws) && !js_trim(line).is_empty()
}

/// `copyDirectoryContents` (qmd.ts:3393-3405): recursive copy of a
/// skill dir; `statSync` follows symlinks like upstream.
///
/// # Errors
/// [`qmd::Error::Io`] on any read/write failure.
fn copy_directory_contents(source_dir: &Path, target_dir: &Path) -> Result<(), qmd::Error> {
    fs::create_dir_all(target_dir).map_err(|err| io_error(target_dir, err))?;
    for entry in fs::read_dir(source_dir).map_err(|err| io_error(source_dir, err))? {
        let entry = entry.map_err(|err| io_error(source_dir, err))?;
        let source_path = source_dir.join(entry.file_name());
        let target_path = target_dir.join(entry.file_name());
        let stat = fs::metadata(&source_path).map_err(|err| io_error(&source_path, err))?;
        if stat.is_dir() {
            copy_directory_contents(&source_path, &target_path)?;
        } else if stat.is_file() {
            fs::copy(&source_path, &target_path).map_err(|err| io_error(&target_path, err))?;
        }
    }
    Ok(())
}

/// `pathExists` (qmd.ts:3238-3245): `lstatSync` — a dangling symlink
/// counts as existing.
fn path_exists(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok()
}

/// `removePath` (qmd.ts:3247-3254): `lstat` directories go through a
/// recursive remove, everything else (files, symlinks — `lstat` never
/// reports a link as a dir) unlinks.
///
/// # Errors
/// [`qmd::Error::Io`] on filesystem failure.
fn remove_path(path: &Path) -> Result<(), qmd::Error> {
    let stat = fs::symlink_metadata(path).map_err(|err| io_error(path, err))?;
    if stat.is_dir() {
        fs::remove_dir_all(path).map_err(|err| io_error(path, err))
    } else {
        fs::remove_file(path).map_err(|err| io_error(path, err))
    }
}

/// Node `path.relative(from, to)` for already-absolute paths: `..` per
/// unconsumed `from` component, then the rest of `to`. An empty result
/// (same path) stays empty — callers map it to `"."` like upstream's
/// `|| "."`.
fn path_relative(from: &Path, to: &Path) -> PathBuf {
    let from_components: Vec<_> = from.components().collect();
    let to_components: Vec<_> = to.components().collect();
    let common = from_components
        .iter()
        .zip(&to_components)
        .take_while(|(a, b)| a == b)
        .count();
    let mut out = PathBuf::new();
    for _ in from_components.iter().skip(common) {
        out.push("..");
    }
    for component in to_components.iter().skip(common) {
        out.push(component.as_os_str());
    }
    out
}

/// `symlinkSync(target, path, "dir")` — a directory symlink.
#[cfg(unix)]
fn symlink_dir(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

/// `symlinkSync(target, path, "dir")` — a directory symlink.
#[cfg(windows)]
fn symlink_dir(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_dir(target, link)
}

impl SkillInfo {
    /// The `skill.dir` accessor (upstream's `SkillInfo.dir`, a plain
    /// string): directory skills already hold their resolved path;
    /// embedded skills materialize the tree first.
    ///
    /// # Errors
    /// [`qmd::Error::Io`] when the embedded tree cannot be written.
    fn dir(&self, env: &Environment) -> Result<PathBuf, qmd::Error> {
        match &self.source {
            SkillSource::Dir(dir) => Ok(dir.clone()),
            SkillSource::Embedded(index) => {
                let base = materialize_embedded_skills(env)?;
                Ok(EMBEDDED_SKILLS
                    .get(*index)
                    .map_or_else(|| base.clone(), |skill| base.join(skill.dir)))
            }
        }
    }
}

/// `outputSkillsJson` (qmd.ts:3453-3455): `JSON.stringify` — single
/// line, stdout.
fn output_skills_json(io: &mut Io<'_>, payload: &Value) {
    // `serde_json` escapes like `JSON.stringify` (see `crate::json`);
    // these payloads cannot fail to serialize.
    let text = serde_json::to_string(payload).unwrap_or_else(|_| "null".to_owned());
    let _ = writeln!(io.out, "{text}");
}

/// `{success: true, data: <data>}` in upstream member order.
fn success_envelope(data: Value) -> Value {
    let mut payload = Map::new();
    payload.insert("success".to_owned(), Value::Bool(true));
    payload.insert("data".to_owned(), data);
    Value::Object(payload)
}

/// `process.stdout.write(content.endsWith("\n") ? content : content + "\n")`.
fn write_body(out: &mut dyn Write, content: &str) {
    if content.ends_with('\n') {
        let _ = write!(out, "{content}");
    } else {
        let _ = writeln!(out, "{content}");
    }
}

/// `String.prototype.trim()` — the JS `\s` set (includes U+FEFF);
/// delegates to the shared [`qmd::js::js_trim`].
fn js_trim(text: &str) -> &str {
    qmd::js::js_trim(text)
}

/// `name.padEnd(maxName)` — pad with spaces to `target` UTF-16 units.
fn pad_end(text: &str, target: usize) -> String {
    let len = utf16_len(text);
    if len >= target {
        return text.to_owned();
    }
    format!("{text}{}", " ".repeat(target - len))
}

/// JS `Array#sort()` default string order — UTF-16 code-unit
/// comparison (Rust's `str::cmp` is code-point order, which differs for
/// astral-plane names).
fn utf16_cmp(a: &str, b: &str) -> Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

/// A `qmd::Error` that prints an upstream message verbatim — the JS
/// `Error.message` equivalent through both error channels.
fn skill_error(reason: impl Into<String>) -> qmd::Error {
    qmd::Error::Skill {
        reason: reason.into(),
    }
}

/// `qmd::Error::Io` shorthand for filesystem failures.
fn io_error(path: &Path, error: std::io::Error) -> qmd::Error {
    qmd::Error::Io {
        path: path.to_path_buf(),
        source: error,
    }
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::missing_assert_message,
    reason = "idiomatic test-code patterns"
)]
mod tests {
    use super::*;

    /// `parseSkillFrontmatter` — the happy path plus the multi-line
    /// `description` continuation join.
    #[test]
    fn frontmatter_parses_name_description_hidden() {
        let parsed = parse_skill_frontmatter(
            "---\nname: qmd\ndescription: First line\n  second line.\nhidden: yes\n---\n\n# Body\n",
        )
        .unwrap();
        assert_eq!(parsed.name, "qmd");
        assert_eq!(parsed.description, "First line second line.");
        assert!(parsed.hidden);
    }

    /// `hidden:` is honored only for `true`/`yes`, case-insensitively.
    #[test]
    fn frontmatter_hidden_values() {
        let hidden = |v: &str| {
            parse_skill_frontmatter(&format!("---\nname: n\nhidden: {v}\n---\n"))
                .unwrap()
                .hidden
        };
        assert!(hidden("true"));
        assert!(hidden("YES"));
        assert!(!hidden("1"));
        assert!(!hidden("no"));
    }

    /// Rejections: no `---` fence, no closing `\n---`, no `name:`.
    #[test]
    fn frontmatter_rejects_unparseable() {
        assert!(parse_skill_frontmatter("no fence\n---\nname: x\n---\n").is_none());
        assert!(parse_skill_frontmatter("---\nname: x\n").is_none());
        assert!(parse_skill_frontmatter("---\ndescription: d\n---\n").is_none());
        // `\r\n` line endings still parse.
        let parsed = parse_skill_frontmatter("---\r\nname: n\r\n---\r\n").unwrap();
        assert_eq!(parsed.name, "n");
    }

    /// `/^\s+\S/` — leading whitespace then a non-whitespace char.
    #[test]
    fn continuation_lines() {
        assert!(is_continuation("  x"));
        assert!(is_continuation("\tx"));
        assert!(!is_continuation("x"));
        assert!(!is_continuation("   "));
        assert!(!is_continuation(""));
    }

    /// Node `path.relative` shapes the `.claude/skills/qmd` link target.
    #[test]
    fn path_relative_shapes() {
        let from = Path::new("/w/.claude/skills");
        assert_eq!(
            path_relative(from, Path::new("/w/.agents/skills/qmd")),
            PathBuf::from("../../.agents/skills/qmd")
        );
        assert_eq!(
            path_relative(from, from),
            PathBuf::new(),
            "same path stays empty — callers map it to `.`"
        );
    }

    /// `padEnd` counts UTF-16 units, not bytes — an astral char is two.
    #[test]
    fn pad_end_utf16() {
        assert_eq!(pad_end("qmd", 5), "qmd  ");
        assert_eq!(pad_end("qmd", 3), "qmd");
        assert_eq!(pad_end("\u{1f600}", 3), "\u{1f600} ");
    }

    /// `Array#sort` order is UTF-16 code units: an astral char sorts
    /// after `z` even though its code point is higher.
    #[test]
    fn utf16_sort_order() {
        assert_eq!(utf16_cmp("z", "\u{1f600}"), Ordering::Less);
        let mut names = vec!["\u{1f600}", "z"];
        names.sort_by(|a, b| utf16_cmp(a, b));
        assert_eq!(names, ["z", "\u{1f600}"]);
    }
}
