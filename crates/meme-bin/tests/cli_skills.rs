//! Ports of upstream `test/cli.test.ts` `CLI Skills` and
//! `CLI Skill Commands` describe blocks, plus coverage for the
//! surfaces those tests don't reach: the `skills --json` envelopes,
//! `skills path` materialization, `MEME_SKILLS_DIR` discovery, the
//! interactive symlink prompt (via an in-process `run` with a TTY
//! [`Io`]), and the deliberate `release`-skill absence (P4-D7 — the
//! maintainer skill is not ported).

#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::shadow_unrelated,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_assert_message,
    clippy::panic,
    reason = "idiomatic test-code patterns"
)]

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use common::CliEnv;

/// Assert the install produced the bootstrap stub (upstream
/// `installedSkillStubContent`) and copied payload files.
fn assert_installed(skill_dir: &Path) {
    let stub = fs::read_to_string(skill_dir.join("SKILL.md")).unwrap();
    assert!(stub.contains("!`meme skill show`"), "stub: {stub}");
    assert!(stub.contains("meme get"), "stub: {stub}");
    assert!(
        !stub.contains("## MCP Tool: `query`"),
        "the stub replaces the full SKILL.md: {stub}"
    );
    let reference = fs::read_to_string(skill_dir.join("references/mcp-setup.md")).unwrap();
    assert!(reference.contains("# MEME MCP Server Setup"), "{reference}");
}

/// Upstream `CLI Skills` — `skills list` (cli.test.ts:255-261). The
/// port ships `meme` only: `release` is a maintainer skill (P4-D7), so
/// its absence is asserted instead of relying on golden scenarios.
#[test]
fn skills_list_shows_meme_without_release() {
    let env = CliEnv::new();
    let out = env.run(&["skills", "list"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stderr.is_empty());
    assert!(out.stdout.contains("meme"), "{}", out.stdout);
    assert!(
        out.stdout.contains("Search local markdown knowledge bases"),
        "{}",
        out.stdout
    );
    assert!(!out.stdout.contains("release"), "{}", out.stdout);
}

/// `skills list --json` — the `{success:true, data:[…]}` envelope.
#[test]
fn skills_list_json_envelope() {
    let env = CliEnv::new();
    let out = env.run(&["skills", "list", "--json"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let json: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(json["success"], true);
    let data = json["data"].as_array().unwrap();
    assert_eq!(data.len(), 1, "{data:?}");
    assert_eq!(data[0]["name"], "meme");
}

/// Upstream `gets version-matched runtime skill content`
/// (cli.test.ts:263-269) — verbatim embedded SKILL.md.
#[test]
fn skills_get_meme_content() {
    let env = CliEnv::new();
    let out = env.run(&["skills", "get", "meme"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("# MEME - Query Markdown Documents"));
    assert!(out.stdout.contains("## MCP Tool: `query`"));
    assert!(!out.stdout.contains("This file is a discovery stub"));
}

/// Upstream `gets runtime skill with supplementary references`
/// (cli.test.ts:272-279).
#[test]
fn skills_get_meme_full_includes_references() {
    let env = CliEnv::new();
    let out = env.run(&["skills", "get", "meme", "--full"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("# MEME - Query Markdown Documents"));
    assert!(out.stdout.contains("--- references/mcp-setup.md ---"));
    assert!(out.stdout.contains("# MEME MCP Server Setup"));
}

/// `skills get` error paths: no name, unknown name, and the `--json`
/// `{success:false,error}` envelope on **stdout** (qmd.ts:4927-4935).
#[test]
fn skills_get_errors() {
    let env = CliEnv::new();
    let out = env.run(&["skills", "get"]);
    assert_eq!(out.exit_code, 1);
    assert_eq!(
        out.stderr,
        "No skill name provided. Usage: meme skills get <name>\n"
    );

    let out = env.run(&["skills", "get", "missing"]);
    assert_eq!(out.exit_code, 1);
    assert_eq!(out.stderr, "Skill not found: missing\n");

    let out = env.run(&["skills", "get", "missing", "--json"]);
    assert_eq!(out.exit_code, 1);
    assert_eq!(
        out.stdout,
        "{\"success\":false,\"error\":\"Skill not found: missing\"}\n"
    );
    assert!(out.stderr.is_empty());
}

/// Unknown `skills` subcommand — bare message on stderr, exit 1, no
/// doctor hint (qmd.ts:4931-4934).
#[test]
fn skills_unknown_subcommand() {
    let env = CliEnv::new();
    let out = env.run(&["skills", "bogus"]);
    assert_eq!(out.exit_code, 1);
    assert_eq!(out.stderr, "Unknown skills subcommand: bogus\n");

    let out = env.run(&["skills", "bogus", "--json"]);
    assert_eq!(out.exit_code, 1);
    assert_eq!(
        out.stdout,
        "{\"success\":false,\"error\":\"Unknown skills subcommand: bogus\"}\n"
    );
}

/// `skills help` prints the dedicated usage (upstream `skills --help`
/// falls through to the general help — covered by the golden replay).
#[test]
fn skills_help_text() {
    let env = CliEnv::new();
    let out = env.run(&["skills", "help"]);
    assert_eq!(out.exit_code, 0);
    assert!(
        out.stdout
            .contains("Usage: meme skills <list|get|path> [options]")
    );
    assert!(out.stdout.contains("get --all"));
}

/// Upstream `prints canonical repository skill path`
/// (cli.test.ts:281-286) — adapted: the embedded tree materializes
/// under `<cache>/skills/<version>` (P4-D7), not a package dir.
#[test]
fn skills_path_materializes_embedded_tree() {
    let env = CliEnv::new();
    let out = env.run(&["skills", "path", "meme"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let dir = out.stdout.trim().to_owned();
    let expected = env
        .root
        .join("xdg-cache/meme/skills")
        .join(env!("CARGO_PKG_VERSION"))
        .join("meme");
    assert_eq!(Path::new(&dir), expected.as_path(), "{dir}");
    assert!(Path::new(&dir).join("SKILL.md").exists());
    assert!(Path::new(&dir).join("references/mcp-setup.md").exists());
}

/// `skills path` without a name prints the search dir — the same
/// materialized skills root.
#[test]
fn skills_path_prints_search_dir() {
    let env = CliEnv::new();
    let out = env.run(&["skills", "path"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let dir = out.stdout.trim().to_owned();
    assert_eq!(
        Path::new(&dir),
        env.root
            .join("xdg-cache/meme/skills")
            .join(env!("CARGO_PKG_VERSION"))
            .as_path(),
        "{dir}"
    );
}

/// Upstream `legacy skill show` (cli.test.ts:288-295).
#[test]
fn skill_show_prints_canonical_skill() {
    let env = CliEnv::new();
    let out = env.run(&["skill", "show"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.starts_with("MEME Skill\n\n"));
    assert!(out.stdout.contains("# MEME - Query Markdown Documents"));
    assert!(out.stdout.contains("## MCP Tool: `query`"));
}

/// Upstream `shows embedded skill with --skill alias`
/// (cli.test.ts:359-365) — including the flag beating a command.
#[test]
fn skill_flag_alias() {
    let env = CliEnv::new();
    for argv in [&["--skill"][..], &["--skill", "search", "x"][..]] {
        let out = env.run(argv);
        assert_eq!(out.exit_code, 0, "{argv:?} stderr: {}", out.stderr);
        assert!(out.stdout.contains("MEME Skill"));
        assert!(out.stdout.contains("name: meme"));
        assert!(
            out.stdout
                .contains("allowed-tools: Bash(meme:*), mcp__meme__*")
        );
    }
}

/// Upstream `shows skill help with -h` (cli.test.ts:367-372) — plus
/// the bare/`help` spellings (exit-0 quirk, qmd.ts:4959-4968).
#[test]
fn skill_usage_spellings() {
    let env = CliEnv::new();
    for argv in [
        &["skill"][..],
        &["skill", "help"][..],
        &["skill", "-h"][..],
        &["skill", "--help"][..],
    ] {
        let out = env.run(argv);
        assert_eq!(out.exit_code, 0, "{argv:?}");
        assert!(
            out.stdout
                .contains("Usage: meme skill <show|install> [options]"),
            "{argv:?}: {}",
            out.stdout
        );
        assert!(out.stdout.contains("install"));
        assert!(out.stdout.contains("--global"));
    }
}

/// Unknown `skill` subcommand — message + usage pointer + doctor
/// hint, exit 1 (qmd.ts:4970-4974).
#[test]
fn skill_unknown_subcommand() {
    let env = CliEnv::new();
    let out = env.run(&["skill", "bogus"]);
    assert_eq!(out.exit_code, 1);
    assert_eq!(
        out.stderr,
        "Unknown subcommand: bogus\nRun 'meme skill help' for usage\n\
         If meme still behaves unexpectedly, run 'meme doctor' for diagnostics.\n"
    );
}

/// Upstream `installs the skill into the current project`
/// (cli.test.ts:375-388) — non-TTY run prints the Tip and does not
/// link.
#[test]
fn skill_install_local_tip() {
    let env = CliEnv::new();
    let project = env.root.join("project");
    fs::create_dir_all(&project).unwrap();
    let out = env.run_at(&["skill", "install"], &project);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let skill_dir = project.join(".agents/skills/meme");
    assert_installed(&skill_dir);
    assert!(
        !project.join(".claude/skills/meme").exists(),
        "non-TTY install leaves no link"
    );
    assert!(
        out.stdout.contains(&format!(
            "✓ Installed MEME skill to {}",
            skill_dir.display()
        )),
        "{}",
        out.stdout
    );
    assert!(
        out.stdout.contains("Tip: create a Claude symlink manually"),
        "{}",
        out.stdout
    );
}

/// Upstream `installs globally and creates the Claude symlink with
/// --yes` (cli.test.ts:390-408) — relative link target.
#[test]
fn skill_install_global_yes() {
    let env = CliEnv::new();
    let fake_home = env.root.join("fake-home");
    fs::create_dir_all(&fake_home).unwrap();
    let out = env.run_opts(
        &["skill", "install", "--global", "--yes"],
        &env.fixtures,
        false,
        &[("HOME", fake_home.to_str().unwrap())],
    );
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let skill_dir = fake_home.join(".agents/skills/meme");
    let link = fake_home.join(".claude/skills/meme");
    assert_installed(&skill_dir);
    let meta = fs::symlink_metadata(&link).unwrap();
    assert!(meta.file_type().is_symlink());
    assert_eq!(
        fs::read_link(&link).unwrap(),
        PathBuf::from("../../.agents/skills/meme")
    );
    assert!(link.join("SKILL.md").exists(), "link resolves");
    assert!(
        out.stdout.contains(&format!(
            "✓ Installed MEME skill to {}",
            skill_dir.display()
        )),
        "{}",
        out.stdout
    );
    assert!(
        out.stdout
            .contains(&format!("✓ Linked Claude skill at {}", link.display())),
        "{}",
        out.stdout
    );
}

/// Upstream `skips Claude qmd symlink when .claude/skills already
/// points to .agents/skills` (cli.test.ts:410-426).
#[test]
fn skill_install_sees_existing_link_parent() {
    let env = CliEnv::new();
    let fake_home = env.root.join("shared-home");
    fs::create_dir_all(fake_home.join(".agents")).unwrap();
    fs::create_dir_all(fake_home.join(".claude")).unwrap();
    std::os::unix::fs::symlink(
        fake_home.join(".agents/skills"),
        fake_home.join(".claude/skills"),
    )
    .unwrap();
    let out = env.run_opts(
        &["skill", "install", "--global", "--yes"],
        &env.fixtures,
        false,
        &[("HOME", fake_home.to_str().unwrap())],
    );
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let skill_dir = fake_home.join(".agents/skills/meme");
    assert!(
        !fs::symlink_metadata(&skill_dir)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_installed(&skill_dir);
    assert!(
        out.stdout.contains(&format!(
            "✓ Claude already sees the skill via {}",
            fake_home.join(".claude/skills").display()
        )),
        "{}",
        out.stdout
    );
}

/// Upstream `refuses to overwrite an existing install without
/// --force` (cli.test.ts:428-440) — second install, then `-f`.
#[test]
fn skill_install_force_replaces() {
    let env = CliEnv::new();
    let project = env.root.join("project-force");
    fs::create_dir_all(&project).unwrap();

    let first = env.run_at(&["skill", "install"], &project);
    assert_eq!(first.exit_code, 0, "stderr: {}", first.stderr);

    let second = env.run_at(&["skill", "install"], &project);
    assert_eq!(second.exit_code, 1);
    assert!(
        second.stderr.contains("Skill already exists"),
        "{}",
        second.stderr
    );
    assert!(second.stderr.contains("--force"), "{}", second.stderr);

    let forced = env.run_at(&["skill", "install", "-f", "--yes"], &project);
    assert_eq!(forced.exit_code, 0, "stderr: {}", forced.stderr);
    let link = project.join(".claude/skills/meme");
    assert!(
        fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink()
    );

    // A relink with an already-correct target succeeds without
    // complaining (upstream idempotent branch).
    let again = env.run_at(&["skill", "install", "-f", "--yes"], &project);
    assert_eq!(again.exit_code, 0, "stderr: {}", again.stderr);
    assert!(
        again.stdout.contains("✓ Linked Claude skill at"),
        "{}",
        again.stdout
    );
}

/// `MEME_SKILLS_DIR` replaces discovery wholesale: fixture skills list,
/// a `hidden:` skill is gettable by name but filtered from `--all`,
/// and `skills path` echoes the env value.
#[test]
fn skills_dir_override() {
    let env = CliEnv::new();
    let skills_dir = env.root.join("fskills");
    fs::create_dir_all(skills_dir.join("custom")).unwrap();
    fs::write(
        skills_dir.join("custom/SKILL.md"),
        "---\nname: custom\ndescription: Fixture skill.\n---\n\n# Custom\n",
    )
    .unwrap();
    fs::create_dir_all(skills_dir.join("hidden")).unwrap();
    fs::write(
        skills_dir.join("hidden/SKILL.md"),
        "---\nname: hidden\ndescription: Not listed.\nhidden: true\n---\n\n# Hidden\n",
    )
    .unwrap();

    let extra = [("MEME_SKILLS_DIR", skills_dir.to_str().unwrap())];
    let out = env.run_opts(&["skills", "list"], &env.fixtures, false, &extra);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("custom"));
    assert!(!out.stdout.contains("hidden"), "{}", out.stdout);

    // Hidden skills are gettable by name (upstream quirk — `findSkill`
    // ignores the flag).
    let out = env.run_opts(&["skills", "get", "hidden"], &env.fixtures, false, &extra);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("# Hidden"), "{}", out.stdout);

    // `get --all` honors the filter.
    let out = env.run_opts(&["skills", "get", "--all"], &env.fixtures, false, &extra);
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("# Custom"));
    assert!(!out.stdout.contains("# Hidden"), "{}", out.stdout);

    // `skills path` echoes the env value verbatim.
    let out = env.run_opts(&["skills", "path"], &env.fixtures, false, &extra);
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.trim(), skills_dir.to_str().unwrap());
}

/// `skill install` honors `MEME_SKILLS_DIR` — the fixture `meme` tree
/// replaces the embedded one (still ending in the stub SKILL.md).
#[test]
fn skill_install_from_skills_dir() {
    let env = CliEnv::new();
    let skills_dir = env.root.join("fskills");
    fs::create_dir_all(skills_dir.join("meme/references")).unwrap();
    fs::write(
        skills_dir.join("meme/SKILL.md"),
        "---\nname: meme\ndescription: Fixture meme.\n---\n\n# Fixture MEME\n",
    )
    .unwrap();
    fs::write(skills_dir.join("meme/references/ext.md"), "# Ext\n").unwrap();

    let project = env.root.join("project-dir");
    fs::create_dir_all(&project).unwrap();
    let out = env.run_opts(
        &["skill", "install"],
        &project,
        false,
        &[("MEME_SKILLS_DIR", skills_dir.to_str().unwrap())],
    );
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let skill_dir = project.join(".agents/skills/meme");
    assert!(
        fs::read_to_string(skill_dir.join("SKILL.md"))
            .unwrap()
            .contains("!`meme skill show`")
    );
    assert!(skill_dir.join("references/ext.md").exists());
    assert!(
        !skill_dir.join("references/mcp-setup.md").exists(),
        "fixture tree replaces the embedded one"
    );
}

/// The interactive prompt — `run` in-process with TTY flags and an
/// injected stdin answer (`y`, `n`, EOF) like `confirmOnTty`.
#[test]
fn skill_install_tty_prompt() {
    for (answer, expect_link) in [("y\n", true), ("n\n", false), ("", false)] {
        let env = CliEnv::new();
        let project = env.root.join("tty-project");
        fs::create_dir_all(&project).unwrap();
        let mut meme_env = meme::Environment::default();
        meme_env.pwd = Some(project.clone());
        meme_env.home = Some(env.home.clone());
        meme_env.xdg_cache_home = Some(env.root.join("xdg-cache"));
        meme_env.meme_config_dir = Some(env.config_dir);
        let mut stdin: &[u8] = answer.as_bytes();
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = {
            let mut io = meme_bin::Io {
                out: &mut out,
                err: &mut err,
                stdin: &mut stdin,
                stdout_tty: true,
                stderr_tty: false,
                stdin_tty: true,
                local_offset: time::UtcOffset::UTC,
            };
            meme_bin::run(
                ["skill", "install"].map(Into::into),
                meme_env,
                &project,
                &mut io,
            )
        };
        assert_eq!(
            code,
            ExitCode::SUCCESS,
            "stderr: {}",
            String::from_utf8_lossy(&err)
        );
        let stdout = String::from_utf8_lossy(&out);
        let link = project.join(".claude/skills/meme");
        assert_eq!(
            link.symlink_metadata()
                .is_ok_and(|m| m.file_type().is_symlink()),
            expect_link,
            "answer {answer:?}: {stdout}"
        );
        if expect_link {
            assert!(
                stdout.contains("Create a symlink in"),
                "prompt shown: {stdout}"
            );
            assert!(stdout.contains("✓ Linked Claude skill at"), "{stdout}");
        }
    }
}

/// `skill install` writes the stub first — a busy `.claude` path only
/// fails afterward (upstream error ordering, qmd.ts:3617-3633).
#[test]
fn skill_install_blocked_link_path() {
    let env = CliEnv::new();
    let project = env.root.join("project-blocked");
    fs::create_dir_all(project.join(".claude/skills")).unwrap();
    fs::write(project.join(".claude/skills/meme"), "blocker\n").unwrap();
    let out = env.run_at(&["skill", "install", "--yes"], &project);
    assert_eq!(out.exit_code, 1);
    assert!(
        out.stdout.contains("✓ Installed MEME skill"),
        "{}",
        out.stdout
    );
    assert!(
        out.stderr.contains("Claude skill path already exists"),
        "{}",
        out.stderr
    );
    // `-f` clears the blocking file and links.
    let forced = env.run_at(&["skill", "install", "-f", "--yes"], &project);
    assert_eq!(forced.exit_code, 0, "stderr: {}", forced.stderr);
    assert!(
        fs::symlink_metadata(project.join(".claude/skills/meme"))
            .unwrap()
            .file_type()
            .is_symlink()
    );
}
