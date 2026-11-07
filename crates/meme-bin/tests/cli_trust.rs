//! Ports of upstream `test/local-config-trust-cli.test.ts` (tobi/qmd#889) and
//! `test/update-hook-trust-cli.test.ts` (tobi/qmd#886), plus the CLI test of
//! `test/local-config.test.ts`.
//!
//! Each fixture plays a freshly cloned repository shipping its own
//! `.meme/index.yml` under `env.root/proj`; `MEME_CONFIG_DIR` (the trust
//! store's home) stays outside the project like upstream's separate
//! `configDir` tempdir. `INDEX_PATH` is left unset so the project-local
//! `.meme/index.sqlite` is used — the Rust counterpart of upstream's
//! `.meme/index.sqlite` (renamed so Rust and Bun indexes do not
//! collide).

#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::missing_assert_message,
    clippy::panic,
    reason = "idiomatic test-code patterns"
)]

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use common::{CliEnv, Out};

/// A checked-out project: `proj/.meme/`, `proj/docs/readme.md` and an
/// `outside/` directory the checked-in config may point at.
struct ProjEnv {
    env: CliEnv,
    proj: PathBuf,
    outside: PathBuf,
}

impl ProjEnv {
    fn new() -> Self {
        let env = CliEnv::new();
        let proj = env.root.join("proj");
        let outside = env.root.join("outside");
        fs::create_dir_all(proj.join(".meme")).unwrap();
        fs::create_dir_all(proj.join("docs")).unwrap();
        fs::write(
            proj.join("docs/readme.md"),
            "# Readme\n\nSome indexable content.\n",
        )
        .unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(
            outside.join("secret.md"),
            "# Secret\n\nShould not be indexed unattended.\n",
        )
        .unwrap();
        Self { env, proj, outside }
    }

    /// Write `.meme/index.yml` inside the project.
    fn write_local_config(&self, body: &str) {
        fs::write(self.proj.join(".meme/index.yml"), body).unwrap();
    }

    /// `runMeme(args)` — cwd inside the project, no `INDEX_PATH`.
    fn run(&self, args: &[&str]) -> Out {
        self.env.run_opts(args, &self.proj, false, &[])
    }

    fn run_with(&self, args: &[&str], extra: &[(&str, &str)]) -> Out {
        self.env.run_opts(args, &self.proj, false, extra)
    }

    fn config_path(&self) -> PathBuf {
        self.proj.join(".meme/index.yml")
    }

    fn marker(&self) -> PathBuf {
        self.proj.join("docs/hook-ran.txt")
    }
}

/// Upstream `outsideConfig()` — the `secrets` collection points out of
/// the project.
fn outside_config(dir: &Path) -> String {
    format!(
        "collections:\n  secrets:\n    path: \"{}\"\n    pattern: \"**/*.md\"\n",
        dir.display()
    )
}

// ── checked-in collection path outside the project ─────────────────

#[test]
fn update_does_not_index_the_outside_path_unattended() {
    let p = ProjEnv::new();
    p.write_local_config(&outside_config(&p.outside));
    let out = p.run(&["update"]);

    assert!(
        out.stdout.contains("Collection paths outside this project"),
        "{}",
        out.stdout
    );
    assert!(out.stdout.contains("meme trust"), "{}", out.stdout);
    assert!(
        out.stdout.contains("Skipping collection 'secrets'"),
        "{}",
        out.stdout
    );
    assert!(!out.stdout.contains("Indexed: 1 new"), "{}", out.stdout);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
}

#[test]
fn update_indexes_the_outside_path_after_meme_trust() {
    let p = ProjEnv::new();
    p.write_local_config(&outside_config(&p.outside));

    let trust = p.run(&["trust"]);
    assert!(trust.stdout.contains("Trusted"), "{}", trust.stdout);
    assert!(
        trust
            .stdout
            .contains(&p.outside.to_string_lossy().as_ref().to_owned()),
        "{}",
        trust.stdout
    );

    let out = p.run(&["update"]);
    assert!(out.stdout.contains("Indexed: 1 new"), "{}", out.stdout);
    assert!(
        !out.stdout.contains("Skipping collection 'secrets'"),
        "{}",
        out.stdout
    );
}

#[test]
fn update_re_arms_the_gate_when_the_path_is_rewritten_after_approval() {
    let p = ProjEnv::new();
    p.write_local_config(&outside_config(&p.outside));
    let trust = p.run(&["trust"]);
    assert!(trust.stdout.contains("Trusted"), "{}", trust.stdout);

    let other = p.env.root.join("other");
    fs::create_dir_all(&other).unwrap();
    fs::write(other.join("other.md"), "# Other\n\nAlso outside.\n").unwrap();
    p.write_local_config(&outside_config(&other));

    let out = p.run(&["update"]);
    assert!(
        out.stdout.contains("Skipping collection 'secrets'"),
        "{}",
        out.stdout
    );
    assert!(!out.stdout.contains("Indexed: 1 new"), "{}", out.stdout);
}

#[test]
fn meme_trust_local_config_env_opts_unattended_runs_back_in() {
    let p = ProjEnv::new();
    p.write_local_config(&outside_config(&p.outside));

    let out = p.run_with(&["update"], &[("MEME_TRUST_LOCAL_CONFIG", "1")]);
    assert!(out.stdout.contains("Indexed: 1 new"), "{}", out.stdout);
    assert!(
        !out.stdout.contains("Skipping collection 'secrets'"),
        "{}",
        out.stdout
    );
}

#[test]
fn meme_trust_list_reports_the_approved_config() {
    let p = ProjEnv::new();
    p.write_local_config(&outside_config(&p.outside));
    let trust = p.run(&["trust"]);
    assert!(trust.stdout.contains("Trusted"), "{}", trust.stdout);

    let list = p.run(&["trust", "list"]);
    assert!(
        list.stdout
            .contains(&p.config_path().to_string_lossy().as_ref().to_owned()),
        "{}",
        list.stdout
    );
}

/// `status` reports the declared config; it never prompts for trust.
#[test]
fn status_is_not_prompted_to_trust_local_config() {
    let p = ProjEnv::new();
    p.write_local_config(&outside_config(&p.outside));

    let out = p.run(&["status"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("secrets"), "{}", out.stdout);
    let combined = format!("{}{}", out.stdout, out.stderr);
    assert!(!combined.contains("meme trust"), "{combined}");
}

#[test]
fn trusted_json_records_the_config_digest_and_timestamp() {
    let p = ProjEnv::new();
    p.write_local_config(&outside_config(&p.outside));
    let trust = p.run(&["trust"]);
    assert!(trust.stdout.contains("Trusted"), "{}", trust.stdout);

    let raw = fs::read_to_string(p.env.config_dir.join("trusted.json")).unwrap();
    let records: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let key = p.config_path().to_string_lossy().into_owned();
    let record = records.get(&key).unwrap_or_else(|| panic!("{raw}"));
    let hooks = record["hooks"].as_str().unwrap();
    let hex = regex::Regex::new(r"^[a-f0-9]{64}$").unwrap();
    assert!(hex.is_match(hooks), "{hooks}");
    assert!(record["trustedAt"].as_str().is_some(), "{record}");
}

// ── checked-in custom model URI ─────────────────────────────────────

/// `docs` inside the project plus a non-default `embed` model.
fn custom_model_config() -> String {
    "collections:\n  docs:\n    path: ./docs\n    pattern: \"**/*.md\"\nmodels:\n  embed: hf:evil/embed/x.gguf\n"
        .to_owned()
}

#[test]
fn custom_model_is_gated_but_the_project_still_indexes() {
    let p = ProjEnv::new();
    p.write_local_config(&custom_model_config());

    let out = p.run(&["update"]);
    assert!(out.stdout.contains("Custom models"), "{}", out.stdout);
    assert!(
        out.stdout.contains("hf:evil/embed/x.gguf"),
        "{}",
        out.stdout
    );
    assert!(out.stdout.contains("Indexed: 1 new"), "{}", out.stdout);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
}

#[test]
fn meme_trust_records_the_custom_model() {
    let p = ProjEnv::new();
    p.write_local_config(&custom_model_config());

    let trust = p.run(&["trust"]);
    assert!(
        trust.stdout.contains("hf:evil/embed/x.gguf"),
        "{}",
        trust.stdout
    );
    assert!(trust.stdout.contains("Trusted"), "{}", trust.stdout);

    let out = p.run(&["update"]);
    assert!(!out.stdout.contains("Custom models"), "{}", out.stdout);
    assert!(out.stdout.contains("Indexed: 1 new"), "{}", out.stdout);
}

// ── only in-project paths ───────────────────────────────────────────

#[test]
fn in_project_paths_index_without_a_trust_prompt() {
    let p = ProjEnv::new();
    p.write_local_config("collections:\n  docs:\n    path: ./docs\n    pattern: \"**/*.md\"\n");

    let out = p.run(&["update"]);
    assert!(out.stdout.contains("Indexed: 1 new"), "{}", out.stdout);
    assert!(!out.stdout.contains("meme trust"), "{}", out.stdout);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
}

// ── checked-in update: hook ─────────────────────────────────────────

/// Upstream `writeLocalConfig(command)`: `docs` in-project with an
/// `update:` hook that writes a marker inside the collection.
fn hook_config(command: &str) -> String {
    format!(
        "collections:\n  docs:\n    path: ./docs\n    pattern: \"**/*.md\"\n    update: {command:?}\n"
    )
}

#[test]
fn update_does_not_run_the_repos_hook_unattended_but_still_indexes() {
    let p = ProjEnv::new();
    p.write_local_config(&hook_config("echo ran > hook-ran.txt"));

    let out = p.run(&["update"]);
    assert!(!p.marker().exists());
    assert!(
        out.stdout.contains("defines update commands"),
        "{}",
        out.stdout
    );
    assert!(out.stdout.contains("meme trust"), "{}", out.stdout);
    assert!(out.stdout.contains("Indexed: 1 new"), "{}", out.stdout);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
}

#[test]
fn update_runs_the_hook_after_meme_trust() {
    let p = ProjEnv::new();
    p.write_local_config(&hook_config("echo ran > hook-ran.txt"));

    let trust = p.run(&["trust"]);
    assert!(trust.stdout.contains("Trusted"), "{}", trust.stdout);

    let out = p.run(&["update"]);
    assert!(
        out.stdout.contains("Running update command"),
        "{}",
        out.stdout
    );
    assert!(p.marker().exists());
}

#[test]
fn update_re_arms_the_gate_when_the_hook_changes_after_approval() {
    let p = ProjEnv::new();
    p.write_local_config(&hook_config("echo ran > hook-ran.txt"));
    let trust = p.run(&["trust"]);
    assert!(trust.stdout.contains("Trusted"), "{}", trust.stdout);

    // Stand-in for a `git pull` rewriting the hook under an approval.
    p.write_local_config(&hook_config("echo pwned > hook-ran.txt && echo extra"));

    let out = p.run(&["update"]);
    assert!(!p.marker().exists());
    assert!(
        out.stdout.contains("defines update commands"),
        "{}",
        out.stdout
    );
}

#[test]
fn meme_trust_revoke_puts_the_gate_back() {
    let p = ProjEnv::new();
    p.write_local_config(&hook_config("echo ran > hook-ran.txt"));
    let trust = p.run(&["trust"]);
    assert!(trust.stdout.contains("Trusted"), "{}", trust.stdout);

    let revoke = p.run(&["trust", "revoke"]);
    assert!(revoke.stdout.contains("Revoked trust"), "{}", revoke.stdout);

    p.run(&["update"]);
    assert!(!p.marker().exists());
}

#[test]
fn meme_trust_update_hooks_env_opts_unattended_runs_back_in() {
    let p = ProjEnv::new();
    p.write_local_config(&hook_config("echo ran > hook-ran.txt"));

    let out = p.run_with(&["update"], &[("MEME_TRUST_UPDATE_HOOKS", "1")]);
    assert!(
        out.stdout.contains("Running update command"),
        "{}",
        out.stdout
    );
    assert!(p.marker().exists());
}

// ── trust laundering (deviation 85, upstream-quirks.md Q39) ─────────

#[test]
fn collection_add_does_not_launder_a_checked_in_hook() {
    // A checked-in `.meme/index.yml` carrying a gated `update:` hook:
    // the user's unrelated `collection add` must not re-record trust,
    // or the hook would run on the next `meme update`.
    let p = ProjEnv::new();
    p.write_local_config(&hook_config("echo ran > hook-ran.txt"));
    fs::create_dir_all(p.proj.join("notes")).unwrap();
    fs::write(p.proj.join("notes/n.md"), "# Notes\n").unwrap();

    let add = p.run(&["collection", "add", "./notes", "--name", "notes"]);
    assert_eq!(add.exit_code, 0, "stderr: {}", add.stderr);

    let out = p.run(&["update"]);
    assert!(!p.marker().exists());
    assert!(
        out.stdout.contains("defines update commands"),
        "{}",
        out.stdout
    );
}

#[test]
fn update_cmd_does_not_launder_other_checked_in_hooks() {
    // The edited collection (`notes`) has no hook; `docs`'s checked-in
    // hook must stay gated through the user's `update-cmd`.
    let p = ProjEnv::new();
    fs::create_dir_all(p.proj.join("notes")).unwrap();
    fs::write(p.proj.join("notes/n.md"), "# Notes\n").unwrap();
    p.write_local_config(
        "collections:\n  docs:\n    path: ./docs\n    pattern: \"**/*.md\"\n    update: \"echo ran > hook-ran.txt\"\n  notes:\n    path: ./notes\n    pattern: \"**/*.md\"\n",
    );

    let set = p.run(&["collection", "update-cmd", "notes", "echo mine"]);
    assert_eq!(set.exit_code, 0, "stderr: {}", set.stderr);

    let out = p.run(&["update"]);
    assert!(!p.marker().exists());
    assert!(
        out.stdout.contains("defines update commands"),
        "{}",
        out.stdout
    );
}

#[test]
fn update_cmd_on_a_trusted_config_stays_trusted() {
    // Keeps the tobi/qmd#886 intent: a config already trusted re-records
    // its digest across the user's own hook edit.
    let p = ProjEnv::new();
    p.write_local_config(&hook_config("echo ran > hook-ran.txt"));
    let trust = p.run(&["trust"]);
    assert!(trust.stdout.contains("Trusted"), "{}", trust.stdout);

    let set = p.run(&[
        "collection",
        "update-cmd",
        "docs",
        "echo again > hook-ran.txt",
    ]);
    assert_eq!(set.exit_code, 0, "stderr: {}", set.stderr);

    let out = p.run(&["update"]);
    assert!(
        out.stdout.contains("Running update command"),
        "{}",
        out.stdout
    );
    assert!(p.marker().exists());
}

#[test]
fn collection_add_in_an_ungated_project_still_auto_trusts() {
    // Keeps the tobi/qmd#889 intent: adding an out-of-project path the
    // user just typed to a config with no gated items re-records
    // trust, so `update` indexes it without warnings.
    let p = ProjEnv::new();
    p.write_local_config("collections:\n  docs:\n    path: ./docs\n    pattern: \"**/*.md\"\n");

    let add = p.run(&[
        "collection",
        "add",
        &p.outside.to_string_lossy(),
        "--name",
        "outside",
    ]);
    assert_eq!(add.exit_code, 0, "stderr: {}", add.stderr);

    let out = p.run(&["update"]);
    assert!(
        !out.stdout.contains("Collection paths outside this project"),
        "{}",
        out.stdout
    );
    assert!(
        !out.stdout.contains("Skipping collection 'outside'"),
        "{}",
        out.stdout
    );
}

// ── local .meme project config (local-config.test.ts) ───────────────

/// The CLI half of upstream `local-config.test.ts`: the local config
/// supplies its own index, collections, contexts and declared models.
#[test]
fn status_uses_local_meme_config_and_index_instead_of_global_cache() {
    let p = ProjEnv::new();
    fs::write(
        p.proj.join(".meme/index.yaml"),
        format!(
            "collections:\n  docs:\n    path: \"{}\"\n    pattern: \"**/*.md\"\n    context:\n      /: Local test docs\nmodels:\n  embed: local-embed-model\n  rerank: local-rerank-model\n  generate: local-generate-model\n",
            p.proj.join("docs").display()
        ),
    )
    .unwrap();

    let out = p.run_with(
        &["status"],
        &[
            ("MEME_EMBED_MODEL", "env-embed-model"),
            ("MEME_RERANK_MODEL", "env-rerank-model"),
            ("MEME_GENERATE_MODEL", "env-generate-model"),
        ],
    );
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);

    let local_index = p.proj.join(".meme/index.sqlite");
    assert!(
        out.stdout
            .contains(&format!("Index: {}", local_index.display())),
        "{}",
        out.stdout
    );
    assert!(out.stdout.contains("docs (meme://docs/)"), "{}", out.stdout);
    assert!(
        out.stdout.contains("Embedding:   local-embed-model"),
        "{}",
        out.stdout
    );
    assert!(
        out.stdout.contains("Reranking:   local-rerank-model"),
        "{}",
        out.stdout
    );
    assert!(
        out.stdout.contains("Generation:  local-generate-model"),
        "{}",
        out.stdout
    );
    assert!(!out.stdout.contains("env-embed-model"), "{}", out.stdout);
    assert!(local_index.exists());
    assert!(!p.env.root.join("xdg-cache/meme/index.sqlite").exists());
    assert!(!p.env.root.join("xdg-cache/meme/index.sqlite").exists());
}

// ── lazy store opening (upstream manageTrust / update-cmd) ──────────

/// `trust`, `manageTrust` and the YAML-only collection mutations never
/// reach `getStore`/`getDb` — the project index must not appear. The
/// `update:` hook the test writes would be gated on the next `update`;
/// `update-cmd` itself re-records trust without touching the db
/// (qmd.ts:4576-4601).
#[test]
fn trust_and_yaml_only_commands_do_not_create_the_project_index() {
    let p = ProjEnv::new();
    p.write_local_config(&outside_config(&p.outside));
    let local_index = p.proj.join(".meme/index.sqlite");

    for args in [
        ["trust"].as_slice(),
        ["trust", "list"].as_slice(),
        // `update-cmd` mutates `.meme/index.yml` in place — still YAML.
        ["collection", "update-cmd", "secrets", "echo hi"].as_slice(),
        ["collection", "show", "secrets"].as_slice(),
        ["context", "rm", "meme://secrets/nope"].as_slice(),
    ] {
        let out = p.run(args);
        assert_ne!(out.exit_code, -1, "{args:?} died on a signal");
        assert!(
            !local_index.exists(),
            "{args:?} created {}",
            local_index.display()
        );
    }

    // `trust revoke` removes the record; still no database.
    let out = p.run(&["trust", "revoke"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(!local_index.exists());
}
