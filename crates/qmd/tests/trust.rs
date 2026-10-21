//! Trust gate for project-local `.qmd` config and the update-hook
//! execution primitive — a port of upstream
//! `test/update-hook-trust.test.ts` — plus the `trust_golden.json`
//! fixture generated from upstream by `scripts/gen_trust_golden.ts`.
//!
//! `hookDigest` and `decideHookGate` are upstream dead code and not
//! ported (P3-D3): their 3 + 12 cases (the `decideHookGate` describe,
//! counting the `test.each` expansion) are recorded N/A. The falsy-env
//! coverage those cases provided is carried by
//! [`falsy_env_values_do_not_opt_in`].

#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_assert_message,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::shadow_unrelated,
    reason = "idiomatic test-code patterns"
)]

use std::path::{Path, PathBuf};

use qmd::Environment;
use qmd::trust::{
    BuiltinModels, CollectionPath, GateAction, LocalConfigGateInput, ModelsSnapshot,
    SensitiveSnapshot, TrustedCheck, UpdateHook, decide_local_config_gate, expand_user_path,
    gated_items, is_collection_path_inside_project, is_local_config_path,
    is_local_config_trust_opted_in, is_trusted, list_trusted, load_trust_store,
    project_root_from_config, record_trust, resolve_config_collection_path, revoke_trust,
    sensitive_digest, trust_file_path,
};
use qmd::update_hook::run_update_hook;

// =============================================================================
// Helpers — upstream `localConfigPath`, `snapshot`, `builtins`, `noEnv`
// =============================================================================

/// Environment pinned to a throwaway config dir + the golden HOME.
fn env_with(config_dir: &Path) -> Environment {
    let mut env = Environment::default();
    env.qmd_config_dir = Some(config_dir.to_path_buf());
    env.home = Some(PathBuf::from("/home/qmd-golden"));
    env.pwd = Some(PathBuf::from("/"));
    env
}

/// Environment for the path-helper tests — only `home` matters; `pwd`
/// stands in for `process.cwd()` (upstream tests pass absolute paths).
fn path_env() -> Environment {
    let mut env = Environment::default();
    env.home = Some(PathBuf::from("/home/qmd-golden"));
    env.pwd = Some(PathBuf::from("/"));
    env
}

/// A `.qmd/index.yml` path inside a throwaway project directory —
/// upstream `localConfigPath` (the project root exists so `realpath`
/// succeeds).
fn local_config_path() -> (tempfile::TempDir, String) {
    let project = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project.path().join(".qmd")).unwrap();
    let config = project.path().join(".qmd").join("index.yml");
    (project, config.to_string_lossy().into_owned())
}

const PROJECT: &str = "/home/me/project/.qmd/index.yml";
const GLOBAL: &str = "/home/me/.config/qmd/index.yml";

fn builtins() -> BuiltinModels {
    BuiltinModels {
        embed: "hf:default/embed/model.gguf".to_owned(),
        rerank: "hf:default/rerank/model.gguf".to_owned(),
        generate: "hf:default/generate/model.gguf".to_owned(),
    }
}

fn snapshot() -> SensitiveSnapshot {
    SensitiveSnapshot {
        hooks: vec![],
        paths: vec![CollectionPath {
            collection: "docs".to_owned(),
            path: "./docs".to_owned(),
        }],
        models: ModelsSnapshot::default(),
    }
}

fn outside_snapshot() -> SensitiveSnapshot {
    SensitiveSnapshot {
        hooks: vec![],
        paths: vec![CollectionPath {
            collection: "secrets".to_owned(),
            path: "/etc".to_owned(),
        }],
        models: ModelsSnapshot::default(),
    }
}

fn custom_model_snapshot() -> SensitiveSnapshot {
    SensitiveSnapshot {
        hooks: vec![],
        paths: vec![],
        models: ModelsSnapshot {
            embed: Some("hf:evil/embed/x.gguf".to_owned()),
            ..ModelsSnapshot::default()
        },
    }
}

// =============================================================================
// `describe("isLocalConfigPath")` — 3 cases
// =============================================================================

#[test]
fn recognizes_a_project_local_config() {
    let env = path_env();
    assert!(is_local_config_path(PROJECT, &env));
    assert!(is_local_config_path(
        "/home/me/project/.qmd/index.yaml",
        &env
    ));
}

#[test]
fn does_not_treat_the_users_own_config_as_project_local() {
    let env = path_env();
    assert!(!is_local_config_path(
        "/home/me/.config/qmd/index.yml",
        &env
    ));
    assert!(!is_local_config_path("/home/me/.config/qmd/work.yml", &env));
}

#[test]
fn handles_the_sdk_inline_sentinel_and_empty_paths() {
    let env = path_env();
    assert!(!is_local_config_path("<inline>", &env));
    assert!(!is_local_config_path("", &env));
}

// =============================================================================
// `describe("trust store")` — 5 cases (digests are opaque strings here;
// upstream feeds `hookDigest` output, the store never interprets it)
// =============================================================================

#[test]
fn records_reads_back_and_revokes() {
    let cfg = tempfile::tempdir().unwrap();
    let env = env_with(cfg.path());
    let (_project, config_path) = local_config_path();

    assert!(!is_trusted(&env, &config_path, "digest-a"));
    record_trust(&env, &config_path, "digest-a").unwrap();
    assert!(is_trusted(&env, &config_path, "digest-a"));
    assert!(list_trusted(&env).iter().any(|r| r.path == config_path));

    assert!(revoke_trust(&env, &config_path).unwrap());
    assert!(!is_trusted(&env, &config_path, "digest-a"));
    assert!(!revoke_trust(&env, &config_path).unwrap());
}

#[test]
fn an_approval_does_not_cover_a_different_hook_set() {
    let cfg = tempfile::tempdir().unwrap();
    let env = env_with(cfg.path());
    let (_project, config_path) = local_config_path();
    record_trust(&env, &config_path, "digest-a").unwrap();
    assert!(!is_trusted(&env, &config_path, "digest-b"));
}

#[test]
fn an_approval_does_not_carry_to_another_project() {
    let cfg = tempfile::tempdir().unwrap();
    let env = env_with(cfg.path());
    let (_p1, first) = local_config_path();
    let (_p2, second) = local_config_path();
    record_trust(&env, &first, "digest-a").unwrap();
    assert!(!is_trusted(&env, &second, "digest-a"));
}

#[test]
fn a_corrupt_trust_file_is_not_read_as_blanket_trust() {
    let cfg = tempfile::tempdir().unwrap();
    let env = env_with(cfg.path());
    std::fs::create_dir_all(cfg.path()).unwrap();
    std::fs::write(trust_file_path(&env), "{ not json").unwrap();
    assert!(load_trust_store(&env).is_empty());
    let (_project, config_path) = local_config_path();
    assert!(!is_trusted(&env, &config_path, "digest-a"));
}

#[test]
fn writes_the_trust_file_into_the_config_directory() {
    let cfg = tempfile::tempdir().unwrap();
    let env = env_with(cfg.path());
    let (_project, config_path) = local_config_path();
    record_trust(&env, &config_path, "digest-a").unwrap();
    let written = cfg.path().join("trusted.json");
    assert!(written.exists());
    let parsed: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(written).unwrap()).unwrap();
    assert!(parsed.is_object());
    // Record key order must be `hooks` then `trustedAt` — upstream's
    // literal `{ hooks: digest, trustedAt }` order (preserve_order on).
    let keys: Vec<&str> = parsed[&config_path]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, ["hooks", "trustedAt"]);
    // `trustedAt` is `new Date().toISOString()` — ms precision + `Z`.
    let ts = parsed[&config_path]["trustedAt"].as_str().unwrap();
    assert!(ts.ends_with('Z') && ts.len() == 24, "{ts}");
}

// =============================================================================
// `describe("isCollectionPathInsideProject")` — 3 cases
// =============================================================================

#[test]
fn allows_relative_paths_inside_the_project() {
    let env = path_env();
    assert!(is_collection_path_inside_project(PROJECT, "./docs", &env));
    assert!(is_collection_path_inside_project(PROJECT, ".", &env));
    assert!(is_collection_path_inside_project(
        PROJECT,
        "docs/notes",
        &env
    ));
}

#[test]
fn rejects_paths_that_leave_the_project() {
    let env = path_env();
    assert!(!is_collection_path_inside_project(PROJECT, "..", &env));
    assert!(!is_collection_path_inside_project(
        PROJECT, "../other", &env
    ));
    assert!(!is_collection_path_inside_project(PROJECT, "/etc", &env));
    assert!(!is_collection_path_inside_project(
        PROJECT,
        "/home/me/secrets",
        &env
    ));
}

#[test]
fn expands_tilde_to_the_home_directory_which_is_outside_the_project() {
    let env = path_env();
    assert!(!is_collection_path_inside_project(
        PROJECT,
        "~/secrets",
        &env
    ));
    assert_eq!(
        resolve_config_collection_path("~/secrets", PROJECT, &env),
        "/home/qmd-golden/secrets"
    );
}

/// `realOrResolve`'s `realpathSync` branch — a symlinked collection path
/// resolves to its real target before the `..` check (upstream relies on
/// `realpathSync`; all golden cases take the nonexistent-path lexical
/// fallback, so this exercises the canonicalize arm).
#[cfg(unix)]
#[test]
fn follows_symlinks_before_the_escape_check() {
    use std::os::unix::fs::symlink;

    let env = path_env();
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let project = root.path().join("project");
    std::fs::create_dir_all(project.join(".qmd")).unwrap();
    std::fs::create_dir_all(project.join("real")).unwrap();
    let config = project.join(".qmd").join("index.yml");
    let config = config.to_string_lossy().into_owned();

    // A symlink inside the project pointing outside counts as leaving.
    symlink(outside.path(), project.join("escape")).unwrap();
    assert!(!is_collection_path_inside_project(&config, "escape", &env));
    // A symlink pointing deeper inside the project is allowed.
    symlink(project.join("real"), project.join("alias")).unwrap();
    assert!(is_collection_path_inside_project(&config, "alias", &env));
    // `..` through a symlinked parent: `link/sub` still resolves outside.
    std::fs::create_dir_all(outside.path().join("sub")).unwrap();
    assert!(!is_collection_path_inside_project(
        &config,
        "escape/sub",
        &env
    ));
}

// =============================================================================
// `describe("sensitiveDigest")` — 4 cases
// =============================================================================

#[test]
fn digest_is_stable_across_collection_ordering() {
    let env = path_env();
    let mut a = snapshot();
    a.paths = vec![
        CollectionPath {
            collection: "docs".to_owned(),
            path: "./docs".to_owned(),
        },
        CollectionPath {
            collection: "notes".to_owned(),
            path: "./notes".to_owned(),
        },
    ];
    let mut b = snapshot();
    b.paths = vec![
        CollectionPath {
            collection: "notes".to_owned(),
            path: "./notes".to_owned(),
        },
        CollectionPath {
            collection: "docs".to_owned(),
            path: "./docs".to_owned(),
        },
    ];
    assert_eq!(
        sensitive_digest(&a, PROJECT, &builtins(), &env),
        sensitive_digest(&b, PROJECT, &builtins(), &env)
    );
}

#[test]
fn in_project_paths_and_default_models_do_not_change_the_digest() {
    let env = path_env();
    let empty = SensitiveSnapshot {
        hooks: vec![],
        paths: vec![],
        models: ModelsSnapshot::default(),
    };
    let b = builtins();
    let in_project = SensitiveSnapshot {
        hooks: vec![],
        paths: vec![CollectionPath {
            collection: "docs".to_owned(),
            path: "./docs".to_owned(),
        }],
        models: ModelsSnapshot {
            embed: Some(b.embed.clone()),
            rerank: Some(b.rerank.clone()),
            generate: Some(b.generate),
        },
    };
    assert_eq!(
        sensitive_digest(&empty, PROJECT, &builtins(), &env),
        sensitive_digest(&in_project, PROJECT, &builtins(), &env)
    );
}

#[test]
fn digest_changes_when_a_collection_path_leaves_the_project() {
    let env = path_env();
    assert_ne!(
        sensitive_digest(&snapshot(), PROJECT, &builtins(), &env),
        sensitive_digest(&outside_snapshot(), PROJECT, &builtins(), &env)
    );
}

#[test]
fn digest_changes_when_a_custom_model_uri_is_set() {
    let env = path_env();
    let defaults = SensitiveSnapshot {
        hooks: vec![],
        paths: vec![],
        models: ModelsSnapshot::default(),
    };
    assert_ne!(
        sensitive_digest(&defaults, PROJECT, &builtins(), &env),
        sensitive_digest(&custom_model_snapshot(), PROJECT, &builtins(), &env)
    );
}

// =============================================================================
// `describe("decideLocalConfigGate")` — 9 cases
// =============================================================================

fn gate<'a>(
    config_path: &'a str,
    snap: &'a SensitiveSnapshot,
    interactive: bool,
    env: &Environment,
    trusted: Option<&'a TrustedCheck<'a>>,
) -> GateAction {
    decide_local_config_gate(
        &LocalConfigGateInput {
            config_path,
            snapshot: snap,
            builtins: &builtins(),
            is_interactive: interactive,
        },
        env,
        trusted,
    )
    .action
}

const fn untrusted(_p: &str, _d: &str) -> bool {
    false
}

#[test]
fn in_project_paths_and_default_models_need_no_decision() {
    let env = path_env();
    assert_eq!(
        gate(PROJECT, &snapshot(), false, &env, Some(&untrusted)),
        GateAction::Run
    );
}

#[test]
fn the_users_own_config_always_runs_even_with_an_outside_path() {
    let env = path_env();
    assert_eq!(
        gate(GLOBAL, &outside_snapshot(), false, &env, Some(&untrusted)),
        GateAction::Run
    );
}

#[test]
fn an_untrusted_outside_path_prompts_on_a_terminal() {
    let env = path_env();
    assert_eq!(
        gate(PROJECT, &outside_snapshot(), true, &env, Some(&untrusted)),
        GateAction::Prompt
    );
}

#[test]
fn an_untrusted_outside_path_is_skipped_with_nobody_to_ask() {
    let env = path_env();
    assert_eq!(
        gate(PROJECT, &outside_snapshot(), false, &env, Some(&untrusted)),
        GateAction::Skip
    );
}

#[test]
fn an_untrusted_custom_model_is_skipped_with_nobody_to_ask() {
    let env = path_env();
    assert_eq!(
        gate(
            PROJECT,
            &custom_model_snapshot(),
            false,
            &env,
            Some(&untrusted)
        ),
        GateAction::Skip
    );
}

#[test]
fn a_previously_approved_snapshot_runs_unattended() {
    let env = path_env();
    let outside = outside_snapshot();
    let digest = sensitive_digest(&outside, PROJECT, &builtins(), &env);
    let approved = move |p: &str, d: &str| p == PROJECT && d == digest;
    assert_eq!(
        gate(PROJECT, &outside, false, &env, Some(&approved)),
        GateAction::Run
    );
}

#[test]
fn pointing_an_approved_collection_outside_the_project_re_arms_the_gate() {
    let env = path_env();
    let approved = sensitive_digest(&snapshot(), PROJECT, &builtins(), &env);
    let check = move |_p: &str, d: &str| d == approved;
    let escaped = SensitiveSnapshot {
        hooks: vec![],
        paths: vec![CollectionPath {
            collection: "docs".to_owned(),
            path: "/etc".to_owned(),
        }],
        models: ModelsSnapshot::default(),
    };
    assert_eq!(
        gate(PROJECT, &escaped, false, &env, Some(&check)),
        GateAction::Skip
    );
}

#[test]
fn qmd_trust_local_config_opts_unattended_runs_back_in() {
    let mut env = path_env();
    env.qmd_trust_local_config = Some("1".to_owned());
    assert_eq!(
        gate(PROJECT, &outside_snapshot(), false, &env, Some(&untrusted)),
        GateAction::Run
    );
}

#[test]
fn qmd_trust_update_hooks_also_opts_the_path_model_gate_in() {
    let mut env = path_env();
    env.qmd_trust_update_hooks = Some("1".to_owned());
    assert_eq!(
        gate(
            PROJECT,
            &custom_model_snapshot(),
            false,
            &env,
            Some(&untrusted)
        ),
        GateAction::Run
    );
}

/// `trusted_check: None` falls through to the real `trusted.json` store —
/// the wiring the P4 orchestration will use.
#[test]
fn an_approved_record_runs_via_the_real_store() {
    let cfg = tempfile::tempdir().unwrap();
    let env = env_with(cfg.path());
    let (_project, config_path) = local_config_path();
    let outside = outside_snapshot();
    let digest = sensitive_digest(&outside, &config_path, &builtins(), &env);

    // Untrusted → non-interactive skip.
    assert_eq!(
        gate(&config_path, &outside, false, &env, None),
        GateAction::Skip
    );
    // Approve against the real store, then the same gate runs.
    record_trust(&env, &config_path, &digest).unwrap();
    assert_eq!(
        gate(&config_path, &outside, false, &env, None),
        GateAction::Run
    );
    // Revoking re-arms the gate.
    revoke_trust(&env, &config_path).unwrap();
    assert_eq!(
        gate(&config_path, &outside, false, &env, None),
        GateAction::Skip
    );
}

/// The falsy-value table upstream pinned via `decideHookGate` (N/A)
/// lives on through `isLocalConfigTrustOptedIn` — the same `isTruthyEnv`.
#[test]
fn falsy_env_values_do_not_opt_in() {
    for value in ["0", "false", "off", "no", "none", ""] {
        let mut env = path_env();
        env.qmd_trust_local_config = Some(value.to_owned());
        env.qmd_trust_update_hooks = Some(value.to_owned());
        assert!(!is_local_config_trust_opted_in(&env), "{value:?}");
    }
    for value in ["1", "true", "yes", "on", "anything"] {
        let mut env = path_env();
        env.qmd_trust_local_config = Some(value.to_owned());
        assert!(is_local_config_trust_opted_in(&env), "{value:?}");
    }
    let env = path_env();
    assert!(!is_local_config_trust_opted_in(&env));
}

// =============================================================================
// Golden — replay every `trust_golden.json` case
// =============================================================================

fn snapshot_from_json(v: &serde_json::Value) -> SensitiveSnapshot {
    let mut snap = SensitiveSnapshot {
        hooks: v["hooks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| UpdateHook {
                collection: h["collection"].as_str().unwrap().to_owned(),
                command: h["command"].as_str().unwrap().to_owned(),
            })
            .collect(),
        paths: v["paths"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| CollectionPath {
                collection: p["collection"].as_str().unwrap().to_owned(),
                path: p["path"].as_str().unwrap().to_owned(),
            })
            .collect(),
        models: ModelsSnapshot::default(),
    };
    let m = &v["models"];
    snap.models.embed = m["embed"].as_str().map(str::to_owned);
    snap.models.rerank = m["rerank"].as_str().map(str::to_owned);
    snap.models.generate = m["generate"].as_str().map(str::to_owned);
    snap
}

fn builtins_from_json(v: &serde_json::Value) -> BuiltinModels {
    BuiltinModels {
        embed: v["embed"].as_str().unwrap().to_owned(),
        rerank: v["rerank"].as_str().unwrap().to_owned(),
        generate: v["generate"].as_str().unwrap().to_owned(),
    }
}

fn gated_to_json(g: &qmd::trust::GatedItems) -> serde_json::Value {
    serde_json::json!({
        "hooks": g.hooks.iter().map(|h| serde_json::json!({
            "collection": h.collection, "command": h.command,
        })).collect::<Vec<_>>(),
        "paths": g.paths.iter().map(|p| serde_json::json!({
            "collection": p.collection, "path": p.path,
        })).collect::<Vec<_>>(),
        "models": g.models.iter().map(|m| serde_json::json!({
            "slot": m.slot.as_str(), "uri": m.uri,
        })).collect::<Vec<_>>(),
    })
}

#[test]
fn golden_cases_match_upstream() {
    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("tests/fixtures/trust_golden.json").unwrap())
            .unwrap();
    let home = fixture["home"].as_str().unwrap();
    let mut env = Environment::default();
    env.home = Some(PathBuf::from(home));
    env.pwd = Some(PathBuf::from("/"));
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 49);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let kind = case["kind"].as_str().unwrap();
        let input = &case["input"];
        let expected = &case["output"];
        let actual: serde_json::Value = match kind {
            "localPath" => {
                serde_json::json!(is_local_config_path(input[0].as_str().unwrap(), &env))
            }
            "projectRoot" => {
                serde_json::json!(project_root_from_config(input[0].as_str().unwrap(), &env))
            }
            "expand" => {
                serde_json::json!(expand_user_path(input[0].as_str().unwrap(), &env))
            }
            "resolve" => serde_json::json!(resolve_config_collection_path(
                input[1].as_str().unwrap(),
                input[0].as_str().unwrap(),
                &env
            )),
            "inside" => serde_json::json!(is_collection_path_inside_project(
                input[0].as_str().unwrap(),
                input[1].as_str().unwrap(),
                &env
            )),
            "gated" => {
                let snap = snapshot_from_json(&input["snapshot"]);
                let b = builtins_from_json(&input["builtins"]);
                gated_to_json(&gated_items(
                    input["configPath"].as_str().unwrap(),
                    &snap,
                    &b,
                    &env,
                ))
            }
            "digest" => {
                let snap = snapshot_from_json(&input["snapshot"]);
                let b = builtins_from_json(&input["builtins"]);
                serde_json::json!(sensitive_digest(
                    &snap,
                    input["configPath"].as_str().unwrap(),
                    &b,
                    &env
                ))
            }
            "storeFormat" => continue, // exercised by trusted_json_write_format
            other => panic!("unknown case kind {other}"),
        };
        assert_eq!(&actual, expected, "case {name}");
    }
}

// =============================================================================
// `trusted.json` interop — bytes written by upstream load, and our write
// format is byte-identical (P3-D2 / AC5)
// =============================================================================

fn golden_store_bytes() -> String {
    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("tests/fixtures/trust_golden.json").unwrap())
            .unwrap();
    fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["kind"].as_str() == Some("storeFormat"))
        .unwrap()["output"]
        .as_str()
        .unwrap()
        .to_owned()
}

#[test]
fn reads_a_trust_file_written_by_upstream() {
    let cfg = tempfile::tempdir().unwrap();
    let env = env_with(cfg.path());
    std::fs::write(trust_file_path(&env), golden_store_bytes()).unwrap();

    // Records parse, insertion order survives, digests match.
    assert!(is_trusted(
        &env,
        "/home/me/project/.qmd/index.yml",
        "abcdef0123456789"
    ));
    assert!(!is_trusted(
        &env,
        "/home/me/project/.qmd/index.yml",
        "0011223344556677"
    ));
    let entries = list_trusted(&env);
    assert_eq!(
        entries.iter().map(|e| e.path.as_str()).collect::<Vec<_>>(),
        [
            // `resolve()` normalizes — same string for already-clean paths.
            "/home/me/project/.qmd/index.yml",
            "/home/me/other/.qmd/index.yaml",
        ]
    );
    assert_eq!(entries[0].hooks.as_deref(), Some("abcdef0123456789"));
    assert_eq!(
        entries[0].trusted_at.as_deref(),
        Some("2026-01-02T03:04:05.006Z")
    );
}

#[test]
fn writes_trust_file_bytes_identical_to_upstream() {
    // Build the same store object upstream's golden covers and compare
    // `JSON.stringify(store, null, 2) + "\n"` to `to_string_pretty`.
    let mut store = qmd::trust::TrustStore::new();
    store.insert(
        "/home/me/project/.qmd/index.yml".to_owned(),
        serde_json::json!({"hooks": "abcdef0123456789", "trustedAt": "2026-01-02T03:04:05.006Z"}),
    );
    store.insert(
        "/home/me/other/.qmd/index.yaml".to_owned(),
        serde_json::json!({"hooks": "0011223344556677", "trustedAt": "2026-02-03T04:05:06.007Z"}),
    );
    let ours = serde_json::to_string_pretty(&store).unwrap() + "\n";
    assert_eq!(ours, golden_store_bytes());
}

// =============================================================================
// Update-hook execution — upstream `qmd.ts:947-979` primitive
// =============================================================================

#[test]
fn hook_captures_stdout_stderr_and_success_code() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_update_hook("echo out; echo err >&2", dir.path()).unwrap();
    assert_eq!(out.stdout, "out\n");
    assert_eq!(out.stderr, "err\n");
    assert_eq!(out.code, 0);
}

#[test]
fn hook_runs_with_the_collection_path_as_cwd() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_update_hook("pwd", dir.path()).unwrap();
    // `pwd` reports the physical cwd — macOS `/var` resolves to
    // `/private/var`, so compare against the canonicalized tempdir.
    assert_eq!(
        out.stdout.trim(),
        dir.path().canonicalize().unwrap().to_string_lossy()
    );
    assert_eq!(out.code, 0);
}

#[test]
fn hook_propagates_a_nonzero_exit_code() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_update_hook("exit 42", dir.path()).unwrap();
    assert_eq!(out.code, 42);
}

#[test]
fn hook_killed_by_signal_reports_code_one() {
    let dir = tempfile::tempdir().unwrap();
    // `kill -9 $$` kills bash itself → `close` code is null → 1.
    let out = run_update_hook("kill -9 $$", dir.path()).unwrap();
    assert_eq!(out.code, 1);
}

#[test]
fn hook_spawn_failure_is_an_error() {
    let missing = tempfile::tempdir().unwrap().path().join("gone");
    assert!(run_update_hook("true", &missing).is_err());
}

#[test]
fn hook_output_is_utf8_lossy() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_update_hook("printf '\\xff\\xfe'", dir.path()).unwrap();
    assert!(out.stdout.contains('\u{fffd}'));
}
