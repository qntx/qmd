#!/usr/bin/env bun
/**
 * Regenerate `crates/qmd-cli/tests/fixtures/cli_golden.json` by running the
 * upstream (tobi/qmd) CLI — `bun src/cli/qmd.ts` — over a fixed corpus and a
 * fixed list of invocations (P4-D3).
 *
 * Isolation (the reason this script exists in this shape):
 *   - every child runs with an explicit, minimal environment — nothing is
 *     inherited from this process except the directory holding `bun`;
 *   - HOME, XDG_CONFIG_HOME, XDG_CACHE_HOME, QMD_CONFIG_DIR, INDEX_PATH and
 *     TMPDIR all point inside a throwaway root under /tmp; bun's transpiler
 *     cache is disabled;
 *   - before/after the run, the real `~/.cache/qmd` / `~/.config/qmd` files
 *     are sha256-compared and the watched directories plus the upstream
 *     checkout are scanned for anything modified after the start marker.
 *     A qmd-related change outside the root aborts without writing the
 *     fixture; unrelated noise (other apps touching ~/Library/Caches) is
 *     listed for review.
 *
 * Fixture shape: `{upstream, generated, groups: [{name, files, config_name,
 * config_yaml, env, setup, delete_after_setup, scenarios: [{name, argv, cwd,
 * check, stdout, stderr, exit_code}]}]}`. Each group has its own root; in
 * recorded output that root is `<ROOT>` and the shared HOME is `<HOME>`.
 * `files` carries every corpus file with its content and mtime so the Rust
 * replay rebuilds the identical state; `config_yaml` is the config as
 * upstream left it after setup (upstream fills in `models:`).
 *
 * `check`: `exact` — stdout, stderr and exit code byte-identical;
 * `stdout-exit` — stderr embeds JS-engine text (`JSON.parse` messages) and
 * is compared by the Rust test with a prefix rule instead; `record` — kept
 * for reference only (help/version text differs by design, P4-D2/D8).
 *
 * Usage:
 *   QMD_UPSTREAM=/path/to/tobi/qmd bun scripts/gen_cli_golden.ts
 */

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  existsSync, lstatSync, mkdirSync, mkdtempSync, readdirSync, readFileSync,
  realpathSync, rmSync, unlinkSync, utimesSync, writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

const upstream = process.env.QMD_UPSTREAM;
if (!upstream) {
  console.error("QMD_UPSTREAM=<path to tobi/qmd checkout> is required");
  process.exit(1);
}
const BUN = process.execPath;
const QMD_TS = join(upstream, "src", "cli", "qmd.ts");
const REAL_HOME = homedir();
const fixturesDir = new URL("../crates/qmd-cli/tests/fixtures/", import.meta.url).pathname;

// ---------------------------------------------------------------------------
// Isolation guard
// ---------------------------------------------------------------------------

const sha256File = (p: string) => createHash("sha256").update(readFileSync(p)).digest("hex");

function realQmdState(): Record<string, string> {
  const out: Record<string, string> = {};
  for (const dir of [join(REAL_HOME, ".cache", "qmd"), join(REAL_HOME, ".config", "qmd")]) {
    if (!existsSync(dir)) continue;
    for (const name of readdirSync(dir)) {
      const p = join(dir, name);
      if (lstatSync(p).isFile()) out[p] = sha256File(p);
    }
  }
  return out;
}

function gitStatus(): string {
  return spawnSync("git", ["-C", upstream!, "status", "--porcelain", "--ignored"], { encoding: "utf8" }).stdout;
}

function changedSince(base: string, markerMs: number, skip: string[]): string[] {
  const out: string[] = [];
  if (!existsSync(base)) return out;
  const walk = (dir: string) => {
    let entries: string[];
    try { entries = readdirSync(dir); } catch { return; }
    for (const name of entries) {
      const p = join(dir, name);
      if (skip.some(s => p === s || p.startsWith(s + "/"))) continue;
      let st;
      try { st = lstatSync(p); } catch { continue; }
      if (Math.max(st.mtimeMs, st.ctimeMs) >= markerMs) out.push(p);
      if (st.isDirectory()) walk(p);
    }
  };
  walk(base);
  return out;
}

const WATCHED = [".cache", ".config", ".bun", ".local", "Library/Caches"].map(d => join(REAL_HOME, d));

// ---------------------------------------------------------------------------
// Corpus
// ---------------------------------------------------------------------------

// 2020-01-01T00:00:00Z: older than six months, so `ls` prints the year and
// the output does not depend on the current date.
const FIXED_MTIME = Date.UTC(2020, 0, 1) / 1000;

const faq = [
  "# FAQ",
  "",
  "## How do I search?",
  "Run qmd search with a keyword.",
  "Search ranks results with BM25.",
  "",
  "## Does search work offline?",
  "Yes, search never needs the network.",
  "",
  "## What about vectors?",
  "Vector search needs embeddings first.",
  "",
  "## Where is the index?",
  "The index lives in the cache directory.",
  "",
].join("\n");

// Exactly 1280 characters: `ls` shows `(1280/1024).toFixed(1)` = "1.3 KB"
// (JS rounds the exact .25 tie up; Rust `{:.1}` would print 1.2).
const bigHead = "# Big Reference\n\nThis big search reference exists to exceed small byte limits.\n";
const big = bigHead + "x".repeat(1280 - bigHead.length - 1) + "\n";

const CORPUS: Record<string, string> = {
  "docs/readme.md": "# QMD Readme\n\nQMD indexes markdown notes for search.\nSearch works offline and stays local.\n",
  "docs/guide/install.md": "# Install Guide\n\nInstall the binary, then run search once to verify.\n",
  "docs/guide/search-tips.md": "# Search \"Tips\", Tricks\n\nSearch tips: quote phrases to search exact text.\nSearch, search, search — repetition raises the score.\n",
  "docs/api.md": "---\nqmd:\n  metadata:\n    status: published\n    priority: 2\n    tags: [api, reference]\n---\n\n# API Reference\n\nThe search API returns ranked documents.\n",
  "docs/changelog.md": "---\nqmd:\n  metadata:\n    status: draft\n    priority: 1\n---\n\n# Changelog\n\nAdded faster search and a new index format.\n",
  "docs/faq.md": faq,
  "docs/big.md": big,
  "docs/cjk.md": "# 中文文档\n\n这是一个关于搜索的中文文档。\n全文搜索支持中文。\n",
  "docs/plain.md": "No heading here, just a plain search note.\n",
  "docs/drafts/secret.md": "# Secret Draft\n\nThis draft mentions search but is ignored.\n",
  "notes/meeting.md": "# Weekly Meeting\n\nDiscussed search quality, the index size and next steps.\n",
  "notes/todo.md": "# Todo\n\n- improve search speed\n- write docs\n",
  "archive/old.md": "# Old Archive Note\n\nAn archived search experiment.\n",
};

if (CORPUS["docs/big.md"]!.length !== 1280) throw new Error("big.md must be exactly 1280 chars");

const docid = (rel: string) => createHash("sha256").update(CORPUS[rel]!).digest("hex").slice(0, 6);

function mainConfig(corpus: string): string {
  return `global_context: Global knowledge base
collections:
  docs:
    path: ${corpus}/docs
    pattern: "**/*.md"
    ignore:
      - "drafts/**"
    context:
      /: Project documentation
      /guide: User guides, with "quotes"
  notes:
    path: ${corpus}/notes
    pattern: "**/*.md"
    context:
      /: Team notes
  archive:
    path: ${corpus}/archive
    pattern: "**/*.md"
    includeByDefault: false
`;
}

function singleCollectionConfig(corpus: string): string {
  return `collections:
  docs:
    path: ${corpus}/docs
    pattern: "**/*.md"
    ignore:
      - "drafts/**"
`;
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

type Check = "exact" | "stdout-exit" | "record";
type Cwd = "corpus" | "docs" | "root" | "outside";
type Scenario = { name: string; argv: string[]; cwd?: Cwd; check?: Check };

const FORMATS = ["--json", "--csv", "--md", "--xml", "--files"];
const each = (prefix: string, argv: string[], extra: Partial<Scenario> = {}): Scenario[] => [
  { name: `${prefix} cli`, argv, ...extra },
  ...FORMATS.map(f => ({ name: `${prefix} ${f.slice(2)}`, argv: [...argv, f], ...extra })),
];

const published = '{"key":"status","operator":"eq","value":"published"}';
const nested = '{"operator":"or","operands":[{"key":"status","operator":"eq","value":"draft"},{"key":"tags","operator":"in","value":["api"]}]}';
const noMatch = '{"key":"status","operator":"eq","value":"archived"}';

const mainScenarios: Scenario[] = [
  // search: formats, empty, min-score. "index" hits four documents across
  // two collections with distinct term frequencies (non-zero BM25); "search"
  // appears almost everywhere (IDF ~ 0) and covers the many-results paths.
  ...each("search", ["search", "index"]),
  ...each("search empty", ["search", "zzqxnomatch"]),
  ...each("search min-score", ["search", "index", "--min-score", "0.99"]),
  { name: "search --format json", argv: ["search", "index", "--format", "json"] },
  { name: "search --format md", argv: ["search", "index", "--format", "md"] },
  { name: "search --format bogus", argv: ["search", "index", "--format", "bogus"] },
  { name: "search --full json", argv: ["search", "index", "--full", "--json"] },
  { name: "search --full md", argv: ["search", "index", "--full", "--md", "-n", "2"] },
  { name: "search --full cli", argv: ["search", "index", "--full", "-n", "2"] },
  { name: "search --full xml", argv: ["search", "index", "--full", "--xml", "-n", "2"] },
  { name: "search --full csv", argv: ["search", "index", "--full", "--csv", "-n", "2"] },
  { name: "search --line-numbers cli", argv: ["search", "index", "--line-numbers"] },
  { name: "search --line-numbers json", argv: ["search", "index", "--line-numbers", "--json"] },
  { name: "search --line-numbers csv", argv: ["search", "index", "--line-numbers", "--csv"] },
  { name: "search -n 2", argv: ["search", "index", "-n", "2"] },
  { name: "search -n 2 json", argv: ["search", "index", "-n", "2", "--json"] },
  { name: "search --all files", argv: ["search", "search", "--all", "--files"] },
  { name: "search offline json", argv: ["search", "offline", "--json"] },
  { name: "search cjk json", argv: ["search", "搜索", "--json"] },
  { name: "search phrase files", argv: ["search", "\"search exact\"", "--files"] },
  { name: "search no query", argv: ["search"] },
  // collections
  { name: "search -c notes json", argv: ["search", "search", "-c", "notes", "--json"] },
  { name: "search -c docs -c notes files", argv: ["search", "search", "-c", "docs", "-c", "notes", "--files"] },
  { name: "search -c archive files", argv: ["search", "search", "-c", "archive", "--files"] },
  { name: "search -c missing", argv: ["search", "search", "-c", "missing"] },
  // metadata filter
  { name: "search filter eq json", argv: ["search", "search", "--filter", published, "--json"] },
  { name: "search filter nested files", argv: ["search", "search", "--filter", nested, "--files"] },
  ...each("search filter no-match", ["search", "search", "--filter", noMatch]),
  { name: "search filter invalid json", argv: ["search", "search", "--filter", "{not json"], check: "stdout-exit" },
  { name: "search filter invalid ast", argv: ["search", "search", "--filter", '{"key":"status","operator":"bogus","value":1}'] },
  // --full-path
  ...each("search full-path", ["search", "index", "--full-path"]),
  { name: "search full-path json from docs", argv: ["search", "index", "--full-path", "--json"], cwd: "docs" },
  { name: "search full-path json from outside", argv: ["search", "index", "--full-path", "--json"], cwd: "outside" },
  { name: "search full-path cli from outside", argv: ["search", "index", "--full-path", "-n", "2"], cwd: "outside" },

  // get
  { name: "get bare", argv: ["get", "docs/readme.md"] },
  { name: "get qmd uri", argv: ["get", "qmd://docs/readme.md"] },
  { name: "get collection path", argv: ["get", "docs/guide/install.md"] },
  { name: "get double slash", argv: ["get", "//docs/readme.md"] },
  { name: "get extra slashes", argv: ["get", "qmd:////docs/readme.md"] },
  { name: "get :line", argv: ["get", "docs/faq.md:5"] },
  { name: "get qmd :from:count", argv: ["get", "qmd://docs/faq.md:5:3"] },
  { name: "get :from:count -l override", argv: ["get", "docs/faq.md:5:3", "-l", "1"] },
  { name: "get --from -l", argv: ["get", "docs/faq.md", "--from", "3", "-l", "2"] },
  { name: "get negative --from", argv: ["get", "docs/faq.md", "--from", "-5", "-l", "2"] },
  { name: "get --no-line-numbers", argv: ["get", "docs/faq.md", "--no-line-numbers"] },
  { name: "get context", argv: ["get", "notes/meeting.md"] },
  { name: "get nested context", argv: ["get", "qmd://docs/guide/search-tips.md"] },
  { name: "get #docid", argv: ["get", `#${docid("docs/readme.md")}`] },
  { name: "get docid bare", argv: ["get", docid("notes/todo.md")] },
  { name: "get missing", argv: ["get", "docs/missing.md"] },
  { name: "get typo similar", argv: ["get", "docs/redme.md"] },
  { name: "get ignored", argv: ["get", "drafts/secret.md"], cwd: "docs" },
  { name: "get no arg", argv: ["get"] },
  { name: "get full-path under pwd", argv: ["get", "docs/readme.md", "--full-path", "-l", "2"] },
  { name: "get full-path outside pwd", argv: ["get", "qmd://docs/readme.md", "--full-path", "-l", "2"], cwd: "outside" },

  // multi-get
  ...each("multi-get glob", ["multi-get", "docs/guide/*.md"]),
  { name: "multi-get comma json", argv: ["multi-get", "docs/readme.md, notes/todo.md", "--json"] },
  { name: "multi-get comma prefixed md", argv: ["multi-get", "docs/readme.md,qmd://notes/todo.md", "--md"] },
  { name: "multi-get comma fragment", argv: ["multi-get", "readme,todo"] },
  { name: "multi-get docid json", argv: ["multi-get", `#${docid("docs/api.md")}`, "--json"] },
  { name: "multi-get missing docid", argv: ["multi-get", "#ffffff"] },
  { name: "multi-get comma docids files", argv: ["multi-get", `#${docid("docs/readme.md")},#${docid("notes/meeting.md")}`, "--files"] },
  // Comma list, not a glob: upstream's glob order is document insertion
  // order = filesystem readdir order (APFS vs ext4 differ); the Rust port
  // sorts the scan (P1 T7), registered as a deviation. The comma list keeps
  // every format's skip path covered with an order fixed by the argument.
  ...each("multi-get max-bytes", ["multi-get", "docs/faq.md,docs/big.md,docs/readme.md,docs/api.md", "--max-bytes", "600"]),
  { name: "multi-get -l 2", argv: ["multi-get", "docs/faq.md,docs/readme.md", "-l", "2"] },
  { name: "multi-get --no-line-numbers md", argv: ["multi-get", "docs/guide/*.md", "--no-line-numbers", "--md"] },
  { name: "multi-get full-path md", argv: ["multi-get", "docs/guide/*.md", "--full-path", "--md"] },
  { name: "multi-get full-path json from outside", argv: ["multi-get", "docs/guide/*.md", "--full-path", "--json"], cwd: "outside" },
  { name: "multi-get no match", argv: ["multi-get", "nothing/*.zz"] },
  { name: "multi-get cjk csv", argv: ["multi-get", "docs/cjk.md,docs/plain.md", "--csv"] },

  // ls
  { name: "ls", argv: ["ls"] },
  { name: "ls collection", argv: ["ls", "docs"] },
  { name: "ls prefix", argv: ["ls", "docs/guide"] },
  { name: "ls qmd uri", argv: ["ls", "qmd://docs/guide"] },
  { name: "ls qmd extra slashes", argv: ["ls", "qmd:////docs/guide"] },
  { name: "ls empty prefix", argv: ["ls", "docs/nowhere"] },
  { name: "ls missing collection", argv: ["ls", "nope"] },

  // dispatcher
  { name: "unknown command", argv: ["nosuchcmd"] },
  { name: "help is not a command", argv: ["help"] },
  { name: "version is not a command", argv: ["version"] },
  { name: "no args", argv: [], check: "record" },
  { name: "--help", argv: ["--help"], check: "record" },
  { name: "--version", argv: ["--version"], check: "record" },
];

const workScenarios: Scenario[] = [
  { name: "index search json", argv: ["--index", "work", "search", "index", "--json", "-n", "2"] },
  { name: "index search cli", argv: ["--index", "work", "search", "index", "-n", "2"] },
  { name: "index search files", argv: ["--index", "work", "search", "index", "--files", "-n", "2"] },
  { name: "get ?index= switches index", argv: ["get", "qmd://docs/readme.md?index=work", "-l", "2"] },
  { name: "index ls", argv: ["--index", "work", "ls", "docs/guide"] },
];

const staleScenarios: Scenario[] = [
  ...each("stale search full-path", ["search", "install", "--full-path"]),
  { name: "stale get full-path", argv: ["get", "docs/guide/install.md", "--full-path"] },
  { name: "stale multi-get full-path md", argv: ["multi-get", "docs/guide/*.md", "--full-path", "--md"] },
  { name: "stale search without full-path", argv: ["search", "install", "--json"] },
];

type Group = {
  name: string;
  configName: string;
  config: (corpus: string) => string;
  indexName?: string;
  // true: INDEX_PATH=<root>/index.sqlite; false: default DB under XDG_CACHE_HOME.
  indexPath: boolean;
  deleteAfterSetup: string[];
  scenarios: Scenario[];
};

const GROUPS: Group[] = [
  { name: "main", configName: "index.yml", config: mainConfig, indexPath: true, deleteAfterSetup: [], scenarios: mainScenarios },
  { name: "work", configName: "work.yml", config: singleCollectionConfig, indexName: "work", indexPath: false, deleteAfterSetup: [], scenarios: workScenarios },
  { name: "stale", configName: "index.yml", config: singleCollectionConfig, indexPath: true, deleteAfterSetup: ["docs/guide/install.md"], scenarios: staleScenarios },
];

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

const before = realQmdState();
const gitBefore = gitStatus();
const top = realpathSync(mkdtempSync("/tmp/qmd-cli-golden-"));
const home = join(top, "home");
const outside = join(top, "outside");
mkdirSync(home);
mkdirSync(outside);
await Bun.sleep(1100);
const marker = Date.now();

const normalize = (s: string, root: string) => s.split(root).join("<ROOT>").split(home).join("<HOME>");

const out = {
  upstream: {
    repo: "tobi/qmd",
    commit: spawnSync("git", ["-C", upstream, "rev-parse", "HEAD"], { encoding: "utf8" }).stdout.trim(),
    version: JSON.parse(readFileSync(join(upstream, "package.json"), "utf8")).version,
  },
  generated: new Date().toISOString(),
  groups: [] as unknown[],
};

let failed = false;
try {
  for (const group of GROUPS) {
    const root = join(top, group.name);
    const corpus = join(root, "corpus");
    const configDir = join(root, "config");
    const tmp = join(root, "tmp");
    for (const d of [corpus, configDir, tmp, join(root, "xdg-config"), join(root, "xdg-cache")]) mkdirSync(d, { recursive: true });
    for (const [rel, body] of Object.entries(CORPUS)) {
      const p = join(corpus, rel);
      mkdirSync(dirname(p), { recursive: true });
      writeFileSync(p, body);
      utimesSync(p, FIXED_MTIME, FIXED_MTIME);
    }
    writeFileSync(join(configDir, group.configName), group.config(corpus));

    const env: Record<string, string> = {
      PATH: `${dirname(BUN)}:/usr/bin:/bin`,
      HOME: home,
      XDG_CONFIG_HOME: join(root, "xdg-config"),
      XDG_CACHE_HOME: join(root, "xdg-cache"),
      QMD_CONFIG_DIR: configDir,
      TZ: "UTC",
      NO_COLOR: "1",
      QMD_DOCTOR_DEVICE_PROBE: "0",
      BUN_RUNTIME_TRANSPILER_CACHE_PATH: "0",
      TMPDIR: tmp,
    };
    if (group.indexPath) env.INDEX_PATH = join(root, "index.sqlite");

    const cwdOf = (c: Cwd = "corpus") => ({ corpus, docs: join(corpus, "docs"), root, outside }[c]);
    const run = (argv: string[], cwd: string) => {
      const r = spawnSync(BUN, [QMD_TS, ...argv], {
        cwd, env: { ...env, PWD: cwd }, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], timeout: 120_000,
      });
      if (r.error) throw r.error;
      return { stdout: normalize(r.stdout, root), stderr: normalize(r.stderr, root), exit_code: r.status ?? 1 };
    };

    const setupArgv = group.indexName ? ["--index", group.indexName, "update"] : ["update"];
    const setup = { argv: setupArgv, ...run(setupArgv, corpus) };
    if (setup.exit_code !== 0) throw new Error(`setup failed for ${group.name}: ${setup.stderr}`);
    for (const rel of group.deleteAfterSetup) unlinkSync(join(corpus, rel));

    const scenarios = group.scenarios.map(s => ({
      name: s.name,
      argv: s.argv,
      cwd: s.cwd ?? "corpus",
      check: s.check ?? "exact",
      ...run(s.argv, cwdOf(s.cwd)),
    }));

    out.groups.push({
      name: group.name,
      files: Object.entries(CORPUS).map(([path, content]) => ({ path, content, mtime: FIXED_MTIME })),
      config_name: group.configName,
      config_yaml: normalize(readFileSync(join(configDir, group.configName), "utf8"), root),
      index_name: group.indexName ?? null,
      index_path: group.indexPath,
      delete_after_setup: group.deleteAfterSetup,
      setup,
      scenarios,
    });
    console.error(`group ${group.name}: ${scenarios.length} scenarios`);
  }
} catch (err) {
  failed = true;
  console.error(err);
}

// ---------------------------------------------------------------------------
// Verify isolation, then clean up
// ---------------------------------------------------------------------------

const after = realQmdState();
const realChanged = [...new Set([...Object.keys(before), ...Object.keys(after)])]
  .filter(p => before[p] !== after[p]);
const watchedChanged = WATCHED.flatMap(w => changedSince(w, marker, []));
const upstreamChanged = changedSince(upstream, marker, [join(upstream, ".git")]);
const tmpChanged = changedSince("/private/tmp", marker, [top]).filter(p => !p.startsWith("/private/tmp/qmd-golden-probe"));
const gitChanged = gitStatus() !== gitBefore;
const qmdRelated = [...watchedChanged, ...tmpChanged].filter(p => /qmd/i.test(p));

rmSync(top, { recursive: true, force: true });

console.error(`isolation: real qmd files changed=${realChanged.length}, upstream files changed=${upstreamChanged.length}, upstream git status changed=${gitChanged}, qmd-related writes outside root=${qmdRelated.length}`);
for (const p of [...realChanged, ...upstreamChanged, ...qmdRelated]) console.error(`  VIOLATION ${p}`);
const noise = [...watchedChanged, ...tmpChanged].filter(p => !qmdRelated.includes(p));
for (const p of noise) console.error(`  note (unrelated write during run): ${p}`);

if (failed || realChanged.length || upstreamChanged.length || gitChanged || qmdRelated.length) {
  console.error("aborting: fixture not written");
  process.exit(1);
}

mkdirSync(fixturesDir, { recursive: true });
const dest = join(fixturesDir, "cli_golden.json");
writeFileSync(dest, JSON.stringify(out, null, 2) + "\n");
const total = (out.groups as { scenarios: unknown[] }[]).reduce((n, g) => n + g.scenarios.length, 0);
console.error(`wrote ${dest}: ${total} scenarios against tobi/qmd@${out.upstream.commit.slice(0, 12)} v${out.upstream.version}`);
