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
 * check, env, write, delete, post_state, stdout, stderr, exit_code, state}]}]}`.
 * Each group has its own root; in recorded output that root is `<ROOT>` and
 * the shared HOME is `<HOME>`. `files` carries every corpus file with its
 * content and mtime so the Rust replay rebuilds the identical state;
 * `config_yaml` is the config as upstream left it after setup (upstream fills
 * in `models:`).
 *
 * P4.2 additions: `env` overrides per scenario; `write`/`delete` mutate the
 * corpus before the invocation (content writes get the current mtime, which
 * `update` detects); `post_state` lists root-relative files captured after
 * the run as `{path, exists, content}` (`config/*.yml`, `.qmd/*`,
 * `trusted.json`) — `"trustedAt"` timestamps normalize to `<TS>`.
 *
 * P4.4 additions: `links` creates root-relative symlinks before the run;
 * `post_state` entries capture `link` targets when the path is a symlink
 * (`exists:true, content:null`); scenario `env` substitutes `<ROOT>`
 * in-place (e.g. `QMD_SKILLS_DIR` pointing at a fixture dir); the skills
 * groups are `noSetup` — they never touch the index.
 *
 * `check`: `exact` — stdout, stderr and exit code byte-identical;
 * `normalized` — like exact after replacing `\d+[smhd] ago` with `<AGO>`,
 * `Size:` values with `<SIZE>` and `trusted <ISO>` with `<TS>` (db bytes and
 * wall-clock-relative times cannot be byte-stable);
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
  readlinkSync, realpathSync, rmSync, symlinkSync, unlinkSync, utimesSync,
  writeFileSync,
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

type Check = "exact" | "normalized" | "stdout-exit" | "record";
type Cwd = "corpus" | "docs" | "root" | "outside" | "proj";
type Scenario = {
  name: string;
  argv: string[];
  cwd?: Cwd;
  check?: Check;
  // Per-scenario env overrides (e.g. QMD_TRUST_LOCAL_CONFIG).
  env?: Record<string, string>;
  // Corpus mutations applied before the invocation (P4.2 `update` tests).
  write?: Record<string, string>;
  delete?: string[];
  // Command run silently before the scenario (e.g. set an update hook).
  pre?: string[];
  // Root-relative files captured after the run (config, trusted.json, .qmd/*).
  post_state?: string[];
  // Root-relative symlinks to create before the invocation (P4.4 skills —
  // e.g. `.claude/skills` already pointing at `.agents/skills`).
  links?: { path: string; target: string }[];
};

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

  // -----------------------------------------------------------------
  // P4.3 — vsearch / vector-search. Setup ran `update` only, so there
  // is no vectors_vec table: vectorSearchQuery returns [] before any
  // expansion or embedding (store.ts:5879-5881) — every format of the
  // empty path is deterministic. stderr carries the checkIndexHealth
  // "need embeddings" warning that `search` never prints.
  // -----------------------------------------------------------------
  { name: "vsearch no query", argv: ["vsearch"] },
  { name: "vector-search no query", argv: ["vector-search"] },
  { name: "vsearch --format bogus", argv: ["vsearch", "--format", "bogus", "index"] },
  { name: "vsearch -c missing", argv: ["vsearch", "index", "-c", "missing"] },
  { name: "vsearch filter invalid json", argv: ["vsearch", "index", "--filter", "{not json"], check: "stdout-exit" },
  ...each("vsearch empty", ["vsearch", "index"]),
  { name: "vsearch empty --min-score", argv: ["vsearch", "index", "--min-score", "0.9"] },
  { name: "vsearch empty -c", argv: ["vsearch", "index", "-c", "notes"] },
  { name: "vsearch empty --filter", argv: ["vsearch", "index", "--filter", published] },
  { name: "vsearch empty --all", argv: ["vsearch", "index", "--all"] },
  { name: "vsearch empty --intent", argv: ["vsearch", "index", "--intent", "find docs"] },
  { name: "vsearch empty --full-path", argv: ["vsearch", "index", "--full-path"] },

  // -----------------------------------------------------------------
  // P4.3 — query / deep-search. Structured `lex:` documents reach real
  // results without any LLM (hybrid fts → RRF → --no-rerank); `vec:`
  // searches skip silently without vectors_vec. Plain queries and the
  // structured-parser/CI error paths die inside Bun's uncaught
  // rejection dump — engine-specific, so `record` only.
  // -----------------------------------------------------------------
  { name: "query no query", argv: ["query"] },
  { name: "deep-search no query", argv: ["deep-search"] },
  { name: "query --format bogus", argv: ["query", "--format", "bogus", "index"] },
  { name: "query -c missing", argv: ["query", "index", "-c", "missing"] },
  { name: "query filter invalid json", argv: ["query", "index", "--filter", "{not json"], check: "stdout-exit" },
  // Upstream validates --chunk-strategy inside parseCLI's cli.opts (qmd.ts:3213)
  // — an uncaught parse-time throw → Bun source dump, so record only.
  { name: "query --chunk-strategy bogus", argv: ["query", "index", "--chunk-strategy", "bogus"], check: "record" },
  ...each("query lex no-rerank", ["query", "lex: index", "--no-rerank"]),
  { name: "query lex+lex no-rerank", argv: ["query", "lex: index\nlex: search", "--no-rerank", "-n", "3"] },
  { name: "query lex -c no-rerank", argv: ["query", "lex: search", "-c", "notes", "--no-rerank"] },
  { name: "query lex --intent no-rerank", argv: ["query", "lex: index", "--intent", "find docs", "--no-rerank"] },
  { name: "query lex --explain no-rerank", argv: ["query", "lex: index", "--no-rerank", "--explain"] },
  { name: "query lex --full-path no-rerank", argv: ["query", "lex: index", "--no-rerank", "--full-path"] },
  { name: "query lex --full no-rerank", argv: ["query", "lex: index", "--no-rerank", "--full", "-n", "2"] },
  { name: "query vec no vectors", argv: ["query", "vec: index", "--no-rerank"] },
  { name: "query --candidate-limit", argv: ["query", "lex: index", "--no-rerank", "--candidate-limit", "1"] },
  { name: "query multi plain lines", argv: ["query", "index\nsearch"], check: "record" },
  { name: "query empty typed", argv: ["query", "lex:"], check: "record" },
  { name: "query intent alone", argv: ["query", "intent: just intent"], check: "record" },
  { name: "query expand mixed", argv: ["query", "expand: index\nlex: search"], check: "record" },
  { name: "query expand ci", argv: ["query", "index"], env: { CI: "1" }, check: "record" },
  { name: "query lex ci rerank", argv: ["query", "lex: index"], env: { CI: "1" }, check: "record" },

  // -----------------------------------------------------------------
  // P4.3 — embed option validation. All five parsers run inside the
  // command's try before the store opens, so each failure is a verbatim
  // `exitWithError` line + the doctor hint.
  // -----------------------------------------------------------------
  { name: "embed -c missing", argv: ["embed", "-c", "missing"] },
  { name: "embed --max-docs-per-batch abc", argv: ["embed", "--max-docs-per-batch", "abc"] },
  { name: "embed --max-docs-per-batch 0", argv: ["embed", "--max-docs-per-batch", "0"] },
  { name: "embed --max-batch-mb 0", argv: ["embed", "--max-batch-mb", "0"] },
  { name: "embed --max-batch-mb 1.5", argv: ["embed", "--max-batch-mb", "1.5"] },
  { name: "embed --timeout -1", argv: ["embed", "--timeout", "-1"] },
  { name: "embed --timeout abc", argv: ["embed", "--timeout", "abc"] },
  // Same parse-time throw as query's (qmd.ts:3213) → Bun dump → record.
  { name: "embed --chunk-strategy bogus", argv: ["embed", "--chunk-strategy", "bogus"], check: "record" },

  // -----------------------------------------------------------------
  // P4.3 — doctor. `normalized`: `Runtime:`/`SQLite runtime:`/
  // `sqlite-vec:`/`better-sqlite3 package:` report versions that differ
  // between Bun's bundled libs and rusqlite; everything else — config,
  // overrides, model cache misses, fingerprints, next steps — is
  // byte-stable against this empty index.
  // -----------------------------------------------------------------
  { name: "doctor", argv: ["doctor"], check: "normalized" },
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

// ---------------------------------------------------------------------------
// P4.2 — mutating commands. Scenarios run in order over shared group state;
// `post_state` records the config/trusted.json bytes after each mutation.
// ---------------------------------------------------------------------------

const CFG = "config/index.yml";
const TRUSTED = "config/trusted.json";

const mutateScenarios: Scenario[] = [
  { name: "collection list baseline", argv: ["collection", "list"], check: "normalized" },
  { name: "collection add", argv: ["collection", "add", "notes"], post_state: [CFG] },
  // basename(name="notes") now exists → next hits the name-dup check first.
  { name: "collection add dup name", argv: ["collection", "add", "archive", "--name", "notes"] },
  // Same path+pattern under a fresh name hits the second check.
  { name: "collection add dup path+pattern", argv: ["collection", "add", "notes", "--name", "other"] },
  { name: "collection add missing", argv: ["collection", "add", "nosuchdir"] },
  { name: "collection add file", argv: ["collection", "add", "docs/big.md"] },
  { name: "collection add no arg", argv: ["collection", "add"] },
  { name: "collection add --name --mask", argv: ["collection", "add", "docs/guide", "--name", "guides", "--mask", "search-*.md"], post_state: [CFG] },
  { name: "collection add --glob alias", argv: ["collection", "add", "archive", "--glob", "*.md"], post_state: [CFG] },
  { name: "collection list", argv: ["collection", "list"], check: "normalized" },
  { name: "collection show", argv: ["collection", "show", "docs"] },
  { name: "collection show missing", argv: ["collection", "show", "nope"] },
  { name: "collection exclude", argv: ["collection", "exclude", "docs"], post_state: [CFG] },
  { name: "collection include", argv: ["collection", "include", "docs"], post_state: [CFG] },
  { name: "collection exclude missing", argv: ["collection", "exclude", "nope"] },
  { name: "collection update-cmd set", argv: ["collection", "update-cmd", "docs", "echo upd-hook"], post_state: [CFG] },
  { name: "collection update-cmd clear", argv: ["collection", "update-cmd", "docs"], post_state: [CFG] },
  { name: "collection update-cmd missing", argv: ["collection", "update-cmd", "nope", "x"] },
  { name: "collection rename", argv: ["collection", "rename", "guides", "tutorials"], post_state: [CFG] },
  { name: "collection rename to existing", argv: ["collection", "rename", "tutorials", "docs"] },
  { name: "collection rm alias", argv: ["collection", "rm", "tutorials"], post_state: [CFG] },
  { name: "collection remove missing", argv: ["collection", "remove", "nope"] },
  { name: "collection bogus sub", argv: ["collection", "bogus"] },

  { name: "context add qmd uri", argv: ["context", "add", "qmd://docs/guide", "Guide files"], post_state: [CFG] },
  { name: "context add global", argv: ["context", "add", "/", "Root context"], post_state: [CFG] },
  { name: "context add fs path", argv: ["context", "add", "docs/faq.md", "FAQ file"], post_state: [CFG] },
  { name: "context add not indexed", argv: ["context", "add", "nosuchdir/x.md", "x"] },
  { name: "context add missing coll", argv: ["context", "add", "qmd://nope/x", "x"] },
  { name: "context list", argv: ["context", "list"], check: "normalized" },
  { name: "context rm", argv: ["context", "rm", "qmd://docs/guide"], post_state: [CFG] },
  { name: "context rm missing", argv: ["context", "rm", "qmd://docs/nowhere"] },

  { name: "status", argv: ["status"], check: "normalized" },
  { name: "cleanup dry-run", argv: ["cleanup", "--dry-run"] },
  { name: "cleanup", argv: ["cleanup"] },

  // `add .` resolves to the corpus dir itself — basename-named "corpus".
  { name: "collection add dot", argv: ["collection", "add", "."], post_state: [CFG] },

  { name: "update no-op", argv: ["update"] },
  { name: "update changed", argv: ["update"], write: { "docs/faq.md": faq + "\nA brand new line about search.\n" } },
  { name: "update new file", argv: ["update"], write: { "docs/newfile.md": "# New File\n\nFresh content about search.\n" } },
  { name: "update removed", argv: ["update"], delete: ["docs/newfile.md"] },
  { name: "update with hook", argv: ["update"], pre: ["collection", "update-cmd", "docs", "echo upd-hook"] },
  { name: "update hook fails", argv: ["update"], pre: ["collection", "update-cmd", "docs", "exit 3"] },
  { name: "update after hook cleared", argv: ["update"], pre: ["collection", "update-cmd", "docs"], post_state: [CFG] },
];

// Project-local `.qmd/index.yaml` gating: the config travels with a checkout,
// so `update` hooks / out-of-project paths / custom models are untrusted until
// approved. `docs` is in-project (hook gated), `outside` points at
// `<root>/external` (path gated). The generator never reaches the interactive
// prompt — piped stdin is not a TTY, so upstream takes the `skip` branch.
const localScenarios: Scenario[] = [
  { name: "update untrusted skips gated", argv: ["update"], post_state: ["corpus/.qmd/index.yaml"] },
  { name: "trust list empty", argv: ["trust", "list"] },
  { name: "trust", argv: ["trust"], post_state: [TRUSTED] },
  { name: "update trusted runs hook", argv: ["update"], check: "normalized" },
  { name: "trust list", argv: ["trust", "list"], check: "normalized" },
  { name: "trust revoke", argv: ["trust", "revoke"], post_state: [TRUSTED] },
  { name: "update env opted-in", argv: ["update"], env: { QMD_TRUST_LOCAL_CONFIG: "1" }, check: "normalized" },
  { name: "trust bogus sub", argv: ["trust", "bogus"] },
  { name: "init existing local config", argv: ["init"], post_state: ["corpus/.qmd/index.yaml"] },
  { name: "status local", argv: ["status"], check: "normalized" },
];

// `init` on a fresh directory — cwd `proj` starts empty.
const initScenarios: Scenario[] = [
  { name: "init", argv: ["init"], cwd: "proj", post_state: ["proj/.qmd/index.yml"] },
  { name: "init again", argv: ["init"], cwd: "proj", post_state: ["proj/.qmd/index.yml"] },
  // homedir() resolves $HOME — pointing it at the cwd exercises the refusal.
  { name: "init refuses HOME", argv: ["init"], cwd: "proj", env: { HOME: "<PROJ>" } },
  // P4.3 — embed on the empty local index: 0 pending hashes → the
  // early-exit line; `--force` skips the check, prints the banner +
  // Model line, and generateEmbeddings returns 0/0 without touching
  // the model (store.ts:2004-2007).
  { name: "embed empty index", argv: ["embed"], cwd: "proj", post_state: ["proj/.qmd/index.yml"] },
  { name: "embed --force empty", argv: ["embed", "--force"], cwd: "proj" },
  // P4.3 — doctor against a project-local index with no collections.
  { name: "doctor local", argv: ["doctor"], cwd: "proj", check: "normalized" },
];

// P4.4 — `skills`/`skill`/`--skill` (qmd.ts:3457-3633, 4920-4977). Pure
// filesystem commands: `noSetup` skips the index, installs run with
// cwd `root` so `.agents/skills/qmd` lands under post_state's reach.
// `record` scenarios are the deliberate divergences: the port does not
// ship `release` (P4-D7) so `list`/`--all` rows and `--full` bodies
// differ, and `path` reports the versioned cache-materialized dir
// instead of upstream's package tree.
const skillsScenarios: Scenario[] = [
  // `list`/`--json` are normalized — the generator folds upstream's
  // `release` row out so the `qmd` row compares byte-exact.
  { name: "skills list", argv: ["skills", "list"], check: "normalized" },
  { name: "skills bare", argv: ["skills"], check: "normalized" },
  { name: "skills list json", argv: ["skills", "list", "--json"], check: "normalized" },
  { name: "skills get qmd", argv: ["skills", "get", "qmd"] },
  { name: "skills get qmd full", argv: ["skills", "get", "qmd", "--full"] },
  { name: "skills get qmd json", argv: ["skills", "get", "qmd", "--json"] },
  { name: "skills get qmd json full", argv: ["skills", "get", "qmd", "--json", "--full"] },
  { name: "skills get dup", argv: ["skills", "get", "qmd", "qmd"] },
  { name: "skills get none", argv: ["skills", "get"] },
  { name: "skills get missing", argv: ["skills", "get", "missing"] },
  { name: "skills get missing json", argv: ["skills", "get", "missing", "--json"] },
  { name: "skills bogus", argv: ["skills", "bogus"] },
  { name: "skills bogus json", argv: ["skills", "bogus", "--json"] },
  { name: "skills help", argv: ["skills", "help"] },
  // Upstream `--help` short-circuits to the GENERAL help (P4-D2 record).
  { name: "skills --help", argv: ["skills", "--help"], check: "record" },
  { name: "skills get --all", argv: ["skills", "get", "--all"], check: "record" },
  { name: "skills get --all full", argv: ["skills", "get", "--all", "--full"], check: "record" },
  { name: "skills get --all json", argv: ["skills", "get", "--all", "--json"], check: "record" },
  // `path` prints upstream's package dir; ours is the cache-materialized
  // embedded dir — record both shapes, the missing-name error is exact.
  { name: "skills path", argv: ["skills", "path"], check: "record" },
  { name: "skills path qmd", argv: ["skills", "path", "qmd"], check: "record" },
  { name: "skills path json", argv: ["skills", "path", "--json"], check: "record" },
  { name: "skills path qmd json", argv: ["skills", "path", "qmd", "--json"], check: "record" },
  { name: "skills path missing", argv: ["skills", "path", "missing"] },
  { name: "skill bare", argv: ["skill"] },
  { name: "skill help", argv: ["skill", "help"] },
  { name: "skill --help", argv: ["skill", "--help"] },
  { name: "skill bogus", argv: ["skill", "bogus"] },
  { name: "skill show", argv: ["skill", "show"] },
  { name: "--skill", argv: ["--skill"] },
  { name: "--skill wins", argv: ["--skill", "search", "x"] },
  // Install sequence — order matters, state persists across scenarios.
  // `.claude/skills/qmd` starts as a regular file (extraFiles): the
  // `--yes` install writes `.agents/...` then fails on the link path.
  {
    name: "skill install blocked", argv: ["skill", "install", "--yes"], cwd: "root",
    check: "normalized",
    post_state: [".agents/skills/qmd/SKILL.md", ".agents/skills/qmd/references/mcp-setup.md"],
  },
  { name: "skill install again", argv: ["skill", "install"], cwd: "root", check: "normalized" },
  // Non-TTY stdin → the Tip line; the blocking file stays untouched
  // because the Tip path never reaches ensureClaudeSymlink.
  {
    name: "skill install force tip", argv: ["skill", "install", "-f"], cwd: "root",
    check: "normalized", post_state: [".claude/skills/qmd"],
  },
  // `-f` now replaces the blocking file with the relative symlink.
  {
    name: "skill install force yes", argv: ["skill", "install", "-f", "--yes"], cwd: "root",
    check: "normalized", post_state: [".claude/skills/qmd"],
  },
  // The existing correct link returns Linked again (idempotent, no force needed).
  {
    name: "skill install relink", argv: ["skill", "install", "-f", "--yes"], cwd: "root",
    check: "normalized", post_state: [".claude/skills/qmd"],
  },
  {
    name: "skill install global", argv: ["skill", "install", "--global", "--yes"], cwd: "root",
    check: "normalized", env: { HOME: "<ROOT>/ghome" },
    post_state: ["ghome/.agents/skills/qmd/SKILL.md", "ghome/.claude/skills/qmd"],
  },
  // QMD_SKILLS_DIR scenarios — env replaces the search dirs wholesale:
  // `list`/`--all` show `custom` only, `hidden` is gettable by name,
  // `path` echoes the env value, and install copies the fixture `qmd`.
  { name: "dir skills list", argv: ["skills", "list"], env: { QMD_SKILLS_DIR: "<ROOT>/fskills" } },
  { name: "dir skills list json", argv: ["skills", "list", "--json"], env: { QMD_SKILLS_DIR: "<ROOT>/fskills" } },
  { name: "dir skills get custom", argv: ["skills", "get", "custom"], env: { QMD_SKILLS_DIR: "<ROOT>/fskills" } },
  { name: "dir skills get custom full", argv: ["skills", "get", "custom", "--full"], env: { QMD_SKILLS_DIR: "<ROOT>/fskills" } },
  { name: "dir skills get hidden", argv: ["skills", "get", "hidden"], env: { QMD_SKILLS_DIR: "<ROOT>/fskills" } },
  { name: "dir skills get --all", argv: ["skills", "get", "--all"], env: { QMD_SKILLS_DIR: "<ROOT>/fskills" } },
  { name: "dir skills path", argv: ["skills", "path"], env: { QMD_SKILLS_DIR: "<ROOT>/fskills" }, check: "normalized" },
  { name: "dir skills path custom", argv: ["skills", "path", "custom"], env: { QMD_SKILLS_DIR: "<ROOT>/fskills" }, check: "normalized" },
  // The env value is echoed verbatim — relative stays relative.
  { name: "dir skills path rel", argv: ["skills", "path"], cwd: "root", env: { QMD_SKILLS_DIR: "fskills" } },
  // `skill show`/`install` with no `qmd` skill: upstream throws to an
  // unhandled Bun dump — record; ours renders the message + hint.
  { name: "dir skill show missing", argv: ["skill", "show"], env: { QMD_SKILLS_DIR: "<ROOT>/fskills/custom" }, check: "record" },
  // The override install copies the fixture `qmd` dir — stub SKILL.md
  // plus `references/ext.md`, and the embedded `mcp-setup.md` is gone.
  {
    name: "dir skill install", argv: ["skill", "install", "-f", "--yes"], cwd: "root",
    check: "normalized", env: { QMD_SKILLS_DIR: "<ROOT>/fskills" },
    post_state: [
      ".agents/skills/qmd/SKILL.md",
      ".agents/skills/qmd/references/ext.md",
      ".agents/skills/qmd/references/mcp-setup.md",
    ],
  },
];

// `.claude/skills` itself is a symlink to `../.agents/skills` — the
// already-visible loop guard prints `✓ Claude already sees...`.
const skillsLinkScenarios: Scenario[] = [
  {
    name: "skill install sees link", argv: ["skill", "install", "--yes"], cwd: "root",
    check: "normalized",
    links: [{ path: ".claude/skills", target: "../.agents/skills" }],
    post_state: [".agents/skills/qmd/SKILL.md", ".claude/skills"],
  },
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
  // Files under <root> outside the corpus (e.g. an out-of-project collection).
  extraFiles?: Record<string, string>;
  // Project-local `.qmd/index.yaml` written inside the corpus instead of the
  // global config dir (P4.2 trust scenarios); setup runs `update`.
  localConfig?: (root: string) => string;
  // Extra cwd root created empty (e.g. `proj` for `qmd init`).
  projDir?: boolean;
  // Skip the automatic `update` setup entirely.
  noSetup?: boolean;
};

const GROUPS: Group[] = [
  { name: "main", configName: "index.yml", config: mainConfig, indexPath: true, deleteAfterSetup: [], scenarios: mainScenarios },
  { name: "work", configName: "work.yml", config: singleCollectionConfig, indexName: "work", indexPath: false, deleteAfterSetup: [], scenarios: workScenarios },
  { name: "stale", configName: "index.yml", config: singleCollectionConfig, indexPath: true, deleteAfterSetup: ["docs/guide/install.md"], scenarios: staleScenarios },
  {
    name: "mutate", configName: "index.yml", config: singleCollectionConfig,
    indexPath: true, deleteAfterSetup: [], scenarios: mutateScenarios,
  },
  {
    name: "local", configName: "unused.yml",
    config: () => "",
    indexPath: false, deleteAfterSetup: [], scenarios: localScenarios,
    extraFiles: { "external/ext.md": "# External\n\nContent outside the project root.\n" },
    localConfig: (root) => `collections:
  docs:
    path: ${root}/corpus/docs
    pattern: "**/*.md"
    update: "echo hook-ran"
  outside:
    path: ${root}/external
    pattern: "**/*.md"
`,
  },
  {
    name: "init", configName: "unused.yml", config: () => "",
    indexPath: true, deleteAfterSetup: [], scenarios: initScenarios,
    projDir: true, noSetup: true,
  },
  {
    name: "skills", configName: "unused.yml", config: () => "",
    indexPath: true, deleteAfterSetup: [], scenarios: skillsScenarios,
    noSetup: true,
    extraFiles: {
      // `.claude/skills/qmd` starts as a regular FILE so the first
      // `--yes` install hits `Claude skill path already exists` — the
      // later `-f` install proves force removes it and links.
      ".claude/skills/qmd": "blocker\n",
      // A `QMD_SKILLS_DIR` fixture tree: one listed skill (multiline
      // frontmatter description), one `hidden:` skill (gettable by
      // name, filtered from list/--all), and a custom `qmd` skill the
      // `skill install` override path copies instead of the embedded one.
      "fskills/custom/SKILL.md":
        "---\nname: custom\ndescription: A fixture skill\n  spanning two lines.\n---\n\n# Custom\n\nCustom content.\n",
      "fskills/custom/references/r.md": "# Reference\n\nCustom reference body.\n",
      "fskills/hidden/SKILL.md":
        "---\nname: hidden\ndescription: Not listed.\nhidden: true\n---\n\n# Hidden\n\n",
      "fskills/qmd/SKILL.md":
        "---\nname: qmd\ndescription: Fixture qmd skill.\n---\n\n# Fixture QMD\n\nFixture-specific instructions.\n",
      "fskills/qmd/references/ext.md": "# Ext\n\nFixture reference body.\n",
    },
  },
  {
    name: "skills-link", configName: "unused.yml", config: () => "",
    indexPath: true, deleteAfterSetup: [], scenarios: skillsLinkScenarios,
    noSetup: true,
  },
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
    for (const [rel, body] of Object.entries(group.extraFiles ?? {})) {
      const p = join(root, rel);
      mkdirSync(dirname(p), { recursive: true });
      writeFileSync(p, body);
      utimesSync(p, FIXED_MTIME, FIXED_MTIME);
    }
    if (group.projDir) mkdirSync(join(root, "proj"), { recursive: true });
    if (group.localConfig) {
      // Project-local config inside the corpus — global config stays absent.
      const qmdDir = join(corpus, ".qmd");
      mkdirSync(qmdDir, { recursive: true });
      writeFileSync(join(qmdDir, "index.yaml"), group.localConfig(root));
    } else {
      writeFileSync(join(configDir, group.configName), group.config(corpus));
    }

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

    const proj = join(root, "proj");
    const cwdOf = (c: Cwd = "corpus") => ({ corpus, docs: join(corpus, "docs"), root, outside, proj }[c]);
    // `<AGO>`/`<SIZE>`/`<TS>`: db file size and clock-relative strings are
    // not byte-stable across implementations or run dates. The AST block is
    // flattened to `<AST>` — the port pins upstream's `not available` branch
    // until P6 while this upstream reports `active` plus a language list.
    // The doctor tokens fold deliberate driver/version differences:
    // `Runtime:` is `bun:sqlite` upstream vs `rusqlite` here; the SQLite
    // and sqlite-vec versions are whatever each build bundles; and the
    // `better-sqlite3 package` line reads upstream's own package.json
    // (a `^13.x` dep) while the port has none to declare.
    const softNormalize = (s: string) => s
      // `Index last updated N days ago` ages one per UTC day off the
      // pinned corpus mtime — fold the count like the compact `<AGO>`s.
      .replace(/Index last updated \d+ days ago/g, "Index last updated <AGO> days ago")
      .replace(/\d+[smhd] ago/g, "<AGO>")
      .replace(/(\n|^)(\s*)Size:.*(\n|$)/g, "$1$2Size: <SIZE>$3")
      .replace(/trusted \d{4}-\d{2}-\d{2}T[^\s)]*/g, "trusted <TS>")
      .replace(/"trustedAt": "[^"]*"/g, '"trustedAt": "<TS>"')
      .replace(/^  Status:.*$/gm, "  Status: <AST>")
      .replace(/^  Languages:.*\n/gm, "")
      .replace(/^  Unavailable:.*\n/gm, "")
      .replace(/^Runtime:.*$/gm, "Runtime: <DRIVER>")
      .replace(/SQLite runtime:.*$/gm, "SQLite runtime: <VER>")
      .replace(/better-sqlite3 package:.*$/gm, "better-sqlite3 package: <PKG>")
      .replace(/sqlite-vec:.*$/gm, "sqlite-vec: <VER>")
      // P4.4: the port does not ship upstream's `release` skill (P4-D7 — a
      // maintainer-only release workflow). Fold it out of `skills list`
      // output so the rows our `qmd` skill produces still compare.
      .replace(/^  release  .*\n/gm, "")
      .replace(/,\{"name":"release","description":"(?:[^"\\]|\\.)*"\}/g, "")
      // ...and the `qmd` row's padding was sized for `release` — fold
      // the gap so the port's tighter padding still compares.
      .replace(/^  qmd {2,}/gm, "  qmd  ");
    const run = (argv: string[], cwd: string, extra?: Record<string, string>) => {
      const e = { ...env, PWD: cwd };
      // `<PROJ>` substitutes whole-value; `<ROOT>` substitutes in-place so a
      // value like `<ROOT>/fskills` stays single-token in the fixture.
      for (const [k, v] of Object.entries(extra ?? {}))
        e[k] = v === "<PROJ>" ? proj : v.split("<ROOT>").join(root);
      const r = spawnSync(BUN, [QMD_TS, ...argv], {
        cwd, env: e, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], timeout: 120_000,
      });
      if (r.error) throw r.error;
      return { stdout: normalize(r.stdout, root), stderr: normalize(r.stderr, root), exit_code: r.status ?? 1 };
    };
    const postState = (paths: string[] | undefined) =>
      (paths ?? []).map(rel => {
        const p = join(root, rel);
        // lstat: a symlink to a directory is `exists` but unreadable —
        // capture the link target instead (P4.4 `.claude/skills/qmd`).
        if (lstatSync(p, { throwIfNoEntry: false })?.isSymbolicLink()) {
          return { path: rel, exists: true, content: null, link: readlinkSync(p) };
        }
        const exists = existsSync(p);
        return {
          path: rel,
          exists,
          content: exists ? softNormalize(normalize(readFileSync(p, "utf8"), root)) : null,
        };
      });

    const setupArgv = group.indexName ? ["--index", group.indexName, "update"] : ["update"];
    const setup = group.noSetup
      ? null
      : { argv: setupArgv, ...run(setupArgv, corpus) };
    if (setup && setup.exit_code !== 0) throw new Error(`setup failed for ${group.name}: ${setup.stderr}`);
    for (const rel of group.deleteAfterSetup) unlinkSync(join(corpus, rel));

    // `config_yaml` is the config at scenario-start — read right after
    // setup (upstream's `update` has already appended `models:`), not at
    // group end: `mutate` scenarios mutate the file further, so the
    // end-of-run bytes are not the setup input the replay needs.
    const configFile = group.localConfig
      ? join(corpus, ".qmd", "index.yaml")
      : join(configDir, group.configName);
    const postSetupConfig = readFileSync(configFile, "utf8");

    const scenarios = group.scenarios.map(s => {
      if (s.pre) run(s.pre, cwdOf(s.cwd));
      for (const [rel, body] of Object.entries(s.write ?? {})) writeFileSync(join(corpus, rel), body);
      for (const rel of s.delete ?? []) unlinkSync(join(corpus, rel));
      for (const l of s.links ?? []) {
        const p = join(root, l.path);
        mkdirSync(dirname(p), { recursive: true });
        symlinkSync(l.target, p);
      }
      const result = run(s.argv, cwdOf(s.cwd), s.env);
      const norm = s.check === "normalized";
      return {
        name: s.name,
        argv: s.argv,
        cwd: s.cwd ?? "corpus",
        check: s.check ?? "exact",
        env: s.env,
        write: s.write,
        delete: s.delete,
        pre: s.pre,
        links: s.links,
        stdout: norm ? softNormalize(result.stdout) : result.stdout,
        stderr: norm ? softNormalize(result.stderr) : result.stderr,
        exit_code: result.exit_code,
        state: postState(s.post_state),
      };
    });

    out.groups.push({
      name: group.name,
      files: Object.entries(CORPUS).map(([path, content]) => ({ path, content, mtime: FIXED_MTIME })),
      extra_files: Object.entries(group.extraFiles ?? {}).map(([path, content]) => ({
        path, content, mtime: FIXED_MTIME,
      })),
      config_name: group.localConfig ? null : group.configName,
      config_yaml: normalize(postSetupConfig, root),
      local_config: group.localConfig ? "corpus/.qmd/index.yaml" : null,
      proj_dir: group.projDir ?? false,
      no_setup: group.noSetup ?? false,
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
