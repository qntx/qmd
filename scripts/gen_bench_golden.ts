/**
 * gen_bench_golden.ts — record upstream `qmd bench` internals for the
 * Rust golden replay (P7.1, issue #103).
 *
 * Three sections, all upstream-authoritative:
 *   1. `score` — every bench-score.test.ts input plus edge cases run
 *      through upstream `scoreResults`/`normalizePath`/`pathsMatch`.
 *   2. `run` — bm25-only `runBenchmark` over a throwaway `eval-docs`
 *      index, stdout+stderr captured, in json AND text mode, for the
 *      vendored `example.json` and the edge fixture (structured
 *      lines, CJK/emoji ids, error branches). Timestamps and
 *      latency_ms are folded so the fixture is stable.
 *   3. `guard` — `assertBenchCollectionReady` message text for the
 *      four guard branches.
 *
 * Run: QMD_UPSTREAM=/path/to/tobi/qmd bun scripts/gen_bench_golden.ts
 * Output: crates/qmd-cli/tests/fixtures/bench_golden.json
 */

import { mkdtempSync, writeFileSync, mkdirSync, cpSync, readFileSync } from "fs";
import { execSync } from "child_process";
import { join } from "path";

const UPSTREAM = process.env.QMD_UPSTREAM ?? "/Users/fjh/code/github/tobi/qmd";
const OUT = "crates/qmd-cli/tests/fixtures/bench_golden.json";

// Isolate EVERYTHING the recorder or upstream CLI touches — a stray
// default path writes to the real index/config. All four env vars
// point inside temp dirs, and a fail-fast guard asserts it before any
// store/CLI call.
import { mkdtempSync as _mk } from "fs";
const _root = _mk("/tmp/bench-golden-");
const _cfg = _root + "/cfg";
const _home = _root + "/home";
const _xdgC = _root + "/xdg-config";
const _xdgCa = _root + "/xdg-cache";
const _db = _cfg + "/index.sqlite";
import { mkdirSync as _md } from "fs";
for (const d of [_cfg, _home, _xdgC, _xdgCa]) _md(d, { recursive: true });
Object.assign(process.env, {
  QMD_CONFIG_DIR: _cfg,
  INDEX_PATH: _db,
  HOME: _home,
  XDG_CONFIG_HOME: _xdgC,
  XDG_CACHE_HOME: _xdgCa,
});
const _expected: [string, string][] = [
  ["QMD_CONFIG_DIR", _cfg],
  ["INDEX_PATH", _db],
  ["HOME", _home],
  ["XDG_CONFIG_HOME", _xdgC],
  ["XDG_CACHE_HOME", _xdgCa],
];
for (const [k, p] of _expected) {
  if (process.env[k] !== p || !p.startsWith(_root)) {
    throw new Error(`recorder isolation broken: ${k}=${process.env[k]} not under ${_root}`);
  }
}

const { scoreResults, normalizePath, pathsMatch } = await import(`${UPSTREAM}/src/bench/score.ts`);
const { runBenchmark, assertBenchCollectionReady } = await import(`${UPSTREAM}/src/bench/bench.ts`);
const rev = execSync(`git -C ${UPSTREAM} rev-parse HEAD`, { encoding: "utf-8" }).trim();

// ---- 1. score table --------------------------------------------------------
const scoreCases: { name: string; results: string[]; expected: string[]; topK: number }[] = [
  // bench-score.test.ts inputs, verbatim.
  { name: "perfect-all-in-topk", results: ["docs/a.md", "docs/b.md"], expected: ["a.md", "b.md"], topK: 2 },
  { name: "zero-none-found", results: ["docs/x.md", "docs/y.md"], expected: ["a.md", "b.md"], topK: 2 },
  { name: "found-outside-topk", results: ["x.md", "y.md", "a.md"], expected: ["a.md"], topK: 1 },
  { name: "mrr-rank2", results: ["x.md", "a.md"], expected: ["a.md"], topK: 2 },
  { name: "recall-at-n-matched", results: ["x.md", "concepts/a.md", "y.md", "b.md", "z.md"], expected: ["concepts/a.md", "b.md"], topK: 5 },
  // Edge cases.
  { name: "qmd-uri-both-sides", results: ["qmd://col/docs/a.md"], expected: ["qmd://other/docs/a.md"], topK: 1 },
  { name: "case-differs", results: ["Docs/A.md"], expected: ["docs/a.md"], topK: 1 },
  { name: "suffix-expected-longer", results: ["a.md"], expected: ["docs/a.md"], topK: 1 },
  { name: "duplicates-in-results", results: ["a.md", "a.md", "a.md"], expected: ["a.md"], topK: 2 },
  { name: "topk-larger-than-results", results: ["a.md"], expected: ["a.md", "b.md"], topK: 10 },
  { name: "topk-zero", results: ["a.md"], expected: ["a.md"], topK: 0 },
  { name: "empty-expected", results: ["a.md"], expected: [], topK: 2 },
];
const normalizeCases = [
  "Resources/Concepts/Context Engineering.md",
  "qmd://collection/docs/readme.md",
  "/docs/readme.md/",
  "readme.md",
];
const pathsMatchCases: [string, string][] = [
  ["docs/readme.md", "docs/readme.md"],
  ["Docs/README.md", "docs/readme.md"],
  ["/full/path/docs/readme.md", "docs/readme.md"],
  ["readme.md", "docs/readme.md"],
  ["qmd://col/docs/readme.md", "docs/readme.md"],
  ["docs/readme.md", "docs/other.md"],
];
const score = {
  results: scoreCases.map(c => ({ ...c, out: scoreResults(c.results, c.expected, c.topK) })),
  normalize: normalizeCases.map(p => ({ in: p, out: normalizePath(p) })),
  match: pathsMatchCases.map(([a, b]) => ({ a, b, out: pathsMatch(a, b) })),
};

// ---- 2. bm25 runBenchmark over a temp eval-docs index ----------------------
const tmp = mkdtempSync(_root + "/corpus-");
const corpusSrc = "crates/qmd/tests/fixtures/eval-docs";
const indexDir = join(tmp, "eval-docs");
mkdirSync(indexDir, { recursive: true });
cpSync(corpusSrc, indexDir, { recursive: true });
const cfgDir = _cfg;
const dbPath = _db;

// Build the index via upstream's own `qmd` CLI (collection add +
// update) so the store is exactly what upstream creates.
execSync(`cd ${UPSTREAM} && QMD_CONFIG_DIR=${cfgDir} INDEX_PATH=${dbPath} bun src/cli/qmd.ts collection add ${indexDir} --name eval-docs`, { stdio: "inherit" });

function capture(fn: () => Promise<unknown>) {
  const out: string[] = [];
  const err: string[] = [];
  const origOut = process.stdout.write.bind(process.stdout);
  const origErr = process.stderr.write.bind(process.stderr);
  const origLog = console.log;
  const origError = console.error;
  (process.stdout.write as any) = (s: any) => { out.push(String(s)); return true; };
  (process.stderr.write as any) = (s: any) => { err.push(String(s)); return true; };
  console.log = (...a: any[]) => out.push(a.join(" ") + "\n");
  console.error = (...a: any[]) => err.push(a.join(" ") + "\n");
  return fn().then(r => ({ r, out: out.join(""), err: err.join("") }))
    .finally(() => {
      process.stdout.write = origOut;
      process.stderr.write = origErr;
      console.log = origLog;
      console.error = origError;
    });
}

function fold(text: string): string {
  return text
    .replace(/"timestamp":\s*"[^"]*"/g, `"timestamp": "<TS>"`)
    .replace(/\d+ms\n/g, " NNNms\n")
    .replace(/Avg=\d+ms/g, "Avg=NNms")
    .replace(/"latency_ms":\s*\d+/g, `"latency_ms": 0`)
    .replace(/"avg_latency_ms":\s*[\d.]+/g, `"avg_latency_ms": 0`)
    .replace(/\s\d+ms$/gm, " NNms");
}
function foldJson(parsed: any): any {
  parsed.timestamp = "<TS>";
  for (const r of parsed.results ?? []) {
    for (const b of Object.values(r.backends ?? {}) as any[]) {
      b.latency_ms = 0;
      b.top_files = b.top_files;
    }
  }
  for (const s of Object.values(parsed.summary ?? {}) as any[]) s.avg_latency_ms = 0;
  return parsed;
}

// Edge fixture — structured lines, error branches, CJK/emoji id,
// no-type query, empty expected, large top_k. The `expand: x` single
// line is NOT a bench prefix → plain query. `intent:` alone and an
// unprefixed multi-line doc exercise the error branch.
const edge = {
  collection: "eval-docs",
  queries: [
    { id: "lex-only", query: "lex: versioning\nlex: fundraising", expected_files: ["api-design-principles.md"], expected_in_top_k: 1 },
    { id: "vec-only-empty-bm25", query: "vec: startup capital", expected_files: ["startup-fundraising-memo.md"], expected_in_top_k: 1 },
    { id: "mixed-intent", query: "lex: theorem\nintent: distributed systems", expected_files: ["distributed-systems-overview.md"], expected_in_top_k: 1 },
    { id: "expand-plain", query: "expand: ignored by bench", expected_files: [], expected_in_top_k: 1 },
    { id: "intent-alone-error", query: "intent: only an intent", expected_files: ["api-design-principles.md"], expected_in_top_k: 1 },
    { id: "unprefixed-multiline-error", query: "lex: versioning\nbare line", expected_files: ["api-design-principles.md"], expected_in_top_k: 1 },
    { id: "cjk-日本語-🚀-长id-over-twentyfive-units-here", query: "API versioning", type: "exact", expected_files: ["api-design-principles.md"], expected_in_top_k: 1 },
    { id: "no-type", query: "CAP theorem", expected_files: ["distributed-systems-overview.md"], expected_in_top_k: 1 },
    { id: "empty-expected", query: "machine learning", expected_files: [], expected_in_top_k: 1 },
    { id: "big-topk", query: "work", expected_files: ["remote-work-policy.md"], expected_in_top_k: 15 },
    // 1 real hit among 8 expected → precision_at_k = 0.125, an exact
    // toFixed(2) tie for the table's `num`.
    { id: "tie-125", query: "API versioning", expected_files: ["api-design-principles.md","fa.md","fb.md","fc.md","fd.md","fe.md","ff.md","fg.md"], expected_in_top_k: 8 },
  ],
};
const edgePath = "crates/qmd-cli/tests/fixtures/bench/edge.json";
writeFileSync(edgePath, JSON.stringify(edge, null, 2) + "\n");

const examplePath = "crates/qmd-cli/tests/fixtures/bench/example.json";

// A 5th run whose bm25 search finds nothing → the all-zero warning
// lands in the captured stderr (upstream writes it inside
// runBenchmark, so it is part of the recorded stream).
const zeroPath = "crates/qmd-cli/tests/fixtures/bench/zero.json";
writeFileSync(zeroPath, JSON.stringify({
  collection: "eval-docs",
  queries: [{ id: "zero", query: "zzz-no-such-term-qqq", expected_files: ["missing.md"], expected_in_top_k: 1 }],
}) + "\n");

const runs: any[] = [];
for (const [name, fixture, json] of [
  ["example-json", examplePath, true],
  ["example-text", examplePath, false],
  ["edge-json", edgePath, true],
  ["edge-text", edgePath, false],
  ["allzero-json", zeroPath, true],
] as [string, string, boolean][]) {
  const cap = await capture(() =>
    runBenchmark(fixture, { backends: ["bm25"], json, dbPath }),
  );
  runs.push({
    name, fixture, json,
    stdout: fold(cap.out),
    stderr: fold(cap.err),
    result: json ? foldJson(JSON.parse(cap.out)) : undefined,
  });
}

// Assert the edge fixture produced at least one metric that is an
// exact toFixed(2) tie (x.xx5) so the Rust rounding path is
// exercised — fail loudly if engineering didn't produce one.
const edgeRun = runs.find(r => r.name === "edge-json")!;
let tie = false;
for (const q of edgeRun.result.results) {
  for (const [k, v] of Object.entries(q.backends.bm25)) {
    if (typeof v === "number" && /\.[0-9]*5$/.test(String(v))) tie = true;
  }
}
if (!tie) {
  throw new Error(`edge fixture produced no .xx5 metric: ${JSON.stringify(edgeRun.result.results.map(r=>r.backends.bm25.precision_at_k))}`);
}

// ---- 3. guard messages -----------------------------------------------------
const guard = {
  missing: (() => { try { assertBenchCollectionReady([{ name: "other", active_count: 3 }], "eval-docs"); } catch (e: any) { return e.message; } })(),
  missingNoOthers: (() => { try { assertBenchCollectionReady([], "eval-docs"); } catch (e: any) { return e.message; } })(),
  empty: (() => { try { assertBenchCollectionReady([{ name: "eval-docs", active_count: 0 }], "eval-docs"); } catch (e: any) { return e.message; } })(),
  nothing: (() => { try { assertBenchCollectionReady([]); } catch (e: any) { return e.message; } })(),
};

const fixture = { upstream_commit: rev, score, runs, guard };
writeFileSync(OUT, JSON.stringify(fixture, null, 2) + "\n");
console.log(`wrote ${OUT} (upstream ${rev.slice(0, 8)})`);
