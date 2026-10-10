#!/usr/bin/env bun
/**
 * Regenerate `crates/qmd/tests/fixtures/chunk_golden.json` by running the
 * upstream (tobi/qmd) chunking pipeline on the vendored corpus.
 *
 * Usage:
 *   QMD_UPSTREAM=/path/to/tobi/qmd bun scripts/gen_chunk_golden.ts
 *
 * Records `scanBreakPoints`/`findCodeFences`/`chunkDocument` outputs per
 * fixture file plus canned `findBestCutoff`/`mergeBreakPoints` cases, so
 * the Rust port can assert identical positions (UTF-16 code units) and
 * texts without needing the TS runtime at test time. `chunkDocument`
 * delegates to `chunkDocumentWithBreakPoints` upstream, so the Rust side
 * compares the golden `scan_break_points`/`code_fences` fed into its own
 * `chunk_document_with_break_points` against these chunks.
 */

const upstream = process.env.QMD_UPSTREAM;
if (!upstream) {
  console.error("QMD_UPSTREAM=<path to tobi/qmd checkout> is required");
  process.exit(1);
}

const store = await import(`${upstream}/src/store.ts`);
const fs = await import("node:fs/promises");
const path = await import("node:path");
const { execSync } = await import("node:child_process");

const commit = execSync(`git -C "${upstream}" rev-parse HEAD`).toString().trim();
const fixturesDir = new URL("../crates/qmd/tests/fixtures/", import.meta.url).pathname;

const dirs = ["eval-docs", "chunk-docs"];
const files: string[] = [];
for (const d of dirs) {
  for (const f of (await fs.readdir(path.join(fixturesDir, d))).sort()) {
    if (f.endsWith(".md")) files.push(`${d}/${f}`);
  }
}

const PARAM_SETS = {
  default: { maxChars: 3600, overlapChars: 540, windowChars: 800 },
  small: { maxChars: 300, overlapChars: 45, windowChars: 120 },
  tight: { maxChars: 100, overlapChars: 20, windowChars: 10 },
};

const docs = [];
for (const file of files) {
  const content = await fs.readFile(path.join(fixturesDir, file), "utf8");
  const breakPoints = store.scanBreakPoints(content);
  const codeFences = store.findCodeFences(content);
  const chunks: Record<string, { text: string; pos: number }[]> = {};
  for (const [name, p] of Object.entries(PARAM_SETS)) {
    chunks[name] = store.chunkDocument(
      content, p.maxChars, p.overlapChars, p.windowChars,
    );
  }
  docs.push({
    file,
    scan_break_points: breakPoints.map((b: any) => ({ pos: b.pos, score: b.score, type: b.type })),
    code_fences: codeFences,
    chunks,
  });
}

const CUTOFF_CASES: {
  bps: { pos: number; score: number; type: string }[];
  target: number; window: number; decay: number;
  fences: { start: number; end: number }[];
}[] = [
  // Ported from test/store.test.ts "findBestCutoff".
  { bps: [{ pos: 100, score: 1, type: "newline" }, { pos: 150, score: 100, type: "h1" }, { pos: 180, score: 20, type: "blank" }],
    target: 200, window: 100, decay: 0.7, fences: [] },
  { bps: [{ pos: 100, score: 90, type: "h2" }, { pos: 195, score: 20, type: "blank" }],
    target: 200, window: 100, decay: 0.7, fences: [] },
  { bps: [{ pos: 150, score: 100, type: "h1" }, { pos: 195, score: 1, type: "newline" }],
    target: 200, window: 100, decay: 0.7, fences: [] },
  { bps: [{ pos: 10, score: 100, type: "h1" }],
    target: 200, window: 100, decay: 0.7, fences: [] },
  { bps: [{ pos: 150, score: 100, type: "h1" }, { pos: 180, score: 20, type: "blank" }],
    target: 200, window: 100, decay: 0.7, fences: [{ start: 140, end: 160 }] },
  { bps: [], target: 200, window: 100, decay: 0.7, fences: [] },
  // Rust-side edge coverage: zero window and window larger than target.
  { bps: [{ pos: 50, score: 20, type: "blank" }], target: 100, window: 0, decay: 0.7, fences: [] },
  { bps: [{ pos: 10, score: 20, type: "blank" }, { pos: 95, score: 5, type: "list" }],
    target: 100, window: 400, decay: 0.7, fences: [] },
];

const MERGE_CASES: {
  a: { pos: number; score: number; type: string }[];
  b: { pos: number; score: number; type: string }[];
}[] = [
  // Ported from test/store.test.ts + test/ast-chunking.test.ts "mergeBreakPoints".
  { a: [{ pos: 10, score: 20, type: "blank" }, { pos: 50, score: 1, type: "newline" }],
    b: [{ pos: 10, score: 90, type: "ast:func" }, { pos: 100, score: 100, type: "ast:class" }] },
  { a: [{ pos: 10, score: 20, type: "blank" }, { pos: 50, score: 1, type: "newline" }, { pos: 100, score: 20, type: "blank" }],
    b: [{ pos: 10, score: 90, type: "ast:func" }, { pos: 75, score: 100, type: "ast:class" }, { pos: 100, score: 60, type: "ast:import" }] },
  { a: [{ pos: 100, score: 10, type: "a" }], b: [{ pos: 5, score: 20, type: "b" }] },
];

const out = {
  upstream_commit: commit,
  param_sets: PARAM_SETS,
  docs,
  cutoff_cases: CUTOFF_CASES.map((c) => ({
    ...c,
    expected: store.findBestCutoff(c.bps, c.target, c.window, c.decay, c.fences),
  })),
  merge_cases: MERGE_CASES.map((c) => ({
    ...c,
    expected: store.mergeBreakPoints(c.a, c.b),
  })),
};

const fixture = new URL("../crates/qmd/tests/fixtures/chunk_golden.json", import.meta.url).pathname;
await fs.writeFile(fixture, JSON.stringify(out, null, 2) + "\n");
console.log(`wrote ${fixture}`);
