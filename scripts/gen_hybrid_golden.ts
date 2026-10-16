#!/usr/bin/env bun
/**
 * Regenerate `crates/qmd/tests/fixtures/hybrid_golden.json` by running
 * the upstream (tobi/qmd) pure functions that P2.4 ports:
 *
 *   - getCacheKey (store.ts)
 *   - reciprocalRankFusion / buildRrfTrace / getHybridRrfWeights (store.ts)
 *   - extractIntentTerms / extractSnippet (store.ts)
 *   - parseStructuredQuery (cli/qmd.ts — module-private, extracted via
 *     Bun.Transpiler type-stripping and evaluated in isolation)
 *
 * Usage:
 *   QMD_UPSTREAM=/path/to/tobi/qmd bun scripts/gen_hybrid_golden.ts
 *
 * No models and no database are involved — every case is a pure
 * function call, so the fixture is deterministic.
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
const version = JSON.parse(
  await fs.readFile(`${upstream}/package.json`, "utf8"),
).version;
const fixturesDir = new URL("../crates/qmd/tests/fixtures/", import.meta.url).pathname;

// ── parseStructuredQuery (not exported — lift from cli/qmd.ts) ────────
const cliSrc = await fs.readFile(`${upstream}/src/cli/qmd.ts`, "utf8");
const fnMatch = cliSrc.match(/function parseStructuredQuery[\s\S]*?\n\}\n/);
if (!fnMatch) {
  console.error("parseStructuredQuery not found in cli/qmd.ts");
  process.exit(1);
}
const stripped = new Bun.Transpiler({ loader: "ts" }).transformSync(
  `${fnMatch[0]}\nreturn parseStructuredQuery;`,
);
const parseStructuredQuery: (q: string) => unknown = new Function(stripped)() as never;

// ── Case inputs ──────────────────────────────────────────────────────

const rr = (file: string, score: number) => ({
  file,
  displayPath: file.replace("qmd://", ""),
  title: file,
  body: `body of ${file}`,
  score,
});

const rrfInputs: {
  name: string;
  lists: unknown[][];
  weights: number[];
  metas: { source: string; queryType: string; query: string }[];
  k: number;
}[] = [
  {
    name: "single_list",
    lists: [[rr("qmd://docs/a.md", 0.9), rr("qmd://docs/b.md", 0.8), rr("qmd://docs/c.md", 0.7)]],
    weights: [2.0],
    metas: [{ source: "fts", queryType: "original", query: "alpha" }],
    k: 60,
  },
  {
    name: "merge_across_lists_original_2x",
    lists: [
      [rr("qmd://docs/a.md", 0.92), rr("qmd://docs/b.md", 0.81)],
      [rr("qmd://docs/b.md", 0.77), rr("qmd://docs/a.md", 0.65)],
      [rr("qmd://docs/c.md", 0.7)],
    ],
    weights: [2.0, 1.0, 2.0],
    metas: [
      { source: "fts", queryType: "original", query: "alpha" },
      { source: "fts", queryType: "lex", query: "alpha terms" },
      { source: "vec", queryType: "original", query: "alpha" },
    ],
    k: 60,
  },
  {
    name: "missing_weight_defaults_to_one",
    lists: [[rr("qmd://docs/a.md", 0.9)], [rr("qmd://docs/b.md", 0.9)]],
    weights: [],
    metas: [
      { source: "fts", queryType: "original", query: "q" },
      { source: "vec", queryType: "vec", query: "q v" },
    ],
    k: 60,
  },
  {
    name: "custom_k",
    lists: [[rr("qmd://docs/a.md", 0.9), rr("qmd://docs/b.md", 0.5)]],
    weights: [1.0],
    metas: [{ source: "fts", queryType: "original", query: "q" }],
    k: 30,
  },
  {
    name: "rank_bonus_boundary",
    lists: [
      [
        rr("qmd://docs/r1.md", 0.9),
        rr("qmd://docs/r2.md", 0.8),
        rr("qmd://docs/r3.md", 0.7),
        rr("qmd://docs/r4.md", 0.6),
      ],
    ],
    weights: [1.0],
    metas: [{ source: "fts", queryType: "original", query: "r" }],
    k: 60,
  },
];

const hybridWeightInputs = [
  [
    { source: "fts", queryType: "original", query: "q" },
    { source: "fts", queryType: "lex", query: "q lex" },
    { source: "vec", queryType: "original", query: "q" },
    { source: "vec", queryType: "vec", query: "q vec" },
    { source: "vec", queryType: "hyde", query: "q hyde" },
  ],
  [{ source: "fts", queryType: "original", query: "q" }],
  [
    { source: "fts", queryType: "lex", query: "a" },
    { source: "fts", queryType: "vec", query: "b" },
  ],
];

const cacheKeyBodies = [
  { url: "expandQuery", body: { query: "alpha bravo", model: "hf:Qwen/Qwen3-0.6B-GGUF" } },
  { url: "expandQuery", body: { query: "C++ \"templates\"\nline2", model: "m-gen" } },
  { url: "expandQuery", body: { query: "日本語 クエリ 🚀", model: "m-gen" } },
  { url: "rerank", body: { query: "intent text\n\nalpha", model: "m-rerank", chunk: "chunk body" } },
  { url: "rerank", body: { query: "q", model: "m-rerank", chunk: "" } },
  { url: "rerank", body: { model: "m-rerank", chunk: "no query field" } },
];

const parseInputs = [
  "just a plain query",
  "expand: something to expand",
  "lex: alpha terms\nvec: semantic variant\nhyde: hypothetical answer",
  "intent: focus on tradeoffs\nlex: cap theorem",
  "vec: embeddings only\nintent: keep it short",
  "lex: one\nLEX: two\nVec: three\nHYDE: four",
  "lex: trailing spaces   \n\n  \nvec: after blanks",
  "expand: first\nlex: second",
  "lex: first\nexpand: second",
  "intent: only intent no searches",
  "intent: dup one\nintent: dup two\nlex: x",
  "lex:",
  "vec:   ",
  "intent:  ",
  "lex: ok\nthis line lacks a prefix\nvec: tail",
  "lex: ok\nrandom middle line",
  "two plain lines\nsecond plain line",
  "\n\n",
  "   ",
  "lex: a\nlex: b\nlex: c",
];

const intentInputs = [
  "quickly find the C++ template metaprogramming guide",
  "I need help with on-call runbooks and post-mortems",
  "the quick brown fox jumps over the lazy dog",
  "a an the of to in for on at by with",
  "multi-word state-of-the-art hyphenated terms keep internal dashes",
  "UPPERCASE MixedCase lowercase",
  "trailing punctuation, like commas, periods! and? question—marks…",
  "emoji 🚀 rocket and ☕ coffee symbols",
  "résumé café naïve übermensch",
  "single x y z letters filtered",
];

const snippetBodies = {
  simple:
    "# Notes\n\nfirst line about alpha\nsecond line is filler\nthird line mentions alpha again\nlast line\n",
  windowed: Array.from({ length: 30 }, (_, i) =>
    `line ${i + 1} ${i === 12 ? "alpha needle here" : "filler"}`,
  ).join("\n"),
  emoji: "🚀🚀🚀 intro line\nsecond line alpha\nthird line\n",
  long: "header\n" + "x".repeat(700) + " alpha\n" + "y".repeat(300) + "\n",
};

const snippetInputs = [
  { body: snippetBodies.simple, query: "alpha", maxLen: 500 },
  { body: snippetBodies.simple, query: "alpha", maxLen: 60 },
  { body: snippetBodies.windowed, query: "needle", maxLen: 500, chunkPos: 0, chunkLen: 400 },
  { body: snippetBodies.windowed, query: "nomatch", maxLen: 500, chunkPos: 0, chunkLen: 400 },
  { body: snippetBodies.windowed, query: "nomatch", maxLen: 500, chunkPos: 100, chunkLen: 400 },
  { body: snippetBodies.emoji, query: "alpha", maxLen: 500, chunkPos: 4, chunkLen: 400 },
  { body: snippetBodies.long, query: "alpha", maxLen: 500 },
  { body: snippetBodies.simple, query: "zzz", maxLen: 500, intent: "about alpha topics" },
  { body: snippetBodies.simple, query: "zzz alpha", maxLen: 500, intent: "filler second" },
];

// ── Run cases ─────────────────────────────────────────────────────────

const cases: Record<string, unknown> = {};

cases.cache_key = cacheKeyBodies.map(({ url, body }) => ({
  url,
  body,
  key: store.getCacheKey(url, body),
}));

cases.rrf = rrfInputs.map(({ name, lists, weights, metas, k }) => ({
  name,
  lists,
  weights,
  metas,
  k,
  fused: store
    .reciprocalRankFusion(lists as never, weights, k)
    .map((r: { file: string; score: number }) => ({ file: r.file, score: r.score })),
}));

cases.rrf_trace = rrfInputs.map(({ name, lists, weights, metas, k }) => {
  const trace = store.buildRrfTrace(lists as never, weights, metas as never, k);
  const out: Record<string, unknown> = {};
  for (const [file, t] of trace as Map<string, Record<string, unknown>>) {
    out[file] = t;
  }
  return { name, lists, weights, metas, k, trace: out };
});

cases.hybrid_weights = hybridWeightInputs.map((metas) => ({
  metas,
  weights: store.getHybridRrfWeights(metas as never),
}));

cases.parse_query_doc = parseInputs.map((input) => {
  try {
    return { input, result: parseStructuredQuery(input) ?? null, error: null };
  } catch (e) {
    return { input, result: null, error: String((e as Error).message) };
  }
});

cases.intent_terms = intentInputs.map((intent) => ({
  intent,
  terms: store.extractIntentTerms(intent),
}));

cases.snippet = snippetInputs.map(({ body, query, maxLen, chunkPos, chunkLen, intent }) => ({
  body,
  query,
  maxLen,
  chunkPos: chunkPos ?? null,
  chunkLen: chunkLen ?? null,
  intent: intent ?? null,
  result: store.extractSnippet(body, query, maxLen, chunkPos, chunkLen, intent),
}));

const fixture = {
  upstream_commit: commit,
  upstream_version: version,
  generated_at: new Date().toISOString(),
  cases,
};

const out = path.join(fixturesDir, "hybrid_golden.json");
await fs.writeFile(out, `${JSON.stringify(fixture, null, 2)}\n`);
console.log(`wrote ${out} (upstream ${commit.slice(0, 8)} v${version})`);
