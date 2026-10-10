#!/usr/bin/env bun
/**
 * Regenerate `crates/qmd/tests/fixtures/eval_bm25_golden.json` for the
 * AC3 acceptance check: upstream `test/eval-docs` (vendored in
 * `fixtures/eval-docs/` with its LICENSE) indexed via
 * `insertContent`/`insertDocument` exactly like
 * `eval-bm25.test.ts`, then `searchFTS(db, q, 5)` for each of its 24
 * eval queries — parsed out of the upstream test file.
 *
 * Usage:
 *   QMD_UPSTREAM=/path/to/tobi/qmd bun scripts/gen_eval_golden.ts
 */

import { execSync } from "node:child_process";

const upstream = process.env.QMD_UPSTREAM;
if (!upstream) {
  console.error("QMD_UPSTREAM=<path to tobi/qmd checkout> is required");
  process.exit(1);
}

const store = await import(`${upstream}/src/store.ts`);

// Pull the evalQueries table out of upstream's test so the fixture can
// never drift from the acceptance corpus.
const testSource = await Bun.file(`${upstream}/test/eval-bm25.test.ts`).text();
const ENTRY = /\{\s*query:\s*"((?:[^"\\]|\\.)*)",\s*expectedDoc:\s*"([^"]+)",\s*difficulty:\s*"([^"]+)"/g;
const queries = [...testSource.matchAll(ENTRY)].map((m) => ({
  query: JSON.parse(`"${m[1]}"`) as string,
  expectedDoc: m[2],
  difficulty: m[3],
}));
if (queries.length !== 24) {
  console.error(`expected 24 eval queries, parsed ${queries.length}`);
  process.exit(1);
}

const docsDir = new URL("../crates/qmd/tests/fixtures/eval-docs/", import.meta.url).pathname;

const tmp = await import("node:fs/promises");
const os = await import("node:os");
const path = await import("node:path");
const { createHash } = await import("node:crypto");

const dir = await tmp.mkdtemp(path.join(os.tmpdir(), "qmd-eval-golden-"));
const s = store.createStore(path.join(dir, "index.sqlite"));
const db = s.db;

// Mirrors eval-bm25.test.ts beforeAll: sha256[0:12] hash, title from the
// first `# ` line, collection "eval-docs", path = bare filename.
const now = new Date().toISOString();
const files = (await tmp.readdir(docsDir)).filter((f) => f.endsWith(".md")).sort();
for (const file of files) {
  const content = await tmp.readFile(path.join(docsDir, file), "utf-8");
  const title = content.split("\n")[0]?.replace(/^#\s*/, "") || file;
  const hash = createHash("sha256").update(content).digest("hex").slice(0, 12);
  store.insertContent(db, hash, content, now);
  store.insertDocument(db, "eval-docs", file, title, hash, now, now);
}

const out = {
  upstream_commit: execSync("git rev-parse HEAD", { cwd: upstream }).toString().trim(),
  collection: "eval-docs",
  queries: queries.map((q) => ({
    ...q,
    results: store
      .searchFTS(db, q.query, 5)
      .map((r: any) => ({ filepath: r.filepath, score: r.score })),
  })),
};

const fixture = new URL("../crates/qmd/tests/fixtures/eval_bm25_golden.json", import.meta.url).pathname;
await tmp.writeFile(fixture, JSON.stringify(out, null, 2) + "\n");
s.close();
await tmp.rm(dir, { recursive: true, force: true });
console.log(`wrote ${fixture} (${queries.length} queries, ${files.length} docs)`);
