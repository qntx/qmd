#!/usr/bin/env bun
/**
 * Regenerate `crates/qmd/tests/fixtures/lex_golden.json` by running the
 * upstream (tobi/qmd) FTS5 pipeline on a fixed corpus.
 *
 * Usage:
 *   QMD_UPSTREAM=/path/to/tobi/qmd bun scripts/gen_lex_golden.ts
 *
 * The fixture carries documents, queries, per-query upstream
 * `searchFTS` results and `validateLexQuery` outcomes, so the Rust test
 * can assert identical ranking and scores (within float tolerance)
 * without needing the TS runtime at test time.
 */

const upstream = process.env.QMD_UPSTREAM;
if (!upstream) {
  console.error("QMD_UPSTREAM=<path to tobi/qmd checkout> is required");
  process.exit(1);
}

const store = await import(`${upstream}/src/store.ts`);

const COLLECTION = "docs";

// ASCII-only corpus: the upstream FTS triggers never CJK-normalize, and
// identical token streams keep BM25 scores comparable across SQLite
// builds.
const DOCS: { path: string; title: string; body: string }[] = [
  { path: "perf.md", title: "Performance Guide",
    body: "# Performance Guide\n\nTuning and optimizing query performance.\nPerformance improves with better indexes.\n" },
  { path: "hello.md", title: "Hello World",
    body: "# Hello World\n\nhello world, this is a greeting document.\n" },
  { path: "ticket.md", title: "Ticket Notes",
    body: "# Ticket Notes\n\nDEC-0054 tracks the search regression.\n" },
  { path: "i18n.md", title: "i18n Module",
    body: "# i18n Module\n\nThe file src/lib/i18n.ts holds locale logic.\n" },
  { path: "agents.md", title: "Agents",
    body: "# Agents\n\nmulti-agent memory sharing across workers.\n" },
  { path: "secrets.md", title: "Secrets",
    body: "# Secrets\n\nDon't forget apply_secrets before deploy.\n" },
  { path: "release.md", title: "Release",
    body: "# Release\n\nVersion 2026.4.10 ships bugfixes.\n" },
  { path: "sports.md", title: "Sports",
    body: "# Sports\n\nA note about sports and games entirely.\n" },
];

const QUERIES: { query: string; limit: number; collection?: string }[] = [
  { query: "hello", limit: 20 },
  { query: "performance", limit: 20 },
  { query: "perform", limit: 20 },
  { query: '"hello world"', limit: 20 },
  { query: "hello -sports", limit: 20 },
  { query: "DEC-0054", limit: 20 },
  { query: "dec-0054", limit: 20 },
  { query: "src/lib/i18n.ts", limit: 20 },
  { query: "i18n", limit: 20 },
  { query: "multi-agent", limit: 20 },
  { query: "-multi-agent", limit: 20 },
  { query: "memory -agents", limit: 20 },
  { query: '"DEC-0054"', limit: 20 },
  { query: "HELLO", limit: 20 },
  { query: "2026.4.10", limit: 20 },
  { query: "don't", limit: 20 },
  { query: "apply_secrets", limit: 20 },
  { query: "nosuchterm", limit: 20 },
  { query: "hello world", limit: 20 },
  { query: "!!!", limit: 20 },
  { query: "-sports", limit: 20 },
  { query: '"sports"', limit: 20 },
  { query: "release version", limit: 20 },
  { query: "performance", limit: 1 },
  { query: "hello", limit: 20, collection: "docs" },
  { query: "hello", limit: 20, collection: "nosuch" },
];

const tmp = await import("node:fs/promises");
const os = await import("node:os");
const path = await import("node:path");

const dir = await tmp.mkdtemp(path.join(os.tmpdir(), "qmd-golden-"));
const dbPath = path.join(dir, "index.sqlite");
const s = store.createStore(dbPath);
const db = s.db;

const now = new Date().toISOString();
store.upsertStoreCollection(db, COLLECTION, { path: "/virtual/docs", pattern: "**/*.md" });
for (const doc of DOCS) {
  const hash = await store.hashContent(doc.body);
  store.insertContent(db, hash, doc.body, now);
  store.insertDocument(db, COLLECTION, doc.path, doc.title, hash, now, now);
}

const out = {
  collection: COLLECTION,
  documents: DOCS,
  queries: QUERIES.map((q) => ({
    query: q.query,
    limit: q.limit,
    collection: q.collection ?? null,
    lex_validation: store.validateLexQuery(q.query),
    results: store
      .searchFTS(db, q.query, q.limit, q.collection)
      .map((r: any) => ({ filepath: r.filepath, displayPath: r.displayPath, title: r.title, score: r.score })),
  })),
};

const fixture = new URL("../crates/qmd/tests/fixtures/lex_golden.json", import.meta.url).pathname;
await tmp.writeFile(fixture, JSON.stringify(out, null, 2) + "\n");
s.close();
await tmp.rm(dir, { recursive: true, force: true });
console.log(`wrote ${fixture}`);
