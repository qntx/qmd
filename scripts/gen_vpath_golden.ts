#!/usr/bin/env bun
/**
 * Regenerate `crates/qmd/tests/fixtures/vpath_golden.json` by running
 * the upstream (tobi/qmd) virtual-path, docid, glob and comma-list
 * resolvers on fixed inputs.
 *
 * Usage:
 *   QMD_UPSTREAM=/path/to/tobi/qmd bun scripts/gen_vpath_golden.ts
 *
 * The fixture records upstream outputs verbatim so the Rust test can
 * assert identical normalization/parsing/matching without needing the
 * TS runtime at test time.
 */

const upstream = process.env.QMD_UPSTREAM;
if (!upstream) {
  console.error("QMD_UPSTREAM=<path to tobi/qmd checkout> is required");
  process.exit(1);
}

const store = await import(`${upstream}/src/store.ts`);

const tmp = await import("node:fs/promises");
const os = await import("node:os");
const path = await import("node:path");

// --- pure functions ---------------------------------------------------------

const PATH_INPUTS = [
  "qmd://notes/a.md",
  "qmd:////notes/a.md",
  "qmd:notes/a.md",
  "//notes/a.md",
  "  qmd://notes/a.md  ",
  "notes/a.md",
  "/abs/a.md",
  "qmd://notes",
  "qmd://notes/",
  "qmd://",
  "qmd:///a.md",
  "qmd://notes/a.md?index=main",
  "qmd://notes/a.md?index=my%20idx",
  "qmd://notes/a.md?index=a+b",
  "qmd://notes/a.md?other=1&index=i",
  "qmd://notes/a.md?index=",
  "a.md:10",
  "a.md:10:20",
];

const DOCID_INPUTS = [
  "#abc123",
  "abc123",
  '"abc123"',
  "'abc123'",
  '"#abc123"',
  "  #abc123  ",
  "abc12",
  "abcdefg",
  "#",
  "",
  "a.md",
  "docs/a.md",
];

const HASHES = ["0123456789abcdef", "abc", ""];

const BUILD_ARGS: [string, string, string | null][] = [
  ["notes", "a.md", null],
  ["notes", "a.md", "main"],
  ["notes", "a.md", "i j"],
  ["notes", "a.md", "a'b(c)"],
];

const LINE_INPUTS = [
  "a.md",
  "a.md:100",
  "a.md:100:40",
  "a.md:10:x",
  "qmd://notes/a.md:5",
  "qmd://notes/a.md:5:6",
];

// Upstream CLI inline regexes (cli/qmd.ts:1261-1282).
function parseLineSuffix(input: string) {
  let m = input.match(/:(\d+):(\d+)$/);
  if (m) {
    return {
      path: input.slice(0, -m[0].length),
      from_line: parseInt(m[1], 10),
      max_lines: parseInt(m[2], 10),
    };
  }
  m = input.match(/:(\d+)$/);
  if (m) {
    return {
      path: input.slice(0, -m[0].length),
      from_line: parseInt(m[1], 10),
      max_lines: null,
    };
  }
  return { path: input, from_line: null, max_lines: null };
}

// --- DB-backed resolvers -----------------------------------------------------

const dir = await tmp.mkdtemp(path.join(os.tmpdir(), "qmd-vpath-golden-"));
const dbPath = path.join(dir, "index.sqlite");
const s = store.createStore(dbPath);
const db = s.db;

const now = new Date().toISOString();
store.upsertStoreCollection(db, "docs", {
  path: "/virtual/docs",
  pattern: "**/*.md",
});
store.upsertStoreCollection(db, "blog", {
  path: "/virtual/blog",
  pattern: "**/*.md",
});

// Two collections share `same.md` to exercise ambiguity; the odd names
// cover the LIKE-suffix boundary (`NTAX.md` must not match `SYNTAX.md`).
const DOCS = [
  { c: "docs", path: "hello.md", body: "# Hello\n" },
  { c: "docs", path: "sub/b.md", body: "# B\n" },
  { c: "docs", path: "sub/deep/c.md", body: "# C\n" },
  { c: "docs", path: "SYNTAX.md", body: "# Syntax\n" },
  { c: "blog", path: "same.md", body: "# Same blog\n" },
  { c: "docs", path: "same.md", body: "# Same docs\n" },
];
for (const d of DOCS) {
  const hash = await store.hashContent(d.body);
  store.insertContent(db, hash, d.body, now);
  store.insertDocument(db, d.c, d.path, d.path, hash, now, now);
}

const GLOB_PATTERNS = [
  "**/*.md",
  "*.md",
  "sub/**",
  "**/b.md",
  "docs/*.md",
  "qmd://docs/*.md",
  "same.md",
  "!hello.md",
  "**/*.xyz",
];

const COMMA_NAMES = [
  "hello.md",
  "docs/hello.md",
  "qmd://docs/hello.md",
  "b.md",          // path-boundary suffix match
  "NTAX.md",       // must NOT match SYNTAX.md
  "same.md",       // ambiguous across collections
  "docs/same.md",  // disambiguated
  "sub/b.md",
  "nope.md",
  "#" + (await (async () => {
    const row = db
      .prepare("SELECT hash FROM documents WHERE collection='docs' AND path='hello.md'")
      .get() as { hash: string };
    return row.hash.slice(0, 6);
  })()),
];

const out = {
  normalize: PATH_INPUTS.map((i) => [i, store.normalizeVirtualPath(i)]),
  is_virtual: PATH_INPUTS.map((i) => [i, store.isVirtualPath(i)]),
  parse: PATH_INPUTS.map((i) => [i, store.parseVirtualPath(i)]),
  build: BUILD_ARGS.map(([c, p, idx]) => [
    [c, p, idx],
    store.buildVirtualPath(c, p, idx ?? undefined),
  ]),
  docid: DOCID_INPUTS.map((i) => [
    i,
    { normalized: store.normalizeDocid(i), is_docid: store.isDocid(i) },
  ]),
  get_docid: HASHES.map((h) => [h, store.getDocid(h)]),
  line_suffix: LINE_INPUTS.map((i) => [i, parseLineSuffix(i)]),
  glob: GLOB_PATTERNS.map((p) => [
    p,
    store.matchFilesByGlob(db, p).map((m: any) => m.filepath),
  ]),
  comma_resolve: COMMA_NAMES.map((n) => [
    n,
    (() => {
      const r = store.resolveCommaListName(db, n);
      return r.ok ? { ok: r.match.virtualPath } : { err: r.error };
    })(),
  ]),
};

const fixture = new URL(
  "../crates/qmd/tests/fixtures/vpath_golden.json",
  import.meta.url,
).pathname;
await tmp.writeFile(fixture, JSON.stringify(out, null, 2) + "\n");
s.close();
await tmp.rm(dir, { recursive: true, force: true });
console.log(`wrote ${fixture}`);
