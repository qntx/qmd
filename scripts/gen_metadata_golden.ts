#!/usr/bin/env bun
/**
 * Regenerate `crates/qmd/tests/fixtures/metadata_golden.json` by running
 * upstream (tobi/qmd) `extractDocumentMetadata` (src/metadata.ts) — the
 * function P3.1 ports — over a fixed corpus of frontmatter cases.
 *
 * Each case records upstream's `metadata` (as `JSON.stringify` output, so
 * JS key order and number formatting are pinned byte-for-byte), the
 * extraction `error` (or null) and `extractionVersion`.
 *
 * Usage:
 *   QMD_UPSTREAM=/path/to/tobi/qmd bun scripts/gen_metadata_golden.ts
 *
 * Pure function calls only — deterministic, no database, no models.
 */

const upstream = process.env.QMD_UPSTREAM;
if (!upstream) {
  console.error("QMD_UPSTREAM=<path to tobi/qmd checkout> is required");
  process.exit(1);
}

const metadata = await import(`${upstream}/src/metadata.ts`);
const fs = await import("node:fs/promises");
const { execSync } = await import("node:child_process");

const commit = execSync(`git -C "${upstream}" rev-parse HEAD`).toString().trim();
const version = JSON.parse(await fs.readFile(`${upstream}/package.json`, "utf8")).version;
const yamlVersion = JSON.parse(
  await fs.readFile(`${upstream}/node_modules/yaml/package.json`, "utf8"),
).version;
const fixturesDir = new URL("../crates/qmd/tests/fixtures/", import.meta.url).pathname;

const doc = (yaml: string, body = "# Body\n") => `---\n${yaml}---\n${body}`;
const meta = (lines: string) =>
  doc(`qmd:\n  metadata:\n${lines.split("\n").filter(l => l !== "").map(l => `    ${l}`).join("\n")}\n`);
const aliases = (n: number) =>
  doc(`a: &a x\nlist: [${Array.from({ length: n }, () => "*a").join(", ")}]\nqmd:\n  metadata:\n    ok: 1\n`);

const cases: { name: string; path: string; content: string }[] = [
  // ── Detection ──
  { name: "no frontmatter", path: "a.md", content: "# Title\n\nbody\n" },
  { name: "no qmd namespace", path: "a.md", content: doc("title: Hello\ntags: [a]\n") },
  { name: "qmd without metadata", path: "a.md", content: doc("qmd:\n  other: 1\n") },
  { name: "empty metadata mapping", path: "a.md", content: doc("qmd:\n  metadata: {}\n") },
  { name: "empty frontmatter", path: "a.md", content: "---\n---\nbody\n" },
  { name: "scalar frontmatter", path: "a.md", content: doc("hello\n") },
  { name: "list frontmatter", path: "a.md", content: doc("- a\n- b\n") },
  { name: "bom crlf", path: "a.md", content: "\ufeff---\r\nqmd:\r\n  metadata:\r\n    status: ok\r\n---\r\nbody\r\n" },
  { name: "dots close", path: "a.md", content: "---\nqmd:\n  metadata:\n    status: ok\n...\nbody\n" },
  { name: "close at eof", path: "a.md", content: "---\nqmd:\n  metadata:\n    status: ok\n---" },
  { name: "trailing ws on fences", path: "a.md", content: "--- \t\nqmd:\n  metadata:\n    status: ok\n---  \nbody\n" },
  { name: "missing close", path: "a.md", content: "---\nqmd:\n  metadata:\n    status: ok\n" },
  { name: "four dashes open", path: "a.md", content: "----\nqmd:\n  metadata:\n    status: ok\n---\n" },
  { name: "comment after open", path: "a.md", content: "--- # c\nqmd:\n  metadata:\n    status: ok\n---\n" },
  { name: "text after close", path: "a.md", content: "---\nqmd:\n  metadata:\n    status: ok\n--- x\n---\n" },
  { name: "leading blank line", path: "a.md", content: "\n---\nqmd:\n  metadata:\n    status: ok\n---\n" },
  { name: "ext txt", path: "a.txt", content: meta("status: ok") },
  { name: "ext upper MD", path: "a.MD", content: meta("status: ok") },
  { name: "ext markdown", path: "dir/a.markdown", content: meta("status: ok") },
  { name: "ext mdx", path: "a.mdx", content: meta("status: ok") },
  { name: "no extension", path: "README", content: meta("status: ok") },
  { name: "dot in dir only", path: "dir.md/file", content: meta("status: ok") },
  { name: "dotfile md", path: ".md", content: meta("status: ok") },

  // ── Scalars ──
  { name: "basic scalars", path: "a.md", content: meta("title: Hello\ncount: 3\nratio: 0.5\ndraft: false") },
  { name: "core bools", path: "a.md", content: meta("a: true\nb: True\nc: TRUE\nd: false\ne: False\nf: FALSE") },
  { name: "yaml11 bools stay strings", path: "a.md", content: meta("a: yes\nb: no\nc: on\nd: off\ne: y\nf: n\ng: Yes\nh: ON") },
  { name: "null forms", path: "a.md", content: meta("a: ~") },
  { name: "null word", path: "a.md", content: meta("a: null") },
  { name: "null Null", path: "a.md", content: meta("a: Null") },
  { name: "empty value", path: "a.md", content: meta("a:") },
  { name: "dates stay strings", path: "a.md", content: meta("d: 2024-01-15\nt: 2024-01-15T10:00:00Z\ns: 2001-12-14 21:59:43.10 -5") },
  { name: "ints", path: "a.md", content: meta("a: 0\nb: -7\nc: +12\nd: 017\ne: 0o17\nf: 0x1F\ng: 0X1f\nh: -0") },
  { name: "underscore number", path: "a.md", content: meta("a: 1_000\nb: 0b101\nc: 1,000") },
  { name: "floats", path: "a.md", content: meta("a: 1.5\nb: 1e3\nc: 1E3\nd: 1.5e-3\ne: .5\nf: 5.\ng: -0.0\nh: 1e21\ni: 1e-7\nj: 123456789012345678901234\nk: 0.1") },
  { name: "big int precision", path: "a.md", content: meta("a: 9007199254740993\nb: 12345678901234567890") },
  { name: "inf", path: "a.md", content: meta("a: .inf") },
  { name: "neg inf", path: "a.md", content: meta("a: -.Inf") },
  { name: "nan", path: "a.md", content: meta("a: .nan") },
  { name: "overflow float", path: "a.md", content: meta("a: 1e400") },
  { name: "quoted scalars", path: "a.md", content: meta("a: \"yes\"\nb: '123'\nc: \"true\"\nd: 'null'\ne: \"\"") },
  { name: "block scalars", path: "a.md", content: meta("a: |\n  line one\n  line two\nb: >\n  folded\n  text") },
  { name: "multiline plain", path: "a.md", content: meta("a: first\n  second") },
  { name: "escapes", path: "a.md", content: meta("a: \"tab\\there\"\nb: \"\\u00e9\\U0001F600\"\nc: 'it''s'") },
  { name: "unicode", path: "a.md", content: meta("a: 中文\nb: émoji 🚀") },
  { name: "flow mapping metadata", path: "a.md", content: doc("qmd: {metadata: {a: 1, b: [x, y]}}\n") },
  { name: "comments", path: "a.md", content: meta("a: 1 # trailing\n# full line\nb: two") },

  // ── Tags ──
  { name: "tag str", path: "a.md", content: meta("a: !!str 123\nb: !!str true") },
  { name: "tag int from string", path: "a.md", content: meta("a: !!int \"5\"") },
  { name: "tag float", path: "a.md", content: meta("a: !!float 1") },
  { name: "tag bool yes", path: "a.md", content: meta("a: !!bool yes") },
  { name: "custom tag", path: "a.md", content: meta("a: !foo bar") },
  { name: "tag timestamp", path: "a.md", content: meta("a: !!timestamp 2024-01-15") },
  { name: "tag binary", path: "a.md", content: meta("a: !!binary aGVsbG8=") },

  // ── Arrays ──
  { name: "homogeneous arrays", path: "a.md", content: meta("tags: [a, b]\nnums: [1, 2.5]\nflags: [true, false]") },
  { name: "block array", path: "a.md", content: meta("tags:\n  - x\n  - y") },
  { name: "array dedupe", path: "a.md", content: meta("tags: [b, a, b, c, a]\nnums: [1, 1.0, 2, 1]\nz: [0, -0]\nf: [true, true]") },
  { name: "array mixed yaml11 strings", path: "a.md", content: meta("a: [yes, no]") },
  { name: "empty array", path: "a.md", content: meta("a: []") },
  { name: "nested array", path: "a.md", content: meta("a: [[1], [2]]") },
  { name: "mixed array", path: "a.md", content: meta("a: [1, x]") },
  { name: "array with null", path: "a.md", content: meta("a: [x, ~]") },
  { name: "array with object", path: "a.md", content: meta("a: [{k: v}]") },
  { name: "array with inf", path: "a.md", content: meta("a: [1, .inf]") },
  { name: "nested object", path: "a.md", content: meta("a:\n  b: 1") },
  { name: "null value", path: "a.md", content: meta("a: 1\nb: ~") },

  // ── Keys ──
  { name: "prototype keys", path: "a.md", content: meta("__proto__: x\nconstructor: y\ntoString: z") },
  { name: "integer-like key order", path: "a.md", content: meta("b: 1\n\"2\": 2\na: 3\n\"10\": 4\n\"01\": 5\n\"-1\": 6\n\"4294967294\": 7\n\"4294967295\": 8\n\"1.5\": 9") },
  { name: "numeric yaml key", path: "a.md", content: meta("1: one\n2.5: two") },
  { name: "bool yaml key", path: "a.md", content: meta("true: t") },
  { name: "null yaml key", path: "a.md", content: meta("~: n") },
  { name: "empty key", path: "a.md", content: meta("\"\": v") },
  { name: "control char key", path: "a.md", content: meta("\"a\\tb\": v") },
  { name: "del char key", path: "a.md", content: meta("\"a\\x7fb\": v") },
  { name: "unicode key", path: "a.md", content: meta("标签: v") },
  { name: "complex key", path: "a.md", content: meta("? [a, b]\n: v") },
  { name: "duplicate keys in metadata", path: "a.md", content: meta("a: 1\na: 2") },
  { name: "duplicate keys top level", path: "a.md", content: doc("qmd:\n  metadata:\n    a: 1\nqmd:\n  metadata:\n    a: 2\n") },
  { name: "merge key", path: "a.md", content: doc("base: &base {a: 1}\nqmd:\n  metadata:\n    <<: *base\n    b: 2\n") },
  { name: "alias scalar value", path: "a.md", content: doc("s: &s shared\nqmd:\n  metadata:\n    a: *s\n    b: [*s, other]\n") },
  { name: "alias as metadata", path: "a.md", content: doc("m: &m {x: 1}\nqmd:\n  metadata: *m\n") },
  { name: "unresolved alias", path: "a.md", content: meta("a: *nope") },
  { name: "recursive seq alias", path: "a.md", content: meta("a: &x [*x]") },
  { name: "recursive map alias", path: "a.md", content: meta("a: &x {k: *x}") },
  { name: "binary tag on map", path: "a.md", content: doc("qmd: !!binary {metadata: {a: 1}}\n") },

  // ── Namespace errors ──
  { name: "qmd scalar", path: "a.md", content: doc("qmd: hello\n") },
  { name: "qmd list", path: "a.md", content: doc("qmd: [a]\n") },
  { name: "qmd null", path: "a.md", content: doc("qmd: ~\n") },
  { name: "metadata list", path: "a.md", content: doc("qmd:\n  metadata: [a]\n") },
  { name: "metadata scalar", path: "a.md", content: doc("qmd:\n  metadata: x\n") },
  { name: "metadata null", path: "a.md", content: doc("qmd:\n  metadata:\n") },
  { name: "malformed yaml", path: "a.md", content: doc("qmd: [unclosed\n") },
  { name: "bad indentation", path: "a.md", content: doc("qmd:\n  metadata:\n    a: 1\n   b: 2\n") },
  { name: "tab indentation", path: "a.md", content: doc("qmd:\n\tmetadata:\n\t\ta: 1\n") },

  // ── Limits ──
  { name: "64 keys", path: "a.md", content: meta(Array.from({ length: 64 }, (_, i) => `k${i}: ${i}`).join("\n")) },
  { name: "65 keys", path: "a.md", content: meta(Array.from({ length: 65 }, (_, i) => `k${i}: ${i}`).join("\n")) },
  { name: "key 128 bytes", path: "a.md", content: meta(`${"k".repeat(128)}: v`) },
  { name: "key 129 bytes", path: "a.md", content: meta(`${"k".repeat(129)}: v`) },
  { name: "key multibyte 129 bytes", path: "a.md", content: meta(`${"é".repeat(64)}k: v`) },
  { name: "string 1024", path: "a.md", content: meta(`a: ${"x".repeat(1024)}`) },
  { name: "string 1025", path: "a.md", content: meta(`a: ${"x".repeat(1025)}`) },
  { name: "string emoji 512", path: "a.md", content: meta(`a: ${"😀".repeat(512)}`) },
  { name: "string emoji 513", path: "a.md", content: meta(`a: ${"😀".repeat(513)}`) },
  { name: "string cjk 1024", path: "a.md", content: meta(`a: ${"中".repeat(1024)}`) },
  { name: "array 128", path: "a.md", content: meta(`a: [${Array.from({ length: 128 }, (_, i) => i).join(", ")}]`) },
  { name: "array 129", path: "a.md", content: meta(`a: [${Array.from({ length: 129 }, (_, i) => i).join(", ")}]`) },
  { name: "array 129 dupes", path: "a.md", content: meta(`a: [${Array.from({ length: 129 }, () => 1).join(", ")}]`) },
  { name: "array string 1025", path: "a.md", content: meta(`a: [ok, ${"x".repeat(1025)}]`) },
  { name: "frontmatter at limit", path: "a.md", content: doc(`qmd:\n  metadata:\n    a: 1\npad: ${"p".repeat(64 * 1024 - 32)}\n`) },
  { name: "frontmatter over limit", path: "a.md", content: doc(`qmd:\n  metadata:\n    a: 1\npad: ${"p".repeat(64 * 1024)}\n`) },
  { name: "frontmatter multibyte over limit", path: "a.md", content: doc(`qmd:\n  metadata:\n    a: 1\npad: ${"é".repeat(32 * 1024)}\n`) },
  { name: "100 aliases", path: "a.md", content: aliases(100) },
  { name: "101 aliases", path: "a.md", content: aliases(101) },
  {
    name: "alias bomb",
    path: "a.md",
    content: doc([
      "a: &a [x, x, x, x, x, x, x, x, x, x]",
      "b: &b [*a, *a, *a, *a, *a, *a, *a, *a, *a, *a]",
      "c: &c [*b, *b, *b, *b, *b, *b, *b, *b, *b, *b]",
      "qmd:",
      "  metadata:",
      "    status: ok",
      "",
    ].join("\n")),
  },
  {
    name: "nested alias weight",
    path: "a.md",
    content: doc([
      "a: &a [x, y]",
      "b: &b [*a, *a, *a]",
      "c: [*b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b, *b]",
      "qmd:",
      "  metadata:",
      "    status: ok",
      "",
    ].join("\n")),
  },
  { name: "empty collection alias", path: "a.md", content: doc(`e: &e []\nl: [${Array.from({ length: 150 }, () => "*e").join(", ")}]\nqmd:\n  metadata:\n    ok: 1\n`) },
  // ── Parser edges (grammar acceptance must match upstream's `yaml`) ──
  { name: "flow seq under-indented", path: "a.md", content: meta("tags: [\na, b\n]") },
  { name: "flow seq indented multiline", path: "a.md", content: meta("tags: [\n  a, b\n  ]") },
  { name: "flow map under-indented", path: "a.md", content: meta("m: {\nk: v\n}") },
  { name: "implicit key over 1024", path: "a.md", content: meta(`${"k".repeat(1030)}: v`) },
  { name: "lone surrogate escape", path: "a.md", content: meta("a: \"\\ud800\"") },
  { name: "surrogate pair escape", path: "a.md", content: meta("a: \"\\ud83d\\ude00\"") },
  { name: "yaml directive", path: "a.md", content: doc("%YAML 1.1\nqmd:\n  metadata:\n    a: yes\n") },
  { name: "anchor redefinition", path: "a.md", content: doc("x: &a 1\ny: &a 2\nqmd:\n  metadata:\n    v: *a\n") },
  { name: "alias as key", path: "a.md", content: doc("k: &k key\nqmd:\n  metadata:\n    *k : v\n") },
  { name: "dup key number vs string", path: "a.md", content: meta("1: a\n\"1\": b") },
  { name: "dup key plain vs quoted", path: "a.md", content: meta("a: 1\n\"a\": 2") },
  { name: "dup key flow map", path: "a.md", content: meta("m: {a: 1, a: 2}") },
  { name: "numeric key forms", path: "a.md", content: meta("10: a\n2: b\n1e3: c\n0x1F: d\n-0: e\n.inf: f\n.nan: g") },
  { name: "tag null", path: "a.md", content: meta("a: !!null") },
  { name: "tag str on seq", path: "a.md", content: meta("a: !!str [x]") },
  { name: "tag map", path: "a.md", content: meta("a: !!map {k: v}") },
  { name: "tag seq", path: "a.md", content: meta("a: !!seq [1, 2]") },
  { name: "tag set", path: "a.md", content: meta("a: !!set {x, y}") },
  { name: "tag omap", path: "a.md", content: meta("a: !!omap [{x: 1}]") },
  { name: "tag on metadata map", path: "a.md", content: doc("qmd:\n  metadata: !!map\n    a: 1\n") },
  ...[".NaN", "+.inf", "-.5", "+.5e+3", "0.", "00", "08", "0o8", "0x", "1.2.3", "1e", "1__0", "0x_1", ".", "-", "+", "0o", "-0o7", "-0x1F", "+0x1F", "0b1", "1.0e+3", "6.02E23", "-1e-400", "1E-7", "0.000001", "100000000000000000000", "1000000000000000000000",
    "0x20000000000001", "0x1fffffffffffffffff", "0xffffffffffffffffffffffffffffffffffff", "0o1777777777777777777777",
    "0o7777777777777777777777777777777", "123456789012345678901234567890", "9007199254740993", "-9007199254740995",
    "4.35", "0.1e-310", "2.2250738585072011e-308"]
    .map(v => ({ name: `number form ${v}`, path: "a.md", content: meta(`a: ${v}`) })),
  { name: "nan in array", path: "a.md", content: meta("a: [.NaN]") },
  { name: "hash in plain", path: "a.md", content: meta("a: a#b\nb: a #b") },
  { name: "tab in value", path: "a.md", content: meta("a: x\ty") },
  { name: "trailing spaces plain", path: "a.md", content: meta("a: value   ") },
  { name: "empty flow entry", path: "a.md", content: meta("a: [x, , y]") },
  { name: "crlf block scalar", path: "a.md", content: "---\r\nqmd:\r\n  metadata:\r\n    a: |\r\n      x\r\n      y\r\n---\r\n" },
  { name: "flow seq single line long", path: "a.md", content: meta(`a: [${Array.from({ length: 40 }, (_, i) => `item${i}`).join(", ")}]`) },
  { name: "nested flow in block seq", path: "a.md", content: meta("a:\n  - [x]") },
  { name: "quoted key escapes", path: "a.md", content: meta("\"a\\u0041\": v\n'b''c': w") },
  { name: "explicit key", path: "a.md", content: meta("? k\n: v") },
  { name: "anchor on metadata key value", path: "a.md", content: meta("a: &x val\nb: *x") },
  { name: "complex map key", path: "a.md", content: meta("? {a: 1}\n: v") },
  { name: "complex quoted key", path: "a.md", content: meta("? [\"x y\", 'z']\n: v") },
  { name: "tag timestamp invalid", path: "a.md", content: meta("a: !!timestamp nope") },
  { name: "tag binary invalid", path: "a.md", content: meta("a: !!binary \"!!!\"") },
  { name: "tag merge scalar", path: "a.md", content: meta("a: !!merge <<") },
  { name: "verbatim tag", path: "a.md", content: meta("a: !<tag:yaml.org,2002:int> 5") },
  { name: "non-specific tag", path: "a.md", content: meta("a: ! 123") },
  { name: "local tag on seq", path: "a.md", content: meta("a: !foo [x]") },
  { name: "tag pairs", path: "a.md", content: meta("a: !!pairs [{x: 1}]") },
  { name: "tag int invalid", path: "a.md", content: meta("a: !!int abc") },
  { name: "close after CR only", path: "a.md", content: "---\nqmd:\n  metadata:\n    a: 1\r---\nbody\n" },
  { name: "close after LS", path: "a.md", content: "---\nqmd:\n  metadata:\n    a: 1\u2028---\nbody\n" },
  { name: "open with CR only", path: "a.md", content: "---\rqmd:\n  metadata:\n    a: 1\n---\n" },
  { name: "bom not first", path: "a.md", content: "x\ufeff---\nqmd:\n  metadata:\n    a: 1\n---\n" },
  { name: "long error truncated", path: "a.md", content: meta(`valid: 1\n"${"k".repeat(600)}x": 1`) },
  { name: "long error multibyte", path: "a.md", content: meta(`"a": "${"中".repeat(1100)}"`) },
  { name: "error whitespace collapsed", path: "a.md", content: meta(`"bad\\nkey": 1`) },
];

const results = cases.map(c => {
  const r = metadata.extractDocumentMetadata(c.content, c.path);
  return {
    name: c.name,
    path: c.path,
    content: c.content,
    metadata_json: JSON.stringify(r.metadata),
    error: r.error ?? null,
    extraction_version: r.extractionVersion,
  };
});

const names = new Set<string>();
for (const c of cases) {
  if (names.has(c.name)) throw new Error(`duplicate case name: ${c.name}`);
  names.add(c.name);
}

const out = {
  upstream_commit: commit,
  upstream_version: version,
  yaml_version: yamlVersion,
  limits: metadata.METADATA_LIMITS,
  extraction_version: metadata.METADATA_EXTRACTION_VERSION,
  cases: results,
};
await fs.writeFile(`${fixturesDir}metadata_golden.json`, `${JSON.stringify(out, null, 2)}\n`);
console.log(`wrote ${results.length} cases (upstream ${version} @ ${commit.slice(0, 8)}, yaml ${yamlVersion})`);
