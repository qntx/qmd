#!/usr/bin/env bun
/**
 * Regenerate `crates/qmd/tests/fixtures/metadata_filter_golden.json` by
 * running upstream (tobi/qmd) `parseMetadataFilter` +
 * `compileMetadataFilter` (src/metadata-filter.ts) — the functions P3.2
 * ports — over a fixed corpus of filter inputs.
 *
 * Each case records, for the raw JSON input:
 *   - valid input  → `{ sql, params }` from `compileMetadataFilter(f, "d")`
 *     (params serialized via JSON.stringify so JS numbers like `1`/`1e+21`
 *     are pinned byte-for-byte), or
 *   - invalid input → `error`, the MetadataFilterError `.message`, which
 *     embeds the failing JSON path (`Invalid metadata filter at $.x: …`).
 *
 * Usage:
 *   QMD_UPSTREAM=/path/to/tobi/qmd bun scripts/gen_metadata_filter_golden.ts
 *
 * Pure function calls only — deterministic, no database, no models.
 */

const upstream = process.env.QMD_UPSTREAM;
if (!upstream) {
  console.error("QMD_UPSTREAM=<path to tobi/qmd checkout> is required");
  process.exit(1);
}

const { parseMetadataFilter, compileMetadataFilter } = await import(
  `${upstream}/src/metadata-filter.ts`
);
const fs = await import("node:fs/promises");
const { execSync } = await import("node:child_process");

const commit = execSync(`git -C "${upstream}" rev-parse HEAD`).toString().trim();
const version = JSON.parse(await fs.readFile(`${upstream}/package.json`, "utf8")).version;
const fixturesDir = new URL("../crates/qmd/tests/fixtures/", import.meta.url).pathname;

const eq = (key: string, value: unknown) => ({ key, operator: "eq", value });
const cond = (key: string, operator: string, value: unknown) => ({ key, operator, value });

const cases: { name: string; input: unknown }[] = [
  // ── Compile: groups / negation ──
  { name: "and two conditions", input: { operator: "and", operands: [eq("a", 1), eq("b", "x")] } },
  { name: "or two conditions", input: { operator: "or", operands: [eq("a", 1), eq("b", "x")] } },
  { name: "not condition", input: { operator: "not", operand: eq("a", 1) } },
  {
    name: "nested groups",
    input: {
      operator: "and",
      operands: [
        { operator: "or", operands: [eq("a", 1), cond("b", "gt", 2)] },
        { operator: "not", operand: cond("c", "exists", false) },
      ],
    },
  },
  { name: "single operand group", input: { operator: "and", operands: [eq("a", 1)] } },
  // ── Compile: comparisons ──
  { name: "eq string", input: eq("status", "published") },
  { name: "eq number", input: eq("rating", 4.5) },
  { name: "eq bool", input: eq("draft", true) },
  { name: "ne string", input: cond("status", "ne", "draft") },
  { name: "ne number", input: cond("rating", "ne", 0) },
  { name: "ne bool", input: cond("draft", "ne", false) },
  { name: "gt string", input: cond("name", "gt", "m") },
  { name: "gte number", input: cond("rating", "gte", 4) },
  { name: "lt number", input: cond("price", "lt", 100) },
  { name: "lte string", input: cond("name", "lte", "z") },
  // ── Compile: membership ──
  { name: "in strings", input: cond("status", "in", ["a", "b", "c"]) },
  { name: "in numbers", input: cond("rating", "in", [1, 2, 3]) },
  { name: "in bools", input: cond("flag", "in", [true, false]) },
  { name: "nin strings", input: cond("status", "nin", ["draft", "archived"]) },
  { name: "nin numbers", input: cond("n", "nin", [0, -1]) },
  { name: "nin bools", input: cond("flag", "nin", [true]) },
  { name: "all strings", input: cond("tags", "all", ["rust", "sqlite"]) },
  { name: "all numbers", input: cond("dims", "all", [1, 2]) },
  { name: "all bools", input: cond("flags", "all", [true, false]) },
  { name: "all single", input: cond("tags", "all", ["only"]) },
  { name: "in dedup", input: cond("tags", "in", ["a", "b", "a", "c", "b"]) },
  { name: "in neg zero", input: cond("n", "in", [0, -0, 1]) },
  { name: "exists true", input: cond("author", "exists", true) },
  { name: "exists false", input: cond("author", "exists", false) },
  // ── Key edge cases (parameterized, never interpolated) ──
  { name: "key unicode", input: eq("café☕", "x") },
  { name: "key sql injection", input: eq("k'; DROP TABLE documents; --", "x") },
  { name: "value sql injection", input: eq("k", "v' OR '1'='1") },
  { name: "value with quotes", input: cond("k", "eq", 'a"b\'c') },
  // ── Parse errors: node shape ──
  { name: "input null", input: null },
  { name: "input string", input: "eq" },
  { name: "input number", input: 42 },
  { name: "input array", input: [eq("a", 1)] },
  { name: "input bool", input: true },
  { name: "missing operator", input: { key: "a", value: 1 } },
  { name: "operator not string", input: { operator: 5, operands: [] } },
  { name: "unknown operator", input: { operator: "xor", operands: [] } },
  { name: "unknown operator fuzzy", input: { operator: "equals", key: "a", value: 1 } },
  // ── Parse errors: groups ──
  { name: "group extra prop", input: { operator: "and", operands: [eq("a", 1)], extra: 1 } },
  { name: "group missing operands", input: { operator: "and" } },
  { name: "group operands not array", input: { operator: "or", operands: "x" } },
  { name: "group empty operands", input: { operator: "and", operands: [] } },
  {
    name: "group too many operands",
    input: { operator: "or", operands: Array.from({ length: 33 }, (_, i) => eq("k", i)) },
  },
  { name: "nested error path", input: { operator: "and", operands: [eq("a", 1), { operator: "and", operands: [null] }] } },
  { name: "deep nesting", input: null /* replaced below */ },
  { name: "too many nodes", input: null /* replaced below */ },
  // ── Parse errors: negation ──
  { name: "not extra prop", input: { operator: "not", operand: eq("a", 1), operand2: eq("b", 2) } },
  { name: "not missing operand", input: { operator: "not" } },
  { name: "not null operand", input: { operator: "not", operand: null } },
  { name: "not operand error path", input: { operator: "not", operand: { operator: "and", operands: [42] } } },
  // ── Parse errors: conditions ──
  { name: "cond extra prop", input: { key: "a", operator: "eq", value: 1, extra: 2 } },
  { name: "cond missing key", input: { operator: "eq", value: 1 } },
  { name: "cond empty key", input: { key: "", operator: "eq", value: 1 } },
  { name: "cond nonstring key", input: { key: 5, operator: "eq", value: 1 } },
  { name: "cond key too long", input: { key: "k".repeat(129), operator: "eq", value: 1 } },
  { name: "cond missing value", input: { key: "a", operator: "eq" } },
  { name: "cond null value", input: { key: "a", operator: "eq", value: null } },
  { name: "cond object value", input: { key: "a", operator: "eq", value: { x: 1 } } },
  { name: "cond array value", input: { key: "a", operator: "eq", value: [1] } },
  { name: "exists nonbool", input: { key: "a", operator: "exists", value: "true" } },
  { name: "exists number", input: { key: "a", operator: "exists", value: 1 } },
  { name: "gt bool", input: { key: "a", operator: "gt", value: true } },
  { name: "lte bool", input: { key: "a", operator: "lte", value: false } },
  { name: "eq bool ok is compile", input: { key: "a", operator: "gte", value: "x" } },
  { name: "string too long", input: { key: "a", operator: "eq", value: "s".repeat(1025) } },
  { name: "number nan impossible in json", input: { key: "a", operator: "gt", value: 9007199254740993 } },
  // ── Parse errors: membership ──
  { name: "in not array", input: { key: "a", operator: "in", value: "x" } },
  { name: "in empty array", input: { key: "a", operator: "in", value: [] } },
  {
    name: "in too many",
    input: { key: "a", operator: "nin", value: Array.from({ length: 65 }, (_, i) => i) },
  },
  { name: "in mixed types", input: { key: "a", operator: "in", value: ["x", 1] } },
  { name: "in mixed num bool", input: { key: "a", operator: "all", value: [1, true] } },
  { name: "in nested array", input: { key: "a", operator: "in", value: [["x"]] } },
  { name: "in null element", input: { key: "a", operator: "in", value: ["x", null] } },
  { name: "in error path", input: { key: "a", operator: "in", value: ["ok", { bad: 1 }] } },
  { name: "in element too long", input: { key: "a", operator: "in", value: ["x".repeat(1025)] } },
];

// Cases that need programmatic construction.
cases[cases.findIndex(c => c.name === "deep nesting")].input = (() => {
  let f: unknown = eq("leaf", 1);
  for (let i = 0; i < 17; i++) f = { operator: "not", operand: f };
  return f;
})();
cases[cases.findIndex(c => c.name === "too many nodes")].input = {
  operator: "or",
  operands: Array.from({ length: 32 }, () => ({
    operator: "and",
    operands: Array.from({ length: 8 }, (_, i) => eq("k", i)),
  })),
};

const out = {
  upstream: { repo: "tobi/qmd", commit, version },
  generated: new Date().toISOString(),
  cases: cases.map(({ name, input }) => {
    const json = JSON.stringify(input);
    try {
      const filter = parseMetadataFilter(input);
      const { sql, params } = compileMetadataFilter(filter, "d");
      return { name, input: json, sql, params: JSON.stringify(params) };
    } catch (e) {
      return { name, input: json, error: (e as Error).message };
    }
  }),
};

const dest = `${fixturesDir}metadata_filter_golden.json`;
await fs.writeFile(dest, JSON.stringify(out, null, 2) + "\n");
console.log(`wrote ${dest}: ${out.cases.length} cases against tobi/qmd@${commit.slice(0, 12)} v${version}`);
