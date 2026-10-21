#!/usr/bin/env bun
/**
 * Regenerate `crates/qmd/tests/fixtures/trust_golden.json` by running
 * upstream (tobi/qmd) `src/trust.ts` over a fixed corpus of inputs —
 * the functions P3.3 ports.
 *
 * Each case records `{name, kind, input, output}` so the Rust test can
 * replay the identical call:
 *   - kind `localPath`/`expand`/`resolve`/`inside` → `input` is the arg
 *     list, `output` the return value;
 *   - kind `gated`/`digest` → `input` is `{configPath, snapshot}`
 *     (builtins are the fixture-level constant), `output` the gated
 *     surface / digest hex;
 *   - kind `storeFormat` → `output` is the exact `trusted.json` bytes
 *     upstream's `saveTrustStore` produces
 *     (`JSON.stringify(store, null, 2) + "\n"`).
 *
 * HOME is pinned so `~` expansion and `realOrResolve` fallbacks are
 * deterministic. `hookDigest`/`decideHookGate` are upstream dead code
 * and not ported (P3-D3), so they are not covered here.
 *
 * Usage:
 *   QMD_UPSTREAM=/path/to/tobi/qmd bun scripts/gen_trust_golden.ts
 *
 * Pure function calls only — deterministic, no database, no models.
 */

const upstream = process.env.QMD_UPSTREAM;
if (!upstream) {
  console.error("QMD_UPSTREAM=<path to tobi/qmd checkout> is required");
  process.exit(1);
}

const { execSync } = await import("node:child_process");
const fs = await import("node:fs/promises");

// Pin HOME before importing trust.ts — `qmdHomedir` reads it per call.
const GOLDEN_HOME = "/home/qmd-golden";
const origHome = process.env.HOME;
process.env.HOME = GOLDEN_HOME;

const {
  expandUserPath,
  gatedItems,
  isCollectionPathInsideProject,
  isLocalConfigPath,
  projectRootFromConfig,
  resolveConfigCollectionPath,
  sensitiveDigest,
} = await import(`${upstream}/src/trust.ts`);

const commit = execSync(`git -C "${upstream}" rev-parse HEAD`).toString().trim();
const version = JSON.parse(await fs.readFile(`${upstream}/package.json`, "utf8")).version;
const fixturesDir = new URL("../crates/qmd/tests/fixtures/", import.meta.url).pathname;

const PROJECT = "/home/me/project/.qmd/index.yml";
const GLOBAL = "/home/me/.config/qmd/index.yml";
const builtins = {
  embed: "hf:default/embed/model.gguf",
  rerank: "hf:default/rerank/model.gguf",
  generate: "hf:default/generate/model.gguf",
};

type Case = { name: string; kind: string; input: unknown; output: unknown };
const cases: Case[] = [];
const add = (name: string, kind: string, input: unknown, run: () => unknown) =>
  cases.push({ name, kind, input, output: run() });

// ── isLocalConfigPath ──
for (const p of [
  PROJECT,
  "/home/me/project/.qmd/index.yaml",
  GLOBAL,
  "/home/me/.config/qmd/work.yml",
  "<inline>",
  "",
  "/x/y/../project/.qmd/index.yml",
]) {
  add(`localPath ${p || "<empty>"}`, "localPath", [p], () => isLocalConfigPath(p));
}

// ── projectRootFromConfig ──
for (const p of [PROJECT, "/x/y/../project/.qmd/index.yml"]) {
  add(`projectRoot ${p}`, "projectRoot", [p], () => projectRootFromConfig(p));
}

// ── expandUserPath ──
for (const p of ["~", "~/", "~/secrets", "~\\secrets", "~user/x", "/abs/path", ""]) {
  add(`expand ${JSON.stringify(p)}`, "expand", [p], () => expandUserPath(p));
}

// ── resolveConfigCollectionPath ──
for (const p of [
  "",
  "   ",
  ".",
  "./docs",
  "docs/notes",
  "../other",
  "/etc",
  "~/secrets",
  "a/../b",
  "  ./docs  ",
]) {
  add(`resolve ${JSON.stringify(p)}`, "resolve", [PROJECT, p], () =>
    resolveConfigCollectionPath(p, PROJECT),
  );
}

// ── isCollectionPathInsideProject ──
for (const p of [
  "./docs",
  ".",
  "docs/notes",
  "a/../b",
  "..",
  "../other",
  "/etc",
  "/home/me/secrets",
  "~/secrets",
  "docs/../../etc",
  "../project-other/docs",
]) {
  add(`inside ${JSON.stringify(p)}`, "inside", [PROJECT, p], () =>
    isCollectionPathInsideProject(PROJECT, p),
  );
}

// ── gatedItems ──
const gatedInputs: [string, string, unknown][] = [
  ["gated empty", PROJECT, { hooks: [], paths: [], models: {} }],
  [
    "gated all-in-project + builtins",
    PROJECT,
    {
      hooks: [],
      paths: [{ collection: "docs", path: "./docs" }],
      models: { embed: builtins.embed },
    },
  ],
  [
    "gated hooks + outside + models",
    PROJECT,
    {
      hooks: [{ collection: "docs", command: "git pull --ff-only" }],
      paths: [
        { collection: "docs", path: "./docs" },
        { collection: "secrets", path: "/etc" },
      ],
      models: { embed: "hf:evil/x.gguf", rerank: builtins.rerank },
    },
  ],
];
for (const [name, configPath, snapshot] of gatedInputs) {
  add(name, "gated", { configPath, snapshot, builtins }, () =>
    gatedItems(configPath, snapshot as never, builtins),
  );
}

// ── sensitiveDigest ──
const digestInputs: [string, string, unknown][] = [
  [
    "digest ordering-insensitive",
    PROJECT,
    {
      hooks: [
        { collection: "b", command: "cmd-b" },
        { collection: "a", command: "cmd-a" },
      ],
      paths: [
        { collection: "b", path: "/outside/b" },
        { collection: "a", path: "/outside/a" },
      ],
      models: { generate: "hf:m/g.gguf", embed: "hf:m/e.gguf" },
    },
  ],
  [
    "digest outside path",
    PROJECT,
    { hooks: [], paths: [{ collection: "docs", path: "/etc" }], models: {} },
  ],
  [
    "digest custom model",
    PROJECT,
    { hooks: [], paths: [], models: { embed: "hf:evil/embed/x.gguf" } },
  ],
  [
    "digest ~ resolved",
    PROJECT,
    { hooks: [], paths: [{ collection: "docs", path: "~/secrets" }], models: {} },
  ],
  [
    // UTF-16 code-unit sort: U+E000 (BMP) vs U+10000 (supplementary)
    // order differs from code-point order — `byFirst` must sort by units.
    "digest utf16 sort edge",
    PROJECT,
    {
      hooks: [
        { collection: "\u{1F600}hook", command: "x" },
        { collection: "\uE000hook", command: "y" },
      ],
      paths: [],
      models: {},
    },
  ],
  [
    "digest escaping",
    PROJECT,
    {
      hooks: [{ collection: 'a"b\\c', command: 'echo "x\\y"' }],
      paths: [],
      models: {},
    },
  ],
  [
    "digest hooks+paths+models",
    PROJECT,
    {
      hooks: [{ collection: "docs", command: "git pull --ff-only" }],
      paths: [{ collection: "secrets", path: "/etc" }],
      models: { embed: "hf:evil/embed/x.gguf", rerank: "hf:evil/r/y.gguf" },
    },
  ],
  [
    "digest global config path",
    GLOBAL,
    { hooks: [], paths: [{ collection: "docs", path: "/etc" }], models: {} },
  ],
];
for (const [name, configPath, snapshot] of digestInputs) {
  add(name, "digest", { configPath, snapshot, builtins }, () =>
    sensitiveDigest(snapshot as never, configPath, builtins),
  );
}

// ── trusted.json write bytes (saveTrustStore format) ──
add("store write format", "storeFormat", null, () =>
  `${JSON.stringify(
    {
      "/home/me/project/.qmd/index.yml": {
        hooks: "abcdef0123456789",
        trustedAt: "2026-01-02T03:04:05.006Z",
      },
      "/home/me/other/.qmd/index.yaml": {
        hooks: "0011223344556677",
        trustedAt: "2026-02-03T04:05:06.007Z",
      },
    },
    null,
    2,
  )}\n`,
);

const out = {
  upstream: { repo: "tobi/qmd", commit, version },
  home: GOLDEN_HOME,
  generated: new Date().toISOString(),
  cases,
};

if (origHome === undefined) delete process.env.HOME;
else process.env.HOME = origHome;

const dest = `${fixturesDir}trust_golden.json`;
await fs.writeFile(dest, JSON.stringify(out, null, 2) + "\n");
console.log(`wrote ${dest}: ${cases.length} cases against tobi/qmd@${commit.slice(0, 12)} v${version}`);
