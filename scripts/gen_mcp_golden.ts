#!/usr/bin/env bun
/**
 * Regenerate `crates/qmd-cli/tests/fixtures/mcp_golden.json` by driving
 * the upstream (tobi/qmd) MCP server — `bun src/cli/qmd.ts mcp` — over
 * stdio with scripted JSON-RPC sessions (P5.2).
 *
 * Isolation mirrors scripts/gen_cli_golden.ts: minimal env, HOME/
 * XDG dirs, QMD_CONFIG_DIR, INDEX_PATH, TMPDIR inside a throwaway /tmp root,
 * nothing touches the real ~/.cache/qmd or ~/.config/qmd.
 *
 * Fixture shape:
 *   { generated, upstream, corpus: { files: [{path, content}], config_yaml },
 *     sessions: [{ name, requests: [<raw request lines>],
 *                  responses: { "<id>": <response object> },
 *                  stderr, exit_code }] }
 * `stderr`/`exit_code` are recorded on every session (they matter for
 * the EOF case); `<ROOT>` stands in for the corpus root inside output.
 *
 * Usage: QMD_UPSTREAM=/path/to/tobi/qmd bun scripts/gen_mcp_golden.ts
 */
import { spawnSync, spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, realpathSync, utimesSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

const upstream = process.env.QMD_UPSTREAM;
if (!upstream) { console.error("QMD_UPSTREAM=<path to tobi/qmd checkout> is required"); process.exit(1); }
const QMD_TS = join(upstream, "src", "cli", "qmd.ts");
const OUT = new URL("../crates/qmd-cli/tests/fixtures/mcp_golden.json", import.meta.url).pathname;

const root = mkdtempSync(join(tmpdir(), "qmd-mcp-golden-"));
const corpus = join(root, "corpus");
mkdirSync(join(corpus, "docs"), { recursive: true });
mkdirSync(join(corpus, "notes"), { recursive: true });
// Fixed mtimes pin the `lastUpdated` collection order (notes newest
// → notes listed before docs) on both record and replay.
const files = [
  ["docs/readme.md", "# Readme\n\nindex search content\n", 1700000000],
  ["docs/install.md", "# Install\n\ninstall the binary then search\n", 1700000010],
  ["notes/meeting.md", "# Meeting\n\nweekly sync notes\n", 1700000060],
] as const;
for (const [rel, body, mtime] of files) {
  const p = join(corpus, rel);
  writeFileSync(p, body);
  utimesSync(p, mtime, mtime);
}
const configYaml = `global_context: Spike corpus
collections:
  docs:
    path: ${corpus}/docs
    pattern: "**/*.md"
    context:
      /: Documentation files
  notes:
    path: ${corpus}/notes
    pattern: "**/*.md"
`;
mkdirSync(join(root, "config"), { recursive: true });
writeFileSync(join(root, "config", "index.yml"), configYaml);
mkdirSync(join(root, "home"), { recursive: true });
mkdirSync(join(root, "cache"), { recursive: true });
mkdirSync(join(root, "tmp"), { recursive: true });

const env = {
  PATH: process.env.PATH,
  HOME: join(root, "home"),
  XDG_CONFIG_HOME: join(root, "config"),
  XDG_CACHE_HOME: join(root, "cache"),
  QMD_CONFIG_DIR: join(root, "config"),
  INDEX_PATH: join(root, "index.sqlite"),
  TMPDIR: join(root, "tmp"),
  TZ: "UTC",
  NO_COLOR: "1",
  QMD_DOCTOR_DEVICE_PROBE: "0",
};

const idx = spawnSync(process.execPath, [QMD_TS, "update"], { env, encoding: "utf8" });
if (idx.status !== 0) { console.error("index failed:", idx.stderr); process.exit(1); }

const META26 = { "io.modelcontextprotocol/protocolVersion": "2026-07-28",
  "io.modelcontextprotocol/clientCapabilities": {},
  "io.modelcontextprotocol/clientInfo": { name: "spike", version: "0" } };
const init25 = (v: string) => ({ jsonrpc: "2.0", id: 1, method: "initialize",
  params: { protocolVersion: v, capabilities: {}, clientInfo: { name: "spike", version: "0" } } });
const initd = { jsonrpc: "2.0", method: "notifications/initialized" };
const tc = (id: number, name: string, args: unknown) =>
  ({ jsonrpc: "2.0", id, method: "tools/call", params: { name, arguments: args } });
const tc26 = (id: number, name: string, args: unknown) =>
  ({ jsonrpc: "2.0", id, method: "tools/call", params: { name, arguments: args, _meta: META26 } });
const m26 = (id: number, method: string, params: unknown = {}) =>
  ({ jsonrpc: "2.0", id, method, params: { ...params as object, _meta: META26 } });

const sessions = [
  { name: "2025-06-18", requests: [
      init25("2025-06-18"), initd,
      { jsonrpc: "2.0", id: 2, method: "tools/list" },
      { jsonrpc: "2.0", id: 3, method: "resources/templates/list" },
      { jsonrpc: "2.0", id: 4, method: "resources/list" },
      tc(5, "status", {}),
      { jsonrpc: "2.0", id: 6, method: "resources/read", params: { uri: "qmd://docs/readme.md" } },
      { jsonrpc: "2.0", id: 7, method: "resources/read", params: { uri: "qmd://docs/nope.md" } },
      { jsonrpc: "2.0", id: 8, method: "no/such.method" },
  ]},
  { name: "2025-11-25", requests: [init25("2025-11-25"), initd, tc(2, "status", {})] },
  { name: "2026-07-28", requests: [
      m26(1, "server/discover"), m26(2, "tools/list"), tc26(3, "status", {}),
      m26(4, "resources/templates/list"),
      m26(5, "resources/read", { uri: "qmd://docs/readme.md" }),
      m26(6, "no/such.method"),
  ]},
  { name: "mixed-era", requests: [
      init25("2025-06-18"), initd,
      m26(2, "tools/list"),
      tc(3, "status", {}),
  ]},
];

async function runSession(requests: unknown[]) {
  const proc = spawn(process.execPath, [QMD_TS, "mcp"], { env, stdio: ["pipe", "pipe", "pipe"] });
  let stderr = "";
  proc.stderr.on("data", (d) => (stderr += d));
  proc.stdout.setEncoding("utf8");
  let buf = "";
  const responses: Record<string, unknown> = {};
  const waiters: (() => void)[] = [];
  proc.stdout.on("data", (d) => {
    buf += d;
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i); buf = buf.slice(i + 1);
      if (!line.trim()) continue;
      const m = JSON.parse(line);
      if (m.id !== undefined) responses[String(m.id)] = m;
      waiters.splice(0).forEach((w) => w());
    }
  });
  const want = requests.filter((r: any) => r.id !== undefined).length;
  for (const r of requests) proc.stdin.write(JSON.stringify(r) + "\n");
  const deadline = Date.now() + 30000;
  while (Object.keys(responses).length < want && Date.now() < deadline)
    await new Promise((r) => { waiters.push(r); setTimeout(r, 200); });
  proc.stdin.end();
  const exit_code = await new Promise<number | null>((r) => proc.on("close", (c) => r(c)));
  return { responses, stderr, exit_code };
}

const out = {
  generated: new Date().toISOString(),
  upstream: "tobi/qmd v2.8.3 @ 04e4dbd8",
  corpus: {
    files: files.map(([path, content, mtime]) => ({ path, content, mtime })),
    config_yaml: configYaml.replaceAll(corpus, "<ROOT>"),
  },
  sessions: [] as unknown[],
};
const roots = [root, realpathSync(root)];
const stripRoot = (x: unknown) => {
  let t = JSON.stringify(x);
  for (const r of roots) t = t.split(r).join("<ROOT>");
  return JSON.parse(t);
};
for (const s of sessions) {
  const r = await runSession(s.requests);
  out.sessions.push(stripRoot({ name: s.name, requests: s.requests, ...r }));
  console.log(`${s.name}: ${Object.keys(r.responses).length} responses, exit ${r.exit_code}`);
}
writeFileSync(OUT, JSON.stringify(out, null, 2) + "\n");
console.log("wrote", OUT);
