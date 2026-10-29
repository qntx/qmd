#!/usr/bin/env bun
/**
 * Regenerate `crates/qmd-cli/tests/fixtures/mcp_golden.json` by driving
 * the upstream (tobi/qmd) MCP server — `bun src/cli/qmd.ts mcp` — over
 * stdio with scripted JSON-RPC sessions (P5.2), and
 * `qmd mcp --http --port N` with scripted HTTP requests (P5.3).
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
import { spawn, spawnSync } from "node:child_process";
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
mkdirSync(join(corpus, "notes", "External Podcast"), { recursive: true });
// Fixed mtimes pin the `lastUpdated` collection order (notes newest
// → notes listed before docs) on both record and replay.
const files = [
  ["docs/readme.md", "# Readme\n\nindex search content\n", 1700000000],
  ["docs/install.md", "# Install\n\ninstall the binary then search\n", 1700000010],
  ["notes/meeting.md", "# Meeting\n\nweekly sync notes\n", 1700000060],
  ["notes/External Podcast/2023 April - Interview.md", "# Interview\n\npodcast transcript\n", 1700000070],
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

// Learn docs/readme.md's docid from upstream `qmd get` (docids are
// content-hash deterministic, so ours matches byte-for-byte).
const getOut = spawnSync(process.execPath, [QMD_TS, "get", "docs/readme.md"], { env, cwd: corpus, encoding: "utf8" });
const docid = (getOut.stdout || "").match(/#[0-9a-f]{6}/)?.[0] || "#000000";

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
      tc(9, "query", { searches: [{ type: "lex", query: "index" }], rerank: false }),
      tc(10, "query", { searches: [{ type: "lex", query: "install" }], rerank: false, collections: ["docs"] }),
      tc(11, "query", {}),
      tc(12, "query", { query: "index", searches: [{ type: "lex", query: "x" }] }),
      tc(13, "query", { searches: Array.from({ length: 11 }, () => ({ type: "lex", query: "x" })) }),
      tc(14, "query", { searches: [{ type: "bogus", query: "x" }] }),
      tc(16, "get", { file: "docs/readme.md" }),
      tc(17, "get", { file: "docs/readme.md:2" }),
      tc(18, "get", { file: "docs/readme.md:1:2" }),
      tc(19, "get", { file: "docs/readme.md", fromLine: 2, maxLines: 1 }),
      tc(20, "get", { file: "docs/nope.md" }),
      tc(21, "get", { file: 5 }),
      tc(22, "get", {}),
      tc(23, "multi_get", { pattern: "docs/*.md" }),
      tc(24, "multi_get", { pattern: "docs/*.md", maxLines: 1 }),
      tc(25, "multi_get", { pattern: "docs/readme.md,notes/meeting.md" }),
      tc(26, "multi_get", { pattern: "nope/*.md" }),
      tc(27, "get", { file: docid }),
      tc(28, "get", { file: docid.slice(1) }),
      tc(29, "get", { file: 1, maxLines: "x" }),
      tc(30, "query", { searches: [{ type: "bad", query: 1 }], limit: "x" }),
      tc(31, "multi_get", { pattern: 5, maxBytes: "x" }),
      // resources/read — encoded paths + malformed escapes
      // (mcp.test.ts:647-742); JS decodeURIComponent throws URIError on
      // the malformed forms.
      { jsonrpc: "2.0", id: 32, method: "resources/read", params: { uri: "qmd://notes%2Fmeeting.md" } },
      { jsonrpc: "2.0", id: 33, method: "resources/read", params: { uri: "qmd://notes/External%20Podcast/2023%20April%20-%20Interview.md" } },
      { jsonrpc: "2.0", id: 34, method: "resources/read", params: { uri: "qmd://notes%252Fmeeting.md" } },
      { jsonrpc: "2.0", id: 35, method: "resources/read", params: { uri: "qmd://meeting.md" } },
      { jsonrpc: "2.0", id: 36, method: "resources/read", params: { uri: "qmd://docs%ZZ.md" } },
      { jsonrpc: "2.0", id: 37, method: "resources/read", params: { uri: "qmd://docs%2.md" } },
      { jsonrpc: "2.0", id: 38, method: "resources/read", params: { uri: "qmd://%FF.md" } },
      { jsonrpc: "2.0", id: 39, method: "resources/read", params: { uri: "qmd://%E2%82.md" } },
      // get — bare filename (mcp.test.ts:463) + near-miss suggestions (:476)
      tc(40, "get", { file: "readme.md" }),
      tc(41, "get", { file: "readmee.md" }),
      // multi_get — missing file in comma list (:541) + maxBytes skip (:555)
      tc(42, "multi_get", { pattern: "docs/readme.md,notes/missing.md" }),
      tc(43, "multi_get", { pattern: "docs/*.md", maxBytes: 32 }),
      // query edge inputs (mcp.test.ts:785-807) — lex-only, no models
      tc(44, "query", { searches: [{ type: "lex", query: "" }], rerank: false }),
      tc(45, "query", { searches: [{ type: "lex", query: "C++ & \"quotes\" (parens)" }], rerank: false }),
      tc(46, "query", { searches: [{ type: "lex", query: "日本語 café" }], rerank: false }),
      tc(47, "query", { searches: [{ type: "lex", query: "index ".repeat(166) + "x" }], rerank: false }),
      tc(48, "query", { searches: [{ type: "lex", query: "the a an of" }], rerank: false }),
  ]},
  { name: "2025-11-25", requests: [init25("2025-11-25"), initd, tc(2, "status", {})] },
  { name: "2026-07-28", requests: [
      m26(1, "server/discover"), m26(2, "tools/list"), tc26(3, "status", {}),
      m26(4, "resources/templates/list"),
      m26(5, "resources/read", { uri: "qmd://docs/readme.md" }),
      m26(6, "no/such.method"),
      tc26(7, "query", { searches: [{ type: "lex", query: "index" }], rerank: false }),
      tc26(8, "get", { file: "docs/readme.md" }),
      tc26(9, "multi_get", { pattern: "docs/*.md" }),
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

// ---------------------------------------------------------------------------
// HTTP transport (P5.3): `qmd mcp --http --port N` + scripted fetch requests.
// ---------------------------------------------------------------------------

const HTTP_ACCEPT = { Accept: "application/json, text/event-stream" };
const H26 = { ...HTTP_ACCEPT, "MCP-Protocol-Version": "2026-07-28" };
const h26 = (method: string, extra: Record<string, string> = {}) =>
  ({ ...H26, "Mcp-Method": method, ...extra });
const tc26h = (id: number, name: string, args: unknown) => ({
  body: tc26(id, name, args), headers: h26("tools/call", { "Mcp-Name": name }),
});
const m26h = (id: number, method: string, params: unknown = {}) => ({
  body: m26(id, method, params), headers: h26(method),
});

type HttpCase = {
  name: string;
  method?: string;
  path?: string;
  headers?: Record<string, string>;
  body?: unknown | string;
  env?: Record<string, string>;
};
const httpCases: HttpCase[] = [
  { name: "health" },
  { name: "health-post", method: "POST" },
  { name: "health-trailing", path: "/health/" },
  { name: "init-2025", method: "POST", path: "/mcp", headers: HTTP_ACCEPT,
    body: init25("2025-06-18") },
  { name: "init-2026-named", method: "POST", path: "/mcp", headers: HTTP_ACCEPT,
    body: init25("2026-07-28") },
  { name: "tools-list-2025", method: "POST", path: "/mcp", headers: HTTP_ACCEPT,
    body: { jsonrpc: "2.0", id: 2, method: "tools/list" } },
  { name: "call-status-2025", method: "POST", path: "/mcp", headers: HTTP_ACCEPT,
    body: tc(3, "status", {}) },
  { name: "res-read-2025", method: "POST", path: "/mcp", headers: HTTP_ACCEPT,
    body: { jsonrpc: "2.0", id: 5, method: "resources/read",
            params: { uri: "qmd://docs/readme.md" } } },
  { name: "discover-2026", method: "POST", path: "/mcp", ...m26h(6, "server/discover") },
  { name: "tools-list-2026", method: "POST", path: "/mcp", ...m26h(7, "tools/list") },
  { name: "call-status-2026", method: "POST", path: "/mcp", ...tc26h(8, "status", {}) },
  { name: "call-query-2026", method: "POST", path: "/mcp",
    ...tc26h(9, "query", { searches: [{ type: "lex", query: "index" }], rerank: false }) },
  { name: "call-get-2025", method: "POST", path: "/mcp", headers: HTTP_ACCEPT,
    body: tc(90, "get", { file: "docs/readme.md" }) },
  { name: "res-read-2026", method: "POST", path: "/mcp",
    ...m26h(10, "resources/read", { uri: "qmd://docs/readme.md" }) },
  { name: "no-method-hdr", method: "POST", path: "/mcp", headers: H26,
    body: m26(11, "tools/list") },
  { name: "method-hdr-mismatch", method: "POST", path: "/mcp",
    headers: h26("resources/list"), body: m26(12, "tools/list") },
  { name: "name-hdr-mismatch", method: "POST", path: "/mcp",
    headers: h26("tools/call", { "Mcp-Name": "query" }),
    body: tc26(13, "status", {}) },
  { name: "name-hdr-missing", method: "POST", path: "/mcp",
    headers: h26("tools/call"), body: tc26(14, "status", {}) },
  { name: "meta-no-version-hdr", method: "POST", path: "/mcp",
    headers: { Accept: HTTP_ACCEPT.Accept, "Mcp-Method": "tools/list" },
    body: m26(15, "tools/list") },
  { name: "version-hdr-no-meta", method: "POST", path: "/mcp",
    headers: h26("tools/list"),
    body: { jsonrpc: "2.0", id: 16, method: "tools/list", params: {} } },
  { name: "version-mismatch", method: "POST", path: "/mcp",
    headers: h26("tools/list"),
    body: { jsonrpc: "2.0", id: 17, method: "tools/list",
            params: { _meta: { ...META26,
              "io.modelcontextprotocol/protocolVersion": "2025-11-25" } } } },
  { name: "version-unsupported", method: "POST", path: "/mcp",
    headers: h26("server/discover", { "MCP-Protocol-Version": "1900-01-01" }),
    body: { jsonrpc: "2.0", id: 18, method: "server/discover",
            params: { _meta: { ...META26,
              "io.modelcontextprotocol/protocolVersion": "1900-01-01" } } } },
  { name: "initialized-2025", method: "POST", path: "/mcp", headers: HTTP_ACCEPT,
    body: initd },
  { name: "notification-2026", method: "POST", path: "/mcp",
    headers: h26("notifications/cancelled"),
    body: { jsonrpc: "2.0", method: "notifications/cancelled", params: { _meta: META26 } } },
  { name: "bad-json", method: "POST", path: "/mcp", headers: HTTP_ACCEPT,
    body: "{not json" },
  { name: "batch-array", method: "POST", path: "/mcp", headers: HTTP_ACCEPT,
    body: [tc(19, "status", {})] },
  { name: "scalar-body", method: "POST", path: "/mcp", headers: HTTP_ACCEPT,
    body: "42" },
  { name: "mcp-get", method: "GET", path: "/mcp" },
  { name: "mcp-delete", method: "DELETE", path: "/mcp" },
  { name: "mcp-put", method: "PUT", path: "/mcp", headers: HTTP_ACCEPT,
    body: tc(20, "status", {}) },
  { name: "accept-json-only", method: "POST", path: "/mcp",
    headers: { Accept: "application/json" }, body: tc(21, "status", {}) },
  { name: "accept-star", method: "POST", path: "/mcp",
    headers: { Accept: "*/*" }, body: tc(22, "status", {}) },
  { name: "accept-missing", method: "POST", path: "/mcp",
    headers: { Accept: "" }, body: tc(23, "status", {}) },
  { name: "ctype-text", method: "POST", path: "/mcp",
    headers: { ...HTTP_ACCEPT, "Content-Type": "text/plain" },
    body: tc(24, "status", {}) },
  { name: "query-hit", method: "POST", path: "/query",
    body: { searches: [{ type: "lex", query: "index" }], rerank: false } },
  { name: "search-alias", method: "POST", path: "/search",
    body: { searches: [{ type: "lex", query: "weekly" }], rerank: false } },
  { name: "query-vec-ftype", method: "POST", path: "/query",
    body: { searches: [{ type: "bogus", query: "index" }], rerank: false } },
  { name: "query-collections", method: "POST", path: "/query",
    body: { searches: [{ type: "lex", query: "install" }], collections: ["docs"], rerank: false } },
  { name: "query-empty-collections", method: "POST", path: "/query",
    body: { searches: [{ type: "lex", query: "index" }], collections: [], rerank: false } },
  { name: "query-limit-opts", method: "POST", path: "/query",
    body: { searches: [{ type: "lex", query: "a" }],
            limit: 1, minScore: 0, intent: "find docs", rerank: false } },
  { name: "query-bad-json", method: "POST", path: "/query", body: "{nope" },
  { name: "query-array-body", method: "POST", path: "/query", body: [1, 2] },
  { name: "query-scalar-body", method: "POST", path: "/query", body: "\"x\"" },
  { name: "query-no-searches", method: "POST", path: "/query",
    body: { limit: 1 } },
  { name: "query-searches-obj", method: "POST", path: "/query",
    body: { searches: { type: "lex" } } },
  { name: "query-searches-empty", method: "POST", path: "/query",
    body: { searches: [] } },
  { name: "query-filter-str", method: "POST", path: "/query",
    body: { searches: [{ type: "lex", query: "index" }], filter: "x", rerank: false } },
  { name: "query-filter-bad", method: "POST", path: "/query",
    body: { searches: [{ type: "lex", query: "index" }],
            filter: { key: { $bad: 1 } }, rerank: false } },
  { name: "query-searches-null-elem", method: "POST", path: "/query",
    body: { searches: [null] } },
  { name: "query-get", method: "GET", path: "/query",
    headers: { "Content-Type": "application/json" } },
  { name: "unknown-path", path: "/nope" },
  { name: "unknown-post", method: "POST", path: "/" },
  { name: "origin-evil", method: "POST", path: "/query",
    headers: { Origin: "http://evil.example", "Content-Type": "application/json" },
    body: { searches: [] } },
  { name: "origin-loopback", method: "POST", path: "/query",
    headers: { Origin: "http://localhost:9999", "Content-Type": "application/json" },
    body: { searches: [] } },
  { name: "origin-localhost-suffix", method: "POST", path: "/query",
    headers: { Origin: "http://app.localhost:81", "Content-Type": "application/json" },
    body: { searches: [] } },
  { name: "host-evil", method: "POST", path: "/query",
    headers: { Host: "evil.example", "Content-Type": "application/json" },
    body: { searches: [] } },
  { name: "host-allowed-bind", path: "/health",
    headers: { Host: "127.0.0.1:1" } },
];

async function runHttp(): Promise<unknown> {
  const port = 18200 + (process.pid % 2000);
  const proc = spawn(process.execPath, [QMD_TS, "mcp", "--http", "--port", String(port)],
    { env, stdio: ["ignore", "pipe", "pipe"] });
  let stderr = "";
  proc.stderr.on("data", (d) => (stderr += d));
  const deadline = Date.now() + 30000;
  while (!stderr.includes("/mcp") && Date.now() < deadline)
    await new Promise((r) => setTimeout(r, 100));
  const base = `http://localhost:${port}`;
  const results = [];
  for (const c of httpCases) {
    const method = c.method ?? "GET";
    const path = c.path ?? "/health";
    const headers: Record<string, string> =
      method === "POST" && !("Content-Type" in (c.headers ?? {}))
        ? { "Content-Type": "application/json", ...(c.headers ?? {}) }
        : (c.headers ?? {});
    const res = await fetch(`${base}${path}`, {
      method, headers,
      ...(method !== "GET" && method !== "HEAD" && c.body !== undefined
        ? { body: typeof c.body === "string" ? c.body : JSON.stringify(c.body) }
        : {}),
    });
    const text = await res.text();
    const keep = ["content-type", "allow", "cache-control", "x-accel-buffering",
      "connection", "mcp-session-id"] as const;
    const recorded: Record<string, string | null> = {};
    for (const k of keep) recorded[k] = res.headers.get(k);
    const sse = [...text.matchAll(/^data: (.*)$/gm)].map((m) => JSON.parse(m[1]));
    results.push(stripRoot({
      name: c.name, method, path, headers: c.headers ?? {},
      body: c.body ?? null,
      response: { status: res.status, headers: recorded,
                  body: text, sse },
    }));
    console.log(`${c.name}: ${res.status}`);
  }
  // SIGTERM shutdown: stderr breadcrumb + exit 0.
  proc.kill("SIGTERM");
  const exit = await new Promise<number | null>((r) => proc.on("close", (c) => r(c)));
  await new Promise((r) => setTimeout(r, 500));
  return { cases: results, sigterm_stderr: stderr.split("\n").pop() ?? "",
           stderr, exit_code: exit };
}
(out as any).http = await runHttp();
writeFileSync(OUT, JSON.stringify(out, null, 2) + "\n");
console.log("wrote", OUT);
