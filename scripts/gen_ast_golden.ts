/**
 * gen_ast_golden.ts — record upstream `getASTBreakPoints` output for
 * the Rust golden replay (P6.1), and upstream `chunkDocumentAsync`
 * chunk output for the `auto`-strategy golden replay (P6.2).
 *
 * Upstream `src/ast.ts` is authoritative for break points: for each
 * corpus entry we record the raw break points `{type, pos, score}` it
 * emits. `pos` is a UTF-16 code-unit index (web-tree-sitter
 * `startIndex` semantics — verified empirically; Rust `start_byte`
 * needs conversion on non-ASCII files). The Rust replay compares
 * `ast_break_points` verbatim, plus byte-level parity on ASCII files.
 *
 * Upstream `store.ts` `chunkDocumentAsync` is authoritative for
 * chunking: for each chunk-corpus entry we record `{path, content,
 * strategy, chunks: [{text, pos}]}` at the upstream defaults
 * (CHUNK_SIZE_CHARS 3600 / CHUNK_OVERLAP_CHARS 540 /
 * CHUNK_WINDOW_CHARS 800). The Rust replay compares
 * `chunk_document_with_strategy` verbatim.
 *
 * Run:
 *   QMD_UPSTREAM=/path/to/tobi/qmd bun scripts/gen_ast_golden.ts
 * Output: crates/qmd/tests/fixtures/ast_golden.json
 *         crates/qmd/tests/fixtures/ast_chunk_golden.json
 */

import { writeFileSync } from "fs";

const UPSTREAM = process.env.QMD_UPSTREAM ?? "/Users/fjh/code/github/tobi/qmd";
const { getASTBreakPoints } = await import(`${UPSTREAM}/src/ast.ts`);
const { chunkDocumentAsync } = await import(`${UPSTREAM}/src/store.ts`);

// One sample per extension (all 11), exercising every capture name in
// every language's query; plus invalid syntax, empty, non-ASCII, and
// non-code files.
const corpus: { path: string; content: string }[] = [
  // typescript (.ts/.mts/.cts share grammar+query; samples differ to
  // cover all captures: import/type/iface/enum/class/func/method/
  // export/lexical-decl funcs)
  { path: "src/auth.ts", content: `import { Database } from './db';
import type { User } from './types';

interface AuthConfig {
  secret: string;
  ttl: number;
}

type UserId = string;

export class AuthService {
  constructor(private db: Database) {}

  async authenticate(user: User, token: string): Promise<boolean> {
    const session = await this.db.findSession(token);
    return session?.userId === user.id;
  }

  validateToken(token: string): boolean {
    return token.length === 64;
  }
}

export function hashPassword(password: string): string {
  return crypto.createHash('sha256').update(password).digest('hex');
}
` },
  { path: "mod/utils.mts", content: `import { join } from 'path';

enum Kind { A, B }

export const parse = (s: string): string => s.trim();

function helper() { return 1; }
` },
  { path: "conf/settings.cts", content: `import fs from 'fs';

export const load = function(path: string) { return fs.readFileSync(path); };

class Cache { get() { return null; } }
` },
  // tsx (.tsx + .jsx)
  { path: "src/App.tsx", content: `import React from 'react';

interface Props { name: string; }

type Size = 'sm' | 'lg';

enum Variant { A, B }

export class Widget {
  render() { return <div/>; }
}

export const Label = (p: Props) => <span>{p.name}</span>;
` },
  { path: "src/View.jsx", content: `import { useState } from 'react';

export default function View() {
  return <main/>;
}

const Btn = () => <button/>;
` },
  // javascript (.js/.mjs/.cjs — typescript grammar, no iface/type/enum)
  { path: "src/util.js", content: `import { y } from 'z';

export function main() {
  class Inner {}
}

const arrow = () => "x";

function legacy() { return null; }
` },
  { path: "src/mod.mjs", content: `import * as os from 'os';

export const detect = () => os.platform();

class Platform {
  name = 'x';
  getName() { return this.name; }
}
` },
  { path: "lib/old.cjs", content: `const fs = require('fs');

function read() { return fs.readFileSync('x'); }

const write = function() {};
` },
  // python
  { path: "auth.py", content: `import os
from typing import Optional

class AuthService:
    def __init__(self, db):
        self.db = db

    async def authenticate(self, user, token):
        session = await self.db.find(token)
        return session.user_id == user.id

    def validate_token(self, token):
        return len(token) == 64

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

@decorator
def decorated_func():
    pass
` },
  // go
  { path: "auth.go", content: `package main

import "fmt"

type AuthService struct {
    db *Database
}

func (s *AuthService) Authenticate(user User) bool {
    return true
}

func HashPassword(password string) string {
    return "hash"
}
` },
  // rust
  { path: "auth.rs", content: `use std::collections::HashMap;

struct AuthService {
    db: Database,
}

impl AuthService {
    fn authenticate(&self, user: &User) -> bool {
        true
    }
}

trait Authenticatable {
    fn validate(&self) -> bool;
}

enum Role {
    Admin,
    User,
}

mod inner {
    fn f() {}
}

type UserId = u64;

fn hash_password(password: &str) -> String {
    String::new()
}
` },
  // edge cases
  { path: "broken.ts", content: `function { broken syntax %%%` },
  { path: "empty.ts", content: "" },
  // CJK — non-ASCII: byte offsets diverge from UTF-16 indices
  { path: "unicode.ts", content: `// 日本語のコメント
export function emoji() { return "🎉"; }

class After {
  x = "éèê";
}
` },
  // emoji/astral python
  { path: "emoji.py", content: `# comment with emoji 🎉🎊
def greet():
    return "日本語"

class Ünïcödé:
    pass
` },
  // non-code
  { path: "readme.md", content: "# Hello World" },
  { path: "data.xyz", content: "not code" },
  { path: "qmd://virtual/doc.py", content: `def v(): pass` },
];

const cases = [] as any[];
for (const entry of corpus) {
  const points = await getASTBreakPoints(entry.content, entry.path);
  cases.push({ path: entry.path, content: entry.content, points });
  console.log(`${entry.path}: ${points.length} break points`);
}

const out = {
  generated: new Date().toISOString().slice(0, 10),
  upstream: "tobi/qmd src/ast.ts getASTBreakPoints (pos = UTF-16 code-unit index)",
  cases,
};
writeFileSync(
  "/Users/fjh/code/github/qntx/qmd/crates/qmd/tests/fixtures/ast_golden.json",
  JSON.stringify(out, null, 2) + "\n",
);
console.log("wrote crates/qmd/tests/fixtures/ast_golden.json");

// =============================================================================
// P6.2 — chunkDocumentAsync corpus (auto-strategy chunk golden)
// =============================================================================
//
// One ASCII code file per grammar language, each sized to produce >= 3
// chunks at the defaults; one markdown and one unknown-extension file
// recorded under BOTH strategies (auto must equal regex — no AST break
// points exist for them); one CJK code file under auto (UTF-16 `pos`
// units diverge from byte offsets there).

function tsHandlers(n: number): string {
  const parts: string[] = [];
  for (let i = 0; i < n; i++) {
    parts.push(`
export function handler${i}(req: Request, res: Response): void {
  const startTime = Date.now();
  const userId = req.params.userId;
  const sessionToken = req.headers.authorization;

  if (!userId || !sessionToken) {
    res.status(400).json({ error: "Missing required parameters" });
    return;
  }

  console.log(\`Processing request ${i} for user \${userId}\`);
  const result = processBusinessLogic${i}(userId, sessionToken);

  const elapsed = Date.now() - startTime;
  res.json({ data: result, processingTimeMs: elapsed });
}
`);
  }
  return parts.join("\n");
}

function pyModule(n: number): string {
  const parts: string[] = [`import os`, `from typing import Optional`];
  for (let i = 0; i < n; i++) {
    parts.push(`
class Handler${i}:
    def __init__(self, db):
        self.db = db
        self.counter = ${i}

    async def handle(self, request):
        session = await self.db.find(request.token)
        return self.render(session, ${i})

    def render(self, session, tag):
        return {"tag": tag, "counter": self.counter}

def helper_${i}(value: int) -> int:
    return value * ${i} + 1
`);
  }
  return parts.join("\n");
}

function goModule(n: number): string {
  const parts: string[] = [`package main`, ``, `import "fmt"`];
  for (let i = 0; i < n; i++) {
    parts.push(`
type Handler${i} struct {
    db     *Database
    counter int
}

func (h *Handler${i}) Handle(req *Request) (*Response, error) {
    session, err := h.db.Find(req.Token)
    if err != nil {
        return nil, fmt.Errorf("handler ${i}: %w", err)
    }
    return h.Render(session, ${i}), nil
}

func helper${i}(value int) int {
    return value*${i} + 1
}
`);
  }
  return parts.join("\n");
}

function rsModule(n: number): string {
  const parts: string[] = [`use std::collections::HashMap;`];
  for (let i = 0; i < n; i++) {
    parts.push(`
pub struct Handler${i} {
    db: Database,
    counter: u32,
}

impl Handler${i} {
    pub fn handle(&self, req: &Request) -> Result<Response, Error> {
        let session = self.db.find(&req.token)?;
        Ok(self.render(session, ${i}))
    }

    fn render(&self, session: Session, tag: u32) -> Response {
        Response::json(session.with_tag(tag + ${i}))
    }
}

fn helper_${i}(value: u32) -> u32 {
    value * ${i} + 1
}
`);
  }
  return parts.join("\n");
}

function jsModule(n: number): string {
  const parts: string[] = [`import { open } from 'fs';`];
  for (let i = 0; i < n; i++) {
    parts.push(`
export function handler${i}(req, res) {
  const startTime = Date.now();
  const userId = req.params.userId;

  if (!userId) {
    res.status(400).json({ error: "Missing userId" });
    return;
  }

  const result = process${i}(userId);
  res.json({ data: result, elapsed: Date.now() - startTime });
}

const arrow${i} = () => ${i} * 2;
`);
  }
  return parts.join("\n");
}

function tsxModule(n: number): string {
  const parts: string[] = [`import React from 'react';`, ``, `interface Props { id: number; }`];
  for (let i = 0; i < n; i++) {
    parts.push(`
export class Widget${i} extends React.Component<Props> {
  state = { count: ${i} };

  render() {
    return <section id={this.props.id}><p>widget ${i}</p></section>;
  }
}

export const Label${i} = (p: Props) => <span>{p.id}</span>;
`);
  }
  return parts.join("\n");
}

function mdDoc(n: number): string {
  const sections: string[] = [];
  for (let i = 0; i < n; i++) {
    sections.push(`# Section ${i}\n\n${"Lorem ipsum dolor sit amet. ".repeat(40)}\n`);
  }
  return sections.join("\n");
}

function plainText(n: number): string {
  const paras: string[] = [];
  for (let i = 0; i < n; i++) {
    paras.push(
      `Paragraph ${i} of an unknown file type. ` +
      `This text is plain prose with no recognized extension, ` +
      `so only regex break points apply. ` +
      `Sentence ${i}.a keeps it going. Sentence ${i}.b adds more.\n`,
    );
  }
  return paras.join("\n");
}

function cjkTs(n: number): string {
  const parts: string[] = [`// 日本語のコメント — CJK 前のファイル`, `import { x } from './y';`];
  for (let i = 0; i < n; i++) {
    parts.push(`
// セクション ${i} の説明です
export function handler${i}(input: string): string {
  const message = "こんにちは世界 ${i} — " + input;
  const detail = \`処理 ${i}: \${message} の結果を返します\`;
  if (input.length === 0) {
    return "空の入力 🎉 " + detail;
  }
  return message + detail;
}
`);
  }
  return parts.join("\n");
}

const chunkCorpus: { path: string; content: string; strategies: ("auto" | "regex")[] }[] = [
  { path: "src/handlers.ts", content: tsHandlers(30), strategies: ["auto"] },
  { path: "src/widgets.tsx", content: tsxModule(30), strategies: ["auto"] },
  { path: "src/util.js", content: jsModule(30), strategies: ["auto"] },
  { path: "pkg/handlers.py", content: pyModule(25), strategies: ["auto"] },
  { path: "pkg/handlers.go", content: goModule(25), strategies: ["auto"] },
  { path: "src/handlers.rs", content: rsModule(25), strategies: ["auto"] },
  { path: "docs/readme.md", content: mdDoc(15), strategies: ["auto", "regex"] },
  { path: "data/notes.xyz", content: plainText(60), strategies: ["auto", "regex"] },
  { path: "src/日本語.ts", content: cjkTs(30), strategies: ["auto"] },
];

const chunkCases = [] as any[];
for (const entry of chunkCorpus) {
  for (const strategy of entry.strategies) {
    const chunks = await chunkDocumentAsync(
      entry.content,
      undefined, undefined, undefined, // CHUNK_SIZE/OVERLAP/WINDOW_CHARS defaults
      entry.path,
      strategy,
    );
    chunkCases.push({
      path: entry.path,
      content: entry.content,
      strategy,
      chunks: chunks.map((c: any) => ({ text: c.text, pos: c.pos })),
    });
    console.log(
      `${entry.path} [${strategy}]: ${entry.content.length} chars → ${chunks.length} chunks`,
    );
  }
}

const chunkOut = {
  generated: new Date().toISOString().slice(0, 10),
  upstream:
    "tobi/qmd store.ts chunkDocumentAsync (defaults 3600/540/800 chars; pos = UTF-16 code-unit index)",
  cases: chunkCases,
};
writeFileSync(
  "/Users/fjh/code/github/qntx/qmd/crates/qmd/tests/fixtures/ast_chunk_golden.json",
  JSON.stringify(chunkOut, null, 2) + "\n",
);
console.log("wrote crates/qmd/tests/fixtures/ast_chunk_golden.json");
