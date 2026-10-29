/**
 * gen_ast_golden.ts — record upstream `getASTBreakPoints` output for
 * the Rust golden replay (P6.1).
 *
 * Upstream `src/ast.ts` is authoritative: for each corpus entry we
 * record the raw break points `{type, pos, score}` it emits. `pos` is
 * a UTF-16 code-unit index (web-tree-sitter `startIndex` semantics —
 * verified empirically; Rust `start_byte` needs conversion on
 * non-ASCII files). The Rust replay compares `ast_break_points`
 * verbatim, plus byte-level parity on ASCII files.
 *
 * Run:
 *   QMD_UPSTREAM=/path/to/tobi/qmd bun scripts/gen_ast_golden.ts
 * Output: crates/qmd/tests/fixtures/ast_golden.json
 */

import { writeFileSync } from "fs";

const UPSTREAM = process.env.QMD_UPSTREAM ?? "/Users/fjh/code/github/tobi/qmd";
const { getASTBreakPoints } = await import(`${UPSTREAM}/src/ast.ts`);

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
