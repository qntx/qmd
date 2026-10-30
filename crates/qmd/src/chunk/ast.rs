//! Tree-sitter AST break points — upstream `src/ast.ts`.
//!
//! Position units: web-tree-sitter's `startIndex` (ast.ts:305) is a
//! **UTF-16 code-unit index** — the golden recording puts a CJK file's
//! `export` at 12 (UTF-16), not 28 (UTF-8 bytes). Rust
//! `Node::start_byte` gives byte offsets, so [`ast_break_points`]
//! converts each to a UTF-16 index; the result matches upstream
//! verbatim (no deviation). `break_points_utf8` keeps the raw byte
//! view internally.
//!
//! Grammars are statically linked (no wasm load), so upstream's
//! grammar-load-failure branch (`formatGrammarLoadError`, ast.ts:75)
//! is unreachable and not ported. `extractSymbols` is not ported —
//! upstream marks it a Phase-2 stub with no callers (D3).

use std::collections::HashMap;
use std::sync::OnceLock;

use tree_sitter::{Parser, Query, QueryCursor, StreamingIterator as _};

use super::BreakPoint;

/// Supported code languages — upstream `SupportedLanguage` (ast.ts:34).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Language {
    /// `.ts`, `.mts`, `.cts` (`tree-sitter-typescript.wasm`).
    TypeScript,
    /// `.tsx`, `.jsx` (`tree-sitter-tsx.wasm`).
    Tsx,
    /// `.js`, `.mjs`, `.cjs` — upstream reuses the TypeScript grammar
    /// (ast.ts:69).
    JavaScript,
    /// `.py` (`tree-sitter-python.wasm`).
    Python,
    /// `.go` (`tree-sitter-go.wasm`).
    Go,
    /// `.rs` (`tree-sitter-rust.wasm`).
    Rust,
}

impl Language {
    /// Every supported language in upstream `GRAMMAR_MAP` order
    /// (ast.ts:66-74) — the `Languages:` list in `qmd status`
    /// (qmd.ts:610); statically linked grammars are always available.
    pub const ALL: [Self; 6] = [
        Self::TypeScript,
        Self::Tsx,
        Self::JavaScript,
        Self::Python,
        Self::Go,
        Self::Rust,
    ];

    /// Upstream `SupportedLanguage` string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TypeScript => "typescript",
            Self::Tsx => "tsx",
            Self::JavaScript => "javascript",
            Self::Python => "python",
            Self::Go => "go",
            Self::Rust => "rust",
        }
    }
}

/// `detectLanguage` (ast.ts:54-58) — extension → language, mirroring
/// node `path.extname`: the last `.` after the last `/`, lowercase;
/// `.ts`-style dotfiles have no extension; `file.` yields `"."`.
///
/// Works on `qmd://` virtual paths — only the basename matters.
#[must_use]
pub fn detect_language(path: &str) -> Option<Language> {
    let name = path.rsplit('/').next().unwrap_or(path);
    let dot = name.rfind('.')?;
    if dot == 0 {
        return None;
    }
    Some(match name[dot..].to_ascii_lowercase().as_str() {
        ".ts" | ".mts" | ".cts" => Language::TypeScript,
        ".tsx" | ".jsx" => Language::Tsx,
        ".js" | ".mjs" | ".cjs" => Language::JavaScript,
        ".py" => Language::Python,
        ".go" => Language::Go,
        ".rs" => Language::Rust,
        _ => return None,
    })
}

// LANGUAGE_QUERIES — verbatim copy of ast.ts:94-151.

const Q_TYPESCRIPT: &str = r"
    (export_statement) @export
    (class_declaration) @class
    (function_declaration) @func
    (method_definition) @method
    (interface_declaration) @iface
    (type_alias_declaration) @type
    (enum_declaration) @enum
    (import_statement) @import
    (lexical_declaration (variable_declarator value: (arrow_function))) @func
    (lexical_declaration (variable_declarator value: (function_expression))) @func
  ";

const Q_TSX: &str = Q_TYPESCRIPT;

const Q_JAVASCRIPT: &str = r"
    (export_statement) @export
    (class_declaration) @class
    (function_declaration) @func
    (method_definition) @method
    (import_statement) @import
    (lexical_declaration (variable_declarator value: (arrow_function))) @func
    (lexical_declaration (variable_declarator value: (function_expression))) @func
  ";

const Q_PYTHON: &str = r"
    (class_definition) @class
    (function_definition) @func
    (decorated_definition) @decorated
    (import_statement) @import
    (import_from_statement) @import
  ";

const Q_GO: &str = r"
    (type_declaration) @type
    (function_declaration) @func
    (method_declaration) @method
    (import_declaration) @import
  ";

const Q_RUST: &str = r"
    (struct_item) @struct
    (impl_item) @impl
    (function_item) @func
    (trait_item) @trait
    (enum_item) @enum
    (use_declaration) @import
    (type_item) @type
    (mod_item) @mod
  ";

/// `SCORE_MAP` (ast.ts:158-172); unlisted captures score 20.
fn capture_score(name: &str) -> u32 {
    match name {
        "class" | "iface" | "struct" | "trait" | "impl" | "mod" => 100,
        "export" | "func" | "method" | "decorated" => 90,
        "type" | "enum" => 80,
        "import" => 60,
        _ => 20,
    }
}

fn ts_language(lang: Language) -> tree_sitter::Language {
    match lang {
        Language::TypeScript | Language::JavaScript => {
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()
        }
        Language::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
        Language::Python => tree_sitter_python::LANGUAGE.into(),
        Language::Go => tree_sitter_go::LANGUAGE.into(),
        Language::Rust => tree_sitter_rust::LANGUAGE.into(),
    }
}

const fn query_source(lang: Language) -> &'static str {
    match lang {
        Language::TypeScript => Q_TYPESCRIPT,
        Language::Tsx => Q_TSX,
        Language::JavaScript => Q_JAVASCRIPT,
        Language::Python => Q_PYTHON,
        Language::Go => Q_GO,
        Language::Rust => Q_RUST,
    }
}

/// Compiled-query cache — upstream `queryCache` (ast.ts:193).
fn query_for(lang: Language) -> Option<&'static Query> {
    static TYPESCRIPT: OnceLock<Option<Query>> = OnceLock::new();
    static TSX: OnceLock<Option<Query>> = OnceLock::new();
    static JAVASCRIPT: OnceLock<Option<Query>> = OnceLock::new();
    static PYTHON: OnceLock<Option<Query>> = OnceLock::new();
    static GO: OnceLock<Option<Query>> = OnceLock::new();
    static RUST: OnceLock<Option<Query>> = OnceLock::new();
    let slot = match lang {
        Language::TypeScript => &TYPESCRIPT,
        Language::Tsx => &TSX,
        Language::JavaScript => &JAVASCRIPT,
        Language::Python => &PYTHON,
        Language::Go => &GO,
        Language::Rust => &RUST,
    };
    slot.get_or_init(|| Query::new(&ts_language(lang), query_source(lang)).ok())
        .as_ref()
}

/// `getASTBreakPoints` (ast.ts:274-318) computed on `start_byte`
/// positions — UTF-8 byte offsets. Internal: the public surface
/// converts to upstream's UTF-16 indices; this view feeds the
/// byte-level parity check on ASCII corpus files.
pub(crate) fn break_points_utf8(content: &str, path: &str) -> Vec<BreakPoint> {
    let Some(lang) = detect_language(path) else {
        return Vec::new();
    };
    let Some(query) = query_for(lang) else {
        return Vec::new();
    };
    let mut parser = Parser::new();
    if parser.set_language(&ts_language(lang)).is_err() {
        return Vec::new();
    }
    let Some(tree) = parser.parse(content, None) else {
        return Vec::new();
    };
    // `captures()` yields captures in upstream `query.captures` order —
    // the tie on equal score at a position keeps the first capture,
    // so iteration order must match upstream's exactly (ast.ts:296-313).
    let mut cursor = QueryCursor::new();
    let mut captures = cursor.captures(query, tree.root_node(), content.as_bytes());
    let mut seen: HashMap<usize, BreakPoint> = HashMap::new();
    while let Some((m, cap_idx)) = captures.next() {
        let Some(cap) = m.captures.get(*cap_idx) else {
            continue;
        };
        let pos = cap.node.start_byte();
        let name = query
            .capture_names()
            .get(cap.index as usize)
            .copied()
            .unwrap_or_default();
        let score = capture_score(name);
        if seen.get(&pos).is_none_or(|x| score > x.score) {
            seen.insert(
                pos,
                BreakPoint {
                    pos,
                    score,
                    kind: format!("ast:{name}").into(),
                },
            );
        }
    }
    let mut points: Vec<BreakPoint> = seen.into_values().collect();
    points.sort_by_key(|p| p.pos);
    points
}

/// `getASTBreakPoints` verbatim — positions as UTF-16 code-unit
/// indices (upstream's `startIndex` semantics). Returns empty
/// for unsupported languages, parse failures, or query errors.
#[must_use]
pub fn ast_break_points(content: &str, path: &str) -> Vec<BreakPoint> {
    let mut points = break_points_utf8(content, path);
    // Byte positions land on char boundaries (tree-sitter node
    // starts), so one char_indices pass converts them all.
    let mut byte = 0usize;
    let mut utf16 = 0usize;
    let mut chars = content.char_indices();
    for bp in &mut points {
        while byte < bp.pos {
            match chars.next() {
                Some((i, c)) => {
                    byte = i + c.len_utf8();
                    utf16 += c.len_utf16();
                }
                None => break,
            }
        }
        bp.pos = utf16;
    }
    points
}

#[cfg(test)]
mod tests {
    //! Ports of upstream `test/ast.test.ts` (detectLanguage :16-70,
    //! per-language break points :74-292, error handling :295-318) plus
    //! the byte → UTF-16 conversion and the `ast_golden.json` replay.
    #![allow(
        clippy::missing_assert_message,
        reason = "idiomatic test-code patterns"
    )]

    use super::*;

    // Sample sources copied verbatim from ast.test.ts fixtures.

    const TS_SAMPLE: &str = r"import { Database } from './db';
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
";

    const PY_SAMPLE: &str = r"import os
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
";

    const GO_SAMPLE: &str = r#"package main

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
"#;

    const RS_SAMPLE: &str = r"use std::collections::HashMap;

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

fn hash_password(password: &str) -> String {
    String::new()
}
";

    // ---- detectLanguage (ast.test.ts:16-70) ----

    #[test]
    fn detects_all_extensions() {
        let cases: &[(&str, Language)] = &[
            ("src/auth.ts", Language::TypeScript),
            ("src/auth.mts", Language::TypeScript),
            ("src/auth.cts", Language::TypeScript),
            ("src/App.tsx", Language::Tsx),
            ("src/util.js", Language::JavaScript),
            ("src/util.mjs", Language::JavaScript),
            ("src/util.cjs", Language::JavaScript),
            ("src/App.jsx", Language::Tsx),
            ("src/auth.py", Language::Python),
            ("src/auth.go", Language::Go),
            ("src/auth.rs", Language::Rust),
        ];
        for (path, want) in cases {
            assert_eq!(detect_language(path), Some(*want), "{path}");
        }
    }

    #[test]
    fn rejects_non_code_and_extensionless() {
        for path in ["docs/README.md", "data/file.csv", "config.yaml", "Makefile"] {
            assert_eq!(detect_language(path), None, "{path}");
        }
    }

    #[test]
    fn detection_is_case_insensitive() {
        assert_eq!(detect_language("src/Auth.TS"), Some(Language::TypeScript));
        assert_eq!(detect_language("src/Auth.PY"), Some(Language::Python));
    }

    #[test]
    fn detection_works_on_virtual_paths() {
        assert_eq!(
            detect_language("qmd://myproject/src/auth.ts"),
            Some(Language::TypeScript)
        );
        assert_eq!(detect_language("qmd://docs/README.md"), None);
    }

    // ---- TypeScript (ast.test.ts:74-140) ----

    #[test]
    fn ts_produces_break_points_at_boundaries() {
        let points = ast_break_points(TS_SAMPLE, "src/auth.ts");
        assert!(!points.is_empty());
        let has = |frag: &str| points.iter().any(|p| p.kind.contains(frag));
        assert!(has("import") && has("iface") && has("type"));
        assert!(has("export") || has("class"));
        assert!(has("method"));
    }

    #[test]
    fn ts_break_points_sorted() {
        let points = ast_break_points(TS_SAMPLE, "src/auth.ts");
        for w in points.windows(2) {
            let first = w.first().expect("window");
            let last = w.last().expect("window");
            assert!(last.pos >= first.pos);
        }
    }

    #[test]
    fn ts_scores_align_with_hierarchy() {
        let points = ast_break_points(TS_SAMPLE, "src/auth.ts");
        let score = |kind: &str| points.iter().find(|p| p.kind == kind).map(|p| p.score);
        assert_eq!(score("ast:iface"), Some(100));
        assert_eq!(score("ast:method"), Some(90));
        assert_eq!(score("ast:import"), Some(60));
    }

    #[test]
    fn ts_break_point_positions_match_content() {
        let points = ast_break_points(TS_SAMPLE, "src/auth.ts");
        let first_import = points.iter().find(|p| p.kind == "ast:import").unwrap();
        assert_eq!(
            TS_SAMPLE
                .get(first_import.pos..first_import.pos + 6)
                .unwrap_or_default(),
            "import"
        );
    }

    // ---- Python (ast.test.ts:144-181) ----

    #[test]
    fn py_produces_class_func_import_decorated() {
        let points = ast_break_points(PY_SAMPLE, "auth.py");
        let has = |frag: &str| points.iter().any(|p| p.kind.contains(frag));
        assert!(has("import") && has("class") && has("func") && has("decorated"));
    }

    #[test]
    fn py_captures_methods_inside_classes() {
        let points = ast_break_points(PY_SAMPLE, "auth.py");
        let funcs = points.iter().filter(|p| p.kind == "ast:func").count();
        assert!(funcs >= 3, "{funcs} func points");
    }

    // ---- Go (ast.test.ts:185-223) ----

    #[test]
    fn go_produces_type_func_method_import() {
        let points = ast_break_points(GO_SAMPLE, "auth.go");
        let has = |frag: &str| points.iter().any(|p| p.kind.contains(frag));
        assert!(has("import") && has("type") && has("method") && has("func"));
    }

    #[test]
    fn go_func_and_method_score_90() {
        let points = ast_break_points(GO_SAMPLE, "auth.go");
        let score = |kind: &str| points.iter().find(|p| p.kind == kind).map(|p| p.score);
        assert_eq!(score("ast:func"), Some(90));
        assert_eq!(score("ast:method"), Some(90));
    }

    // ---- Rust (ast.test.ts:227-292) ----

    #[test]
    fn rs_produces_all_node_kinds() {
        let points = ast_break_points(RS_SAMPLE, "auth.rs");
        let has = |frag: &str| points.iter().any(|p| p.kind.contains(frag));
        assert!(has("import") && has("struct") && has("impl"));
        assert!(has("trait") && has("enum") && has("func"));
    }

    #[test]
    fn rs_struct_impl_trait_score_100() {
        let points = ast_break_points(RS_SAMPLE, "auth.rs");
        let score = |kind: &str| points.iter().find(|p| p.kind == kind).map(|p| p.score);
        assert_eq!(score("ast:struct"), Some(100));
        assert_eq!(score("ast:impl"), Some(100));
        assert_eq!(score("ast:trait"), Some(100));
    }

    // ---- Error handling (ast.test.ts:295-318, first 4) ----

    #[test]
    fn empty_for_unsupported_types() {
        assert!(ast_break_points("# Hello World", "readme.md").is_empty());
        assert!(ast_break_points("data,here", "file.csv").is_empty());
    }

    #[test]
    fn empty_content_parses_to_nothing() {
        assert!(ast_break_points("", "empty.ts").is_empty());
    }

    #[test]
    fn invalid_syntax_does_not_panic() {
        // Tree-sitter is error-tolerant — a broken file yields a tree
        // with ERROR nodes; result may be empty or partial, never a panic.
        let _points = ast_break_points("function { broken syntax %%%", "broken.ts");
    }

    // ---- byte → UTF-16 conversion (upstream `startIndex` units) ----

    #[test]
    fn ascii_byte_positions_equal_utf16() {
        let content = "import x;\n\nclass Foo {\n  m() {}\n}\n";
        let utf8 = break_points_utf8(content, "a.ts");
        let utf16 = ast_break_points(content, "a.ts");
        assert_eq!(utf8.len(), utf16.len());
        for (a, b) in utf8.iter().zip(&utf16) {
            assert_eq!(a.pos, b.pos);
            assert_eq!(a.score, b.score);
            assert_eq!(a.kind, b.kind);
        }
    }

    #[test]
    fn cjk_and_astral_convert_to_utf16_indices() {
        // 4-byte emoji + 3-byte CJK shift byte offsets well past the
        // UTF-16 indices; conversion must land on JS-string positions.
        let content = "// 日本語\nexport function f() { return \"🎉\"; }\n";
        let utf8 = break_points_utf8(content, "a.ts");
        let utf16 = ast_break_points(content, "a.ts");
        assert_eq!(utf8.len(), utf16.len());
        for (a, b) in utf8.iter().zip(&utf16) {
            let want = content
                .get(..a.pos)
                .map_or(0, |s| s.chars().map(char::len_utf16).sum());
            assert_eq!(b.pos, want, "kind {}", b.kind);
            assert!(a.pos > b.pos, "byte pos should exceed utf16 pos here");
        }
    }

    // ---- Golden replay: upstream recordings are byte-authoritative ----

    #[test]
    fn ast_golden_replay() {
        let fixture = include_str!("../../tests/fixtures/ast_golden.json");
        let root: serde_json::Value = serde_json::from_str(fixture).unwrap();
        let cases = root
            .get("cases")
            .and_then(serde_json::Value::as_array)
            .unwrap();
        assert_eq!(cases.len(), 18);
        for case in cases {
            let path = case["path"].as_str().unwrap();
            let content = case["content"].as_str().unwrap();
            let recorded = case["points"].as_array().unwrap();
            // Upstream records UTF-16 positions — the public API
            // must match verbatim on every file.
            let got = ast_break_points(content, path);
            assert_eq!(got.len(), recorded.len(), "{path} count");
            for (g, r) in got.iter().zip(recorded) {
                assert_eq!(
                    g.pos,
                    usize::try_from(r["pos"].as_u64().unwrap()).unwrap(),
                    "{path} pos"
                );
                assert_eq!(
                    g.score,
                    u32::try_from(r["score"].as_u64().unwrap()).unwrap(),
                    "{path} score"
                );
                assert_eq!(g.kind.as_ref(), r["type"].as_str().unwrap(), "{path} type");
            }
            // ASCII files are also byte-identical to the raw view.
            if content.is_ascii() {
                let raw = break_points_utf8(content, path);
                assert!(
                    raw.iter().zip(&got).all(|(a, b)| a.pos == b.pos),
                    "{path} byte-vs-utf16"
                );
            }
        }
    }
}
