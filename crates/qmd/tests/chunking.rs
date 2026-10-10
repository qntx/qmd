//! Chunking tests (P2.1): `scan_break_points`, `find_code_fences`,
//! `is_inside_code_fence`, `find_best_cutoff`, `merge_break_points`,
//! `chunk_document`, `chunk_document_with_break_points`,
//! `chunk_document_with_strategy`, `chunk_document_by_tokens`, and the
//! upstream `chunk_golden.json` fixture.
//!
//! Ported upstream coverage: `test/store.test.ts` groups "Document
//! Chunking", "Token-based Chunking" (fake-tokenizer shape), "Smart
//! Chunking - Break Point Detection", "Smart Chunking Integration",
//! "mergeBreakPoints", "chunkDocumentWithBreakPoints",
//! "AST-aware chunkDocumentAsync" (degraded-auto assertions), "Token
//! chunking guardrails"; and `test/ast-chunking.test.ts` groups
//! "mergeBreakPoints" and "chunkDocumentWithBreakPoints equivalence".

#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::shadow_unrelated,
    clippy::field_reassign_with_default,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_assert_message,
    clippy::panic,
    reason = "idiomatic test-code patterns"
)]

use std::borrow::Cow;
use std::fs;
use std::path::Path;

use qmd::{
    BREAK_PATTERNS, BreakPoint, CharChunkOptions, ChunkStrategy, CodeFenceRegion,
    TokenChunkOptions, chunk_document, chunk_document_by_tokens, chunk_document_with_break_points,
    chunk_document_with_strategy, find_best_cutoff, find_code_fences, is_inside_code_fence,
    merge_break_points, scan_break_points, utf16_len, utf16_to_byte_offset,
};

fn bp(pos: usize, score: u32, kind: &str) -> BreakPoint {
    BreakPoint {
        pos,
        score,
        kind: Cow::Owned(kind.to_owned()),
    }
}

const fn opts(max_chars: usize, overlap_chars: usize, window_chars: usize) -> CharChunkOptions {
    CharChunkOptions {
        max_chars,
        overlap_chars,
        window_chars,
    }
}

// =============================================================================
// Document Chunking — store.test.ts:581-655
// =============================================================================

#[test]
fn chunk_document_returns_single_chunk_for_small_documents() {
    let content = "Small document content";
    let chunks = chunk_document(content, &opts(1000, 0, 800));
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, content);
    assert_eq!(chunks[0].pos, 0);
}

#[test]
fn chunk_document_splits_large_documents() {
    let content = "A".repeat(10_000);
    let chunks = chunk_document(&content, &opts(1000, 0, 800));
    assert!(chunks.len() > 1);
    for (i, chunk) in chunks.iter().enumerate() {
        assert!(i == 0 || chunk.pos > chunks[i - 1].pos);
    }
}

#[test]
fn chunk_document_with_overlap_creates_overlapping_chunks() {
    let content = "A".repeat(3000);
    let chunks = chunk_document(&content, &opts(1000, 150, 800));
    assert!(chunks.len() > 1);
    for i in 1..chunks.len() {
        let prev_end = chunks[i - 1].pos + utf16_len(chunks[i - 1].text);
        assert!(chunks[i].pos < prev_end);
        assert!(chunks[i].pos > chunks[i - 1].pos);
    }
}

#[test]
fn chunk_document_prefers_paragraph_breaks() {
    let content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.".repeat(50);
    let chunks = chunk_document(&content, &opts(500, 0, 800));
    assert!(chunks.len() > 1);
}

#[test]
fn chunk_document_handles_utf8_characters_correctly() {
    let content = "こんにちは世界".repeat(500);
    let chunks = chunk_document(&content, &opts(1000, 0, 800));
    // &str is always well-formed; assert chunk text actually covers the
    // document (no dropped content).
    let covered: usize = chunks.iter().map(|c| utf16_len(c.text)).sum();
    assert!(covered >= utf16_len(&content));
    assert!(chunks.len() > 1);
}

#[test]
fn chunk_document_with_default_params_uses_900_token_chunks() {
    let content = "Word ".repeat(2500); // ~12500 chars
    let chunks = chunk_document(&content, &CharChunkOptions::default());
    assert!(chunks.len() > 1);
    assert!(utf16_len(chunks[0].text) > 2800);
    assert!(utf16_len(chunks[0].text) <= 3600);
}

// =============================================================================
// Token-based Chunking — store.test.ts:657-714 (fake-tokenizer shape)
// =============================================================================

#[cfg(feature = "testing")]
mod token_chunks {
    use qmd::FakeEmbedder;

    use super::*;

    fn char_tokenizer_embedder() -> FakeEmbedder {
        FakeEmbedder::new("fake").with_tokenizer(|s| s.chars().map(|_| 1).collect())
    }

    fn token_opts(max_tokens: usize, overlap: usize, window: usize) -> TokenChunkOptions {
        TokenChunkOptions {
            max_tokens,
            overlap_tokens: overlap,
            window_tokens: window,
            ..TokenChunkOptions::default()
        }
    }

    #[test]
    fn chunk_document_by_tokens_returns_single_chunk_for_small_documents() {
        let e = char_tokenizer_embedder();
        let content = "This is a small document.";
        let chunks = chunk_document_by_tokens(&e, content, &token_opts(900, 135, 200)).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, content);
        assert_eq!(chunks[0].pos, 0);
        assert!(chunks[0].tokens > 0);
        assert!(chunks[0].tokens < 900);
    }

    #[test]
    fn chunk_document_by_tokens_splits_large_documents() {
        let e = char_tokenizer_embedder();
        let content = "The quick brown fox jumps over the lazy dog. ".repeat(250);
        let chunks = chunk_document_by_tokens(&e, &content, &token_opts(900, 135, 200)).unwrap();
        assert!(chunks.len() > 1);
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(chunk.tokens <= 950);
            assert!(chunk.tokens > 0);
            assert!(i == 0 || chunk.pos > chunks[i - 1].pos);
        }
    }

    #[test]
    fn chunk_document_by_tokens_creates_overlapping_chunks() {
        let e = char_tokenizer_embedder();
        let content = "Word ".repeat(500);
        let chunks = chunk_document_by_tokens(&e, &content, &token_opts(200, 30, 200)).unwrap();
        assert!(chunks.len() > 1);
        for i in 1..chunks.len() {
            let prev_end = chunks[i - 1].pos + utf16_len(&chunks[i - 1].text);
            assert!(chunks[i].pos < prev_end);
        }
    }

    #[test]
    fn chunk_document_by_tokens_returns_actual_token_counts() {
        let e = char_tokenizer_embedder();
        let content = "Hello world, this is a test.";
        let chunks = chunk_document_by_tokens(&e, content, &TokenChunkOptions::default()).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].tokens, content.chars().count());
    }
}

// =============================================================================
// scanBreakPoints — store.test.ts:720-808
// =============================================================================

#[test]
fn scan_break_points_detects_h1_headings() {
    let text = "Intro\n# Heading 1\nMore text";
    let breaks = scan_break_points(text);
    let h1 = breaks.iter().find(|b| b.kind == "h1").unwrap();
    assert_eq!(h1.score, 100);
    assert_eq!(h1.pos, 5);
}

#[test]
fn scan_break_points_detects_multiple_heading_levels() {
    let text = "Text\n# H1\n## H2\n### H3\nMore";
    let breaks = scan_break_points(text);
    let score_of = |kind: &str| breaks.iter().find(|b| b.kind == kind).unwrap().score;
    assert_eq!(score_of("h1"), 100);
    assert_eq!(score_of("h2"), 90);
    assert_eq!(score_of("h3"), 80);
}

#[test]
fn scan_break_points_detects_code_blocks() {
    let text = "Before\n```js\ncode\n```\nAfter";
    let breaks = scan_break_points(text);
    let blocks: Vec<_> = breaks.iter().filter(|b| b.kind == "codeblock").collect();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].score, 80);
}

#[test]
fn scan_break_points_detects_horizontal_rules() {
    let text = "Text\n---\nMore text";
    let breaks = scan_break_points(text);
    let hr = breaks.iter().find(|b| b.kind == "hr").unwrap();
    assert_eq!(hr.score, 60);
}

#[test]
fn scan_break_points_detects_blank_lines() {
    let text = "First paragraph.\n\nSecond paragraph.";
    let breaks = scan_break_points(text);
    let blank = breaks.iter().find(|b| b.kind == "blank").unwrap();
    assert_eq!(blank.score, 20);
}

#[test]
fn scan_break_points_detects_list_items() {
    let text = "Intro\n- Item 1\n- Item 2\n1. Numbered";
    let breaks = scan_break_points(text);
    let lists: Vec<_> = breaks.iter().filter(|b| b.kind == "list").collect();
    let nums: Vec<_> = breaks.iter().filter(|b| b.kind == "numlist").collect();
    assert_eq!(lists.len(), 2);
    assert_eq!(nums.len(), 1);
    assert_eq!(lists[0].score, 5);
    assert_eq!(nums[0].score, 5);
}

#[test]
fn scan_break_points_detects_newlines_as_fallback() {
    let text = "Line 1\nLine 2\nLine 3";
    let breaks = scan_break_points(text);
    let newlines: Vec<_> = breaks.iter().filter(|b| b.kind == "newline").collect();
    assert_eq!(newlines.len(), 2);
    assert_eq!(newlines[0].score, 1);
}

#[test]
fn scan_break_points_returns_breaks_sorted_by_position() {
    let text = "A\n# B\n\nC\n## D";
    let breaks = scan_break_points(text);
    for w in breaks.windows(2) {
        assert!(w[1].pos > w[0].pos);
    }
}

#[test]
fn scan_break_points_higher_scoring_pattern_wins_at_same_position() {
    let text = "Text\n# Heading";
    let breaks = scan_break_points(text);
    let at_pos: Vec<_> = breaks.iter().filter(|b| b.pos == 4).collect();
    assert_eq!(at_pos.len(), 1);
    assert_eq!(at_pos[0].kind, "h1");
    assert_eq!(at_pos[0].score, 100);
}

#[test]
fn scan_break_points_heading_run_lengths() {
    // `\n#+` run-length classification: every heading pattern carries a
    // `(?!#)` lookahead upstream, so exactly-k → h1..h6 and a run of 7+
    // matches NO heading pattern — only the `newline` fallback. Verified
    // verbatim against upstream `scanBreakPoints` on this input.
    let text = "\n#\n##\n###\n####\n#####\n######\n#######\n########";
    let breaks = scan_break_points(text);
    let kinds: Vec<(&str, u32)> = breaks.iter().map(|b| (b.kind.as_ref(), b.score)).collect();
    assert_eq!(
        kinds,
        [
            ("h1", 100),
            ("h2", 90),
            ("h3", 80),
            ("h4", 70),
            ("h5", 60),
            ("h6", 50),
            ("newline", 1),
            ("newline", 1),
        ]
    );
}

#[test]
fn scan_break_points_js_whitespace_excludes_nel() {
    // JS `\s` does not match U+0085 (NEL); `\p{White_Space}` would.
    // Verified against upstream: these inputs yield only `newline`s.
    for (text, expected_kinds) in [
        ("a\n---\u{85}\nb", ["newline", "newline"].as_slice()),
        ("a\n-\u{85}item", ["newline"].as_slice()),
        ("a\n1.\u{85}item", ["newline"].as_slice()),
    ] {
        let breaks = scan_break_points(text);
        let kinds: Vec<&str> = breaks.iter().map(|b| b.kind.as_ref()).collect();
        assert_eq!(kinds, expected_kinds, "input {text:?}");
    }
    // Sanity: real whitespace still matches.
    let breaks = scan_break_points("a\n--- \nb");
    assert!(breaks.iter().any(|b| b.kind == "hr"));
}

#[test]
fn break_patterns_table_matches_upstream_scores() {
    let scores: Vec<(&str, u32)> = BREAK_PATTERNS.iter().map(|p| (p.kind, p.score)).collect();
    assert_eq!(
        scores,
        [
            ("h1", 100),
            ("h2", 90),
            ("h3", 80),
            ("h4", 70),
            ("h5", 60),
            ("h6", 50),
            ("codeblock", 80),
            ("hr", 60),
            ("blank", 20),
            ("list", 5),
            ("numlist", 5),
            ("newline", 1),
        ]
    );
}

// =============================================================================
// findCodeFences / isInsideCodeFence — store.test.ts:810-868
// =============================================================================

#[test]
fn find_code_fences_finds_single_code_fence() {
    let text = "Before\n```js\ncode here\n```\nAfter";
    let fences = find_code_fences(text);
    assert_eq!(fences.len(), 1);
    assert_eq!(fences[0].start, 6);
    assert_eq!(fences[0].end, 26);
}

#[test]
fn find_code_fences_finds_multiple_code_fences() {
    let text = "Intro\n```\nblock1\n```\nMiddle\n```\nblock2\n```\nEnd";
    let fences = find_code_fences(text);
    assert_eq!(fences.len(), 2);
}

#[test]
fn find_code_fences_handles_unclosed_code_fence() {
    let text = "Before\n```\nunclosed code block";
    let fences = find_code_fences(text);
    assert_eq!(fences.len(), 1);
    assert_eq!(fences[0].end, utf16_len(text));
}

#[test]
fn find_code_fences_returns_empty_for_no_fences() {
    assert!(find_code_fences("No code fences here").is_empty());
}

#[test]
fn is_inside_code_fence_positions() {
    let fences = [CodeFenceRegion { start: 10, end: 30 }];
    assert!(is_inside_code_fence(15, &fences));
    assert!(is_inside_code_fence(20, &fences));
    assert!(!is_inside_code_fence(5, &fences));
    assert!(!is_inside_code_fence(35, &fences));
    assert!(!is_inside_code_fence(10, &fences)); // at start
    assert!(!is_inside_code_fence(30, &fences)); // at end
}

#[test]
fn is_inside_code_fence_handles_multiple_fences() {
    let fences = [
        CodeFenceRegion { start: 10, end: 30 },
        CodeFenceRegion { start: 50, end: 70 },
    ];
    assert!(is_inside_code_fence(20, &fences));
    assert!(is_inside_code_fence(60, &fences));
    assert!(!is_inside_code_fence(40, &fences));
}

// =============================================================================
// findBestCutoff — store.test.ts:870-929
// =============================================================================

#[test]
fn find_best_cutoff_prefers_higher_scoring_break_points() {
    let bps = [
        bp(100, 1, "newline"),
        bp(150, 100, "h1"),
        bp(180, 20, "blank"),
    ];
    assert_eq!(find_best_cutoff(&bps, 200, 100, 0.7, &[]), 150);
}

#[test]
fn find_best_cutoff_h2_at_window_edge_beats_blank_at_target() {
    let bps = [bp(100, 90, "h2"), bp(195, 20, "blank")];
    assert_eq!(find_best_cutoff(&bps, 200, 100, 0.7, &[]), 100);
}

#[test]
fn find_best_cutoff_high_score_easily_overcomes_distance() {
    let bps = [bp(150, 100, "h1"), bp(195, 1, "newline")];
    assert_eq!(find_best_cutoff(&bps, 200, 100, 0.7, &[]), 150);
}

#[test]
fn find_best_cutoff_returns_target_when_no_breaks_in_window() {
    let bps = [bp(10, 100, "h1")];
    assert_eq!(find_best_cutoff(&bps, 200, 100, 0.7, &[]), 200);
}

#[test]
fn find_best_cutoff_skips_break_points_inside_code_fences() {
    let bps = [bp(150, 100, "h1"), bp(180, 20, "blank")];
    let fences = [CodeFenceRegion {
        start: 140,
        end: 160,
    }];
    assert_eq!(find_best_cutoff(&bps, 200, 100, 0.7, &fences), 180);
}

#[test]
fn find_best_cutoff_handles_empty_break_points() {
    assert_eq!(find_best_cutoff(&[], 200, 100, 0.7, &[]), 200);
}

// =============================================================================
// Smart Chunking Integration — store.test.ts:931-1016
// =============================================================================

#[test]
fn chunk_document_prefers_headings_over_arbitrary_breaks() {
    let section1 = "Introduction text here. ".repeat(70); // ~1680 chars
    let section2 = "Main content text here. ".repeat(50);
    let content = format!("{section1}\n# Main Section\n{section2}");
    let chunks = chunk_document(&content, &opts(2000, 0, 800));
    let heading_pos = content.find("\n# Main Section").unwrap();
    assert!(chunks.len() >= 2);
    assert_eq!(utf16_len(chunks[0].text), heading_pos);
}

#[test]
fn chunk_document_does_not_split_inside_code_blocks() {
    let before_code = "Some intro text. ".repeat(30);
    let code_block = format!("```typescript\n{}", "const x = 1;\n".repeat(100));
    let after_code = "More text after code. ".repeat(30);
    let content = format!("{before_code}{code_block}\n{after_code}");
    let chunks = chunk_document(&content, &opts(1000, 0, 400));
    assert!(chunks.len() > 1);
}

#[test]
fn chunk_document_handles_markdown_with_mixed_elements() {
    let content = "# Introduction\n\nThis is the introduction paragraph with some text.\n\n\
        ## Section 1\n\nSome content in section 1.\n\n- List item 1\n- List item 2\n- List item 3\n\n\
        ## Section 2\n\n```javascript\nfunction hello() {\n  console.log(\"Hello\");\n}\n```\n\n\
        More text after the code block.\n\n---\n\n## Section 3\n\nFinal section content.\n"
        .repeat(10);
    let chunks = chunk_document(&content, &opts(500, 75, 200));
    assert!(chunks.len() > 5);
    for chunk in &chunks {
        assert!(!chunk.text.is_empty());
    }
}

// =============================================================================
// mergeBreakPoints — store.test.ts:1022-1055 + ast-chunking.test.ts:24-53
// =============================================================================

#[test]
fn merge_break_points_keeps_highest_score_at_each_position() {
    let regex_points = [bp(10, 20, "blank"), bp(50, 1, "newline")];
    let ast_points = [bp(10, 90, "ast:func"), bp(100, 100, "ast:class")];
    let merged = merge_break_points(&regex_points, &ast_points);
    assert_eq!(merged.len(), 3);
    let at = |pos: usize| merged.iter().find(|p| p.pos == pos).unwrap();
    assert_eq!(at(10).score, 90);
    assert_eq!(at(10).kind, "ast:func");
    assert_eq!(at(50).score, 1);
    assert_eq!(at(100).score, 100);
}

#[test]
fn merge_break_points_returns_sorted_by_position() {
    let a = [bp(100, 10, "a")];
    let b = [bp(5, 20, "b")];
    let merged = merge_break_points(&a, &b);
    assert_eq!(merged[0].pos, 5);
    assert_eq!(merged[1].pos, 100);
}

#[test]
fn merge_break_points_ast_chunking_suite_case() {
    // ast-chunking.test.ts:25-44.
    let regex_points = [
        bp(10, 20, "blank"),
        bp(50, 1, "newline"),
        bp(100, 20, "blank"),
    ];
    let ast_points = [
        bp(10, 90, "ast:func"),
        bp(75, 100, "ast:class"),
        bp(100, 60, "ast:import"),
    ];
    let merged = merge_break_points(&regex_points, &ast_points);
    assert_eq!(merged.len(), 4);
    let at = |pos: usize| merged.iter().find(|p| p.pos == pos).unwrap().score;
    assert_eq!(at(10), 90);
    assert_eq!(at(50), 1);
    assert_eq!(at(75), 100);
    assert_eq!(at(100), 60);
}

// =============================================================================
// chunkDocumentWithBreakPoints — store.test.ts:1057-1108
// + ast-chunking.test.ts:156-168
// =============================================================================

#[test]
fn chunk_document_with_break_points_equivalent_to_chunk_document() {
    let content = format!("{}\n\n{}", "a".repeat(5000), "b".repeat(5000));
    let bps = scan_break_points(&content);
    let fences = find_code_fences(&content);
    let original = chunk_document(&content, &CharChunkOptions::default());
    let with_bp =
        chunk_document_with_break_points(&content, &bps, &fences, &CharChunkOptions::default());
    assert_eq!(with_bp, original);
}

#[test]
fn chunk_document_does_not_split_surrogate_pair_at_raw_boundary() {
    // 🚀 occupies UTF-16 units 99-100; targetEndPos=100 lands mid-pair.
    let content = format!("{}\u{1F680}{}", "x".repeat(99), "y".repeat(50));
    let chunks = chunk_document_with_break_points(&content, &[], &[], &opts(100, 20, 10));
    assert!(chunks.len() > 1);
    for chunk in &chunks {
        // Well-formedness is guaranteed by &str; assert the emoji is
        // never split across chunks (each chunk holds it whole or none).
        let rockets = chunk.text.matches('\u{1F680}').count();
        assert!(rockets <= 1);
    }
    let total_rockets: usize = chunks
        .iter()
        .map(|c| c.text.matches('\u{1F680}').count())
        .sum();
    assert_eq!(total_rockets, 1);
}

#[test]
fn chunk_document_does_not_split_surrogate_pair_at_overlap_start() {
    // Boundary is clean; the next chunk's overlap start (100-20=80) lands
    // between high (79) and low (80) surrogate.
    let content = format!("{}\u{1F680}{}", "x".repeat(79), "y".repeat(69));
    let chunks = chunk_document_with_break_points(&content, &[], &[], &opts(100, 20, 10));
    assert!(chunks.len() > 1);
    let total_rockets: usize = chunks
        .iter()
        .map(|c| c.text.matches('\u{1F680}').count())
        .sum();
    // The pair stays intact; it may appear in two overlapping chunks.
    assert!(total_rockets >= 1);
}

// =============================================================================
// AST-aware chunkDocumentAsync (degraded Auto) — store.test.ts:1110-1169
// + ast-chunking.test.ts:113-149
// =============================================================================

fn ts_code() -> String {
    "import { Database } from './db';\n\nexport class AuthService {\n  \
    constructor(private db: Database) {}\n\n  async authenticate(user: User, token: string): \
    Promise<boolean> {\n    const session = await this.db.findSession(token);\n    \
    return session?.userId === user.id;\n  }\n\n  validateToken(token: string): boolean {\n    \
    return token.length === 64;\n  }\n}\n\nexport function hashPassword(password: string): string {\n  \
    return crypto.createHash('sha256').update(password).digest('hex');\n}\n"
        .repeat(10)
}

#[test]
fn auto_strategy_returns_chunks_for_code_files() {
    let content = ts_code();
    let chunks = chunk_document_with_strategy(
        &content,
        Some("auth.ts"),
        ChunkStrategy::Auto,
        &CharChunkOptions::default(),
    );
    assert!(!chunks.is_empty());
    for chunk in &chunks {
        assert!(!chunk.text.is_empty());
    }
}

#[test]
fn regex_strategy_produces_same_output_as_chunk_document() {
    let content = ts_code();
    let via_strategy = chunk_document_with_strategy(
        &content,
        Some("auth.ts"),
        ChunkStrategy::Regex,
        &CharChunkOptions::default(),
    );
    let direct = chunk_document(&content, &CharChunkOptions::default());
    assert_eq!(via_strategy, direct);
}

#[test]
fn auto_strategy_degrades_to_regex_until_p6() {
    // Auto without an AST backend (P6) is identical to Regex — covers
    // upstream's "markdown files unchanged in auto mode" and "no filepath
    // falls back to regex" cases.
    let content = ts_code();
    let auto = chunk_document_with_strategy(
        &content,
        Some("auth.ts"),
        ChunkStrategy::Auto,
        &CharChunkOptions::default(),
    );
    let no_path = chunk_document_with_strategy(
        &content,
        None,
        ChunkStrategy::Auto,
        &CharChunkOptions::default(),
    );
    let regex = chunk_document(&content, &CharChunkOptions::default());
    assert_eq!(auto, regex);
    assert_eq!(no_path, regex);

    let md = ("# Heading\n\n".to_owned() + &"Some text. ".repeat(200) + "\n\n").repeat(10);
    let auto_md = chunk_document_with_strategy(
        &md,
        Some("readme.md"),
        ChunkStrategy::Auto,
        &CharChunkOptions::default(),
    );
    let regex_md = chunk_document(&md, &CharChunkOptions::default());
    assert_eq!(auto_md, regex_md);
}

#[test]
fn small_file_produces_single_chunk() {
    let chunks = chunk_document_with_strategy(
        "export const x = 1;",
        Some("s.ts"),
        ChunkStrategy::Auto,
        &CharChunkOptions::default(),
    );
    assert_eq!(chunks.len(), 1);
}

// =============================================================================
// Token chunking guardrails — store.test.ts:4647-4812 (fake tokenizer)
// =============================================================================
//
// Upstream cases exercising lone surrogates in *source* text or in
// `detokenize` output have no Rust analogue: `&str`/`String` is always
// well-formed UTF-16, so neither `"\uD800"` input nor a lone-surrogate
// detokenize result can be constructed. Those three cases are covered by
// the type system instead of tests (see `upstream-quirks.md`).

#[cfg(feature = "testing")]
mod token_guardrails {
    use qmd::{CancelToken, FakeEmbedder};

    use super::*;

    /// Upstream fake: one token per UTF-16 code unit.
    fn unit_tokenizer_embedder() -> FakeEmbedder {
        FakeEmbedder::new("fake")
            .with_tokenizer(|s| vec![1; s.encode_utf16().count()])
            .with_detokenizer(|t| "x".repeat(t.len()))
    }

    fn opts(max_tokens: usize, overlap: usize, window: usize) -> TokenChunkOptions {
        TokenChunkOptions {
            max_tokens,
            overlap_tokens: overlap,
            window_tokens: window,
            ..TokenChunkOptions::default()
        }
    }

    #[test]
    fn keeps_pathological_single_line_blobs_under_token_limit() {
        let e = unit_tokenizer_embedder();
        let content = "x".repeat(1200);
        let chunks = chunk_document_by_tokens(&e, &content, &opts(100, 15, 20)).unwrap();
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|c| c.tokens <= 100));
        for i in 1..chunks.len() {
            assert!(chunks[i].pos > chunks[i - 1].pos);
        }
    }

    #[test]
    fn keeps_surrogate_pairs_intact_when_token_budget_shrinks_char_budget() {
        let e = unit_tokenizer_embedder();
        let content = "\u{1F680}".repeat(30); // 60 UTF-16 units
        let chunks = chunk_document_by_tokens(&e, &content, &opts(2, 0, 0)).unwrap();
        assert!(chunks.len() > 1);
        // Every chunk text is well-formed by construction and the emoji
        // count per chunk is an integer number of pairs.
        for chunk in &chunks {
            assert_eq!(utf16_len(&chunk.text) % 2, 0);
        }
    }

    #[test]
    fn drops_chunks_whose_detokenize_fallback_is_empty() {
        // Adapted from upstream's lone-surrogate detokenize case: the
        // Rust counterpart of "strip produces empty text" is a detokenize
        // result that is itself empty — the chunk must be dropped.
        let e = unit_tokenizer_embedder().with_detokenizer(|t| {
            if t.len() == 1 {
                String::new()
            } else {
                "x".repeat(t.len())
            }
        });
        let chunks = chunk_document_by_tokens(&e, "\u{1F680}", &opts(1, 0, 0)).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn empty_source_produces_no_chunks() {
        let e = unit_tokenizer_embedder();
        let chunks = chunk_document_by_tokens(&e, "", &opts(900, 135, 200)).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn cancel_token_stops_chunk_emission() {
        let e = FakeEmbedder::new("fake").with_tokenizer(|s| s.chars().map(|_| 1).collect());
        let cancel = CancelToken::new();
        cancel.cancel();
        let mut o = opts(100, 15, 20);
        o.cancel = cancel;
        let content = "word ".repeat(500);
        let chunks = chunk_document_by_tokens(&e, &content, &o).unwrap();
        assert!(chunks.is_empty());
    }
}

// =============================================================================
// UTF-16 position helpers (P2-D1)
// =============================================================================

#[test]
fn utf16_len_counts_code_units() {
    assert_eq!(utf16_len("abc"), 3);
    assert_eq!(utf16_len("こんにちは"), 5); // BMP: 1 unit each
    assert_eq!(utf16_len("\u{1F680}"), 2); // astral: surrogate pair
    assert_eq!(utf16_len("a\u{1F680}b"), 4);
}

#[test]
fn utf16_to_byte_offset_converts_boundaries() {
    let text = "ab\u{1F680}cd"; // utf16: a0 b1 🚀2-3 c4 d5; bytes: a0 b1 🚀2-5 c6 d7
    assert_eq!(utf16_to_byte_offset(text, 0), 0);
    assert_eq!(utf16_to_byte_offset(text, 2), 2); // before 🚀
    assert_eq!(utf16_to_byte_offset(text, 3), 2); // mid-pair rounds down
    assert_eq!(utf16_to_byte_offset(text, 4), 6); // after 🚀
    assert_eq!(utf16_to_byte_offset(text, 6), 8); // end
    // Round-trip: slicing at any returned offset is a valid char boundary.
    for u in 0..=utf16_len(text) {
        let b = utf16_to_byte_offset(text, u);
        assert!(text.is_char_boundary(b));
    }
}

#[test]
fn chunk_positions_are_utf16_offsets() {
    // A doc whose utf16 offsets diverge from byte offsets: verify `pos`
    // counts code units (upstream semantics), not bytes.
    let content = format!("{}\u{1F680}{}", "x".repeat(99), "y".repeat(400));
    let chunks = chunk_document(&content, &opts(200, 0, 10));
    assert!(chunks.len() > 1);
    for chunk in &chunks {
        // `pos` must slice cleanly via utf16_to_byte_offset.
        let byte_start = utf16_to_byte_offset(&content, chunk.pos);
        assert_eq!(
            &content[byte_start..byte_start + chunk.text.len()],
            chunk.text
        );
    }
}

// =============================================================================
// chunk_golden.json — upstream parity fixture (AC3)
// =============================================================================

#[derive(serde::Deserialize)]
struct Golden {
    upstream_commit: String,
    param_sets: std::collections::HashMap<String, GoldenParams>,
    docs: Vec<GoldenDoc>,
    cutoff_cases: Vec<GoldenCutoff>,
    merge_cases: Vec<GoldenMerge>,
}

#[derive(serde::Deserialize)]
struct GoldenParams {
    #[serde(rename = "maxChars")]
    max_chars: usize,
    #[serde(rename = "overlapChars")]
    overlap_chars: usize,
    #[serde(rename = "windowChars")]
    window_chars: usize,
}

#[derive(serde::Deserialize)]
struct GoldenDoc {
    file: String,
    scan_break_points: Vec<GoldenBp>,
    code_fences: Vec<GoldenFence>,
    chunks: std::collections::HashMap<String, Vec<GoldenChunk>>,
}

#[derive(serde::Deserialize)]
struct GoldenBp {
    pos: usize,
    score: u32,
    #[serde(rename = "type")]
    kind: String,
}

#[derive(serde::Deserialize)]
struct GoldenFence {
    start: usize,
    end: usize,
}

#[derive(serde::Deserialize)]
struct GoldenChunk {
    text: String,
    pos: usize,
}

#[derive(serde::Deserialize)]
struct GoldenCutoff {
    bps: Vec<GoldenBp>,
    target: usize,
    window: usize,
    decay: f64,
    fences: Vec<GoldenFence>,
    expected: usize,
}

#[derive(serde::Deserialize)]
struct GoldenMerge {
    a: Vec<GoldenBp>,
    b: Vec<GoldenBp>,
    expected: Vec<GoldenBp>,
}

#[test]
fn chunk_golden_matches_upstream() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let golden: Golden = serde_json::from_str(
        &fs::read_to_string(root.join("tests/fixtures/chunk_golden.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        golden.upstream_commit, "04e4dbd8245c527a88f1a8f0bda547aef9ca81fb",
        "fixture generated against a different upstream commit — regenerate with \
         `QMD_UPSTREAM=… bun scripts/gen_chunk_golden.ts` and re-review"
    );

    // Sizing parameters come from the fixture itself — the same record
    // the generator fed upstream — so there is no duplicated constant to
    // drift out of sync.
    let param_sets: std::collections::HashMap<&str, CharChunkOptions> = golden
        .param_sets
        .iter()
        .map(|(name, p)| {
            (
                name.as_str(),
                opts(p.max_chars, p.overlap_chars, p.window_chars),
            )
        })
        .collect();

    for doc in &golden.docs {
        let content = fs::read_to_string(root.join("tests/fixtures").join(&doc.file)).unwrap();

        // scanBreakPoints parity (pos + score + type).
        let bps = scan_break_points(&content);
        let ours: Vec<(usize, u32, &str)> = bps
            .iter()
            .map(|b| (b.pos, b.score, b.kind.as_ref()))
            .collect();
        let theirs: Vec<(usize, u32, &str)> = doc
            .scan_break_points
            .iter()
            .map(|b| (b.pos, b.score, b.kind.as_str()))
            .collect();
        assert_eq!(ours, theirs, "scan_break_points mismatch in {}", doc.file);

        // findCodeFences parity.
        let fences = find_code_fences(&content);
        let ours_f: Vec<(usize, usize)> = fences.iter().map(|f| (f.start, f.end)).collect();
        let theirs_f: Vec<(usize, usize)> =
            doc.code_fences.iter().map(|f| (f.start, f.end)).collect();
        assert_eq!(
            ours_f, theirs_f,
            "find_code_fences mismatch in {}",
            doc.file
        );

        // chunkDocumentWithBreakPoints parity per param set (text + pos).
        for (name, params) in &param_sets {
            let golden_chunks = doc.chunks.get(*name).unwrap();
            let ours_chunks = chunk_document_with_break_points(&content, &bps, &fences, params);
            let ours_c: Vec<(&str, usize)> = ours_chunks.iter().map(|c| (c.text, c.pos)).collect();
            let theirs_c: Vec<(&str, usize)> = golden_chunks
                .iter()
                .map(|c| (c.text.as_str(), c.pos))
                .collect();
            assert_eq!(
                ours_c, theirs_c,
                "chunk mismatch in {} @ {}",
                doc.file, name
            );
        }
    }

    for (i, case) in golden.cutoff_cases.iter().enumerate() {
        let bps: Vec<BreakPoint> = case
            .bps
            .iter()
            .map(|b| bp(b.pos, b.score, &b.kind))
            .collect();
        let fences: Vec<CodeFenceRegion> = case
            .fences
            .iter()
            .map(|f| CodeFenceRegion {
                start: f.start,
                end: f.end,
            })
            .collect();
        assert_eq!(
            find_best_cutoff(&bps, case.target, case.window, case.decay, &fences),
            case.expected,
            "cutoff case {i}"
        );
    }

    for (i, case) in golden.merge_cases.iter().enumerate() {
        let a: Vec<BreakPoint> = case.a.iter().map(|b| bp(b.pos, b.score, &b.kind)).collect();
        let b: Vec<BreakPoint> = case.b.iter().map(|b| bp(b.pos, b.score, &b.kind)).collect();
        let merged = merge_break_points(&a, &b);
        let ours: Vec<(usize, u32, &str)> = merged
            .iter()
            .map(|p| (p.pos, p.score, p.kind.as_ref()))
            .collect();
        let theirs: Vec<(usize, u32, &str)> = case
            .expected
            .iter()
            .map(|p| (p.pos, p.score, p.kind.as_str()))
            .collect();
        assert_eq!(ours, theirs, "merge case {i}");
    }
}
