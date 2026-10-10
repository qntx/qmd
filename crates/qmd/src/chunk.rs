//! Smart chunking — ported from upstream `store.ts:112-121, 142-422,
//! 3146-3370`.
//!
//! **All positions and sizes are UTF-16 code units** (P2-D1): upstream JS
//! strings index UTF-16 units, and `pos` is persisted in
//! `content_vectors`. Rust slices are byte offsets, so this module keeps a
//! per-document `Utf16Index` and converts to byte offsets only when
//! slicing — every adjustment that could land inside a surrogate pair is
//! resolved in UTF-16 space first, exactly as upstream.
//!
//! Rust `&str` is always well-formed UTF-16, so upstream's
//! `stripUnpairedSurrogates` safety net (`store.ts:3216-3244`) has no
//! input to act on: the source text can never contain a lone surrogate
//! and every boundary adjustment preserves pairing. The one remaining
//! producer of non-slice text — the tokenizer's `detokenize` truncation
//! fallback — must still return a `String`, so it can only deliver
//! well-formed text (a lossy decode surfaces as U+FFFD instead of a half
//! surrogate pair; registered as a parity deviation).

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

use crate::llm::{CancelToken, Embedder, InferenceError};

/// Upstream `CHUNK_SIZE_TOKENS` (store.ts:114): 900 tokens per chunk.
pub const CHUNK_SIZE_TOKENS: usize = 900;
/// Upstream `CHUNK_OVERLAP_TOKENS` (store.ts:115): 15% of the chunk size.
pub const CHUNK_OVERLAP_TOKENS: usize = CHUNK_SIZE_TOKENS * 15 / 100;
/// Upstream `CHUNK_SIZE_CHARS` (store.ts:117): ~4 chars per token.
pub const CHUNK_SIZE_CHARS: usize = CHUNK_SIZE_TOKENS * 4;
/// Upstream `CHUNK_OVERLAP_CHARS` (store.ts:118).
pub const CHUNK_OVERLAP_CHARS: usize = CHUNK_OVERLAP_TOKENS * 4;
/// Upstream `CHUNK_WINDOW_TOKENS` (store.ts:120): break-point search window.
pub const CHUNK_WINDOW_TOKENS: usize = 200;
/// Upstream `CHUNK_WINDOW_CHARS` (store.ts:121).
pub const CHUNK_WINDOW_CHARS: usize = CHUNK_WINDOW_TOKENS * 4;

/// Upstream's in-loop decay factor (`store.ts:390`): `findBestCutoff` is
/// always called with `0.7`.
const CUTOFF_DECAY: f64 = 0.7;

/// UTF-16 code-unit length of `text` — the counterpart of upstream
/// `String.prototype.length` used for every position/size in this module.
#[must_use]
pub fn utf16_len(text: &str) -> usize {
    text.encode_utf16().count()
}

/// A scored break-point pattern descriptor — the table behind
/// [`scan_break_points`], mirroring upstream `BREAK_PATTERNS`
/// (`store.ts:170-183`).
///
/// `pattern` keeps the upstream JS regex source for reference; the
/// heading patterns' `(?!#)` lookahead has no `regex`-crate equivalent
/// and is implemented as a `\n#+` run whose length selects the kind
/// (identical results — `(?!#)` makes each `h{k}` match exactly `k`
/// hashes, so a run of 7+ produces no heading break point at all).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct BreakPattern {
    /// Upstream JS regex source (documentation value).
    pub pattern: &'static str,
    /// Base score — higher wins at equal positions.
    pub score: u32,
    /// Pattern label (`h1`, `codeblock`, `blank`, …).
    pub kind: &'static str,
}

/// Upstream `BREAK_PATTERNS` (store.ts:170-183), in upstream order.
pub static BREAK_PATTERNS: &[BreakPattern] = &[
    BreakPattern {
        pattern: r"\n#{1}(?!#)",
        score: 100,
        kind: "h1",
    },
    BreakPattern {
        pattern: r"\n#{2}(?!#)",
        score: 90,
        kind: "h2",
    },
    BreakPattern {
        pattern: r"\n#{3}(?!#)",
        score: 80,
        kind: "h3",
    },
    BreakPattern {
        pattern: r"\n#{4}(?!#)",
        score: 70,
        kind: "h4",
    },
    BreakPattern {
        pattern: r"\n#{5}(?!#)",
        score: 60,
        kind: "h5",
    },
    BreakPattern {
        pattern: r"\n#{6}(?!#)",
        score: 50,
        kind: "h6",
    },
    BreakPattern {
        pattern: r"\n```",
        score: 80,
        kind: "codeblock",
    },
    BreakPattern {
        pattern: r"\n(?:---|\*\*\*|___)\s*\n",
        score: 60,
        kind: "hr",
    },
    BreakPattern {
        pattern: r"\n\n+",
        score: 20,
        kind: "blank",
    },
    BreakPattern {
        pattern: r"\n[-*]\s",
        score: 5,
        kind: "list",
    },
    BreakPattern {
        pattern: r"\n\d+\.\s",
        score: 5,
        kind: "numlist",
    },
    BreakPattern {
        pattern: r"\n",
        score: 1,
        kind: "newline",
    },
];

/// A potential break point with a base score — upstream `BreakPoint`
/// (`store.ts:149-153`). `pos` is a UTF-16 code-unit offset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BreakPoint {
    /// UTF-16 offset of the match start.
    pub pos: usize,
    /// Base score (higher is a better break point).
    pub score: u32,
    /// Debug label (`h1`, `blank`, `ast:func`, …). Dynamic because AST
    /// break points (P6) carry synthesized kinds.
    pub kind: Cow<'static, str>,
}

/// A code-fence region that must not be split inside — upstream
/// `CodeFenceRegion` (`store.ts:159-162`), UTF-16 offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodeFenceRegion {
    /// UTF-16 offset of the opening `\n``` `.
    pub start: usize,
    /// UTF-16 offset just after the closing `\n``` ` (or the document
    /// end when unclosed).
    pub end: usize,
}

/// Break-point strategy — upstream `ChunkStrategy` (`store.ts:303`).
/// `Auto` merges tree-sitter AST break points for supported code files;
/// that backend arrives in P6, so `Auto` currently degrades to `Regex`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ChunkStrategy {
    /// AST-aware break points merged with regex ones (degrades to
    /// [`Self::Regex`] until P6).
    Auto,
    /// Regex break points only (upstream default).
    #[default]
    Regex,
}

/// A chunk of the source text — upstream `{ text, pos }`.
/// `pos` is a UTF-16 code-unit offset into the original document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk<'a> {
    /// The sliced text.
    pub text: &'a str,
    /// UTF-16 offset where the chunk starts.
    pub pos: usize,
}

/// A token-bounded chunk — upstream `{ text, pos, tokens }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenChunk<'a> {
    /// The chunk text (borrowed from the source, or owned when produced
    /// by the `detokenize` truncation fallback).
    pub text: Cow<'a, str>,
    /// UTF-16 offset where the chunk starts.
    pub pos: usize,
    /// Token count as reported by the tokenizer at emit time.
    pub tokens: usize,
}

/// Sizing parameters for [`chunk_document`] and
/// [`chunk_document_with_break_points`] — UTF-16 code units, matching
/// upstream's `maxChars`/`overlapChars`/`windowChars`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharChunkOptions {
    /// Maximum chunk size.
    pub max_chars: usize,
    /// Overlap between consecutive chunks.
    pub overlap_chars: usize,
    /// How far back from the target end to search for break points.
    pub window_chars: usize,
}

impl Default for CharChunkOptions {
    fn default() -> Self {
        Self {
            max_chars: CHUNK_SIZE_CHARS,
            overlap_chars: CHUNK_OVERLAP_CHARS,
            window_chars: CHUNK_WINDOW_CHARS,
        }
    }
}

/// Parameters for [`chunk_document_by_tokens`] — mirrors the upstream
/// `chunkDocumentByTokens` signature (`store.ts:3351-3370`).
#[derive(Debug, Clone)]
pub struct TokenChunkOptions {
    /// Maximum tokens per chunk.
    pub max_tokens: usize,
    /// Token overlap between consecutive chunks.
    pub overlap_tokens: usize,
    /// Break-point search window in tokens.
    pub window_tokens: usize,
    /// Source path used to pick a language for AST break points.
    pub filepath: Option<String>,
    /// Break-point strategy (see [`ChunkStrategy`]).
    pub strategy: ChunkStrategy,
    /// Cooperative cancellation — upstream `signal`.
    pub cancel: CancelToken,
}

impl Default for TokenChunkOptions {
    fn default() -> Self {
        Self {
            max_tokens: CHUNK_SIZE_TOKENS,
            overlap_tokens: CHUNK_OVERLAP_TOKENS,
            window_tokens: CHUNK_WINDOW_TOKENS,
            filepath: None,
            strategy: ChunkStrategy::default(),
            cancel: CancelToken::default(),
        }
    }
}

/// Byte↔UTF-16 offset conversion table — one entry per char boundary
/// plus the end sentinel, built once per document (T2).
struct Utf16Index {
    /// `(utf16_offset, byte_offset)` sorted by construction.
    bounds: Vec<(usize, usize)>,
}

impl Utf16Index {
    fn new(text: &str) -> Self {
        let mut bounds = Vec::with_capacity(text.len() / 4 + 2);
        let mut u = 0usize;
        for (b, ch) in text.char_indices() {
            bounds.push((u, b));
            u += ch.len_utf16();
        }
        bounds.push((u, text.len()));
        Self { bounds }
    }

    fn len_utf16(&self) -> usize {
        self.bounds.last().map_or(0, |b| b.0)
    }

    /// Byte offset of `u` — the greatest char boundary whose UTF-16
    /// offset is `<= u`, so positions inside a surrogate pair round down
    /// to the pair's start.
    fn byte_of_utf16(&self, u: usize) -> usize {
        let i = self.bounds.partition_point(|&(upos, _)| upos <= u);
        self.bounds.get(i.wrapping_sub(1)).map_or(0, |b| b.1)
    }

    /// UTF-16 offset of byte `b` (`b` must be a char boundary for exact
    /// results; mid-char positions round down).
    fn utf16_of_byte(&self, b: usize) -> usize {
        let i = self.bounds.partition_point(|&(_, bpos)| bpos <= b);
        self.bounds.get(i.wrapping_sub(1)).map_or(0, |t| t.0)
    }

    /// True when `pos` sits between a high surrogate (at `pos-1`) and a
    /// low surrogate (at `pos`) — upstream `isSurrogatePairBoundary`
    /// (`store.ts:331-336`). In index terms: the char occupying `pos-1`
    /// starts exactly there and is two UTF-16 units wide.
    fn is_pair_boundary(&self, pos: usize) -> bool {
        if pos == 0 || pos >= self.len_utf16() {
            return false;
        }
        let i = self.bounds.partition_point(|&(u, _)| u < pos);
        match (self.bounds.get(i.wrapping_sub(1)), self.bounds.get(i)) {
            (Some(&(start, _)), Some(&(next, _))) => start == pos - 1 && next - start == 2,
            _ => false,
        }
    }
}

/// Byte offset of `utf16_offset` in `text` (P2-D1) — the public helper
/// for callers that need to slice `&str` by a recorded `pos`. Positions
/// inside a surrogate pair round down to the containing char's start.
///
/// Builds the conversion table on each call (O(n)); code doing many
/// lookups should slice via the chunk functions instead.
#[must_use]
pub fn utf16_to_byte_offset(text: &str, utf16_offset: usize) -> usize {
    Utf16Index::new(text).byte_of_utf16(utf16_offset)
}

/// JS `\s` spelled out exactly: Unicode `White_Space` minus U+0085 (NEL)
/// plus U+FEFF. `\p{White_Space}` alone would wrongly match U+0085,
/// which JS `\s` does not cover. Mirrors upstream (upstream-quirks.md
/// Q16).
const JS_WS: &str = r"[\u{9}-\u{d}\u{20}\u{a0}\u{1680}\u{2000}-\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}\u{feff}]";

#[allow(
    clippy::expect_used,
    reason = "the break-point patterns are fixed literals; a failure is a compile-time bug"
)]
fn compile(pattern: &str) -> Regex {
    Regex::new(pattern).expect("static break-point pattern must compile")
}

static HEADING_RE: LazyLock<Regex> = LazyLock::new(|| compile(r"\n#+"));
static FENCE_RE: LazyLock<Regex> = LazyLock::new(|| compile("\n```"));
/// The non-heading rows of [`BREAK_PATTERNS`], in upstream order.
static REST_PATTERNS: LazyLock<[(Regex, u32, &'static str); 6]> = LazyLock::new(|| {
    [
        (compile("\n```"), 80, "codeblock"),
        (
            compile(&format!(r"\n(?:---|\*\*\*|___){JS_WS}*\n")),
            60,
            "hr",
        ),
        (compile(r"\n\n+"), 20, "blank"),
        (compile(&format!(r"\n[-*]{JS_WS}")), 5, "list"),
        (compile(&format!(r"\n[0-9]+\.{JS_WS}")), 5, "numlist"),
        (compile(r"\n"), 1, "newline"),
    ]
});

/// Scan text for all potential break points — upstream `scanBreakPoints`
/// (`store.ts:190-211`). Sorted by position; the highest-scoring pattern
/// wins at a shared position.
#[must_use]
pub fn scan_break_points(text: &str) -> Vec<BreakPoint> {
    const HEADING_SCORES: [u32; 6] = [100, 90, 80, 70, 60, 50];
    const HEADING_KINDS: [&str; 6] = ["h1", "h2", "h3", "h4", "h5", "h6"];

    let index = Utf16Index::new(text);
    let mut best: HashMap<usize, BreakPoint> = HashMap::new();
    let mut offer = |pos: usize, score: u32, kind: &'static str| {
        if best.get(&pos).is_none_or(|bp| score > bp.score) {
            best.insert(
                pos,
                BreakPoint {
                    pos,
                    score,
                    kind: Cow::Borrowed(kind),
                },
            );
        }
    };

    for m in HEADING_RE.find_iter(text) {
        // `m` is `\n` + N hashes (all ASCII); upstream h1..h6 each
        // require exactly k hashes via `(?!#)` — a run of 7+ matches no
        // heading pattern at all (only the `newline` fallback).
        // Mirrors upstream (upstream-quirks.md Q15).
        let run = m.len() - 1;
        if run <= 6 {
            offer(
                index.utf16_of_byte(m.start()),
                HEADING_SCORES.get(run - 1).copied().unwrap_or(50),
                HEADING_KINDS.get(run - 1).copied().unwrap_or("h6"),
            );
        }
    }
    for (re, score, kind) in REST_PATTERNS.iter() {
        for m in re.find_iter(text) {
            offer(index.utf16_of_byte(m.start()), *score, kind);
        }
    }

    let mut points: Vec<BreakPoint> = best.into_values().collect();
    points.sort_by_key(|bp| bp.pos);
    points
}

/// Find all `\n``` ` code-fence regions — upstream `findCodeFences`
/// (`store.ts:217-239`). An unclosed fence extends to the document end.
#[must_use]
pub fn find_code_fences(text: &str) -> Vec<CodeFenceRegion> {
    let index = Utf16Index::new(text);
    let mut regions = Vec::new();
    let mut in_fence = false;
    let mut fence_start = 0usize;
    for m in FENCE_RE.find_iter(text) {
        let pos = index.utf16_of_byte(m.start());
        if in_fence {
            regions.push(CodeFenceRegion {
                start: fence_start,
                end: pos + m.as_str().encode_utf16().count(),
            });
            in_fence = false;
        } else {
            fence_start = pos;
            in_fence = true;
        }
    }
    if in_fence {
        regions.push(CodeFenceRegion {
            start: fence_start,
            end: index.len_utf16(),
        });
    }
    regions
}

/// Whether `pos` is strictly inside a fence region — upstream
/// `isInsideCodeFence` (`store.ts:244-246`).
#[must_use]
pub fn is_inside_code_fence(pos: usize, fences: &[CodeFenceRegion]) -> bool {
    fences.iter().any(|f| pos > f.start && pos < f.end)
}

/// Merge two break-point sets, keeping the highest score at each
/// position — upstream `mergeBreakPoints` (`store.ts:309-324`).
/// Result sorted by position; `a` wins ties.
#[must_use]
pub fn merge_break_points(a: &[BreakPoint], b: &[BreakPoint]) -> Vec<BreakPoint> {
    let mut best: HashMap<usize, BreakPoint> = HashMap::with_capacity(a.len() + b.len());
    for bp in a.iter().chain(b.iter()) {
        if best.get(&bp.pos).is_none_or(|x| bp.score > x.score) {
            best.insert(bp.pos, bp.clone());
        }
    }
    let mut points: Vec<BreakPoint> = best.into_values().collect();
    points.sort_by_key(|bp| bp.pos);
    points
}

/// Best cut position with squared-distance decay — upstream
/// `findBestCutoff` (`store.ts:261-297`). Positions inside code fences
/// are skipped; returns `target_pos` when nothing in the window wins.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "upstream arithmetic is IEEE-754 doubles; positions fit well inside 2^53"
)]
pub fn find_best_cutoff(
    break_points: &[BreakPoint],
    target_pos: usize,
    window_chars: usize,
    decay_factor: f64,
    code_fences: &[CodeFenceRegion],
) -> usize {
    let window_start = target_pos.saturating_sub(window_chars);
    let mut best_score = -1.0_f64;
    let mut best_pos = target_pos;
    for bp in break_points {
        if bp.pos < window_start {
            continue;
        }
        if bp.pos > target_pos {
            break;
        }
        if is_inside_code_fence(bp.pos, code_fences) {
            continue;
        }
        let distance = (target_pos - bp.pos) as f64;
        // Squared decay (upstream): 1.0 at target → 1-decay at the edge.
        let normalized = distance / window_chars as f64;
        #[allow(
            clippy::suboptimal_flops,
            reason = "must match upstream's exact `1.0 - d*d*decay` rounding order"
        )]
        let multiplier = 1.0 - normalized * normalized * decay_factor;
        let score = f64::from(bp.score) * multiplier;
        if score > best_score {
            best_score = score;
            best_pos = bp.pos;
        }
    }
    best_pos
}

/// Nudge `pos` off a surrogate-pair boundary — upstream
/// `adjustSurrogateBoundary` (`store.ts:356-360`). Retreats by one unit
/// when that keeps `pos > floor`, else advances by one.
fn adjust_surrogate_boundary(index: &Utf16Index, pos: usize, floor: usize) -> usize {
    if !index.is_pair_boundary(pos) {
        return pos;
    }
    if pos - 1 > floor { pos - 1 } else { pos + 1 }
}

/// Core chunk algorithm on precomputed break points — upstream
/// `chunkDocumentWithBreakPoints` (`store.ts:366-422`). All positions
/// and options are UTF-16 code units.
///
/// Like upstream, `max_chars <= 0` never terminates
/// (upstream-quirks.md Q17).
#[must_use]
pub fn chunk_document_with_break_points<'a>(
    content: &'a str,
    break_points: &[BreakPoint],
    code_fences: &[CodeFenceRegion],
    options: &CharChunkOptions,
) -> Vec<Chunk<'a>> {
    let index = Utf16Index::new(content);
    let len = index.len_utf16();
    if len <= options.max_chars {
        return vec![Chunk {
            text: content,
            pos: 0,
        }];
    }

    let mut chunks = Vec::new();
    let mut char_pos = 0usize;
    while char_pos < len {
        let target_end = (char_pos + options.max_chars).min(len);
        let mut end_pos = target_end;
        if end_pos < len {
            let cutoff = find_best_cutoff(
                break_points,
                target_end,
                options.window_chars,
                CUTOFF_DECAY,
                code_fences,
            );
            if cutoff > char_pos && cutoff <= target_end {
                end_pos = cutoff;
            }
        }
        if end_pos <= char_pos {
            end_pos = (char_pos + options.max_chars).min(len);
        }
        end_pos = adjust_surrogate_boundary(&index, end_pos, char_pos);

        let start_byte = index.byte_of_utf16(char_pos);
        let end_byte = index.byte_of_utf16(end_pos);
        chunks.push(Chunk {
            text: &content[start_byte..end_byte],
            pos: char_pos,
        });
        if end_pos >= len {
            break;
        }

        char_pos = end_pos.saturating_sub(options.overlap_chars);
        let last_chunk_pos = chunks.last().map_or(0, |c| c.pos);
        if char_pos <= last_chunk_pos {
            char_pos = end_pos;
        }
        char_pos = adjust_surrogate_boundary(&index, char_pos, last_chunk_pos);
    }
    chunks
}

/// Regex-breakpoint chunking — upstream `chunkDocument`
/// (`store.ts:3152-3161`).
#[must_use]
pub fn chunk_document<'a>(content: &'a str, options: &CharChunkOptions) -> Vec<Chunk<'a>> {
    let break_points = scan_break_points(content);
    let code_fences = find_code_fences(content);
    chunk_document_with_break_points(content, &break_points, &code_fences, options)
}

/// Strategy-dispatched chunking — upstream `chunkDocumentAsync`
/// (`store.ts:3171-3192`, minus the await).
///
/// `Auto` + `filepath` merges AST break points in upstream; the AST
/// backend arrives in P6, so `Auto` currently produces the same chunks
/// as `Regex`.
#[must_use]
pub fn chunk_document_with_strategy<'a>(
    content: &'a str,
    filepath: Option<&str>,
    strategy: ChunkStrategy,
    options: &CharChunkOptions,
) -> Vec<Chunk<'a>> {
    // P6: when `strategy == Auto` and `filepath` maps to a supported
    // language, merge `ast_break_points(content, filepath)` via
    // `merge_break_points` before delegating.
    let _ = (filepath, strategy);
    chunk_document(content, options)
}

/// `clampOverlapChars` (store.ts:3276-3279).
fn clamp_overlap_chars(value: f64, max_chars: usize) -> usize {
    if max_chars <= 1 {
        return 0;
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "upstream Math.floor; the value is clamped into [0, max_chars) below"
    )]
    let floored = value.floor() as usize;
    floored.min(max_chars - 1)
}

/// Recursive token-budget enforcement — upstream
/// `pushChunkWithinTokenLimit` (`store.ts:3281-3342`).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "upstream arithmetic is IEEE-754 doubles; the truncation mirrors Math.floor"
)]
fn push_within_token_limit<'a, E: Embedder + ?Sized>(
    embedder: &E,
    text: &'a str,
    pos: usize,
    options: &TokenChunkOptions,
    out: &mut Vec<TokenChunk<'a>>,
) -> Result<(), InferenceError> {
    if options.cancel.is_cancelled() {
        return Ok(());
    }

    let tokens = embedder.tokenize(text)?;
    let text_units = utf16_len(text);
    // `text_units <= 1` mirrors upstream `text.length <= 1`; Rust &str is
    // always well-formed so `stripUnpairedSurrogates` is unnecessary.
    if tokens.len() <= options.max_tokens || text_units <= 1 {
        if text.is_empty() {
            return Ok(());
        }
        out.push(TokenChunk {
            text: Cow::Borrowed(text),
            pos,
            tokens: tokens.len(),
        });
        return Ok(());
    }

    let chars_per_token = text_units as f64 / tokens.len() as f64;
    let mut safe_max_chars = (options.max_tokens as f64 * chars_per_token * 0.95).floor();
    if !safe_max_chars.is_finite() || safe_max_chars < 1.0 {
        safe_max_chars = (text_units as f64 / 2.0).floor();
    }
    let mut safe_max_chars = (safe_max_chars as usize).clamp(1, text_units - 1);

    let overlap_chars = clamp_overlap_chars(
        options.overlap_tokens as f64 * 3.0 * chars_per_token / 2.0,
        safe_max_chars,
    );
    let window_chars = (options.window_tokens as f64 * 3.0 * chars_per_token / 2.0)
        .floor()
        .max(0.0) as usize;

    let mut sub_chunks = chunk_document(
        text,
        &CharChunkOptions {
            max_chars: safe_max_chars,
            overlap_chars,
            window_chars,
        },
    );
    // Pathological single-line blobs can produce no meaningful break-point
    // progress; halve so every recursion step strictly shrinks.
    if sub_chunks.len() <= 1
        || sub_chunks
            .first()
            .is_some_and(|c| utf16_len(c.text) == text_units)
    {
        safe_max_chars = (text_units / 2).max(1);
        sub_chunks = chunk_document(
            text,
            &CharChunkOptions {
                max_chars: safe_max_chars,
                overlap_chars: 0,
                window_chars: 0,
            },
        );
    }
    if sub_chunks.len() <= 1
        || sub_chunks
            .first()
            .is_some_and(|c| utf16_len(c.text) == text_units)
    {
        let keep = options.max_tokens.max(1).min(tokens.len());
        let fallback_tokens = tokens.get(..keep).unwrap_or_default();
        // `detokenize` must return a well-formed String, so upstream's
        // lone-surrogate drop cannot trigger here; an empty result is
        // still skipped like upstream's stripped-empty case.
        let truncated = embedder.detokenize(fallback_tokens)?;
        if truncated.is_empty() {
            return Ok(());
        }
        out.push(TokenChunk {
            text: Cow::Owned(truncated),
            pos,
            tokens: fallback_tokens.len(),
        });
        return Ok(());
    }

    for sub in &sub_chunks {
        push_within_token_limit(embedder, sub.text, pos + sub.pos, options, out)?;
    }
    Ok(())
}

/// Token-bounded chunking — upstream `chunkDocumentByTokens` /
/// `chunkDocumentByTokensWithLlm` (`store.ts:3253-3370`). The tokenizer
/// comes from the injected [`Embedder`].
///
/// # Errors
/// [`InferenceError::Backend`] on tokenizer failure.
#[allow(
    clippy::arithmetic_side_effects,
    reason = "chars-per-token estimate multiplications mirror upstream"
)]
pub fn chunk_document_by_tokens<'a, E: Embedder + ?Sized>(
    embedder: &E,
    content: &'a str,
    options: &TokenChunkOptions,
) -> Result<Vec<TokenChunk<'a>>, InferenceError> {
    // Moderate estimate (prose ~4, code ~2, mixed ~3 chars/token);
    // oversize chunks are re-split with the actual ratio below.
    const AVG_CHARS_PER_TOKEN: usize = 3;
    let char_options = CharChunkOptions {
        max_chars: options.max_tokens * AVG_CHARS_PER_TOKEN,
        overlap_chars: options.overlap_tokens * AVG_CHARS_PER_TOKEN,
        window_chars: options.window_tokens * AVG_CHARS_PER_TOKEN,
    };
    let char_chunks = chunk_document_with_strategy(
        content,
        options.filepath.as_deref(),
        options.strategy,
        &char_options,
    );

    let mut results = Vec::new();
    for chunk in char_chunks {
        push_within_token_limit(embedder, chunk.text, chunk.pos, options, &mut results)?;
    }
    Ok(results)
}
