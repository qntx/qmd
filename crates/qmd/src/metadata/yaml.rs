//! Frontmatter YAML composer on `granit-parser` events — a deliberate
//! mirror of npm `yaml` 2.9 (`parse(doc, { maxAliasCount: 100 })` with
//! the default core schema), the library upstream uses. It reproduces
//! the pieces upstream metadata extraction depends on: core-schema
//! scalar resolution, explicit-tag handling, `uniqueKeys` duplicate
//! detection on resolved scalar values, forward-only alias resolution
//! with the alias-count resource guard, JavaScript key stringification
//! (`stringifyKey`), and JavaScript object key ordering.
//!
//! Deliberate deviations, all inside error paths where callers only
//! assert the `invalid frontmatter YAML: ` prefix: granit's own syntax
//! errors are surfaced through `info()`, and an alias inside a
//! collection map key stringifies to its resolved value rather than
//! `*name` (npm yaml never exercises that path with an unresolved
//! anchor — granit rejects it earlier).

use std::collections::HashMap;
use std::sync::LazyLock;

use granit_parser::{Event, Parser, ScalarStyle, Tag};
use regex::Regex;

use crate::metadata::js_num_to_string;

/// Upstream `METADATA_LIMITS.maxYamlAliasCount` — npm `yaml`
/// `maxAliasCount` default.
const MAX_ALIAS_COUNT: usize = 100;

/// JavaScript value tree — what npm `yaml`'s `toJS` produces for the
/// core schema, plus [`Js::Unsupported`] for values JSON cannot
/// represent (timestamps, binary, sets, …), which metadata
/// normalization rejects as "unsupported value type".
#[derive(Debug, Clone)]
pub(super) enum Js {
    /// `null`
    Null,
    /// `boolean`
    Bool(bool),
    /// `number`
    Num(f64),
    /// `string`
    Str(String),
    /// `Array`
    Seq(Vec<Self>),
    /// Plain object — pairs in document order, deduplicated like JS
    /// assignment (later value wins, first position kept).
    Map(JsMap),
    /// Non-JSON object (Date, Buffer, Set, …).
    Unsupported,
}

/// A JavaScript-like object: key → value in insertion order.
#[derive(Debug, Clone)]
pub(super) struct JsMap {
    entries: Vec<(String, Js)>,
}

impl JsMap {
    /// `map[key]`.
    pub(super) fn get(&self, key: &str) -> Option<&Js> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Keys in JavaScript object order: canonical array-index keys
    /// (decimal integers 0..=4294967294 without leading zeros) first,
    /// ascending, then all others in insertion order. Mirrors the
    /// enumeration order `JSON.stringify` uses upstream.
    pub(super) fn js_ordered_keys(&self) -> Vec<&str> {
        let mut indexed: Vec<(u64, &str)> = Vec::new();
        let mut rest: Vec<&str> = Vec::new();
        for (k, _) in &self.entries {
            match canonical_array_index(k) {
                Some(i) => indexed.push((i, k)),
                None => rest.push(k),
            }
        }
        indexed.sort_by_key(|(i, _)| *i);
        indexed.into_iter().map(|(_, k)| k).chain(rest).collect()
    }
}

/// Whether `s` is a canonical JS array-index key.
fn canonical_array_index(s: &str) -> Option<u64> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) || (s.len() > 1 && s.starts_with('0'))
    {
        return None;
    }
    s.parse::<u64>().ok().filter(|&i| i <= 4_294_967_294)
}

/// `YAMLParseError` text for a second document in the stream — npm
/// yaml's `parse` accepts a single document.
const MULTI_DOC_ERROR: &str = "Source contains multiple documents; please use YAML.parseAllDocuments or YAML.parseAll to parse them";

/// Compose and convert `yaml_source` to a [`Js`] value. Errors carry
/// messages suitable for the `invalid frontmatter YAML: ` prefix.
///
/// # Errors
/// Syntax errors, multiple documents, unresolved anchors, duplicate
/// keys, alias-count exhaustion and invalid `!!timestamp` scalars.
pub(super) fn parse_frontmatter(yaml_source: &str) -> Result<Js, String> {
    let nodes = compose(yaml_source)?;
    let Some(root) = nodes.root else {
        return Ok(Js::Null);
    };
    let mut composer = Composer {
        nodes: &nodes.arena,
        anchors: HashMap::new(),
    };
    composer.js_value(root)
}

/// Node arena: granit events nest, but npm yaml composes the whole
/// document before converting, so anchors may only be referenced by
/// aliases appearing textually later.
struct NodeTree {
    arena: Vec<Node>,
    root: Option<usize>,
}

enum Node {
    Scalar {
        src: String,
        style: ScalarStyle,
        anchor: usize,
        tag: YamlTag,
    },
    Seq {
        items: Vec<usize>,
        anchor: usize,
        tag: YamlTag,
    },
    /// `(key, value)` node-index pairs.
    Map {
        pairs: Vec<(usize, usize)>,
        anchor: usize,
        tag: YamlTag,
    },
    Alias(usize),
}

impl Node {
    const fn anchor(&self) -> usize {
        match self {
            Self::Scalar { anchor, .. } | Self::Seq { anchor, .. } | Self::Map { anchor, .. } => {
                *anchor
            }
            Self::Alias(_) => 0,
        }
    }
}

/// Resolved tag classification — only tags that change semantics are
/// distinguished; everything else behaves like npm yaml's unresolved
/// tag fallback.
#[derive(Debug, Clone, Copy)]
enum YamlTag {
    /// No tag.
    None,
    /// `!` — forces string.
    NonSpecific,
    /// `!!str`
    Str,
    /// `!!null`
    Null,
    /// `!!bool`
    Bool,
    /// `!!int`
    Int,
    /// `!!float`
    Float,
    /// `!!binary`
    Binary,
    /// `!!timestamp`
    Timestamp,
    /// `!!merge`
    Merge,
    /// `!!set`
    Set,
    /// `!!omap` / `!!pairs`
    OrderedPairs,
    /// `!!map`, `!!seq`, local and unknown tags — ignored.
    Other,
}

/// npm yaml's `!!omap` and `!!pairs` share a sequence shape (`Set`-like
/// single-pair maps); both are grouped as [`YamlTag::OrderedPairs`]
/// since neither reaches metadata values.
fn classify(tag: &Tag) -> YamlTag {
    if let Some(name) = tag.suffix_in_namespace("tag:yaml.org,2002:") {
        return match name.as_ref() {
            "str" => YamlTag::Str,
            "null" => YamlTag::Null,
            "bool" => YamlTag::Bool,
            "int" => YamlTag::Int,
            "float" => YamlTag::Float,
            "binary" => YamlTag::Binary,
            "timestamp" => YamlTag::Timestamp,
            "merge" => YamlTag::Merge,
            "set" => YamlTag::Set,
            "omap" | "pairs" => YamlTag::OrderedPairs,
            _ => YamlTag::Other,
        };
    }
    if tag.handle() == "!" && tag.suffix().is_empty() {
        return YamlTag::NonSpecific;
    }
    YamlTag::Other
}

/// Open collection being filled by the event stream. `flow` records the
/// `StructureStyle`: a *flow* close consumes `]`/`}` source tokens
/// (ending a pending scalar for the trailing-`\r` quirk), while a block
/// collection ends implicitly after the last scalar.
enum Open {
    Seq {
        items: Vec<usize>,
        anchor: usize,
        tag: YamlTag,
        flow: bool,
    },
    Map {
        pairs: Vec<(usize, usize)>,
        pending_key: Option<usize>,
        anchor: usize,
        tag: YamlTag,
        flow: bool,
    },
}

/// npm yaml quirk: a `\r` not followed by `\n` is buffered by the
/// lexer waiting for a possible `\n`; at end of input it is flushed as
/// scalar content. So a trailing bare `\r` appends to the final plain
/// scalar (`a: 1\r` → `"1\r"`), and is an "Unexpected scalar" error
/// after a completed node.
#[derive(Clone, Copy)]
enum Tail {
    /// Last event produced a plain scalar — index into the arena.
    PlainScalar(usize),
    /// Quoted/block scalar or a completed collection.
    Other,
    /// Nothing parsed yet.
    None,
}

/// Compose granit events into a [`NodeTree`].
fn compose(yaml_source: &str) -> Result<NodeTree, String> {
    let mut arena: Vec<Node> = Vec::new();
    let mut stack: Vec<Open> = Vec::new();
    let mut root: Option<usize> = None;
    let mut seen_document = false;
    let mut tail = Tail::None;

    for event in Parser::new_from_str(yaml_source) {
        let (event, _span) = event.map_err(|e| e.info())?;
        match event {
            Event::DocumentStart(..) => {
                if seen_document {
                    return Err(MULTI_DOC_ERROR.to_owned());
                }
                seen_document = true;
            }
            Event::Scalar(src, style, anchor, tag) => {
                arena.push(Node::Scalar {
                    src: src.into_owned(),
                    style,
                    anchor,
                    tag: tag.map_or(YamlTag::None, |t| classify(&t)),
                });
                tail = if matches!(style, ScalarStyle::Plain) {
                    Tail::PlainScalar(arena.len() - 1)
                } else {
                    Tail::Other
                };
                attach(&mut stack, &mut root, arena.len() - 1)?;
            }
            Event::Alias(anchor) => {
                arena.push(Node::Alias(anchor));
                tail = Tail::Other;
                attach(&mut stack, &mut root, arena.len() - 1)?;
            }
            Event::SequenceStart(style, anchor, tag) => stack.push(Open::Seq {
                items: Vec::new(),
                anchor,
                tag: tag.map_or(YamlTag::None, |t| classify(&t)),
                flow: matches!(style, granit_parser::StructureStyle::Flow),
            }),
            Event::SequenceEnd => {
                let Some(Open::Seq {
                    items,
                    anchor,
                    tag,
                    flow,
                }) = stack.pop()
                else {
                    return Err("unexpected sequence end".to_owned());
                };
                arena.push(Node::Seq { items, anchor, tag });
                if flow {
                    tail = Tail::Other;
                }
                attach(&mut stack, &mut root, arena.len() - 1)?;
            }
            Event::MappingStart(style, anchor, tag) => stack.push(Open::Map {
                pairs: Vec::new(),
                pending_key: None,
                anchor,
                tag: tag.map_or(YamlTag::None, |t| classify(&t)),
                flow: matches!(style, granit_parser::StructureStyle::Flow),
            }),
            Event::MappingEnd => {
                let Some(Open::Map {
                    mut pairs,
                    pending_key,
                    anchor,
                    tag,
                    flow,
                }) = stack.pop()
                else {
                    return Err("unexpected mapping end".to_owned());
                };
                if let Some(dangling) = pending_key {
                    // npm yaml pairs a trailing key with an empty value.
                    arena.push(Node::Scalar {
                        src: String::new(),
                        style: ScalarStyle::Plain,
                        anchor: 0,
                        tag: YamlTag::None,
                    });
                    pairs.push((dangling, arena.len() - 1));
                }
                arena.push(Node::Map { pairs, anchor, tag });
                if flow {
                    tail = Tail::Other;
                }
                attach(&mut stack, &mut root, arena.len() - 1)?;
            }
            // Stream markers, comments and future additions carry no
            // node data.
            _ => {}
        }
    }
    if !stack.is_empty() {
        return Err("unexpected end of stream".to_owned());
    }
    apply_trailing_cr(&mut arena, tail, yaml_source)?;
    Ok(NodeTree { arena, root })
}

/// Attach a finished node to the innermost collection, or set it as the
/// document root.
fn attach(stack: &mut [Open], root: &mut Option<usize>, idx: usize) -> Result<(), String> {
    match stack.last_mut() {
        None => {
            if root.is_some() {
                return Err(MULTI_DOC_ERROR.to_owned());
            }
            *root = Some(idx);
        }
        Some(Open::Seq { items, .. }) => items.push(idx),
        Some(Open::Map {
            pairs, pending_key, ..
        }) => match pending_key.take() {
            Some(key) => pairs.push((key, idx)),
            None => *pending_key = Some(idx),
        },
    }
    Ok(())
}

/// Apply the trailing-bare-`\r` quirk (see [`Tail`]) once the stream is
/// fully composed.
fn apply_trailing_cr(arena: &mut [Node], tail: Tail, yaml_source: &str) -> Result<(), String> {
    if !yaml_source.ends_with('\r') {
        return Ok(());
    }
    let carriage = "\r".repeat(yaml_source.chars().rev().take_while(|c| *c == '\r').count());
    match tail {
        Tail::PlainScalar(idx) => {
            if let Some(Node::Scalar { src, .. }) = arena.get_mut(idx) {
                src.push_str(&carriage);
            }
            Ok(())
        }
        Tail::Other => Err("Unexpected scalar at node end".to_owned()),
        Tail::None => Ok(()),
    }
}

/// npm `nodes/toJS.ts` conversion with `nodes/Alias.js` anchor data.
struct Composer<'a> {
    nodes: &'a [Node],
    /// anchor id → `AnchorData` (registered at first conversion,
    /// `count` starting at 1 like upstream).
    anchors: HashMap<usize, AnchorData>,
}

/// npm yaml `Alias.source.anchored` bookkeeping.
struct AnchorData {
    count: usize,
    alias_count: usize,
    node: usize,
    res: Option<Js>,
}

impl Composer<'_> {
    /// `toJS(node)` — registers the anchor before converting so
    /// self-references inside collections resolve to the in-flight node.
    fn js_value(&mut self, idx: usize) -> Result<Js, String> {
        match self.nodes.get(idx) {
            Some(Node::Alias(anchor)) => self.resolve_alias(*anchor),
            Some(node) => {
                let anchor = node.anchor();
                if anchor != 0 {
                    self.anchors.insert(
                        anchor,
                        AnchorData {
                            count: 1,
                            alias_count: 0,
                            node: idx,
                            res: None,
                        },
                    );
                }
                let res = self.convert_node(idx)?;
                if anchor != 0
                    && let Some(data) = self.anchors.get_mut(&anchor)
                {
                    data.res = Some(res.clone());
                }
                Ok(res)
            }
            None => Err("internal: missing node".to_owned()),
        }
    }

    /// npm `Alias.toJSON`/`resolve`: bump the anchor count, lazily
    /// compute `aliasCount`, then enforce `count * aliasCount <= 100`.
    fn resolve_alias(&mut self, anchor: usize) -> Result<Js, String> {
        let needs_count = {
            let Some(entry) = self.anchors.get_mut(&anchor) else {
                return Err(format!(
                    "Unresolved alias (the anchor must be set before the alias): {anchor}"
                ));
            };
            entry.count += 1;
            (entry.alias_count == 0).then_some(entry.node)
        };
        if let Some(node) = needs_count {
            let alias_count = self.get_alias_count(node);
            self.anchors
                .entry(anchor)
                .and_modify(|entry| entry.alias_count = alias_count);
        }
        let Some(data) = self.anchors.get(&anchor) else {
            return Err("internal: missing anchor".to_owned());
        };
        if data.count.saturating_mul(data.alias_count) > MAX_ALIAS_COUNT {
            return Err("Excessive alias count indicates a resource exhaustion attack".to_owned());
        }
        if let Some(res) = data.res.clone() {
            return Ok(res);
        }
        // npm yaml registers a collection's result before filling it, so a
        // self-referencing alias resolves to the in-flight (circular) node.
        // A circular value can never be valid metadata — normalization
        // rejects nested structures — so a shape placeholder preserves the
        // upstream error path (`[*x]` → "nested arrays are not supported",
        // `{k: *x}` → "unsupported value type").
        match self.nodes.get(data.node) {
            Some(Node::Seq { .. }) => Ok(Js::Seq(vec![Js::Unsupported])),
            Some(Node::Map { .. }) => Ok(Js::Map(JsMap {
                entries: Vec::new(),
            })),
            _ => Err("This should not happen: Alias anchor was not resolved?".to_owned()),
        }
    }

    /// npm `getAliasCount` — recursive over collections; an alias item
    /// contributes its anchor's `count * aliasCount`.
    fn get_alias_count(&self, idx: usize) -> usize {
        match self.nodes.get(idx) {
            Some(Node::Alias(anchor)) => self
                .anchors
                .get(anchor)
                .map_or(0, |d| d.count * d.alias_count),
            Some(Node::Seq { items, .. }) => items
                .iter()
                .map(|i| self.get_alias_count(*i))
                .max()
                .unwrap_or(0),
            Some(Node::Map { pairs, .. }) => pairs
                .iter()
                .map(|(k, v)| self.get_alias_count(*k).max(self.get_alias_count(*v)))
                .max()
                .unwrap_or(0),
            Some(Node::Scalar { .. }) => 1,
            None => 0,
        }
    }

    fn convert_node(&mut self, idx: usize) -> Result<Js, String> {
        match self.nodes.get(idx) {
            Some(Node::Scalar {
                src, style, tag, ..
            }) => resolve_scalar(src, *style, *tag),
            Some(Node::Seq { items, tag, .. }) => match tag {
                YamlTag::Set => Err("Expected a mapping for this tag".to_owned()),
                YamlTag::OrderedPairs => {
                    // Still convert items so nested aliases resolve.
                    for i in items {
                        self.js_value(*i)?;
                    }
                    Ok(Js::Unsupported)
                }
                _ => {
                    let mut out = Vec::with_capacity(items.len());
                    for i in items {
                        out.push(self.js_value(*i)?);
                    }
                    Ok(Js::Seq(out))
                }
            },
            Some(Node::Map { pairs, tag, .. }) => match tag {
                YamlTag::OrderedPairs => Err("Expected a sequence for this tag".to_owned()),
                _ => self.convert_map(pairs, *tag),
            },
            _ => Err("internal: missing node".to_owned()),
        }
    }

    /// `addPairToJSMap` per pair: check `uniqueKeys` on the resolved
    /// scalar key, convert key then value, assign JS-style (later value
    /// wins, first position kept). `!!set` requires all-null values.
    fn convert_map(&mut self, pairs: &[(usize, usize)], tag: YamlTag) -> Result<Js, String> {
        let mut entries: Vec<(String, Js)> = Vec::new();
        let mut key_scalars: Vec<KeyScalar> = Vec::new();
        for (k_idx, v_idx) in pairs.iter().copied() {
            let key_scalar = self.key_scalar(k_idx);
            if key_scalars
                .iter()
                .any(|prev| scalar_key_eq(prev, &key_scalar))
            {
                return Err("Map keys must be unique".to_owned());
            }
            key_scalars.push(key_scalar);
            let skey = self.stringify_key(k_idx)?;
            let value = self.js_value(v_idx)?;
            if let Some((_, existing)) = entries.iter_mut().find(|(k, _)| *k == skey) {
                *existing = value;
            } else {
                entries.push((skey, value));
            }
        }
        if matches!(tag, YamlTag::Set) {
            if entries.iter().all(|(_, v)| matches!(v, Js::Null)) {
                return Ok(Js::Unsupported);
            }
            return Err("Set items must all have null values".to_owned());
        }
        Ok(Js::Map(JsMap { entries }))
    }

    /// Resolved scalar identity for `uniqueKeys` — npm compares the
    /// composed scalar `value`s (`a === b || (both scalars &&
    /// a.value === b.value)`), so `1` ≠ `"1"` but `1` == `1.0`; NaN is
    /// never equal. Non-scalar keys and aliases never collide.
    fn key_scalar(&self, idx: usize) -> KeyScalar {
        match self.nodes.get(idx) {
            Some(Node::Scalar {
                src, style, tag, ..
            }) => match resolve_scalar(src, *style, *tag) {
                Ok(Js::Null) => KeyScalar::Null,
                Ok(Js::Bool(b)) => KeyScalar::Bool(b),
                Ok(Js::Num(n)) => KeyScalar::Num(n),
                Ok(Js::Str(s)) => KeyScalar::Str(s),
                _ => KeyScalar::Other,
            },
            _ => KeyScalar::Other,
        }
    }

    /// npm `stringifyKey`: `String(value)` for non-object values
    /// (null → `""`), flow `toString` for collection keys, alias → its
    /// resolved value first.
    fn stringify_key(&mut self, idx: usize) -> Result<String, String> {
        let jskey = self.js_value(idx)?;
        match jskey {
            Js::Null => Ok(String::new()),
            Js::Str(s) => Ok(s),
            Js::Num(n) => Ok(js_num_to_string(n)),
            Js::Bool(b) => Ok(if b { "true" } else { "false" }.to_owned()),
            Js::Seq(_) | Js::Map(_) => self.flow_key(idx),
            Js::Unsupported => Err("Cannot convert a Symbol value to a string".to_owned()),
        }
    }

    /// npm collection-node `toString` in flow context for map keys:
    /// `[ a, b ]` / `{ a: 1 }`, scalars keeping their quote style.
    fn flow_key(&self, idx: usize) -> Result<String, String> {
        match self.nodes.get(idx) {
            Some(Node::Scalar {
                src, style, tag, ..
            }) => flow_scalar(src, *style, *tag),
            Some(Node::Seq { items, .. }) => {
                let mut parts = Vec::with_capacity(items.len());
                for item in items {
                    parts.push(self.flow_key(*item)?);
                }
                Ok(format!("[ {} ]", parts.join(", ")))
            }
            Some(Node::Map { pairs, .. }) => {
                let mut parts = Vec::with_capacity(pairs.len());
                for (k, v) in pairs {
                    parts.push(format!("{}: {}", self.flow_key(*k)?, self.flow_key(*v)?));
                }
                Ok(format!("{{ {} }}", parts.join(", ")))
            }
            Some(Node::Alias(anchor)) => {
                // npm prints `*name`; the resolved value is equivalent
                // for key purposes and granit hides the name.
                let res = self
                    .anchors
                    .get(anchor)
                    .and_then(|d| d.res.clone())
                    .ok_or_else(|| {
                        format!(
                            "Unresolved alias (the anchor must be set before the alias): {anchor}"
                        )
                    })?;
                Ok(match res {
                    Js::Str(s) => s,
                    Js::Num(n) => js_num_to_string(n),
                    Js::Bool(b) => if b { "true" } else { "false" }.to_owned(),
                    Js::Null => "null".to_owned(),
                    Js::Seq(_) | Js::Map(_) | Js::Unsupported => String::new(),
                })
            }
            None => Err("internal: missing node".to_owned()),
        }
    }
}

/// Scalar identity for `uniqueKeys` (see [`Composer::key_scalar`]).
enum KeyScalar {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    /// Non-scalar or alias — `===` on distinct objects is false.
    Other,
}

/// `a === b || (both scalars && a.value === b.value)`.
fn scalar_key_eq(a: &KeyScalar, b: &KeyScalar) -> bool {
    match (a, b) {
        (KeyScalar::Null, KeyScalar::Null) => true,
        (KeyScalar::Bool(x), KeyScalar::Bool(y)) => x == y,
        (KeyScalar::Num(x), KeyScalar::Num(y)) => x == y,
        (KeyScalar::Str(x), KeyScalar::Str(y)) => x == y,
        _ => false,
    }
}

// Core-schema scalar resolution — npm yaml `schema/core/*.js` and
// `schema/common/null.js`. Tests run in this exact order on the plain
// scalar source; quoted and block scalars are always strings.
static NULL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?:~|[Nn]ull|NULL)?$").unwrap_or_else(|_| unreachable!()));
static BOOL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:[Tt]rue|TRUE|[Ff]alse|FALSE)$").unwrap_or_else(|_| unreachable!())
});
static INT_OCT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^0o[0-7]+$").unwrap_or_else(|_| unreachable!()));
static INT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[-+]?[0-9]+$").unwrap_or_else(|_| unreachable!()));
static INT_HEX_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^0x[0-9a-fA-F]+$").unwrap_or_else(|_| unreachable!()));
static FLOAT_NAN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:[-+]?\.(?:inf|Inf|INF)|\.nan|\.NaN|\.NAN)$").unwrap_or_else(|_| unreachable!())
});
static FLOAT_EXP_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)[eE][-+]?[0-9]+$")
        .unwrap_or_else(|_| unreachable!())
});
static FLOAT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[-+]?(?:\.[0-9]+|[0-9]+\.[0-9]*)$").unwrap_or_else(|_| unreachable!())
});
// npm yaml `schema/yaml-1.1/timestamp.js` `test` — date with optional
// time and zone; month/day/hour are 1–2 digits.
static TIMESTAMP_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}(?:(?:t|T|[ \t]+)[0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}(?:\.[0-9]+)?(?:[ \t]*(?:Z|[-+][012]?[0-9](?::[0-9]{2})?))?)?$",
    )
    .unwrap_or_else(|_| unreachable!())
});

/// Resolve a scalar node to a [`Js`] value — npm
/// `compose/compose-scalar.js` `findScalarTagByName` semantics: plain
/// untagged scalars go through the core-schema tag tests in order;
/// explicit tags test only their own type and fall back to string.
fn resolve_scalar(src: &str, style: ScalarStyle, tag: YamlTag) -> Result<Js, String> {
    match tag {
        YamlTag::None => {
            if matches!(style, ScalarStyle::Plain) {
                Ok(resolve_plain(src))
            } else {
                Ok(Js::Str(src.to_owned()))
            }
        }
        // `!` forces string; `!!str`, unknown tags and collection tags
        // on scalars are all unresolved → string.
        YamlTag::NonSpecific
        | YamlTag::Str
        | YamlTag::Other
        | YamlTag::Set
        | YamlTag::OrderedPairs => Ok(Js::Str(src.to_owned())),
        YamlTag::Null => Ok(if NULL_RE.is_match(src) {
            Js::Null
        } else {
            Js::Str(src.to_owned())
        }),
        YamlTag::Bool => Ok(if BOOL_RE.is_match(src) {
            Js::Bool(src.starts_with('t') || src.starts_with('T'))
        } else {
            Js::Str(src.to_owned())
        }),
        YamlTag::Int => Ok(resolve_int_tagged(src).unwrap_or_else(|| Js::Str(src.to_owned()))),
        YamlTag::Float => Ok(resolve_float_tagged(src).unwrap_or_else(|| Js::Str(src.to_owned()))),
        YamlTag::Binary | YamlTag::Merge => Ok(Js::Unsupported),
        YamlTag::Timestamp => {
            if TIMESTAMP_RE.is_match(src) {
                Ok(Js::Unsupported)
            } else {
                Err("!!timestamp expects a date, starting with yyyy-mm-dd".to_owned())
            }
        }
    }
}

/// npm `resolvePlainScalar` tag-test order: null, bool, intOct, int,
/// intHex, floatNaN, floatExp, float, else string.
fn resolve_plain(s: &str) -> Js {
    if NULL_RE.is_match(s) {
        return Js::Null;
    }
    if BOOL_RE.is_match(s) {
        return Js::Bool(s.starts_with('t') || s.starts_with('T'));
    }
    if let Some(num) = resolve_int_tagged(s) {
        return num;
    }
    if let Some(num) = resolve_float_tagged(s) {
        return num;
    }
    Js::Str(s.to_owned())
}

/// `!!int` resolution — octal, decimal, hexadecimal in tag order.
fn resolve_int_tagged(s: &str) -> Option<Js> {
    if INT_OCT_RE.is_match(s) {
        return Some(Js::Num(parse_radix(s.get(2..).unwrap_or_default(), 8.0)));
    }
    if INT_RE.is_match(s) {
        // `parseInt(s, 10)` — for a pure-decimal string this is `f64`
        // parsing modulo overflow to ±Infinity.
        return Some(Js::Num(s.parse::<f64>().unwrap_or(f64::INFINITY)));
    }
    if INT_HEX_RE.is_match(s) {
        return Some(Js::Num(parse_radix(s.get(2..).unwrap_or_default(), 16.0)));
    }
    None
}

/// `!!float` resolution — NaN/Inf spellings, exponent, plain decimal.
fn resolve_float_tagged(s: &str) -> Option<Js> {
    if FLOAT_NAN_RE.is_match(s) {
        // `str.slice(-3).toLowerCase() === 'nan'` — the regex guarantees
        // an ASCII tail, so a byte slice is safe.
        let is_nan = s
            .get(s.len().saturating_sub(3)..)
            .is_some_and(|t| t.eq_ignore_ascii_case("nan"));
        if is_nan {
            return Some(Js::Num(f64::NAN));
        }
        return Some(Js::Num(if s.starts_with('-') {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        }));
    }
    if FLOAT_EXP_RE.is_match(s) || FLOAT_RE.is_match(s) {
        return Some(Js::Num(s.parse::<f64>().unwrap_or(f64::INFINITY)));
    }
    None
}

/// Digit-wise accumulation like `parseInt` — matches f64 rounding for
/// magnitudes beyond `u64`.
fn parse_radix(digits: &str, radix: f64) -> f64 {
    let mut value: f64 = 0.0;
    for c in digits.chars() {
        let Some(d) = c.to_digit(16) else { break };
        value = value.mul_add(radix, f64::from(d));
    }
    value
}

/// Scalar inside a flow-collection map key — npm `toString` with
/// `inFlow`: resolved `String(value)` for plain scalars, quote style
/// preserved for quoted and block scalars.
fn flow_scalar(src: &str, style: ScalarStyle, tag: YamlTag) -> Result<String, String> {
    let display = match resolve_scalar(src, style, tag)? {
        Js::Str(s) => s,
        Js::Num(n) => js_num_to_string(n),
        Js::Bool(b) => if b { "true" } else { "false" }.to_owned(),
        Js::Null => "null".to_owned(),
        Js::Seq(_) | Js::Map(_) | Js::Unsupported => return Err("bad key".to_owned()),
    };
    Ok(match style {
        ScalarStyle::Plain => display,
        ScalarStyle::SingleQuoted => format!("'{}'", display.replace('\'', "''")),
        _ => crate::metadata::json_string(&display),
    })
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::missing_assert_message,
    clippy::shadow_unrelated,
    reason = "idiomatic test-code patterns"
)]
mod tests {
    use super::*;

    fn ordered_keys(yaml_source: &str) -> Vec<String> {
        let Ok(Js::Map(map)) = parse_frontmatter(yaml_source) else {
            return Vec::new();
        };
        map.js_ordered_keys()
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn js_order_puts_canonical_index_keys_first_ascending() {
        // ECMAScript own-property order: integer-index keys ascending,
        // then the rest in insertion order.
        let src = "b: 1\n10: 1\n2: 1\na: 1\n0.5: 1\n4294967295: 1\n0: 1\n";
        let keys = ordered_keys(src);
        assert_eq!(keys, ["0", "2", "10", "b", "a", "0.5", "4294967295"]);
    }

    #[test]
    fn reassignment_keeps_first_position() {
        // `1` and `1.0` resolve to the same scalar → the duplicate is a
        // YAML error, so exercise reassignment through a JS-level
        // collision instead (`1` and `"1"` are *not* YAML duplicates
        // but share the JS key `"1"`).
        let Ok(Js::Map(map)) = parse_frontmatter("a: 1\n1: 1\n\"1\": 2\nb: 1\n") else {
            panic!("expected map");
        };
        assert_eq!(map.js_ordered_keys(), ["1", "a", "b"]);
        // Later value wins, first position kept.
        let Js::Num(v) = map.get("1").unwrap_or(&Js::Null) else {
            panic!("expected number");
        };
        assert_eq!(*v, 2.0);
    }

    #[test]
    fn duplicate_scalar_keys_error_but_not_cross_type() {
        assert!(parse_frontmatter("a: 1\n\"a\": 2\n").is_err());
        assert!(parse_frontmatter("1: 1\n1.0: 2\n").is_err());
        // `1` (number) and `"1"` (string) are different YAML scalars.
        assert!(parse_frontmatter("1: 1\n\"1\": 2\n").is_ok());
        assert!(parse_frontmatter("~: 1\nnull: 2\n").is_err());
        assert!(parse_frontmatter("true: 1\nTRUE: 2\n").is_err());
    }

    #[test]
    fn aliases_resolve_and_anchor_counts_bound() {
        let Ok(Js::Map(map)) = parse_frontmatter("a: &x v\nb: *x\n") else {
            panic!("expected map");
        };
        assert!(matches!(map.get("b"), Some(Js::Str(s)) if s == "v"));

        // Boundary pins the npm `count * aliasCount > 100` check: 99
        // references pass, the 100th pushes `count` to 101 → error.
        let items = std::iter::repeat_n("*x", 99).collect::<Vec<_>>().join(", ");
        assert!(parse_frontmatter(&format!("a: &x v\nb: [{items}]")).is_ok());
        let items = std::iter::repeat_n("*x", 100)
            .collect::<Vec<_>>()
            .join(", ");
        assert!(parse_frontmatter(&format!("a: &x v\nb: [{items}]")).is_err());
    }

    #[test]
    fn self_referencing_aliases_resolve_to_shape_placeholders() {
        // npm yaml resolves the self-reference to the in-flight
        // collection (a circular value); we return a same-shaped
        // placeholder so callers see a Seq/Map rather than a YAML error.
        let Ok(Js::Map(map)) = parse_frontmatter("a: &x [*x]\n") else {
            panic!("expected map");
        };
        assert!(matches!(map.get("a"), Some(Js::Seq(_))));

        let Ok(Js::Map(map)) = parse_frontmatter("a: &x {k: *x}\n") else {
            panic!("expected map");
        };
        assert!(matches!(map.get("a"), Some(Js::Map(_))));
    }
}
