//! Document retrieval, ported from upstream `src/store.ts`:
//! `escapeLikePattern` (5014), `findDocumentByDocid` (3436-3450),
//! `findSimilarFiles` (3452-3465), `matchFilesByGlob` (3467-3487),
//! `getIgnoredLookupMatch` (4784-4839), `findDocument` (4851-4957),
//! `getDocumentBody` (4963-5007), `resolveCommaListName` (5086-5129),
//! `findDocuments` (5135-5233) and `resolveVirtualPath` (791-801);
//! plus the `qmd ls` query (cli/qmd.ts:1712-1736).
//!
//! Deviations:
//! - `strsim::levenshtein` counts characters; upstream's inline
//!   implementation counts UTF-16 code units (astral chars only).
//! - Glob matching uses `globset`; picomatch extglob syntax
//!   (`!(...)`, `@(...)`) is unsupported and dot-segment rules are moot
//!   because indexed documents never contain dot segments.
//! - A missing `vectors_vec`/`content` row cannot occur: every lookup
//!   goes through `JOIN content`.

use std::path::PathBuf;

use rusqlite::{Connection, OptionalExtension, params_from_iter};

use super::collections::{DbCollection, get_store_collections, get_store_global_context};
use super::search::get_context_for_file;
use crate::docid::{get_docid, is_docid, normalize_docid};
use crate::env::Environment;
use crate::error::{DbError, Error, Result};
use crate::paths::{expand_home, is_path_inside_dir, resolve_lexical};
use crate::vpath::{is_virtual_path, parse_virtual_path};

/// Upstream `escapeLikePattern` (store.ts:5014-5016): `#` is the LIKE
/// escape character, avoiding `'`/`"` pitfalls.
fn escape_like_pattern(value: &str) -> String {
    value
        .replace('#', "##")
        .replace('%', "#%")
        .replace('_', "#_")
}

/// A `documents` row joined with `content`, as selected by the lookup
/// queries (upstream `DbDocRow`, store.ts:4772-4782).
struct DocRow {
    virtual_path: String,
    display_path: String,
    title: String,
    hash: String,
    collection: String,
    modified_at: String,
    /// SQLite `LENGTH()` — characters, matching upstream.
    body_length: usize,
    body: Option<String>,
}

const SELECT_COLS_NO_BODY: &str = "
    'qmd://' || d.collection || '/' || d.path AS virtual_path,
    d.collection || '/' || d.path AS display_path,
    d.title,
    d.hash,
    d.collection,
    d.modified_at,
    LENGTH(content.doc) AS body_length";

const SELECT_COLS_BODY: &str = "
    'qmd://' || d.collection || '/' || d.path AS virtual_path,
    d.collection || '/' || d.path AS display_path,
    d.title,
    d.hash,
    d.collection,
    d.modified_at,
    LENGTH(content.doc) AS body_length,
    content.doc AS body";

const fn select_cols(include_body: bool) -> &'static str {
    if include_body {
        SELECT_COLS_BODY
    } else {
        SELECT_COLS_NO_BODY
    }
}

fn doc_row(row: &rusqlite::Row<'_>, include_body: bool) -> rusqlite::Result<DocRow> {
    Ok(DocRow {
        virtual_path: row.get(0)?,
        display_path: row.get(1)?,
        title: row.get(2)?,
        hash: row.get(3)?,
        collection: row.get(4)?,
        modified_at: row.get(5)?,
        body_length: row.get::<_, i64>(6)?.try_into().unwrap_or(0),
        body: if include_body {
            Some(row.get(7)?)
        } else {
            None
        },
    })
}

/// Assemble the public [`crate::document::Document`] from a `DocRow`,
/// attaching the resolved context chain.
fn to_document(
    conn: &Connection,
    row: &DocRow,
    collections: &[DbCollection],
    global_context: Option<&str>,
) -> Result<crate::document::Document> {
    let context = get_context_for_file(conn, &row.virtual_path, collections, global_context)?;
    Ok(crate::document::Document {
        virtual_path: row.virtual_path.clone(),
        display_path: row.display_path.clone(),
        title: row.title.clone(),
        context,
        hash: row.hash.clone(),
        docid: get_docid(&row.hash),
        collection: row.collection.clone(),
        modified_at: row.modified_at.clone(),
        body_length: row.body_length,
        body: row.body.clone(),
    })
}

/// Upstream `findDocumentByDocid` (store.ts:3436-3450): `LIKE 'hash%'`
/// prefix lookup, first match wins on collisions.
fn find_document_by_docid(conn: &Connection, docid: &str) -> Result<Option<(String, String)>> {
    let short_hash = normalize_docid(docid);
    if short_hash.is_empty() {
        return Ok(None);
    }
    conn.query_row(
        "SELECT 'qmd://' || d.collection || '/' || d.path AS filepath, d.hash \
         FROM documents d WHERE d.hash LIKE ? AND d.active = 1 LIMIT 1",
        [format!("{short_hash}%")],
        |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
    )
    .optional()
    .map_err(|e| DbError::from(e).into())
}

/// Upstream `findSimilarFiles` (store.ts:3452-3465): Levenshtein over
/// active `d.path` values, lowercased, `<= max_distance`, best first.
pub(crate) fn find_similar_files(
    conn: &Connection,
    query: &str,
    max_distance: usize,
    limit: usize,
) -> Result<Vec<String>> {
    let mut stmt = conn
        .prepare("SELECT d.path FROM documents d WHERE d.active = 1")
        .map_err(DbError::from)?;
    let paths = stmt
        .query_map([], |r| r.get::<_, String>(0))
        .map_err(DbError::from)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)?;
    let query_lower = query.to_lowercase();
    let mut scored: Vec<(String, usize)> = paths
        .into_iter()
        .map(|p| {
            let dist = strsim::levenshtein(&p.to_lowercase(), &query_lower);
            (p, dist)
        })
        .filter(|(_, d)| *d <= max_distance)
        .collect();
    scored.sort_by_key(|(_, d)| *d);
    scored.truncate(limit);
    Ok(scored.into_iter().map(|(p, _)| p).collect())
}

/// `picomatch` parity: `*` and `?` never cross `/` separators
/// (`literal_separator` is off by default in `globset`).
fn picomatch_glob(pattern: &str) -> Result<globset::Glob> {
    globset::GlobBuilder::new(pattern)
        .literal_separator(true)
        .build()
        .map_err(|e| Error::InvalidInput {
            reason: format!("invalid glob pattern '{pattern}': {e}"),
        })
}

/// Upstream `matchesIgnoreRule` (store.ts:4792-4797): `picomatch` with
/// `dot: true`. `globset` has no dot flag, but indexed paths never carry
/// dot segments, so the distinction is unreachable here.
fn matches_ignore_rule(relative_path: &str, rule: &str) -> Result<bool> {
    let path = relative_path.replace('\\', "/");
    let path = path.trim_start_matches('/');
    let rule = rule.replace('\\', "/");
    let rule = rule.trim_start_matches('/');
    let glob = picomatch_glob(rule)?;
    Ok(glob.compile_matcher().is_match(path))
}

/// One `excluded_by_ignore` hit (upstream `DocumentExcludedByIgnore`).
pub(crate) struct IgnoredMatch {
    /// Collection owning the rule.
    pub(crate) collection: String,
    /// Collection-relative path that matched.
    pub(crate) path: String,
    /// The ignore rule that matched.
    pub(crate) rule: String,
}

/// Upstream `normalizeLookupPathForIgnore` (store.ts:4784-4790):
/// backslashes to `/`, trim, `~/` expansion.
fn normalize_lookup_path(path: &str, env: &Environment) -> String {
    let normalized = path.replace('\\', "/");
    let normalized = normalized.trim();
    if let Some(rest) = normalized.strip_prefix("~/") {
        let home = expand_home("~", env).to_string_lossy().into_owned();
        return format!("{home}/{rest}");
    }
    normalized.to_owned()
}

/// Upstream `getIgnoredLookupMatch` (store.ts:4799-4839): whether the
/// would-be lookup target is excluded by a collection ignore rule.
fn get_ignored_lookup_match(
    conn: &Connection,
    query: &str,
    env: &Environment,
) -> Result<Option<IgnoredMatch>> {
    let normalized = normalize_lookup_path(query, env);
    let parsed_virtual = if is_virtual_path(&normalized) {
        parse_virtual_path(&normalized)
    } else {
        None
    };
    for coll in get_store_collections(conn)? {
        let Some(ignore) = &coll.ignore else { continue };
        if ignore.is_empty() {
            continue;
        }
        let mut candidates: Vec<String> = Vec::new();
        if let Some(vp) = &parsed_virtual {
            if vp.collection_name != coll.name {
                continue;
            }
            candidates.push(vp.path.clone());
        } else if let Some(rel) = normalized.strip_prefix(&format!("{}/", coll.path)) {
            candidates.push(rel.to_owned());
        } else if !normalized.starts_with('/') {
            let prefix = format!("{}/", coll.name);
            if let Some(rel) = normalized.strip_prefix(&prefix) {
                candidates.push(rel.to_owned());
            } else {
                candidates.push(normalized.clone());
            }
        }
        if let Some(hit) = first_ignore_match(&coll.name, &candidates, ignore)? {
            return Ok(Some(hit));
        }
    }
    Ok(None)
}

fn first_ignore_match(
    collection: &str,
    candidates: &[String],
    rules: &[String],
) -> Result<Option<IgnoredMatch>> {
    for candidate in candidates {
        for rule in rules {
            if matches_ignore_rule(candidate, rule)? {
                return Ok(Some(IgnoredMatch {
                    collection: collection.to_owned(),
                    path: candidate.clone(),
                    rule: rule.clone(),
                }));
            }
        }
    }
    Ok(None)
}

/// Upstream `findDocument` (store.ts:4851-4957): docid → virtual path →
/// fuzzy virtual-path LIKE → filesystem path per collection →
/// ignore-rule check → similar files. `qmd:`/`//` spellings are
/// intentionally not normalized before lookup (upstream-quirks.md Q1).
#[allow(
    clippy::significant_drop_tightening,
    reason = "the prepared statements must live until the last row is read"
)]
pub(crate) fn find_document(
    conn: &Connection,
    env: &Environment,
    filename: &str,
    include_body: bool,
) -> Result<crate::document::Document> {
    // `/:(\d+)$/` — strip a trailing `:line` before lookup.
    let mut filepath = match filename.rsplit_once(':') {
        Some((head, digits))
            if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) =>
        {
            head.to_owned()
        }
        _ => filename.to_owned(),
    };

    if is_docid(&filepath) {
        match find_document_by_docid(conn, &filepath)? {
            Some((virtual_path, _)) => filepath = virtual_path,
            None => {
                return Err(Error::DocumentNotFound {
                    query: filename.to_owned(),
                    similar_files: Vec::new(),
                });
            }
        }
    }

    if filepath.starts_with("~/") {
        filepath = expand_home(&filepath, env).to_string_lossy().into_owned();
    }

    let cols = select_cols(include_body);
    let row = |sql: &str, params: Vec<rusqlite::types::Value>| -> Result<Option<DocRow>> {
        conn.query_row(
            &format!("SELECT {cols} FROM documents d JOIN content ON content.hash = d.hash {sql}"),
            params_from_iter(params),
            |r| doc_row(r, include_body),
        )
        .optional()
        .map_err(|e| DbError::from(e).into())
    };

    // 1. Exact virtual path.
    let mut doc = row(
        "WHERE 'qmd://' || d.collection || '/' || d.path = ? AND d.active = 1",
        vec![filepath.clone().into()],
    )?;

    // 2. Fuzzy virtual-path substring match (`%query%`).
    if doc.is_none() {
        doc = row(
            "WHERE 'qmd://' || d.collection || '/' || d.path LIKE ? ESCAPE '#' \
             AND d.active = 1 LIMIT 1",
            vec![format!("%{}%", escape_like_pattern(&filepath)).into()],
        )?;
    }

    // 3. Absolute or collection-relative filesystem path.
    if doc.is_none() && !filepath.starts_with("qmd://") {
        for coll in get_store_collections(conn)? {
            let relative: Option<String> = filepath
                .strip_prefix(&format!("{}/", coll.path))
                .map(str::to_owned)
                .or_else(|| (!filepath.starts_with('/')).then(|| filepath.clone()));
            let Some(rel) = relative else { continue };
            doc = row(
                "WHERE d.collection = ? AND d.path = ? AND d.active = 1",
                vec![coll.name.into(), rel.into()],
            )?;
            if doc.is_some() {
                break;
            }
        }
    }

    let Some(doc) = doc else {
        if let Some(ignored) = get_ignored_lookup_match(conn, &filepath, env)? {
            return Err(Error::ExcludedByIgnore {
                query: filename.to_owned(),
                collection: ignored.collection,
                path: ignored.path,
                rule: ignored.rule,
            });
        }
        // Upstream scores the raw lookup string against
        // collection-relative paths, so absolute/virtual-path misses get
        // no suggestions (upstream-quirks.md Q2). Kept verbatim.
        let similar = find_similar_files(conn, &filepath, 5, 5)?;
        return Err(Error::DocumentNotFound {
            query: filename.to_owned(),
            similar_files: similar,
        });
    };

    let collections = get_store_collections(conn)?;
    let global_context = get_store_global_context(conn)?;
    to_document(conn, &doc, &collections, global_context.as_deref())
}

/// Upstream `getDocumentBody` (store.ts:4963-5007): resolve the document
/// first, then slice its content by 1-based line window.
///
/// `from` is clamped to line 1; `max_lines` of `None` reads to the end.
pub(crate) fn get_document_body(
    conn: &Connection,
    env: &Environment,
    path_or_docid: &str,
    from_line: Option<u64>,
    max_lines: Option<u64>,
) -> Result<Option<String>> {
    // Upstream resolves the document first (not-found → null) and slices
    // afterwards; lookup errors become `None`, database errors propagate.
    let doc = match find_document(conn, env, path_or_docid, false) {
        Ok(doc) => doc,
        Err(Error::DocumentNotFound { .. } | Error::ExcludedByIgnore { .. }) => {
            return Ok(None);
        }
        Err(e) => return Err(e),
    };
    let body = conn
        .query_row(
            "SELECT content.doc FROM documents d \
             JOIN content ON content.hash = d.hash \
             WHERE 'qmd://' || d.collection || '/' || d.path = ? AND d.active = 1",
            [&doc.virtual_path],
            |r| r.get::<_, String>(0),
        )
        .optional()
        .map_err(DbError::from)?;
    let Some(body) = body else { return Ok(None) };
    if from_line.is_none() && max_lines.is_none() {
        return Ok(Some(body));
    }
    let lines: Vec<&str> = body.split('\n').collect();
    let start = usize::try_from(from_line.unwrap_or(1).saturating_sub(1)).unwrap_or(0);
    let end = max_lines.map_or(lines.len(), |n| {
        start.saturating_add(usize::try_from(n).unwrap_or(usize::MAX))
    });
    let end = end.min(lines.len());
    if start >= lines.len() {
        return Ok(Some(String::new()));
    }
    Ok(Some(lines.get(start..end).unwrap_or(&[]).join("\n")))
}

/// One resolved comma-list entry (upstream `CommaListMatch`).
struct CommaMatch {
    collection: String,
    path: String,
    virtual_path: String,
}

fn comma_list_select(
    conn: &Connection,
    where_sql: &str,
    params: Vec<rusqlite::types::Value>,
) -> Result<Vec<CommaMatch>> {
    let sql = format!(
        "SELECT d.collection, d.path, \
         'qmd://' || d.collection || '/' || d.path AS virtual_path \
         FROM documents d JOIN content ON content.hash = d.hash \
         WHERE d.active = 1 AND ({where_sql}) \
         ORDER BY d.collection, d.path"
    );
    let mut stmt = conn.prepare(&sql).map_err(DbError::from)?;
    let rows = stmt
        .query_map(params_from_iter(params), |r| {
            Ok(CommaMatch {
                collection: r.get(0)?,
                path: r.get(1)?,
                virtual_path: r.get(2)?,
            })
        })
        .map_err(DbError::from)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)?;
    Ok(rows)
}

fn select_doc_row(
    conn: &Connection,
    collection: &str,
    path: &str,
    include_body: bool,
) -> Result<Option<DocRow>> {
    let sql = format!(
        "SELECT {} FROM documents d JOIN content ON content.hash = d.hash \
         WHERE d.collection = ? AND d.path = ? AND d.active = 1",
        select_cols(include_body)
    );
    conn.query_row(&sql, [collection, path], |r| doc_row(r, include_body))
        .optional()
        .map_err(|e| DbError::from(e).into())
}

fn finish_comma_resolve(
    conn: &Connection,
    name: &str,
    rows: Vec<CommaMatch>,
) -> Result<CommaMatch> {
    let mut rows = rows;
    match rows.len() {
        1 => Ok(rows.swap_remove(0)),
        0 => {
            let similar = find_similar_files(conn, name, 5, 3)?;
            let mut msg = format!("File not found: {name}");
            if !similar.is_empty() {
                msg.push_str(" (did you mean: ");
                msg.push_str(&similar.join(", "));
                msg.push_str("?)");
            }
            Err(Error::InvalidInput { reason: msg })
        }
        _ => Err(Error::InvalidInput {
            reason: format!(
                "Ambiguous path {name}: {}",
                rows.iter()
                    .map(|r| r.virtual_path.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }),
    }
}

/// Upstream `resolveCommaListName` (store.ts:5086-5129): docid or
/// `qmd://` (exact), then exact `collection/path`, exact `d.path`, then
/// path-boundary suffix `%/name` — never an unanchored LIKE (#759).
fn resolve_comma_list_name(conn: &Connection, name: &str) -> Result<CommaMatch> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err(Error::InvalidInput {
            reason: format!("File not found: {name}"),
        });
    }

    if is_docid(trimmed) {
        let rows = match find_document_by_docid(conn, trimmed)? {
            Some((virtual_path, _)) => comma_list_select(
                conn,
                "'qmd://' || d.collection || '/' || d.path = ?",
                vec![virtual_path.into()],
            )?,
            None => Vec::new(),
        };
        return finish_comma_resolve(conn, trimmed, rows);
    }

    if is_virtual_path(trimmed) {
        let rows = match parse_virtual_path(trimmed) {
            Some(vp) => comma_list_select(
                conn,
                "d.collection = ? AND d.path = ?",
                vec![vp.collection_name.into(), vp.path.into()],
            )?,
            None => Vec::new(),
        };
        return finish_comma_resolve(conn, trimmed, rows);
    }

    let rows = comma_list_select(
        conn,
        "d.collection || '/' || d.path = ?",
        vec![trimmed.to_owned().into()],
    )?;
    if !rows.is_empty() {
        return finish_comma_resolve(conn, trimmed, rows);
    }

    let path_rows = comma_list_select(conn, "d.path = ?", vec![trimmed.to_owned().into()])?;
    if !path_rows.is_empty() {
        return finish_comma_resolve(conn, trimmed, path_rows);
    }

    let like_rows = comma_list_select(
        conn,
        "d.path LIKE ? ESCAPE '#'",
        vec![format!("%/{}", escape_like_pattern(trimmed)).into()],
    )?;
    finish_comma_resolve(conn, trimmed, like_rows)
}

/// Upstream `matchFilesByGlob` (store.ts:3467-3487): match the glob
/// against `qmd://coll/path`, `path`, and `coll/path`.
fn match_files_by_glob(conn: &Connection, pattern: &str) -> Result<Vec<CommaMatch>> {
    // picomatch treats a leading `!` as negation; globset has no such
    // syntax, so it is applied manually.
    let (negated, pattern) = pattern
        .strip_prefix('!')
        .map_or((false, pattern), |rest| (true, rest));
    let glob = picomatch_glob(pattern)?;
    let matcher = glob.compile_matcher();
    let is_match = |candidate: &str| matcher.is_match(candidate) != negated;

    let mut stmt = conn
        .prepare(
            "SELECT d.collection, d.path, \
             'qmd://' || d.collection || '/' || d.path AS virtual_path \
             FROM documents d JOIN content ON content.hash = d.hash \
             WHERE d.active = 1",
        )
        .map_err(DbError::from)?;
    let rows = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })
        .map_err(DbError::from)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)?;
    Ok(rows
        .into_iter()
        .filter(|(collection, path, virtual_path)| {
            is_match(virtual_path) || is_match(path) || is_match(&format!("{collection}/{path}"))
        })
        .map(|(collection, path, virtual_path)| CommaMatch {
            collection,
            path,
            virtual_path,
        })
        .collect())
}

/// Upstream `findDocuments` (store.ts:5135-5233): comma-separated names
/// (or a lone docid) via [`resolve_comma_list_name`], otherwise a glob
/// over every active document.
pub(crate) fn find_documents(
    conn: &Connection,
    pattern: &str,
    include_body: bool,
    max_bytes: usize,
) -> Result<crate::document::MultiGet> {
    let is_comma_separated = pattern.contains(',')
        && !pattern.contains('*')
        && !pattern.contains('?')
        && !pattern.contains('{');
    let mut errors: Vec<String> = Vec::new();

    let file_rows: Vec<DocRow> = if is_comma_separated || is_docid(pattern) {
        let names: Vec<&str> = if is_comma_separated {
            pattern
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect()
        } else {
            vec![pattern.trim()]
        };
        let mut rows = Vec::new();
        for name in names {
            match resolve_comma_list_name(conn, name) {
                Ok(m) => match select_doc_row(conn, &m.collection, &m.path, include_body)? {
                    Some(row) => rows.push(row),
                    None => errors.push(format!("File not found: {name}")),
                },
                Err(Error::InvalidInput { reason }) => errors.push(reason),
                Err(e) => return Err(e),
            }
        }
        rows
    } else {
        let matched = match_files_by_glob(conn, pattern)?;
        if matched.is_empty() {
            errors.push(format!("No files matched pattern: {pattern}"));
            return Ok(crate::document::MultiGet {
                docs: Vec::new(),
                errors,
            });
        }
        let placeholders = matched.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT {} FROM documents d JOIN content ON content.hash = d.hash \
             WHERE 'qmd://' || d.collection || '/' || d.path IN ({placeholders}) \
             AND d.active = 1",
            select_cols(include_body)
        );
        let params = params_from_iter(matched.iter().map(|m| m.virtual_path.clone()));
        let mut stmt = conn.prepare(&sql).map_err(DbError::from)?;
        stmt.query_map(params, |r| doc_row(r, include_body))
            .map_err(DbError::from)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(DbError::from)?
    };

    let collections = get_store_collections(conn)?;
    let global_context = get_store_global_context(conn)?;

    let mut docs = Vec::with_capacity(file_rows.len());
    for row in &file_rows {
        if row.body_length > max_bytes {
            // `Math.round(x / 1024)` — round-half-up, not ceil.
            let kb = |n: usize| (n + 512) / 1024;
            docs.push(crate::document::MultiGetEntry::Skipped {
                virtual_path: row.virtual_path.clone(),
                display_path: row.display_path.clone(),
                reason: format!(
                    "File too large ({}KB > {}KB)",
                    kb(row.body_length),
                    kb(max_bytes),
                ),
            });
            continue;
        }
        let mut doc = to_document(conn, row, &collections, global_context.as_deref())?;
        if doc.title.is_empty() {
            doc.display_path
                .rsplit('/')
                .next()
                .unwrap_or(&doc.display_path)
                .clone_into(&mut doc.title);
        }
        docs.push(crate::document::MultiGetEntry::Hit(doc));
    }
    Ok(crate::document::MultiGet { docs, errors })
}

/// Upstream `resolveVirtualPath` (store.ts:791-801): `qmd://coll/path`
/// → absolute filesystem path inside the collection root.
pub(crate) fn resolve_virtual_path(conn: &Connection, uri: &str) -> Result<Option<PathBuf>> {
    let Some(parsed) = parse_virtual_path(uri) else {
        return Ok(None);
    };
    let collections = get_store_collections(conn)?;
    let Some(coll) = collections
        .iter()
        .find(|c| c.name == parsed.collection_name)
    else {
        return Ok(None);
    };
    let resolved = resolve_lexical(std::path::Path::new(&coll.path), &parsed.path);
    if !is_path_inside_dir(std::path::Path::new(&coll.path), &resolved) {
        return Ok(None);
    }
    Ok(Some(resolved))
}

/// The `qmd ls` document list (cli/qmd.ts:1712-1736): `d.path LIKE
/// 'prefix%'` when a prefix is given, ordered by path. `size` is
/// `LENGTH(ct.doc)` characters, as upstream prints it.
pub(crate) fn list_documents(
    conn: &Connection,
    collection: &str,
    prefix: Option<&str>,
) -> Result<Vec<crate::document::DocumentEntry>> {
    let exists: Option<String> = conn
        .query_row(
            "SELECT name FROM store_collections WHERE name = ?",
            [collection],
            |r| r.get(0),
        )
        .optional()
        .map_err(DbError::from)?;
    if exists.is_none() {
        return Err(Error::CollectionNotFound {
            name: collection.to_owned(),
        });
    }
    let (where_sql, params): (&str, Vec<rusqlite::types::Value>) = prefix.map_or_else(
        || ("", Vec::new()),
        |p| {
            (
                "AND d.path LIKE ? ESCAPE '#'",
                vec![format!("{}%", escape_like_pattern(p)).into()],
            )
        },
    );
    let sql = format!(
        "SELECT d.path, d.title, d.modified_at, LENGTH(ct.doc) AS size \
         FROM documents d JOIN content ct ON d.hash = ct.hash \
         WHERE d.collection = ? AND d.active = 1 {where_sql} ORDER BY d.path"
    );
    let mut all_params = vec![rusqlite::types::Value::from(collection.to_owned())];
    all_params.extend(params);
    let mut stmt = conn.prepare(&sql).map_err(DbError::from)?;
    let rows = stmt
        .query_map(params_from_iter(all_params), |r| {
            Ok(crate::document::DocumentEntry {
                path: r.get(0)?,
                title: r.get(1)?,
                modified_at: r.get(2)?,
                size: r.get::<_, i64>(3)?.try_into().unwrap_or(0),
            })
        })
        .map_err(DbError::from)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)?;
    Ok(rows)
}
