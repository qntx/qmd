//! FTS5 read side, ported from upstream `src/store.ts`: `getDocid`
//! (2364), `getContextForFile` (3547-3630), `searchFTS` (4081-4180) and
//! `mergeSearchResultsByScore` (4076-4079).
//!
//! Deviations:
//! - `searchFTS`'s `document_metadata` join and metadata `filter` arrive
//!   with P3, so `metadata` is absent from [`SearchResult`].
//! - `body_length` counts bytes (`String::len`); upstream uses JS string
//!   length (UTF-16 code units). Equal for ASCII and BMP text.

use rusqlite::{Connection, OptionalExtension, params_from_iter};

use super::collections::{DbCollection, get_store_collections, get_store_global_context};
use super::fts::build_fts5_query;
use crate::docid::get_docid;
use crate::error::{DbError, Result};
use crate::vpath::parse_virtual_path;

/// A ranked FTS search hit — upstream `SearchResult`/`DocumentResult`
/// (store.ts:2348-2363, 2441-2446).
#[derive(Debug)]
#[non_exhaustive]
pub struct SearchResult {
    /// `qmd://collection/path` URI.
    pub filepath: String,
    /// `collection/path` for display.
    pub display_path: String,
    /// Document title.
    pub title: String,
    /// Content hash.
    pub hash: String,
    /// Short docid (first 6 hash chars).
    pub docid: String,
    /// Parent collection name.
    pub collection_name: String,
    /// Last modification timestamp — `""` for FTS hits (upstream cannot
    /// provide it from this query shape).
    pub modified_at: String,
    /// Body length in bytes.
    pub body_length: usize,
    /// Full document body.
    pub body: String,
    /// Folder context (global + longest-prefix path contexts).
    pub context: Option<String>,
    /// Relevance score in `[0, 1)` — `|bm25| / (1 + |bm25|)`.
    pub score: f64,
    /// Search source.
    pub source: SearchSource,
}

/// Where a search hit came from (upstream `source` field).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SearchSource {
    /// Full-text (BM25) search.
    Fts,
    /// Vector similarity search (P2).
    Vec,
}

/// Upstream `getContextForFile` (store.ts:3547-3630): global context
/// first, then every matching path context from most general to most
/// specific, joined by `"\n\n"`.
///
/// Callers pass preloaded `collections`/`global_context` so a search
/// loop does not re-read `store_collections`/`store_config` per hit.
pub(crate) fn get_context_for_file(
    conn: &Connection,
    filepath: &str,
    collections: &[DbCollection],
    global_context: Option<&str>,
) -> Result<Option<String>> {
    if filepath.is_empty() {
        return Ok(None);
    }

    let (collection_name, relative_path) = if filepath.starts_with("qmd://") {
        let Some(vp) = parse_virtual_path(filepath) else {
            return Ok(None);
        };
        (vp.collection_name, vp.path)
    } else {
        let mut found: Option<(String, String)> = None;
        for coll in collections {
            if coll.path.is_empty() {
                continue;
            }
            let prefix = format!("{}/", coll.path);
            if filepath == coll.path {
                found = Some((coll.name.clone(), String::new()));
                break;
            }
            if filepath.starts_with(&prefix) {
                found = Some((coll.name.clone(), filepath[prefix.len()..].to_owned()));
                break;
            }
        }
        let Some(pair) = found else {
            return Ok(None);
        };
        pair
    };

    let Some(coll) = collections.iter().find(|c| c.name == collection_name) else {
        return Ok(None);
    };

    let exists: Option<String> = conn
        .query_row(
            "SELECT d.path FROM documents d \
             WHERE d.collection = ? AND d.path = ? AND d.active = 1 LIMIT 1",
            rusqlite::params![collection_name, relative_path],
            |r| r.get(0),
        )
        .optional()
        .map_err(DbError::from)?;
    if exists.is_none() {
        return Ok(None);
    }

    let mut contexts: Vec<String> = Vec::new();
    if let Some(global) = global_context {
        contexts.push(global.to_owned());
    }
    if let Some(context_map) = &coll.context {
        let normalized_path = if relative_path.starts_with('/') {
            relative_path
        } else {
            format!("/{relative_path}")
        };
        let mut matching: Vec<(String, &String)> = context_map
            .iter()
            .filter_map(|(prefix, ctx)| {
                let p = if prefix.starts_with('/') {
                    prefix.clone()
                } else {
                    format!("/{prefix}")
                };
                normalized_path.starts_with(&p).then_some((p, ctx))
            })
            .collect();
        matching.sort_by_key(|(prefix, _)| prefix.len());
        for (_, ctx) in matching {
            contexts.push(ctx.clone());
        }
    }
    if contexts.is_empty() {
        Ok(None)
    } else {
        Ok(Some(contexts.join("\n\n")))
    }
}

/// Upstream `searchFTS` (store.ts:4081-4180).
///
/// The CTE forces the FTS5 index scan before the collection filter is
/// applied; upstream found the planner otherwise abandons the index on
/// filtered queries (17s → 8ms). `LIMIT*10` over-fetch happens when a
/// collection filter is set. Multiple collections are queried separately
/// and merged by score so a large unrelated collection cannot starve the
/// smaller ones (upstream #775).
pub(crate) fn search_fts(
    conn: &Connection,
    query: &str,
    limit: usize,
    collection_scope: Option<Vec<String>>,
) -> Result<Vec<SearchResult>> {
    // One collection, several (OR), or all (None) — upstream
    // `scopedCollectionNames` (store.ts:4071-4074).
    let names = collection_scope.map(|scope| {
        scope
            .into_iter()
            .map(|n| n.trim().to_owned())
            .filter(|n| !n.is_empty())
            .collect::<Vec<_>>()
    });
    if let Some(names) = names.as_ref().filter(|n| n.len() > 1) {
        let lists = names
            .iter()
            .map(|name| search_fts(conn, query, limit, Some(vec![name.clone()])))
            .collect::<Result<Vec<_>>>()?;
        return Ok(merge_search_results_by_score(lists, limit));
    }
    let collection_filter = names.and_then(|n| n.into_iter().next());

    let Some(fts_query) = build_fts5_query(query) else {
        return Ok(Vec::new());
    };

    let fts_limit = if collection_filter.is_some() {
        limit * 10
    } else {
        limit
    };

    let mut sql = String::from(
        "WITH fts_matches AS ( \
           SELECT rowid, bm25(documents_fts, 1.5, 4.0, 1.0) AS bm25_score \
           FROM documents_fts \
           WHERE documents_fts MATCH ? \
           ORDER BY bm25_score ASC \
           LIMIT ? \
         ) \
         SELECT \
           'qmd://' || d.collection || '/' || d.path AS filepath, \
           d.collection || '/' || d.path AS display_path, \
           d.collection, \
           d.title, \
           content.doc AS body, \
           d.hash, \
           fm.bm25_score \
         FROM fts_matches fm \
         JOIN documents d ON d.id = fm.rowid \
         JOIN content ON content.hash = d.hash \
         WHERE d.active = 1",
    );
    if collection_filter.is_some() {
        sql.push_str(" AND d.collection = ?");
    }
    sql.push_str(" ORDER BY fm.bm25_score ASC LIMIT ?");

    let as_i64 = |n: usize| i64::try_from(n).unwrap_or(i64::MAX);
    let mut params: Vec<rusqlite::types::Value> = vec![fts_query.into(), as_i64(fts_limit).into()];
    if let Some(name) = collection_filter {
        params.push(name.into());
    }
    params.push(as_i64(limit).into());

    let mut stmt = conn.prepare(&sql).map_err(DbError::from)?;
    let rows = stmt
        .query_map(params_from_iter(params), |r| {
            Ok(Row {
                filepath: r.get(0)?,
                display_path: r.get(1)?,
                collection_name: r.get(2)?,
                title: r.get(3)?,
                body: r.get(4)?,
                hash: r.get(5)?,
                bm25_score: r.get(6)?,
            })
        })
        .map_err(DbError::from)?
        .collect::<rusqlite::Result<Vec<Row>>>()
        .map_err(DbError::from)?;

    // Loaded once per query instead of inside `get_context_for_file`
    // per result (upstream reloads them per hit too; hoisting is an
    // internal-only deviation with identical results).
    let collections = get_store_collections(conn)?;
    let global_context = get_store_global_context(conn)?;

    let mut results = Vec::with_capacity(rows.len());
    for row in rows {
        let bm25 = row.bm25_score.abs();
        results.push(SearchResult {
            docid: get_docid(&row.hash),
            context: get_context_for_file(
                conn,
                &row.filepath,
                &collections,
                global_context.as_deref(),
            )?,
            body_length: row.body.len(),
            body: row.body,
            score: bm25 / (1.0 + bm25),
            source: SearchSource::Fts,
            collection_name: row.collection_name,
            filepath: row.filepath,
            display_path: row.display_path,
            title: row.title,
            hash: row.hash,
            modified_at: String::new(),
        });
    }
    Ok(results)
}

/// One row of the `searchFTS` join.
struct Row {
    filepath: String,
    display_path: String,
    collection_name: String,
    title: String,
    body: String,
    hash: String,
    bm25_score: f64,
}

/// Upstream `mergeSearchResultsByScore` (store.ts:4076-4079): keep the
/// best-scoring entry per filepath, sort by score descending, truncate.
fn merge_search_results_by_score(lists: Vec<Vec<SearchResult>>, limit: usize) -> Vec<SearchResult> {
    let mut best: std::collections::HashMap<String, SearchResult> =
        std::collections::HashMap::new();
    for list in lists {
        for r in list {
            match best.get(&r.filepath) {
                Some(prev) if prev.score >= r.score => {}
                _ => {
                    best.insert(r.filepath.clone(), r);
                }
            }
        }
    }
    let mut merged: Vec<SearchResult> = best.into_values().collect();
    merged.sort_by(|a, b| b.score.total_cmp(&a.score));
    merged.truncate(limit);
    merged
}
