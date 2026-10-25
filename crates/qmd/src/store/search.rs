//! FTS5 read side, ported from upstream `src/store.ts`: `getDocid`
//! (2364), `getContextForFile` (3547-3630), `searchFTS` (4081-4180),
//! `searchVec` (4236-4384) and `mergeSearchResultsByScore` (4076-4079).
//!
//! Deviation:
//! - `body_length` counts bytes (`String::len`); upstream uses JS string
//!   length (UTF-16 code units). Equal for ASCII and BMP text.

use rusqlite::{Connection, OptionalExtension, params_from_iter};

use super::collections::{DbCollection, get_store_collections, get_store_global_context};
use super::fts::build_fts5_query;
use super::metadata::parse_metadata_json;
use crate::docid::get_docid;
use crate::error::{DbError, Result};
use crate::metadata::filter::{MetadataFilter, compile_metadata_filter};
use crate::metadata::{DocumentMetadata, METADATA_EXTRACTION_VERSION};
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
    /// Relevance score in `[0, 1)` — `|bm25| / (1 + |bm25|)` for FTS,
    /// `1 - cosine distance` for vector hits.
    pub score: f64,
    /// Search source.
    pub source: SearchSource,
    /// Chunk position (UTF-16 code units) within the document — `Some`
    /// only for vector hits (upstream `chunkPos`, P2.3).
    pub chunk_pos: Option<usize>,
    /// Canonical metadata from `document_metadata` (`{}` when the
    /// document has no extraction row or holds no metadata) — upstream
    /// `metadata` (store.ts:4166, 4378).
    pub metadata: DocumentMetadata,
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

impl SearchSource {
    /// Upstream's literal tag (`"fts"`/`"vec"`, store.ts:2460) — used by
    /// `--explain` rendering and its JSON payload.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fts => "fts",
            Self::Vec => "vec",
        }
    }
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
/// collection or metadata filter is set. Multiple collections are
/// queried separately and merged by score so a large unrelated
/// collection cannot starve the smaller ones (upstream #775).
///
/// With a metadata `filter`, only documents with a current-version,
/// error-free extraction can match — an unprocessed document must not
/// accidentally satisfy `exists: false` (store.ts:4135-4141).
pub(crate) fn search_fts(
    conn: &Connection,
    query: &str,
    limit: usize,
    collection_scope: Option<Vec<String>>,
    filter: Option<&MetadataFilter>,
) -> Result<Vec<SearchResult>> {
    // One collection, several (OR), or all (None) — upstream
    // `scopedCollectionNames` (store.ts:4071-4074).
    let names = collection_scope.map(|scope| {
        scope
            .into_iter()
            .map(|n| crate::chunk::js_trim(&n).to_owned())
            .filter(|n| !n.is_empty())
            .collect::<Vec<_>>()
    });
    if let Some(names) = names.as_ref().filter(|n| n.len() > 1) {
        let lists = names
            .iter()
            .map(|name| search_fts(conn, query, limit, Some(vec![name.clone()]), filter))
            .collect::<Result<Vec<_>>>()?;
        return Ok(merge_search_results_by_score(lists, limit));
    }
    let collection_filter = names.and_then(|n| n.into_iter().next());

    let Some(fts_query) = build_fts5_query(query) else {
        return Ok(Vec::new());
    };

    let fts_limit = if collection_filter.is_some() || filter.is_some() {
        limit.saturating_mul(10)
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
           fm.bm25_score, \
           dm.metadata_json \
         FROM fts_matches fm \
         JOIN documents d ON d.id = fm.rowid \
         JOIN content ON content.hash = d.hash \
         LEFT JOIN document_metadata dm ON dm.document_id = d.id \
         WHERE d.active = 1",
    );
    let mut filter_params = Vec::new();
    if collection_filter.is_some() {
        sql.push_str(" AND d.collection = ?");
    }
    if let Some(f) = filter {
        // Only documents with current, error-free extraction can match —
        // an unprocessed document must not accidentally satisfy
        // `exists: false` (store.ts:4135-4141).
        let compiled = compile_metadata_filter(f, "d");
        let gate = format!(
            " AND dm.extraction_version = {METADATA_EXTRACTION_VERSION} \
             AND dm.extraction_error IS NULL AND {}",
            compiled.sql
        );
        sql.push_str(&gate);
        filter_params = compiled.params;
    }
    sql.push_str(" ORDER BY fm.bm25_score ASC LIMIT ?");

    let as_i64 = |n: usize| i64::try_from(n).unwrap_or(i64::MAX);
    let mut params: Vec<rusqlite::types::Value> = vec![fts_query.into(), as_i64(fts_limit).into()];
    if let Some(name) = collection_filter {
        params.push(name.into());
    }
    params.extend(filter_params);
    params.push(as_i64(limit).into());
    run_fts_query(conn, &sql, params)
}

/// Shared tail of [`search_fts`] — run the prepared statement and map
/// rows to results.
fn run_fts_query(
    conn: &Connection,
    sql: &str,
    params: Vec<rusqlite::types::Value>,
) -> Result<Vec<SearchResult>> {
    let mut stmt = conn.prepare(sql).map_err(DbError::from)?;
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
                metadata_json: r.get(7)?,
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
            chunk_pos: None,
            collection_name: row.collection_name,
            filepath: row.filepath,
            display_path: row.display_path,
            title: row.title,
            hash: row.hash,
            modified_at: String::new(),
            metadata: parse_metadata_json(row.metadata_json.as_deref()),
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
    metadata_json: Option<String>,
}

/// Upstream `mergeSearchResultsByScore` (store.ts:4076-4079): keep the
/// best-scoring entry per filepath, sort by score descending, truncate.
/// `IndexMap` preserves first-insertion order like upstream's JS `Map`
/// (`set` on an existing key updates in place), so equal scores keep
/// first-seen order through the stable sort.
fn merge_search_results_by_score(lists: Vec<Vec<SearchResult>>, limit: usize) -> Vec<SearchResult> {
    let mut best: indexmap::IndexMap<String, SearchResult> = indexmap::IndexMap::new();
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

// =============================================================================
// Vector search (P2.3)
// =============================================================================

/// Upstream `SQLITE_VEC_MAX_K` (store.ts:4178): sqlite-vec's hard cap on
/// `k` in a MATCH query.
const SQLITE_VEC_MAX_K: usize = 4096;
/// Upstream `FILTERED_VEC_EXACT_SCAN_MAX` (store.ts:4187): eligible-set
/// size below which we exact-scan instead of capped ANN + post-filter.
const FILTERED_VEC_EXACT_SCAN_MAX: usize = 20_000;
/// Upstream `VEC_HASH_SEQ_IN_CHUNK` (store.ts:4190): `hash_seq IN (…)`
/// chunk width for the exact scan.
const VEC_HASH_SEQ_IN_CHUNK: usize = 400;

/// Upstream `exactVecScanByHashSeq` (store.ts:4195-4221): exact
/// `vec_distance_cosine` scan over a known `hash_seq` set, IN-list
/// chunked, over-fetching `max(limit * 3, limit)` so multi-chunk docs
/// still yield `limit` unique files after dedupe.
fn exact_vec_scan_by_hash_seq(
    conn: &Connection,
    embedding: &[f32],
    hash_seqs: &[String],
    limit: usize,
) -> Result<Vec<(String, f64)>> {
    if hash_seqs.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    let fetch_limit = limit.saturating_mul(3).max(limit);
    let query = rusqlite::types::Value::Blob(super::vec::vec_blob(embedding));
    let mut scored: Vec<(String, f64)> = Vec::new();
    for chunk in hash_seqs.chunks(VEC_HASH_SEQ_IN_CHUNK) {
        let placeholders = vec!["?"; chunk.len()].join(",");
        let sql = format!(
            "SELECT hash_seq, vec_distance_cosine(embedding, ?) AS distance \
             FROM vectors_vec WHERE hash_seq IN ({placeholders})"
        );
        let mut params: Vec<rusqlite::types::Value> = Vec::with_capacity(chunk.len() + 1);
        params.push(query.clone());
        params.extend(chunk.iter().map(|s| s.clone().into()));
        let mut rows = conn
            .prepare(&sql)
            .and_then(|mut s| {
                s.query_map(params_from_iter(params), |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?))
                })
                .and_then(|m| m.collect::<rusqlite::Result<Vec<_>>>())
            })
            .map_err(DbError::from)?;
        scored.append(&mut rows);
    }
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored.truncate(fetch_limit);
    Ok(scored)
}

/// Upstream `annVecScan` (store.ts:4223-4234): `embedding MATCH ?` with
/// `k` clamped to `[1, SQLITE_VEC_MAX_K]`.
fn ann_vec_scan(conn: &Connection, embedding: &[f32], k: usize) -> Result<Vec<(String, f64)>> {
    let vec_k = k.clamp(1, SQLITE_VEC_MAX_K);
    let rows = conn
        .prepare("SELECT hash_seq, distance FROM vectors_vec WHERE embedding MATCH ? AND k = ?")
        .and_then(|mut s| {
            s.query_map(
                rusqlite::params![
                    rusqlite::types::Value::Blob(super::vec::vec_blob(embedding)),
                    i64::try_from(vec_k).unwrap_or(i64::MAX)
                ],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?)),
            )
            .and_then(|m| m.collect::<rusqlite::Result<Vec<_>>>())
        })
        .map_err(DbError::from)?;
    Ok(rows)
}

/// Query embedding for `searchVec` — upstream
/// `getEmbedding(query, model, isQuery=true)` (store.ts:4249-4257).
/// `Ok(None)` is the upstream null: callers return/skip results.
///
/// Upstream `llm.embed`/`embedBatch` catch every error and return
/// null (llm.ts:1473-1495, 1502-1566), so upstream degrades to "no
/// vector results"; we deliberately propagate backend errors instead
/// (registered deviation). An empty/absent slot remains the `null`
/// analogue.
///
/// Called WITHOUT the connection lock — the caller checks
/// `has_vectors_vec` first, then releases the guard before embedding.
pub(crate) fn embed_query(
    embedder: &dyn crate::llm::Embedder,
    query: &str,
) -> Result<Option<Vec<f32>>> {
    let text = crate::llm::format_query_for_embedding(query, Some(embedder.model_id()));
    embedder
        .embed(&[&text])
        .map_err(crate::Error::from)
        .map(|v| v.into_iter().next().filter(|v| !v.is_empty()))
}

/// Upstream `searchVec` (store.ts:4236-4384) with a precomputed query
/// embedding (upstream's `precomputedEmbedding` parameter, store.ts:4263,
/// 5641-5644, 6048-6051). Every caller embeds first, outside the
/// connection lock — the hybrid and structured pipelines batch all vec
/// queries in one call; `search_vector`/`vector_query` use
/// [`embed_query`].
///
/// Two-step shape is load-bearing: sqlite-vec virtual tables hang on a
/// `JOIN` in the same query (upstream PR #23) — the vec step never
/// joins, the document lookup never touches `vectors_vec`.
///
/// With a collection or metadata `filter` the eligible `hash_seq` set is
/// computed first: small sets exact-scan (global ANN would starve them,
/// upstream #791/#803), large sets fall back to capped ANN. The filter
/// is re-applied at the document join because vectors are content-scoped
/// — one hash can back both matching and non-matching documents
/// (store.ts:4339-4345).
pub(crate) fn search_vec_with_embedding(
    conn: &Connection,
    embedding: &[f32],
    limit: usize,
    collection_scope: Option<Vec<String>>,
    filter: Option<&MetadataFilter>,
) -> Result<Vec<SearchResult>> {
    let names = collection_scope.map(|scope| {
        scope
            .into_iter()
            .map(|n| crate::chunk::js_trim(&n).to_owned())
            .filter(|n| !n.is_empty())
            .collect::<Vec<_>>()
    });
    if let Some(names) = names.as_ref().filter(|n| n.len() > 1) {
        let lists = names
            .iter()
            .map(|name| search_vec_multi(conn, embedding, limit, Some(name), filter))
            .collect::<Result<Vec<_>>>()?;
        return Ok(merge_search_results_by_score(lists, limit));
    }
    search_vec_multi(
        conn,
        embedding,
        limit,
        names.and_then(|n| n.into_iter().next()).as_deref(),
        filter,
    )
}

/// Eligible `hash_seq` rows for a collection/metadata filter — the
/// `SELECT DISTINCT` over `content_vectors ⨝ documents` upstream
/// builds before deciding exact-scan vs ANN (store.ts:4272-4296).
/// Metadata-filtered queries join `document_metadata` and require a
/// current, error-free extraction row.
fn eligible_hash_seqs(
    conn: &Connection,
    collection_filter: Option<&str>,
    filter: Option<&MetadataFilter>,
) -> Result<Vec<String>> {
    let mut sql = String::from(
        "SELECT DISTINCT cv.hash || '_' || cv.seq AS hash_seq \
         FROM content_vectors cv \
         JOIN documents d ON d.hash = cv.hash AND d.active = 1",
    );
    let mut conditions: Vec<String> = Vec::new();
    let mut params: Vec<rusqlite::types::Value> = Vec::new();
    if let Some(name) = collection_filter {
        conditions.push("d.collection = ?".to_owned());
        params.push(name.to_owned().into());
    }
    if let Some(f) = filter {
        let compiled = compile_metadata_filter(f, "d");
        sql.push_str(" JOIN document_metadata dm ON dm.document_id = d.id");
        conditions.push(format!(
            "dm.extraction_version = {METADATA_EXTRACTION_VERSION}"
        ));
        conditions.push("dm.extraction_error IS NULL".to_owned());
        conditions.push(compiled.sql);
        params.extend(compiled.params);
    }
    sql.push_str(" WHERE ");
    sql.push_str(&conditions.join(" AND "));
    let eligible = conn
        .prepare(&sql)
        .and_then(|mut s| {
            s.query_map(params_from_iter(params), |r| r.get::<_, String>(0))
                .and_then(|m| m.collect::<rusqlite::Result<Vec<_>>>())
        })
        .map_err(DbError::from)?;
    Ok(eligible)
}

/// The query-embeddings-free inner half of `searchVec` — multi-
/// collection merge reuses the already-computed query vector
/// (upstream passes `precomputedEmbedding`).
pub(crate) fn search_vec_multi(
    conn: &Connection,
    embedding: &[f32],
    limit: usize,
    collection_filter: Option<&str>,
    filter: Option<&MetadataFilter>,
) -> Result<Vec<SearchResult>> {
    let vec_results: Vec<(String, f64)> = if collection_filter.is_some() || filter.is_some() {
        let eligible = eligible_hash_seqs(conn, collection_filter, filter)?;
        if eligible.is_empty() {
            return Ok(Vec::new());
        }
        if eligible.len() <= FILTERED_VEC_EXACT_SCAN_MAX {
            exact_vec_scan_by_hash_seq(conn, embedding, &eligible, limit)?
        } else {
            // Large eligible set: ANN with over-fetch, k capped at
            // sqlite-vec's max (store.ts:4302).
            ann_vec_scan(conn, embedding, limit.saturating_mul(30))?
        }
    } else {
        ann_vec_scan(conn, embedding, limit.saturating_mul(3))?
    };
    if vec_results.is_empty() {
        return Ok(Vec::new());
    }

    let distance_map: std::collections::HashMap<String, f64> =
        vec_results.iter().cloned().collect();
    let hash_seqs: Vec<String> = vec_results.iter().map(|r| r.0.clone()).collect();
    let placeholders = vec!["?"; hash_seqs.len()].join(",");
    let mut sql = format!(
        "SELECT cv.hash || '_' || cv.seq AS hash_seq, cv.hash, cv.pos, \
           'qmd://' || d.collection || '/' || d.path AS filepath, \
           d.collection || '/' || d.path AS display_path, \
           d.title, content.doc AS body, dm.metadata_json \
         FROM content_vectors cv \
         JOIN documents d ON d.hash = cv.hash AND d.active = 1 \
         JOIN content ON content.hash = d.hash \
         LEFT JOIN document_metadata dm ON dm.document_id = d.id \
         WHERE cv.hash || '_' || cv.seq IN ({placeholders})"
    );
    let mut params: Vec<rusqlite::types::Value> = hash_seqs.into_iter().map(Into::into).collect();
    if let Some(name) = collection_filter {
        sql.push_str(" AND d.collection = ?");
        params.push(name.to_owned().into());
    }
    if let Some(f) = filter {
        // Re-apply the filter on the document join: vectors are
        // content-scoped, so one hash can belong to both matching and
        // non-matching documents (store.ts:4339-4345).
        let compiled = compile_metadata_filter(f, "d");
        let gate = format!(
            " AND dm.extraction_version = {METADATA_EXTRACTION_VERSION} \
             AND dm.extraction_error IS NULL AND {}",
            compiled.sql
        );
        sql.push_str(&gate);
        params.extend(compiled.params);
    }

    let mut stmt = conn.prepare(&sql).map_err(DbError::from)?;
    let doc_rows = stmt
        .query_map(params_from_iter(params), |r| {
            Ok(VecRow {
                hash_seq: r.get(0)?,
                hash: r.get(1)?,
                pos: r.get(2)?,
                filepath: r.get(3)?,
                display_path: r.get(4)?,
                title: r.get(5)?,
                body: r.get(6)?,
                metadata_json: r.get(7)?,
            })
        })
        .map_err(DbError::from)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)?;

    // Dedupe by filepath keeping the smallest distance (store.ts:4334).
    // Upstream `Map` iterates in insertion order and `sort` is stable,
    // so equal distances keep first-seen order — track insertion order
    // ourselves to mirror that.
    let mut best: std::collections::HashMap<String, (VecRow, f64)> =
        std::collections::HashMap::new();
    let mut filepath_order: Vec<String> = Vec::new();
    for row in doc_rows {
        let distance = distance_map.get(&row.hash_seq).copied().unwrap_or(1.0);
        let dominated = matches!(best.get(&row.filepath), Some((_, b)) if *b <= distance);
        if !dominated {
            if !best.contains_key(&row.filepath) {
                filepath_order.push(row.filepath.clone());
            }
            best.insert(row.filepath.clone(), (row, distance));
        }
    }
    let mut winners: Vec<(VecRow, f64)> = filepath_order
        .into_iter()
        .filter_map(|f| best.remove(&f))
        .collect();
    winners.sort_by(|a, b| a.1.total_cmp(&b.1));
    winners.truncate(limit);

    let collections = get_store_collections(conn)?;
    let global_context = get_store_global_context(conn)?;

    let mut results = Vec::with_capacity(winners.len());
    for (row, dist) in winners {
        let collection_name = row
            .filepath
            .strip_prefix("qmd://")
            .and_then(|rest| rest.split('/').next())
            .unwrap_or_default()
            .to_owned();
        let context =
            get_context_for_file(conn, &row.filepath, &collections, global_context.as_deref())?;
        results.push(SearchResult {
            filepath: row.filepath.clone(),
            display_path: row.display_path,
            title: row.title,
            hash: row.hash.clone(),
            docid: get_docid(&row.hash),
            collection_name,
            // Upstream `modifiedAt: ""` — not available in this query
            // shape (store.ts:4357).
            modified_at: String::new(),
            body_length: row.body.len(),
            body: row.body,
            context,
            score: 1.0 - dist,
            source: SearchSource::Vec,
            chunk_pos: Some(usize::try_from(row.pos).unwrap_or(0)),
            metadata: parse_metadata_json(row.metadata_json.as_deref()),
        });
    }
    Ok(results)
}

/// One row of the `searchVec` document lookup (step 2).
struct VecRow {
    hash_seq: String,
    hash: String,
    pos: i64,
    filepath: String,
    display_path: String,
    title: String,
    body: String,
    metadata_json: Option<String>,
}
