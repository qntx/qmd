//! `llm_cache` read/write — upstream `getCacheKey` /
//! `getCachedResult` / `setCachedResult` /
//! `deleteExpansionCacheEntry` (store.ts:2681-2699, 4596-4599).
//!
//! T5 deviation: upstream prunes probabilistically
//! (`Math.random() < 0.01`); we prune deterministically every
//! [`PRUNE_EVERY`] writes so the cadence matches in expectation without
//! an RNG dependency (registered in parity-matrix).

use std::sync::atomic::{AtomicUsize, Ordering};

use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::error::{DbError, Result};
use crate::store::documents::now_iso;

/// Writes between deterministic prunes — same expected cadence as
/// upstream's 1% random prune.
const PRUNE_EVERY: usize = 100;

/// Rows kept by each prune — upstream `LIMIT 1000` (store.ts:2697).
const PRUNE_KEEP: i64 = 1000;

/// Cache-key body — upstream `CacheKeyBody` (store.ts:2674-2679). The
/// field declaration order fixes `serde_json` key order to match the
/// object literals upstream builds (`{query, model}` for expansion,
/// `{query, model, chunk}` for rerank). The legacy `{query, file,
/// model, chunk}` rerank key is not emitted — D2 registers that
/// pre-#764 entries are simply missed (and lazily replaced).
#[derive(Serialize)]
pub(crate) struct CacheKeyBody<'a> {
    /// Query text (expansion query, or `"{intent}\n\n{query}"` for rerank).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query: Option<&'a str>,
    /// Resolved model name/URI.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<&'a str>,
    /// Chunk text (rerank entries only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk: Option<&'a str>,
}

/// Upstream `getCacheKey` (store.ts:2681-2686):
/// `sha256(url ‖ JSON.stringify(body))` hex.
///
/// `serde_json` escapes `&<>'` etc. differently than `JSON.stringify`
/// only inside HTML-sensitive contexts — both emit the same bytes for
/// the string values we serialize (golden fixture covers quotes,
/// newlines and non-ASCII).
#[must_use]
pub(crate) fn cache_key(url: &str, body: &CacheKeyBody<'_>) -> String {
    #[allow(
        clippy::expect_used,
        reason = "serializing a flat struct of Option<&str> fields cannot fail"
    )]
    let body_json = serde_json::to_string(body).expect("infallible: only &str fields");
    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
    hasher.update(body_json.as_bytes());
    // Lowercase hex — upstream `digest('hex')`. Manual nibble-split,
    // same pattern as `documents::hash_content`.
    let mut s = String::with_capacity(64);
    for b in hasher.finalize() {
        s.push(char::from_digit(u32::from(b >> 4), 16).unwrap_or('0'));
        s.push(char::from_digit(u32::from(b & 0x0f), 16).unwrap_or('0'));
    }
    s
}

/// Upstream `getCachedResult` (store.ts:2688-2691).
pub(crate) fn get_cached_result(conn: &Connection, hash: &str) -> Result<Option<String>> {
    conn.query_row(
        "SELECT result FROM llm_cache WHERE hash = ?",
        params![hash],
        |r| r.get(0),
    )
    .optional()
    .map_err(DbError::from)
    .map_err(Into::into)
}

/// Upstream `setCachedResult` (store.ts:2693-2699): `INSERT OR REPLACE`
/// then prune to the newest [`PRUNE_KEEP`] rows — deterministically
/// every [`PRUNE_EVERY`] writes instead of upstream's 1% chance (T5).
pub(crate) fn set_cached_result(
    conn: &Connection,
    hash: &str,
    result: &str,
    writes: &AtomicUsize,
) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO llm_cache (hash, result, created_at) \
         VALUES (?, ?, ?)",
        params![hash, result, now_iso()],
    )
    .map_err(DbError::from)?;
    if writes.fetch_add(1, Ordering::Relaxed) + 1 >= PRUNE_EVERY {
        writes.store(0, Ordering::Relaxed);
        conn.execute(
            "DELETE FROM llm_cache WHERE hash NOT IN \
             (SELECT hash FROM llm_cache ORDER BY created_at DESC LIMIT ?)",
            params![PRUNE_KEEP],
        )
        .map_err(DbError::from)?;
    }
    Ok(())
}

/// Upstream `deleteExpansionCacheEntry` (store.ts:4596-4599) — dud
/// eviction keyed by the same `{query, model}` body used on write.
pub(crate) fn delete_expansion_cache_entry(
    conn: &Connection,
    query: &str,
    model: &str,
) -> Result<()> {
    let key = cache_key(
        "expandQuery",
        &CacheKeyBody {
            query: Some(query),
            model: Some(model),
            chunk: None,
        },
    );
    conn.execute("DELETE FROM llm_cache WHERE hash = ?", params![key])
        .map_err(DbError::from)?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, reason = "idiomatic test-code patterns")]
mod tests {
    use super::*;

    fn mem_conn() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE llm_cache (
               hash TEXT PRIMARY KEY,
               result TEXT NOT NULL,
               created_at TEXT NOT NULL
             );",
        )
        .unwrap();
        conn
    }

    fn count(conn: &Connection) -> i64 {
        conn.query_row("SELECT COUNT(*) FROM llm_cache", [], |r| r.get(0))
            .unwrap()
    }

    #[test]
    fn cache_key_is_deterministic_and_field_sensitive() {
        let body = CacheKeyBody {
            query: Some("q"),
            model: Some("m"),
            chunk: None,
        };
        let k1 = cache_key("expandQuery", &body);
        let k2 = cache_key("expandQuery", &body);
        assert_eq!(k1, k2);
        assert_eq!(k1.len(), 64);
        assert!(
            k1.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );

        // Every component participates in the digest.
        assert_ne!(k1, cache_key("rerank", &body));
        assert_ne!(
            k1,
            cache_key(
                "expandQuery",
                &CacheKeyBody {
                    query: Some("other"),
                    model: Some("m"),
                    chunk: None,
                },
            )
        );
        assert_ne!(
            k1,
            cache_key(
                "expandQuery",
                &CacheKeyBody {
                    query: Some("q"),
                    model: Some("m2"),
                    chunk: None,
                },
            )
        );
        assert_ne!(
            k1,
            cache_key(
                "expandQuery",
                &CacheKeyBody {
                    query: Some("q"),
                    model: Some("m"),
                    chunk: Some("c"),
                },
            )
        );
        // Absent fields must not serialize at all — `{model}` alone is
        // a different JSON body than `{query:null, model}`.
        assert_eq!(
            cache_key(
                "expandQuery",
                &CacheKeyBody {
                    query: None,
                    model: Some("m"),
                    chunk: None,
                },
            ),
            cache_key(
                "expandQuery",
                &CacheKeyBody {
                    query: None,
                    model: Some("m"),
                    chunk: None,
                },
            )
        );
    }

    #[test]
    fn get_set_overwrite_roundtrip() {
        let conn = mem_conn();
        let w = AtomicUsize::new(0);
        assert_eq!(get_cached_result(&conn, "h1").unwrap(), None);
        set_cached_result(&conn, "h1", "v1", &w).unwrap();
        assert_eq!(
            get_cached_result(&conn, "h1").unwrap().as_deref(),
            Some("v1")
        );
        // INSERT OR REPLACE overwrites.
        set_cached_result(&conn, "h1", "v2", &w).unwrap();
        assert_eq!(
            get_cached_result(&conn, "h1").unwrap().as_deref(),
            Some("v2")
        );
    }

    #[test]
    fn prune_keeps_newest_1000_rows() {
        let conn = mem_conn();
        for i in 0..1005 {
            conn.execute(
                "INSERT INTO llm_cache (hash, result, created_at) VALUES (?, 'x', ?)",
                params![
                    format!("old{i:04}"),
                    format!("2024-01-01T00:{:02}:{:02}Z", i / 60, i % 60)
                ],
            )
            .unwrap();
        }
        // Counter at the threshold minus one — the next write triggers
        // the deterministic prune (T5: every PRUNE_EVERY writes).
        let w = AtomicUsize::new(PRUNE_EVERY - 1);
        set_cached_result(&conn, "fresh", "y", &w).unwrap();
        assert_eq!(count(&conn), 1000);
        // The just-written row is newest.
        assert_eq!(
            get_cached_result(&conn, "fresh").unwrap().as_deref(),
            Some("y")
        );
        // The 1000 survivors are the newest by created_at — the six
        // oldest rows (old0000..old0005) are gone, old0006 lives.
        assert_eq!(get_cached_result(&conn, "old0000").unwrap(), None);
        assert_eq!(get_cached_result(&conn, "old0005").unwrap(), None);
        assert!(get_cached_result(&conn, "old0006").unwrap().is_some());
        // The counter reset — 99 more writes must not re-prune.
        for i in 0..99 {
            set_cached_result(&conn, &format!("n{i}"), "z", &w).unwrap();
        }
        assert_eq!(count(&conn), 1099);
    }

    #[test]
    fn delete_expansion_entry_removes_only_matching_key() {
        let conn = mem_conn();
        let w = AtomicUsize::new(0);
        let key = cache_key(
            "expandQuery",
            &CacheKeyBody {
                query: Some("q"),
                model: Some("m"),
                chunk: None,
            },
        );
        set_cached_result(&conn, &key, "[]", &w).unwrap();
        set_cached_result(&conn, "other", "x", &w).unwrap();
        delete_expansion_cache_entry(&conn, "q", "m").unwrap();
        assert_eq!(get_cached_result(&conn, &key).unwrap(), None);
        assert_eq!(
            get_cached_result(&conn, "other").unwrap().as_deref(),
            Some("x")
        );
        // A different model resolves a different key — nothing deleted.
        set_cached_result(&conn, &key, "[]", &w).unwrap();
        delete_expansion_cache_entry(&conn, "q", "other-model").unwrap();
        assert!(get_cached_result(&conn, &key).unwrap().is_some());
    }

    // ── AC5 golden (fixtures/hybrid_golden.json → cases.cache_key) ──

    #[test]
    fn cache_key_matches_upstream_golden() {
        let g: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/hybrid_golden.json")).unwrap();
        for case in g["cases"]["cache_key"].as_array().unwrap() {
            let url = case["url"].as_str().unwrap();
            let b = &case["body"];
            let body = CacheKeyBody {
                query: b.get("query").and_then(|v| v.as_str()),
                model: b.get("model").and_then(|v| v.as_str()),
                chunk: b.get("chunk").and_then(|v| v.as_str()),
            };
            assert_eq!(
                cache_key(url, &body),
                case["key"].as_str().unwrap(),
                "{case}"
            );
        }
    }
}
