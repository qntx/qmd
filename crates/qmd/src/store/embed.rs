//! Embedding pipeline — upstream `generateEmbeddings`
//! (store.ts:1984-2240) and the storage primitives it drives:
//! `getPendingEmbeddingDocs` (1907), `buildEmbeddingBatches` (1931),
//! `getEmbeddingDocsForBatch` (1962), `ensureVecTableInternal` (1474),
//! `insertEmbedding` (4498), `removeIncompleteEmbeddings` (4524), and
//! the collection-scoped half of `clearAllEmbeddings` (4444-4481).
//!
//! Transaction granularity follows T7: every 32-chunk slice's
//! `content_vectors` + `vectors_vec` writes commit in a single
//! transaction *after* the slice's inference work finishes (upstream is
//! per-statement autocommit); forced-retry writes at a document-batch
//! end and `remove_incomplete_embeddings` each get their own
//! transaction.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params_from_iter};

use super::documents::{extract_title, now_iso};
use super::status::has_vectors_vec;
use super::vec::vec_blob;
use crate::error::{DbError, Result};
use crate::llm::{CancelToken, Embedder, InferenceError, format_doc_for_embedding};
use crate::{Error, chunk};

/// Upstream `BATCH_SIZE` (store.ts:2015): chunks per embed call.
const BATCH_SIZE: usize = 32;
/// Upstream `RETRY_AFTER_SUCCESSFUL_CHUNKS` (2016): unrelated successes
/// that unlock another retry pass over the failure queue.
const RETRY_AFTER_SUCCESSFUL_CHUNKS: usize = 64;
/// Upstream `MAX_RETRY_ATTEMPTS` (2017): per-chunk attempt cap.
const MAX_RETRY_ATTEMPTS: u32 = 3;
/// Upstream `DEFAULT_EMBED_MAX_DOCS_PER_BATCH` (store.ts:104).
pub(crate) const DEFAULT_MAX_DOCS_PER_BATCH: usize = 64;
/// Upstream `DEFAULT_EMBED_MAX_BATCH_BYTES` (store.ts:105).
pub(crate) const DEFAULT_MAX_BATCH_BYTES: usize = 64 * 1024 * 1024;
/// Upstream `DEFAULT_EMBED_MAX_DURATION_MS` (store.ts:106).
pub(crate) const DEFAULT_MAX_DURATION: Duration = Duration::from_mins(30);

/// Upstream `PendingEmbeddingDoc` (store.ts:1793-1797).
struct PendingDoc {
    hash: String,
    path: String,
    bytes: u64,
}

/// Upstream `ChunkItem` (store.ts:1807-1818). Owned text — retry-queue
/// entries outlive the document bodies they were chunked from.
#[derive(Debug, Clone)]
struct ChunkItem {
    hash: String,
    path: String,
    title: String,
    text: String,
    seq: usize,
    pos: usize,
    bytes: usize,
    expected_total_chunks: usize,
}

/// One recorded chunk failure — upstream `EmbedFailure`
/// (store.ts:1749-1755); `Qmd::embed` re-exports the public twin.
#[derive(Debug, Clone)]
pub(crate) struct EmbedFailureRow {
    pub(crate) path: String,
    pub(crate) hash: String,
    pub(crate) seq: usize,
    pub(crate) attempts: u32,
    pub(crate) reason: String,
    /// Upstream `Map` insertion order — kept on re-record, refreshed
    /// when the chunk re-fails after a `clear_failure`.
    pub(crate) order: u64,
}

/// What `Qmd::embed` turns into the public `EmbedReport`
/// (upstream `EmbedResult`, store.ts:1767-1774).
pub(crate) struct EmbedOutcome {
    /// `docsProcessed` — upstream reports the pending-document count
    /// selected at call start, not actually processed (store.ts:2237;
    /// even on mid-run timeout).
    pub(crate) docs_processed: usize,
    /// `chunksEmbedded` after `remove_incomplete_embeddings` adjustment.
    pub(crate) chunks_embedded: usize,
    /// `errors`: outstanding failed chunks (upstream `failures.size`).
    pub(crate) errors: usize,
    /// First-failure-ordered failure records (upstream `failures` list).
    pub(crate) failures: Vec<EmbedFailureRow>,
    /// `durationMs` — measured from after the pending scan
    /// (upstream `startTime`, store.ts:2006); `0` on the empty early
    /// return.
    pub(crate) duration: Duration,
}

/// Embedding-run inputs, resolved by `Qmd::embed` from the public
/// `EmbedOptions` (upstream `resolveEmbedOptions`, store.ts:1827-1831).
pub(crate) struct EmbedRunOptions {
    /// `force`: clear existing vectors first (upstream
    /// `clearAllEmbeddings`).
    pub(crate) force: bool,
    /// Restrict pending docs to one collection.
    pub(crate) collection: Option<String>,
    /// `maxDocsPerBatch`.
    pub(crate) max_docs_per_batch: usize,
    /// `maxBatchBytes`.
    pub(crate) max_batch_bytes: usize,
    /// `chunkStrategy`.
    pub(crate) chunk_strategy: chunk::ChunkStrategy,
    /// Session cap; `None`/zero = unlimited (upstream `maxDuration <= 0`
    /// skips the timer entirely).
    pub(crate) max_duration: Option<Duration>,
    /// Cooperative cancellation — upstream `options.signal`.
    pub(crate) cancel: CancelToken,
}

/// Progress snapshot for the caller's callback — the crate-local twin
/// of public `EmbedProgress` (upstream store.ts:1757-1765).
pub(crate) struct Progress {
    pub(crate) chunks_embedded: usize,
    pub(crate) total_chunks: usize,
    pub(crate) bytes_processed: u64,
    pub(crate) total_bytes: u64,
    pub(crate) errors: usize,
    pub(crate) failures: Vec<EmbedFailureRow>,
}

/// Upstream `SessionReleasedError`'s default message (llm.ts:1977-1981)
/// — `withOperation` throws it on every `!isValid` call, so this is the
/// only reason string that ever surfaces through `session.embed`/
/// `embedBatch` after abort (the reason-forwarding branch at
/// llm.ts:2060-2062 is unreachable).
const SESSION_ABORTED_REASON: &str = "LLM session has been released or aborted";

/// Session validity — upstream `session.isValid` is
/// `!released && !abortController.signal.aborted` where the abort fires
/// from the caller's signal or the `maxDuration` timer
/// (llm.ts:2012-2025). `isValid` only ever flips one way.
struct Session {
    /// The caller's token — upstream `LLMSessionOptions.signal`.
    user_cancel: CancelToken,
    /// Token handed to long-running work (the chunker) — upstream
    /// `session.signal` fires on *both* the caller's signal and the
    /// `maxDuration` timer, so when a cap is set a watcher thread flips
    /// this token at the deadline. Uncapped sessions reuse the caller's
    /// token directly and spawn no thread.
    work_cancel: CancelToken,
    deadline: Option<Instant>,
    /// Stops the deadline watcher once the run ends (`Drop`).
    watcher_done: Arc<AtomicBool>,
}

impl Session {
    fn new(cancel: CancelToken, max_duration: Option<Duration>) -> Self {
        // Upstream `if (maxDuration > 0)` — a zero duration disables the
        // cap.
        let max_duration = max_duration.filter(|d| *d > Duration::ZERO);
        let Some(cap) = max_duration else {
            return Self {
                work_cancel: cancel.clone(),
                user_cancel: cancel,
                deadline: None,
                watcher_done: Arc::new(AtomicBool::new(false)),
            };
        };
        let deadline = Instant::now() + cap;
        let work_cancel = CancelToken::new();
        let watcher_done = Arc::new(AtomicBool::new(false));
        spawn_deadline_watcher(
            cancel.clone(),
            work_cancel.clone(),
            deadline,
            Arc::clone(&watcher_done),
        );
        Self {
            user_cancel: cancel,
            work_cancel,
            deadline: Some(deadline),
            watcher_done,
        }
    }

    fn is_valid(&self) -> bool {
        !self.user_cancel.is_cancelled() && self.deadline.is_none_or(|d| Instant::now() < d)
    }

    /// The error upstream propagates when an invalid session is used —
    /// `SessionReleasedError` (llm.ts:2053-2054). A user cancellation
    /// maps to [`Error::Cancelled`] instead.
    fn aborted_error(&self) -> Error {
        if self.user_cancel.is_cancelled() {
            Error::Cancelled
        } else {
            Error::from(InferenceError::Backend(SESSION_ABORTED_REASON.to_owned()))
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.watcher_done.store(true, Ordering::Relaxed);
    }
}

/// Upstream's `maxDuration` timer (llm.ts:2014-2018): the session's
/// abort signal fires on the caller's token *or* the deadline. Poll
/// both and flip `work` — the token long-running work (the chunker)
/// observes mid-operation.
#[allow(
    clippy::disallowed_methods,
    reason = "dedicated watcher thread — a timed sleep until the next checkpoint is exactly what is needed; there is no async runtime here"
)]
fn spawn_deadline_watcher(
    user: CancelToken,
    work: CancelToken,
    deadline: Instant,
    done: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        loop {
            if done.load(Ordering::Relaxed) {
                break;
            }
            if user.is_cancelled() || Instant::now() >= deadline {
                work.cancel();
                break;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            std::thread::sleep(remaining.min(Duration::from_millis(25)));
        }
    });
}

/// `hash:seq` — upstream `chunkKey` (store.ts:2022).
fn chunk_key(hash: &str, seq: usize) -> String {
    format!("{hash}:{seq}")
}

/// Upstream `reasonFromError` (store.ts:2027-2030): messages longer
/// than 180 UTF-16 code units truncate to 177 plus an ellipsis.
fn reason_from_error(err: &Error) -> String {
    let raw = err.to_string();
    if chunk::utf16_len(&raw) <= 180 {
        return raw;
    }
    let cut = chunk::utf16_to_byte_offset(&raw, 177);
    format!("{}...", &raw[..cut])
}

/// Upstream `getPendingEmbeddingDocs` (store.ts:1907-1929): active
/// document hashes whose `content_vectors` rows for `model` are absent
/// or incomplete (`chunk_count < expected_chunks`), one row per hash,
/// ordered by `MIN(path)`.
fn pending_embedding_docs(
    conn: &Connection,
    collection: Option<&str>,
    model: &str,
    fingerprint: &str,
) -> Result<Vec<PendingDoc>> {
    let sql = format!(
        "SELECT d.hash, MIN(d.path) AS path, length(CAST(c.doc AS BLOB)) AS bytes \
         FROM documents d \
         JOIN content c ON d.hash = c.hash \
         LEFT JOIN ( \
           SELECT hash, model, COUNT(*) AS chunk_count, MAX(total_chunks) AS expected_chunks \
           FROM content_vectors \
           WHERE model = ? AND embed_fingerprint = ? \
           GROUP BY hash, model, embed_fingerprint \
         ) v ON d.hash = v.hash \
         WHERE d.active = 1 \
           AND (v.hash IS NULL OR v.chunk_count < v.expected_chunks) {} \
         GROUP BY d.hash \
         ORDER BY MIN(d.path)",
        if collection.is_some() {
            "AND d.collection = ?"
        } else {
            ""
        }
    );
    let mut params = vec![
        rusqlite::types::Value::from(model.to_owned()),
        rusqlite::types::Value::from(fingerprint.to_owned()),
    ];
    if let Some(c) = collection {
        params.push(c.to_owned().into());
    }
    let docs = conn
        .prepare(&sql)
        .and_then(|mut s| {
            s.query_map(params_from_iter(params), |r| {
                Ok(PendingDoc {
                    hash: r.get(0)?,
                    path: r.get(1)?,
                    bytes: u64::try_from(r.get::<_, i64>(2)?).unwrap_or(0),
                })
            })
            .and_then(|m| m.collect::<rusqlite::Result<Vec<_>>>())
        })
        .map_err(DbError::from)?;
    Ok(docs)
}

/// Upstream `buildEmbeddingBatches` (store.ts:1931-1960): greedy fill —
/// a doc closes the current batch when it would push past
/// `max_docs_per_batch`, or past `max_batch_bytes` with a non-empty
/// batch already.
fn build_embedding_batches(
    docs: &[PendingDoc],
    max_docs_per_batch: usize,
    max_batch_bytes: usize,
) -> Vec<Vec<&PendingDoc>> {
    let mut batches: Vec<Vec<&PendingDoc>> = Vec::new();
    let mut current: Vec<&PendingDoc> = Vec::new();
    let mut current_bytes = 0u64;
    let max_bytes = u64::try_from(max_batch_bytes).unwrap_or(u64::MAX);

    for doc in docs {
        let doc_bytes = doc.bytes;
        let over_docs = current.len() >= max_docs_per_batch;
        let over_bytes = !current.is_empty() && current_bytes.saturating_add(doc_bytes) > max_bytes;
        if over_docs || over_bytes {
            batches.push(std::mem::take(&mut current));
            current_bytes = 0;
        }
        current.push(doc);
        current_bytes += doc_bytes;
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

/// Upstream `getEmbeddingDocsForBatch` (store.ts:1962-1978): hash →
/// body, preserving batch order; a hash absent from `content` maps to
/// `""`.
fn embedding_docs_for_batch<'a>(
    conn: &Connection,
    batch: &[&'a PendingDoc],
) -> Result<Vec<(&'a PendingDoc, String)>> {
    if batch.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = vec!["?"; batch.len()].join(",");
    let sql = format!("SELECT hash, doc AS body FROM content WHERE hash IN ({placeholders})");
    let rows = conn
        .prepare(&sql)
        .and_then(|mut s| {
            s.query_map(
                params_from_iter(batch.iter().map(|d| d.hash.clone())),
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
            )
            .and_then(|m| m.collect::<rusqlite::Result<Vec<_>>>())
        })
        .map_err(DbError::from)?;
    let body_by_hash: BTreeMap<String, String> = rows.into_iter().collect();
    Ok(batch
        .iter()
        .map(|doc| {
            (
                *doc,
                body_by_hash.get(&doc.hash).cloned().unwrap_or_default(),
            )
        })
        .collect())
}

/// Upstream `ensureVecTableInternal` (store.ts:1474-1496): create
/// `vectors_vec` at `dimensions`; reuse a table with matching
/// dims/`hash_seq`/cosine, throw on a dimension mismatch, rebuild one
/// missing those markers.
fn ensure_vec_table(conn: &Connection, dimensions: usize) -> Result<()> {
    let existing: Option<String> = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='vectors_vec'",
            [],
            |r| r.get(0),
        )
        .optional()
        .map_err(DbError::from)?;

    if let Some(sql) = existing {
        let existing_dims = sql
            .split("float[")
            .nth(1)
            .and_then(|rest| rest.split(']').next())
            .and_then(|n| n.parse::<usize>().ok());
        let has_hash_seq = sql.contains("hash_seq");
        let has_cosine = sql.contains("distance_metric=cosine");
        if existing_dims == Some(dimensions) && has_hash_seq && has_cosine {
            return Ok(());
        }
        if let Some(dims) = existing_dims
            && dims != dimensions
        {
            // Upstream appends "Run 'qmd embed -f' to re-embed with the
            // new model." — a CLI hint the structured error drops.
            return Err(Error::EmbeddingDimensionMismatch {
                existing: dims,
                current: dimensions,
            });
        }
        conn.execute_batch("DROP TABLE IF EXISTS vectors_vec")
            .map_err(DbError::from)?;
    }
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE vectors_vec USING vec0(\
           hash_seq TEXT PRIMARY KEY, embedding float[{dimensions}] distance_metric=cosine)"
    ))
    .map_err(DbError::from)?;
    Ok(())
}

/// Upstream `insertEmbedding` (store.ts:4498-4522): `content_vectors`
/// first (crash-safe ordering — the pending scan only reads
/// `content_vectors`), then `vectors_vec` via DELETE + INSERT (vec0
/// silently ignores OR REPLACE). Called inside the slice's transaction.
fn insert_embedding(
    conn: &Connection,
    chunk: &ChunkItem,
    embedding: &[f32],
    model: &str,
    embedded_at: &str,
    fingerprint: &str,
) -> Result<()> {
    let hash_seq = format!("{}_{}", chunk.hash, chunk.seq);
    conn.execute(
        "INSERT OR REPLACE INTO content_vectors \
           (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        rusqlite::params![
            chunk.hash,
            i64::try_from(chunk.seq).unwrap_or(0),
            i64::try_from(chunk.pos).unwrap_or(0),
            model,
            fingerprint,
            i64::try_from(chunk.expected_total_chunks).unwrap_or(0),
            embedded_at
        ],
    )
    .map_err(DbError::from)?;
    conn.execute(
        "DELETE FROM vectors_vec WHERE hash_seq = ?",
        rusqlite::params![hash_seq],
    )
    .map_err(DbError::from)?;
    conn.execute(
        "INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)",
        rusqlite::params![hash_seq, rusqlite::types::Value::Blob(vec_blob(embedding))],
    )
    .map_err(DbError::from)?;
    Ok(())
}

/// Upstream `removeIncompleteEmbeddings` (store.ts:4524-4547): for each
/// expected hash, drop *all* vec + content rows when the stored chunk
/// count differs from the expected total; returns rows removed.
fn remove_incomplete_embeddings(
    conn: &Connection,
    expected_chunks_by_hash: &BTreeMap<String, usize>,
    model: &str,
) -> Result<usize> {
    let mut removed = 0usize;
    let mut rows_stmt = conn
        .prepare("SELECT seq FROM content_vectors WHERE hash = ? AND model = ?")
        .map_err(DbError::from)?;
    for (hash, expected) in expected_chunks_by_hash {
        let seqs = rows_stmt
            .query_map(rusqlite::params![hash, model], |r| r.get::<_, i64>(0))
            .and_then(|m| m.collect::<rusqlite::Result<Vec<_>>>())
            .map_err(DbError::from)?;
        if seqs.is_empty() || seqs.len() == *expected {
            continue;
        }
        for seq in &seqs {
            conn.execute(
                "DELETE FROM vectors_vec WHERE hash_seq = ?",
                rusqlite::params![format!("{hash}_{seq}")],
            )
            .map_err(DbError::from)?;
        }
        conn.execute(
            "DELETE FROM content_vectors WHERE hash = ? AND model = ?",
            rusqlite::params![hash, model],
        )
        .map_err(DbError::from)?;
        removed += seqs.len();
    }
    Ok(removed)
}

/// Collection-scoped half of upstream `clearAllEmbeddings`
/// (store.ts:4444-4481): delete vectors whose hash is referenced
/// exclusively by active documents in `collection`; drop `vectors_vec`
/// when `content_vectors` ends up empty.
pub(crate) fn clear_collection_embeddings(conn: &Connection, collection: &str) -> Result<()> {
    let exclusive = "SELECT DISTINCT d.hash FROM documents d \
        WHERE d.collection = ? AND d.active = 1 \
          AND NOT EXISTS ( \
            SELECT 1 FROM documents d2 \
            WHERE d2.hash = d.hash AND d2.active = 1 AND d2.collection != d.collection \
          )";

    if has_vectors_vec(conn)? {
        let rows = conn
            .prepare(&format!(
                "SELECT cv.hash, cv.seq FROM content_vectors cv WHERE cv.hash IN ({exclusive})"
            ))
            .and_then(|mut s| {
                s.query_map(rusqlite::params![collection], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
                })
                .and_then(|m| m.collect::<rusqlite::Result<Vec<_>>>())
            })
            .map_err(DbError::from)?;
        for (hash, seq) in rows {
            conn.execute(
                "DELETE FROM vectors_vec WHERE hash_seq = ?",
                rusqlite::params![format!("{hash}_{seq}")],
            )
            .map_err(DbError::from)?;
        }
    }

    conn.execute(
        &format!("DELETE FROM content_vectors WHERE hash IN ({exclusive})"),
        rusqlite::params![collection],
    )
    .map_err(DbError::from)?;

    let remaining: i64 = conn
        .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))
        .map_err(DbError::from)?;
    if remaining == 0 {
        conn.execute_batch("DROP TABLE IF EXISTS vectors_vec")
            .map_err(DbError::from)?;
    }
    Ok(())
}

/// Per-run state — upstream's session-scoped `failures`/`retryQueue`/
/// `successesSinceRetry` (store.ts:2019-2021), which deliberately
/// accumulate across document batches.
struct EmbedRun<'e, 'p> {
    embedder: &'e dyn Embedder,
    /// Upstream `options.onProgress` — invoked after every chunk slice
    /// and every document batch.
    progress: &'p mut dyn FnMut(&Progress),
    model: String,
    fingerprint: String,
    /// `now` — upstream stamps one ISO timestamp on every row of a run.
    embedded_at: String,
    session: Session,
    /// Upstream's insertion-ordered `failures`/`retryQueue` Maps —
    /// keyed lookup here; `EmbedFailureRow::order` reproduces the
    /// iteration order for reports and retry passes.
    failures: BTreeMap<String, EmbedFailureRow>,
    retry_queue: BTreeMap<String, ChunkItem>,
    next_order: u64,
    successes_since_retry: usize,
    chunks_embedded: usize,
    /// Successful writes of the current slice, flushed in one
    /// transaction (T7).
    pending_writes: Vec<(ChunkItem, Vec<f32>)>,
    /// Chunks discovered so far across processed batches
    /// (`totalChunks` progress field).
    total_chunks: usize,
    /// Byte total of the pending set (`totalBytes` progress field).
    total_bytes: u64,
    /// Bytes of completed document batches (`bytesProcessed` — slices
    /// report `bytes_processed + proportional` without mutating it).
    bytes_processed: u64,
    /// Upstream `vectorTableInitialized` — reset per run so a stale
    /// wrong-dimension table still raises `EmbeddingDimensionMismatch`.
    vector_table_initialized: bool,
}

impl EmbedRun<'_, '_> {
    /// Upstream `recordFailure` (store.ts:2031-2039): `Map.set` on a
    /// live key updates in place (order preserved); a key that was
    /// cleared re-enters at the end.
    fn record_failure(&mut self, chunk: &ChunkItem, reason: String) {
        let key = chunk_key(&chunk.hash, chunk.seq);
        let (attempts, order) = if let Some(f) = self.failures.get(&key) {
            (f.attempts, f.order)
        } else {
            let o = self.next_order;
            self.next_order += 1;
            (0, o)
        };
        self.failures.insert(
            key.clone(),
            EmbedFailureRow {
                path: chunk.path.clone(),
                hash: chunk.hash.clone(),
                seq: chunk.seq,
                attempts: attempts + 1,
                reason,
                order,
            },
        );
        self.retry_queue.insert(key, chunk.clone());
    }

    /// Record the same failure for a run of chunks (upstream marks each
    /// `remainingChunks` element — store.ts:2158-2169).
    fn mark_failed(&mut self, chunks: &[ChunkItem], reason: &str) {
        for c in chunks {
            self.record_failure(c, reason.to_owned());
        }
    }

    /// Upstream `clearFailure` (store.ts:2040-2043).
    fn clear_failure(&mut self, chunk: &ChunkItem) {
        let key = chunk_key(&chunk.hash, chunk.seq);
        self.failures.remove(&key);
        self.retry_queue.remove(&key);
    }

    /// Upstream `tryEmbedChunk` (store.ts:2041-2063): single-text embed;
    /// success queues the write and clears any recorded failure.
    /// Returns whether the chunk embedded.
    fn try_embed_chunk(&mut self, chunk: &ChunkItem) -> bool {
        // Upstream `session.embed` throws `SessionReleasedError` on an
        // aborted session (llm.ts:2053-2054) → the chunk is recorded
        // failed, not embedded.
        if !self.session.is_valid() {
            self.record_failure(chunk, SESSION_ABORTED_REASON.to_owned());
            return false;
        }
        let text = format_doc_for_embedding(&chunk.text, Some(&chunk.title), Some(&self.model));
        match self.embedder.embed(&[&text]) {
            Ok(mut vectors) => {
                if let Some(v) = vectors.pop().filter(|v| !v.is_empty()) {
                    self.pending_writes.push((chunk.clone(), v));
                    self.chunks_embedded += 1;
                    self.successes_since_retry += 1;
                    self.clear_failure(chunk);
                    true
                } else {
                    // Upstream `if (!result)` — "embedding returned no
                    // vector".
                    self.record_failure(chunk, "embedding returned no vector".to_owned());
                    false
                }
            }
            Err(e) => {
                self.record_failure(chunk, reason_from_error(&Error::from(e)));
                false
            }
        }
    }

    /// Failure records in upstream `Map` insertion order
    /// (`failureList`, store.ts:2024).
    fn ordered_failures(&self) -> Vec<EmbedFailureRow> {
        let mut v: Vec<_> = self.failures.values().cloned().collect();
        v.sort_by_key(|f| f.order);
        v
    }

    /// Upstream `retryFailedChunks` (store.ts:2064-2086): normal mode
    /// runs once `RETRY_AFTER_SUCCESSFUL_CHUNKS` unrelated chunks have
    /// succeeded; force mode loops until every outstanding failure
    /// recovers or hits `MAX_RETRY_ATTEMPTS`.
    fn retry_failed(&mut self, force: bool) {
        if !self.session.is_valid() || self.retry_queue.is_empty() {
            return;
        }
        if !force && self.successes_since_retry < RETRY_AFTER_SUCCESSFUL_CHUNKS {
            return;
        }
        self.successes_since_retry = 0;

        loop {
            let mut retried = 0usize;
            // Snapshot keys in upstream insertion order:
            // `try_embed_chunk` mutates the queue.
            let mut keys: Vec<_> = self.retry_queue.keys().cloned().collect();
            keys.sort_by_key(|k| self.failures.get(k).map_or(u64::MAX, |f| f.order));
            for key in keys {
                retried += usize::from(self.retry_one(&key));
            }
            if !force || retried == 0 {
                break;
            }
            let any_under_cap = self.retry_queue.keys().any(|k| {
                self.failures
                    .get(k)
                    .is_some_and(|f| f.attempts < MAX_RETRY_ATTEMPTS)
            });
            if !(self.session.is_valid() && any_under_cap) {
                break;
            }
        }
    }

    /// Upstream `options.onProgress` after a completed document batch.
    fn report(&mut self) {
        let p = Progress {
            chunks_embedded: self.chunks_embedded,
            total_chunks: self.total_chunks,
            bytes_processed: self.bytes_processed,
            total_bytes: self.total_bytes,
            errors: self.failures.len(),
            failures: self.ordered_failures(),
        };
        (self.progress)(&p);
    }

    /// Per-slice progress: upstream reports `bytesProcessed +
    /// proportionalBytes` (store.ts:2209-2216) — completed batches plus
    /// the current batch's chunk-prorated share.
    fn report_slice(&mut self, batch_bytes: u64, total_chunk_bytes: u64, done_chunk_bytes: u64) {
        let proportional = if total_chunk_bytes == 0 {
            batch_bytes
        } else {
            // Upstream `Math.round(processed/total * batchBytes)`.
            batch_bytes
                .saturating_mul(done_chunk_bytes)
                .saturating_add(total_chunk_bytes / 2)
                .checked_div(total_chunk_bytes)
                .unwrap_or(batch_bytes)
                .min(batch_bytes)
        };
        let p = Progress {
            chunks_embedded: self.chunks_embedded,
            total_chunks: self.total_chunks,
            bytes_processed: self.bytes_processed + proportional,
            total_bytes: self.total_bytes,
            errors: self.failures.len(),
            failures: self.ordered_failures(),
        };
        (self.progress)(&p);
    }

    /// One queued retry if the chunk is still under the attempt cap
    /// (upstream skips `attempts >= MAX_RETRY_ATTEMPTS`,
    /// store.ts:2079-2081); returns whether a retry ran.
    fn retry_one(&mut self, key: &str) -> bool {
        let under_cap = self
            .failures
            .get(key)
            .is_some_and(|f| f.attempts < MAX_RETRY_ATTEMPTS);
        if !under_cap {
            return false;
        }
        self.retry_queue.get(key).cloned().is_some_and(|chunk| {
            self.try_embed_chunk(&chunk);
            true
        })
    }

    /// Flush `pending_writes` in a single transaction (T7).
    fn flush_writes(&mut self, conn: &mut Connection) -> Result<()> {
        if self.pending_writes.is_empty() {
            return Ok(());
        }
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(DbError::from)?;
        for (chunk, vector) in self.pending_writes.drain(..) {
            insert_embedding(
                &tx,
                &chunk,
                &vector,
                &self.model,
                &self.embedded_at,
                &self.fingerprint,
            )?;
        }
        tx.commit().map_err(DbError::from)?;
        Ok(())
    }
}

impl EmbedRun<'_, '_> {
    /// One `BATCH_SIZE` slice of a document batch (store.ts:2154-2218):
    /// the session/circuit-breaker checkpoints, the batched embed call
    /// with per-chunk fallback, the slice's write flush (T7) and the
    /// progress event.
    fn embed_slice(
        &mut self,
        conn: &mut Connection,
        batch_chunks: &[ChunkItem],
        batch_start: usize,
        batch_bytes: u64,
        total_batch_chunk_bytes: u64,
        batch_chunk_bytes_processed: &mut u64,
    ) -> Result<SliceAction> {
        // Chunk-slice checkpoint (store.ts:2156-2160): an invalid
        // session marks every remaining chunk failed and skips them.
        if !self.session.is_valid() {
            self.mark_failed(
                batch_chunks.get(batch_start..).unwrap_or_default(),
                "LLM session expired before embedding chunk",
            );
            tracing::warn!("session expired — skipping remaining chunks");
            return Ok(SliceAction::Break);
        }

        // Circuit breaker (store.ts:2163-2170): once a full slice has
        // been attempted and >80% of attempts are failures, the rest
        // of the batch is marked and skipped.
        let processed = self.chunks_embedded + self.failures.len();
        if processed >= BATCH_SIZE && self.failures.len() * 5 > processed * 4 {
            self.mark_failed(
                batch_chunks.get(batch_start..).unwrap_or_default(),
                "embedding aborted because error rate was too high",
            );
            tracing::warn!(
                "error rate too high ({}/{processed}) — aborting embedding",
                self.failures.len()
            );
            return Ok(SliceAction::Break);
        }

        let batch_end = (batch_start + BATCH_SIZE).min(batch_chunks.len());
        let slice = batch_chunks.get(batch_start..batch_end).unwrap_or_default();
        let texts: Vec<String> = slice
            .iter()
            .map(|c| format_doc_for_embedding(&c.text, Some(&c.title), Some(&self.model)))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        match self.embedder.embed(&text_refs) {
            Ok(embeddings) => {
                self.record_slice_results(slice, &embeddings, batch_chunk_bytes_processed);
                self.retry_failed(false);
            }
            Err(e) => self.embed_slice_failed(slice, e, batch_chunk_bytes_processed),
        }

        self.flush_writes(conn)?;
        self.report_slice(
            batch_bytes,
            total_batch_chunk_bytes,
            *batch_chunk_bytes_processed,
        );
        Ok(SliceAction::Continue(batch_end))
    }

    /// Successful batch embed — per-item `null` analogue
    /// (store.ts:2173-2184): the trait contracts
    /// `embeddings.len() == texts.len()`; a short/missing/empty slot is
    /// that chunk's "batch embedding returned no vector".
    fn record_slice_results(
        &mut self,
        slice: &[ChunkItem],
        embeddings: &[Vec<f32>],
        batch_chunk_bytes_processed: &mut u64,
    ) {
        for (i, c) in slice.iter().enumerate() {
            if let Some(v) = embeddings.get(i).filter(|v| !v.is_empty()) {
                self.pending_writes.push((c.clone(), v.clone()));
                self.chunks_embedded += 1;
                self.successes_since_retry += 1;
                self.clear_failure(c);
            } else {
                self.record_failure(c, "batch embedding returned no vector".to_owned());
            }
            *batch_chunk_bytes_processed += c.bytes as u64;
        }
    }

    /// Failed batch embed (store.ts:2193-2207): valid sessions fall back
    /// to per-chunk singles with a retry pass after each; an invalid
    /// session marks the slice "batch failed and session expired".
    fn embed_slice_failed(
        &mut self,
        slice: &[ChunkItem],
        error: InferenceError,
        batch_chunk_bytes_processed: &mut u64,
    ) {
        let batch_reason = reason_from_error(&Error::from(error));
        if self.session.is_valid() {
            for c in slice {
                self.try_embed_chunk(c);
                *batch_chunk_bytes_processed += c.bytes as u64;
                self.retry_failed(false);
            }
        } else {
            self.mark_failed(
                slice,
                &format!("batch failed and session expired: {batch_reason}"),
            );
            *batch_chunk_bytes_processed += slice.iter().map(|c| c.bytes as u64).sum::<u64>();
        }
    }

    /// One document batch (store.ts:2088-2225): fetch bodies → chunk →
    /// dimension probe → chunk slices → forced retry → incomplete-hash
    /// cleanup.
    fn embed_doc_batch(
        &mut self,
        conn: &mut Connection,
        batch_meta: &[&PendingDoc],
        chunk_strategy: chunk::ChunkStrategy,
    ) -> Result<()> {
        let batch_docs = embedding_docs_for_batch(conn, batch_meta)?;
        let mut batch_chunks: Vec<ChunkItem> = Vec::new();
        let mut expected_chunks_by_hash: BTreeMap<String, usize> = BTreeMap::new();
        let batch_bytes: u64 = batch_meta.iter().map(|d| d.bytes).sum();

        for (doc, body) in &batch_docs {
            if body.trim().is_empty() {
                continue;
            }
            let title = extract_title(body, &doc.path);
            let chunks = chunk::chunk_document_by_tokens(
                self.embedder,
                body,
                &chunk::TokenChunkOptions {
                    filepath: Some(doc.path.clone()),
                    strategy: chunk_strategy,
                    cancel: self.session.work_cancel.clone(),
                    ..Default::default()
                },
            )
            .map_err(Error::from)?;

            for (seq, c) in chunks.iter().enumerate() {
                batch_chunks.push(ChunkItem {
                    hash: doc.hash.clone(),
                    path: doc.path.clone(),
                    title: title.clone(),
                    text: c.text.to_string(),
                    seq,
                    pos: c.pos,
                    bytes: c.text.len(),
                    expected_total_chunks: chunks.len(),
                });
            }
            expected_chunks_by_hash.insert(doc.hash.clone(), chunks.len());
        }

        self.total_chunks += batch_chunks.len();

        if batch_chunks.is_empty() {
            self.bytes_processed += batch_bytes;
            self.report();
            return Ok(());
        }

        if !self.vector_table_initialized {
            // Upstream probes the dimension through `session.embed`,
            // which throws on an invalid session (llm.ts:2053) — the
            // whole run rejects rather than embedding.
            if !self.session.is_valid() {
                return Err(self.session.aborted_error());
            }
            // Upstream probes with the first chunk's real embedding
            // (store.ts:2145-2151); `null` is a hard error.
            let Some(first) = batch_chunks.first() else {
                return Ok(());
            };
            let text = format_doc_for_embedding(&first.text, Some(&first.title), Some(&self.model));
            let vector = self
                .embedder
                .embed(&[&text])
                .map_err(Error::from)?
                .into_iter()
                .next()
                .filter(|v| !v.is_empty())
                .ok_or_else(|| {
                    Error::from(InferenceError::Backend(
                        "Failed to get embedding dimensions from first chunk".to_owned(),
                    ))
                })?;
            ensure_vec_table(conn, vector.len())?;
            self.vector_table_initialized = true;
        }

        let total_batch_chunk_bytes: u64 = batch_chunks.iter().map(|c| c.bytes as u64).sum();
        let mut batch_chunk_bytes_processed = 0u64;

        let mut batch_start = 0usize;
        while batch_start < batch_chunks.len() {
            match self.embed_slice(
                conn,
                &batch_chunks,
                batch_start,
                batch_bytes,
                total_batch_chunk_bytes,
                &mut batch_chunk_bytes_processed,
            )? {
                SliceAction::Continue(next) => batch_start = next,
                SliceAction::Break => break,
            }
        }

        // Document-batch end (store.ts:2199-2217): forced retry pass with
        // its own write flush, then incomplete-hash cleanup in its own
        // transaction (T7).
        self.retry_failed(true);
        self.flush_writes(conn)?;

        let removed_partial_chunks = {
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(DbError::from)?;
            let removed = remove_incomplete_embeddings(&tx, &expected_chunks_by_hash, &self.model)?;
            tx.commit().map_err(DbError::from)?;
            removed
        };
        self.chunks_embedded = self.chunks_embedded.saturating_sub(removed_partial_chunks);

        self.bytes_processed += batch_bytes;
        self.report();
        Ok(())
    }
}

/// Whether a chunk-slice pass continues or aborts the document batch.
enum SliceAction {
    Continue(usize),
    Break,
}

/// Upstream `generateEmbeddings` (store.ts:1984-2240).
///
/// `progress` mirrors upstream `onProgress` — after every chunk slice
/// and after every document batch.
pub(crate) fn generate_embeddings(
    conn: &mut Connection,
    embedder: &dyn Embedder,
    opts: &EmbedRunOptions,
    progress: &mut dyn FnMut(&Progress),
) -> Result<EmbedOutcome> {
    let model = embedder.model_id().to_owned();
    let fingerprint_extra = embedder
        .fingerprint_extra()
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let fingerprint = crate::llm::embedding_fingerprint(&model, fingerprint_extra.as_deref());
    // Upstream `const now = new Date().toISOString()` — one timestamp
    // per run, stamped on every row.
    let embedded_at = now_iso();

    if opts.force {
        match &opts.collection {
            None => super::maintenance::clear_all_embeddings(conn)?,
            Some(c) => clear_collection_embeddings(conn, c)?,
        }
    }

    let docs_to_embed =
        pending_embedding_docs(conn, opts.collection.as_deref(), &model, &fingerprint)?;
    if docs_to_embed.is_empty() {
        return Ok(EmbedOutcome {
            docs_processed: 0,
            chunks_embedded: 0,
            errors: 0,
            failures: Vec::new(),
            duration: Duration::ZERO,
        });
    }
    let docs_processed = docs_to_embed.len();
    // Upstream `const startTime = Date.now()` sits between the pending
    // scan and the session.
    let start = Instant::now();
    let total_bytes: u64 = docs_to_embed.iter().map(|d| d.bytes).sum();

    let mut run = EmbedRun {
        embedder,
        progress,
        model,
        fingerprint,
        embedded_at,
        session: Session::new(opts.cancel.clone(), opts.max_duration),
        failures: BTreeMap::new(),
        retry_queue: BTreeMap::new(),
        next_order: 0,
        successes_since_retry: 0,
        chunks_embedded: 0,
        pending_writes: Vec::new(),
        total_chunks: 0,
        total_bytes,
        bytes_processed: 0,
        vector_table_initialized: false,
    };

    let batches = build_embedding_batches(
        &docs_to_embed,
        opts.max_docs_per_batch,
        opts.max_batch_bytes,
    );

    for batch_meta in &batches {
        // Doc-batch checkpoint (store.ts:2090-2093): an invalid session
        // skips the remaining batches silently.
        if !run.session.is_valid() {
            tracing::warn!("session expired — skipping remaining document batches");
            break;
        }
        run.embed_doc_batch(conn, batch_meta, opts.chunk_strategy)?;
    }

    Ok(EmbedOutcome {
        docs_processed,
        chunks_embedded: run.chunks_embedded,
        errors: run.failures.len(),
        failures: run.ordered_failures(),
        duration: start.elapsed(),
    })
}
