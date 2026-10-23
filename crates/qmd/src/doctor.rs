//! Diagnostic data for `qmd doctor` — pure reads and the two upstream
//! verification probes; rendering stays in qmd-cli.
//!
//! Ports the data side of upstream `showDoctor` (cli/qmd.ts:4210-4310):
//! the SQLite/sqlite-vec version probes, the active-document and
//! `vectors_vec` checks, the `(model, embed_fingerprint)` grouping of
//! `content_vectors`, `maybeAdoptLegacyEmbeddingFingerprint`
//! (store.ts:2581-2650) and `checkEmbeddingVectorSamples`
//! (cli/qmd.ts:4001-4066). Every entry point takes `&Qmd` so the
//! connection stays private; the two embed-dependent probes take the
//! caller's `&dyn Embedder` (the real [`Qmd::embedder`] or a test
//! fake), matching upstream's `getLlm`/session lookup.
//!
//! Upstream wraps the legacy-fingerprint queries in
//! `withLazyContentVectorMigration`; this schema always has the
//! `embed_fingerprint`/`total_chunks` columns, so the wrapper is a
//! no-op here and the queries run directly.

use crate::Qmd;
use crate::chunk::{TokenChunkOptions, chunk_document_by_tokens};
use crate::error::Result;
use crate::llm::{Embedder, embedding_fingerprint, format_doc_for_embedding};
use crate::store::documents::extract_title;
use crate::store::status;
use crate::{js, utf16_len};

/// Upstream's cosine-distance match threshold (store.ts:2639,
/// cli/qmd.ts:4024) — a stored vector this close to a fresh embed
/// counts as a reproduction.
const VECTOR_MATCH_THRESHOLD: f64 = 0.0001;

/// `SELECT sqlite_version()` — upstream's "SQLite runtime" probe
/// (cli/qmd.ts:4218-4224).
///
/// # Errors
/// [`crate::Error::Db`] when the query fails — upstream catches the same
/// failure and prints the error text as the check detail.
pub fn sqlite_version(qmd: &Qmd) -> Result<String> {
    let conn = qmd.lock_conn();
    status::sqlite_version(&conn)
}

/// `SELECT vec_version()` — upstream's "sqlite-vec" probe
/// (cli/qmd.ts:4228-4234). `Err` (extension absent or query failure)
/// carries the message the CLI prints as the check detail.
///
/// # Errors
/// [`crate::Error::Db`] when `vec_version()` is missing or fails.
pub fn vec_version(qmd: &Qmd) -> Result<String> {
    let conn = qmd.lock_conn();
    status::vec_version(&conn)
}

/// `SELECT COUNT(*) FROM documents WHERE active = 1` — upstream's
/// `activeDocs`/`totalDocs` probes (cli/qmd.ts:4003, store.ts:2658).
///
/// # Errors
/// [`crate::Error::Db`] on query failure.
pub fn active_doc_count(qmd: &Qmd) -> Result<usize> {
    let conn = qmd.lock_conn();
    status::active_doc_count(&conn)
}

/// Whether the `vectors_vec` shadow table exists — upstream's
/// `sqlite_master` lookup (`vecTableExists`, cli/qmd.ts:4008, and
/// `hasVectorIndex`, store.ts:2607-2609).
///
/// # Errors
/// [`crate::Error::Db`] on query failure.
pub fn vectors_table_exists(qmd: &Qmd) -> Result<bool> {
    let conn = qmd.lock_conn();
    status::has_vectors_vec(&conn)
}

/// One `(model, embed_fingerprint)` group of `content_vectors` —
/// upstream's fingerprint-summary row (cli/qmd.ts:4264-4268).
#[derive(Debug)]
#[non_exhaustive]
pub struct FingerprintGroup {
    /// `content_vectors.model`.
    pub model: String,
    /// `content_vectors.embed_fingerprint` (`""` = legacy rows).
    pub fingerprint: String,
    /// `COUNT(DISTINCT hash)` — distinct documents in the group.
    pub docs: usize,
    /// `COUNT(*)` — embedded chunks in the group.
    pub chunks: usize,
}

/// Upstream's fingerprint-groups query (cli/qmd.ts:4263-4269): rows for
/// the "embedding fingerprints"/"mixed named embedding fingerprints"
/// checks, ordered most-chunks-first.
///
/// # Errors
/// [`crate::Error::Db`] on query failure.
pub fn fingerprint_groups(qmd: &Qmd) -> Result<Vec<FingerprintGroup>> {
    let conn = qmd.lock_conn();
    Ok(status::fingerprint_groups(&conn)?
        .into_iter()
        .map(|r| FingerprintGroup {
            model: r.model,
            fingerprint: r.fingerprint,
            docs: r.docs,
            chunks: r.chunks,
        })
        .collect())
}

/// The model `qmd embed` would tag vectors with — upstream
/// `resolveEmbedModelForCli()` modulo an injected [`Embedder`], which
/// is authoritative here (P2-D4).
///
/// # Errors
/// Propagates config load errors from [`Qmd::config`].
pub fn resolved_embed_model(qmd: &Qmd) -> Result<String> {
    let config = qmd.config()?;
    Ok(qmd.embed_model(&config))
}

/// The fingerprint `qmd embed` would stamp now — upstream
/// `getEmbeddingFingerprint(embedModel)` plus this port's
/// `embed_pooling` extra (D14, `hashes_needing_embedding`).
///
/// # Errors
/// Propagates config load errors from [`Qmd::config`].
pub fn current_embed_fingerprint(qmd: &Qmd) -> Result<String> {
    let config = qmd.config()?;
    let model = qmd.embed_model(&config);
    let extra = qmd.embed_fingerprint_extra(&config);
    Ok(embedding_fingerprint(&model, extra.as_deref()))
}

/// Pending-embedding count for an explicit model + fingerprint extra.
///
/// `getHashesNeedingEmbedding(db, undefined, model)` (cli/qmd.ts:4253,
/// calling store.ts:2546-2566). `qmd doctor` passes the *declared*
/// model — the `resolveModelsForCli` triple, not the trust-gated
/// runtime one — matching upstream's display semantics (#889).
///
/// # Errors
/// [`crate::Error::Db`] on query failure.
pub fn pending_embeddings_for(qmd: &Qmd, model: &str, extra: Option<&str>) -> Result<usize> {
    let conn = qmd.lock_conn();
    status::hashes_needing_embedding(&conn, None, model, extra)
}

/// Upstream `LegacyFingerprintAdoptionResult` (store.ts:2575-2579).
#[derive(Debug)]
#[non_exhaustive]
pub struct LegacyFingerprintAdoption {
    /// Whether the probe ran a compatibility check (`false` = no legacy
    /// rows, or no testable sample).
    pub checked: bool,
    /// `content_vectors` rows adopted to the current fingerprint.
    pub adopted: usize,
    /// Upstream's `reason` string, verbatim.
    pub reason: String,
}

/// `maybeAdoptLegacyEmbeddingFingerprint` (store.ts:2581-2650): when
/// legacy empty-fingerprint vectors reproduce under `embedder`, stamp
/// them with the current fingerprint instead of forcing a re-embed.
///
/// Model and fingerprint derive from `embedder` — the same identity
/// `generate_embeddings` stamps (store/embed.rs). A failed re-embed is
/// a checked-but-not-adopted verdict, not an error — upstream's
/// `session.embed` returns `null` the same way.
///
/// # Errors
/// [`crate::Error::Db`] on query failure; [`crate::Error::Inference`]
/// when tokenization/chunking fails (upstream's rejected
/// `chunkDocumentByTokens` lands in the caller's catch the same way).
pub fn maybe_adopt_legacy_embedding_fingerprint(
    qmd: &Qmd,
    embedder: &dyn Embedder,
) -> Result<LegacyFingerprintAdoption> {
    let model = embedder.model_id();
    let fingerprint = embedding_fingerprint(
        model,
        embedder.fingerprint_extra().filter(|s| !s.is_empty()),
    );
    let result = |checked, adopted, reason: String| {
        Ok(LegacyFingerprintAdoption {
            checked,
            adopted,
            reason,
        })
    };

    // The probe phase ends at the block boundary — re-chunking and
    // re-embedding can be slow, so the store lock is not held across
    // them (upstream's async store holds no lock either).
    let (legacy_count, sample, has_vec) = {
        let conn = qmd.lock_conn();
        (
            status::legacy_fingerprint_doc_count(&conn, model)?,
            status::legacy_fingerprint_sample(&conn, model)?,
            status::has_vectors_vec(&conn)?,
        )
    };

    if legacy_count == 0 {
        return result(
            false,
            0,
            "no legacy empty-fingerprint embeddings".to_owned(),
        );
    }
    let Some(sample) = sample else {
        return result(
            false,
            0,
            format!("{legacy_count} legacy docs have no active sample"),
        );
    };
    if !has_vec {
        return result(false, 0, "vectors_vec table is missing".to_owned());
    }

    let expected = format!("{}_{}", sample.hash, sample.seq);
    let chunks = chunk_document_by_tokens(
        embedder,
        &sample.body,
        &TokenChunkOptions {
            filepath: Some(sample.path.clone()),
            ..TokenChunkOptions::default()
        },
    )?;
    let Some(chunk) = chunks.get(sample.seq) else {
        return result(true, 0, format!("sample chunk {expected} no longer exists"));
    };

    let title = extract_title(&sample.body, &sample.path);
    let text = format_doc_for_embedding(
        &chunk.text,
        Some(title.as_str()).filter(|s| !s.is_empty()),
        Some(model),
    );
    let embedding = embedder
        .embed(&[&text])
        .ok()
        .and_then(|v| v.into_iter().next())
        .filter(|v| !v.is_empty());
    let Some(embedding) = embedding else {
        return result(true, 0, "failed to embed legacy sample".to_owned());
    };

    let conn = qmd.lock_conn();
    let Some((hash_seq, distance)) = status::vec_nearest(&conn, &embedding)? else {
        return result(true, 0, "legacy sample vector not found".to_owned());
    };
    if hash_seq != expected || distance > VECTOR_MATCH_THRESHOLD {
        return result(
            true,
            0,
            format!(
                "legacy sample differs from current fingerprint (nearest {hash_seq}, distance {})",
                js::to_fixed(distance, 6)
            ),
        );
    }

    let adopted = status::adopt_legacy_fingerprints(&conn, model, &fingerprint)?;
    drop(conn);
    result(
        true,
        adopted,
        format!(
            "sample {expected} matched current fingerprint at distance {}",
            js::to_fixed(distance, 6)
        ),
    )
}

/// Upstream `DoctorVectorSampleResult` (cli/qmd.ts:3776-3779).
#[derive(Debug)]
#[non_exhaustive]
pub struct VectorSampleReport {
    /// Every sampled stored vector reproduced within the threshold.
    pub ok: bool,
    /// Upstream's `details` string, verbatim — the CLI prints it as the
    /// check detail.
    pub details: String,
}

/// `checkEmbeddingVectorSamples` (cli/qmd.ts:4001-4066): re-chunk and
/// re-embed up to `sample_size` random current-fingerprint chunks and
/// compare each against the stored `vectors_vec` blob.
///
/// A failed re-embed joins the mismatch list (upstream's `null` from
/// `session.embed`), it does not abort the probe.
///
/// # Errors
/// [`crate::Error::Db`] on query failure; [`crate::Error::Inference`]
/// when tokenization/chunking fails — upstream's rejected
/// `chunkDocumentByTokens` lands in the caller's catch the same way.
pub fn check_embedding_vector_samples(
    qmd: &Qmd,
    embedder: &dyn Embedder,
    sample_size: usize,
) -> Result<VectorSampleReport> {
    let conn = qmd.lock_conn();
    let report = |ok, details: String| Ok(VectorSampleReport { ok, details });

    if status::active_doc_count(&conn)? == 0 {
        return report(true, "no active documents indexed".to_owned());
    }
    if !status::has_vectors_vec(&conn)? {
        return report(
            false,
            "no vector table to test; please run qmd embed again".to_owned(),
        );
    }

    let model = embedder.model_id();
    let fingerprint = embedding_fingerprint(
        model,
        embedder.fingerprint_extra().filter(|s| !s.is_empty()),
    );
    let samples = status::current_fingerprint_samples(&conn, model, &fingerprint, sample_size)?;
    if samples.is_empty() {
        return report(
            false,
            "no current embedded chunks to test; please run qmd embed again".to_owned(),
        );
    }
    // Stored blobs come along with the samples so the store lock is
    // released before re-chunking/re-embedding (upstream's async store
    // holds no lock across `session.embed` either).
    let stored = samples
        .iter()
        .map(|s| status::stored_embedding(&conn, &format!("{}_{}", s.hash, s.seq)))
        .collect::<Result<Vec<_>>>()?;
    drop(conn);

    let mut mismatches: Vec<String> = Vec::new();
    for (sample, stored_blob) in samples.iter().zip(&stored) {
        let hash_seq = format!("{}_{}", sample.hash, sample.seq);
        let short = short_hash_seq(&hash_seq);
        let chunks = chunk_document_by_tokens(
            embedder,
            &sample.body,
            &TokenChunkOptions {
                filepath: Some(sample.path.clone()),
                ..TokenChunkOptions::default()
            },
        )?;
        let Some(chunk) = chunks.get(sample.seq) else {
            mismatches.push(format!("{short}: chunk no longer exists"));
            continue;
        };
        let title = extract_title(&sample.body, &sample.path);
        let text = format_doc_for_embedding(
            &chunk.text,
            Some(title.as_str()).filter(|s| !s.is_empty()),
            Some(model),
        );
        let embedding = embedder
            .embed(&[&text])
            .ok()
            .and_then(|v| v.into_iter().next())
            .filter(|v| !v.is_empty());
        let Some(embedding) = embedding else {
            mismatches.push(format!("{short}: embedding failed"));
            continue;
        };
        let Some(blob) = stored_blob else {
            mismatches.push(format!("{short}: stored vector missing"));
            continue;
        };
        let distance = cosine_distance(&embedding, &vec_from_blob(blob));
        if distance > VECTOR_MATCH_THRESHOLD {
            mismatches.push(format!(
                "{short}: stored vector distance {}",
                js::to_fixed(distance, 6)
            ));
        }
    }

    if let Some(first) = mismatches.first() {
        return report(
            false,
            format!(
                "{}/{} sampled chunks differ from stored vectors ({first}). \
                 Rebuild with `qmd embed --force`",
                mismatches.len(),
                samples.len()
            ),
        );
    }
    report(
        true,
        format!(
            "{} sampled {} reproduce stored vectors",
            samples.len(),
            if samples.len() == 1 {
                "chunk"
            } else {
                "chunks"
            }
        ),
    )
}

/// Upstream `shortHashSeq` (cli/qmd.ts:3770-3774): a `hash_seq` renders
/// as `{first 12 hash chars}_{seq}`; without a `_`, strings over 18
/// units truncate to 18 + `"..."`.
#[must_use]
pub fn short_hash_seq(hash_seq: &str) -> String {
    match hash_seq.rfind('_') {
        // `hash` is a hex digest, so byte and UTF-16 boundaries agree.
        Some(i) => format!(
            "{}_{}",
            hash_seq.get(..12).unwrap_or(hash_seq),
            hash_seq.get(i + 1..).unwrap_or_default()
        ),
        None if utf16_len(hash_seq) > 18 => {
            format!("{}...", hash_seq.get(..18).unwrap_or(hash_seq))
        }
        None => hash_seq.to_owned(),
    }
}

/// `decodeStoredEmbedding` (cli/qmd.ts:3781-3783): the stored
/// little-endian `float32[]` blob back into a vector. A trailing
/// partial f32 is dropped, like the `Float32Array` view upstream.
fn vec_from_blob(blob: &[u8]) -> Vec<f32> {
    blob.as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// `cosineDistance` (cli/qmd.ts:3785-3799): `1 - a·b/(|a||b|)` in f64,
/// `Infinity` on a length mismatch, empty input, or a zero vector.
#[allow(
    clippy::suboptimal_flops,
    reason = "upstream's separate multiply-add — `mul_add` would contract to FMA and shift the 0.0001 threshold's borderline"
)]
fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return f64::INFINITY;
    }
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return f64::INFINITY;
    }
    1.0 - dot / (norm_a.sqrt() * norm_b.sqrt())
}
