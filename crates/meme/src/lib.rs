//! On-device hybrid search for markdown: BM25, vectors and LLM reranking.
//!
//! Rust port of [tobi/qmd](https://github.com/tobi/qmd) (v2.8.3).
//!
//! # Overview
//!
//! [`Meme::builder`] opens a SQLite index (WAL, busy timeout,
//! sqlite-vec); collections come from a YAML config, an inline
//! [`Config`], or the database mirror. [`Meme::update`] scans and indexes
//! collections, and [`Meme::embed`] builds chunk vectors. Search runs
//! through [`Meme::search_lex`] (BM25), [`Meme::search_vector`], and
//! [`Meme::search`] (hybrid: query expansion → BM25 + vector → RRF →
//! rerank — [`Meme::expand_query`] and [`Meme::vector_query`] expose the
//! steps). Documents are fetched by `meme://` virtual path, `#docid`, or
//! filesystem path via [`Meme::get`], [`Meme::document_body`],
//! [`Meme::multi_get`] and [`Meme::list_documents`]; [`MetadataFilter`]
//! gates results on front-matter scalars. [`Meme::status`],
//! [`Meme::index_health`] and [`Meme::maintenance`] report and clean up.
//!
//! # Inference backends
//!
//! Embedding, reranking and query expansion go through the
//! [`Embedder`]/[`Reranker`]/[`Generator`] traits. With the `llama`
//! feature (default) the backends are built lazily from the
//! configured/env models; custom implementations plug in through
//! [`MemeBuilder::embedder`]/[`MemeBuilder::reranker`]/
//! [`MemeBuilder::generator`]. The `testing` feature provides
//! [`FakeEmbedder`]/[`FakeReranker`]/[`FakeGenerator`].
//!
//! # Cargo features
//!
//! | feature | purpose |
//! |---------|---------|
//! | `llama` (default) | llama.cpp inference + Hugging Face model download; needs cmake and clang/libclang |
//! | `ast` (default) | tree-sitter AST break points for [`ChunkStrategy::Auto`] |
//! | `testing` | deterministic in-memory `Fake*` backends |
//! | `metal`/`cuda`/`vulkan` | GPU passthroughs, opt-in per machine — `--all-features` is not supported |
//!
//! # Process exit on macOS
//!
//! Drop every `Meme` handle (and backend `Arc`s) before the process
//! exits — llama.cpp's Metal static destructor asserts while contexts
//! are alive, so a signal handler should exit through `_exit` (e.g.
//! `signal_hook::low_level::exit`), not `std::process::exit`, whose
//! libc `exit` runs that destructor and aborts.
//!
//! # API stability
//!
//! 0.x. The modules `js`, `paths`, `doctor`, `trust`, `update_hook`,
//! `vpath` and `docid` exist mainly to serve `meme-bin`/`meme-mcp` and
//! may change in minor releases.
//!
//! # Upstream differences
//!
//! Behavior deliberately differs from upstream in a few places; see the
//! [Notable upstream differences](https://github.com/qntx/meme/blob/main/docs/upstream-differences.mdx)
//! section of the README and `CHANGELOG.md`.
//!
//! ```
//! # fn main() -> meme::Result<()> {
//! let db = std::env::temp_dir().join("meme-example.sqlite");
//! let meme = meme::Meme::builder(&db).build()?;
//! meme.add_collection(
//!     "notes",
//!     &meme::CollectionSpec {
//!         path: "/tmp/notes".to_string(),
//!         ..meme::CollectionSpec::default()
//!     },
//! )?;
//! assert_eq!(meme.collections()?.len(), 1);
//! meme.close()?;
//! # std::fs::remove_file(&db).ok();
//! # std::fs::remove_file(db.with_extension("sqlite-wal")).ok();
//! # std::fs::remove_file(db.with_extension("sqlite-shm")).ok();
//! # Ok(())
//! # }
//! ```

#![allow(
    unused_crate_dependencies,
    reason = "tempfile is a dev-dependency used only by tests/"
)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod chunk;
pub mod collection;
pub mod config;
pub mod docid;
pub mod doctor;
pub mod document;
mod env;
mod error;
pub mod js;
pub mod llm;
pub mod maintenance;
mod meme;
pub mod metadata;
pub mod paths;
pub mod query;
pub mod snippet;
mod store;
#[cfg(feature = "testing")]
#[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
pub mod testing;
pub mod trust;
pub mod update_hook;
pub mod vpath;

pub use chunk::{
    BREAK_PATTERNS, BreakPattern, BreakPoint, CHUNK_OVERLAP_CHARS, CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_CHARS, CHUNK_SIZE_TOKENS, CHUNK_WINDOW_CHARS, CHUNK_WINDOW_TOKENS, CharChunkOptions,
    Chunk, ChunkStrategy, CodeFenceRegion, TokenChunk, TokenChunkOptions, chunk_document,
    chunk_document_by_tokens, chunk_document_with_break_points, chunk_document_with_strategy,
    find_best_cutoff, find_code_fences, is_inside_code_fence, merge_break_points,
    scan_break_points, utf16_len, utf16_to_byte_offset,
};
pub use collection::{CollectionUpdateReport, SkippedFile, split_glob_mask};
pub use config::{Collection, Config, EmbedPooling, ModelsConfig};
pub use docid::get_docid;
pub use doctor::{
    FingerprintGroup, LegacyFingerprintAdoption, VectorSampleReport, active_doc_count,
    check_embedding_vector_samples, current_embed_fingerprint, fingerprint_groups,
    maybe_adopt_legacy_embedding_fingerprint, pending_embeddings_for, resolved_embed_model,
    short_hash_seq, sqlite_version, vec_version, vectors_table_exists,
};
pub use document::{
    DEFAULT_MULTI_GET_MAX_BYTES, Document, DocumentEntry, GetOptions, LineRange, MultiGet,
    MultiGetEntry, MultiGetFile, MultiGetOptions, MultiGetSelection,
};
pub use env::Environment;
pub use error::{ConfigError, DbError, Error, Result};
pub use llm::{CancelToken, Capability, Embedder, Generator, InferenceError, Reranker};
pub use maintenance::{CleanupCounts, Maintenance};
pub use meme::{
    CollectionInfo, CollectionPath, CollectionRemoval, CollectionSettings, CollectionSpec,
    ContextEntry, EmbedFailure, EmbedOptions, EmbedProgress, EmbedReport, IndexHealth, LexOptions,
    Meme, MemeBuilder, SearchInput, SearchOptions, Status, StatusCollection, UpdateOptions,
    UpdateProgress, UpdateReport, UpdateTailCounts, VectorOptions, VectorQueryOptions,
};
pub use metadata::filter::{
    CompiledMetadataFilter, METADATA_FILTER_LIMITS, MetadataFilter, MetadataFilterError,
    MetadataFilterLimits, compile_metadata_filter, parse_metadata_filter,
};
pub use metadata::{
    DocumentMetadata, METADATA_EXTRACTION_VERSION, METADATA_LIMITS, MetadataExtraction,
    MetadataLimits, MetadataScalar, MetadataValue, extract_document_metadata,
};
pub use query::{
    ExpandedQuery, ParsedQueryDocument, QueryKind, extract_intent_terms, parse_query_document,
};
pub use snippet::{ChunkSpan, SnippetResult, add_line_numbers, extract_snippet};
pub use store::documents::extract_title;
pub use store::fts::{
    build_fts5_query, sanitize_fts5_term, validate_lex_query, validate_semantic_query,
};
pub use store::hybrid::{Explain, HybridResult, RrfTrace, SearchHooks, VectorQueryResult};
pub use store::rrf::RrfContribution;
pub use store::search::{SearchResult, SearchSource};
#[cfg(feature = "testing")]
pub use testing::{FakeEmbedder, FakeGenerator, FakeReranker};
pub use vpath::{
    LineSuffix, VirtualPath, build_virtual_path, encode_meme_path, is_virtual_path,
    normalize_virtual_path, parse_line_suffix, parse_virtual_path,
};
