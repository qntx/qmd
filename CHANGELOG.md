# Changelog

All notable changes to the `qmd` workspace are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `qmd` crate inference abstraction and smart chunking (P2.1): `Embedder`/`Reranker`/`Generator` traits with `Capability`, `InferenceError`, `CancelToken` and `QmdBuilder::embedder`/`reranker`/`generator` injection (`Error::NoBackend` when unset); `chunk` module porting upstream smart chunking — `scan_break_points`, `find_code_fences`/`is_inside_code_fence`, `find_best_cutoff`, `merge_break_points`, `chunk_document`/`chunk_document_with_break_points`/`chunk_document_with_strategy`, `chunk_document_by_tokens` (via `Embedder::tokenize`/`detokenize`), surrogate-pair-safe boundaries, UTF-16 `pos` reporting plus `utf16_len`/`utf16_to_byte_offset` helpers, and `ChunkStrategy::{Regex,Auto}` (Auto degrades to Regex until the AST pass lands); `testing` feature with deterministic `FakeEmbedder`/`FakeReranker`/`FakeGenerator`; `Error::Cancelled`/`Model`/`Inference`/`EmbeddingDimensionMismatch`; `embedding_fingerprint` accepts `fingerprint_extra`, and an injected `Embedder`'s `model_id()` is authoritative for vector identity (config/env resolution only when uninjected); chunk golden fixture (`scripts/gen_chunk_golden.ts`) asserting `text`/`pos` parity with upstream `store.ts` across 14 documents × 3 parameter sets, including JS `\s`/heading-lookahead edge cases.
- P1 acceptance fixtures: `eval-docs` BM25 golden replaying upstream `eval-bm25.test.ts` (24 queries, score tolerance <1e-9) via `scripts/gen_eval_golden.ts`; opt-in YAML interop check (`config_interop_upstream_round_trip` + `scripts/interop_yaml.ts`, requires `QMD_UPSTREAM` + bun).
- `qmd` crate retrieval, index status and maintenance (P1.4): `Qmd::get`/`document_body`/`multi_get`/`list_documents` with docid (`#abc123`), `qmd://` virtual paths, `:from:count` line suffixes, `DocumentNotFound`/`DocumentExcludedByIgnore` errors and Levenshtein "did you mean" suggestions; `Qmd::status`/`index_health`/`maintenance` with `MaintenancePreview`/`MaintenanceReport`; virtual-path and docid helpers (`vpath`/`docid` modules); embedding-fingerprint model resolution with `models` config and `QMD_*_MODEL` env precedence; `vector_schema_version` "2" migration adding `embed_fingerprint`/`total_chunks`; `Document`/`ListEntry`/`StatusReport`/`IndexHealth` types; generated golden fixtures for virtual-path and docid helpers (`scripts/gen_vpath_golden.ts`).
- `qmd` crate BM25 full-text search (P1.3): `Qmd::search_lex` with `LexOptions`; FTS5 query builder (`build_fts5_query`, phrase/negation/compound/CJK terms); `validate_lex_query`/`validate_semantic_query`; `SearchResult`/`SearchSource`/`get_docid`; `extract_snippet`/`add_line_numbers`.
- `qmd` crate collection scanning and incremental indexing (P1.2): `Qmd::update` with `UpdateOptions`/`UpdateProgress`/`UpdateReport`; `split_glob_mask`; mask/ignore/hidden/excluded-dir filtering; file-symlink boundary checks; sha256 content-addressed storage; write-side FTS5 maintenance with CJK normalization; `extract_title`.
- `qmd` crate storage and configuration foundation (P1.1): `Qmd` handle and builder; file / inline / DB-only config sources; YAML config model preserving unknown keys, key order and `editor_uri` aliases; injectable `Environment`; path helpers (`<index>-rs.sqlite`, `.qmd/` discovery); SQLite store with busy timeout, WAL, three-layer versioning, `application_id` foreign-index guard and sqlite-vec; collection and context management with CLI semantics.

### Fixed

- Local config discovery now checks `.qmd/index.yaml` (preferred) and `.qmd/index.yml`, matching upstream — P1.1 mistakenly looked for `qmd.yml`.

### Changed

- Complete rewrite: this workspace is being rebuilt from scratch as a Rust port of [tobi/qmd](https://github.com/tobi/qmd) v2.8.3. The previous implementation is archived at tag `archive/v0.5.0-abandoned`.
