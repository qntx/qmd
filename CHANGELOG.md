# Changelog

All notable changes to the `qmd` workspace are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- P1 acceptance fixtures: `eval-docs` BM25 golden replaying upstream `eval-bm25.test.ts` (24 queries, score tolerance <1e-9) via `scripts/gen_eval_golden.ts`; opt-in YAML interop check (`config_interop_upstream_round_trip` + `scripts/interop_yaml.ts`, requires `QMD_UPSTREAM` + bun).
- `qmd` crate retrieval, index status and maintenance (P1.4): `Qmd::get`/`document_body`/`multi_get`/`list_documents` with docid (`#abc123`), `qmd://` virtual paths, `:from:count` line suffixes, `DocumentNotFound`/`DocumentExcludedByIgnore` errors and Levenshtein "did you mean" suggestions; `Qmd::status`/`index_health`/`maintenance` with `MaintenancePreview`/`MaintenanceReport`; virtual-path and docid helpers (`vpath`/`docid` modules); embedding-fingerprint model resolution with `models` config and `QMD_*_MODEL` env precedence; `vector_schema_version` "2" migration adding `embed_fingerprint`/`total_chunks`; `Document`/`ListEntry`/`StatusReport`/`IndexHealth` types; generated golden fixtures for virtual-path and docid helpers (`scripts/gen_vpath_golden.ts`).
- `qmd` crate BM25 full-text search (P1.3): `Qmd::search_lex` with `LexOptions`; FTS5 query builder (`build_fts5_query`, phrase/negation/compound/CJK terms); `validate_lex_query`/`validate_semantic_query`; `SearchResult`/`SearchSource`/`get_docid`; `extract_snippet`/`add_line_numbers`.
- `qmd` crate collection scanning and incremental indexing (P1.2): `Qmd::update` with `UpdateOptions`/`UpdateProgress`/`UpdateReport`; `split_glob_mask`; mask/ignore/hidden/excluded-dir filtering; file-symlink boundary checks; sha256 content-addressed storage; write-side FTS5 maintenance with CJK normalization; `extract_title`.
- `qmd` crate storage and configuration foundation (P1.1): `Qmd` handle and builder; file / inline / DB-only config sources; YAML config model preserving unknown keys, key order and `editor_uri` aliases; injectable `Environment`; path helpers (`<index>-rs.sqlite`, `.qmd/` discovery); SQLite store with busy timeout, WAL, three-layer versioning, `application_id` foreign-index guard and sqlite-vec; collection and context management with CLI semantics.

### Fixed

- Local config discovery now checks `.qmd/index.yaml` (preferred) and `.qmd/index.yml`, matching upstream — P1.1 mistakenly looked for `qmd.yml`.

### Changed

- Complete rewrite: this workspace is being rebuilt from scratch as a Rust port of [tobi/qmd](https://github.com/tobi/qmd) v2.8.3. The previous implementation is archived at tag `archive/v0.5.0-abandoned`.
