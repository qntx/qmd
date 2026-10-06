# Changelog

All notable changes to the `qmd` workspace are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `qmd` crate storage and configuration foundation (P1.1): `Qmd` handle and builder; file / inline / DB-only config sources; YAML config model preserving unknown keys, key order and `editor_uri` aliases; injectable `Environment`; path helpers (`<index>-rs.sqlite`, `.qmd/` discovery); SQLite store with busy timeout, WAL, three-layer versioning, `application_id` foreign-index guard and sqlite-vec; collection and context management with CLI semantics.

### Changed

- Complete rewrite: this workspace is being rebuilt from scratch as a Rust port of [tobi/qmd](https://github.com/tobi/qmd) v2.8.3. The previous implementation is archived at tag `archive/v0.5.0-abandoned`.
