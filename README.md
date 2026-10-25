<!-- markdownlint-disable MD033 MD041 -->

# qmd

Rust port of [tobi/qmd](https://github.com/tobi/qmd) (v2.8.3) — search
local markdown knowledge bases with BM25, vector similarity and hybrid
query expansion, from a single binary.

> **Status:** rewrite in progress. Indexing, retrieval, collections,
> contexts, models, diagnostics, the skills commands and the stdio MCP
> server are implemented; `qmd mcp --http`/`--daemon`/`stop` and
> `qmd bench` are not yet ported.

See [docs/](docs/) for the full reference.

## CLI

```text
qmd collection add <path> [--name <n>] [--mask '**/*.md']  # register a folder
qmd init                    # create a project-local .qmd index
qmd update                  # re-index (runs update hooks first)
qmd search <query>          # BM25 keyword search — no model needed
qmd embed                   # generate embeddings (llama.cpp, local)
qmd vsearch <query>         # vector similarity only
qmd query <query>           # expansion + hybrid RRF + reranking
qmd get <file|#docid>       # fetch a document (line-numbered)
qmd multi-get <pattern>     # batch fetch by glob or comma list
qmd status / doctor         # index health and diagnostics
qmd pull                    # download configured models
qmd skills list|get|path    # inspect bundled skills
qmd skill show|install      # print or install the QMD skill
qmd mcp                     # MCP server over stdio for agents/IDEs
```

`query` also accepts a structured multi-line document of
`lex:`/`vec:`/`hyde:`/`intent:`/`expand:` lines (see
[docs/query-syntax](docs/query-syntax.mdx)). Shared options:
`-c/--collection`, `-n`, `--all`, `--min-score`,
`--format cli|json|csv|md|xml|files`, `--filter '<json>'`, `--full`,
`--full-path`, `--intent`; `query` also takes `--explain`, `--no-rerank`
and `-C/--candidate-limit`.

The index lives at `~/.cache/qmd/<index>.sqlite` (global) or
`.qmd/index-rs.sqlite` next to a project-local `.qmd/index.y{a,}ml`.
`--index <name>` switches indexes; `INDEX_PATH` overrides the db path.

## Notable upstream differences

Deliberate deviations from `tobi/qmd` — each is also registered in the
parity matrix in `qmd-study`:

- **Single binary, no package dir.** The `skills/` tree is embedded via
  `include_str!` (a byte-verbatim copy of upstream `skills/qmd`) and
  materialized under `<cache>/qmd/skills/<version>` when `skills path`
  or `skill install` needs a real path. `QMD_SKILLS_DIR` still overrides
  discovery wholesale. Upstream's `skills/release` maintainer skill is
  not ported, so `skills list`/`get --all` show `qmd` only.
- **Project-local index naming.** Local databases are
  `.qmd/index-rs.sqlite` so a Rust and an upstream index can coexist in
  the same project without clobbering `index.sqlite`.
- **Embed locking.** `qmd embed` takes an OS `try_lock` on
  `<db>.embed.lock` instead of upstream's directory-level PID file;
  a busy lock prints `Another embed process is already running.
  Skipping.` and exits 0.
- **`AST: not available`** in `qmd status`/doctor until the AST chunking
  pass lands (`--chunk-strategy auto` falls back to regex).
- **Unnoted crash paths** (e.g. `skill show`/`--skill` with no skill,
  invalid `--chunk-strategy`) print the upstream message plus the doctor
  hint instead of a Bun stack dump.
- **`qmd mcp`** serves the upstream tools and resource byte-for-byte,
  but `qmd mcp <anything>` exits 1 instead of silently starting a
  server, and a checked-in `.qmd` config's models go through the trust
  gate like every other command. `--http`/`--daemon`/`stop` and
  `qmd bench` are not yet ported; `--help` omits them.
- The embedded `SKILL.md` still mentions `npm install -g @tobilu/qmd`
  (kept verbatim for parity); install this port via cargo or a release
  binary instead.

## Development

```text
just all          # fmt + clippy + deny + tests + rustdoc + docs lint
cargo test --workspace --features llama,testing
```

`llama` (llama.cpp inference) and `testing` (deterministic backends) are
the supported local feature set; `metal`/`cuda`/`vulkan` are opt-in
passthroughs. `scripts/gen_cli_golden.ts` regenerates
`crates/qmd-cli/tests/fixtures/cli_golden.json` — byte-for-byte CLI
output recorded from upstream — with
`QMD_UPSTREAM=/path/to/tobi/qmd bun scripts/gen_cli_golden.ts`;
`scripts/gen_mcp_golden.ts` does the same for the MCP stdio sessions
in `mcp_golden.json`.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project shall be dual-licensed as above, without any additional terms or conditions.

## Acknowledgements

This project is a Rust port of [tobi/qmd](https://github.com/tobi/qmd) by Tobi Lutke, used under the MIT License (Copyright (c) 2024-2026 Tobi Lutke).

---

<div align="center">

A **[QuantX](https://qntx.org)** open-source project.

<a href="https://qntx.org"><img alt="QuantX" width="369" src="https://raw.githubusercontent.com/qntx/.github/main/profile/qntx.svg" /></a>

Code is law. We write both.

</div>
