<!-- markdownlint-disable MD033 MD041 -->

# qmd

Rust port of [tobi/qmd](https://github.com/tobi/qmd) (v2.8.3) — search
local markdown knowledge bases with BM25, vector similarity and hybrid
query expansion, from a single binary.

> **Status:** port complete — indexing, retrieval, collections,
> contexts, models, diagnostics, the skills commands, `qmd bench` and
> the MCP server (stdio, HTTP, daemon lifecycle) are implemented.
> v0.6.0 is the first release of the Rust port (supersedes the
> pre-rewrite 0.3.2 on crates.io).

See [docs/](docs/) for the full reference.

## Install

Prebuilt binaries are attached to each
[GitHub Release](https://github.com/qntx/qmd/releases):

| Target | Notes |
| --- | --- |
| `x86_64-unknown-linux-gnu` | glibc ≥ 2.39 (Ubuntu 24.04+, Debian 13+); x86 CPU needs AVX2 (Haswell 2013 or newer) |
| `aarch64-unknown-linux-gnu` | same glibc ≥ 2.39 baseline |
| `aarch64-apple-darwin` | Apple Silicon; Metal acceleration built in |
| `x86_64-pc-windows-msvc` | Windows x64 — second-tier support: builds and ships, release-smoke verified, but the test suite is Unix-oriented and `qmd mcp stop` cannot kill a live daemon (same as upstream; use `taskkill`). ARM64 Windows is not built — MSVC cannot compile llama.cpp's ARM kernels |

Older Linux systems (glibc < 2.39, pre-AVX2 CPUs) can still build from
source:

```text
cargo install qmd-cli          # crates.io
# or from a checkout:
cargo install --path crates/qmd-cli
```

The default `llama` feature compiles llama.cpp with cmake and runs
bindgen, so the build needs `cmake`, a C/C++ toolchain and `clang`/
`libclang` (on Windows: MSVC Build Tools, CMake and LLVM). GPU
backends are opt-in passthrough features — Metal is built on Apple
targets regardless; add `--features cuda` or `--features vulkan` for
the matching toolchain (CUDA verified on real hardware — RTX 5060 Ti,
Windows; Vulkan build-verified only).

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
qmd bench <fixture.json>    # search-quality benchmark (P/R, MRR, F1)
qmd skills list|get|path    # inspect bundled skills
qmd skill show|install      # print or install the QMD skill
qmd mcp                     # MCP server over stdio for agents/IDEs
qmd mcp --http [--port N] [--host H]   # or HTTP (default :8181, localhost)
qmd mcp --http --daemon     # detached daemon (pidfile + log under cache dir)
qmd mcp stop                # stop the daemon
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
- **AST grammars are statically linked.** `qmd status` always reports
  `AST Chunking: active` with the full language list — upstream's
  grammar-download failure paths (`Unavailable: <lang>` rows, the
  `unavailable` fallback state) do not exist here.
- **Unnoted crash paths** (e.g. `skill show`/`--skill` with no skill,
  invalid `--chunk-strategy`) print the upstream message plus the doctor
  hint instead of a Bun stack dump.
- **`qmd mcp`** serves the upstream tools and resource byte-for-byte
  over stdio and HTTP (`--http`, `--daemon`, `stop`), but
  `qmd mcp <anything>` other than `stop` exits 1 instead of silently
  starting a server, and a checked-in `.qmd` config's models go through
  the trust gate like every other command.
- The embedded `SKILL.md` still mentions `npm install -g @tobilu/qmd`
  (kept verbatim for parity); install this port via cargo or a release
  binary instead.

## Models

`qmd` fetches GGUF models on demand (`qmd pull`, or lazily the first
time a search path needs them); nothing is bundled into the binary.
The upstream defaults are:

| Role | Model (GGUF) | License |
| --- | --- | --- |
| Embedding | [embeddinggemma-300M](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) (Q8_0, Google's EmbeddingGemma) | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| Reranking | [Qwen3-Reranker-0.6B](https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF) (Q8_0) | Apache-2.0 |
| Query expansion | [qmd-query-expansion-1.7B](https://huggingface.co/tobil/qmd-query-expansion-1.7B-gguf) (Qwen3-1.7B fine-tune) | MIT |

Set `HF_ENDPOINT=https://hf-mirror.com` to download through the
mirror. A project's `.qmd/index.y{a,}ml` may declare different
`models:` entries — they only take effect in trusted directories, and
their licenses are the user's responsibility (this project only
fetches the files you configure).

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
