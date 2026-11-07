# Contributing to qmd

This document covers what you need before opening a pull request.

## Code of Conduct

Be kind, technical, and concise. No harassment, no surprises, no
backchannel exploits. Report vulnerabilities through GitHub security
advisories, not public issues. Features and bugs go through GitHub
issues and pull requests.

## Development Setup

```bash
git clone https://github.com/qntx/meme
cd meme

# Build the entire workspace with the supported feature set
# (llama.cpp inference + deterministic test backends).
cargo build --workspace --features llama,testing

# Run unit + integration + doc tests across the workspace.
cargo test --workspace --features llama,testing
```

`--all-features` is intentionally unsupported: the `metal`/`cuda`/`vulkan`
passthroughs need their platform toolkits. The supported set is
`llama,testing` (see `FEATURES` in the `Justfile`); GPU backends are
opt-in per machine, e.g. `--features llama,testing,metal`.

The MSRV is **Rust 1.95** (`workspace.package.rust-version` in `Cargo.toml`).
CI builds on stable; match that locally with `rustup`.

CI clippy is **stable** `-D warnings` via `qntx/workflows`
`ci-rust.yml@v2` (GitHub currently rustc 1.98). Do not add
`-A unknown-lints` to CI. Justfile `fmt` / `clippy` / `doc` stay
`cargo +nightly` and are **local-only**. Install nightly once
(`rustup toolchain install nightly`) if you run those recipes.

### Quality Gates (run before pushing)

```bash
just all
```

CI-equivalent cargo invocations:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --features llama,testing -- -D warnings
cargo test --workspace --features llama,testing
# No-llama configuration (the ci-minimal job; needs no cmake/clang):
cargo check -p qmd --no-default-features --features testing
cargo test -p qmd --no-default-features --features testing
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --features llama,testing
cargo publish --workspace --dry-run --locked
```

Local Justfile equivalents (`+nightly`, not CI):

```bash
cargo +nightly fmt --all --check
cargo +nightly clippy --workspace --all-targets --features llama,testing -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo +nightly doc --workspace --features llama,testing --no-deps
```

For dependency hygiene:

```bash
cargo install cargo-audit cargo-deny
cargo audit
cargo deny check all
```

## TOML style

TOML is formatted by taplo (`.taplo.toml`, aligned `=` entries) — the
`taplo` CI job and `just toml` both run:

```bash
npx --yes @taplo/cli@0.7.0 fmt --check
```

Inside one table, a contiguous run of `key = value` lines (no blank
line) is a group; taplo pads keys so every `=` in that group shares a
column. Do not reorder keys to make padding easier.

## Pull Request Workflow

1. **Open an issue** if your change is non-trivial. Architecture-level
   changes benefit from a design discussion before coding.
2. **Branch from `main`** — keep PRs focused on one logical concern.
3. **Add tests** alongside any behaviour change.
4. **Update the changelog** (`CHANGELOG.md`, Keep-a-Changelog format) if
   you make a user-visible or breaking API change.
5. **Run the quality gates** above. CI will reject `clippy` or `fmt`
   violations.
6. **Open the PR** with a clear title and description; reference any
   related issues.

## Commit Message Convention

Follow [Conventional Commits][cc]. Subject lines stay under 50
characters and use the imperative mood:

```text
feat(cli): add `qmd mcp` subcommand
fix(search): return BM25 results in descending score order
docs(readme): document hybrid search usage
```

[cc]: https://www.conventionalcommits.org/en/v1.0.0/

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`,
`test`, `chore`, `ci`, `build`, `revert`. Multi-paragraph bodies are
welcome for non-trivial changes; explain *why*, not just *what*.

## Reviewing Changes

If you are reviewing a PR, check for:

- Behaviour parity with upstream [tobi/qmd](https://github.com/tobi/qmd)
  v2.8.3, or the notes call out the divergence.
- Security implications (filesystem access, subprocesses, network
  calls).
- Breaking changes appearing in `CHANGELOG.md`.
- New `unsafe`, `unwrap`, or `expect` usages — they need a comment with
  the safety / panic argument.

## Reporting Security Issues

Do **not** open a public issue. Use GitHub security advisories for
private disclosure.

## Licence

By contributing you agree your work is dual-licensed under the MIT and
Apache-2.0 licences distributed with this repository.
