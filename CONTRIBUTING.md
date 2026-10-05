# Contributing to qmd

This document covers what you need before opening a pull request.

## Code of Conduct

Be kind, technical, and concise. No harassment, no surprises, no
backchannel exploits. Report vulnerabilities through GitHub security
advisories, not public issues. Features and bugs go through GitHub
issues and pull requests.

## Development Setup

```bash
git clone https://github.com/qntx/qmd
cd qmd

# Build the entire workspace with all features.
cargo build --workspace --all-features

# Run unit + integration + doc tests across the workspace.
cargo test --workspace --all-features
```

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
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

Local Justfile equivalents (`+nightly`, not CI):

```bash
cargo +nightly fmt --all --check
cargo +nightly clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo +nightly doc --workspace --all-features --no-deps
```

For dependency hygiene:

```bash
cargo install cargo-audit cargo-deny
cargo audit
cargo deny check all
```

## TOML style

Match kobe’s grouped `=` alignment. Inside one table, a contiguous run of
`key = value` lines (no blank line) is a group; pad keys so every `=` in
that group shares a column. One space on each side of `=`. Keep the
file’s existing trailing-comma style. Do not reorder keys to make
padding easier. There is no taplo/dprint gate; rustfmt remains the only
format check.

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
