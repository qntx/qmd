# justfile for Rust project using Cargo

default: all

# The supported feature set for local dev and CI: llama.cpp inference
# (CPU-only on Linux; Metal is always built on Apple) plus the deterministic
# test backends. GPU passthroughs (metal/cuda/vulkan) are opt-in — add them
# here per machine, e.g. `just FEATURES="--features llama,testing,metal" all`.
# `--all-features` is not a supported configuration (see crates/qmd/Cargo.toml).
FEATURES := "--features llama,testing"

all: fmt clippy-fix deny test doc-check lint-docs

# Docs lint mirrors the shared ci-docs workflow (markdownlint-cli2).
lint-docs:
    npx --yes markdownlint-cli2 "docs/*.mdx"

list:
    @just --list

build:
    cargo build --workspace --release {{FEATURES}}

check:
    cargo check --workspace {{FEATURES}}

update:
    cargo update

test:
    cargo test --workspace {{FEATURES}}

bench:
    cargo bench {{FEATURES}}

# Prerequisites: `rustup toolchain install nightly --component clippy`
clippy:
    cargo +nightly clippy --workspace \
        --all-targets \
        {{FEATURES}} \
        -- -D warnings

# Prerequisites: `rustup toolchain install nightly --component clippy`
clippy-fix:
    cargo +nightly clippy --workspace \
        --fix \
        --all-targets \
        {{FEATURES}} \
        --allow-dirty \
        --allow-staged \
        -- -D warnings

# Nightly rustfmt provides import grouping.
# Prerequisites: `rustup toolchain install nightly --component rustfmt`
fmt:
    cargo +nightly fmt --all -- \
        --config unstable_features=true,group_imports=StdExternalCrate,imports_granularity=Module

fmt-check:
    cargo +nightly fmt --all -- \
        --check \
        --config unstable_features=true,group_imports=StdExternalCrate,imports_granularity=Module

# Prerequisites: `rustup toolchain install nightly`
doc:
    cargo +nightly doc --workspace {{FEATURES}} --no-deps --open

doc-check:
    RUSTDOCFLAGS="-D warnings" cargo +nightly doc --workspace {{FEATURES}} --no-deps

# Prerequisites: `cargo install cargo-deny`
deny:
    cargo deny check

clean:
    cargo clean
