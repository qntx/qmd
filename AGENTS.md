# AGENTS.md

- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current requirements. Avoid speculative abstractions, configuration, and indirection.
- Grow the system in layers. Start from the smallest version that works end to end, and add each new capability on top of a product that already works. Never trade a working product for unfinished complexity.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own implementation or adding packages. Do not assume a library lacks a capability without checking its documentation and types.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.

## Comments and documentation

Following the conventions in `qntx/r402`:

- Every module gets a `//!` doc comment; every public item gets `///` docs (`missing_docs` is a workspace lint). Fallible functions get a `# Errors` section; constructors and pure getters get `#[must_use]`.
- Inline `//` comments are sparse and explain *why*, never *what* — one line per non-obvious decision. `#[allow(...)]` always carries a `reason = "..."`.
- Porting-specific rule: where behavior intentionally mirrors upstream `tobi/qmd`, the comment names the upstream file (and lines when useful), e.g. `mirrors upstream db.ts:84-95`. Where we deliberately deviate from upstream (bug fixes, dropped legacy paths), say so in the module or item doc and state the reason.
- `unsafe` is forbidden except inside `src/store/vec.rs` (sqlite-vec registration); every unsafe block needs a `// SAFETY:` comment.
- Test files carry the standard header allows (`unused_crate_dependencies`, test-code patterns) each with a `reason`, followed by a `//!` doc naming the upstream test files they port.
