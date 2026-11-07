# meme-mcp

MCP server for meme — library crate backing the `meme mcp` subcommand of `meme-bin`.

## Embedding

stdio: `serve_stdio(meme, StdioOptions::default())` serves process stdin/stdout
(`serve` accepts injected streams for tests). HTTP: `HttpServer::bind(meme, options)` (`HttpOptions::default()`
with `options.port = 0` for an OS-assigned port) binds and returns the
actual port via `server.port()`; `server.serve(shutdown_future)` runs until
the future resolves — the CLI's `serve_http` is `bind` + a
SIGINT/SIGTERM shutdown future. See the crate-level docs for a full example.

## License

Licensed under either of MIT or Apache-2.0, at your option.

Portions are derived from [tobi/qmd](https://github.com/tobi/qmd), Copyright (c) 2024-2026 Tobi Lutke, under the MIT License — see `LICENSE-UPSTREAM`.
