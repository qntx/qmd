# qmd-mcp

MCP server for qmd — library crate backing the `qmd mcp` subcommand of `qmd-cli`.

## Embedding

stdio: `serve_stdio(qmd, StdioOptions::default())` serves process stdin/stdout
(`serve` accepts injected streams for tests). HTTP: `HttpServer::bind(qmd, options)` (`HttpOptions::default()`
with `options.port = 0` for an OS-assigned port) binds and returns the
actual port via `server.port()`; `server.serve(shutdown_future)` runs until
the future resolves — the CLI's `serve_http` is `bind` + a
SIGINT/SIGTERM shutdown future. See the crate-level docs for a full example.

## License

Licensed under either of MIT or Apache-2.0, at your option.
