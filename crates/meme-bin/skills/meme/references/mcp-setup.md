# MEME MCP Server Setup

## Install

```bash
cargo install meme-bin
meme collection add ~/path/to/markdown --name myknowledge
meme embed
```

## Configure MCP Client

**Claude Code** (`~/.claude/settings.json`):
```json
{
  "mcpServers": {
    "meme": { "command": "meme", "args": ["mcp"] }
  }
}
```

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "meme": { "command": "meme", "args": ["mcp"] }
  }
}
```

**OpenClaw** (`~/.openclaw/openclaw.json`):
```json
{
  "mcp": {
    "servers": {
      "meme": { "command": "meme", "args": ["mcp"] }
    }
  }
}
```

## Protocol

MEME speaks MCP **2026-07-28** (SDK 2.x) and dual-speaks 2025-era stdio clients.

- **stdio** (`meme mcp`): hosts still launch this as a subprocess. Opening
  `initialize` (2025) or a `_meta`-enveloped request / `server/discover`
  (2026) pins the connection's era.
- **HTTP** (`meme mcp --http`): sessionless Streamable HTTP. No
  `Mcp-Session-Id`, no handshake. Each POST is independent. 2026 clients
  MUST send `MCP-Protocol-Version`, `Mcp-Method`, and (for `tools/call`)
  `Mcp-Name`. Version/caps travel in `_meta`. Call `server/discover` to
  learn supported versions and capabilities. `tools/list` is cacheable
  (`ttlMs` / `cacheScope`) and returns tools in a stable order.

## HTTP Mode

```bash
meme mcp --http              # Port 8181
meme mcp --http --daemon     # Background
meme mcp stop                # Stop daemon
```

`POST /mcp` is the MCP endpoint (JSON). `GET /health` is a liveness check.
There is no session GET stream and no idle-session TTL.

## Tools

### query

Search with pre-expanded queries.

```json
{
  "searches": [
    { "type": "lex", "query": "keyword phrases" },
    { "type": "vec", "query": "natural language question" },
    { "type": "hyde", "query": "hypothetical answer passage..." }
  ],
  "limit": 10,
  "collection": "optional",
  "minScore": 0.0
}
```

| Type | Method | Input |
|------|--------|-------|
| `lex` | BM25 | Keywords (2-5 terms) |
| `vec` | Vector | Question |
| `hyde` | Vector | Answer passage (50-100 words) |

### get

Retrieve document by path or `#docid`.

| Param | Type | Description |
|-------|------|-------------|
| `path` | string | File path or `#docid` |
| `full` | bool? | Return full content |
| `lineNumbers` | bool? | Add line numbers |

### multi_get

Retrieve multiple documents.

| Param | Type | Description |
|-------|------|-------------|
| `pattern` | string | Glob or comma-separated list |
| `maxBytes` | number? | Skip large files (default 64KB) |

### status

Index health and collections. No params.

## Troubleshooting

- **Not starting**: `which meme`, `meme mcp` manually
- **No results**: `meme collection list`, `meme embed`
- **Slow first search**: Normal, models loading (~3GB)
