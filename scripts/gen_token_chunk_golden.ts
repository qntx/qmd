#!/usr/bin/env bun
/**
 * Regenerate `crates/qmd/tests/fixtures/token_chunk_golden.json` by
 * running the upstream (tobi/qmd) token-based chunking pipeline —
 * `chunkDocumentByTokens` with the real embeddinggemma-300M tokenizer —
 * over the vendored corpus plus two deterministic synthetic documents.
 *
 * Usage:
 *   QMD_UPSTREAM=/path/to/tobi/qmd bun scripts/gen_token_chunk_golden.ts
 *
 * The first run downloads the embedding model to `~/.cache/qmd/models`
 * (upstream's shared cache, ~350MB).
 *
 * The fixture is self-contained: every doc's full input `content` is
 * recorded alongside the upstream chunk output `{text, pos, tokens}`,
 * so the Rust test needs neither the TS runtime nor the model.
 */

const upstream = process.env.QMD_UPSTREAM;
if (!upstream) {
  console.error("QMD_UPSTREAM=<path to tobi/qmd checkout> is required");
  process.exit(1);
}

const store = await import(`${upstream}/src/store.ts`);
const fs = await import("node:fs/promises");
const path = await import("node:path");
const { execSync } = await import("node:child_process");

const commit = execSync(`git -C "${upstream}" rev-parse HEAD`).toString().trim();
const fixturesDir = new URL("../crates/qmd/tests/fixtures/", import.meta.url).pathname;

const PARAMS = { max_tokens: 900, overlap_tokens: 135, window_tokens: 200 };

// ── Corpus ────────────────────────────────────────────────────────────
// Vendored edge-case corpus (small docs — mostly single-chunk, but they
// pin down the tokenizer-aware path for CJK/emoji/surrogates/fences).
const corpus: { name: string; content: string }[] = [];
for (const d of ["eval-docs", "chunk-docs"]) {
  for (const f of (await fs.readdir(path.join(fixturesDir, d))).sort()) {
    if (!f.endsWith(".md")) continue;
    corpus.push({
      name: `${d}/${f}`,
      content: await fs.readFile(path.join(fixturesDir, d, f), "utf8"),
    });
  }
}

// Deterministic synthetic docs that actually exercise the >max_tokens
// re-split path: the char-estimated first pass (3 chars/token) must
// produce chunks that tokenize over the 900-token cap.
const longSection = (n: number) =>
  Array.from({ length: n }, (_, i) =>
    `## Section ${i}\n\n` +
    `Paragraph ${i} discusses retrieval-augmented generation, vector ` +
    `similarity search, and hybrid ranking. BM25 tokenization splits on ` +
    `word boundaries; embedding models read byte-pair encoded tokens. ` +
    `The pipeline chunks documents, embeds each chunk, and stores the ` +
    `vectors alongside the relational index for later cosine search.\n\n` +
    `- bullet ${i}.a: deterministic filler text with unicode ünïcodé\n` +
    `- bullet ${i}.b: numbers ${i * 7}.${i * 13} and punctuation, dense!\n`,
  ).join("\n");

corpus.push({ name: "synthetic/long-prose.md", content: longSection(60) });

// Dense token/char ratio (code-ish, ~2 chars/token) forces the
// char-estimate chunks well past the 900-token cap → re-split path.
const denseCode = Array.from({ length: 40 }, (_, i) =>
  `\`\`\`rust\nfn handler_${i}(req: &Request) -> Result<Response, Error> {\n` +
  `    let payload = serde_json::from_str::<Payload>(&req.body)?;\n` +
  `    Ok(Response::json(&payload.map(|p| p.transform())))?;\n` +
  `}\n\`\`\`\n`,
).join("\n");
corpus.push({ name: "synthetic/dense-code.md", content: denseCode });

// ── Generate ──────────────────────────────────────────────────────────
console.error(`upstream ${commit}; ${corpus.length} docs`);
const docs = [];
for (const doc of corpus) {
  const chunks = await store.chunkDocumentByTokens(
    doc.content,
    PARAMS.max_tokens,
    PARAMS.overlap_tokens,
    PARAMS.window_tokens,
    doc.name,
    "regex",
  );
  docs.push({
    name: doc.name,
    content: doc.content,
    chunks: chunks.map((c: any) => ({ text: c.text, pos: c.pos, tokens: c.tokens })),
  });
  console.error(
    `  ${doc.name}: ${doc.content.length} chars → ${chunks.length} chunks ` +
      `(${chunks.map((c: any) => c.tokens).join(",")} tokens)`,
  );
}

const out = {
  upstream_commit: commit,
  upstream_version: "2.8.3",
  model: "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf",
  params: PARAMS,
  docs,
};
await fs.writeFile(
  path.join(fixturesDir, "token_chunk_golden.json"),
  JSON.stringify(out, null, 2) + "\n",
);
console.error("wrote token_chunk_golden.json");
