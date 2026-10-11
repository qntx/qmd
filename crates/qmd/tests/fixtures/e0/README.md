# E0 reference data — upstream llm.ts measurements

Captured from upstream `tobi/qmd` (v2.8.3, node-llama-cpp backend) on
Apple Silicon, for the AC4 inference-parity checks in
`crates/qmd/src/llm/llamacpp.rs` (`#[ignore]` tests — run with
`cargo test --features llama -- --ignored`).

- `embed.json` — 21 inputs: `id` (`q:` = query text via
  `formatQueryForEmbedding`, `d:` = eval-doc via `formatDocForEmbedding`,
  CJK/emoji/long-doc probes), `n_tokens`, `token_ids` (content tokens, no
  specials), `vector` (768-dim embeddinggemma-300M output).
- `rerank.json` — 10 queries × 6 eval-docs: per-doc `token_ids` (the
  fully-rendered `tokenizer.chat_template.rerank` eval input) and
  `score`; plus `sample_eval_input` with the rendered template text.
- `generate.json` — `rendered_prompt` (QwenChatWrapper output, verified
  byte-identical to `apply_chat_template` in P2.2 verification ③) and 50
  sampled expansion outputs.
- `prompt_tokens.json` — 10 queries × tokenized rendered prompt.

Input texts are reconstructible by detokenizing `token_ids`; document
inputs are the files in `../eval-docs/`. Models come from
`~/.cache/qmd/models` (upstream-compatible `hf_*` names); pull them via
`llm::download::pull_models` or upstream `qmd embed`.
