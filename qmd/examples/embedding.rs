//! Embedding generation and similarity computation - fully self-contained.
//!
//! Run: `cargo run --example embedding`

use anyhow::Result;
use qmd::{
    CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, EmbeddingEngine, chunk_document_by_tokens,
    cosine_similarity, llm::DEFAULT_EMBED_MODEL_URI, pull_model,
};

const DOC_ERROR: &str = include_str!("data/error-handling.md");
const DOC_ASYNC: &str = include_str!("data/async-await.md");

fn main() -> Result<()> {
    println!("Loading embedding model...");
    let model = pull_model(DEFAULT_EMBED_MODEL_URI, false)?;
    let mut engine = EmbeddingEngine::new(&model.path)?;

    // Embed documents
    let emb1 = engine.embed_document(DOC_ERROR, Some("Error Handling"))?;
    let emb2 = engine.embed_document(DOC_ASYNC, Some("Async Await"))?;

    println!("Embedding dimensions: {}", emb1.embedding.len());
    println!(
        "Doc similarity: {:.4}\n",
        cosine_similarity(&emb1.embedding, &emb2.embedding)
    );

    // Query similarity
    println!("Query similarities to 'Error Handling' doc:");
    for q in [
        "Result and Option types",
        "async runtime",
        "python exceptions",
    ] {
        let q_emb = engine.embed_query(q)?;
        let sim = cosine_similarity(&emb1.embedding, &q_emb.embedding);
        println!("  '{}' -> {:.4}", q, sim);
    }

    // Document chunking
    println!("\nChunking:");
    let chunks =
        chunk_document_by_tokens(&engine, DOC_ERROR, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)?;
    println!(
        "  {} chunks (size={}, overlap={})",
        chunks.len(),
        CHUNK_SIZE_TOKENS,
        CHUNK_OVERLAP_TOKENS
    );

    Ok(())
}
