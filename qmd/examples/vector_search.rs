//! Vector semantic search using embeddings - fully self-contained.
//!
//! Run: `cargo run --example vector_search`

use anyhow::Result;
use qmd::{EmbeddingEngine, Store, llm::DEFAULT_EMBED_MODEL_URI, pull_model};

const SAMPLE_DOCS: &[(&str, &str)] = &[
    ("rust-basics.md", include_str!("data/rust-basics.md")),
    ("error-handling.md", include_str!("data/error-handling.md")),
    ("async-await.md", include_str!("data/async-await.md")),
    ("ownership.md", include_str!("data/ownership.md")),
];

fn main() -> Result<()> {
    let db_path = std::env::temp_dir().join("qmd_vector.db");
    let _ = std::fs::remove_file(&db_path);
    let store = Store::open(&db_path)?;

    // Index documents
    let now = chrono::Utc::now().to_rfc3339();
    for (name, content) in SAMPLE_DOCS {
        let hash = Store::hash_content(content);
        let title = Store::extract_title(content);
        store.insert_content(&hash, content, &now)?;
        store.insert_document("samples", name, &title, &hash, &now, &now)?;
    }

    // Load embedding model
    println!("Loading embedding model...");
    let model = pull_model(DEFAULT_EMBED_MODEL_URI, false)?;
    let mut engine = EmbeddingEngine::new(&model.path)?;

    // Generate embeddings
    println!("Generating embeddings...");
    store.ensure_vector_table(768)?;
    for (name, content) in SAMPLE_DOCS {
        let hash = Store::hash_content(content);
        let emb = engine.embed_document(content, Some(name))?;
        store.insert_embedding(&hash, 0, 0, &emb.embedding, &emb.model, &now)?;
    }

    // Vector search
    let query = "how to handle errors";
    let query_emb = engine.embed_query(query)?;
    let results = store.search_vec(&query_emb.embedding, 5, None)?;

    println!("\nVector search: '{}'\n", query);
    for r in &results {
        println!("[{:.4}] {}", r.score, r.doc.path);
    }

    // Compare with FTS
    println!("\nFTS comparison:");
    for r in store.search_fts(query, 3, None)? {
        println!("[{:.2}] {}", r.score, r.doc.path);
    }

    let _ = std::fs::remove_file(&db_path);
    Ok(())
}
