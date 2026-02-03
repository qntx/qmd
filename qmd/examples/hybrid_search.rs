//! Hybrid search with RRF (Reciprocal Rank Fusion) - fully self-contained.
//!
//! Run: `cargo run --example hybrid_search`

use anyhow::Result;
use qmd::{EmbeddingEngine, Store, hybrid_search_rrf, llm::DEFAULT_EMBED_MODEL_URI, pull_model};

const SAMPLE_DOCS: &[(&str, &str)] = &[
    ("rust-basics.md", include_str!("data/rust-basics.md")),
    ("error-handling.md", include_str!("data/error-handling.md")),
    ("async-await.md", include_str!("data/async-await.md")),
    ("ownership.md", include_str!("data/ownership.md")),
];

fn main() -> Result<()> {
    let db_path = std::env::temp_dir().join("qmd_hybrid.db");
    let _ = std::fs::remove_file(&db_path);
    let store = Store::open(&db_path)?;

    let now = chrono::Utc::now().to_rfc3339();
    for (name, content) in SAMPLE_DOCS {
        let hash = Store::hash_content(content);
        let title = Store::extract_title(content);
        store.insert_content(&hash, content, &now)?;
        store.insert_document("samples", name, &title, &hash, &now, &now)?;
    }

    // Setup embeddings
    println!("Preparing embeddings...");
    let model = pull_model(DEFAULT_EMBED_MODEL_URI, false)?;
    let mut engine = EmbeddingEngine::new(&model.path)?;
    store.ensure_vector_table(768)?;

    for (name, content) in SAMPLE_DOCS {
        let hash = Store::hash_content(content);
        let emb = engine.embed_document(content, Some(name))?;
        store.insert_embedding(&hash, 0, 0, &emb.embedding, &emb.model, &now)?;
    }

    // Hybrid search
    let query = "error handling best practices";
    let fts = store.search_fts(query, 10, None)?;
    let query_emb = engine.embed_query(query)?;
    let vec = store.search_vec(&query_emb.embedding, 10, None)?;

    let fts_tuples: Vec<_> = fts
        .iter()
        .map(|r| {
            (
                r.doc.filepath.clone(),
                r.doc.display_path.clone(),
                r.doc.title.clone(),
                r.doc.body.clone().unwrap_or_default(),
            )
        })
        .collect();
    let vec_tuples: Vec<_> = vec
        .iter()
        .map(|r| {
            (
                r.doc.filepath.clone(),
                r.doc.display_path.clone(),
                r.doc.title.clone(),
                String::new(),
            )
        })
        .collect();

    let results = hybrid_search_rrf(fts_tuples, vec_tuples, 60);

    println!("\nHybrid search: '{}'\n", query);
    println!(
        "FTS: {} | Vec: {} | Hybrid: {}",
        fts.len(),
        vec.len(),
        results.len()
    );
    for (i, r) in results.iter().take(5).enumerate() {
        println!("{}. [{:.4}] {}", i + 1, r.score, r.display_path);
    }

    let _ = std::fs::remove_file(&db_path);
    Ok(())
}
