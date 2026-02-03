//! Cross-encoder reranking for improved relevance - fully self-contained.
//!
//! Run: `cargo run --example rerank`

use anyhow::Result;
use qmd::{RerankDocument, RerankEngine, Store, llm::DEFAULT_RERANK_MODEL_URI, pull_model};

const SAMPLE_DOCS: &[(&str, &str)] = &[
    ("rust-basics.md", include_str!("data/rust-basics.md")),
    ("error-handling.md", include_str!("data/error-handling.md")),
    ("async-await.md", include_str!("data/async-await.md")),
    ("ownership.md", include_str!("data/ownership.md")),
];

fn main() -> Result<()> {
    let db_path = std::env::temp_dir().join("qmd_rerank.db");
    let _ = std::fs::remove_file(&db_path);
    let store = Store::open(&db_path)?;

    let now = chrono::Utc::now().to_rfc3339();
    for (name, content) in SAMPLE_DOCS {
        let hash = Store::hash_content(content);
        let title = Store::extract_title(content);
        store.insert_content(&hash, content, &now)?;
        store.insert_document("samples", name, &title, &hash, &now, &now)?;
    }

    println!("Loading rerank model...");
    let model = pull_model(DEFAULT_RERANK_MODEL_URI, false)?;
    let mut reranker = RerankEngine::new(&model.path)?;

    // Get FTS results and convert to rerank format
    let query = "error handling";
    let initial = store.search_fts(query, 10, None)?;
    let docs: Vec<RerankDocument> = initial
        .iter()
        .filter_map(|r| {
            store
                .get_document(&r.doc.collection_name, &r.doc.path)
                .ok()
                .flatten()
                .map(|d| RerankDocument {
                    file: r.doc.path.clone(),
                    text: d.body.unwrap_or_default(),
                    title: Some(d.title),
                })
        })
        .collect();

    let result = reranker.rerank(query, &docs)?;

    println!("\nQuery: '{}'\n", query);
    println!("Before (BM25):");
    for (i, d) in docs.iter().take(5).enumerate() {
        println!("  {}. {}", i + 1, d.file);
    }
    println!("\nAfter (Reranked):");
    for (i, r) in result.results.iter().take(5).enumerate() {
        println!("  {}. [{:.4}] {}", i + 1, r.score, r.file);
    }

    let _ = std::fs::remove_file(&db_path);
    Ok(())
}
