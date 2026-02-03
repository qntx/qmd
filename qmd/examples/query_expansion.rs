//! Query expansion for improved search coverage - fully self-contained.
//!
//! Run: `cargo run --example query_expansion`

use anyhow::Result;
use qmd::{GenerationEngine, QueryType, Queryable, Store, expand_query_simple};

const SAMPLE_DOCS: &[(&str, &str)] = &[
    ("rust-basics.md", include_str!("data/rust-basics.md")),
    ("error-handling.md", include_str!("data/error-handling.md")),
];

fn main() -> Result<()> {
    let db_path = std::env::temp_dir().join("qmd_expansion.db");
    let _ = std::fs::remove_file(&db_path);
    let store = Store::open(&db_path)?;

    let now = chrono::Utc::now().to_rfc3339();
    for (name, content) in SAMPLE_DOCS {
        let hash = Store::hash_content(content);
        let title = Store::extract_title(content);
        store.insert_content(&hash, content, &now)?;
        store.insert_document("samples", name, &title, &hash, &now, &now)?;
    }

    let query = "rust error handling";
    println!("Query: '{}'\n", query);

    // Simple expansion
    println!("Simple expansion:");
    for q in expand_query_simple(query) {
        let t = match q.query_type {
            QueryType::Lex => "LEX",
            QueryType::Vec => "VEC",
            QueryType::Hyde => "HYD",
        };
        println!("  [{}] {}", t, q.text);
    }

    // Search with expanded queries
    println!("\nSearch results:");
    for q in expand_query_simple(query) {
        if q.query_type == QueryType::Lex {
            let n = store.search_fts(&q.text, 5, None)?.len();
            println!("  '{}': {} results", q.text, n);
        }
    }

    // LLM expansion
    println!("\nLLM expansion:");
    if GenerationEngine::is_available() {
        let engine = GenerationEngine::load_default()?;
        for q in engine.expand_query(query, true)? {
            println!("  [{:?}] {}", q.query_type, q.text);
        }
    } else {
        println!("  (not available)");
    }

    // Manual construction
    println!("\nManual:");
    for q in [
        Queryable::lex("rust error"),
        Queryable::vec("exception handling"),
    ] {
        println!("  [{:?}] {}", q.query_type, q.text);
    }

    let _ = std::fs::remove_file(&db_path);
    Ok(())
}
