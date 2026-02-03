//! Full-text search with BM25 ranking - fully self-contained.
//!
//! Run: `cargo run --example fts_search`

use anyhow::Result;
use qmd::{OutputFormat, Store, format_search_results};

const SAMPLE_DOCS: &[(&str, &str)] = &[
    ("rust-basics.md", include_str!("data/rust-basics.md")),
    ("error-handling.md", include_str!("data/error-handling.md")),
    ("async-await.md", include_str!("data/async-await.md")),
    ("ownership.md", include_str!("data/ownership.md")),
    ("traits.md", include_str!("data/traits.md")),
];

fn main() -> Result<()> {
    let db_path = std::env::temp_dir().join("qmd_fts.db");
    let _ = std::fs::remove_file(&db_path);
    let store = Store::open(&db_path)?;

    let now = chrono::Utc::now().to_rfc3339();
    for (name, content) in SAMPLE_DOCS {
        let hash = Store::hash_content(content);
        let title = Store::extract_title(content);
        store.insert_content(&hash, content, &now)?;
        store.insert_document("samples", name, &title, &hash, &now, &now)?;
    }

    // Search
    let query = "error handling Result";
    let results = store.search_fts(query, 5, None)?;

    println!("Query: '{}'\n", query);
    for r in &results {
        println!("[{:.2}] {} - {}", r.score, r.doc.path, r.doc.title);
    }

    // Multiple queries
    println!("\nMore searches:");
    for q in ["async await", "ownership borrowing", "trait bounds"] {
        let n = store.search_fts(q, 10, None)?.len();
        println!("  '{}': {} results", q, n);
    }

    // JSON output
    println!("\nJSON format:");
    format_search_results(&results, &OutputFormat::Json, false);

    let _ = std::fs::remove_file(&db_path);
    Ok(())
}
