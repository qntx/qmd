//! Minimal quickstart example - fully self-contained.
//!
//! Run: `cargo run --example quickstart`

use anyhow::Result;
use qmd::Store;

/// Sample documents embedded at compile time.
const SAMPLE_DOCS: &[(&str, &str)] = &[
    ("rust-basics.md", include_str!("data/rust-basics.md")),
    ("error-handling.md", include_str!("data/error-handling.md")),
    ("async-await.md", include_str!("data/async-await.md")),
    ("ownership.md", include_str!("data/ownership.md")),
];

fn main() -> Result<()> {
    // Create temporary database
    let db_path = std::env::temp_dir().join("qmd_quickstart.db");
    let _ = std::fs::remove_file(&db_path);
    let store = Store::open(&db_path)?;

    // Index sample documents
    let now = chrono::Utc::now().to_rfc3339();
    for (name, content) in SAMPLE_DOCS {
        let hash = Store::hash_content(content);
        let title = Store::extract_title(content);
        store.insert_content(&hash, content, &now)?;
        store.insert_document("samples", name, &title, &hash, &now, &now)?;
    }

    // Search
    let results = store.search_fts("rust ownership", 5, None)?;
    println!("Search 'rust ownership':");
    for r in &results {
        println!("  [{:.2}] {}", r.score, r.doc.path);
    }

    let _ = std::fs::remove_file(&db_path);
    Ok(())
}
