//! Document retrieval and utility functions - fully self-contained.
//!
//! Run: `cargo run --example documents`

use anyhow::Result;
use qmd::{Store, format_bytes, is_docid};

const SAMPLE_DOCS: &[(&str, &str)] = &[
    ("rust-basics.md", include_str!("data/rust-basics.md")),
    ("error-handling.md", include_str!("data/error-handling.md")),
];

fn main() -> Result<()> {
    let db_path = std::env::temp_dir().join("qmd_documents.db");
    let _ = std::fs::remove_file(&db_path);
    let store = Store::open(&db_path)?;

    let now = chrono::Utc::now().to_rfc3339();
    for (name, content) in SAMPLE_DOCS {
        let hash = Store::hash_content(content);
        let title = Store::extract_title(content);
        store.insert_content(&hash, content, &now)?;
        store.insert_document("samples", name, &title, &hash, &now, &now)?;
    }

    // Get document
    if let Some(doc) = store.get_document("samples", "rust-basics.md")? {
        println!("Document: {}", doc.path);
        println!("  Title: {}", doc.title);
        println!("  Size: {}", format_bytes(doc.body_length));
        println!("  Hash: {}...", &doc.hash[..16]);
        println!("  DocID: {}", doc.docid);

        // Find by docid
        if let Some((c, p)) = store.find_document_by_docid(&doc.docid)? {
            println!("  Lookup: {}/{}", c, p);
        }
    }

    // Hash utilities
    println!("\nHash:");
    let hash = Store::hash_content("Hello, world!");
    println!("  'Hello, world!' -> {}...", &hash[..16]);

    // DocID validation
    println!("\nDocID check:");
    for id in ["#abc123", "abc123", "#ab", "#ABCDEF"] {
        println!("  '{}' -> {}", id, is_docid(id));
    }

    let _ = std::fs::remove_file(&db_path);
    Ok(())
}
