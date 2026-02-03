//! Database maintenance and health checks - fully self-contained.
//!
//! Run: `cargo run --example maintenance`

use anyhow::Result;
use qmd::{Store, format_bytes};

const SAMPLE_DOCS: &[(&str, &str)] = &[
    ("rust-basics.md", include_str!("data/rust-basics.md")),
    ("error-handling.md", include_str!("data/error-handling.md")),
];

fn main() -> Result<()> {
    let db_path = std::env::temp_dir().join("qmd_maintenance.db");
    let _ = std::fs::remove_file(&db_path);
    let store = Store::open(&db_path)?;

    let now = chrono::Utc::now().to_rfc3339();
    for (name, content) in SAMPLE_DOCS {
        let hash = Store::hash_content(content);
        let title = Store::extract_title(content);
        store.insert_content(&hash, content, &now)?;
        store.insert_document("samples", name, &title, &hash, &now, &now)?;
    }

    let status = store.get_status()?;
    let size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);

    println!("Database: {}", db_path.display());
    println!("Size: {}", format_bytes(size as usize));

    println!("\nStatus:");
    println!("  Documents: {}", status.total_documents);
    println!("  Needs embedding: {}", status.needs_embedding);
    println!("  Vector index: {}", status.has_vector_index);

    // Cleanup demo
    println!("\nCleanup:");
    println!("  Cache cleared: {}", store.clear_cache()?);
    println!("  Orphaned removed: {}", store.cleanup_orphaned_content()?);
    store.vacuum()?;
    let new_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "  Vacuum: {} -> {}",
        format_bytes(size as usize),
        format_bytes(new_size as usize)
    );

    let _ = std::fs::remove_file(&db_path);
    Ok(())
}
