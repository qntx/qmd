//! Collection management operations - fully self-contained.
//!
//! Run: `cargo run --example collections`

use anyhow::Result;
use qmd::{Store, is_virtual_path, parse_virtual_path};

const SAMPLE_DOCS: &[(&str, &str)] = &[
    ("rust-basics.md", include_str!("data/rust-basics.md")),
    ("error-handling.md", include_str!("data/error-handling.md")),
    ("async-await.md", include_str!("data/async-await.md")),
];

fn main() -> Result<()> {
    let db_path = std::env::temp_dir().join("qmd_collections.db");
    let _ = std::fs::remove_file(&db_path);
    let store = Store::open(&db_path)?;

    let now = chrono::Utc::now().to_rfc3339();
    for (name, content) in SAMPLE_DOCS {
        let hash = Store::hash_content(content);
        let title = Store::extract_title(content);
        store.insert_content(&hash, content, &now)?;
        store.insert_document("samples", name, &title, &hash, &now, &now)?;
    }

    // List files
    println!("Files in 'samples':");
    for (path, title, _, size) in store.list_files("samples", None)? {
        println!("  {} - {} ({} bytes)", path, title, size);
    }

    // Virtual path parsing
    println!("\nVirtual path parsing:");
    for path in [
        "qmd://samples/rust-basics.md",
        "/local/file.md",
        "relative.md",
    ] {
        if is_virtual_path(path) {
            if let Some((coll, file)) = parse_virtual_path(path) {
                println!("  {} -> [{}] {}", path, coll, file);
            }
        } else {
            println!("  {} -> local", path);
        }
    }

    let _ = std::fs::remove_file(&db_path);
    Ok(())
}
