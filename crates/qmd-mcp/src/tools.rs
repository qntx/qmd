//! Tool handlers — upstream `qmd_status` (mcp/server.ts:547-580) and
//! the stage-A shape of the others. All calls run through
//! `spawn_blocking` against the shared `Arc<Qmd>` (parity-matrix
//! deviation 43 groundwork).

use std::sync::Arc;

use qmd::Qmd;
use rmcp::model::{CallToolResult, ContentBlock};
use serde_json::json;

/// `status` — text summary plus `structuredContent` equal to upstream's
/// `getStatus()` shape (mcp/server.ts:553-580).
pub(crate) fn status(qmd: &Arc<Qmd>) -> Result<CallToolResult, qmd::Error> {
    let s = qmd.status()?;
    let mut summary = vec![
        "QMD Index Status:".to_owned(),
        format!("  Total documents: {}", s.total_documents),
        format!("  Needs embedding: {}", s.needs_embedding),
        format!(
            "  Vector index: {}",
            if s.has_vector_index { "yes" } else { "no" }
        ),
        format!("  Collections: {}", s.collections.len()),
    ];
    for col in &s.collections {
        // JS prints `null` for a missing path (String(null)).
        let path = col.path.as_deref().unwrap_or("null");
        summary.push(format!(
            "    - {}: {} ({} docs)",
            col.name, path, col.documents
        ));
    }
    let mut r = CallToolResult::success(vec![ContentBlock::text(summary.join("\n"))]);
    r.structured_content = Some(json!({
        "totalDocuments": s.total_documents,
        "needsEmbedding": s.needs_embedding,
        "hasVectorIndex": s.has_vector_index,
        "pendingMetadata": s.pending_metadata,
        "collections": s.collections.iter().map(|c| json!({
            "name": c.name,
            "path": c.path,
            "pattern": c.pattern,
            "documents": c.documents,
            "lastUpdated": c.last_updated,
        })).collect::<Vec<_>>(),
    }));
    // Upstream emits no `isError` field on success.
    r.is_error = None;
    Ok(r)
}
