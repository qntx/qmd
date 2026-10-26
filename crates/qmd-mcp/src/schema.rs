//! Upstream-verbatim tool and resource-template definitions — the
//! `tools` array of upstream `tools/list` and the `resourceTemplates`
//! array of `resources/templates/list`, recorded from
//! `tobi/qmd` v2.8.3 (spike transcripts). Kept as raw JSON so clients
//! see exactly upstream's schemas; parsed once via serde.

use rmcp::model::{ResourceTemplate, Tool};

/// The four upstream tool definitions (`tools/list` `tools` array).
///
/// # Errors
/// [`serde_json::Error`] if the pinned JSON stops deserializing.
pub(crate) fn tools() -> Result<Vec<Tool>, serde_json::Error> {
    serde_json::from_str(include_str!("tools.json"))
}

/// The single upstream resource template (`resources/templates/list`).
///
/// # Errors
/// [`serde_json::Error`] if the pinned JSON stops deserializing.
pub(crate) fn resource_templates() -> Result<Vec<ResourceTemplate>, serde_json::Error> {
    serde_json::from_str(include_str!("resource-templates.json"))
}
