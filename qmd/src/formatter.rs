//! Output formatting utilities.

use crate::store::{DocumentResult, SearchResult};
use chrono::{Datelike, Timelike};
use colored::Colorize;

/// Output format options.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OutputFormat {
    /// CLI-friendly output with colors.
    #[default]
    Cli,
    /// JSON output.
    Json,
    /// CSV output.
    Csv,
    /// Markdown output.
    Md,
    /// XML output.
    Xml,
    /// Just file paths (one per line).
    Files,
}

/// Format search results and return as string.
#[must_use]
pub fn format_search_results(
    results: &[SearchResult],
    format: &OutputFormat,
    full: bool,
) -> String {
    match format {
        OutputFormat::Json => format_search_json(results, full),
        OutputFormat::Csv => format_search_csv(results),
        OutputFormat::Md => format_search_md(results, full),
        OutputFormat::Xml => format_search_xml(results, full),
        OutputFormat::Files => format_search_files(results),
        OutputFormat::Cli => format_search_cli(results, full),
    }
}

/// Format documents and return as string.
#[must_use]
pub fn format_documents(
    docs: &[(DocumentResult, bool, Option<String>)],
    format: &OutputFormat,
) -> String {
    match format {
        OutputFormat::Json => format_docs_json(docs),
        OutputFormat::Csv => format_docs_csv(docs),
        OutputFormat::Md => format_docs_md(docs),
        OutputFormat::Xml => format_docs_xml(docs),
        OutputFormat::Files => format_docs_files(docs),
        OutputFormat::Cli => format_docs_cli(docs),
    }
}

// JSON output.
fn format_search_json(results: &[SearchResult], full: bool) -> String {
    let output: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            let mut obj = serde_json::json!({
                "docid": format!("#{}", r.doc.docid),
                "score": r.score,
                "file": r.doc.display_path,
                "title": r.doc.title,
            });
            if let Some(ref ctx) = r.doc.context {
                obj["context"] = serde_json::Value::String(ctx.clone());
            }
            if full && let Some(ref body) = r.doc.body {
                obj["body"] = serde_json::Value::String(body.clone());
            }
            obj
        })
        .collect();
    serde_json::to_string_pretty(&output).unwrap_or_default()
}

fn format_docs_json(docs: &[(DocumentResult, bool, Option<String>)]) -> String {
    let output: Vec<serde_json::Value> = docs
        .iter()
        .map(|(doc, skipped, skip_reason)| {
            if *skipped {
                serde_json::json!({
                    "file": doc.display_path,
                    "skipped": true,
                    "reason": skip_reason.as_deref().unwrap_or("unknown"),
                })
            } else {
                let mut obj = serde_json::json!({
                    "file": doc.display_path,
                    "title": doc.title,
                });
                if let Some(ref ctx) = doc.context {
                    obj["context"] = serde_json::Value::String(ctx.clone());
                }
                if let Some(ref body) = doc.body {
                    obj["body"] = serde_json::Value::String(body.clone());
                }
                obj
            }
        })
        .collect();
    serde_json::to_string_pretty(&output).unwrap_or_default()
}

// CSV output.
fn format_search_csv(results: &[SearchResult]) -> String {
    let mut lines = vec!["docid,score,file,title,context".to_string()];
    for r in results {
        lines.push(format!(
            "{},{:.4},{},{},{}",
            escape_csv(&format!("#{}", r.doc.docid)),
            r.score,
            escape_csv(&r.doc.display_path),
            escape_csv(&r.doc.title),
            escape_csv(r.doc.context.as_deref().unwrap_or(""))
        ));
    }
    lines.join("\n")
}

fn format_docs_csv(docs: &[(DocumentResult, bool, Option<String>)]) -> String {
    let mut lines = vec!["file,title,context,skipped,body".to_string()];
    for (doc, skipped, skip_reason) in docs {
        if *skipped {
            lines.push(format!(
                "{},{},{},true,{}",
                escape_csv(&doc.display_path),
                escape_csv(&doc.title),
                escape_csv(doc.context.as_deref().unwrap_or("")),
                escape_csv(skip_reason.as_deref().unwrap_or(""))
            ));
        } else {
            lines.push(format!(
                "{},{},{},false,{}",
                escape_csv(&doc.display_path),
                escape_csv(&doc.title),
                escape_csv(doc.context.as_deref().unwrap_or("")),
                escape_csv(doc.body.as_deref().unwrap_or(""))
            ));
        }
    }
    lines.join("\n")
}

// Markdown output.
fn format_search_md(results: &[SearchResult], full: bool) -> String {
    let mut out = String::new();
    for r in results {
        out.push_str(&format!(
            "## {} (score: {:.4})\n\n",
            r.doc.display_path, r.score
        ));
        out.push_str(&format!("**Title:** {}\n\n", r.doc.title));
        if let Some(ref ctx) = r.doc.context {
            out.push_str(&format!("**Context:** {ctx}\n\n"));
        }
        if full && let Some(ref body) = r.doc.body {
            out.push_str(&format!("```\n{body}\n```\n\n"));
        }
    }
    out
}

fn format_docs_md(docs: &[(DocumentResult, bool, Option<String>)]) -> String {
    let mut out = String::new();
    for (doc, skipped, skip_reason) in docs {
        out.push_str(&format!("## {}\n\n", doc.display_path));
        if !doc.title.is_empty() && doc.title != doc.display_path {
            out.push_str(&format!("**Title:** {}\n\n", doc.title));
        }
        if let Some(ref ctx) = doc.context {
            out.push_str(&format!("**Context:** {ctx}\n\n"));
        }
        if *skipped {
            out.push_str(&format!(
                "> {}\n\n",
                skip_reason.as_deref().unwrap_or("Skipped")
            ));
        } else if let Some(ref body) = doc.body {
            out.push_str(&format!("```\n{body}\n```\n\n"));
        }
    }
    out
}

// XML output.
fn format_search_xml(results: &[SearchResult], full: bool) -> String {
    let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<results>\n");
    for r in results {
        out.push_str("  <result>\n");
        out.push_str(&format!(
            "    <docid>#{}</docid>\n",
            escape_xml(&r.doc.docid)
        ));
        out.push_str(&format!("    <score>{:.4}</score>\n", r.score));
        out.push_str(&format!(
            "    <file>{}</file>\n",
            escape_xml(&r.doc.display_path)
        ));
        out.push_str(&format!(
            "    <title>{}</title>\n",
            escape_xml(&r.doc.title)
        ));
        if let Some(ref ctx) = r.doc.context {
            out.push_str(&format!("    <context>{}</context>\n", escape_xml(ctx)));
        }
        if full && let Some(ref body) = r.doc.body {
            out.push_str(&format!("    <body>{}</body>\n", escape_xml(body)));
        }
        out.push_str("  </result>\n");
    }
    out.push_str("</results>");
    out
}

fn format_docs_xml(docs: &[(DocumentResult, bool, Option<String>)]) -> String {
    let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<documents>\n");
    for (doc, skipped, skip_reason) in docs {
        out.push_str("  <document>\n");
        out.push_str(&format!(
            "    <file>{}</file>\n",
            escape_xml(&doc.display_path)
        ));
        out.push_str(&format!("    <title>{}</title>\n", escape_xml(&doc.title)));
        if let Some(ref ctx) = doc.context {
            out.push_str(&format!("    <context>{}</context>\n", escape_xml(ctx)));
        }
        if *skipped {
            out.push_str("    <skipped>true</skipped>\n");
            out.push_str(&format!(
                "    <reason>{}</reason>\n",
                escape_xml(skip_reason.as_deref().unwrap_or(""))
            ));
        } else if let Some(ref body) = doc.body {
            out.push_str(&format!("    <body>{}</body>\n", escape_xml(body)));
        }
        out.push_str("  </document>\n");
    }
    out.push_str("</documents>");
    out
}

// Files output (just paths).
fn format_search_files(results: &[SearchResult]) -> String {
    results
        .iter()
        .map(|r| r.doc.display_path.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_docs_files(docs: &[(DocumentResult, bool, Option<String>)]) -> String {
    docs.iter()
        .map(|(doc, skipped, _)| {
            if *skipped {
                format!("{} [SKIPPED]", doc.display_path)
            } else {
                doc.display_path.clone()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// CLI output (colored, human-friendly).
fn format_search_cli(results: &[SearchResult], full: bool) -> String {
    if results.is_empty() {
        return "No results found.".dimmed().to_string();
    }

    let mut out = String::new();
    for r in results {
        out.push_str(&format!(
            "{} {} {}\n",
            format!("#{}", r.doc.docid).cyan(),
            format!("{:.2}", r.score).dimmed(),
            r.doc.display_path.bold()
        ));
        if !r.doc.title.is_empty() {
            out.push_str(&format!("  {}\n", r.doc.title));
        }
        if let Some(ref ctx) = r.doc.context {
            out.push_str(&format!("  {}\n", format!("Context: {ctx}").dimmed()));
        }
        if full && let Some(ref body) = r.doc.body {
            out.push_str(&format!("\n{body}\n"));
        }
        out.push('\n');
    }
    out
}

fn format_docs_cli(docs: &[(DocumentResult, bool, Option<String>)]) -> String {
    let mut out = String::new();
    for (doc, skipped, skip_reason) in docs {
        out.push_str(&format!("\n{}\n", "=".repeat(60)));
        out.push_str(&format!("File: {}\n", doc.display_path));
        out.push_str(&format!("{}\n\n", "=".repeat(60)));

        if *skipped {
            out.push_str(&format!(
                "[SKIPPED: {}]\n",
                skip_reason.as_deref().unwrap_or("unknown")
            ));
            continue;
        }

        if let Some(ref ctx) = doc.context {
            out.push_str(&format!("Folder Context: {ctx}\n---\n\n"));
        }
        if let Some(ref body) = doc.body {
            out.push_str(&format!("{body}\n"));
        }
    }
    out
}

/// Escape a string for CSV output.
#[must_use]
pub fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Escape a string for XML output.
#[must_use]
pub fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Add line numbers to content.
#[must_use]
pub fn add_line_numbers(content: &str, start_line: usize) -> String {
    content
        .lines()
        .enumerate()
        .map(|(i, line)| format!("{:>6}\t{line}", start_line + i))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Format bytes into human-readable size.
#[must_use]
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes < KB {
        format!("{bytes} B")
    } else if bytes < MB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else if bytes < GB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    }
}

/// Format time ago.
#[must_use]
pub fn format_time_ago(timestamp: &str) -> String {
    let Ok(dt) = chrono::DateTime::parse_from_rfc3339(timestamp) else {
        return timestamp.to_string();
    };

    let now = chrono::Utc::now();
    let duration = now.signed_duration_since(dt);

    let seconds = duration.num_seconds();
    if seconds < 60 {
        return format!("{seconds}s ago");
    }

    let minutes = duration.num_minutes();
    if minutes < 60 {
        return format!("{minutes}m ago");
    }

    let hours = duration.num_hours();
    if hours < 24 {
        return format!("{hours}h ago");
    }

    let days = duration.num_days();
    format!("{days}d ago")
}

/// Format date/time like ls -l.
#[must_use]
pub fn format_ls_time(timestamp: &str) -> String {
    let Ok(dt) = chrono::DateTime::parse_from_rfc3339(timestamp) else {
        return timestamp.to_string();
    };

    let now = chrono::Utc::now();
    let six_months_ago = now - chrono::Duration::days(180);

    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let month = months[dt.month0() as usize];
    let day = dt.day();

    if dt < six_months_ago {
        format!("{month} {day:>2}  {}", dt.year())
    } else {
        format!("{month} {day:>2} {:02}:{:02}", dt.hour(), dt.minute())
    }
}
