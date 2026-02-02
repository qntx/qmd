//! Output formatting utilities.

use crate::cli::OutputFormat;
use crate::store::{DocumentResult, SearchResult};
use chrono::{Datelike, Timelike};
use colored::Colorize;

/// Format search results for output.
pub fn format_search_results(results: &[SearchResult], format: &OutputFormat, full: bool) {
    match format {
        OutputFormat::Json => print_search_json(results, full),
        OutputFormat::Csv => print_search_csv(results),
        OutputFormat::Md => print_search_md(results, full),
        OutputFormat::Xml => print_search_xml(results, full),
        OutputFormat::Files => print_search_files(results),
        OutputFormat::Cli => print_search_cli(results, full),
    }
}

/// Format documents for output.
pub fn format_documents(docs: &[(DocumentResult, bool, Option<String>)], format: &OutputFormat) {
    match format {
        OutputFormat::Json => print_docs_json(docs),
        OutputFormat::Csv => print_docs_csv(docs),
        OutputFormat::Md => print_docs_md(docs),
        OutputFormat::Xml => print_docs_xml(docs),
        OutputFormat::Files => print_docs_files(docs),
        OutputFormat::Cli => print_docs_cli(docs),
    }
}

// JSON output.
fn print_search_json(results: &[SearchResult], full: bool) {
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
            if full {
                if let Some(ref body) = r.doc.body {
                    obj["body"] = serde_json::Value::String(body.clone());
                }
            }
            obj
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );
}

fn print_docs_json(docs: &[(DocumentResult, bool, Option<String>)]) {
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
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );
}

// CSV output.
fn print_search_csv(results: &[SearchResult]) {
    println!("docid,score,file,title,context");
    for r in results {
        println!(
            "{},{:.4},{},{},{}",
            escape_csv(&format!("#{}", r.doc.docid)),
            r.score,
            escape_csv(&r.doc.display_path),
            escape_csv(&r.doc.title),
            escape_csv(r.doc.context.as_deref().unwrap_or(""))
        );
    }
}

fn print_docs_csv(docs: &[(DocumentResult, bool, Option<String>)]) {
    println!("file,title,context,skipped,body");
    for (doc, skipped, skip_reason) in docs {
        if *skipped {
            println!(
                "{},{},{},true,{}",
                escape_csv(&doc.display_path),
                escape_csv(&doc.title),
                escape_csv(doc.context.as_deref().unwrap_or("")),
                escape_csv(skip_reason.as_deref().unwrap_or(""))
            );
        } else {
            println!(
                "{},{},{},false,{}",
                escape_csv(&doc.display_path),
                escape_csv(&doc.title),
                escape_csv(doc.context.as_deref().unwrap_or("")),
                escape_csv(doc.body.as_deref().unwrap_or(""))
            );
        }
    }
}

// Markdown output.
fn print_search_md(results: &[SearchResult], full: bool) {
    for r in results {
        println!("## {} (score: {:.4})\n", r.doc.display_path, r.score);
        println!("**Title:** {}\n", r.doc.title);
        if let Some(ref ctx) = r.doc.context {
            println!("**Context:** {ctx}\n");
        }
        if full {
            if let Some(ref body) = r.doc.body {
                println!("```\n{body}\n```\n");
            }
        }
    }
}

fn print_docs_md(docs: &[(DocumentResult, bool, Option<String>)]) {
    for (doc, skipped, skip_reason) in docs {
        println!("## {}\n", doc.display_path);
        if !doc.title.is_empty() && doc.title != doc.display_path {
            println!("**Title:** {}\n", doc.title);
        }
        if let Some(ref ctx) = doc.context {
            println!("**Context:** {ctx}\n");
        }
        if *skipped {
            println!("> {}\n", skip_reason.as_deref().unwrap_or("Skipped"));
        } else if let Some(ref body) = doc.body {
            println!("```\n{body}\n```\n");
        }
    }
}

// XML output.
fn print_search_xml(results: &[SearchResult], full: bool) {
    println!(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    println!("<results>");
    for r in results {
        println!("  <result>");
        println!("    <docid>#{}</docid>", escape_xml(&r.doc.docid));
        println!("    <score>{:.4}</score>", r.score);
        println!("    <file>{}</file>", escape_xml(&r.doc.display_path));
        println!("    <title>{}</title>", escape_xml(&r.doc.title));
        if let Some(ref ctx) = r.doc.context {
            println!("    <context>{}</context>", escape_xml(ctx));
        }
        if full {
            if let Some(ref body) = r.doc.body {
                println!("    <body>{}</body>", escape_xml(body));
            }
        }
        println!("  </result>");
    }
    println!("</results>");
}

fn print_docs_xml(docs: &[(DocumentResult, bool, Option<String>)]) {
    println!(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    println!("<documents>");
    for (doc, skipped, skip_reason) in docs {
        println!("  <document>");
        println!("    <file>{}</file>", escape_xml(&doc.display_path));
        println!("    <title>{}</title>", escape_xml(&doc.title));
        if let Some(ref ctx) = doc.context {
            println!("    <context>{}</context>", escape_xml(ctx));
        }
        if *skipped {
            println!("    <skipped>true</skipped>");
            println!(
                "    <reason>{}</reason>",
                escape_xml(skip_reason.as_deref().unwrap_or(""))
            );
        } else if let Some(ref body) = doc.body {
            println!("    <body>{}</body>", escape_xml(body));
        }
        println!("  </document>");
    }
    println!("</documents>");
}

// Files output (just paths).
fn print_search_files(results: &[SearchResult]) {
    for r in results {
        println!("{}", r.doc.display_path);
    }
}

fn print_docs_files(docs: &[(DocumentResult, bool, Option<String>)]) {
    for (doc, skipped, _) in docs {
        let status = if *skipped { " [SKIPPED]" } else { "" };
        println!("{}{status}", doc.display_path);
    }
}

// CLI output (colored, human-friendly).
fn print_search_cli(results: &[SearchResult], full: bool) {
    if results.is_empty() {
        println!("{}", "No results found.".dimmed());
        return;
    }

    for r in results {
        println!(
            "{} {} {}",
            format!("#{}", r.doc.docid).cyan(),
            format!("{:.2}", r.score).dimmed(),
            r.doc.display_path.bold()
        );
        if !r.doc.title.is_empty() {
            println!("  {}", r.doc.title);
        }
        if let Some(ref ctx) = r.doc.context {
            println!("  {}", format!("Context: {ctx}").dimmed());
        }
        if full {
            if let Some(ref body) = r.doc.body {
                println!("\n{body}\n");
            }
        }
        println!();
    }
}

fn print_docs_cli(docs: &[(DocumentResult, bool, Option<String>)]) {
    for (doc, skipped, skip_reason) in docs {
        println!("\n{}", "=".repeat(60));
        println!("File: {}", doc.display_path);
        println!("{}\n", "=".repeat(60));

        if *skipped {
            println!("[SKIPPED: {}]", skip_reason.as_deref().unwrap_or("unknown"));
            continue;
        }

        if let Some(ref ctx) = doc.context {
            println!("Folder Context: {ctx}\n---\n");
        }
        if let Some(ref body) = doc.body {
            println!("{body}");
        }
    }
}

/// Escape a string for CSV output.
pub fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Escape a string for XML output.
pub fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Add line numbers to content.
pub fn add_line_numbers(content: &str, start_line: usize) -> String {
    content
        .lines()
        .enumerate()
        .map(|(i, line)| format!("{:>6}\t{line}", start_line + i))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Format bytes into human-readable size.
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
