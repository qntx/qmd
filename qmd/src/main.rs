//! QMD - Query Markdown Documents
//!
//! A full-text search CLI for markdown files.

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use qmd::cli::{
    Cli, CollectionCommands, Commands, ContextCommands, DbCommands, ModelCommands, OutputFormat,
};
use qmd::collections::{
    add_collection as yaml_add_collection, add_context, get_collection, list_all_contexts,
    list_collections as yaml_list_collections, remove_collection as yaml_remove_collection,
    remove_context, rename_collection as yaml_rename_collection, set_global_context,
};
use qmd::formatter::{
    add_line_numbers, format_bytes, format_documents, format_ls_time, format_search_results,
    format_time_ago,
};
use qmd::store::{Store, is_docid, is_virtual_path, parse_virtual_path, should_exclude};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Collection(cmd) => handle_collection(cmd),
        Commands::Context(cmd) => handle_context(cmd),
        Commands::Ls { path } => handle_ls(path),
        Commands::Get {
            file,
            from_line,
            max_lines,
            line_numbers,
        } => handle_get(&file, from_line, max_lines, line_numbers),
        Commands::MultiGet {
            pattern,
            max_lines,
            max_bytes,
            format,
        } => handle_multi_get(&pattern, max_lines, max_bytes, &format),
        Commands::Status => handle_status(),
        Commands::Update { pull } => handle_update(pull),
        Commands::Search {
            query,
            collection,
            limit,
            min_score,
            full,
            line_numbers: _,
            format,
        } => handle_search(
            &query,
            collection.as_deref(),
            limit,
            min_score,
            full,
            &format,
        ),
        Commands::Vsearch {
            query,
            collection,
            limit,
            min_score,
            full,
            line_numbers: _,
            format,
            model,
        } => handle_vsearch(
            &query,
            collection.as_deref(),
            limit,
            min_score,
            full,
            &format,
            model.as_deref(),
        ),
        Commands::Embed { force, model } => handle_embed(force, model.as_deref()),
        Commands::Models(cmd) => handle_models(cmd),
        Commands::Db(cmd) => handle_db(cmd),
    }
}

/// Handle collection subcommands.
fn handle_collection(cmd: CollectionCommands) -> Result<()> {
    match cmd {
        CollectionCommands::Add { path, name, mask } => {
            let abs_path = fs::canonicalize(&path)?;
            let abs_path_str = abs_path.to_string_lossy().to_string();

            // Generate name from path if not provided.
            let coll_name = name.unwrap_or_else(|| {
                abs_path
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "root".to_string())
            });

            // Check if collection exists.
            if get_collection(&coll_name)?.is_some() {
                eprintln!(
                    "{} Collection '{}' already exists.",
                    "Error:".red(),
                    coll_name
                );
                eprintln!("Use a different name with --name <name>");
                std::process::exit(1);
            }

            // Add to YAML config.
            yaml_add_collection(&coll_name, &abs_path_str, &mask)?;

            // Index files.
            println!("Creating collection '{}'...", coll_name);
            index_files(&abs_path_str, &mask, &coll_name)?;
            println!(
                "{} Collection '{}' created successfully",
                "✓".green(),
                coll_name
            );
        }
        CollectionCommands::List => {
            let store = Store::new()?;
            let collections = store.list_collections()?;

            if collections.is_empty() {
                println!("No collections found. Run 'qmd collection add .' to create one.");
                return Ok(());
            }

            println!("{}\n", "Collections:".bold());
            for coll in &collections {
                let time_ago = coll
                    .last_modified
                    .as_ref()
                    .map(|t| format_time_ago(t))
                    .unwrap_or_else(|| "never".to_string());

                println!(
                    "{} {}",
                    coll.name.cyan(),
                    format!("(qmd://{}/)", coll.name).dimmed()
                );
                println!("  {} {}", "Pattern:".dimmed(), coll.glob_pattern);
                println!("  {} {}", "Files:".dimmed(), coll.active_count);
                println!("  {} {}", "Updated:".dimmed(), time_ago);
                println!();
            }
        }
        CollectionCommands::Remove { name } => {
            // Check if collection exists.
            if get_collection(&name)?.is_none() {
                eprintln!("{} Collection not found: {}", "Error:".red(), name);
                std::process::exit(1);
            }

            let store = Store::new()?;
            let (deleted_docs, cleaned) = store.remove_collection_documents(&name)?;
            yaml_remove_collection(&name)?;

            println!("{} Removed collection '{}'", "✓".green(), name);
            println!("  Deleted {} documents", deleted_docs);
            if cleaned > 0 {
                println!("  Cleaned up {} orphaned content hashes", cleaned);
            }
        }
        CollectionCommands::Rename { old_name, new_name } => {
            // Check if old collection exists.
            if get_collection(&old_name)?.is_none() {
                eprintln!("{} Collection not found: {}", "Error:".red(), old_name);
                std::process::exit(1);
            }

            // Check if new name already exists.
            if get_collection(&new_name)?.is_some() {
                eprintln!(
                    "{} Collection name already exists: {}",
                    "Error:".red(),
                    new_name
                );
                std::process::exit(1);
            }

            let store = Store::new()?;
            store.rename_collection_documents(&old_name, &new_name)?;
            yaml_rename_collection(&old_name, &new_name)?;

            println!(
                "{} Renamed collection '{}' to '{}'",
                "✓".green(),
                old_name,
                new_name
            );
        }
    }
    Ok(())
}

/// Handle context subcommands.
fn handle_context(cmd: ContextCommands) -> Result<()> {
    match cmd {
        ContextCommands::Add { path, text } => {
            let path_arg = path.as_deref().unwrap_or(".");

            // Handle global context.
            if path_arg == "/" {
                set_global_context(Some(&text))?;
                println!("{} Set global context", "✓".green());
                println!("{}", format!("Context: {text}").dimmed());
                return Ok(());
            }

            // Handle virtual paths.
            if is_virtual_path(path_arg) {
                let Some((coll_name, file_path)) = parse_virtual_path(path_arg) else {
                    eprintln!("{} Invalid virtual path: {}", "Error:".red(), path_arg);
                    std::process::exit(1);
                };

                if get_collection(&coll_name)?.is_none() {
                    eprintln!("{} Collection not found: {}", "Error:".red(), coll_name);
                    std::process::exit(1);
                }

                add_context(&coll_name, &file_path, &text)?;
                let display = if file_path.is_empty() {
                    format!("qmd://{coll_name}/ (collection root)")
                } else {
                    format!("qmd://{coll_name}/{file_path}")
                };
                println!("{} Added context for: {}", "✓".green(), display);
                println!("{}", format!("Context: {text}").dimmed());
                return Ok(());
            }

            // Filesystem path - detect collection.
            let abs_path = fs::canonicalize(path_arg)?;
            let abs_path_str = abs_path.to_string_lossy().to_string();

            // Find matching collection.
            let collections = yaml_list_collections()?;
            let mut best_match: Option<(&str, String)> = None;

            for coll in &collections {
                if abs_path_str.starts_with(&format!("{}/", coll.path)) || abs_path_str == coll.path
                {
                    let rel_path = if abs_path_str.starts_with(&format!("{}/", coll.path)) {
                        abs_path_str[coll.path.len() + 1..].to_string()
                    } else {
                        String::new()
                    };

                    if best_match.is_none()
                        || coll.path.len() > best_match.as_ref().unwrap().0.len()
                    {
                        best_match = Some((&coll.name, rel_path));
                    }
                }
            }

            let Some((coll_name, rel_path)) = best_match else {
                eprintln!(
                    "{} Path is not in any indexed collection: {}",
                    "Error:".red(),
                    abs_path_str
                );
                std::process::exit(1);
            };

            add_context(coll_name, &rel_path, &text)?;
            let display = if rel_path.is_empty() {
                format!("qmd://{coll_name}/")
            } else {
                format!("qmd://{coll_name}/{rel_path}")
            };
            println!("{} Added context for: {}", "✓".green(), display);
            println!("{}", format!("Context: {text}").dimmed());
        }
        ContextCommands::List => {
            let contexts = list_all_contexts()?;

            if contexts.is_empty() {
                println!(
                    "{}",
                    "No contexts configured. Use 'qmd context add' to add one.".dimmed()
                );
                return Ok(());
            }

            println!("\n{}\n", "Configured Contexts".bold());
            let mut last_collection = String::new();

            for ctx in &contexts {
                if ctx.collection != last_collection {
                    println!("{}", ctx.collection.cyan());
                    last_collection.clone_from(&ctx.collection);
                }

                let path_display = if ctx.path.is_empty() || ctx.path == "/" {
                    "  / (root)".to_string()
                } else {
                    format!("  {}", ctx.path)
                };
                println!("{path_display}");
                println!("    {}", ctx.context.dimmed());
            }
        }
        ContextCommands::Check => {
            let store = Store::new()?;
            let collections = store.list_collections()?;
            let contexts = list_all_contexts()?;

            // Find collections without any context.
            let collections_with_context: HashSet<_> =
                contexts.iter().map(|c| c.collection.as_str()).collect();

            let mut missing = Vec::new();
            for coll in &collections {
                if !collections_with_context.contains(coll.name.as_str()) && coll.name != "*" {
                    missing.push(&coll.name);
                }
            }

            if missing.is_empty() {
                println!(
                    "\n{} {}\n",
                    "✓".green(),
                    "All collections have context configured".bold()
                );
            } else {
                println!("\n{}\n", "Collections without any context:".yellow());
                for name in missing {
                    println!("{}", name.cyan());
                    println!(
                        "  {}",
                        format!(
                            "Suggestion: qmd context add qmd://{name}/ \"Description of {name}\""
                        )
                        .dimmed()
                    );
                }
            }
        }
        ContextCommands::Rm { path } => {
            if path == "/" {
                set_global_context(None)?;
                println!("{} Removed global context", "✓".green());
                return Ok(());
            }

            if is_virtual_path(&path) {
                let Some((coll_name, file_path)) = parse_virtual_path(&path) else {
                    eprintln!("{} Invalid virtual path: {}", "Error:".red(), path);
                    std::process::exit(1);
                };

                if !remove_context(&coll_name, &file_path)? {
                    eprintln!("{} No context found for: {}", "Error:".red(), path);
                    std::process::exit(1);
                }

                println!("{} Removed context for: {}", "✓".green(), path);
            } else {
                eprintln!(
                    "{} Use virtual path format (qmd://collection/path)",
                    "Error:".red()
                );
                std::process::exit(1);
            }
        }
    }
    Ok(())
}

/// Handle ls command.
fn handle_ls(path: Option<String>) -> Result<()> {
    let store = Store::new()?;

    let Some(path_arg) = path else {
        // List all collections.
        let collections = yaml_list_collections()?;

        if collections.is_empty() {
            println!("No collections found. Run 'qmd collection add .' to index files.");
            return Ok(());
        }

        println!("{}\n", "Collections:".bold());
        for coll in collections {
            // Get file count from database.
            let files = store.list_files(&coll.name, None)?;
            println!(
                "  {}{}{}  {}",
                "qmd://".dimmed(),
                coll.name.cyan(),
                "/".dimmed(),
                format!("({} files)", files.len()).dimmed()
            );
        }
        return Ok(());
    };

    // Parse path argument.
    let (coll_name, path_prefix) = if is_virtual_path(&path_arg) {
        parse_virtual_path(&path_arg).unwrap_or_else(|| {
            eprintln!("{} Invalid virtual path: {}", "Error:".red(), path_arg);
            std::process::exit(1);
        })
    } else {
        // Assume collection name or collection/path format.
        let parts: Vec<&str> = path_arg.splitn(2, '/').collect();
        (
            parts[0].to_string(),
            parts.get(1).map(|s| s.to_string()).unwrap_or_default(),
        )
    };

    // Check collection exists.
    if get_collection(&coll_name)?.is_none() {
        eprintln!("{} Collection not found: {}", "Error:".red(), coll_name);
        eprintln!("Run 'qmd ls' to see available collections.");
        std::process::exit(1);
    }

    let prefix = if path_prefix.is_empty() {
        None
    } else {
        Some(path_prefix.as_str())
    };

    let files = store.list_files(&coll_name, prefix)?;

    if files.is_empty() {
        if prefix.is_some() {
            println!("No files found under qmd://{}/{}", coll_name, path_prefix);
        } else {
            println!("No files found in collection: {}", coll_name);
        }
        return Ok(());
    }

    // Calculate max width for size alignment.
    let max_size = files
        .iter()
        .map(|(_, _, _, size)| format_bytes(*size).len())
        .max()
        .unwrap_or(0);

    for (file_path, _title, modified_at, size) in files {
        let size_str = format!("{:>width$}", format_bytes(size), width = max_size);
        let time_str = format_ls_time(&modified_at);
        println!(
            "{}  {}  {}{}{}",
            size_str,
            time_str,
            format!("qmd://{coll_name}/").dimmed(),
            file_path.cyan(),
            ""
        );
    }

    Ok(())
}

/// Handle get command.
fn handle_get(
    file: &str,
    from_line: Option<usize>,
    max_lines: Option<usize>,
    line_numbers: bool,
) -> Result<()> {
    let store = Store::new()?;

    // Parse :linenum suffix.
    let (input_path, parsed_from_line) = if let Some(pos) = file.rfind(':') {
        let suffix = &file[pos + 1..];
        if let Ok(line) = suffix.parse::<usize>() {
            (&file[..pos], Some(line))
        } else {
            (file, None)
        }
    } else {
        (file, None)
    };

    let from_line = from_line.or(parsed_from_line);

    // Resolve document.
    let (collection, path) = if is_docid(input_path) {
        store
            .find_document_by_docid(input_path)?
            .ok_or_else(|| anyhow::anyhow!("Document not found: {}", input_path))?
    } else if is_virtual_path(input_path) {
        parse_virtual_path(input_path)
            .ok_or_else(|| anyhow::anyhow!("Invalid virtual path: {}", input_path))?
    } else {
        // Try as collection/path format.
        let parts: Vec<&str> = input_path.splitn(2, '/').collect();
        if parts.len() == 2 {
            (parts[0].to_string(), parts[1].to_string())
        } else {
            return Err(anyhow::anyhow!(
                "Could not resolve path: {}. Use qmd://collection/path format.",
                input_path
            ));
        }
    };

    let doc = store
        .get_document(&collection, &path)?
        .ok_or_else(|| anyhow::anyhow!("Document not found: qmd://{}/{}", collection, path))?;

    let mut body = doc.body.unwrap_or_default();
    let start_line = from_line.unwrap_or(1);

    // Apply line filtering.
    if from_line.is_some() || max_lines.is_some() {
        let lines: Vec<&str> = body.lines().collect();
        let start = start_line.saturating_sub(1);
        let end = max_lines.map_or(lines.len(), |n| (start + n).min(lines.len()));
        body = lines[start..end].join("\n");
    }

    // Add line numbers.
    if line_numbers {
        body = add_line_numbers(&body, start_line);
    }

    // Output context if exists.
    if let Some(ref ctx) = doc.context {
        println!("Folder Context: {ctx}\n---\n");
    }

    println!("{body}");
    Ok(())
}

/// Handle multi-get command.
fn handle_multi_get(
    pattern: &str,
    max_lines: Option<usize>,
    max_bytes: usize,
    format: &OutputFormat,
) -> Result<()> {
    let store = Store::new()?;

    // Parse pattern - comma-separated list or glob.
    let is_comma_list = pattern.contains(',') && !pattern.contains('*') && !pattern.contains('?');

    let mut results: Vec<(qmd::store::DocumentResult, bool, Option<String>)> = Vec::new();

    if is_comma_list {
        for name in pattern.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (collection, path) = if is_virtual_path(name) {
                match parse_virtual_path(name) {
                    Some(p) => p,
                    None => {
                        eprintln!("Invalid path: {}", name);
                        continue;
                    }
                }
            } else {
                let parts: Vec<&str> = name.splitn(2, '/').collect();
                if parts.len() == 2 {
                    (parts[0].to_string(), parts[1].to_string())
                } else {
                    eprintln!("Invalid path format: {}", name);
                    continue;
                }
            };

            match store.get_document(&collection, &path)? {
                Some(mut doc) => {
                    if doc.body_length > max_bytes {
                        let reason = format!(
                            "File too large ({}KB > {}KB)",
                            doc.body_length / 1024,
                            max_bytes / 1024
                        );
                        doc.body = None;
                        results.push((doc, true, Some(reason)));
                    } else {
                        // Apply line limit.
                        if let Some(limit) = max_lines {
                            if let Some(ref mut body) = doc.body {
                                let lines: Vec<&str> = body.lines().take(limit).collect();
                                *body = lines.join("\n");
                            }
                        }
                        results.push((doc, false, None));
                    }
                }
                None => {
                    eprintln!("File not found: {}", name);
                }
            }
        }
    } else {
        // Glob pattern - not fully implemented, just search by pattern.
        eprintln!("Glob patterns not yet implemented. Use comma-separated list.");
        std::process::exit(1);
    }

    format_documents(&results, format);
    Ok(())
}

/// Handle status command.
fn handle_status() -> Result<()> {
    let store = Store::new()?;
    let db_path = store.db_path().to_string_lossy().to_string();

    // Get database size.
    let index_size = fs::metadata(store.db_path())
        .map(|m| m.len() as usize)
        .unwrap_or(0);

    let status = store.get_status()?;
    let contexts = list_all_contexts()?;

    println!("{}\n", "QMD Status".bold());
    println!("Index: {db_path}");
    println!("Size:  {}\n", format_bytes(index_size));

    println!("{}", "Documents".bold());
    println!("  Total:    {} files indexed", status.total_documents);
    if status.needs_embedding > 0 {
        println!(
            "  {} {} (run 'qmd embed')",
            "Pending:".yellow(),
            format!("{} need embedding", status.needs_embedding)
        );
    }

    if !status.collections.is_empty() {
        println!("\n{}", "Collections".bold());

        for coll in &status.collections {
            let time_ago = coll
                .last_modified
                .as_ref()
                .map(|t| format_time_ago(t))
                .unwrap_or_else(|| "never".to_string());

            // Get contexts for this collection.
            let coll_contexts: Vec<_> = contexts
                .iter()
                .filter(|c| c.collection == coll.name)
                .collect();

            println!(
                "  {} {}",
                coll.name.cyan(),
                format!("(qmd://{}/)", coll.name).dimmed()
            );
            println!("    {} {}", "Pattern:".dimmed(), coll.glob_pattern);
            println!(
                "    {} {} (updated {})",
                "Files:".dimmed(),
                coll.active_count,
                time_ago
            );

            if !coll_contexts.is_empty() {
                println!("    {} {}", "Contexts:".dimmed(), coll_contexts.len());
                for ctx in coll_contexts {
                    let path_display = if ctx.path.is_empty() || ctx.path == "/" {
                        "/".to_string()
                    } else {
                        format!("/{}", ctx.path)
                    };
                    let preview = if ctx.context.len() > 60 {
                        format!("{}...", &ctx.context[..57])
                    } else {
                        ctx.context.clone()
                    };
                    println!("      {} {}", format!("{path_display}:").dimmed(), preview);
                }
            }
        }
    } else {
        println!(
            "\n{}",
            "No collections. Run 'qmd collection add .' to index markdown files.".dimmed()
        );
    }

    Ok(())
}

/// Handle update command.
fn handle_update(pull: bool) -> Result<()> {
    let store = Store::new()?;
    store.clear_cache()?;

    let collections = store.list_collections()?;

    if collections.is_empty() {
        println!(
            "{}",
            "No collections found. Run 'qmd collection add .' to index markdown files.".dimmed()
        );
        return Ok(());
    }

    println!(
        "{}\n",
        format!("Updating {} collection(s)...", collections.len()).bold()
    );

    for (i, coll) in collections.iter().enumerate() {
        println!(
            "{} {} {}",
            format!("[{}/{}]", i + 1, collections.len()).cyan(),
            coll.name.bold(),
            format!("({})", coll.glob_pattern).dimmed()
        );

        // Git pull if requested.
        if pull {
            let git_dir = Path::new(&coll.pwd).join(".git");
            if git_dir.exists() {
                println!("    Running git pull...");
                let output = std::process::Command::new("git")
                    .arg("pull")
                    .current_dir(&coll.pwd)
                    .output();

                match output {
                    Ok(o) if o.status.success() => {
                        let stdout = String::from_utf8_lossy(&o.stdout);
                        if !stdout.trim().is_empty() {
                            for line in stdout.lines() {
                                println!("    {line}");
                            }
                        }
                    }
                    Ok(o) => {
                        eprintln!("    {} git pull failed", "Warning:".yellow());
                        eprintln!("    {}", String::from_utf8_lossy(&o.stderr));
                    }
                    Err(e) => {
                        eprintln!("    {} Could not run git pull: {}", "Warning:".yellow(), e);
                    }
                }
            }
        }

        index_files(&coll.pwd, &coll.glob_pattern, &coll.name)?;
        println!();
    }

    println!("{} All collections updated.", "✓".green());
    Ok(())
}

/// Handle search command.
fn handle_search(
    query: &str,
    collection: Option<&str>,
    limit: usize,
    min_score: Option<f64>,
    full: bool,
    format: &OutputFormat,
) -> Result<()> {
    let store = Store::new()?;

    let mut results = store.search_fts(query, limit, collection)?;

    // Apply minimum score filter.
    if let Some(min) = min_score {
        results.retain(|r| r.score >= min);
    }

    // Load full body if requested.
    if full {
        for result in &mut results {
            if result.doc.body.is_none() {
                if let Ok(Some(doc)) =
                    store.get_document(&result.doc.collection_name, &result.doc.path)
                {
                    result.doc.body = doc.body;
                }
            }
        }
    }

    format_search_results(&results, format, full);
    Ok(())
}

/// Handle vector search command.
fn handle_vsearch(
    query: &str,
    collection: Option<&str>,
    limit: usize,
    min_score: Option<f64>,
    full: bool,
    format: &OutputFormat,
    model_path: Option<&str>,
) -> Result<()> {
    use qmd::llm::EmbeddingEngine;
    use std::path::PathBuf;

    let store = Store::new()?;

    // Load embedding model
    let mut engine = if let Some(path) = model_path {
        EmbeddingEngine::new(&PathBuf::from(path))?
    } else {
        match EmbeddingEngine::load_default() {
            Ok(e) => e,
            Err(_) => {
                eprintln!(
                    "{} Embedding model not found. Please specify --model or download a model.",
                    "Error:".red()
                );
                eprintln!(
                    "Place a GGUF embedding model in: {}",
                    qmd::config::get_model_cache_dir().display()
                );
                std::process::exit(1);
            }
        }
    };

    // Generate query embedding
    println!("Generating query embedding...");
    let query_result = engine.embed_query(query)?;

    // Search
    let mut results = store.search_vec(&query_result.embedding, limit, collection)?;

    // Apply minimum score filter
    if let Some(min) = min_score {
        results.retain(|r| r.score >= min);
    }

    if results.is_empty() {
        println!("No results found. Run 'qmd embed' to generate embeddings first.");
        return Ok(());
    }

    // Load full body if requested
    if full {
        for result in &mut results {
            if result.doc.body.is_none() {
                if let Ok(Some(doc)) =
                    store.get_document(&result.doc.collection_name, &result.doc.path)
                {
                    result.doc.body = doc.body;
                }
            }
        }
    }

    format_search_results(&results, format, full);
    Ok(())
}

/// Handle embed command.
fn handle_embed(force: bool, model_path: Option<&str>) -> Result<()> {
    use qmd::llm::EmbeddingEngine;
    use std::path::PathBuf;

    let store = Store::new()?;

    // Clear existing embeddings if force
    if force {
        let cleared = store.clear_embeddings()?;
        println!("Cleared {} existing embeddings", cleared);
    }

    // Get documents needing embedding
    let pending = store.get_hashes_needing_embedding()?;

    if pending.is_empty() {
        println!("{} All documents already have embeddings.", "✓".green());
        return Ok(());
    }

    println!(
        "{}",
        format!("Generating embeddings for {} documents...", pending.len()).bold()
    );

    // Load embedding model
    let mut engine = if let Some(path) = model_path {
        EmbeddingEngine::new(&PathBuf::from(path))?
    } else {
        match EmbeddingEngine::load_default() {
            Ok(e) => e,
            Err(_) => {
                eprintln!(
                    "{} Embedding model not found. Please specify --model or download a model.",
                    "Error:".red()
                );
                eprintln!(
                    "Place a GGUF embedding model in: {}",
                    qmd::config::get_model_cache_dir().display()
                );
                std::process::exit(1);
            }
        }
    };

    let now = chrono::Utc::now().to_rfc3339();
    let mut success = 0;
    let mut failed = 0;

    // Ensure vector table exists
    store.ensure_vector_table(768)?; // Common embedding dimension

    for (i, (hash, path, content)) in pending.iter().enumerate() {
        print!("\r  [{}/{}] Processing {}...", i + 1, pending.len(), path);

        match engine.embed(content) {
            Ok(result) => {
                store.insert_embedding(hash, 0, 0, &result.embedding, &result.model, &now)?;
                success += 1;
            }
            Err(e) => {
                eprintln!(
                    "\n  {} Failed to embed {}: {}",
                    "Warning:".yellow(),
                    path,
                    e
                );
                failed += 1;
            }
        }
    }

    println!(
        "\n{} Embedded {} documents ({} failed)",
        "✓".green(),
        success,
        failed
    );
    Ok(())
}

/// Handle models subcommand.
fn handle_models(cmd: ModelCommands) -> Result<()> {
    use qmd::llm::{DEFAULT_EMBED_MODEL, list_cached_models};

    match cmd {
        ModelCommands::List => {
            let models = list_cached_models();
            let cache_dir = qmd::config::get_model_cache_dir();

            println!("{}\n", "Available Models".bold());
            println!("Cache directory: {}\n", cache_dir.display());

            if models.is_empty() {
                println!("No models found in cache.");
                println!(
                    "\n{}",
                    "To use vector search, download a GGUF embedding model:".dimmed()
                );
                println!("  1. Download a model (e.g., embeddinggemma-300M-Q8_0.gguf)");
                println!("  2. Place it in: {}", cache_dir.display());
            } else {
                println!("{}", "Cached models:".cyan());
                for model in &models {
                    let is_default = model == DEFAULT_EMBED_MODEL;
                    if is_default {
                        println!("  {} {}", model, "(default)".green());
                    } else {
                        println!("  {}", model);
                    }
                }
            }
        }
        ModelCommands::Info { name } => {
            let model_name = name.as_deref().unwrap_or(DEFAULT_EMBED_MODEL);
            let cache_dir = qmd::config::get_model_cache_dir();
            let model_path = cache_dir.join(model_name);

            println!("{}\n", "Model Info".bold());
            println!("Name: {}", model_name);
            println!("Path: {}", model_path.display());

            if model_path.exists() {
                let size = fs::metadata(&model_path)
                    .map(|m| format_bytes(m.len() as usize))
                    .unwrap_or_else(|_| "unknown".to_string());
                println!("Status: {} ({})", "Downloaded".green(), size);
            } else {
                println!("Status: {}", "Not downloaded".red());
            }
        }
    }

    Ok(())
}

/// Handle database maintenance commands.
fn handle_db(cmd: DbCommands) -> Result<()> {
    let store = Store::new()?;

    match cmd {
        DbCommands::Cleanup => {
            let inactive = store.delete_inactive_documents()?;
            let orphaned_content = store.cleanup_orphaned_content()?;
            let orphaned_vectors = store.cleanup_orphaned_vectors()?;

            println!("{} Database cleanup complete", "✓".green());
            println!("  Removed {} inactive documents", inactive);
            println!("  Removed {} orphaned content entries", orphaned_content);
            println!("  Removed {} orphaned vector entries", orphaned_vectors);
        }
        DbCommands::Vacuum => {
            println!("Vacuuming database...");
            store.vacuum()?;
            println!("{} Database vacuumed", "✓".green());
        }
        DbCommands::ClearCache => {
            let cleared = store.clear_cache()?;
            println!("{} Cleared {} cached entries", "✓".green(), cleared);
        }
    }

    Ok(())
}

/// Index files in a directory.
fn index_files(pwd: &str, glob_pattern: &str, collection_name: &str) -> Result<()> {
    let store = Store::new()?;
    let now = chrono::Utc::now().to_rfc3339();

    // Collect matching files.
    let glob_matcher = glob::Pattern::new(glob_pattern)?;
    let mut files = Vec::new();

    for entry in WalkDir::new(pwd)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();

        // Skip directories.
        if !path.is_file() {
            continue;
        }

        // Skip excluded paths.
        if should_exclude(path) {
            continue;
        }

        // Check glob match.
        let rel_path = path.strip_prefix(pwd).unwrap_or(path);
        let rel_path_str = rel_path.to_string_lossy();

        if glob_matcher.matches(&rel_path_str) {
            files.push((path.to_path_buf(), rel_path_str.to_string()));
        }
    }

    if files.is_empty() {
        println!("  No files found matching pattern.");
        return Ok(());
    }

    let mut indexed = 0;
    let mut updated = 0;
    let mut unchanged = 0;
    let mut seen_paths = HashSet::new();

    for (abs_path, rel_path) in &files {
        let normalized_path = Store::handelize(rel_path);
        seen_paths.insert(normalized_path.clone());

        // Read file content.
        let content = match fs::read_to_string(abs_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  Warning: Could not read {}: {}", rel_path, e);
                continue;
            }
        };

        let hash = Store::hash_content(&content);
        let title = Store::extract_title(&content);

        // Check if document exists.
        if let Some((doc_id, existing_hash, existing_title)) =
            store.find_active_document(collection_name, &normalized_path)?
        {
            if existing_hash == hash {
                // Check if title changed.
                if existing_title != title {
                    store.update_document_title(doc_id, &title, &now)?;
                }
                unchanged += 1;
            } else {
                // Content changed - update.
                store.insert_content(&hash, &content, &now)?;
                store.update_document(doc_id, &title, &hash, &now)?;
                updated += 1;
            }
        } else {
            // New document.
            store.insert_content(&hash, &content, &now)?;
            store.insert_document(collection_name, &normalized_path, &title, &hash, &now, &now)?;
            indexed += 1;
        }
    }

    // Deactivate removed files.
    let existing_paths = store.get_active_document_paths(collection_name)?;
    let mut deactivated = 0;

    for path in existing_paths {
        if !seen_paths.contains(&path) {
            store.deactivate_document(collection_name, &path)?;
            deactivated += 1;
        }
    }

    println!(
        "  {} indexed, {} updated, {} unchanged, {} removed",
        indexed, updated, unchanged, deactivated
    );

    Ok(())
}
