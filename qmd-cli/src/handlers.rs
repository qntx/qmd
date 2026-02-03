// Command handler implementations for qmd-cli

fn handle_cleanup() -> Result<()> {
    let store = Store::new()?;
    println!("{}\n", "Database Cleanup".bold());
    let cache_cleared = store.clear_cache()?;
    println!("{} Cleared {} cached entries", "✓".green(), cache_cleared);
    let inactive = store.delete_inactive_documents()?;
    if inactive > 0 { println!("{} Removed {} inactive documents", "✓".green(), inactive); }
    let orphaned_content = store.cleanup_orphaned_content()?;
    if orphaned_content > 0 { println!("{} Removed {} orphaned content entries", "✓".green(), orphaned_content); }
    let orphaned_vectors = store.cleanup_orphaned_vectors()?;
    if orphaned_vectors > 0 { println!("{} Removed {} orphaned vector entries", "✓".green(), orphaned_vectors); }
    store.vacuum()?;
    println!("{} Database vacuumed", "✓".green());
    println!("\n{} Cleanup complete", "✓".green());
    Ok(())
}

fn handle_collection(cmd: CollectionCommands) -> Result<()> {
    match cmd {
        CollectionCommands::Add { path, name, mask } => {
            let abs_path = fs::canonicalize(&path)?;
            let abs_path_str = abs_path.to_string_lossy().to_string();
            let coll_name = name.unwrap_or_else(|| {
                abs_path.file_name().map_or_else(|| "root".to_string(), |s| s.to_string_lossy().to_string())
            });
            if get_collection(&coll_name)?.is_some() {
                eprintln!("{} Collection '{}' already exists.", "Error:".red(), coll_name);
                std::process::exit(1);
            }
            yaml_add_collection(&coll_name, &abs_path_str, &mask)?;
            println!("Creating collection '{coll_name}'...");
            index_files(&abs_path_str, &mask, &coll_name)?;
            println!("{} Collection '{}' created successfully", "✓".green(), coll_name);
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
                let time_ago = coll.last_modified.as_ref().map_or_else(|| "never".to_string(), |t| format_time_ago(t));
                println!("{} {}", coll.name.cyan(), format!("(qmd://{}/)", coll.name).dimmed());
                println!("  {} {}", "Pattern:".dimmed(), coll.glob_pattern);
                println!("  {} {}", "Files:".dimmed(), coll.active_count);
                println!("  {} {}", "Updated:".dimmed(), time_ago);
                println!();
            }
        }
        CollectionCommands::Remove { name } => {
            if get_collection(&name)?.is_none() {
                eprintln!("{} Collection not found: {}", "Error:".red(), name);
                std::process::exit(1);
            }
            let store = Store::new()?;
            let (deleted_docs, cleaned) = store.remove_collection_documents(&name)?;
            yaml_remove_collection(&name)?;
            println!("{} Removed collection '{}'", "✓".green(), name);
            println!("  Deleted {deleted_docs} documents");
            if cleaned > 0 { println!("  Cleaned up {cleaned} orphaned content hashes"); }
        }
        CollectionCommands::Rename { old_name, new_name } => {
            if get_collection(&old_name)?.is_none() {
                eprintln!("{} Collection not found: {}", "Error:".red(), old_name);
                std::process::exit(1);
            }
            if get_collection(&new_name)?.is_some() {
                eprintln!("{} Collection name already exists: {}", "Error:".red(), new_name);
                std::process::exit(1);
            }
            let store = Store::new()?;
            store.rename_collection_documents(&old_name, &new_name)?;
            yaml_rename_collection(&old_name, &new_name)?;
            println!("{} Renamed collection '{}' to '{}'", "✓".green(), old_name, new_name);
        }
    }
    Ok(())
}

fn handle_context(cmd: ContextCommands) -> Result<()> {
    match cmd {
        ContextCommands::Add { path, text } => {
            let path_arg = path.as_deref().unwrap_or(".");
            if path_arg == "/" {
                set_global_context(Some(&text))?;
                println!("{} Set global context", "✓".green());
                return Ok(());
            }
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
                println!("{} Added context", "✓".green());
                return Ok(());
            }
            let abs_path = fs::canonicalize(path_arg)?;
            let abs_path_str = abs_path.to_string_lossy().to_string();
            let collections = yaml_list_collections()?;
            let mut best_match: Option<(&str, String)> = None;
            for coll in &collections {
                if abs_path_str.starts_with(&format!("{}/", coll.path)) || abs_path_str == coll.path {
                    let rel_path = if abs_path_str.starts_with(&format!("{}/", coll.path)) {
                        abs_path_str[coll.path.len() + 1..].to_string()
                    } else { String::new() };
                    if best_match.is_none() || coll.path.len() > best_match.as_ref().unwrap().0.len() {
                        best_match = Some((&coll.name, rel_path));
                    }
                }
            }
            let Some((coll_name, rel_path)) = best_match else {
                eprintln!("{} Path is not in any indexed collection", "Error:".red());
                std::process::exit(1);
            };
            add_context(coll_name, &rel_path, &text)?;
            println!("{} Added context", "✓".green());
        }
        ContextCommands::List => {
            let contexts = list_all_contexts()?;
            if contexts.is_empty() {
                println!("{}", "No contexts configured.".dimmed());
                return Ok(());
            }
            println!("\n{}\n", "Configured Contexts".bold());
            let mut last_collection = String::new();
            for ctx in &contexts {
                if ctx.collection != last_collection {
                    println!("{}", ctx.collection.cyan());
                    last_collection.clone_from(&ctx.collection);
                }
                let path_display = if ctx.path.is_empty() || ctx.path == "/" { "  / (root)".to_string() } else { format!("  {}", ctx.path) };
                println!("{path_display}");
                println!("    {}", ctx.context.dimmed());
            }
        }
        ContextCommands::Check => {
            let store = Store::new()?;
            let collections = store.list_collections()?;
            let contexts = list_all_contexts()?;
            let collections_with_context: HashSet<_> = contexts.iter().map(|c| c.collection.as_str()).collect();
            let mut missing = Vec::new();
            for coll in &collections {
                if !collections_with_context.contains(coll.name.as_str()) && coll.name != "*" {
                    missing.push(&coll.name);
                }
            }
            if missing.is_empty() {
                println!("\n{} {}\n", "✓".green(), "All collections have context configured".bold());
            } else {
                println!("\n{}\n", "Collections without any context:".yellow());
                for name in missing { println!("{}", name.cyan()); }
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
                eprintln!("{} Use virtual path format (qmd://collection/path)", "Error:".red());
                std::process::exit(1);
            }
        }
    }
    Ok(())
}

fn handle_ls(path: Option<String>) -> Result<()> {
    let store = Store::new()?;
    let Some(path_arg) = path else {
        let collections = yaml_list_collections()?;
        if collections.is_empty() {
            println!("No collections found.");
            return Ok(());
        }
        println!("{}\n", "Collections:".bold());
        for coll in collections {
            let files = store.list_files(&coll.name, None)?;
            println!("  {}{}{}  {}", "qmd://".dimmed(), coll.name.cyan(), "/".dimmed(), format!("({} files)", files.len()).dimmed());
        }
        return Ok(());
    };
    let (coll_name, path_prefix) = if is_virtual_path(&path_arg) {
        parse_virtual_path(&path_arg).unwrap_or_else(|| {
            eprintln!("{} Invalid virtual path: {}", "Error:".red(), path_arg);
            std::process::exit(1);
        })
    } else {
        let parts: Vec<&str> = path_arg.splitn(2, '/').collect();
        (parts[0].to_string(), parts.get(1).map(ToString::to_string).unwrap_or_default())
    };
    if get_collection(&coll_name)?.is_none() {
        eprintln!("{} Collection not found: {}", "Error:".red(), coll_name);
        std::process::exit(1);
    }
    let prefix = if path_prefix.is_empty() { None } else { Some(path_prefix.as_str()) };
    let files = store.list_files(&coll_name, prefix)?;
    if files.is_empty() {
        println!("No files found.");
        return Ok(());
    }
    let max_size = files.iter().map(|(_, _, _, size)| format_bytes(*size).len()).max().unwrap_or(0);
    for (file_path, _title, modified_at, size) in files {
        let size_str = format!("{:>width$}", format_bytes(size), width = max_size);
        let time_str = format_ls_time(&modified_at);
        println!("{}  {}  {}{}", size_str, time_str, format!("qmd://{coll_name}/").dimmed(), file_path.cyan());
    }
    Ok(())
}

fn handle_get(file: &str, from_line: Option<usize>, max_lines: Option<usize>, line_numbers: bool) -> Result<()> {
    let store = Store::new()?;
    let (input_path, parsed_from_line) = if let Some(pos) = file.rfind(':') {
        let suffix = &file[pos + 1..];
        if let Ok(line) = suffix.parse::<usize>() { (&file[..pos], Some(line)) } else { (file, None) }
    } else { (file, None) };
    let from_line = from_line.or(parsed_from_line);
    let (collection, path) = if is_docid(input_path) {
        store.find_document_by_docid(input_path)?.ok_or_else(|| anyhow::anyhow!("Document not found: {input_path}"))?
    } else if is_virtual_path(input_path) {
        parse_virtual_path(input_path).ok_or_else(|| anyhow::anyhow!("Invalid virtual path: {input_path}"))?
    } else {
        let parts: Vec<&str> = input_path.splitn(2, '/').collect();
        if parts.len() == 2 { (parts[0].to_string(), parts[1].to_string()) }
        else { return Err(anyhow::anyhow!("Could not resolve path: {input_path}")); }
    };
    let doc = store.get_document(&collection, &path)?.ok_or_else(|| anyhow::anyhow!("Document not found"))?;
    let mut body = doc.body.unwrap_or_default();
    let start_line = from_line.unwrap_or(1);
    if from_line.is_some() || max_lines.is_some() {
        let lines: Vec<&str> = body.lines().collect();
        let start = start_line.saturating_sub(1);
        let end = max_lines.map_or(lines.len(), |n| (start + n).min(lines.len()));
        body = lines[start..end].join("\n");
    }
    if line_numbers { body = add_line_numbers(&body, start_line); }
    if let Some(ref ctx) = doc.context { println!("Folder Context: {ctx}\n---\n"); }
    println!("{body}");
    Ok(())
}

fn handle_multi_get(pattern: &str, max_lines: Option<usize>, max_bytes: usize, format: &OutputFormat) -> Result<()> {
    let store = Store::new()?;
    let is_comma_list = pattern.contains(',') && !pattern.contains('*') && !pattern.contains('?');
    let mut results: Vec<(qmd::DocumentResult, bool, Option<String>)> = Vec::new();
    if is_comma_list {
        for name in pattern.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (collection, path) = if is_virtual_path(name) {
                if let Some(p) = parse_virtual_path(name) { p } else { continue; }
            } else {
                let parts: Vec<&str> = name.splitn(2, '/').collect();
                if parts.len() == 2 { (parts[0].to_string(), parts[1].to_string()) } else { continue; }
            };
            if let Ok(Some(mut doc)) = store.get_document(&collection, &path) {
                if doc.body_length > max_bytes {
                    doc.body = None;
                    results.push((doc, true, Some("File too large".to_string())));
                } else {
                    if let Some(limit) = max_lines { if let Some(ref mut body) = doc.body {
                        let lines: Vec<&str> = body.lines().take(limit).collect();
                        *body = lines.join("\n");
                    }}
                    results.push((doc, false, None));
                }
            }
        }
    } else {
        let matched_docs = match_files_by_glob(&store, pattern)?;
        for mut doc in matched_docs {
            if doc.body_length > max_bytes {
                doc.body = None;
                results.push((doc, true, Some("File too large".to_string())));
            } else if let Ok(Some(mut full_doc)) = store.get_document(&doc.collection_name, &doc.path) {
                if let Some(limit) = max_lines { if let Some(ref body) = full_doc.body {
                    let lines: Vec<&str> = body.lines().take(limit).collect();
                    full_doc.body = Some(lines.join("\n"));
                }}
                results.push((full_doc, false, None));
            }
        }
    }
    println!("{}", format_documents(&results, format));
    Ok(())
}

fn handle_status() -> Result<()> {
    let store = Store::new()?;
    let db_path = store.db_path().to_string_lossy().to_string();
    let index_size = fs::metadata(store.db_path()).map_or(0, |m| m.len() as usize);
    let status = store.get_status()?;
    let contexts = list_all_contexts()?;
    println!("{}\n", "QMD Status".bold());
    println!("Index: {db_path}");
    println!("Size:  {}\n", format_bytes(index_size));
    println!("{}", "Documents".bold());
    println!("  Total:    {} files indexed", status.total_documents);
    if status.needs_embedding > 0 {
        println!("  {} {} (run 'qmd embed')", "Pending:".yellow(), format!("{} need embedding", status.needs_embedding));
    }
    if status.collections.is_empty() {
        println!("\n{}", "No collections.".dimmed());
    } else {
        println!("\n{}", "Collections".bold());
        for coll in &status.collections {
            let time_ago = coll.last_modified.as_ref().map_or_else(|| "never".to_string(), |t| format_time_ago(t));
            let coll_contexts: Vec<_> = contexts.iter().filter(|c| c.collection == coll.name).collect();
            println!("  {} {}", coll.name.cyan(), format!("(qmd://{}/)", coll.name).dimmed());
            println!("    {} {}", "Pattern:".dimmed(), coll.glob_pattern);
            println!("    {} {} (updated {})", "Files:".dimmed(), coll.active_count, time_ago);
            if !coll_contexts.is_empty() { println!("    {} {}", "Contexts:".dimmed(), coll_contexts.len()); }
        }
    }
    Ok(())
}

fn handle_update(pull: bool) -> Result<()> {
    let store = Store::new()?;
    store.clear_cache()?;
    let collections = store.list_collections()?;
    if collections.is_empty() {
        println!("{}", "No collections found.".dimmed());
        return Ok(());
    }
    let yaml_collections = yaml_list_collections().unwrap_or_default();
    println!("{}\n", format!("Updating {} collection(s)...", collections.len()).bold());
    for (i, coll) in collections.iter().enumerate() {
        println!("{} {} {}", format!("[{}/{}]", i + 1, collections.len()).cyan(), coll.name.bold(), format!("({})", coll.glob_pattern).dimmed());
        if let Some(yaml_coll) = yaml_collections.iter().find(|c| c.name == coll.name) {
            if let Some(ref update_cmd) = yaml_coll.update {
                println!("    Running update command: {}", update_cmd.dimmed());
                let output = if cfg!(target_os = "windows") {
                    std::process::Command::new("cmd").args(["/C", update_cmd]).current_dir(&coll.pwd).output()
                } else {
                    std::process::Command::new("sh").args(["-c", update_cmd]).current_dir(&coll.pwd).output()
                };
                if let Ok(o) = output { if o.status.success() {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    for line in stdout.lines().take(10) { println!("    {line}"); }
                }}
            }
        }
        if pull {
            let git_dir = Path::new(&coll.pwd).join(".git");
            if git_dir.exists() {
                println!("    Running git pull...");
                let _ = std::process::Command::new("git").arg("pull").current_dir(&coll.pwd).output();
            }
        }
        index_files(&coll.pwd, &coll.glob_pattern, &coll.name)?;
        println!();
    }
    println!("{} All collections updated.", "✓".green());
    Ok(())
}

fn handle_search(query: &str, collection: Option<&str>, limit: usize, min_score: Option<f64>, full: bool, format: &OutputFormat) -> Result<()> {
    let store = Store::new()?;
    let mut results = store.search_fts(query, limit, collection)?;
    if let Some(min) = min_score { results.retain(|r| r.score >= min); }
    if full {
        for result in &mut results {
            if result.doc.body.is_none() { if let Ok(Some(doc)) = store.get_document(&result.doc.collection_name, &result.doc.path) { result.doc.body = doc.body; }}
        }
    }
    println!("{}", format_search_results(&results, format, full));
    Ok(())
}

fn handle_vsearch(query: &str, collection: Option<&str>, limit: usize, min_score: Option<f64>, full: bool, format: &OutputFormat, model_path: Option<&str>) -> Result<()> {
    use qmd::EmbeddingEngine;
    use std::path::PathBuf;
    let store = Store::new()?;
    store.check_and_warn_health();
    let mut engine = if let Some(path) = model_path {
        EmbeddingEngine::new(&PathBuf::from(path))?
    } else if let Ok(e) = EmbeddingEngine::load_default() { e }
    else {
        eprintln!("{} Embedding model not found.", "Error:".red());
        std::process::exit(1);
    };
    println!("Generating query embedding...");
    let query_result = engine.embed_query(query)?;
    let mut results = store.search_vec(&query_result.embedding, limit, collection)?;
    if let Some(min) = min_score { results.retain(|r| r.score >= min); }
    if results.is_empty() { println!("No results found."); return Ok(()); }
    if full {
        for result in &mut results {
            if result.doc.body.is_none() { if let Ok(Some(doc)) = store.get_document(&result.doc.collection_name, &result.doc.path) { result.doc.body = doc.body; }}
        }
    }
    println!("{}", format_search_results(&results, format, full));
    Ok(())
}

fn handle_embed(force: bool, model_path: Option<&str>) -> Result<()> {
    use qmd::{CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, Cursor, EmbeddingEngine, Progress, chunk_document_by_tokens, format_doc_for_embedding, format_eta, render_progress_bar};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Instant;
    let store = Store::new()?;
    if force { let cleared = store.clear_embeddings()?; println!("Cleared {cleared} existing embeddings"); }
    let pending = store.get_hashes_needing_embedding()?;
    if pending.is_empty() { println!("{} All documents already have embeddings.", "✓".green()); return Ok(()); }
    let mut engine = if let Some(path) = model_path { EmbeddingEngine::new(&PathBuf::from(path))? }
    else if let Ok(e) = EmbeddingEngine::load_default() { e }
    else { eprintln!("{} Embedding model not found.", "Error:".red()); std::process::exit(1); };
    eprintln!("Chunking {} documents...", pending.len());
    #[allow(dead_code)]
    struct ChunkItem { hash: String, title: String, text: String, seq: usize, pos: usize, bytes: usize, display_name: String }
    let mut all_chunks: Vec<ChunkItem> = Vec::new();
    for (hash, path, content) in &pending {
        if content.is_empty() { continue; }
        let title = Store::extract_title(content);
        match chunk_document_by_tokens(&engine, content, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS) {
            Ok(chunks) => { for (seq, chunk) in chunks.into_iter().enumerate() {
                all_chunks.push(ChunkItem { hash: hash.clone(), title: title.clone(), text: chunk.text, seq, pos: chunk.pos, bytes: chunk.bytes, display_name: path.clone() });
            }}
            Err(_) => { all_chunks.push(ChunkItem { hash: hash.clone(), title: title.clone(), text: content.clone(), seq: 0, pos: 0, bytes: content.len(), display_name: path.clone() }); }
        }
    }
    if all_chunks.is_empty() { println!("{} No non-empty documents to embed.", "✓".green()); return Ok(()); }
    let total_bytes: usize = all_chunks.iter().map(|c| c.bytes).sum();
    let total_chunks = all_chunks.len();
    let total_docs = pending.len();
    println!("{} {} {}", "Embedding".bold(), format!("{total_docs} documents").bold(), format!("({total_chunks} chunks, {})", format_bytes(total_bytes)).dimmed());
    let progress = Progress::new();
    progress.indeterminate();
    let first_chunk = &all_chunks[0];
    let first_text = format_doc_for_embedding(&first_chunk.text, Some(&first_chunk.title));
    let first_result = engine.embed(&first_text)?;
    let dims = first_result.embedding.len();
    store.ensure_vector_table(dims)?;
    Cursor::hide();
    let now = chrono::Utc::now().to_rfc3339();
    let start_time = Instant::now();
    let mut chunks_embedded = 0usize;
    let mut errors = 0usize;
    let mut bytes_processed = 0usize;
    store.insert_embedding(&first_chunk.hash, first_chunk.seq, first_chunk.pos, &first_result.embedding, &first_result.model, &now)?;
    chunks_embedded += 1;
    bytes_processed += first_chunk.bytes;
    for chunk in all_chunks.iter().skip(1) {
        let formatted = format_doc_for_embedding(&chunk.text, Some(&chunk.title));
        match engine.embed(&formatted) {
            Ok(result) => { store.insert_embedding(&chunk.hash, chunk.seq, chunk.pos, &result.embedding, &result.model, &now)?; chunks_embedded += 1; }
            Err(e) => { errors += 1; eprintln!("\n{} Error embedding: {}", "⚠".yellow(), e); }
        }
        bytes_processed += chunk.bytes;
        let percent = (bytes_processed as f64 / total_bytes as f64) * 100.0;
        progress.set(percent);
        let elapsed = start_time.elapsed().as_secs_f64();
        let bytes_per_sec = bytes_processed as f64 / elapsed;
        let remaining_bytes = total_bytes.saturating_sub(bytes_processed);
        let eta_sec = remaining_bytes as f64 / bytes_per_sec;
        let bar = render_progress_bar(percent, 20);
        let eta = if elapsed > 2.0 { format_eta(eta_sec) } else { "...".to_string() };
        eprint!("\r{} {:3.0}% {}/{} {} ETA {}   ", bar.cyan(), percent, chunks_embedded, total_chunks, format!("{}/s", format_bytes(bytes_per_sec as usize)).dimmed(), eta.dimmed());
        std::io::stderr().flush().ok();
    }
    progress.clear();
    Cursor::show();
    let total_time_sec = start_time.elapsed().as_secs_f64();
    println!("\r{} {}                                    ", render_progress_bar(100.0, 20).green(), "100%".bold());
    println!("\n{} Embedded {} chunks from {} documents in {}", "✓".green(), chunks_embedded.to_string().bold(), total_docs.to_string().bold(), format_eta(total_time_sec).bold());
    if errors > 0 { println!("{} {} chunks failed", "⚠".yellow(), errors); }
    Ok(())
}

fn handle_models(cmd: ModelCommands) -> Result<()> {
    use qmd::llm::{DEFAULT_EMBED_MODEL, list_cached_models};
    match cmd {
        ModelCommands::List => {
            let models = list_cached_models();
            let cache_dir = qmd::config::get_model_cache_dir();
            println!("{}\n", "Available Models".bold());
            println!("Cache directory: {}\n", cache_dir.display());
            if models.is_empty() { println!("No models found in cache."); }
            else { println!("{}", "Cached models:".cyan()); for model in &models { println!("  {model}"); }}
        }
        ModelCommands::Info { name } => {
            let model_name = name.as_deref().unwrap_or(DEFAULT_EMBED_MODEL);
            let cache_dir = qmd::config::get_model_cache_dir();
            let model_path = cache_dir.join(model_name);
            println!("{}\n", "Model Info".bold());
            println!("Name: {model_name}");
            println!("Path: {}", model_path.display());
            if model_path.exists() { println!("Status: {}", "Downloaded".green()); }
            else { println!("Status: {}", "Not downloaded".red()); }
        }
        ModelCommands::Pull { model, refresh } => {
            use qmd::{pull_model, pull_models, llm::{DEFAULT_EMBED_MODEL_URI, DEFAULT_RERANK_MODEL_URI}};
            println!("{}\n", "Pulling Models".bold());
            let results = if model == "all" {
                let default_models = [DEFAULT_EMBED_MODEL_URI, DEFAULT_RERANK_MODEL_URI];
                println!("Downloading {} default models...\n", default_models.len());
                pull_models(&default_models, refresh)?
            } else { vec![pull_model(&model, refresh)?] };
            println!();
            for result in &results {
                let status = if result.refreshed { "Downloaded".green() } else { "Cached".cyan() };
                println!("{} {} ({})", status, result.path.file_name().unwrap_or_default().to_string_lossy(), format_bytes(result.size_bytes as usize));
            }
            println!("\n{} {} model(s) ready", "✓".green(), results.len());
        }
    }
    Ok(())
}

fn handle_db(cmd: DbCommands) -> Result<()> {
    let store = Store::new()?;
    match cmd {
        DbCommands::Cleanup => {
            let inactive = store.delete_inactive_documents()?;
            let orphaned_content = store.cleanup_orphaned_content()?;
            let orphaned_vectors = store.cleanup_orphaned_vectors()?;
            println!("{} Database cleanup complete", "✓".green());
            println!("  Removed {inactive} inactive documents");
            println!("  Removed {orphaned_content} orphaned content entries");
            println!("  Removed {orphaned_vectors} orphaned vector entries");
        }
        DbCommands::Vacuum => { println!("Vacuuming database..."); store.vacuum()?; println!("{} Database vacuumed", "✓".green()); }
        DbCommands::ClearCache => { let cleared = store.clear_cache()?; println!("{} Cleared {} cached entries", "✓".green(), cleared); }
    }
    Ok(())
}

fn handle_qsearch(query: &str, collection: Option<&str>, limit: usize, full: bool, no_expand: bool, no_rerank: bool, format: &OutputFormat) -> Result<()> {
    use qmd::{EmbeddingEngine, GenerationEngine, RerankDocument, RerankEngine};
    let store = Store::new()?;
    store.check_and_warn_health();
    let queries = if no_expand || !GenerationEngine::is_available() {
        vec![qmd::Queryable::lex(query), qmd::Queryable::vec(query)]
    } else {
        println!("Expanding query...");
        match GenerationEngine::load_default() {
            Ok(engine) => match engine.expand_query(query, true) { Ok(q) => q, Err(_) => qmd::expand_query_simple(query) },
            Err(_) => qmd::expand_query_simple(query),
        }
    };
    let mut fts_results: Vec<(String, String, String, String)> = Vec::new();
    let mut vec_results: Vec<(String, String, String, String)> = Vec::new();
    for q in &queries {
        match q.query_type {
            qmd::QueryType::Lex => {
                if let Ok(results) = store.search_fts(&q.text, limit * 2, collection) {
                    for r in results { fts_results.push((r.doc.filepath.clone(), r.doc.display_path.clone(), r.doc.title.clone(), r.doc.body.clone().unwrap_or_default())); }
                }
            }
            qmd::QueryType::Vec | qmd::QueryType::Hyde => {
                if let Ok(mut engine) = EmbeddingEngine::load_default() {
                    if let Ok(query_result) = engine.embed_query(&q.text) {
                        if let Ok(results) = store.search_vec(&query_result.embedding, limit * 2, collection) {
                            for r in results {
                                let body = store.get_document(&r.doc.collection_name, &r.doc.path).ok().flatten().and_then(|d| d.body).unwrap_or_default();
                                vec_results.push((r.doc.filepath.clone(), r.doc.display_path.clone(), r.doc.title.clone(), body));
                            }
                        }
                    }
                }
            }
        }
    }
    let mut rrf_results = qmd::hybrid_search_rrf(fts_results, vec_results, 60);
    if !no_rerank && RerankEngine::is_available() && !rrf_results.is_empty() {
        println!("Reranking {} results...", rrf_results.len().min(limit * 2));
        if let Ok(mut reranker) = RerankEngine::load_default() {
            let docs: Vec<RerankDocument> = rrf_results.iter().take(limit * 2).map(|r| RerankDocument { file: r.file.clone(), text: r.body.clone(), title: Some(r.title.clone()) }).collect();
            if let Ok(reranked) = reranker.rerank(query, &docs) {
                let mut reordered = Vec::new();
                for rr in reranked.results { if let Some(orig) = rrf_results.iter().find(|r| r.file == rr.file) { reordered.push(orig.clone()); }}
                rrf_results = reordered;
            }
        }
    }
    rrf_results.truncate(limit);
    if rrf_results.is_empty() { println!("{}", "No results found.".dimmed()); return Ok(()); }
    let search_results: Vec<qmd::SearchResult> = rrf_results.iter().map(|r| {
        let parts: Vec<&str> = r.file.strip_prefix("qmd://").unwrap_or(&r.file).splitn(2, '/').collect();
        let (collection_name, path) = if parts.len() == 2 { (parts[0].to_string(), parts[1].to_string()) } else { (String::new(), r.file.clone()) };
        qmd::SearchResult { doc: qmd::DocumentResult { filepath: r.file.clone(), display_path: r.display_path.clone(), title: r.title.clone(), context: None, hash: String::new(), docid: String::new(), collection_name, path, modified_at: String::new(), body_length: r.body.len(), body: if full { Some(r.body.clone()) } else { None } }, score: r.score, source: qmd::SearchSource::Fts, chunk_pos: None }
    }).collect();
    println!("{}", format_search_results(&search_results, format, full));
    Ok(())
}

fn handle_expand(query: &str, include_lexical: bool) -> Result<()> {
    use qmd::GenerationEngine;
    println!("{}\n", "Query Expansion".bold());
    println!("Original: {query}\n");
    let queries = if GenerationEngine::is_available() {
        match GenerationEngine::load_default() {
            Ok(engine) => match engine.expand_query(query, include_lexical) { Ok(q) => q, Err(_) => qmd::expand_query_simple(query) },
            Err(_) => qmd::expand_query_simple(query),
        }
    } else { qmd::expand_query_simple(query) };
    println!("{}", "Expanded queries:".cyan());
    for q in &queries {
        let type_str = match q.query_type { qmd::QueryType::Lex => "lex".green(), qmd::QueryType::Vec => "vec".blue(), qmd::QueryType::Hyde => "hyde".magenta() };
        println!("  {}: {}", type_str, q.text);
    }
    Ok(())
}

fn handle_rerank(query: &str, files: &str, limit: usize, format: &OutputFormat) -> Result<()> {
    use qmd::{RerankDocument, RerankEngine};
    let store = Store::new()?;
    let file_list: Vec<&str> = files.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    if file_list.is_empty() { eprintln!("{} No files specified", "Error:".red()); std::process::exit(1); }
    let mut docs: Vec<RerankDocument> = Vec::new();
    for file in &file_list {
        let (collection, path) = if is_virtual_path(file) { parse_virtual_path(file).unwrap_or((String::new(), file.to_string())) }
        else { let parts: Vec<&str> = file.splitn(2, '/').collect(); if parts.len() == 2 { (parts[0].to_string(), parts[1].to_string()) } else { continue; }};
        if let Ok(Some(doc)) = store.get_document(&collection, &path) { docs.push(RerankDocument { file: doc.filepath.clone(), text: doc.body.unwrap_or_default(), title: Some(doc.title) }); }
    }
    if docs.is_empty() { eprintln!("{} No valid documents found", "Error:".red()); std::process::exit(1); }
    println!("Reranking {} documents...", docs.len());
    let mut engine = RerankEngine::load_default().map_err(|e| { eprintln!("{} Could not load rerank model: {}", "Error:".red(), e); std::process::exit(1); })?;
    let result = engine.rerank(query, &docs)?;
    match format {
        OutputFormat::Json => {
            let output: Vec<serde_json::Value> = result.results.iter().take(limit).map(|r| serde_json::json!({"file": r.file, "score": r.score, "rank": r.index + 1})).collect();
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        _ => { println!("\n{}", "Reranked Results:".bold()); for (i, r) in result.results.iter().take(limit).enumerate() { println!("{}. {} {}", (i + 1).to_string().cyan(), format!("{:.4}", r.score).dimmed(), r.file.bold()); }}
    }
    Ok(())
}

fn handle_ask(question: &str, collection: Option<&str>, limit: usize, max_tokens: usize) -> Result<()> {
    use qmd::{EmbeddingEngine, GenerationEngine};
    let store = Store::new()?;
    println!("{}", "Searching for relevant documents...".dimmed());
    let context_docs = if let Ok(mut engine) = EmbeddingEngine::load_default() {
        if let Ok(query_result) = engine.embed_query(question) { store.search_vec(&query_result.embedding, limit, collection).unwrap_or_default() }
        else { store.search_fts(question, limit, collection).unwrap_or_default() }
    } else { store.search_fts(question, limit, collection).unwrap_or_default() };
    if context_docs.is_empty() { println!("{}", "No relevant documents found.".yellow()); return Ok(()); }
    let mut context = String::new();
    for (i, result) in context_docs.iter().enumerate() {
        let body = store.get_document(&result.doc.collection_name, &result.doc.path).ok().flatten().and_then(|d| d.body).unwrap_or_default();
        let truncated: String = body.chars().take(1000).collect();
        context.push_str(&format!("\n--- Document {} ({}): ---\n{}\n", i + 1, result.doc.display_path, truncated));
    }
    println!("Found {} relevant documents. Generating answer...\n", context_docs.len());
    let gen_engine = GenerationEngine::load_default().map_err(|e| { eprintln!("{} Could not load generation model: {}", "Error:".red(), e); std::process::exit(1); })?;
    let prompt = format!("Based on the following documents, answer the question concisely.\n\nDocuments:\n{context}\n\nQuestion: {question}\n\nAnswer:");
    let result = gen_engine.generate(&prompt, max_tokens)?;
    println!("{}\n", "Answer:".green().bold());
    println!("{}", result.text);
    println!("\n{}", "Sources:".dimmed());
    for result in &context_docs { println!("  - {}", result.doc.display_path); }
    Ok(())
}

fn handle_index(name: Option<&str>) -> Result<()> {
    use qmd::collections::set_config_index_name;
    match name {
        Some(index_name) => {
            set_config_index_name(index_name);
            let db_path = qmd::config::get_default_db_path(index_name).unwrap_or_else(|| std::path::PathBuf::from("unknown"));
            println!("{} Switched to index: {}", "✓".green(), index_name.cyan());
            println!("  Database: {}", db_path.display());
        }
        None => {
            let default_path = qmd::config::get_default_db_path("index").unwrap_or_else(|| std::path::PathBuf::from("unknown"));
            println!("{}", "Current Index".bold());
            println!("  Name: {}", "index".cyan());
            println!("  Path: {}", default_path.display());
        }
    }
    Ok(())
}

fn index_files(pwd: &str, glob_pattern: &str, collection_name: &str) -> Result<()> {
    let store = Store::new()?;
    let now = chrono::Utc::now().to_rfc3339();
    let glob_matcher = glob::Pattern::new(glob_pattern)?;
    let mut files = Vec::new();
    for entry in WalkDir::new(pwd).follow_links(true).into_iter().filter_map(std::result::Result::ok) {
        let path = entry.path();
        if !path.is_file() { continue; }
        if should_exclude(path) { continue; }
        let rel_path = path.strip_prefix(pwd).unwrap_or(path);
        let rel_path_str = rel_path.to_string_lossy();
        if glob_matcher.matches(&rel_path_str) { files.push((path.to_path_buf(), rel_path_str.to_string())); }
    }
    if files.is_empty() { println!("  No files found matching pattern."); return Ok(()); }
    let mut indexed = 0; let mut updated = 0; let mut unchanged = 0;
    let mut seen_paths = HashSet::new();
    for (abs_path, rel_path) in &files {
        let normalized_path = Store::handelize(rel_path);
        seen_paths.insert(normalized_path.clone());
        let content = match fs::read_to_string(abs_path) { Ok(c) => c, Err(_) => continue };
        let hash = Store::hash_content(&content);
        let title = Store::extract_title(&content);
        if let Some((doc_id, existing_hash, existing_title)) = store.find_active_document(collection_name, &normalized_path)? {
            if existing_hash == hash { if existing_title != title { store.update_document_title(doc_id, &title, &now)?; } unchanged += 1; }
            else { store.insert_content(&hash, &content, &now)?; store.update_document(doc_id, &title, &hash, &now)?; updated += 1; }
        } else { store.insert_content(&hash, &content, &now)?; store.insert_document(collection_name, &normalized_path, &title, &hash, &now, &now)?; indexed += 1; }
    }
    let existing_paths = store.get_active_document_paths(collection_name)?;
    let mut deactivated = 0;
    for path in existing_paths { if !seen_paths.contains(&path) { store.deactivate_document(collection_name, &path)?; deactivated += 1; }}
    println!("  {indexed} indexed, {updated} updated, {unchanged} unchanged, {deactivated} removed");
    Ok(())
}
