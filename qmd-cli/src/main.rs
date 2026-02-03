//! QMD CLI - Command-line interface for qmd search engine.

mod cli;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, CollectionCommands, Commands, ContextCommands, DbCommands, ModelCommands};
use colored::Colorize;
use qmd::{
    OutputFormat, Store, add_collection as yaml_add_collection, add_context, add_line_numbers,
    format_bytes, format_documents, format_ls_time, format_search_results, format_time_ago,
    get_collection, is_docid, is_virtual_path, list_all_contexts,
    list_collections as yaml_list_collections, match_files_by_glob, parse_virtual_path,
    remove_collection as yaml_remove_collection, remove_context,
    rename_collection as yaml_rename_collection, set_global_context, should_exclude,
};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

fn main() -> Result<()> {
    let cli = Cli::parse();
    dispatch_command(cli.command)
}

fn dispatch_command(cmd: Commands) -> Result<()> {
    match cmd {
        Commands::Collection(c) => handle_collection(c),
        Commands::Context(c) => handle_context(c),
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
        } => handle_multi_get(&pattern, max_lines, max_bytes, &format.into()),
        Commands::Status => handle_status(),
        Commands::Update { pull } => handle_update(pull),
        Commands::Search {
            query,
            collection,
            limit,
            min_score,
            full,
            format,
            ..
        } => handle_search(
            &query,
            collection.as_deref(),
            limit,
            min_score,
            full,
            &format.into(),
        ),
        Commands::Vsearch {
            query,
            collection,
            limit,
            min_score,
            full,
            format,
            model,
            ..
        } => handle_vsearch(
            &query,
            collection.as_deref(),
            limit,
            min_score,
            full,
            &format.into(),
            model.as_deref(),
        ),
        Commands::Embed { force, model } => handle_embed(force, model.as_deref()),
        Commands::Models(c) => handle_models(c),
        Commands::Db(c) => handle_db(c),
        Commands::Qsearch {
            query,
            collection,
            limit,
            full,
            no_expand,
            no_rerank,
            format,
        } => handle_qsearch(
            &query,
            collection.as_deref(),
            limit,
            full,
            no_expand,
            no_rerank,
            &format.into(),
        ),
        Commands::Expand { query, lexical } => handle_expand(&query, lexical),
        Commands::Rerank {
            query,
            files,
            limit,
            format,
        } => handle_rerank(&query, &files, limit, &format.into()),
        Commands::Ask {
            question,
            collection,
            limit,
            max_tokens,
        } => handle_ask(&question, collection.as_deref(), limit, max_tokens),
        Commands::Index { name } => handle_index(name.as_deref()),
        Commands::Cleanup => handle_cleanup(),
    }
}

// Include command handlers from separate module
include!("handlers.rs");
