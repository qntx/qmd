//! Command-line interface definitions.

use clap::{Parser, Subcommand, ValueEnum};

/// Query Markdown Documents - Full-text search for markdown files.
#[derive(Parser, Debug)]
#[command(name = "qmd")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Commands,
}

/// Available commands.
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Manage collections.
    #[command(subcommand)]
    Collection(CollectionCommands),

    /// Manage context descriptions.
    #[command(subcommand)]
    Context(ContextCommands),

    /// List collections or files in a collection.
    Ls {
        /// Collection name or path (e.g., "notes" or "<qmd://notes/path>").
        path: Option<String>,
    },

    /// Get a document by path or docid.
    Get {
        /// File path, virtual path (qmd://), or docid (#abc123).
        file: String,

        /// Starting line number.
        #[arg(short = 'f', long)]
        from_line: Option<usize>,

        /// Maximum lines to show.
        #[arg(short = 'l', long)]
        max_lines: Option<usize>,

        /// Show line numbers.
        #[arg(short = 'n', long)]
        line_numbers: bool,
    },

    /// Get multiple documents by glob pattern or comma-separated list.
    MultiGet {
        /// Glob pattern or comma-separated list of files.
        pattern: String,

        /// Maximum lines per file.
        #[arg(short = 'l', long)]
        max_lines: Option<usize>,

        /// Skip files larger than this (bytes, default 10KB).
        #[arg(long, default_value = "10240")]
        max_bytes: usize,

        /// Output format.
        #[arg(long, value_enum, default_value = "cli")]
        format: OutputFormat,
    },

    /// Show index status and collections.
    Status,

    /// Re-index all collections.
    Update {
        /// Run git pull first in each collection.
        #[arg(long)]
        pull: bool,
    },

    /// BM25 full-text search.
    Search {
        /// Search query.
        query: String,

        /// Restrict to a collection.
        #[arg(short, long)]
        collection: Option<String>,

        /// Number of results.
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,

        /// Minimum score threshold.
        #[arg(long)]
        min_score: Option<f64>,

        /// Show full document content.
        #[arg(long)]
        full: bool,

        /// Add line numbers to output.
        #[arg(long)]
        line_numbers: bool,

        /// Output format.
        #[arg(long, value_enum, default_value = "cli")]
        format: OutputFormat,
    },

    /// Vector semantic search.
    Vsearch {
        /// Search query.
        query: String,

        /// Restrict to a collection.
        #[arg(short, long)]
        collection: Option<String>,

        /// Number of results.
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,

        /// Minimum score threshold.
        #[arg(long)]
        min_score: Option<f64>,

        /// Show full document content.
        #[arg(long)]
        full: bool,

        /// Add line numbers to output.
        #[arg(long)]
        line_numbers: bool,

        /// Output format.
        #[arg(long, value_enum, default_value = "cli")]
        format: OutputFormat,

        /// Embedding model path.
        #[arg(long)]
        model: Option<String>,
    },

    /// Generate embeddings for all documents.
    Embed {
        /// Force re-embedding of all documents.
        #[arg(long)]
        force: bool,

        /// Model path (GGUF file).
        #[arg(long)]
        model: Option<String>,
    },

    /// Model management commands.
    #[command(subcommand)]
    Models(ModelCommands),

    /// Database maintenance commands.
    #[command(subcommand)]
    Db(DbCommands),
}

/// Model management commands.
#[derive(Subcommand, Debug)]
pub enum ModelCommands {
    /// List available models.
    List,

    /// Show model info and download status.
    Info {
        /// Model name.
        name: Option<String>,
    },

    /// Download models from `HuggingFace`.
    Pull {
        /// Model URI (e.g., "hf:user/repo/file.gguf") or "all" for default models.
        #[arg(default_value = "all")]
        model: String,

        /// Force re-download even if cached.
        #[arg(short, long)]
        refresh: bool,
    },
}

/// Collection management commands.
#[derive(Subcommand, Debug)]
pub enum CollectionCommands {
    /// Add a new collection.
    Add {
        /// Directory path to index.
        path: String,

        /// Collection name (defaults to directory name).
        #[arg(short, long)]
        name: Option<String>,

        /// Glob pattern for files (default: **/*.md).
        #[arg(short, long, default_value = "**/*.md")]
        mask: String,
    },

    /// List all collections.
    List,

    /// Remove a collection.
    Remove {
        /// Collection name to remove.
        name: String,
    },

    /// Rename a collection.
    Rename {
        /// Old collection name.
        old_name: String,

        /// New collection name.
        new_name: String,
    },
}

/// Context management commands.
#[derive(Subcommand, Debug)]
pub enum ContextCommands {
    /// Add context for a path.
    Add {
        /// Path (filesystem, virtual, or "/" for global).
        path: Option<String>,

        /// Context description.
        text: String,
    },

    /// List all contexts.
    List,

    /// Check for paths missing context.
    Check,

    /// Remove context for a path.
    Rm {
        /// Path to remove context from.
        path: String,
    },
}

/// Database maintenance commands.
#[derive(Subcommand, Debug, Clone, Copy)]
pub enum DbCommands {
    /// Clean up orphaned content and vectors.
    Cleanup,

    /// Vacuum the database.
    Vacuum,

    /// Clear LLM cache.
    ClearCache,
}

/// Output format options.
#[derive(ValueEnum, Clone, Copy, Debug, Default)]
pub enum OutputFormat {
    /// CLI-friendly output.
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
    /// Just file paths.
    Files,
}
