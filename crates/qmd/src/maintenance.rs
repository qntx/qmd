//! `qmd cleanup` maintenance operations — upstream `src/maintenance.ts`.
//!
//! [`Maintenance`] borrows the [`Qmd`] handle; every method
//! locks the connection for the duration of its own work.

use crate::Qmd;
use crate::error::Result;
pub use crate::store::maintenance::CleanupCounts;
use crate::store::{documents, maintenance as store};

/// Maintenance operations on a [`Qmd`] handle — upstream
/// `Maintenance` class (maintenance.ts:22-76).
pub struct Maintenance<'q> {
    pub(crate) qmd: &'q Qmd,
}

impl Maintenance<'_> {
    /// `VACUUM` — rebuild the database file, reclaiming space
    /// (upstream `vacuum`, maintenance.ts:30).
    ///
    /// # Errors
    /// [`crate::Error::Db`] on SQLite failure.
    pub fn vacuum(&self) -> Result<()> {
        store::vacuum_database(&self.qmd.lock_conn())
    }

    /// Delete `content` rows no document references
    /// (upstream `cleanupOrphanedContent`).
    ///
    /// # Errors
    /// [`crate::Error::Db`] on SQLite failure.
    pub fn cleanup_orphaned_content(&self) -> Result<usize> {
        documents::cleanup_orphaned_content(&self.qmd.lock_conn())
    }

    /// Delete `content_vectors`/`vectors_vec` rows whose hash no active
    /// document references (upstream `cleanupOrphanedVectors`; atomic).
    ///
    /// # Errors
    /// [`crate::Error::Db`] on SQLite failure.
    pub fn cleanup_orphaned_vectors(&self) -> Result<usize> {
        store::cleanup_orphaned_vectors(&mut self.qmd.lock_conn())
    }

    /// Clear the LLM response cache; returns rows removed
    /// (upstream `clearLLMCache` → `deleteLLMCache`).
    ///
    /// # Errors
    /// [`crate::Error::Db`] on SQLite failure.
    pub fn clear_llm_cache(&self) -> Result<usize> {
        store::delete_llm_cache(&self.qmd.lock_conn())
    }

    /// Hard-delete documents marked inactive
    /// (upstream `deleteInactiveDocs` → `deleteInactiveDocuments`).
    ///
    /// # Errors
    /// [`crate::Error::Db`] on SQLite failure.
    pub fn delete_inactive_documents(&self) -> Result<usize> {
        store::delete_inactive_documents(&self.qmd.lock_conn())
    }

    /// Clear all vector embeddings — `content_vectors` emptied,
    /// `vectors_vec` dropped (upstream `clearEmbeddings` →
    /// `clearAllEmbeddings` whole-index variant).
    ///
    /// # Errors
    /// [`crate::Error::Db`] on SQLite failure.
    pub fn clear_embeddings(&self) -> Result<()> {
        store::clear_all_embeddings(&self.qmd.lock_conn())
    }

    /// Compact the FTS5 index so deleted rows leave
    /// `documents_fts_data` (upstream `optimizeFts`, #550).
    ///
    /// # Errors
    /// [`crate::Error::Db`] on SQLite failure.
    pub fn optimize_fts(&self) -> Result<()> {
        store::optimize_documents_fts(&self.qmd.lock_conn())
    }

    /// What [`run`](Self::run) would remove, without writing
    /// (upstream `preview` → `previewCleanup`).
    ///
    /// # Errors
    /// [`crate::Error::Db`] on SQLite failure.
    pub fn preview(&self) -> Result<CleanupCounts> {
        store::preview_cleanup(&self.qmd.lock_conn())
    }

    /// Full cleanup sequence in upstream order (maintenance.ts:73,
    /// store.ts:2888-2896): LLM cache → orphaned vectors → inactive
    /// documents → orphaned content → FTS optimize → VACUUM.
    ///
    /// # Errors
    /// [`crate::Error::Db`] on SQLite failure; the steps are not wrapped
    /// in a single transaction, same as upstream.
    pub fn run(&self) -> Result<CleanupCounts> {
        store::run_cleanup(&mut self.qmd.lock_conn())
    }
}

impl std::fmt::Debug for Maintenance<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Maintenance").finish_non_exhaustive()
    }
}
