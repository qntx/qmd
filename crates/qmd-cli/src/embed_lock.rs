//! Per-database embed lock — upstream `embed-lock.ts` (P4-D4).
//!
//! Upstream keeps a *directory*-scoped `.qmd-embed.lock` PID file and
//! recovers stale locks by probing the recorded PID
//! (`isLiveEmbedLockHolder`). This port deliberately deviates: the lock
//! is the OS file lock on `<db_path>.embed.lock` —
//!
//! - **per database**, so sibling databases in one directory
//!   (`index.sqlite` and `index-rs.sqlite`) do not block each other,
//!   and
//! - **OS-held**, so a killed process releases the lock automatically —
//!   there is no stale-PID recovery to port (`flock`/`LockFile` never
//!   outlives the holder).
//!
//! The lock file itself is best-effort deleted on drop; a leftover
//! file from a `kill -9` is harmless — the next acquirer truncates it
//! and locks anew.

use std::fs::{File, OpenOptions, TryLockError};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Message for a skipped second embed — upstream
/// `EMBED_LOCK_BUSY_MESSAGE` (embed-lock.ts:100-101), printed by the
/// `embed` command (stage B) to stdout like upstream's `console.log`.
pub(crate) const BUSY_MESSAGE: &str = "Another embed process is already running. Skipping.";

/// A held embed lock — the OS lock releases and the lock file is
/// best-effort removed on `Drop` (upstream `release()`,
/// embed-lock.ts:46-58).
#[derive(Debug)]
pub(crate) struct EmbedLock {
    /// `<db_path>.embed.lock`.
    path: PathBuf,
    /// Held handle — the OS lock is released when this closes.
    file: File,
}

impl EmbedLock {
    /// The lock file for `db_path` — `<db_path>.embed.lock`
    /// (deviation: upstream's `embedLockPathForDb` is the
    /// directory-level `dirname(db)/.qmd-embed.lock`, embed-lock.ts:20-22).
    #[must_use]
    pub(crate) fn path_for_db(db_path: &Path) -> PathBuf {
        let mut name = db_path.file_name().map_or_else(
            || "index.sqlite".to_owned(),
            |n| n.to_string_lossy().into_owned(),
        );
        name.push_str(".embed.lock");
        db_path.with_file_name(name)
    }

    /// Upstream `tryAcquireEmbedLock` (embed-lock.ts:65-97): `Ok(None)`
    /// while another live process holds the lock; real I/O failures are
    /// `Err` like upstream's throw (a `None` would wrongly report
    /// "already running").
    ///
    /// # Errors
    /// [`std::io::Error`] creating the lock file, or from
    /// [`File::try_lock`] for reasons other than contention.
    pub(crate) fn try_acquire(db_path: &Path) -> std::io::Result<Option<Self>> {
        let path = Self::path_for_db(db_path);
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        match file.try_lock() {
            Ok(()) => {
                // The PID contents are debug aid only — the OS lock, not
                // the file's bytes, is the mutual exclusion.
                let _ = writeln!(file, "{}", std::process::id());
                Ok(Some(Self { path, file }))
            }
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Error(err)) => Err(err),
        }
    }
}

impl Drop for EmbedLock {
    fn drop(&mut self) {
        // Unlock before unlinking: a racing acquirer that wins the
        // create-after-unlink race must not find itself locking a
        // doomed inode while we still hold the real lock.
        let _ = self.file.unlock();
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::missing_assert_message,
    reason = "idiomatic test-code patterns"
)]
mod tests {
    use super::*;

    #[test]
    fn lock_path_is_sibling_of_db() {
        assert_eq!(
            EmbedLock::path_for_db(Path::new("/x/index-rs.sqlite")),
            Path::new("/x/index-rs.sqlite.embed.lock")
        );
    }

    /// flock/`LockFile` conflicts across open file descriptions even
    /// within one process, so a second acquire observes the busy lock.
    #[test]
    fn busy_then_reacquire() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index-rs.sqlite");
        std::fs::write(&db, b"").unwrap();

        let lock = EmbedLock::try_acquire(&db).unwrap().unwrap();
        assert!(EmbedLock::try_acquire(&db).unwrap().is_none());
        drop(lock);
        assert!(EmbedLock::try_acquire(&db).unwrap().is_some());
    }

    /// Sibling databases get independent locks (the deviation reason —
    /// upstream's directory-level lockfile would block this).
    #[test]
    fn sibling_databases_do_not_block() {
        let dir = tempfile::tempdir().unwrap();
        let db_a = dir.path().join("index.sqlite");
        let db_b = dir.path().join("index-rs.sqlite");
        let _a = EmbedLock::try_acquire(&db_a).unwrap().unwrap();
        assert!(EmbedLock::try_acquire(&db_b).unwrap().is_some());
    }
}
