//! sqlite-vec registration.
//!
//! This is the only module in the crate allowed to use `unsafe`, needed to
//! declare and register the C extension entry point. Registration is
//! process-global and performed once via [`OnceLock`]; it must run before
//! the first connection is opened, which [`crate::store::open`] guarantees.

#[allow(
    unsafe_code,
    reason = "FFI declaration and registration of the sqlite-vec entry point"
)]
mod imp {
    use std::ffi::{c_char, c_int};
    use std::sync::OnceLock;

    use rusqlite::ffi;
    // Linking dependency: pulls in the static sqlite-vec object code.
    use sqlite_vec as _;

    unsafe extern "C" {
        /// sqlite-vec extension initialization entry point, with the
        /// standard `sqlite3_api` signature expected by
        /// `sqlite3_auto_extension`.
        fn sqlite3_vec_init(
            db: *mut ffi::sqlite3,
            pz_err_msg: *mut *mut c_char,
            p_api: *const ffi::sqlite3_api_routines,
        ) -> c_int;
    }

    /// Register `sqlite3_vec_init` as an auto-extension once per process.
    ///
    /// # Errors
    /// Returns the registration error reported by SQLite, mapped into
    /// [`rusqlite::Error`].
    pub(crate) fn register() -> rusqlite::Result<()> {
        static RESULT: OnceLock<Result<(), ffi::Error>> = OnceLock::new();
        let outcome = RESULT.get_or_init(|| {
            // SAFETY: `sqlite3_vec_init` has the exact signature required
            // by `sqlite3_auto_extension`; the symbols come from the
            // statically linked `sqlite-vec` crate.
            unsafe { rusqlite::auto_extension::register_auto_extension(sqlite3_vec_init) }.map_err(
                |e| match e {
                    rusqlite::Error::SqliteFailure(code, _) => code,
                    // Registration can only fail with a SQLite result code;
                    // other variants are unreachable in practice.
                    _ => ffi::Error::new(ffi::SQLITE_ERROR),
                },
            )
        });
        match outcome {
            Ok(()) => Ok(()),
            Err(code) => Err(rusqlite::Error::SqliteFailure(*code, None)),
        }
    }
}

pub(crate) use imp::register;
