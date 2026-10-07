//! Collection filesystem scanning and incremental indexing.

pub(crate) mod index;
mod scan;

pub use scan::split_glob_mask;
