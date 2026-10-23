//! `qmd ls` — upstream `case "ls"` (qmd.ts:4508-4511) and `listFiles`
//! (qmd.ts:1606-1764).
//!
//! No argument prints the `Collections:` block from the *config* with
//! active-document counts from the database; an argument resolves to a
//! collection plus optional path prefix through upstream's four
//! branches (absolute-path collection `qmd:///…`, normal `qmd://`, raw
//! absolute path, `name[/prefix]`).

use std::collections::HashMap;

use qmd::vpath::parse_virtual_path;
use qmd::{Environment, Qmd, js};
use time::format_description::BorrowedFormatItem;
use time::macros::format_description;
use time::{OffsetDateTime, UtcOffset};

use crate::Failure;
use crate::args::LsArgs;
use crate::context::CliContext;
use crate::output::{Io, Palette};

/// `modified_at` is `YYYY-MM-DDTHH:MM:SS.sssZ` (upstream `toISOString`).
const ISO: &[BorrowedFormatItem<'_>] =
    format_description!("[year]-[month]-[day]T[hour]:[minute]:[second].[subsecond digits:3]Z");

/// `qmd ls` entry.
///
/// # Errors
/// [`Failure::Reported`] after writing the upstream message;
/// [`Failure::Error`] wraps store/open failures.
pub(crate) fn run(
    env: &Environment,
    ctx: &CliContext,
    args: &LsArgs,
    io: &mut Io<'_>,
) -> Result<(), Failure> {
    // Upstream opens the db first (`const db = getDb()`).
    let qmd = ctx.open_store(env)?;
    let palette = Palette::new(Palette::enabled(env, io));
    let config = qmd.config()?;

    let Some(arg) = args.path.first() else {
        return list_collections(&qmd, &config, io, palette);
    };

    // The four parse branches (qmd.ts:1638-1693).
    let (collection_name, path_prefix) = parse_path_arg(arg, &config).inspect_err(|_| {
        let _ = writeln!(io.err, "Invalid virtual path: {arg}");
    })?;

    if !config.collections.contains_key(&collection_name) {
        let _ = writeln!(io.err, "Collection not found: {collection_name}");
        let _ = writeln!(io.err, "Run 'qmd ls' to see available collections.");
        return Err(Failure::Reported(1));
    }

    let files = match qmd.list_documents(&collection_name, path_prefix.as_deref()) {
        Ok(files) => files,
        // Upstream queries `documents` directly — a configured collection
        // with no index rows prints the empty message, not an error.
        Err(qmd::Error::CollectionNotFound { .. }) => Vec::new(),
        Err(e) => return Err(Failure::Error(e)),
    };

    if files.is_empty() {
        if let Some(prefix) = &path_prefix {
            let _ = writeln!(
                io.out,
                "No files found under qmd://{collection_name}/{prefix}"
            );
        } else {
            let _ = writeln!(io.out, "No files found in collection: {collection_name}");
        }
        return Ok(());
    }

    let max_size = files
        .iter()
        .map(|f| format_bytes(f.size).len())
        .max()
        .unwrap_or(0);
    let now = OffsetDateTime::now_utc();
    for file in &files {
        let size_str = format!("{:>width$}", format_bytes(file.size), width = max_size);
        let time_str = format_ls_time(&file.modified_at, now, io.local_offset);
        let _ = writeln!(
            io.out,
            "{size_str}  {time_str}  {}qmd://{collection_name}/{}{}{}{}",
            palette.dim, palette.reset, palette.cyan, file.path, palette.reset
        );
    }
    Ok(())
}

/// The `Collections:` block (qmd.ts:1611-1636): names come from the
/// config in file order, counts from `documents` by `active = 1`.
fn list_collections(
    qmd: &Qmd,
    config: &qmd::Config,
    io: &mut Io<'_>,
    c: Palette,
) -> Result<(), Failure> {
    if config.collections.is_empty() {
        let _ = writeln!(
            io.out,
            "No collections found. Run 'qmd collection add .' to index files."
        );
        return Ok(());
    }
    let status = qmd.status()?;
    let counts: HashMap<&str, usize> = status
        .collections
        .iter()
        .map(|col| (col.name.as_str(), col.documents))
        .collect();
    let _ = writeln!(io.out, "{}Collections:{}\n", c.bold, c.reset);
    for name in config.collections.keys() {
        let count = counts.get(name.as_str()).copied().unwrap_or(0);
        let _ = writeln!(
            io.out,
            "  {}qmd://{}{}{}/{}  {}({count} files){}",
            c.dim, c.reset, c.cyan, name, c.reset, c.dim, c.reset
        );
    }
    Ok(())
}

/// Upstream's four path-argument branches (qmd.ts:1638-1693); returns
/// `(collection name, optional path prefix)`.
fn parse_path_arg(arg: &str, config: &qmd::Config) -> Result<(String, Option<String>), Failure> {
    if let Some(after_scheme) = arg.strip_prefix("qmd://") {
        if after_scheme.starts_with('/') {
            // Absolute-path collection: `qmd:///Users/foo/bar` — one
            // trailing slash stripped, then longest-prefix name match.
            let normalized = after_scheme.strip_suffix('/').unwrap_or(after_scheme);
            if let Some(name) = longest_prefix(config, normalized) {
                let rest = normalized
                    .get(name.len()..)
                    .unwrap_or("")
                    .trim_start_matches('/');
                return Ok((name.to_owned(), non_empty(rest)));
            }
            // Historical `qmd:////collection/path` alias (qmd.ts:1656-1666).
            let parsed = parse_virtual_path(arg).ok_or(Failure::Reported(1))?;
            return Ok((parsed.collection_name, non_empty(&parsed.path)));
        }
        let parsed = parse_virtual_path(arg).ok_or(Failure::Reported(1))?;
        Ok((parsed.collection_name, non_empty(&parsed.path)))
    } else if arg.starts_with('/') {
        let normalized = arg.strip_suffix('/').unwrap_or(arg);
        longest_prefix(config, normalized).map_or_else(
            // No match: the whole path becomes the collection name
            // (which then reports "Collection not found").
            || Ok((normalized.to_owned(), None)),
            |name| {
                let rest = normalized
                    .get(name.len()..)
                    .unwrap_or("")
                    .trim_start_matches('/');
                Ok((name.to_owned(), non_empty(rest)))
            },
        )
    } else {
        let mut parts = arg.split('/');
        let name = parts.next().unwrap_or_default().to_owned();
        let rest = parts.collect::<Vec<_>>().join("/");
        Ok((name, non_empty(&rest)))
    }
}

/// Longest config-collection name that equals `normalized` or is a
/// `/`-separated prefix of it (qmd.ts:1645-1649, 1677-1681).
fn longest_prefix<'a>(config: &'a qmd::Config, normalized: &str) -> Option<&'a str> {
    let mut best: Option<&'a str> = None;
    for name in config.collections.keys() {
        if (normalized == name.as_str() || normalized.starts_with(&format!("{name}/")))
            // JS `.sort` by descending length is stable — the first
            // same-length name wins, so keep strict `>`.
            && best.is_none_or(|b: &'a str| name.len() > b.len())
        {
            best = Some(name.as_str());
        }
    }
    best
}

/// `rest || null` — the empty string means "whole collection".
fn non_empty(s: &str) -> Option<String> {
    (!s.is_empty()).then(|| s.to_owned())
}

/// `formatBytes` (qmd.ts:433-438): `<1024` bytes stay unitless, then
/// KB/MB/GB at `toFixed(1)`.
fn format_bytes(bytes: usize) -> String {
    let b = bytes as f64;
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    if bytes < 1024 * 1024 {
        return format!("{} KB", js::to_fixed(b / 1024.0, 1));
    }
    if bytes < 1024 * 1024 * 1024 {
        return format!("{} MB", js::to_fixed(b / (1024.0 * 1024.0), 1));
    }
    format!("{} GB", js::to_fixed(b / (1024.0 * 1024.0 * 1024.0), 1))
}

/// `formatLsTime` (qmd.ts:1767-1783): `Jan  1 09:05`-style for the last
/// six 30-day months, `Jan  1  2020` beyond. `now` is injected so the
/// function stays pure; `offset` is the process-local UTC offset.
fn format_ls_time(modified_at: &str, now: OffsetDateTime, offset: UtcOffset) -> String {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let Ok(parsed) =
        time::PrimitiveDateTime::parse(modified_at, ISO).map(time::PrimitiveDateTime::assume_utc)
    else {
        // Upstream's `new Date()` on an unparseable string yields
        // Invalid Date — its getters are NaN/undefined.
        return "undefined NaN NaN:NaN".to_owned();
    };
    let date = parsed.to_offset(offset);
    let month = MONTHS
        .get(usize::from(u8::from(date.month())) - 1)
        .unwrap_or(&"???");
    let day = date.day();
    if parsed < now - time::Duration::days(6 * 30) {
        format!("{month} {day:>2}  {}", date.year())
    } else {
        format!("{month} {day:>2} {:02}:{:02}", date.hour(), date.minute())
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::missing_assert_message,
    reason = "idiomatic test-code patterns"
)]
mod tests {
    //! `formatLsTime` both branches with a fixed `now` (upstream
    //! `test/cli.test.ts` has no unit coverage of this — the golden
    //! replay pins the rendered output end to end).

    use time::format_description::well_known::Rfc3339;
    use time::{OffsetDateTime, UtcOffset};

    use super::{format_bytes, format_ls_time};

    fn now() -> OffsetDateTime {
        OffsetDateTime::parse("2025-06-01T12:00:00Z", &Rfc3339).unwrap()
    }

    #[test]
    fn format_ls_time_old_file_shows_year() {
        // 2020-01-01 is more than 6×30 days before 2025-06-01.
        assert_eq!(
            format_ls_time("2020-01-01T00:00:00.000Z", now(), UtcOffset::UTC),
            "Jan  1  2020"
        );
    }

    #[test]
    fn format_ls_time_recent_file_shows_hour_minute() {
        assert_eq!(
            format_ls_time("2025-05-30T09:05:00.000Z", now(), UtcOffset::UTC),
            "May 30 09:05"
        );
    }

    #[test]
    fn format_ls_time_recent_file_respects_local_offset() {
        // UTC+10 turns 2025-05-30T23:30Z into May 31 09:30 local.
        assert_eq!(
            format_ls_time(
                "2025-05-30T23:30:00.000Z",
                now(),
                UtcOffset::from_hms(10, 0, 0).unwrap()
            ),
            "May 31 09:30"
        );
    }

    #[test]
    fn format_ls_time_invalid_date_matches_js_invalid_date() {
        assert_eq!(
            format_ls_time("not-a-date", now(), UtcOffset::UTC),
            "undefined NaN NaN:NaN"
        );
    }

    #[test]
    fn format_bytes_js_tie_rounding() {
        // 1280 B = 1.25 KB — JS `toFixed(1)` rounds the tie up.
        assert_eq!(format_bytes(1280), "1.3 KB");
        assert_eq!(format_bytes(69), "69 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
    }
}
