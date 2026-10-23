//! OSC 9;4 terminal progress — upstream's `progress` helper
//! (cli/qmd.ts:336-350), `formatETA` (qmd.ts:353-357) and the shared
//! `onProgress` body of `updateCollections`/`indexFiles`
//! (qmd.ts:987-994, 2042-2047).
//!
//! Every write is gated on `stderr.isTTY` (`isTTY`, qmd.ts:336): a
//! non-tty run emits zero progress bytes, which keeps the golden replay
//! byte-stable. `progress.error` (`\x1b]9;4;2`) is omitted — the port's
//! indexing surface has no failure-rendering call for it (upstream only
//! calls it from `embed`).

use std::time::Instant;

use crate::output::Io;

/// `progress.indeterminate()` — `\x1b]9;4;3\x07` (qmd.ts:344-346).
pub(crate) fn indeterminate(io: &mut Io<'_>) {
    if io.stderr_tty {
        let _ = io.err.write_all(b"\x1b]9;4;3\x07");
    }
}

/// `progress.set(percent)` — `\x1b]9;4;1;{Math.round(p)}\x07`
/// (qmd.ts:338-340).
pub(crate) fn set(io: &mut Io<'_>, percent: f64) {
    if io.stderr_tty {
        let _ = write!(
            io.err,
            "\x1b]9;4;1;{}\x07",
            qmd::js::number_to_string(qmd::js::math_round(percent))
        );
    }
}

/// `progress.clear()` — `\x1b]9;4;0\x07` (qmd.ts:341-343).
pub(crate) fn clear(io: &mut Io<'_>) {
    if io.stderr_tty {
        let _ = io.err.write_all(b"\x1b]9;4;0\x07");
    }
}

/// The shared `onProgress` body — `updateCollections` (qmd.ts:987-994)
/// and `indexFiles` (qmd.ts:2042-2047) render identically: OSC 9;4
/// percent, then the `\rIndexing:` status line. The ETA appears only
/// after two files so the rate estimate settles, and the line carries
/// upstream's eight trailing spaces. `current`/`total` are
/// [`qmd::UpdateProgress`]'s `info.current`/`info.total`.
pub(crate) fn indexing_tick(io: &mut Io<'_>, current: usize, total: usize, start: Instant) {
    set(io, current as f64 / total as f64 * 100.0);
    let elapsed = start.elapsed().as_secs_f64();
    let rate = current as f64 / elapsed;
    let remaining = (total as f64 - current as f64) / rate;
    // `info.current > 2` — the ETA needs a few samples to be meaningful.
    let eta = if current > 2 {
        format!(" ETA: {}", format_eta(remaining))
    } else {
        String::new()
    };
    if io.stderr_tty {
        let _ = write!(io.err, "\rIndexing: {current}/{total}{eta}        ");
    }
}

/// Upstream `formatETA` (qmd.ts:353-357): `<60s` → rounded seconds,
/// `<1h` → `floor(m)m round(s)s`, else `floor(h)h floor(m)m`. The
/// JS `Math.round`/`Math.floor` ports keep NaN/∞ formatting as `NaN`/
/// `Infinity` via [`qmd::js::number_to_string`].
fn format_eta(seconds: f64) -> String {
    if seconds < 60.0 {
        return format!(
            "{}s",
            qmd::js::number_to_string(qmd::js::math_round(seconds))
        );
    }
    if seconds < 3600.0 {
        return format!(
            "{}m {}s",
            qmd::js::number_to_string((seconds / 60.0).floor()),
            qmd::js::number_to_string(qmd::js::math_round(seconds % 60.0))
        );
    }
    format!(
        "{}h {}m",
        qmd::js::number_to_string((seconds / 3600.0).floor()),
        qmd::js::number_to_string(((seconds % 3600.0) / 60.0).floor())
    )
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::missing_assert_message,
    reason = "idiomatic test-code patterns"
)]
mod tests {
    use std::io::BufRead;

    use super::*;

    fn test_io(err: &mut Vec<u8>, stderr_tty: bool) -> Io<'_> {
        let empty: &'static mut dyn BufRead = Box::leak(Box::new(std::io::empty()));
        Io {
            out: Box::leak(Box::new(Vec::new())),
            err,
            stdin: empty,
            stdout_tty: false,
            stderr_tty,
            stdin_tty: false,
            local_offset: time::UtcOffset::UTC,
        }
    }

    #[test]
    fn osc_sequences_match_upstream() {
        let mut err = Vec::new();
        let mut io = test_io(&mut err, true);
        indeterminate(&mut io);
        set(&mut io, 42.4);
        set(&mut io, 42.6);
        clear(&mut io);
        assert_eq!(
            String::from_utf8(err).unwrap(),
            "\x1b]9;4;3\x07\x1b]9;4;1;42\x07\x1b]9;4;1;43\x07\x1b]9;4;0\x07"
        );
    }

    /// `isTTY` gates every write (qmd.ts:336) — a non-tty run emits no
    /// progress bytes (the golden replay proves the same end to end).
    #[test]
    fn non_tty_emits_nothing() {
        let mut err = Vec::new();
        let mut io = test_io(&mut err, false);
        indeterminate(&mut io);
        set(&mut io, 50.0);
        indexing_tick(&mut io, 3, 10, Instant::now());
        clear(&mut io);
        assert!(err.is_empty());
    }

    /// The `\rIndexing:` line and its ETA gate (`current > 2`) come out
    /// only on a tty.
    #[test]
    fn indexing_tick_writes_indexing_line_on_tty() {
        let mut err = Vec::new();
        let mut io = test_io(&mut err, true);
        indexing_tick(&mut io, 1, 10, Instant::now());
        assert_eq!(
            String::from_utf8(err).unwrap(),
            "\x1b]9;4;1;10\x07\rIndexing: 1/10        "
        );

        let mut err2 = Vec::new();
        let mut io2 = test_io(&mut err2, true);
        indexing_tick(&mut io2, 5, 10, Instant::now());
        let text = String::from_utf8(err2).unwrap();
        assert!(
            text.starts_with("\x1b]9;4;1;50\x07\rIndexing: 5/10 ETA: "),
            "{text:?}"
        );
        assert!(text.ends_with("        "), "{text:?}");
    }

    #[test]
    fn format_eta_matches_js_branches() {
        assert_eq!(format_eta(5.0), "5s");
        assert_eq!(format_eta(5.6), "6s"); // Math.round
        assert_eq!(format_eta(90.0), "1m 30s");
        assert_eq!(format_eta(89.9), "1m 30s");
        assert_eq!(format_eta(3600.0), "1h 0m");
        assert_eq!(format_eta(3700.0), "1h 1m");
        assert_eq!(format_eta(7325.0), "2h 2m");
        // JS: NaN fails every comparison → falls into the hour branch.
        assert_eq!(format_eta(f64::NAN), "NaNh NaNm");
        // JS: +∞ lands in the hour branch; `∞ % 3600` is NaN.
        assert_eq!(format_eta(f64::INFINITY), "Infinityh NaNm");
    }
}
