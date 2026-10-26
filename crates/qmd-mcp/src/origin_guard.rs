//! DNS-rebinding protection for the HTTP MCP transport — ports upstream
//! `src/mcp/origin-guard.ts` (v2.8.3) 1:1.
//!
//! Binding to localhost keeps remote clients out but not web pages: a page
//! served from `attacker.example` can re-point that hostname at `127.0.0.1`
//! after load (DNS rebinding); the browser then treats
//! `http://attacker.example:8181` as same-origin and hands the response body
//! to attacker JavaScript. The only server-side signal that distinguishes
//! such a request from a legitimate local client is the `Origin` and `Host`
//! headers, so the MCP spec requires local HTTP servers to validate them.
//! The predicates here are pure so they can be unit-tested without a socket;
//! the HTTP middleware applies them to every request before routing.

use std::env;

/// Wildcard bind addresses — the request's legitimate `Host` is unknowable
/// (origin-guard.ts:22).
const WILDCARD_BIND_HOSTS: [&str; 5] = ["", "0.0.0.0", "::", "[::]", "*"];

/// `isLoopbackHostname` (origin-guard.ts:33): hostnames that resolve to this
/// machine's loopback interface. `0.0.0.0` is deliberately excluded —
/// browsers route it to loopback on Linux and macOS, so
/// `Origin: http://0.0.0.0:8181` is a rebinding-free path to local servers
/// and is treated as untrusted like any other.
#[must_use]
pub(crate) fn is_loopback_hostname(hostname: &str) -> bool {
    // `/^\[|\]$/g` strips one leading `[` and one trailing `]`, not nested pairs.
    let lower = hostname.trim().to_lowercase();
    let unbracketed = lower.strip_prefix('[').unwrap_or(&lower);
    let h = unbracketed.strip_suffix(']').unwrap_or(unbracketed);
    if h == "localhost" || h.ends_with(".localhost") {
        return true;
    }
    if h == "::1" || h == "0:0:0:0:0:0:0:1" {
        return true;
    }
    // IPv4-mapped IPv6 (::ffff:127.0.0.1) and the whole 127.0.0.0/8 range —
    // `/^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/` (origin-guard.ts:32).
    let v4 = h.strip_prefix("::ffff:").unwrap_or(h);
    let parts: Vec<&str> = v4.split('.').collect();
    parts.len() == 4
        && parts.first() == Some(&"127")
        && parts
            .iter()
            .skip(1)
            .all(|o| (1..=3).contains(&o.len()) && o.bytes().all(|b| b.is_ascii_digit()))
}

/// `originHostname` (origin-guard.ts:39-47): parse an `Origin` header into
/// its hostname, or `None` if unusable.
fn origin_hostname(origin: &str) -> Option<String> {
    let url = url::Url::parse(origin.trim()).ok()?;
    if url.scheme() != "http" && url.scheme() != "https" {
        return None;
    }
    url.host_str().map(str::to_owned)
}

/// `normalizeOrigin` (origin-guard.ts:50-55): normalize an origin for
/// comparison against the allowlist (`scheme://host[:port]`, lowercased,
/// default port dropped).
fn normalize_origin(origin: &str) -> Option<String> {
    let url = url::Url::parse(origin.trim()).ok()?;
    Some(url.origin().ascii_serialization().to_lowercase())
}

/// `hostHeaderHostname` (origin-guard.ts:58-63): parse a `Host` header
/// (`name`, `name:port`, `[::1]:port`) into a hostname.
fn host_header_hostname(host: &str) -> Option<String> {
    url::Url::parse(&format!("http://{}", host.trim()))
        .ok()
        .and_then(|u| u.host_str().map(str::to_owned))
}

/// `splitList` (origin-guard.ts:74-77): comma-separated env list.
fn split_list(value: Option<&str>) -> Vec<String> {
    value.map_or_else(Vec::new, |v| {
        v.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect()
    })
}

/// Upstream `OriginGuard` (origin-guard.ts:64-72).
#[derive(Debug)]
pub(crate) struct OriginGuard {
    /// All checks disabled (`QMD_ALLOWED_ORIGINS=*`).
    pub(crate) disabled: bool,
    /// Whether the `Host` header is checked at all.
    pub(crate) enforce_host: bool,
    /// Extra origins accepted beyond the loopback defaults, normalized.
    allowed_origins: Vec<String>,
    /// Extra `Host` values accepted beyond the loopback defaults, lowercased.
    allowed_hosts: Vec<String>,
}

/// `resolveOriginGuard` (origin-guard.ts:88-112).
///
/// `Host` validation is enforced whenever the bind address names a concrete
/// interface, since the legitimate `Host` header is then known (loopback, or
/// the bind address itself). For a wildcard bind — `--host 0.0.0.0`, e.g.
/// Docker — any of the container's names may legitimately appear, so the
/// check is only enforced when the operator supplies an explicit allowlist.
/// `Origin` validation always applies; browsers are the threat, and they
/// always send it.
pub(crate) fn resolve_origin_guard(host: &str) -> OriginGuard {
    resolve_origin_guard_with(host, None, None, |k| env::var(k).ok())
}

/// Resolve with injectable values so tests do not touch the process env —
/// `allowed_origins`/`allowed_hosts` `Some` wins over env like upstream's
/// `??` (origin-guard.ts:91-92).
pub(crate) fn resolve_origin_guard_with(
    host: &str,
    allowed_origins: Option<Vec<String>>,
    allowed_hosts: Option<Vec<String>>,
    env: impl Fn(&str) -> Option<String>,
) -> OriginGuard {
    let raw_origins =
        allowed_origins.unwrap_or_else(|| split_list(env("QMD_ALLOWED_ORIGINS").as_deref()));
    let raw_hosts =
        allowed_hosts.unwrap_or_else(|| split_list(env("QMD_ALLOWED_HOSTS").as_deref()));

    if raw_origins.iter().any(|o| o == "*") {
        return OriginGuard {
            disabled: true,
            enforce_host: false,
            allowed_origins: Vec::new(),
            allowed_hosts: Vec::new(),
        };
    }

    let origins: Vec<String> = raw_origins
        .iter()
        .filter_map(|o| normalize_origin(o))
        .collect();
    let mut hosts: Vec<String> = raw_hosts.iter().map(|h| h.to_lowercase()).collect();

    let bind_host = host.trim().to_lowercase();
    let wildcard = WILDCARD_BIND_HOSTS.contains(&bind_host.as_str());
    if !wildcard && !is_loopback_hostname(&bind_host) {
        // Explicit non-loopback interface: its own address is a legitimate Host.
        hosts.push(bind_host);
    }

    OriginGuard {
        disabled: false,
        enforce_host: !wildcard || !hosts.is_empty(),
        allowed_origins: origins,
        allowed_hosts: hosts,
    }
}

/// `checkRequestOrigin` (origin-guard.ts:133-163). Returns the rejection
/// reason, or `None` when the request passes.
///
/// A missing `Origin` is allowed: non-browser clients (curl, the MCP SDK's
/// HTTP client, editors) omit it, and browsers cannot. The header is the
/// signal that a request came from page JavaScript, so its *absence* is not
/// suspicious while a foreign *value* is conclusive.
pub(crate) fn check_request_origin(
    origin: Option<&str>,
    host: Option<&str>,
    guard: &OriginGuard,
) -> Option<String> {
    if guard.disabled {
        return None;
    }

    if let Some(origin) = origin.map(str::trim).filter(|o| !o.is_empty()) {
        let normalized = normalize_origin(origin);
        let hostname = origin_hostname(origin);
        let allowed = hostname.as_deref().is_some_and(is_loopback_hostname)
            || normalized.is_some_and(|n| guard.allowed_origins.contains(&n));
        if !allowed {
            return Some(format!("Origin not allowed: {origin}"));
        }
    }

    if guard.enforce_host
        && let Some(host) = host.map(str::trim).filter(|h| !h.is_empty())
    {
        let lowered = host.to_lowercase();
        let hostname = host_header_hostname(host);
        let allowed = hostname.as_deref().is_some_and(is_loopback_hostname)
            || guard.allowed_hosts.contains(&lowered)
            || hostname.is_some_and(|h| guard.allowed_hosts.contains(&h.to_lowercase()));
        if !allowed {
            return Some(format!("Host not allowed: {host}"));
        }
    }

    None
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::missing_assert_message,
    reason = "idiomatic test-code patterns"
)]
mod tests {
    //! Ports `test/mcp-origin-guard.test.ts` — the predicate cases (the
    //! live-server block lives in `qmd-cli`'s subprocess tests).

    use super::{
        OriginGuard, check_request_origin, is_loopback_hostname, resolve_origin_guard_with,
    };

    /// Upstream `resolveOriginGuard({host, env: {}})` — an empty env.
    fn guard_for(host: &str) -> OriginGuard {
        resolve_origin_guard_with(host, None, None, |_| None)
    }

    fn guard_with(
        host: &str,
        allowed_origins: Option<Vec<String>>,
        allowed_hosts: Option<Vec<String>>,
    ) -> OriginGuard {
        resolve_origin_guard_with(host, allowed_origins, allowed_hosts, |_| None)
    }

    fn guard_with_env(host: &str, env: &[(&str, &str)]) -> OriginGuard {
        resolve_origin_guard_with(host, None, None, |k| {
            env.iter()
                .find(|(key, _)| *key == k)
                .map(|(_, v)| (*v).to_owned())
        })
    }

    /// `checkRequestOrigin` verdict helper — `None` means `{ok: true}`.
    fn check(origin: Option<&str>, host: Option<&str>, guard: &OriginGuard) -> Option<String> {
        check_request_origin(origin, host, guard)
    }

    #[test]
    fn accepts_loopback_hostnames() {
        // upstream test.each "accepts %s".
        for h in [
            "localhost",
            "LOCALHOST",
            "app.localhost",
            "127.0.0.1",
            "127.1.2.3",
            "::1",
            "[::1]",
            "::ffff:127.0.0.1",
        ] {
            assert!(is_loopback_hostname(h), "{h}");
        }
    }

    #[test]
    fn rejects_non_loopback_hostnames() {
        // upstream test.each "rejects %s".
        for h in [
            "evil.example",
            "notlocalhost",
            "localhost.evil.example",
            "127.0.0.1.evil.example",
            "10.0.0.1",
            "::2",
        ] {
            assert!(!is_loopback_hostname(h), "{h}");
        }
    }

    #[test]
    fn rejects_0_0_0_0() {
        // "rejects 0.0.0.0" — browsers route it to loopback on
        // Linux/macOS, so it is not a trusted origin.
        assert!(!is_loopback_hostname("0.0.0.0"));
    }

    #[test]
    fn loopback_bind_enforces_host_header() {
        let guard = guard_for("localhost");
        assert!(!guard.disabled);
        assert!(guard.enforce_host);
    }

    #[test]
    fn wildcard_bind_without_allowlist_skips_host_enforcement() {
        assert!(!guard_for("0.0.0.0").enforce_host);
    }

    #[test]
    fn wildcard_bind_with_allowlist_enforces_host() {
        let guard = guard_with("0.0.0.0", None, Some(vec!["qmd.internal".to_owned()]));
        assert!(guard.enforce_host);
        assert!(guard.allowed_hosts.contains(&"qmd.internal".to_owned()));
    }

    #[test]
    fn concrete_non_loopback_bind_trusts_own_address() {
        let guard = guard_for("192.168.1.5");
        assert!(guard.enforce_host);
        assert!(guard.allowed_hosts.contains(&"192.168.1.5".to_owned()));
    }

    #[test]
    fn allowed_origins_star_disables_every_check() {
        let guard = guard_with_env("localhost", &[("QMD_ALLOWED_ORIGINS", "*")]);
        assert!(guard.disabled);
        assert!(check(Some("https://evil.example"), Some("evil.example"), &guard).is_none());
    }

    #[test]
    fn reads_comma_separated_allowlists_from_env() {
        let guard = guard_with_env(
            "localhost",
            &[
                (
                    "QMD_ALLOWED_ORIGINS",
                    "https://notes.internal , https://app.internal",
                ),
                ("QMD_ALLOWED_HOSTS", "notes.internal"),
            ],
        );
        assert_eq!(
            guard.allowed_origins,
            vec!["https://notes.internal", "https://app.internal"]
        );
        assert_eq!(guard.allowed_hosts, vec!["notes.internal"]);
    }

    #[test]
    fn allows_request_with_no_origin_header() {
        let guard = guard_for("localhost");
        assert!(check(None, Some("localhost:8181"), &guard).is_none());
    }

    #[test]
    fn allows_loopback_origins() {
        let guard = guard_for("localhost");
        for origin in [
            "http://localhost:8181",
            "http://127.0.0.1:8181",
            "http://[::1]:8181",
        ] {
            assert!(
                check(Some(origin), Some("localhost:8181"), &guard).is_none(),
                "{origin}"
            );
        }
    }

    #[test]
    fn rejects_foreign_origin() {
        let guard = guard_for("localhost");
        let verdict = check(Some("https://evil.example"), Some("localhost:8181"), &guard);
        assert!(verdict.is_some_and(|r| r.contains("Origin not allowed")));
    }

    #[test]
    fn rejects_rebound_host_even_when_origin_absent() {
        let guard = guard_for("localhost");
        let verdict = check(None, Some("evil.example:8181"), &guard);
        assert!(verdict.is_some_and(|r| r.contains("Host not allowed")));
    }

    #[test]
    fn rejects_origin_merely_embedding_loopback_name() {
        let guard = guard_for("localhost");
        assert!(check(Some("https://localhost.evil.example"), None, &guard).is_some());
        assert!(check(Some("https://evil.example#localhost"), None, &guard).is_some());
    }

    #[test]
    fn rejects_opaque_null_origin() {
        let guard = guard_for("localhost");
        assert!(check(Some("null"), Some("localhost:8181"), &guard).is_some());
    }

    #[test]
    fn rejects_non_http_schemes() {
        let guard = guard_for("localhost");
        assert!(check(Some("file://"), Some("localhost:8181"), &guard).is_some());
    }

    #[test]
    fn honours_explicit_origin_allowlist() {
        let guard = guard_with(
            "localhost",
            Some(vec!["https://notes.internal".to_owned()]),
            Some(vec!["notes.internal".to_owned()]),
        );
        assert!(
            check(
                Some("https://notes.internal"),
                Some("notes.internal"),
                &guard
            )
            .is_none()
        );
        assert!(
            check(
                Some("https://other.internal"),
                Some("notes.internal"),
                &guard
            )
            .is_some()
        );
    }

    #[test]
    fn wildcard_bind_skips_host_validation_but_screens_origin() {
        let guard = guard_for("0.0.0.0");
        assert!(check(None, Some("qmd-container:8181"), &guard).is_none());
        assert!(
            check(
                Some("https://evil.example"),
                Some("qmd-container:8181"),
                &guard
            )
            .is_some()
        );
    }
}
