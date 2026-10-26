//! HTTP transport — ports upstream `startMcpHttpServer`
//! (src/mcp/server.ts:889-1180) plus the upstream SDK's stateless
//! streamable-HTTP semantics, pinned by the recorded responses in
//! `crates/qmd-cli/tests/fixtures/mcp_golden.json`.
//!
//! rmcp's `StreamableHttpService` is deliberately *not* used: it requires
//! `S: ServerHandler`, and rmcp blanket-implements `Service<RoleServer>`
//! for every `ServerHandler`, which conflicts with `QmdMcpService`'s
//! interception layer that emits raw JSON for `tools/call`/`resources/read`.
//! Each POST is dispatched through a oneshot transport into
//! [`rmcp::service::serve_directly`] instead, which accepts
//! `Service<RoleServer>` — the same service object stdio uses,
//! interception included.

use std::net::ToSocketAddrs as _;
use std::sync::Arc;
use std::time::Instant;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, Method, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::Response;
use qmd::Qmd;
use rmcp::model::{ClientJsonRpcMessage, ClientRequest, InitializeRequestParams, ProtocolVersion};
use rmcp::service::serve_directly;
use rmcp::transport::OneshotTransport;
use rmcp::{RoleServer, Service};
use serde_json::{Value, json};

use crate::origin_guard::{OriginGuard, check_request_origin, resolve_origin_guard};
use crate::rest::RestState;
use crate::{Error, QmdMcpService};

/// Request-body cap — the one bound upstream lacks (`collectBody` is
/// unbounded); recorded as an intentional deviation.
const MAX_BODY: usize = 4 * 1024 * 1024;

/// `MCP-Protocol-Version`/`Mcp-Method`/`Mcp-Name` (SEP-2243 headers; the
/// `HeaderMap` lookup is case-insensitive).
const HDR_VERSION: &str = "mcp-protocol-version";
const HDR_METHOD: &str = "mcp-method";
const HDR_NAME: &str = "mcp-name";

/// The single protocol version the 2026-era HTTP path serves — upstream's
/// `data.supported` in the recorded `-32022` response is `["2026-07-28"]`.
const MODERN: &str = "2026-07-28";

/// `_meta` keys the recorded `-32602` response demands when the
/// `MCP-Protocol-Version` header names the modern revision.
const MODERN_REQUIRED_META: [&str; 3] = [
    "io.modelcontextprotocol/protocolVersion",
    "io.modelcontextprotocol/clientInfo",
    "io.modelcontextprotocol/clientCapabilities",
];

/// Options for [`serve_http`], mirroring upstream's
/// `startMcpHttpServer(port, options)` (server.ts:889-893).
#[derive(Debug)]
#[non_exhaustive]
pub struct HttpOptions {
    /// Port to bind; `0` asks the OS for a free port (tests).
    pub port: u16,
    /// Bind host; `None` falls back to `QMD_HOST` then `localhost`
    /// (server.ts:977).
    pub host: Option<String>,
    /// Suppress startup and per-request stderr logging (upstream `quiet`).
    pub quiet: bool,
}

impl Default for HttpOptions {
    /// Upstream defaults: port `8181`, host `None` (env/localhost
    /// fallback at bind time), logging on.
    fn default() -> Self {
        Self {
            port: 8181,
            host: None,
            quiet: false,
        }
    }
}

/// State shared by every route.
pub(crate) struct AppState {
    qmd: Arc<Qmd>,
    guard: OriginGuard,
    pub(crate) rest: Arc<RestState>,
    start: Instant,
    quiet: bool,
}

/// Log-label a handler stores in response extensions — the middleware
/// prints `{ts()} {method} {label-or-path} ({ms}ms)` (server.ts log calls).
#[derive(Clone)]
pub(crate) struct RequestLabel(pub(crate) String);

fn log_line(state: &AppState, line: &str) {
    use std::io::Write as _;
    if !state.quiet {
        drop(writeln!(std::io::stderr(), "{line}"));
    }
}

/// `ts()` — `new Date().toISOString().slice(11, 23)` (server.ts:918):
/// UTC `HH:mm:ss.SSS`.
fn ts() -> String {
    let t = time::OffsetDateTime::now_utc().time();
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        t.hour(),
        t.minute(),
        t.second(),
        t.millisecond()
    )
}

/// Run the HTTP MCP server until SIGTERM/SIGINT.
///
/// # Errors
/// [`Error::Io`] on bind/listen failure (`EADDRINUSE` arrives as
/// [`std::io::ErrorKind::AddrInUse`]); [`Error::Store`] on the
/// `defaultCollectionNames` read; [`Error::Serve`] on
/// `QmdMcpService` construction.
pub async fn serve_http(qmd: Arc<Qmd>, options: HttpOptions) -> Result<(), Error> {
    // `host = options.host ?? QMD_HOST ?? "localhost"` (server.ts:977).
    let host = options
        .host
        .unwrap_or_else(|| std::env::var("QMD_HOST").unwrap_or_else(|_| "localhost".to_owned()));
    let guard = resolve_origin_guard(&host);

    // `defaultCollectionNames` for the REST endpoint (server.ts:904).
    let default_collections = qmd
        .collections()
        .map_err(Error::Store)?
        .into_iter()
        .filter(|c| c.include_by_default)
        .map(|c| c.name)
        .collect();

    let state = Arc::new(AppState {
        rest: Arc::new(RestState {
            qmd: Arc::clone(&qmd),
            default_collections,
        }),
        qmd,
        guard,
        start: Instant::now(),
        quiet: options.quiet,
    });

    // Upstream tests the pathname *and* verb (`POST /health` → 404
    // without an `Allow` header). `on()` would inject `Allow` even on a
    // fallback response and `get()` answers HEAD, so each route is an
    // `any()` handler that gates the verb itself (`any` skips the
    // Allow header).
    let app = Router::new()
        .route("/health", axum::routing::any(health_gate))
        .route("/query", axum::routing::any(search_gate))
        .route("/search", axum::routing::any(search_gate))
        .route("/mcp", axum::routing::any(mcp))
        .fallback(not_found)
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            guard_and_log,
        ))
        .with_state(Arc::clone(&state));

    // A wildcard host string is a guard concept; the socket needs a
    // concrete address (`""`/`*` → v4-any like Node's `listen`).
    let bind_host = match host.trim() {
        "" | "*" => "0.0.0.0".to_owned(),
        h if h.contains(':') && !h.starts_with('[') => format!("[{h}]"),
        h => h.to_owned(),
    };
    // Node's `listen(port, host)` binds the FIRST resolved address and
    // fails EADDRINUSE when it's taken; tokio's multi-address bind
    // would silently fall through to another family, so resolve once
    // and bind exactly one address (`getaddrinfo` on `localhost` is
    // dual-stack — first result decides, like Node).
    let bind_addr = format!("{bind_host}:{}", options.port)
        .to_socket_addrs()
        .map_err(Error::Io)?
        .next()
        .ok_or_else(|| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("could not resolve host {bind_host}"),
            ))
        })?;
    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .map_err(Error::Io)?;
    let actual_port = listener.local_addr().map_err(Error::Io)?.port();

    // server.ts:1175-1183 — startup line then the two guard warnings.
    log_line(
        &state,
        &format!("QMD MCP server listening on http://{host}:{actual_port}/mcp"),
    );
    if state.guard.disabled {
        log_line(
            &state,
            "Warning: QMD_ALLOWED_ORIGINS=* — DNS-rebinding protection is off. Only do this behind your own authenticating proxy.",
        );
    } else if !state.guard.enforce_host {
        log_line(
            &state,
            &format!(
                "Warning: bound to {host} with no QMD_ALLOWED_HOSTS — Host validation is off and the index is readable by anyone who can reach this port."
            ),
        );
    }

    let result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await
        .map_err(Error::Io);
    // `store.close()` (server.ts:1156-1160).
    drop(state);
    result
}

/// Prints `Shutting down (SIGINT|SIGTERM)...` then lets axum finish
/// in-flight requests — upstream's two handlers (server.ts:1162-1169).
#[expect(
    clippy::expect_used,
    reason = "SIGTERM handler registration only fails without a runtime; serve_http owns one"
)]
async fn shutdown() {
    use std::io::Write as _;
    #[cfg(unix)]
    async fn sigterm() {
        let mut s = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler");
        s.recv().await;
    }
    #[cfg(not(unix))]
    async fn sigterm() {
        std::future::pending::<()>().await;
    }
    let sig = tokio::select! {
        _ = tokio::signal::ctrl_c() => "SIGINT",
        () = sigterm() => "SIGTERM",
    };
    drop(writeln!(std::io::stderr(), "Shutting down ({sig})..."));
}

/// Upstream's catch-all 404 — `res.end("Not Found")`, no `Content-Type`.
#[expect(clippy::expect_used, reason = "a static body cannot fail to build")]
async fn not_found() -> Response {
    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .body(Body::from("Not Found"))
        .expect("static")
}

/// Shared 413 — the body cap is our one deliberate deviation (upstream's
/// `collectBody` is unbounded); every route answers the same shape.
#[expect(clippy::expect_used, reason = "a static body cannot fail to build")]
fn payload_too_large() -> Response {
    Response::builder()
        .status(StatusCode::PAYLOAD_TOO_LARGE)
        .body(Body::from("Payload Too Large"))
        .expect("static")
}

/// 405 `-32000 "Method not allowed."` — recorded for GET/DELETE/PUT
/// on `/mcp` (SDK transport check, `charset` variant like the other
/// transport-validation errors).
fn method_not_allowed() -> Response {
    rpc_error(
        StatusCode::METHOD_NOT_ALLOWED,
        -32000,
        "Method not allowed.",
        &Value::Null,
        None,
        true,
    )
}

/// `/health` serves GET only — everything else is the catch-all 404
/// (`POST /health` → `Not Found`, server.ts:1003).
async fn health_gate(State(state): State<Arc<AppState>>, method: Method) -> Response {
    if method == Method::GET {
        health(State(state)).await
    } else {
        not_found().await
    }
}

/// `/query` and `/search` serve POST only (server.ts:1012-1013). The
/// body is buffered manually so the same `MAX_BODY` cap and 413 shape
/// apply as `/mcp` — axum's per-extractor `DefaultBodyLimit` would
/// otherwise answer a different (2 MiB, plain-text) 413.
async fn search_gate(State(state): State<Arc<AppState>>, req: Request) -> Response {
    if req.method() != Method::POST {
        return not_found().await;
    }
    let Ok(body) = axum::body::to_bytes(req.into_body(), MAX_BODY).await else {
        // No RequestLabel: upstream only logs /query on success, and the
        // cap itself is our deviation.
        return payload_too_large();
    };
    crate::rest::search(State(state), body).await
}

/// `GET /health` → `{status:'ok', uptime:<floor secs>}`
/// (server.ts:1006-1011).
#[expect(clippy::unused_async, reason = "axum handlers return a future")]
async fn health(State(state): State<Arc<AppState>>) -> Response {
    let mut resp = json_response(
        StatusCode::OK,
        &json!({ "status": "ok", "uptime": state.start.elapsed().as_secs() }),
        false,
    );
    resp.extensions_mut().insert(RequestLabel(String::new()));
    resp
}

/// JSON body — `charset` selects the `application/json;charset=utf-8`
/// variant the recorded responses show only on the SDK's inner-dispatch
/// errors (`-32020`/`-32602`/`-32022`/`-32600`/405/415).
#[expect(
    clippy::expect_used,
    reason = "only invalid status/header values fail; both are constant"
)]
fn json_response(status: StatusCode, body: &Value, charset: bool) -> Response {
    let body = serde_json::to_vec(body).unwrap_or_default();
    Response::builder()
        .status(status)
        .header(
            header::CONTENT_TYPE,
            if charset {
                "application/json;charset=utf-8"
            } else {
                "application/json"
            },
        )
        .body(Body::from(body))
        .expect("json response")
}

/// `{"jsonrpc":"2.0","error":{code,message,data?},"id":…}` — field order
/// and charset follow the recorded fixtures.
fn rpc_error(
    status: StatusCode,
    code: i64,
    message: &str,
    id: &Value,
    data: Option<Value>,
    charset: bool,
) -> Response {
    let mut err = json!({ "code": code, "message": message });
    if let Some(d) = data
        && let Some(map) = err.as_object_mut()
    {
        map.insert("data".to_owned(), d);
    }
    json_response(
        status,
        &json!({ "jsonrpc": "2.0", "error": err, "id": id }),
        charset,
    )
}

/// Middleware: origin guard ahead of routing so REST is covered too
/// (server.ts:984-1001, upstream tobi/qmd#881), then the per-request log.
async fn guard_and_log(State(state): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_owned();
    let start = Instant::now();

    let origin = req
        .headers()
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok());
    let host_h = req
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok());
    if let Some(reason) = check_request_origin(origin, host_h, &state.guard) {
        log_line(&state, &format!("{} {method} {path} 403 — {reason}", ts()));
        return rpc_error(
            StatusCode::FORBIDDEN,
            -32003,
            &format!("Forbidden: {reason}"),
            &Value::Null,
            None,
            false,
        );
    }

    let resp = next.run(req).await;
    if let Some(RequestLabel(label)) = resp.extensions().get::<RequestLabel>() {
        let ms = start.elapsed().as_millis();
        let path_or_label = if label.is_empty() {
            path.as_str()
        } else {
            label
        };
        log_line(
            &state,
            &format!("{} {method} {path_or_label} ({ms}ms)", ts()),
        );
    }
    resp
}

/// `describeRequest` (server.ts:935-951): label for the `/mcp` log line.
fn describe_request(body: &Value) -> String {
    let method = body
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    if method != "tools/call" {
        return method.to_owned();
    }
    let params = body.get("params");
    let tool = params
        .and_then(|p| p.get("name"))
        .map_or_else(|| "?".to_owned(), crate::js::js_string);
    if let Some(args) = params
        .and_then(|p| p.get("arguments"))
        .filter(|a| a.is_object())
    {
        for key in ["query", "file", "path", "pattern"] {
            if let Some(v) = args.get(key).filter(|v| crate::js::is_truthy(v)) {
                return tool_arg_label(&tool, key, &crate::js::js_string(v));
            }
        }
    }
    format!("tools/call {tool}")
}

/// `tools/call <tool> "<q≤80>"` for `query`, `tools/call <tool> <v>`
/// for the path-ish keys (server.ts:941-947).
fn tool_arg_label(tool: &str, key: &str, value: &str) -> String {
    if key == "query" {
        format!("tools/call {tool} \"{}\"", crate::js::str_prefix(value, 80))
    } else {
        format!("tools/call {tool} {value}")
    }
}

// ---------------------------------------------------------------------------
// POST /mcp — the stateless streamable-HTTP bridge
// ---------------------------------------------------------------------------

/// Era check for one message: the upstream SDK serves the modern path
/// whenever `params._meta` is present *or* the `MCP-Protocol-Version`
/// header names the modern revision (recorded: meta without the version
/// header → 200; version header without `_meta` → -32602).
fn is_modern(el: &Value, headers: &HeaderMap) -> bool {
    version_header(headers) == Some(MODERN) || el.pointer("/params/_meta").is_some()
}

fn version_header(headers: &HeaderMap) -> Option<&str> {
    headers.get(HDR_VERSION).and_then(|v| v.to_str().ok())
}

fn header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|v| v.to_str().ok())
}

/// Upstream's 2026-era header/envelope rules, in recorded order:
/// missing `_meta` keys → `-32602`; header/body version mismatch →
/// `-32020`; unsupported version → `-32022`; `Mcp-Method`/`Mcp-Name`
/// disagreement → `-32020`. Legacy-era messages skip all of this.
fn check_modern(el: &Value, headers: &HeaderMap) -> Option<Response> {
    if !is_modern(el, headers) {
        return None;
    }
    let id = el.get("id").cloned().unwrap_or(Value::Null);
    let meta = el.pointer("/params/_meta");
    let hdr_version = version_header(headers);

    // `-32602` — header names 2026-07-28 but required envelope keys are
    // absent (`{"envelope":{"missing":[…]}}`).
    if hdr_version == Some(MODERN) {
        let missing: Vec<&str> = match meta {
            None | Some(Value::Null) => vec!["_meta"],
            Some(m) => MODERN_REQUIRED_META
                .iter()
                .copied()
                .filter(|k| m.get(k).is_none())
                .collect(),
        };
        if !missing.is_empty() {
            return Some(rpc_error(
                StatusCode::BAD_REQUEST,
                -32602,
                &format!(
                    "Invalid params: the MCP-Protocol-Version header names protocol revision {MODERN}, but the request is missing the required per-request envelope key(s): {}",
                    missing.join(", ")
                ),
                &id,
                Some(json!({ "envelope": { "missing": missing } })),
                true,
            ));
        }
    }

    // `-32020` — header/body protocol-version disagreement.
    let meta_version = meta
        .and_then(|m| m.get("io.modelcontextprotocol/protocolVersion"))
        .and_then(Value::as_str);
    if let (Some(h), Some(m)) = (hdr_version, meta_version)
        && h != m
    {
        let detail = format!(
            "the body envelope names protocol version {m} but the MCP-Protocol-Version header names {h}"
        );
        return Some(rpc_error(
            StatusCode::BAD_REQUEST,
            -32020,
            &format!("Bad Request: the request headers and body disagree: {detail}"),
            &id,
            Some(json!({ "mismatch": { "header": h, "body": detail } })),
            true,
        ));
    }

    // `-32022` — effective version outside the modern supported set.
    let effective = hdr_version.or(meta_version);
    if let Some(v) = effective
        && v != MODERN
    {
        return Some(rpc_error(
            StatusCode::BAD_REQUEST,
            -32022,
            &format!("Unsupported protocol version: {v}"),
            &id,
            Some(json!({ "supported": [MODERN], "requested": v })),
            true,
        ));
    }

    method_header_check(el, headers, &id).or_else(|| name_header_check(el, headers, &id))
}

/// `-32020` — `Mcp-Method` must be present and match `body.method`.
fn method_header_check(el: &Value, headers: &HeaderMap, id: &Value) -> Option<Response> {
    let body_method = el
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let detail = match header(headers, HDR_METHOD) {
        None => format!(
            "the body names method {body_method} but the required Mcp-Method header is absent"
        ),
        Some(h) if h != body_method => {
            format!("the body names method {body_method} but the Mcp-Method header names {h}")
        }
        _ => return None,
    };
    Some(rpc_error(
        StatusCode::BAD_REQUEST,
        -32020,
        &format!("Bad Request: the request headers and body disagree: {detail}"),
        id,
        Some(
            json!({ "mismatch": { "header": header(headers, HDR_METHOD).unwrap_or("(missing)"), "body": detail } }),
        ),
        true,
    ))
}

/// `-32020` — `Mcp-Name` must match the operation name in params:
/// `params.name` (tools/call) or `params.uri` (resources/read).
fn name_header_check(el: &Value, headers: &HeaderMap, id: &Value) -> Option<Response> {
    let params = el.get("params").filter(|p| p.is_object());
    let (key, value) = params
        .and_then(|p| p.get("name").map(|v| ("name", v)))
        .or_else(|| params.and_then(|p| p.get("uri").map(|v| ("uri", v))))?;
    // Messages quote `JSON.stringify(value)` and `JSON.stringify(header)`;
    // the comparison is `String(value)`.
    let display_v = serde_json::to_string(value).unwrap_or_default();
    let expected = crate::js::js_string(value);
    let detail = match header(headers, HDR_NAME) {
        None => format!(
            "the body carries params.{key}={display_v} but the required Mcp-Name header is absent"
        ),
        Some(h) if h != expected => format!(
            "the body carries params.{key}={display_v} but the Mcp-Name header names {}",
            serde_json::to_string(&Value::String(h.to_owned())).unwrap_or_default()
        ),
        _ => return None,
    };
    Some(rpc_error(
        StatusCode::BAD_REQUEST,
        -32020,
        &format!("Bad Request: the request headers and body disagree: {detail}"),
        id,
        Some(
            json!({ "mismatch": { "header": header(headers, HDR_NAME).unwrap_or("(missing)"), "body": detail } }),
        ),
        true,
    ))
}

/// `peer_info_for_stateless_request` equivalent (tower.rs:2417), except
/// that the `_meta` version is a valid source too — upstream marks a
/// `_meta`-carrying request as modern even without the header. The
/// struct is `#[non_exhaustive]`, so it is built through serde.
///
/// `version` is clamped to the server's supported set: rmcp picks the
/// response era with a `>= "2026-07-28"` *string* compare on this
/// value, so an unrecognized revision (a bogus `MCP-Protocol-Version`
/// header, a future date) would leak the 2026 envelope into a legacy
/// frame. Upstream's negotiation treats it as a legacy client.
fn peer_info_for(
    request: &Value,
    headers: &HeaderMap,
    supported: &[ProtocolVersion],
) -> Option<InitializeRequestParams> {
    let mut version = if request.get("method").and_then(Value::as_str) == Some("initialize") {
        request
            .pointer("/params/protocolVersion")
            .and_then(Value::as_str)
            .unwrap_or("2025-03-26")
            .to_owned()
    } else {
        header(headers, HDR_VERSION)
            .or_else(|| {
                request
                    .pointer("/params/_meta/io.modelcontextprotocol~1protocolVersion")
                    .and_then(Value::as_str)
            })
            .unwrap_or("2025-03-26")
            .to_owned()
    };
    if !supported.iter().any(|v| v.as_str() == version) {
        ProtocolVersion::V_2025_03_26
            .as_str()
            .clone_into(&mut version);
    }
    serde_json::from_value(json!({
        "protocolVersion": version,
        "capabilities": {},
        "clientInfo": { "name": "qmd-http", "version": env!("CARGO_PKG_VERSION") },
    }))
    .ok()
}

/// Attach the `/mcp` request label — upstream logs *every* `/mcp`
/// request, error responses included (server.ts:1143).
fn with_label(mut resp: Response, label: &str) -> Response {
    resp.extensions_mut()
        .insert(RequestLabel(format!("/mcp {label}")));
    resp
}

/// POST /mcp (and 405 for the other verbs — recorded GET/DELETE/PUT).
async fn mcp(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let method = req.method().clone();
    // Upstream never collects a body for GET/HEAD (server.ts:1106-1108)
    // — the 405 fires even when the request carries a payload.
    if method == Method::GET || method == Method::HEAD {
        return with_label(method_not_allowed(), method.as_str());
    }
    let headers = req.headers().clone();
    let Ok(body) = axum::body::to_bytes(req.into_body(), MAX_BODY).await else {
        return with_label(payload_too_large(), method.as_str());
    };
    // `describeRequest(parsedBody) ?? method` — other non-POST verbs
    // with a JSON-RPC body (the recorded PUT carries `tools/call`) label
    // from the body, not the verb (server.ts:1106-1120).
    if method != Method::POST {
        return with_label(method_not_allowed(), &request_label(&body, &method));
    }
    mcp_post(&state, &headers, body).await
}

/// `const label = describeRequest(parsedBody) ?? method` — objects AND
/// arrays (JS `typeof [] === 'object'`) go through `describeRequest`,
/// scalars and unparseable bodies fall back to the method name; computed
/// before any validation so early errors log it too (server.ts:1106-1120).
fn request_label(body: &Bytes, method: &Method) -> String {
    serde_json::from_slice::<Value>(body)
        .ok()
        .filter(|v| v.is_object() || v.is_array())
        .map_or_else(|| method.as_str().to_owned(), |v| describe_request(&v))
}

/// `POST /mcp`: transport validation, then dispatch.
async fn mcp_post(state: &Arc<AppState>, headers: &HeaderMap, body: Bytes) -> Response {
    let label = request_label(&body, &Method::POST);
    // 406 — Accept must name BOTH types; `*/*` does not count (recorded).
    let accept = header(headers, "accept").unwrap_or("").to_lowercase();
    let accepts: Vec<&str> = accept
        .split(',')
        .map(|t| t.split(';').next().unwrap_or("").trim())
        .collect();
    if !accepts.contains(&"application/json") || !accepts.contains(&"text/event-stream") {
        return with_label(
            rpc_error(
                StatusCode::NOT_ACCEPTABLE,
                -32000,
                "Not Acceptable: Client must accept both application/json and text/event-stream",
                &Value::Null,
                None,
                false,
            ),
            &label,
        );
    }

    // 415 — Content-Type must be application/json (recorded).
    let ctype = header(headers, "content-type")
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_lowercase();
    if ctype != "application/json" {
        return with_label(
            rpc_error(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                -32000,
                "Unsupported Media Type: Content-Type must be application/json",
                &Value::Null,
                None,
                true,
            ),
            &label,
        );
    }

    // -32700 invalid JSON (bare charset) → -32600 non-JSON-RPC (charset).
    let Ok(raw) = serde_json::from_slice::<Value>(&body) else {
        return with_label(
            rpc_error(
                StatusCode::BAD_REQUEST,
                -32700,
                "Parse error: Invalid JSON",
                &Value::Null,
                None,
                false,
            ),
            &label,
        );
    };
    let elements: Vec<Value> = match &raw {
        Value::Array(items) => items.clone(),
        other => vec![other.clone()],
    };
    let bad_shape = || {
        with_label(
            rpc_error(
                StatusCode::BAD_REQUEST,
                -32600,
                "Bad Request: the request body is not a valid JSON-RPC message",
                &Value::Null,
                None,
                true,
            ),
            &label,
        )
    };
    if elements.is_empty() {
        return bad_shape();
    }
    for el in &elements {
        if !el.is_object() {
            return bad_shape();
        }
        if let Some(resp) = check_modern(el, headers) {
            return with_label(resp, &label);
        }
    }
    serve_elements(state, headers, elements, &label).await
}

/// Serve each message through a fresh `OneshotTransport` +
/// `serve_directly` (rmcp's stateless path equivalent) and collect the
/// responses; then shape the HTTP answer — 202 when every element was a
/// notification, `application/json` for the 2026 era, SSE frames for
/// the legacy era.
async fn serve_elements(
    state: &Arc<AppState>,
    headers: &HeaderMap,
    elements: Vec<Value>,
    label: &str,
) -> Response {
    use std::io::Write as _;
    let mut events: Vec<String> = Vec::new();
    let mut results: Vec<Value> = Vec::new();
    let mut modern = false;
    for el in elements {
        let el_modern = is_modern(&el, headers);
        modern |= el_modern;
        let Ok(msg) = serde_json::from_value::<ClientJsonRpcMessage>(el.clone()) else {
            return with_label(
                rpc_error(
                    StatusCode::BAD_REQUEST,
                    -32600,
                    "Bad Request: the request body is not a valid JSON-RPC message",
                    &Value::Null,
                    None,
                    true,
                ),
                label,
            );
        };
        let ClientJsonRpcMessage::Request(request) = msg else {
            // Notifications and stray responses get no reply (upstream
            // answers the POST with a bare 202 when nothing responds).
            continue;
        };
        let init_requested = match &request.request {
            ClientRequest::InitializeRequest(init) => {
                Some(init.params.protocol_version.as_str().to_owned())
            }
            _ => None,
        };
        let (transport, mut receiver) =
            OneshotTransport::<RoleServer>::new(ClientJsonRpcMessage::Request(request));
        let service = match QmdMcpService::new(Arc::clone(&state.qmd)) {
            Ok(s) => s,
            Err(e) => {
                drop(writeln!(std::io::stderr(), "HTTP handler error: {e}"));
                return with_label(
                    rpc_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        -32603,
                        "Internal error",
                        &Value::Null,
                        None,
                        true,
                    ),
                    label,
                );
            }
        };
        let supported: Vec<ProtocolVersion> =
            Service::supported_protocol_versions(&service).into_owned();
        let peer_info = peer_info_for(&el, headers, &supported);
        let _running = serve_directly(service, transport, peer_info);
        let mut out: Option<Value> = None;
        while let Some(out_msg) = receiver.recv().await {
            let Ok(mut v) = serde_json::to_value(&out_msg) else {
                continue;
            };
            // Initialize negotiation: echo a requested version the server
            // can serve over the legacy handshake, else fall back to the
            // newest legacy version (recorded: 2026-07-28 → 2025-11-25,
            // mirroring rmcp's `negotiate_protocol_version`).
            if let Some(requested) = &init_requested {
                negotiate_init_version(&mut v, requested, &supported);
            }
            out = Some(v);
            break;
        }
        let Some(response) = out else {
            continue;
        };
        events.push(format!(
            "data: {}",
            serde_json::to_string(&response).unwrap_or_default()
        ));
        results.push(response);
    }
    assemble(&events, results, modern, label)
}

/// Initialize fallback — echo the requested `protocolVersion` when the
/// server can serve it over the legacy handshake (2026-07-28 excluded:
/// that era never initializes), else answer `LATEST_WITH_INITIALIZE`.
fn negotiate_init_version(v: &mut Value, requested: &str, supported: &[ProtocolVersion]) {
    let Some(pv) = v
        .get_mut("result")
        .and_then(|r| r.get_mut("protocolVersion"))
    else {
        return;
    };
    let servable = supported
        .iter()
        .any(|s| s.as_str() == requested && *s != ProtocolVersion::V_2026_07_28);
    if !servable {
        *pv = json!(ProtocolVersion::LATEST_WITH_INITIALIZE.as_str());
    }
}

/// Response shape — 202 for all-notification posts, `application/json`
/// for the 2026 era, `text/event-stream` frames for the legacy era
/// (recorded headers verbatim).
#[expect(
    clippy::expect_used,
    reason = "static headers/bodies cannot fail to build"
)]
fn assemble(events: &[String], mut results: Vec<Value>, modern: bool, label: &str) -> Response {
    if results.is_empty() {
        return with_label(
            Response::builder()
                .status(StatusCode::ACCEPTED)
                .body(Body::empty())
                .expect("static"),
            label,
        );
    }
    let resp = if modern {
        // Single response object, batch array for several.
        let payload = if results.len() == 1 {
            results.remove(0)
        } else {
            Value::Array(results)
        };
        json_response(StatusCode::OK, &payload, false)
    } else {
        let mut text = String::new();
        for d in events {
            text.push_str("event: message\n");
            text.push_str(d);
            text.push_str("\n\n");
        }
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache, no-transform")
            .header("x-accel-buffering", "no")
            .header(header::CONNECTION, "keep-alive")
            .body(Body::from(text))
            .expect("sse")
    };
    with_label(resp, label)
}
