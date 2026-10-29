//! rmcp `ServerHandler` — envelope parity with upstream's SDK 2.0
//! server (mcp/server.ts:176-196 for instructions/cacheHints; the
//! protocol envelope verified in `runbooks/p5-mcp-spikes/`).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use qmd::Qmd;
use rmcp::model::{
    CacheScope, CallToolRequestParams, ClientRequest, CustomResult, DiscoverResult, Implementation,
    ListResourceTemplatesResult, ListResourcesResult, ListToolsResult, PaginatedRequestParams,
    ProtocolVersion, ReadResourceRequestParams, ServerCapabilities, ServerConfig, ServerResult,
    Tool,
};
use rmcp::service::{NotificationContext, RequestContext, Service};
use rmcp::{ErrorData, RoleServer, ServerHandler};
use serde_json::{Value, json};
use tokio::sync::Notify;

use crate::{Error, instructions, resource, schema, tools};

/// In-flight handler gate — upstream `InflightGate`
/// (mcp/server.ts:649-709). `serve` waits on it after stdin EOF before
/// dropping the store, bounded by `StdioOptions::drain_timeout`.
#[derive(Default)]
pub(crate) struct Inflight {
    active: AtomicUsize,
    idle: Notify,
}

impl Inflight {
    fn enter(self: &Arc<Self>) -> InflightGuard {
        self.active.fetch_add(1, Ordering::SeqCst);
        InflightGuard {
            gate: Arc::clone(self),
        }
    }

    /// `true` when no handler runs — resolves early or at timeout.
    /// The `Notified` future is created and enabled before re-checking
    /// `active` — the tokio-documented pattern that cannot miss a
    /// wakeup that fires between check and wait.
    pub(crate) async fn wait_for_idle(&self, timeout: std::time::Duration) -> bool {
        if self.active.load(Ordering::SeqCst) == 0 {
            return true;
        }
        tokio::time::timeout(timeout, self.wait_until_idle())
            .await
            .is_ok()
    }

    /// Resolves once `active` reaches 0 (no timeout applied — callers
    /// wrap in [`tokio::time::timeout`]).
    async fn wait_until_idle(&self) {
        loop {
            let notified = self.idle.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.active.load(Ordering::SeqCst) == 0 {
                return;
            }
            notified.await;
        }
    }
}

struct InflightGuard {
    gate: Arc<Inflight>,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        if self.gate.active.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.gate.idle.notify_waiters();
        }
    }
}

/// `true` when this request speaks the 2026-07-28 era: namespaced
/// `_meta` on the request (no-initialize flow) or a negotiated 2026
/// initialize. Upstream emits `ttlMs`/`cacheScope`/`resultType` only
/// on 2026-era responses (transcript-2025.txt vs transcript-2026-spec.txt).
fn is_2026(ctx: &RequestContext<RoleServer>) -> bool {
    ctx.meta
        .0
        .0
        .contains_key("io.modelcontextprotocol/protocolVersion")
        || ctx
            .peer
            .peer_info()
            .is_some_and(|i| i.protocol_version == ProtocolVersion::V_2026_07_28)
}

/// Cache-hint + `_meta` envelope fields — set only for 2026-era peers;
/// rmcp's own negotiation fills `resultType` for 2026 and strips the
/// rest for 2025-era responses. `meta` mirrors upstream's per-result
/// `io.modelcontextprotocol/serverInfo` (transcript-2026-spec.txt).
struct Envelope {
    ttl_ms: u64,
    meta: rmcp::model::MetaObject,
}

fn envelope_2026(ctx: &RequestContext<RoleServer>, ttl_ms: u64) -> Option<Envelope> {
    is_2026(ctx).then(|| Envelope {
        ttl_ms,
        meta: rmcp::model::MetaObject(serde_json::Map::from_iter([(
            "io.modelcontextprotocol/serverInfo".to_owned(),
            json!({"name": "qmd", "version": env!("CARGO_PKG_VERSION")}),
        )])),
    })
}

/// The MCP server: one shared `Arc<Qmd>` per process, matching
/// upstream's single-store-per-process model (server.ts:798-818).
impl std::fmt::Debug for QmdMcpServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QmdMcpServer").finish_non_exhaustive()
    }
}

/// MCP server over one shared `Arc<Qmd>`.
pub(crate) struct QmdMcpServer {
    qmd: Arc<Qmd>,
    info: ServerConfig,
    tools: Vec<Tool>,
    pub(crate) inflight: Arc<Inflight>,
}

impl QmdMcpServer {
    /// Build the server: instructions are read once from the store, as
    /// upstream does at `createMcpServer` time (server.ts:184-196).
    ///
    /// # Errors
    /// [`Error::Store`] on status/context read failure;
    /// [`Error::Serve`] if the pinned JSON stops deserializing.
    pub(crate) fn new(qmd: Arc<Qmd>) -> Result<Self, Error> {
        let mut info = ServerConfig::new(capabilities());
        info.instructions = Some(instructions::build_instructions(&qmd)?);
        info.server_info = Implementation::new("qmd", env!("CARGO_PKG_VERSION"));
        Ok(Self {
            qmd,
            info,
            tools: schema::tools()?,
            inflight: Arc::new(Inflight::default()),
        })
    }
}

/// Upstream declares `listChanged: true` on both capabilities
/// (transcript-2025.txt initialize result).
fn capabilities() -> ServerCapabilities {
    // Typed-state builder: list_changed is offered only after the
    // capability itself is enabled.
    ServerCapabilities::builder()
        .enable_tools()
        .enable_tool_list_changed()
        .enable_resources()
        .enable_resources_list_changed()
        .build()
}

impl ServerHandler for QmdMcpServer {
    fn get_info(&self) -> ServerConfig {
        self.info.clone()
    }

    /// Upstream advertises only the 2026-07-28 version in discover
    /// (transcript-2026-spec.txt) while still negotiating 2025
    /// initialize — so override `discover`, not
    /// `supported_protocol_versions`.
    fn discover(
        &self,
        _ctx: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<DiscoverResult, ErrorData>> + '_ {
        let mut r = DiscoverResult::from_server_info(
            vec![ProtocolVersion::V_2026_07_28],
            self.info.clone(),
        );
        r.ttl_ms = 60_000;
        r.cache_scope = CacheScope::Private;
        std::future::ready(Ok(r))
    }

    /// The four upstream tool definitions verbatim (schema.rs).
    fn list_tools(
        &self,
        _p: Option<PaginatedRequestParams>,
        ctx: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<ListToolsResult, ErrorData>> + '_ {
        let mut r = ListToolsResult {
            tools: self.tools.clone(),
            ..ListToolsResult::default()
        };
        if let Some(env) = envelope_2026(&ctx, 60_000) {
            r.ttl_ms = Some(env.ttl_ms);
            r.cache_scope = Some(CacheScope::Private);
            r.meta = Some(env.meta);
        }
        std::future::ready(Ok(r))
    }

    /// Upstream's template has `list: undefined` — `resources/list`
    /// answers empty (transcript-2025.txt id 4).
    fn list_resources(
        &self,
        _p: Option<PaginatedRequestParams>,
        ctx: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<ListResourcesResult, ErrorData>> + '_ {
        let mut r = ListResourcesResult::default();
        if let Some(env) = envelope_2026(&ctx, 0) {
            r.ttl_ms = Some(env.ttl_ms);
            r.cache_scope = Some(CacheScope::Private);
            r.meta = Some(env.meta);
        }
        std::future::ready(Ok(r))
    }

    fn list_resource_templates(
        &self,
        _p: Option<PaginatedRequestParams>,
        ctx: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<ListResourceTemplatesResult, ErrorData>> + '_ {
        let templates = schema::resource_templates();
        std::future::ready(
            templates
                .map_err(|e| ErrorData::internal_error(e.to_string(), None))
                .map(|t| {
                    let mut r = ListResourceTemplatesResult {
                        resource_templates: t,
                        ..ListResourceTemplatesResult::default()
                    };
                    if let Some(env) = envelope_2026(&ctx, 0) {
                        r.ttl_ms = Some(env.ttl_ms);
                        r.cache_scope = Some(CacheScope::Private);
                        r.meta = Some(env.meta);
                    }
                    r
                }),
        )
    }
}

// ---------------------------------------------------------------------------
// `Service` wrapper — raw-JSON interception
// ---------------------------------------------------------------------------

/// `Service<RoleServer>` implementation wrapping `QmdMcpServer`.
///
/// rmcp's typed `ResourceContents` cannot carry the `name`/`title`
/// fields upstream emits on `resources/read` contents (server.ts:
/// 213-231), so that method is answered with a raw
/// [`ServerResult::CustomResult`] instead — verified to pass through
/// rmcp's envelope post-processing untouched. All other requests
/// delegate to the `ServerHandler` dispatch.
pub struct QmdMcpService {
    inner: QmdMcpServer,
    /// In-flight handler gate shared with the inner handler — `serve`
    /// drains it after stdin EOF.
    pub(crate) inflight: Arc<Inflight>,
}

impl std::fmt::Debug for QmdMcpService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QmdMcpService").finish_non_exhaustive()
    }
}

impl QmdMcpService {
    /// Build the service over one shared `Arc<Qmd>`.
    ///
    /// # Errors
    /// Same as `QmdMcpServer::new`.
    pub fn new(qmd: Arc<Qmd>) -> Result<Self, Error> {
        let inner = QmdMcpServer::new(qmd)?;
        Ok(Self {
            inflight: Arc::clone(&inner.inflight),
            inner,
        })
    }
}

/// Mirrors rmcp's pre-dispatch era validation for methods we answer
/// through [`Service::handle_request`] (rmcp
/// `handler/server.rs:56-97`) — intercepted methods must reject bad
/// `_meta`/versions the same way dispatched ones do.
fn validate_inline_era(
    inner: &QmdMcpServer,
    request: &ClientRequest,
    context: &RequestContext<RoleServer>,
) -> Result<(), ErrorData> {
    let uses_inline_negotiation = !matches!(request, ClientRequest::InitializeRequest(_));
    let requested_version = context.meta.protocol_version();
    if uses_inline_negotiation && let Some(requested_version) = requested_version.as_ref() {
        let supported = ServerHandler::supported_protocol_versions(inner);
        if !supported.contains(requested_version) {
            return Err(ErrorData::unsupported_protocol_version(
                requested_version.clone(),
                &supported,
            ));
        }
    }
    // `request_metadata_required` is crate-private; the equivalent
    // observable state is "no negotiated initialize" — an inline
    // (2026) lifecycle session must carry request `_meta` anyway.
    let requires_metadata = uses_inline_negotiation
        && (matches!(request, ClientRequest::DiscoverRequest(_))
            || context.protocol_version().is_none()
            || requested_version
                .as_ref()
                .is_some_and(|v| v.as_str() >= ProtocolVersion::V_2026_07_28.as_str()));
    if requires_metadata {
        let missing = context
            .meta
            .missing_required_keys(&ProtocolVersion::V_2026_07_28);
        if !missing.is_empty() {
            return Err(ErrorData::invalid_params(
                format!(
                    "request _meta is missing or has malformed required fields: {}",
                    missing.join(", ")
                ),
                None,
            ));
        }
    }
    Ok(())
}

impl QmdMcpServer {
    /// `tools/call` as raw JSON — same `CustomResult` bypass as
    /// `resources/read` (upstream's `resource` content blocks and the
    /// `name`/`title` fields are beyond rmcp's typed result).
    async fn call_tool_custom(
        &self,
        params: &CallToolRequestParams,
        ctx: &RequestContext<RoleServer>,
    ) -> Result<ServerResult, ErrorData> {
        let qmd = Arc::clone(&self.qmd);
        let name = params.name.to_string();
        let args = serde_json::to_value(&params.arguments).unwrap_or(Value::Null);
        // F4: the guard is created before `spawn_blocking` and moved in —
        // the drain gate must count the call between dispatch and start.
        let guard = self.inflight.enter();
        let mut value = tokio::task::spawn_blocking(move || {
            let _g = guard;
            tools::call(&qmd, &name, &args)
        })
        .await
        .map_err(|e| ErrorData::internal_error(e.to_string(), None))??;
        // Upstream stamps resultType + `_meta.serverInfo` on 2026-era
        // tools/call results (transcript-2026-spec.txt).
        if let Some(env) = envelope_2026(ctx, 0)
            && let Some(o) = value.as_object_mut()
        {
            o.insert("resultType".to_owned(), json!("complete"));
            o.insert(
                "_meta".to_owned(),
                serde_json::to_value(env.meta.0)
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))?,
            );
        }
        Ok(ServerResult::CustomResult(CustomResult(value)))
    }

    /// `resources/read` as raw JSON: upstream's contents item carries
    /// `{uri, name, title, mimeType, text}` — beyond the typed
    /// `ResourceContents`. The 2026 envelope keys are added by the
    /// caller's era check, matching upstream's per-request envelope.
    async fn read_resource_custom(
        &self,
        params: &ReadResourceRequestParams,
        ctx: &RequestContext<RoleServer>,
    ) -> Result<ServerResult, ErrorData> {
        let qmd = Arc::clone(&self.qmd);
        let uri = params.uri.clone();
        let guard = self.inflight.enter();
        let mut value = tokio::task::spawn_blocking(move || {
            let _g = guard;
            resource::read(&qmd, &uri)
        })
        .await
        .map_err(|e| ErrorData::internal_error(e.to_string(), None))?
        .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        if let Some(env) = envelope_2026(ctx, 0)
            && let Some(o) = value.as_object_mut()
        {
            o.insert("resultType".to_owned(), json!("complete"));
            o.insert("ttlMs".to_owned(), json!(env.ttl_ms));
            o.insert("cacheScope".to_owned(), json!("private"));
            o.insert(
                "_meta".to_owned(),
                serde_json::to_value(env.meta.0)
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))?,
            );
        }
        Ok(ServerResult::CustomResult(CustomResult(value)))
    }
}

impl Service<RoleServer> for QmdMcpService {
    async fn handle_request(
        &self,
        request: ClientRequest,
        context: RequestContext<RoleServer>,
    ) -> Result<ServerResult, ErrorData> {
        match &request {
            ClientRequest::CallToolRequest(req) => {
                validate_inline_era(&self.inner, &request, &context)?;
                return self.inner.call_tool_custom(&req.params, &context).await;
            }
            ClientRequest::ReadResourceRequest(req) => {
                validate_inline_era(&self.inner, &request, &context)?;
                return self.inner.read_resource_custom(&req.params, &context).await;
            }
            _ => {}
        }
        Service::handle_request(&self.inner, request, context).await
    }

    fn handle_notification(
        &self,
        notification: rmcp::model::ClientNotification,
        context: NotificationContext<RoleServer>,
    ) -> impl Future<Output = Result<(), ErrorData>> + '_ {
        Service::handle_notification(&self.inner, notification, context)
    }

    fn get_info(&self) -> ServerConfig {
        Service::get_info(&self.inner)
    }

    fn supported_protocol_versions(&self) -> std::borrow::Cow<'static, [ProtocolVersion]> {
        Service::supported_protocol_versions(&self.inner)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// The drain gate must not miss the last guard's wakeup —
    /// regression test for the race where `active` hits 0 between the
    /// load and `notified()` registration.
    async fn wait_idle_owned(inflight: Arc<Inflight>) -> bool {
        inflight.wait_for_idle(Duration::from_secs(10)).await
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn wait_for_idle_no_missed_wakeup() {
        for _ in 0..200 {
            let inflight = Arc::new(Inflight::default());
            let guard = inflight.enter();
            let waiter = tokio::spawn(wait_idle_owned(inflight));
            drop(guard);
            assert!(
                tokio::time::timeout(Duration::from_secs(2), waiter)
                    .await
                    .expect("wait_for_idle missed the wakeup")
                    .unwrap(),
            );
        }
    }

    /// Nothing tracked → `wait_for_idle` returns `true` on the fast
    /// path without waiting (upstream `waitForIdle` resolves
    /// immediately, mcp-stdio-lifecycle.test.ts:226).
    #[tokio::test]
    async fn wait_for_idle_returns_true_when_nothing_tracked() {
        let inflight = Inflight::default();
        // A 60 s drain budget must not be consumed: the outer 100 ms
        // bound proves the fast path, not just an eventual `true`.
        let idle = tokio::time::timeout(
            Duration::from_millis(100),
            inflight.wait_for_idle(Duration::from_secs(60)),
        )
        .await;
        assert_eq!(idle, Ok(true));
    }

    /// A blocking handler whose closure panics still releases the
    /// gate — the guard lives inside the closure and drops on unwind
    /// (upstream: a rejecting handler releases the gate,
    /// mcp-stdio-lifecycle.test.ts:257).
    #[tokio::test(flavor = "multi_thread")]
    #[expect(clippy::panic, reason = "the test drives a panicking closure")]
    async fn wait_for_idle_after_panicked_blocking_call() {
        let inflight = Arc::new(Inflight::default());
        let guard = inflight.enter();
        let task = tokio::task::spawn_blocking(move || {
            let _guard = guard;
            panic!("handler failed");
        });
        assert!(
            task.await.is_err(),
            "closure panic should surface as JoinError"
        );
        assert!(
            inflight.wait_for_idle(Duration::from_secs(1)).await,
            "gate stayed held after the panicked closure"
        );
    }

    /// A handler finishing with an error value (not a panic) releases
    /// the gate the same way a success does.
    #[tokio::test(flavor = "multi_thread")]
    async fn wait_for_idle_after_error_result() {
        let inflight = Arc::new(Inflight::default());
        let guard = inflight.enter();
        let result: Result<(), String> = tokio::task::spawn_blocking(move || {
            let _guard = guard;
            Err("handler error".to_owned())
        })
        .await
        .expect("blocking task itself completed");
        assert!(result.is_err());
        assert!(
            inflight.wait_for_idle(Duration::from_secs(1)).await,
            "gate stayed held after the error result"
        );
    }
}
