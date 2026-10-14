//! Read-only MCP server that exposes demo CRM users and support tickets.

mod store;

use anyhow::Context;
use rmcp::{
    handler::server::wrapper::{Json, Parameters},
    model::{Implementation, ServerCapabilities, ServerConfig},
    schemars::{self, JsonSchema},
    tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    },
    ServerHandler,
};
use serde::{Deserialize, Serialize};
use store::{Store, Ticket, User};

const DEFAULT_ADDR: &str = "127.0.0.1:8094";
const DEFAULT_DATA_DIR: &str = "support/data";

#[derive(Debug, Deserialize, JsonSchema)]
struct IdParams {
    /// Exact identifier, for example TCK-1001 or USR-001.
    id: String,
}

#[derive(Debug, Serialize, JsonSchema)]
struct TicketContext {
    ticket: Ticket,
    user: User,
}

#[derive(Debug, Serialize, JsonSchema)]
struct TicketList {
    user: User,
    tickets: Vec<Ticket>,
}

#[derive(Clone)]
struct SupportServer {
    store: Store,
}

#[tool_router]
impl SupportServer {
    #[tool(
        name = "get_ticket",
        description = "Return a support ticket together with its customer profile. Use this first whenever the user mentions a ticket id, so the answer accounts for account status, plan, environment and ticket history.",
        annotations(read_only_hint = true)
    )]
    async fn get_ticket(
        &self,
        Parameters(p): Parameters<IdParams>,
    ) -> Result<Json<TicketContext>, String> {
        let ticket = self
            .store
            .ticket(&p.id)
            .ok_or_else(|| format!("тикет {} не найден", p.id))?;
        let user = self
            .store
            .user(&ticket.user_id)
            .ok_or_else(|| format!("пользователь {} не найден", ticket.user_id))?;
        Ok(Json(TicketContext { ticket, user }))
    }

    #[tool(
        name = "get_user",
        description = "Return a CRM customer profile by exact user id.",
        annotations(read_only_hint = true)
    )]
    async fn get_user(&self, Parameters(p): Parameters<IdParams>) -> Result<Json<User>, String> {
        self.store
            .user(&p.id)
            .map(Json)
            .ok_or_else(|| format!("пользователь {} не найден", p.id))
    }

    #[tool(
        name = "list_user_tickets",
        description = "Return the customer profile and all their tickets, newest first. Use it when earlier incidents may explain the current problem.",
        annotations(read_only_hint = true)
    )]
    async fn list_user_tickets(
        &self,
        Parameters(p): Parameters<IdParams>,
    ) -> Result<Json<TicketList>, String> {
        let user = self
            .store
            .user(&p.id)
            .ok_or_else(|| format!("пользователь {} не найден", p.id))?;
        let tickets = self.store.user_tickets(&p.id);
        Ok(Json(TicketList { user, tickets }))
    }
}

#[tool_handler]
impl ServerHandler for SupportServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("support-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Read-only CRM for user support. If a request contains a ticket id, call get_ticket before answering. Use list_user_tickets only when incident history is relevant. Combine CRM facts with the documentation supplied by RAG; never invent missing customer data.",
            )
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let addr = std::env::var("SUPPORT_MCP_ADDR").unwrap_or_else(|_| DEFAULT_ADDR.into());
    let data_dir = std::env::var("SUPPORT_DATA_DIR").unwrap_or_else(|_| DEFAULT_DATA_DIR.into());
    let store = Store::load(&data_dir)?;
    eprintln!("support-mcp: данные из {data_dir}");

    let service = StreamableHttpService::new(
        move || Ok(SupportServer { store: store.clone() }),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );
    let router = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("не удалось занять адрес {addr}"))?;
    eprintln!("support-mcp: Streamable HTTP на http://{addr}/mcp");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
