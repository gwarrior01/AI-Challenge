//! MCP-сервер для профилирования Java-приложений, доступный по Streamable HTTP.
//!
//! Запуск: `cargo run -p java-profiler-mcp` — сервер слушает
//! `http://127.0.0.1:8091/mcp` (адрес меняется переменной
//! `JAVA_PROFILER_MCP_ADDR`). Работает с JVM на этой машине через утилиты JDK
//! (`jps`, `jcmd`, `jfr`) и ничего в них не меняет.

mod jvm;
mod parse;

use std::time::Duration;

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

use jvm::{Jdk, Jvm, Target};
use parse::{Histogram, JvmProcess, ThreadDumpSummary};

const DEFAULT_ADDR: &str = "127.0.0.1:8091";
const DEFAULT_MAX_THREADS: usize = 20;
const MAX_STACK_FRAMES: usize = 12;
const DEFAULT_TOP_CLASSES: usize = 20;
const DEFAULT_PROFILE_SEC: u64 = 10;
const MAX_PROFILE_SEC: u64 = 120;
const DEFAULT_VIEWS: &[&str] = &["hot-methods", "allocation-by-class", "contention-by-site", "gc-pauses"];
const MAX_VIEWS: usize = 8;
const MAX_VIEW_LINES: usize = 40;

#[derive(Debug, Deserialize, JsonSchema)]
struct PidParams {
    /// pid Java-процесса (из java_list_processes).
    pid: u32,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ThreadDumpParams {
    /// pid Java-процесса (из java_list_processes).
    pid: u32,
    /// Сколько потоков показать (по убыванию потреблённого CPU), по умолчанию 20.
    #[serde(default)]
    max_threads: Option<usize>,
    /// Показывать и служебные потоки JVM (GC, компилятор, Finalizer …). По умолчанию нет.
    #[serde(default)]
    include_system: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct HistogramParams {
    /// pid Java-процесса (из java_list_processes).
    pid: u32,
    /// Сколько классов показать (по убыванию занятой памяти), по умолчанию 20.
    #[serde(default)]
    top: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ProfileParams {
    /// pid Java-процесса (из java_list_processes).
    pid: u32,
    /// Длительность записи в секундах: от 1 до 120, по умолчанию 10. Столько же
    /// длится вызов инструмента.
    #[serde(default)]
    duration_sec: Option<u64>,
    /// Какие сводки `jfr view` построить по записи. По умолчанию hot-methods,
    /// allocation-by-class, contention-by-site, gc-pauses. Полезны также
    /// cpu-time-hot-methods, allocation-by-site, exception-by-type, gc,
    /// thread-cpu-load, latencies-by-type, memory-leaks-by-class.
    #[serde(default)]
    views: Option<Vec<String>>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct ProcessList {
    processes: Vec<JvmProcess>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct ProcessInfo {
    pid: u32,
    /// Версия JVM и JDK.
    version: String,
    /// Время работы JVM.
    uptime: String,
    /// Сборщик мусора и заполненность кучи.
    heap: String,
    /// Флаги JVM, отличные от значений по умолчанию.
    flags: Vec<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct ViewOutput {
    view: String,
    /// Текстовая таблица `jfr view` или сообщение об ошибке построения сводки.
    output: String,
}

#[derive(Debug, Serialize, JsonSchema)]
struct ProfileResult {
    pid: u32,
    duration_sec: u64,
    views: Vec<ViewOutput>,
}

#[derive(Clone)]
struct ProfilerServer {
    jdk: Jdk,
}

#[tool_router]
impl ProfilerServer {
    #[tool(
        name = "java_list_processes",
        description = "Список Java-процессов (JVM), запущенных на этой машине под текущим пользователем: \
            pid, главный класс или jar, аргументы программы и флаги JVM. С него начинается \
            профилирование — остальным инструментам нужен pid.",
        annotations(read_only_hint = true)
    )]
    async fn list_processes(&self) -> Result<Json<ProcessList>, String> {
        let processes = self.jdk.list_local_jvms().await.map_err(error_text)?;
        Ok(Json(ProcessList { processes }))
    }

    #[tool(
        name = "java_process_info",
        description = "Общие сведения о JVM: версия, время работы, сборщик мусора и заполненность \
            кучи, флаги JVM, отличные от значений по умолчанию (-Xmx, выбранный GC и т. п.).",
        annotations(read_only_hint = true)
    )]
    async fn process_info(&self, Parameters(PidParams { pid }): Parameters<PidParams>) -> Result<Json<ProcessInfo>, String> {
        let jvm = self.jdk.connect(Target::Local { pid });
        // По очереди: одновременные jcmd к одной JVM всё равно выстроятся в
        // очередь (см. Jdk::attach_lock).
        let version = jvm.diagnostic_command("VM.version", &[]).await.map_err(error_text)?;
        let uptime = jvm.diagnostic_command("VM.uptime", &[]).await.map_err(error_text)?;
        let heap = jvm.diagnostic_command("GC.heap_info", &[]).await.map_err(error_text)?;
        let flags = jvm.diagnostic_command("VM.flags", &[]).await.map_err(error_text)?;
        Ok(Json(ProcessInfo {
            pid,
            version,
            uptime,
            heap,
            flags: flags.split_whitespace().map(str::to_string).collect(),
        }))
    }

    #[tool(
        name = "java_thread_dump",
        description = "Снимок потоков JVM (Thread.print): число потоков по состояниям (RUNNABLE, \
            BLOCKED, WAITING …), найденные взаимные блокировки (deadlock) и стеки самых \
            загруженных по CPU потоков с информацией о захваченных и ожидаемых мониторах. \
            Помогает понять, чем заняты потоки прямо сейчас, где они зависли или ждут блокировку.",
        annotations(read_only_hint = true)
    )]
    async fn thread_dump(
        &self,
        Parameters(params): Parameters<ThreadDumpParams>,
    ) -> Result<Json<ThreadDumpSummary>, String> {
        let jvm = self.jdk.connect(Target::Local { pid: params.pid });
        let text = jvm.diagnostic_command("Thread.print", &[]).await.map_err(error_text)?;
        let max_threads = params.max_threads.unwrap_or(DEFAULT_MAX_THREADS).max(1);
        Ok(Json(parse::thread_dump(&text, max_threads, MAX_STACK_FRAMES, params.include_system)))
    }

    #[tool(
        name = "java_heap_histogram",
        description = "Гистограмма кучи (GC.class_histogram): какие классы занимают больше всего памяти — \
            число экземпляров и байты, плюс итог по куче. Помогает искать утечки и раздутые \
            коллекции. Внимание: перед подсчётом JVM выполняет полную сборку мусора (пауза \
            приложения, на большой куче — заметная).",
        annotations(read_only_hint = true)
    )]
    async fn heap_histogram(&self, Parameters(params): Parameters<HistogramParams>) -> Result<Json<Histogram>, String> {
        let jvm = self.jdk.connect(Target::Local { pid: params.pid });
        let text = jvm.diagnostic_command("GC.class_histogram", &[]).await.map_err(error_text)?;
        Ok(Json(parse::histogram(&text, params.top.unwrap_or(DEFAULT_TOP_CLASSES).max(1))))
    }

    #[tool(
        name = "java_profile",
        description = "Профилирует JVM с помощью Java Flight Recorder: пишет запись заданной длительности \
            (по умолчанию 10 с, вызов длится столько же) и возвращает готовые сводки jfr view — \
            по умолчанию самые горячие методы (hot-methods), что больше всего аллоцируется \
            (allocation-by-class), где потоки ждут мониторы (contention-by-site) и паузы GC \
            (gc-pauses). Главный инструмент, чтобы понять, почему приложение тормозит или \
            ест CPU и память. Накладные расходы JFR невелики (единицы процентов).",
        annotations(read_only_hint = true)
    )]
    async fn profile(&self, Parameters(params): Parameters<ProfileParams>) -> Result<Json<ProfileResult>, String> {
        let duration_sec = params.duration_sec.unwrap_or(DEFAULT_PROFILE_SEC).clamp(1, MAX_PROFILE_SEC);
        let views = match params.views.filter(|v| !v.is_empty()) {
            Some(views) => views,
            None => DEFAULT_VIEWS.iter().map(|v| v.to_string()).collect(),
        };
        if views.len() > MAX_VIEWS {
            return Err(format!("не больше {MAX_VIEWS} сводок за вызов"));
        }
        if let Some(bad) = views.iter().find(|v| !is_view_name(v)) {
            return Err(format!("некорректное имя сводки «{bad}»: ожидается имя вроде hot-methods"));
        }

        let jvm = self.jdk.connect(Target::Local { pid: params.pid });
        let recording = jvm.record_jfr(Duration::from_secs(duration_sec)).await.map_err(error_text)?;
        let mut outputs = Vec::with_capacity(views.len());
        for view in views {
            let output = match self.jdk.jfr_view(recording.path(), &view).await {
                Ok(text) => parse::compact_view(&text, MAX_VIEW_LINES),
                Err(e) => format!("ошибка: {e:#}"),
            };
            outputs.push(ViewOutput { view, output });
        }
        Ok(Json(ProfileResult { pid: params.pid, duration_sec, views: outputs }))
    }
}

#[tool_handler]
impl ServerHandler for ProfilerServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("java-profiler-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Профилирование Java-приложений на этой машине. Начните с java_list_processes, \
                 чтобы узнать pid; java_profile показывает, на что уходят CPU, память и ожидание \
                 блокировок, java_thread_dump — что потоки делают прямо сейчас.",
            )
    }
}

/// Имя сводки `jfr view` или типа события: латиница, цифры, «-», «.» и «_».
fn is_view_name(view: &str) -> bool {
    !view.is_empty()
        && !view.starts_with('-')
        && view.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_'))
}

fn error_text(error: anyhow::Error) -> String {
    format!("{error:#}")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let addr = std::env::var("JAVA_PROFILER_MCP_ADDR").unwrap_or_else(|_| DEFAULT_ADDR.into());
    let jdk = Jdk::detect();
    eprintln!("java-profiler-mcp: {}", jdk.describe());

    let service = StreamableHttpService::new(
        move || Ok(ProfilerServer { jdk: jdk.clone() }),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );
    let router = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("не удалось занять адрес {addr}"))?;

    eprintln!("java-profiler-mcp: Streamable HTTP на http://{addr}/mcp");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
