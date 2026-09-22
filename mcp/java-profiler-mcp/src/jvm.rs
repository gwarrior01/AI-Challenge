//! Доступ к JVM: запуск утилит JDK и подключение к процессам.
//!
//! Все диагностические данные берутся диагностическими командами HotSpot
//! (`Thread.print`, `GC.class_histogram`, `JFR.start` и т. д.). Локально их
//! выполняет `jcmd` через attach-механизм. Удалённая JVM выполняет те же команды
//! через MBean `com.sun.management.DiagnosticCommand` по JMX и отдаёт тот же
//! текст, поэтому разбор вывода (`parse`) от способа подключения не зависит:
//! для удалённых JVM достаточно добавить вариант в [`Target`] и реализацию
//! трейта [`Jvm`].

use std::ffi::OsStr;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::parse::{self, JvmProcess};

/// Сколько ждём обычную команду jcmd/jps/jfr.
const TOOL_TIMEOUT: Duration = Duration::from_secs(60);
/// Запас, после которого запись JFR остановится сама, если сервер не дождался
/// её окончания (например, клиент отменил вызов инструмента).
const JFR_SAFETY_MARGIN: Duration = Duration::from_secs(30);

/// Какой JVM касается вызов. Пока поддерживаются только процессы на этой
/// машине; удалённые JVM появятся здесь отдельным вариантом (например,
/// `Remote { jmx_url }`) со своей реализацией [`Jvm`].
#[derive(Debug, Clone, Copy)]
pub enum Target {
    Local { pid: u32 },
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Target::Local { pid } => write!(f, "pid {pid}"),
        }
    }
}

/// Одна JVM, к которой подключился сервер.
pub trait Jvm {
    /// Выполняет диагностическую команду HotSpot и возвращает её текстовый вывод.
    async fn diagnostic_command(&self, command: &str, args: &[&str]) -> Result<String>;

    /// Записывает профиль JFR (настройки `profile`) в течение `duration` и
    /// возвращает файл записи на этой машине.
    async fn record_jfr(&self, duration: Duration) -> Result<JfrFile>;
}

/// Файл записи JFR во временном каталоге; удаляется, когда больше не нужен.
pub struct JfrFile {
    path: PathBuf,
}

impl JfrFile {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for JfrFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Утилиты JDK этой машины: `jps`, `jcmd`, `jfr`.
#[derive(Debug, Clone)]
pub struct Jdk {
    /// Каталог `bin` из `JAVA_HOME`; `None` — утилиты ищутся в `PATH`.
    bin: Option<PathBuf>,
    /// jcmd запускаются по одному: при первом подключении к JVM каждый создаёт
    /// файл `.attach_pid<pid>`, и одновременные подключения падают с
    /// «java.io.IOException: File exists». Общий для всех копий `Jdk`, то есть
    /// для всех сессий сервера.
    attach_lock: Arc<Mutex<()>>,
}

impl Jdk {
    pub fn detect() -> Self {
        let bin = std::env::var_os("JAVA_HOME")
            .map(|home| PathBuf::from(home).join("bin"))
            .filter(|bin| bin.join(exe("jcmd")).is_file());
        Self { bin, attach_lock: Arc::default() }
    }

    /// Откуда берутся утилиты — для сообщения при запуске сервера.
    pub fn describe(&self) -> String {
        match &self.bin {
            Some(bin) => format!("утилиты JDK из {}", bin.display()),
            None => "утилиты JDK из PATH".into(),
        }
    }

    pub fn connect(&self, target: Target) -> LocalJvm {
        match target {
            Target::Local { pid } => LocalJvm { jdk: self.clone(), pid },
        }
    }

    /// JVM, запущенные на этой машине под текущим пользователем.
    pub async fn list_local_jvms(&self) -> Result<Vec<JvmProcess>> {
        // `-lm` даёт главный класс с аргументами программы, `-lv` — с флагами
        // JVM; в одной строке их не различить, поэтому запросов два.
        let (commands, flags) = tokio::try_join!(self.run("jps", ["-lm"]), self.run("jps", ["-lv"]))?;
        Ok(parse::jps(&commands, &flags))
    }

    /// Готовая текстовая сводка `jfr view` по файлу записи.
    pub async fn jfr_view(&self, file: &Path, view: &str) -> Result<String> {
        self.run("jfr", [OsStr::new("view"), OsStr::new("--width"), OsStr::new("120"), OsStr::new(view), file.as_os_str()])
            .await
    }

    /// Запускает утилиту JDK и возвращает её stdout; при ошибке — понятное сообщение.
    async fn run<I, S>(&self, tool: &str, args: I) -> Result<String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let program = match &self.bin {
            Some(bin) => bin.join(exe(tool)),
            None => PathBuf::from(exe(tool)),
        };
        let child = Command::new(&program)
            .args(args)
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .output();
        let output = match tokio::time::timeout(TOOL_TIMEOUT, child).await {
            Err(_) => bail!("{tool} не ответил за {} с", TOOL_TIMEOUT.as_secs()),
            Ok(Err(e)) if e.kind() == std::io::ErrorKind::NotFound => bail!(
                "не найдена утилита {tool}: установите JDK и добавьте его bin в PATH или задайте JAVA_HOME"
            ),
            Ok(result) => result.with_context(|| format!("не удалось запустить {}", program.display()))?,
        };
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("{}", parse::tool_error(tool, &stdout, &stderr));
        }
        Ok(stdout)
    }
}

/// JVM на этой машине, команды к ней выполняет `jcmd`.
pub struct LocalJvm {
    jdk: Jdk,
    pid: u32,
}

impl Jvm for LocalJvm {
    async fn diagnostic_command(&self, command: &str, args: &[&str]) -> Result<String> {
        let pid = self.pid.to_string();
        let _attach = self.jdk.attach_lock.lock().await;
        let output = self
            .jdk
            .run("jcmd", [pid.as_str(), command].into_iter().chain(args.iter().copied()))
            .await
            .map_err(|e| anyhow::anyhow!("JVM pid {}: {e}", self.pid))?;
        Ok(parse::strip_jcmd_header(&output, self.pid).to_string())
    }

    async fn record_jfr(&self, duration: Duration) -> Result<JfrFile> {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
        let name = format!("java-profiler-mcp-{stamp}");
        let file = JfrFile {
            path: std::env::temp_dir().join(format!("{name}-{}.jfr", self.pid)),
        };
        // Файл пишет сама целевая JVM, поэтому путь должен быть абсолютным.
        let name_arg = format!("name={name}");
        let filename_arg = format!("filename={}", file.path.display());
        let guard_arg = format!("duration={}s", (duration + JFR_SAFETY_MARGIN).as_secs());
        self.diagnostic_command("JFR.start", &[&name_arg, "settings=profile", &guard_arg, &filename_arg])
            .await?;
        tokio::time::sleep(duration).await;
        // Остановленная запись сама пишется в файл, заданный при старте.
        self.diagnostic_command("JFR.stop", &[&name_arg]).await?;
        if !file.path.is_file() {
            bail!("JVM pid {} не записала файл JFR {}", self.pid, file.path.display());
        }
        Ok(file)
    }
}

fn exe(tool: &str) -> String {
    format!("{tool}{}", std::env::consts::EXE_SUFFIX)
}
