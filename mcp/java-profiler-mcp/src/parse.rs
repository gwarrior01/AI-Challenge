//! Разбор текстового вывода утилит JDK и диагностических команд HotSpot.
//!
//! Числа в выводе зависят от локали JVM: `cpu=252,14ms` в русской и
//! `cpu=252.14ms` в английской, поэтому дробная часть принимается с любым
//! разделителем.

use std::collections::BTreeMap;

use rmcp::schemars::{self, JsonSchema};
use serde::Serialize;

/// Служебные Java-потоки самой JVM; вместе с потоками без Java-состояния
/// (GC, компилятор, VM Thread) по умолчанию не показываются.
const JVM_SERVICE_THREADS: &[&str] = &[
    "Reference Handler",
    "Finalizer",
    "Signal Dispatcher",
    "Service Thread",
    "Monitor Deflation Thread",
    "Notification Thread",
    "Common-Cleaner",
    "Attach Listener",
    "JFR Recorder Thread",
    "JFR Periodic Tasks",
    "JFR Shutdown Hook",
];
const JVM_SERVICE_THREAD_PREFIXES: &[&str] = &["C1 CompilerThread", "C2 CompilerThread", "JFR "];
const DEADLOCK_MARKER: &str = "Found one Java-level deadlock";
const DEADLOCKS_MARKER: &str = "Found Java-level deadlocks";

#[derive(Debug, PartialEq, Serialize, JsonSchema)]
pub struct JvmProcess {
    pub pid: u32,
    /// Главный класс или путь к jar; `null`, если JVM не сообщила его.
    pub main_class: Option<String>,
    /// Аргументы программы.
    pub arguments: String,
    /// Флаги JVM (-Xmx, -D…, -XX:…).
    pub jvm_flags: String,
}

/// Объединяет вывод `jps -lm` (главный класс и аргументы) и `jps -lv` (флаги
/// JVM) по pid. Сами `jps` и `jcmd` в список не попадают.
pub fn jps(commands: &str, flags: &str) -> Vec<JvmProcess> {
    let flags: BTreeMap<u32, String> = jps_lines(flags).map(|(pid, _, rest)| (pid, rest)).collect();
    jps_lines(commands)
        .filter(|(_, main, _)| {
            !main.as_deref().is_some_and(|m| m.ends_with("sun.tools.jps.Jps") || m.ends_with("sun.tools.jcmd.JCmd"))
        })
        .map(|(pid, main_class, arguments)| JvmProcess {
            pid,
            main_class,
            arguments,
            jvm_flags: flags.get(&pid).cloned().unwrap_or_default(),
        })
        .collect()
}

/// Строки jps: pid, главный класс (если есть) и остаток строки.
fn jps_lines(output: &str) -> impl Iterator<Item = (u32, Option<String>, String)> + '_ {
    output.lines().filter_map(|line| {
        let (pid, rest) = line.trim().split_once(' ').unwrap_or((line.trim(), ""));
        let pid = pid.parse().ok()?;
        let rest = rest.trim();
        if rest.is_empty() || rest.starts_with('-') {
            // «-- process information unavailable» или флаги без главного класса.
            let rest = if rest.starts_with("--") { "" } else { rest };
            return Some((pid, None, rest.to_string()));
        }
        let (main, rest) = rest.split_once(' ').unwrap_or((rest, ""));
        Some((pid, Some(main.to_string()), rest.trim().to_string()))
    })
}

/// Убирает первую строку `<pid>:`, которой jcmd начинает любой вывод.
pub fn strip_jcmd_header(output: &str, pid: u32) -> &str {
    let header = format!("{pid}:");
    let trimmed = output.trim_start();
    match trimmed.split_once('\n') {
        Some((first, rest)) if first.trim() == header => rest.trim(),
        _ if trimmed.trim() == header => "",
        _ => trimmed.trim(),
    }
}

/// Короткое понятное сообщение об ошибке утилиты вместо стека Java-исключения.
pub fn tool_error(tool: &str, stdout: &str, stderr: &str) -> String {
    let text = format!("{stdout}\n{stderr}");
    if text.contains("AttachNotSupportedException") || text.contains("No such process") {
        return "процесс не найден или к нему нельзя подключиться: это не JVM, она запущена \
                другим пользователем или с -XX:+DisableAttachMechanism"
            .into();
    }
    let lines = || text.lines().map(str::trim).filter(|l| !l.is_empty());
    // Первая строка вида «java.lang.IllegalArgumentException: сообщение».
    if let Some(message) = lines()
        .filter(|l| !l.starts_with("at "))
        .find_map(|l| l.split_once("Exception: ").map(|(_, message)| message))
    {
        return message.to_string();
    }
    lines()
        .find(|l| !l.ends_with(':') || l.contains(' '))
        .map(str::to_string)
        .unwrap_or_else(|| format!("{tool} завершился с ошибкой"))
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct ThreadInfo {
    pub name: String,
    /// Состояние Java-потока (RUNNABLE, BLOCKED, WAITING …); `null` у внутренних
    /// потоков JVM.
    pub state: Option<String>,
    pub daemon: bool,
    /// Процессорное время потока с его старта, мс.
    pub cpu_ms: Option<f64>,
    /// Верхние кадры стека вместе со строками о мониторах («- locked …»,
    /// «- waiting to lock …»).
    pub stack: Vec<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct ThreadDumpSummary {
    /// Сколько всего потоков в дампе, включая внутренние потоки JVM.
    pub total_threads: usize,
    /// Число рассмотренных потоков по состояниям.
    pub by_state: BTreeMap<String, usize>,
    /// Описание взаимных блокировок, если JVM их нашла.
    pub deadlocks: Option<String>,
    /// Потоки по убыванию потреблённого процессорного времени.
    pub threads: Vec<ThreadInfo>,
    /// Сколько рассмотренных потоков не вошло в список из-за лимита.
    pub threads_not_shown: usize,
}

/// Сводка по выводу `Thread.print`.
pub fn thread_dump(text: &str, max_threads: usize, max_frames: usize, include_system: bool) -> ThreadDumpSummary {
    let deadlock_start = [DEADLOCK_MARKER, DEADLOCKS_MARKER].iter().filter_map(|m| text.find(m)).min();
    let (threads_text, deadlocks) = match deadlock_start {
        Some(start) => (&text[..start], Some(text[start..].trim().to_string())),
        None => (text, None),
    };

    let all: Vec<ThreadInfo> = threads_text
        .split("\n\n")
        .map(str::trim_start)
        .filter(|block| block.starts_with('"'))
        .filter_map(|block| parse_thread(block, max_frames))
        .collect();
    let total_threads = all.len();

    let mut threads: Vec<ThreadInfo> = all.into_iter().filter(|t| include_system || !is_system_thread(t)).collect();
    let mut by_state = BTreeMap::new();
    for thread in &threads {
        let state = thread.state.as_deref().map_or("—", |s| s.split_whitespace().next().unwrap_or(s));
        *by_state.entry(state.to_string()).or_default() += 1;
    }
    threads.sort_by(|a, b| b.cpu_ms.unwrap_or(0.0).total_cmp(&a.cpu_ms.unwrap_or(0.0)));
    let threads_not_shown = threads.len().saturating_sub(max_threads);
    threads.truncate(max_threads);

    ThreadDumpSummary { total_threads, by_state, deadlocks, threads, threads_not_shown }
}

fn parse_thread(block: &str, max_frames: usize) -> Option<ThreadInfo> {
    let mut lines = block.lines();
    let header = lines.next()?;
    let name_end = header[1..].find('"')? + 1;
    let name = header[1..name_end].to_string();
    let attributes = &header[name_end + 1..];
    let daemon = attributes.split_whitespace().any(|w| w == "daemon");
    let cpu_ms = attributes
        .split_whitespace()
        .find_map(|w| w.strip_prefix("cpu="))
        .and_then(|v| v.strip_suffix("ms"))
        .and_then(parse_decimal);

    let mut state = None;
    let mut stack = Vec::new();
    for line in lines.map(str::trim) {
        if let Some(s) = line.strip_prefix("java.lang.Thread.State:") {
            state = Some(s.trim().to_string());
        } else if line.starts_with("at ") || line.starts_with("- ") {
            stack.push(line.to_string());
        }
    }
    let frames = stack.iter().filter(|l| l.starts_with("at ")).count();
    if frames > max_frames {
        // Обрезаем по числу кадров, сохраняя строки о мониторах внутри них.
        let mut seen = 0;
        let cut = stack
            .iter()
            .position(|l| {
                seen += l.starts_with("at ") as usize;
                seen > max_frames
            })
            .unwrap_or(stack.len());
        stack.truncate(cut);
        stack.push(format!("… ещё {} кадр(ов)", frames - max_frames));
    }
    Some(ThreadInfo { name, state, daemon, cpu_ms, stack })
}

fn is_system_thread(thread: &ThreadInfo) -> bool {
    thread.state.is_none()
        || JVM_SERVICE_THREADS.contains(&thread.name.as_str())
        || JVM_SERVICE_THREAD_PREFIXES.iter().any(|p| thread.name.starts_with(p))
}

#[derive(Debug, PartialEq, Serialize, JsonSchema)]
pub struct ClassEntry {
    /// Имя класса как в JVM: `[B` — byte[], `[Ljava.lang.Object;` — Object[].
    pub class: String,
    pub instances: u64,
    pub bytes: u64,
}

#[derive(Debug, PartialEq, Serialize, JsonSchema)]
pub struct Histogram {
    pub total_instances: u64,
    pub total_bytes: u64,
    /// Классы по убыванию занятой памяти.
    pub classes: Vec<ClassEntry>,
}

/// Разбор `GC.class_histogram`: первые `top` классов и итоговая строка.
pub fn histogram(text: &str, top: usize) -> Histogram {
    let mut classes = Vec::new();
    let (mut total_instances, mut total_bytes) = (0, 0);
    for line in text.lines().map(str::trim) {
        let mut words = line.split_whitespace();
        match words.next() {
            Some("Total") => {
                total_instances = words.next().and_then(|n| n.parse().ok()).unwrap_or(0);
                total_bytes = words.next().and_then(|n| n.parse().ok()).unwrap_or(0);
            }
            Some(rank) if rank.ends_with(':') && rank[..rank.len() - 1].parse::<u32>().is_ok() => {
                let (Some(instances), Some(bytes), Some(class)) = (
                    words.next().and_then(|n| n.parse().ok()),
                    words.next().and_then(|n| n.parse().ok()),
                    words.next(),
                ) else {
                    continue;
                };
                if classes.len() < top {
                    classes.push(ClassEntry { class: class.to_string(), instances, bytes });
                }
            }
            _ => {}
        }
    }
    Histogram { total_instances, total_bytes, classes }
}

/// Приводит вывод `jfr view` к компактному виду: без хвостовых пробелов и
/// повторных пустых строк, не длиннее `max_lines` строк.
pub fn compact_view(text: &str, max_lines: usize) -> String {
    let mut lines: Vec<&str> = Vec::new();
    for line in text.lines().map(str::trim_end) {
        if line.is_empty() && lines.last().is_none_or(|l| l.is_empty()) {
            continue;
        }
        lines.push(line);
    }
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    if lines.len() > max_lines {
        let rest = lines.len() - max_lines;
        lines.truncate(max_lines);
        return format!("{}\n… ещё {rest} строк", lines.join("\n"));
    }
    lines.join("\n")
}

/// Дробное число с точкой или запятой: «252,14» и «252.14».
fn parse_decimal(value: &str) -> Option<f64> {
    value.replace(',', ".").parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const THREAD_DUMP: &str = include_str!("testdata/thread_dump.txt");
    const THREAD_DUMP_DEADLOCK: &str = include_str!("testdata/thread_dump_deadlock.txt");
    const HISTOGRAM: &str = include_str!("testdata/histogram.txt");

    #[test]
    fn merges_jps_outputs() {
        let commands = "\
88681 jdk.jcmd/sun.tools.jps.Jps -lm
88666 jdk.compiler/com.sun.tools.javac.launcher.SourceLauncher Busy.java
90001 /opt/app/service.jar --port 8080
90002 -- process information unavailable
";
        let flags = "\
88666 jdk.compiler/com.sun.tools.javac.launcher.SourceLauncher --add-modules=ALL-DEFAULT
90001 /opt/app/service.jar -Xmx512m -Denv=prod
";
        let processes = jps(commands, flags);
        assert_eq!(
            processes,
            vec![
                JvmProcess {
                    pid: 88666,
                    main_class: Some("jdk.compiler/com.sun.tools.javac.launcher.SourceLauncher".into()),
                    arguments: "Busy.java".into(),
                    jvm_flags: "--add-modules=ALL-DEFAULT".into(),
                },
                JvmProcess {
                    pid: 90001,
                    main_class: Some("/opt/app/service.jar".into()),
                    arguments: "--port 8080".into(),
                    jvm_flags: "-Xmx512m -Denv=prod".into(),
                },
                JvmProcess { pid: 90002, main_class: None, arguments: String::new(), jvm_flags: String::new() },
            ]
        );
    }

    #[test]
    fn strips_jcmd_header() {
        assert_eq!(strip_jcmd_header("88666:\n12,359 s\n", 88666), "12,359 s");
        assert_eq!(strip_jcmd_header("88666:\n", 88666), "");
        assert_eq!(strip_jcmd_header("other\n", 88666), "other");
    }

    #[test]
    fn explains_tool_errors() {
        let attach = "99999:\ncom.sun.tools.attach.AttachNotSupportedException: pid: 99999, state is not ready\n\tat jdk.attach/...";
        assert!(tool_error("jcmd", attach, "").starts_with("процесс не найден"));
        let unknown = "88666:\njava.lang.IllegalArgumentException: Unknown diagnostic command\n";
        assert_eq!(tool_error("jcmd", unknown, ""), "Unknown diagnostic command");
        let jfr = "jfr view: Could not find a view or an event type named foo\n";
        assert_eq!(tool_error("jfr", "", jfr), "jfr view: Could not find a view or an event type named foo");
    }

    #[test]
    fn summarizes_application_threads() {
        let summary = thread_dump(THREAD_DUMP, 3, 5, false);
        assert_eq!(summary.total_threads, 32);
        assert_eq!(summary.by_state.get("RUNNABLE"), Some(&3));
        assert_eq!(summary.by_state.get("BLOCKED"), Some(&2));
        assert_eq!(summary.by_state.get("WAITING"), Some(&1));
        assert!(summary.deadlocks.is_none());

        let names: Vec<_> = summary.threads.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["hot-loop", "allocator", "worker-1"]);
        assert_eq!(summary.threads_not_shown, 3);

        let hot = &summary.threads[0];
        assert_eq!(hot.state.as_deref(), Some("RUNNABLE"));
        assert!(hot.daemon);
        assert_eq!(hot.cpu_ms, Some(11439.88));
        assert!(hot.stack[0].starts_with("at Busy.isPrime"));

        let blocked = &summary.threads[2];
        assert!(blocked.stack.iter().any(|l| l.starts_with("- waiting to lock")));
    }

    #[test]
    fn includes_system_threads_on_request() {
        let summary = thread_dump(THREAD_DUMP, 100, 5, true);
        assert_eq!(summary.threads.len(), 32);
        assert_eq!(summary.by_state.get("—"), Some(&16));
    }

    #[test]
    fn limits_stack_frames() {
        let summary = thread_dump(THREAD_DUMP, 100, 2, false);
        let main = summary.threads.iter().find(|t| t.name == "main").unwrap();
        // Два кадра, строки о мониторах между ними и пометка об обрезке.
        assert_eq!(main.stack.iter().filter(|l| l.starts_with("at ")).count(), 2);
        assert!(main.stack.last().unwrap().starts_with("… ещё"));
    }

    #[test]
    fn extracts_deadlocks() {
        let summary = thread_dump(THREAD_DUMP_DEADLOCK, 20, 10, false);
        let deadlocks = summary.deadlocks.expect("в дампе есть deadlock");
        assert!(deadlocks.starts_with("Found one Java-level deadlock"));
        assert!(deadlocks.contains("which is held by \"right\""));
        assert!(deadlocks.ends_with("Found 1 deadlock."));
        // Раздел о deadlock'е не должен давать «лишних» потоков.
        assert_eq!(summary.threads.iter().filter(|t| t.name == "left").count(), 1);
        assert_eq!(summary.by_state.get("BLOCKED"), Some(&2));
    }

    #[test]
    fn parses_histogram() {
        let histogram = histogram(HISTOGRAM, 2);
        assert_eq!(histogram.total_instances, 107271);
        assert_eq!(histogram.total_bytes, 49810064);
        assert_eq!(
            histogram.classes,
            vec![
                ClassEntry { class: "[B".into(), instances: 36856, bytes: 46894568 },
                ClassEntry { class: "java.lang.String".into(), instances: 26695, bytes: 640680 },
            ]
        );
    }

    #[test]
    fn compacts_views() {
        let view = "\n  Title  \n\n\n\nrow 1   \nrow 2\nrow 3\n\n";
        assert_eq!(compact_view(view, 10), "  Title\n\nrow 1\nrow 2\nrow 3");
        assert_eq!(compact_view(view, 3), "  Title\n\nrow 1\n… ещё 2 строк");
    }

    #[test]
    fn parses_decimals_in_any_locale() {
        assert_eq!(parse_decimal("252,14"), Some(252.14));
        assert_eq!(parse_decimal("252.14"), Some(252.14));
    }
}
