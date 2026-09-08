//! Агент — самостоятельная сущность поверх [`LlmClient`]: у неё есть своя
//! конфигурация (системный промпт, модель, параметры генерации, показывать ли
//! токены) и жизненный цикл (запущен/остановлен). Запрос пользователя
//! обрабатывается агентом, а не единичным вызовом клиента напрямую — пока
//! агент остановлен, он отказывается отвечать.
//!
//! Конфигурация агентов и история их диалогов хранятся в SQLite (см. [`Db`]),
//! поэтому они переживают перезапуск приложения: агент, запущенный до
//! перезапуска, при следующем старте снова видит все прежние сообщения и
//! продолжает диалог, как будто его не выключали. Удаление агента удаляет и
//! его историю (каскадно, через `ON DELETE CASCADE`).

use crate::{ChatMessage, ChatOptions, LlmClient, Usage};
use anyhow::{anyhow, bail, Context, Result};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

/// Настраиваемое поведение агента: системный промпт, модель и параметры
/// генерации, а также флаг вывода количества использованных токенов.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Модель для запросов этого агента; если не задана — берётся модель клиента (LLM_MODEL).
    #[serde(default)]
    pub model: Option<String>,
    /// Если true — в текст ответа агента добавляется строка с расходом токенов.
    #[serde(default)]
    pub show_tokens: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub reasoning: Option<bool>,
}

impl AgentConfig {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            system_prompt: None,
            model: None,
            show_tokens: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            reasoning: None,
        }
    }
}

/// Снимок состояния агента для отображения в интерфейсах и для сохранения на диск.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub config: AgentConfig,
    pub running: bool,
}

/// Результат обработки запроса агентом: итоговый текст (уже с учётом
/// `show_tokens`) и сырые метрики расхода токенов.
#[derive(Debug, Clone)]
pub struct AgentReply {
    pub text: String,
    pub usage: Option<Usage>,
}

/// Агент — отдельная сущность, инкапсулирующая обращение к LLM через API.
/// Хранит свою конфигурацию и состояние запуска; пока агент не запущен,
/// обработка запросов отклоняется без обращения к API.
pub struct Agent {
    name: String,
    client: LlmClient,
    config: RwLock<AgentConfig>,
    running: AtomicBool,
    /// Накопленная история диалога (без системного промпта — тот добавляется
    /// заново из актуальной конфигурации при каждом запросе), восстановленная
    /// из SQLite при создании агента. Каждое новое сообщение дописывается и в
    /// эту память, и в БД — без этого при перезапуске приложения агент забывал
    /// бы прошлые реплики и начинал разговор с чистого листа.
    history: Mutex<Vec<ChatMessage>>,
    db: Arc<Db>,
}

impl Agent {
    fn new(config: AgentConfig, client: LlmClient, db: Arc<Db>, history: Vec<ChatMessage>) -> Self {
        let name = config.name.clone();
        Self {
            name,
            client,
            config: RwLock::new(config),
            running: AtomicBool::new(false),
            history: Mutex::new(history),
            db,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn config(&self) -> AgentConfig {
        self.config.read().expect("конфигурация агента отравлена паникой").clone()
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Текущая история диалога (для отображения в интерфейсах, например при
    /// открытии чата с агентом после перезапуска приложения).
    pub fn history(&self) -> Vec<ChatMessage> {
        self.history.lock().expect("история агента отравлена паникой").clone()
    }

    /// Запускает агента. История диалога не сбрасывается — она хранится в
    /// SQLite и переживает и остановку/запуск, и перезапуск всего приложения,
    /// так что диалог продолжается так, будто агент не выключался.
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    pub fn info(&self) -> AgentInfo {
        AgentInfo { config: self.config(), running: self.is_running() }
    }

    /// Принимает запрос пользователя и обращается к LLM через API, добавляя его
    /// в контекст текущего диалога вместе с историей предыдущих сообщений. Если
    /// агент остановлен, запрос отклоняется без обращения к сети.
    pub async fn handle_request(&self, prompt: &str) -> Result<AgentReply> {
        if !self.is_running() {
            bail!("агент «{}» остановлен — сначала запустите его", self.name);
        }

        let config = self.config();
        let mut messages = Vec::new();
        if let Some(system_prompt) = &config.system_prompt {
            if !system_prompt.trim().is_empty() {
                messages.push(ChatMessage::system(system_prompt.clone()));
            }
        }
        {
            let history = self.history.lock().expect("история агента отравлена паникой");
            messages.extend(history.iter().cloned());
        }
        messages.push(ChatMessage::user(prompt));

        let options = ChatOptions {
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            reasoning: config.reasoning,
            ..ChatOptions::default()
        };

        let model = config.model.clone().unwrap_or_else(|| self.client.model().to_string());
        let completion = self.client.chat_with_model(&model, &messages, &options).await?;

        let user_message = ChatMessage::user(prompt.to_string());
        let assistant_message = ChatMessage::assistant(completion.content.clone());
        {
            let mut history = self.history.lock().expect("история агента отравлена паникой");
            history.push(user_message.clone());
            history.push(assistant_message.clone());
        }
        // Лучшая попытка: ответ уже получен и не должен потеряться для
        // пользователя из-за сбоя записи в БД — при ошибке лишь предупреждаем.
        if let Err(err) = self.db.append_message(&self.name, &user_message) {
            eprintln!("не удалось сохранить сообщение пользователя в БД: {err:#}");
        }
        if let Err(err) = self.db.append_message(&self.name, &assistant_message) {
            eprintln!("не удалось сохранить ответ агента в БД: {err:#}");
        }

        let mut text = completion.content;
        if config.show_tokens {
            if let Some(usage) = completion.usage {
                text.push_str(&format!(
                    "\n\n[токены: запрос {} + ответ {} = всего {}]",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                ));
            }
        }

        Ok(AgentReply { text, usage: completion.usage })
    }
}

/// Тонкая обёртка над SQLite-соединением: хранит реестр агентов (таблица
/// `agents`) и их сообщения (таблица `messages`, `ON DELETE CASCADE` при
/// удалении агента). Соединение защищено мьютексом — операции короткие и
/// нечастые, отдельный пул не нужен.
struct Db(Mutex<Connection>);

impl Db {
    fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("не удалось открыть БД агентов: {}", path.display()))?;
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             CREATE TABLE IF NOT EXISTS agents (
                 name    TEXT PRIMARY KEY,
                 config  TEXT NOT NULL,
                 running INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE IF NOT EXISTS messages (
                 id         INTEGER PRIMARY KEY AUTOINCREMENT,
                 agent_name TEXT NOT NULL REFERENCES agents(name) ON DELETE CASCADE,
                 role       TEXT NOT NULL,
                 content    TEXT NOT NULL,
                 created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
             );
             CREATE INDEX IF NOT EXISTS idx_messages_agent_name ON messages(agent_name);",
        )
        .context("не удалось создать таблицы БД агентов")?;
        Ok(Self(Mutex::new(conn)))
    }

    fn conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.0.lock().expect("соединение с БД агентов отравлено паникой")
    }

    /// Загружает все агенты (конфигурация + флаг запуска) для восстановления реестра при старте.
    fn load_agents(&self) -> Result<Vec<(AgentConfig, bool)>> {
        let conn = self.conn();
        let mut stmt = conn.prepare("SELECT config, running FROM agents")?;
        let rows = stmt
            .query_map([], |row| {
                let config_json: String = row.get(0)?;
                let running: i64 = row.get(1)?;
                Ok((config_json, running != 0))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        rows.into_iter()
            .map(|(config_json, running)| {
                let config: AgentConfig = serde_json::from_str(&config_json)
                    .context("не удалось разобрать конфигурацию агента из БД")?;
                Ok((config, running))
            })
            .collect()
    }

    /// Загружает историю сообщений одного агента в порядке их появления.
    fn load_messages(&self, name: &str) -> Result<Vec<ChatMessage>> {
        let conn = self.conn();
        let mut stmt =
            conn.prepare("SELECT role, content FROM messages WHERE agent_name = ?1 ORDER BY id ASC")?;
        let rows = stmt
            .query_map([name], |row| {
                Ok(ChatMessage { role: row.get(0)?, content: row.get(1)? })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    fn insert_agent(&self, config: &AgentConfig, running: bool) -> Result<()> {
        let config_json = serde_json::to_string(config)?;
        self.conn().execute(
            "INSERT INTO agents (name, config, running) VALUES (?1, ?2, ?3)",
            rusqlite::params![config.name, config_json, running as i64],
        )?;
        Ok(())
    }

    fn set_running(&self, name: &str, running: bool) -> Result<()> {
        self.conn().execute(
            "UPDATE agents SET running = ?1 WHERE name = ?2",
            rusqlite::params![running as i64, name],
        )?;
        Ok(())
    }

    /// Удаляет агента; связанные сообщения удаляются каскадно (`ON DELETE CASCADE`).
    fn delete_agent(&self, name: &str) -> Result<()> {
        self.conn().execute("DELETE FROM agents WHERE name = ?1", [name])?;
        Ok(())
    }

    fn append_message(&self, agent_name: &str, message: &ChatMessage) -> Result<()> {
        self.conn().execute(
            "INSERT INTO messages (agent_name, role, content) VALUES (?1, ?2, ?3)",
            rusqlite::params![agent_name, message.role, message.content],
        )?;
        Ok(())
    }
}

/// Реестр именованных агентов с сохранением в SQLite, общий для CLI, TUI и
/// веб-интерфейса — агент, добавленный в одном из них, виден в других, а его
/// диалог переживает перезапуск процесса.
pub struct AgentManager {
    client: LlmClient,
    db: Arc<Db>,
    agents: RwLock<HashMap<String, Arc<Agent>>>,
}

impl AgentManager {
    pub fn new(client: LlmClient, store_path: impl Into<PathBuf>) -> Result<Self> {
        let db = Arc::new(Db::open(&store_path.into())?);
        let manager = Self { client, db, agents: RwLock::new(HashMap::new()) };
        manager.load()?;
        Ok(manager)
    }

    /// Путь к файлу БД берётся из AGENTS_STORE_PATH, по умолчанию — `agents.db`
    /// в текущей рабочей директории.
    pub fn from_env(client: LlmClient) -> Result<Self> {
        let path = std::env::var("AGENTS_STORE_PATH").unwrap_or_else(|_| "agents.db".to_string());
        Self::new(client, path)
    }

    fn load(&self) -> Result<()> {
        let records = self.db.load_agents()?;
        let mut agents = self.agents.write().expect("реестр агентов отравлен паникой");
        for (config, running) in records {
            let history = self.db.load_messages(&config.name)?;
            let agent = Agent::new(config, self.client.clone(), self.db.clone(), history);
            if running {
                agent.start();
            }
            agents.insert(agent.name().to_string(), Arc::new(agent));
        }
        Ok(())
    }

    pub fn list(&self) -> Vec<AgentInfo> {
        let agents = self.agents.read().expect("реестр агентов отравлен паникой");
        let mut list: Vec<AgentInfo> = agents.values().map(|a| a.info()).collect();
        list.sort_by(|a, b| a.config.name.cmp(&b.config.name));
        list
    }

    pub fn get(&self, name: &str) -> Option<Arc<Agent>> {
        self.agents.read().expect("реестр агентов отравлен паникой").get(name).cloned()
    }

    pub fn create(&self, mut config: AgentConfig) -> Result<AgentInfo> {
        let name = config.name.trim().to_string();
        if name.is_empty() {
            bail!("имя агента не может быть пустым");
        }
        config.name = name.clone();

        let mut agents = self.agents.write().expect("реестр агентов отравлен паникой");
        if agents.contains_key(&name) {
            bail!("агент с именем «{name}» уже существует");
        }
        self.db.insert_agent(&config, false)?;
        let agent = Agent::new(config, self.client.clone(), self.db.clone(), Vec::new());
        let info = agent.info();
        agents.insert(name, Arc::new(agent));

        Ok(info)
    }

    /// Удаляет агента вместе со всей его историей диалога в БД.
    pub fn remove(&self, name: &str) -> Result<()> {
        let mut agents = self.agents.write().expect("реестр агентов отравлен паникой");
        if agents.remove(name).is_none() {
            bail!("агент «{name}» не найден");
        }
        drop(agents);
        self.db.delete_agent(name)
    }

    pub fn start(&self, name: &str) -> Result<AgentInfo> {
        let agent = self.get(name).ok_or_else(|| anyhow!("агент «{name}» не найден"))?;
        agent.start();
        self.db.set_running(name, true)?;
        Ok(agent.info())
    }

    pub fn stop(&self, name: &str) -> Result<AgentInfo> {
        let agent = self.get(name).ok_or_else(|| anyhow!("агент «{name}» не найден"))?;
        agent.stop();
        self.db.set_running(name, false)?;
        Ok(agent.info())
    }
}
