//! Агент — самостоятельная сущность поверх [`LlmClient`]: у неё есть своя
//! конфигурация (системный промпт, модель, параметры генерации, показывать ли
//! токены) и жизненный цикл (запущен/остановлен). Запрос пользователя
//! обрабатывается агентом, а не единичным вызовом клиента напрямую — пока
//! агент остановлен, он отказывается отвечать.

use crate::{ChatMessage, ChatOptions, LlmClient, Usage};
use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
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
    /// заново из актуальной конфигурации при каждом запросе). Без этого каждое
    /// сообщение обрабатывалось бы как разговор с чистого листа, и агент не
    /// «видел» бы предыдущие реплики пользователя.
    history: Mutex<Vec<ChatMessage>>,
}

impl Agent {
    pub fn new(config: AgentConfig, client: LlmClient) -> Self {
        let name = config.name.clone();
        Self { name, client, config: RwLock::new(config), running: AtomicBool::new(false), history: Mutex::new(Vec::new()) }
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

    /// Запускает агента. Начинает новый диалог с чистой историей — если нужно
    /// продолжить прежний разговор, останавливать агента не следует.
    pub fn start(&self) {
        self.history.lock().expect("история агента отравлена паникой").clear();
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

        {
            let mut history = self.history.lock().expect("история агента отравлена паникой");
            history.push(ChatMessage::user(prompt.to_string()));
            history.push(ChatMessage::assistant(completion.content.clone()));
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

#[derive(Serialize, Deserialize)]
struct StoredAgent {
    config: AgentConfig,
    running: bool,
}

/// Реестр именованных агентов с сохранением на диск (JSON), общий для CLI,
/// TUI и веб-интерфейса — агент, добавленный в одном из них, виден в других.
pub struct AgentManager {
    client: LlmClient,
    store_path: PathBuf,
    agents: RwLock<HashMap<String, Arc<Agent>>>,
}

impl AgentManager {
    pub fn new(client: LlmClient, store_path: impl Into<PathBuf>) -> Self {
        let manager = Self { client, store_path: store_path.into(), agents: RwLock::new(HashMap::new()) };
        manager.load();
        manager
    }

    /// Путь к файлу реестра берётся из AGENTS_STORE_PATH, по умолчанию — `agents.json`
    /// в текущей рабочей директории.
    pub fn from_env(client: LlmClient) -> Self {
        let path = std::env::var("AGENTS_STORE_PATH").unwrap_or_else(|_| "agents.json".to_string());
        Self::new(client, path)
    }

    fn load(&self) {
        let Ok(raw) = std::fs::read_to_string(&self.store_path) else { return };
        let Ok(records) = serde_json::from_str::<Vec<StoredAgent>>(&raw) else { return };
        let mut agents = self.agents.write().expect("реестр агентов отравлен паникой");
        for record in records {
            let agent = Agent::new(record.config, self.client.clone());
            if record.running {
                agent.start();
            }
            agents.insert(agent.name().to_string(), Arc::new(agent));
        }
    }

    fn save(&self) -> Result<()> {
        let agents = self.agents.read().expect("реестр агентов отравлен паникой");
        let records: Vec<StoredAgent> = agents
            .values()
            .map(|a| StoredAgent { config: a.config(), running: a.is_running() })
            .collect();
        drop(agents);
        let json = serde_json::to_string_pretty(&records)?;
        std::fs::write(&self.store_path, json)?;
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
        let agent = Agent::new(config, self.client.clone());
        let info = agent.info();
        agents.insert(name, Arc::new(agent));
        drop(agents);

        self.save()?;
        Ok(info)
    }

    pub fn remove(&self, name: &str) -> Result<()> {
        let mut agents = self.agents.write().expect("реестр агентов отравлен паникой");
        if agents.remove(name).is_none() {
            bail!("агент «{name}» не найден");
        }
        drop(agents);
        self.save()
    }

    pub fn start(&self, name: &str) -> Result<AgentInfo> {
        let agent = self.get(name).ok_or_else(|| anyhow!("агент «{name}» не найден"))?;
        agent.start();
        let info = agent.info();
        self.save()?;
        Ok(info)
    }

    pub fn stop(&self, name: &str) -> Result<AgentInfo> {
        let agent = self.get(name).ok_or_else(|| anyhow!("агент «{name}» не найден"))?;
        agent.stop();
        let info = agent.info();
        self.save()?;
        Ok(info)
    }
}
