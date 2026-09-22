//! Модель памяти агента: три раздельно хранимых уровня, каждый со своим типом
//! данных и своим способом попадания в него — ни один из них не заполняется
//! автоматически по догадке модели, всё **явно** выбирается вызывающей
//! стороной (CLI/веб/TUI-командой), какой именно уровень использовать.
//!
//! - **Краткосрочная память** — текущий диалог. Отдельного типа под неё в этом
//!   модуле нет: это уже существующая история сообщений агента ([`crate::agent::Agent::history`],
//!   ветки — [`crate::agent::Branches`] внутри `agent.rs`) вместе с одной из
//!   стратегий её показа модели ([`crate::context::ContextStrategy`] — вся
//!   история, сводка, скользящее окно и т.п.). Она привязана к диалогу и живёт
//!   в его темпе: пополняется каждым сообщением, "забывается" ровно так, как
//!   решает выбранная стратегия.
//! - **Рабочая память** ([`TaskState`]) — данные ТЕКУЩЕЙ ЗАДАЧИ. Задача — это
//!   **общая, независимая от агента сущность**, адресуемая по имени: один агент
//!   создаёт её (`task start <имя> [--goal ...]`), любые другие агенты
//!   присоединяются к ней по тому же имени (`task join <имя>`) — с этого
//!   момента все участники читают и пишут в один и тот же набор пар
//!   ключ/значение (`task set`), поэтому задачу можно использовать как общий
//!   "блокнот" для нескольких агентов, работающих над одной целью. У каждого
//!   отдельного агента в любой момент не более одной активной задачи (нужно
//!   выйти из текущей, прежде чем начать/присоединиться к другой), но у самой
//!   задачи участников может быть сколько угодно. `task finish` завершает
//!   задачу **для всех** её участников разом и необратимо удаляет её данные —
//!   рабочая память не предназначена пережить саму задачу. Если что-то из неё
//!   должно остаться насовсем — это нужно явно переложить в долговременную
//!   память через [`crate::agent::Agent::remember`] ДО завершения задачи.
//! - **Долговременная память** ([`LongTermItem`]) — принятые решения и знания:
//!   пары ключ/значение с категорией. Это **ЕДИНЫЙ набор данных на всё
//!   приложение** — не приватный для отдельного агента и не привязанный к
//!   задаче: правка через одного агента (`agent remember`/`agent forget`)
//!   сразу видна через любого другого агента, независимо от того, какую
//!   задачу он выполняет. Единственный способ туда что-то положить — явный
//!   вызов `agent remember` (или программный [`crate::agent::Agent::remember`]);
//!   ничего не удаляется, пока не удалят явно (`agent forget`). Раньше сюда же
//!   писали и профиль персонализации (запись с `--category profile`) — это
//!   было соглашением, а не отдельным механизмом, и не переживало смешение с
//!   произвольными фактами. Персонализация (манера общения, язык ответа,
//!   ограничения) теперь живёт отдельно, в markdown-файле — см. [`crate::profile`].
//!
//! Все три уровня хранятся в отдельных таблицах SQLite (см. `Db` в `agent.rs`:
//! `messages`/ветки для краткосрочной, `shared_tasks`/`shared_task_data`/
//! `agent_current_task` для рабочей, `long_term_memory` для долговременной —
//! без столбца `agent_name`, поскольку она не принадлежит ни одному агенту) —
//! ни физически, ни в памяти процесса они не смешиваются, поэтому у каждого
//! факта всегда есть ровно одно однозначное место хранения. Ни рабочая, ни
//! долговременная память не кешируются в памяти процесса ни у одного агента —
//! обе читаются из БД заново при каждом обращении ([`crate::agent::Agent::task_state`],
//! [`crate::agent::Agent::long_term_memory`]), потому что их может менять
//! любой другой агент в любой момент, и кеш неизбежно разошёлся бы с тем, что
//! видят остальные.
//!
//! При каждом запросе к LLM ([`crate::agent::Agent::handle_request`]) в
//! сообщения подмешиваются блоки долговременной и рабочей памяти (см.
//! [`format_long_term_block`]/[`format_task_block`]) — независимо от того,
//! какая стратегия управляет краткосрочной историей, потому что это
//! ортогональные оси: одна решает, что модель помнит из диалога, другая — что
//! она знает о задаче и о мире помимо диалога.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Одна запись долговременной памяти: значение и произвольная категория
/// (например, `profile`, `decision`, `knowledge` — категория не проверяется по
/// списку, это просто метка для человека и для модели).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LongTermItem {
    pub category: String,
    pub value: String,
}

/// Долговременная память целиком — ОДНА на всё приложение (не на агента):
/// ключ уникален глобально, а не в рамках отдельного агента.
pub type LongTermMemory = BTreeMap<String, LongTermItem>;

/// Этап конечного автомата задачи (см. [`TaskState`]) — формализует, на какой
/// стадии сейчас находится общая задача. Порядок соответствует линейному
/// happy path `planning → execution → validation → done`; дополнительно
/// разрешены два отката назад: `validation → execution` (доработка после
/// проваленной проверки) и `execution → planning` (план оказался негодным —
/// после пересмотра выход из планирования снова требует утверждения человеком)
/// — см. [`Stage::allowed_next`]. Прямых переходов через этап
/// (например, `planning → validation`) автомат не допускает — см.
/// [`crate::agent::Agent::task_advance`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    #[default]
    Planning,
    Execution,
    Validation,
    Done,
}

impl Stage {
    /// Этапы, в которые можно легально перейти из данного (см. документацию
    /// [`Stage`] за обоснованием откатов `validation → execution` и
    /// `execution → planning`).
    pub fn allowed_next(self) -> &'static [Stage] {
        match self {
            Stage::Planning => &[Stage::Execution],
            Stage::Execution => &[Stage::Validation, Stage::Planning],
            Stage::Validation => &[Stage::Execution, Stage::Done],
            Stage::Done => &[],
        }
    }

    /// Машинное имя этапа (`planning`/`execution`/`validation`/`done`) — то же,
    /// что принимают CLI/TUI/веб и что хранится в БД.
    pub fn as_str(self) -> &'static str {
        match self {
            Stage::Planning => "planning",
            Stage::Execution => "execution",
            Stage::Validation => "validation",
            Stage::Done => "done",
        }
    }

    /// `true`, если переход `self → target` требует утверждения человеком,
    /// прежде чем реально применится (см. `move_stage` в
    /// `crate::agent::task_machine` — вызов модели такой переход не проводит
    /// сразу, а кладёт его в `TaskState::pending_stage`/`pending_outcome` и
    /// ждёт `Agent::task_approve`). Роли разведены по мотивам двух моментов,
    /// где модель иначе могла бы провести всю работу за один обмен, ничего не
    /// спросив: выход из планирования в исполнение (нужно подтвердить план,
    /// прежде чем агент начнёт действовать) и объявление задачи завершённой
    /// (нужно подтвердить итог). Откаты `validation → execution` (доработка),
    /// `execution → planning` (пересмотр плана) и переход `execution →
    /// validation` (запрос проверки) применяются сразу — это безопасное,
    /// обратимое направление, ворота там не нужны: после пересмотра плана
    /// обратный выход в execution всё равно снова пройдёт через утверждение.
    pub fn requires_approval_to(self, target: Stage) -> bool {
        matches!((self, target), (Stage::Planning, Stage::Execution) | (Stage::Validation, Stage::Done))
    }

    /// Инструкция модели о том, как вести себя на этом этапе — попадает в
    /// каждый запрос (см. [`format_stage_block`]) и работает вместе с
    /// описаниями инструментов `move_stage`/`update_step`
    /// (`crate::agent::task_machine::tool_definitions`) как основной механизм,
    /// не позволяющий модели решить задачу целиком в первом же ответе, минуя
    /// этап планирования.
    pub fn directive(self) -> &'static str {
        match self {
            Stage::Planning => {
                "Сейчас этап PLANNING. Не решай задачу и не выдавай финальный результат — сначала \
                 собери требования, уточни неясности и предложи план работы (шаги, критерии \
                 готовности), вызвав update_step. Инструменты внешних (MCP) серверов тебе \
                 предложены, но на этом этапе они не выполняются: такой вызов вернёт отказ. Не \
                 вызывай их — вместо этого распиши план в терминах именно этих инструментов: для \
                 каждого шага, который получает данные или что-то делает, назови инструмент и \
                 параметры вызова (они включатся, как только человек утвердит план). Когда план \
                 готов — вызови move_stage в execution с итогом (кратким описанием плана); этот \
                 переход требует подтверждения человека, так что после вызова остановись, покажи \
                 план и жди — не начинай выполнять его сам."
            }
            Stage::Execution => {
                "Сейчас этап EXECUTION. Выполняй задачу согласно согласованному плану (см. «Текущий \
                 шаг»/«Ожидаемое действие» ниже), отмечая прогресс через update_step. Когда работа \
                 сделана, вызови move_stage в validation с итогом (что именно сделано) — этот переход \
                 применяется сразу, без подтверждения. Если выяснилось, что план не годится, — вызови \
                 move_stage обратно в planning с итогом (что не так с планом): новый план снова \
                 потребует утверждения человеком."
            }
            Stage::Validation => {
                "Сейчас этап VALIDATION. Не производи новую работу — проверь результат этапа execution \
                 на соответствие цели и критериям. Если что-то не так — вызови move_stage обратно в \
                 execution с итогом (что исправить), это применяется сразу. Если всё в порядке — вызови \
                 move_stage в done с итогом проверки; этот переход требует подтверждения человека, так \
                 что после вызова остановись и жди."
            }
            Stage::Done => "Задача завершена (DONE) — новая работа по ней в рамках этого автомата не ведётся.",
        }
    }

    /// Можно ли на этом этапе ВЫПОЛНЯТЬ вызовы инструментов MCP-серверов.
    /// Только на execution (сама работа по утверждённому плану) и validation
    /// (проверка результата теми же инструментами, например повторный замер):
    /// живой прогон показал, что на planning модель, получив инструменты,
    /// выполняла всю работу до утверждения плана, и утверждение теряло смысл.
    pub fn allows_external_tools(self) -> bool {
        matches!(self, Stage::Execution | Stage::Validation)
    }

    /// Показывать ли инструменты MCP-серверов модели на этом этапе. На planning
    /// они показываются, хотя и не выполняются: модель судит о своих
    /// возможностях по списку инструментов в запросе, а не по тексту — без них
    /// она в живых прогонах отвечала, что доступа к машине у неё нет, и
    /// планировала работу вокруг shell-команд вместо доступных инструментов.
    /// Вызов на planning отклоняется с объяснением (см.
    /// [`crate::agent::Agent::execute_tool`]), а на done инструменты не нужны
    /// вовсе — работа по задаче закончена.
    pub fn offers_external_tools(self) -> bool {
        self.allows_external_tools() || self == Stage::Planning
    }
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Stage {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "planning" => Ok(Stage::Planning),
            "execution" => Ok(Stage::Execution),
            "validation" => Ok(Stage::Validation),
            "done" => Ok(Stage::Done),
            other => bail!(
                "неизвестный этап «{other}» — допустимые значения: planning, execution, validation, done"
            ),
        }
    }
}

/// Рабочая память ОБЩЕЙ задачи: имя, необязательная цель, произвольные пары
/// ключ/значение (заполняемые по ходу работы над задачей любым из
/// присоединившихся к ней агентов) и формализованное состояние конечного
/// автомата — текущий этап, текущий шаг и ожидаемое действие (см. [`Stage`]).
/// `paused` — независимый от этапа флаг: задачу можно поставить на паузу на
/// ЛЮБОМ этапе (см. [`crate::agent::Agent::task_pause`]) — это не отдельное
/// состояние автомата, а отметка "работа временно приостановлена здесь".
/// Поскольку весь этот снимок целиком попадает в каждый запрос к LLM (см.
/// [`format_task_block`]), возобновление задачи (`task_resume`) не требует
/// заново объяснять агенту, на чём остановились — этап/шаг/ожидание уже в
/// контексте. `None` на уровне [`crate::agent::Agent::task_state`] означает
/// "этот агент сейчас не состоит ни в какой задаче" — сама задача при этом
/// может продолжать существовать (и быть видна другим её участникам), пока её
/// не завершат явно.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskState {
    pub name: String,
    pub goal: Option<String>,
    pub data: BTreeMap<String, String>,
    #[serde(default)]
    pub stage: Stage,
    #[serde(default)]
    pub current_step: Option<String>,
    #[serde(default)]
    pub expected_action: Option<String>,
    #[serde(default)]
    pub paused: bool,
    /// Этап, на который модель предложила перейти вызовом `move_stage`, но
    /// переход требует утверждения человеком (см. [`Stage::requires_approval_to`])
    /// и поэтому ЕЩЁ НЕ применён — `stage` выше остаётся прежним, пока человек
    /// не вызовет `Agent::task_approve` (применяет) или `Agent::task_reject`
    /// (снимает предложение, `stage` не меняется). `None` — переход либо не
    /// предлагался, либо уже разрешён (применён или отклонён).
    #[serde(default)]
    pub pending_stage: Option<Stage>,
    /// Итог этапа, которым модель сопроводила предложенный переход
    /// (`pending_stage`) — что было сделано/к чему пришли. `None` синхронно с
    /// `pending_stage`.
    #[serde(default)]
    pub pending_outcome: Option<String>,
    /// Текстовые инварианты, привязанные ТОЛЬКО к этой задаче — обязательны
    /// наравне с глобальными (см. [`crate::invariants`] и [`crate::agent::Agent::handle_request`]),
    /// но действуют и видны ТОЛЬКО пока агент присоединён именно к этой
    /// задаче: например, правило конкретного отчёта или клиента, не нужное
    /// после `task finish`. Заполняются явно —
    /// [`crate::agent::Agent::task_invariant_set`]/[`crate::agent::Agent::task_invariant_remove`].
    #[serde(default)]
    pub invariants: BTreeMap<String, String>,
    /// Переходы автомата, дополнительно ЗАПРЕЩЁННЫЕ для этой конкретной
    /// задачи сверх общей карты [`Stage::allowed_next`] — например, "для
    /// этой задачи откат `validation -> execution` не имеет смысла, запретить
    /// совсем". Это абсолютный запрет: в отличие от `extra_approval_transitions`
    /// ниже, действует одинаково и на ручной переход человеком
    /// ([`crate::agent::Agent::task_advance`]), и на переход, предложенный
    /// моделью (`move_stage`) — снять его может только человек
    /// ([`crate::agent::Agent::task_allow_transition`]).
    #[serde(default)]
    pub blocked_transitions: BTreeSet<(Stage, Stage)>,
    /// Переходы, которые для этой задачи ДОПОЛНИТЕЛЬНО требуют подтверждения
    /// человеком сверх двух переходов, гейтящихся всегда
    /// ([`Stage::requires_approval_to`]) — например, "для этой задачи откат
    /// `validation -> execution` тоже нужно подтверждать, а не только выход
    /// из planning и объявление done". В отличие от `blocked_transitions`
    /// выше, это гейт СОГЛАСИЯ, а не запрет — он действует только на переход,
    /// который предлагает МОДЕЛЬ (`move_stage`): ручной `task_advance` — это
    /// и так решение человека, гейтить его нечем и незачем.
    #[serde(default)]
    pub extra_approval_transitions: BTreeSet<(Stage, Stage)>,
    /// Журнал попыток сменить этап — применённых, предложенных, утверждённых,
    /// отклонённых человеком и отвергнутых автоматом, — последние
    /// [`TRANSITION_LOG_LIMIT`] записей, от старых к новым (см.
    /// [`TransitionRecord`]). Хранится в БД, поэтому переживает паузу и
    /// перезапуск приложения: видно, что и когда пытались «перепрыгнуть».
    #[serde(default)]
    pub transitions: Vec<TransitionRecord>,
    /// Цепочка принятых переходов (применённых или утверждённых), которая
    /// привела задачу на текущий этап, от старых к новым (см. [`stage_path`]):
    /// их итоги — это то, что уже сделано на пройденных этапах (утверждённый
    /// план, что реализовано, что исправить после проверки). Попадает в каждый
    /// запрос (см. [`format_task_block`]), поэтому после паузы или перезапуска
    /// модель продолжает с незавершённого этапа, опираясь на них, даже если
    /// сами реплики уже выпали из окна контекста.
    #[serde(default)]
    pub stage_path: Vec<TransitionRecord>,
    /// Запрос человека, ответ на который был прерван постановкой задачи на
    /// паузу (см. `Agent::handle_request`): ответ модели отброшен, не попал
    /// в историю, а сам запрос отправляется заново при возобновлении (см.
    /// `Agent::task_resume`). Хранится в БД, поэтому переживает перезапуск.
    #[serde(default)]
    pub interrupted_prompt: Option<String>,
}

/// Сколько последних записей журнала переходов попадает в [`TaskState::transitions`].
pub const TRANSITION_LOG_LIMIT: usize = 20;

/// Кто инициировал переход: модель (инструмент `move_stage`) или человек
/// (команда/кнопка, либо просьба в сообщении, перехваченная до обращения к
/// модели — см. [`skip_request_target`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionActor {
    Model,
    Human,
}

/// Чем закончилась попытка перехода.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionOutcome {
    /// Переход применён сразу.
    Applied,
    /// Модель предложила переход, требующий подтверждения, — он ждёт решения.
    Proposed,
    /// Человек утвердил предложенный переход — он применён.
    Approved,
    /// Предложенный переход снят человеком (reject или ручной переход поверх него).
    Rejected,
    /// Автомат отверг попытку: нет такого перехода, запрет инвариантом, пауза…
    Refused,
}

macro_rules! str_enum {
    ($ty:ident { $($variant:ident => $name:literal),+ $(,)? }) => {
        impl $ty {
            pub fn as_str(self) -> &'static str {
                match self { $($ty::$variant => $name),+ }
            }
        }

        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl std::str::FromStr for $ty {
            type Err = anyhow::Error;

            fn from_str(s: &str) -> Result<Self> {
                match s {
                    $($name => Ok($ty::$variant),)+
                    other => bail!("неизвестное значение «{other}» для {}", stringify!($ty)),
                }
            }
        }
    };
}

str_enum!(TransitionActor { Model => "model", Human => "human" });
str_enum!(TransitionOutcome {
    Applied => "applied",
    Proposed => "proposed",
    Approved => "approved",
    Rejected => "rejected",
    Refused => "refused",
});

/// Одна запись журнала переходов задачи (см. [`TaskState::transitions`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitionRecord {
    pub from: Stage,
    pub to: Stage,
    pub actor: TransitionActor,
    pub outcome: TransitionOutcome,
    /// Итог этапа (для применённых/предложенных) или причина отказа.
    pub reason: String,
    /// Время записи, UTC, ISO 8601.
    pub at: String,
}

impl fmt::Display for TransitionRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mark = match self.outcome {
            TransitionOutcome::Applied | TransitionOutcome::Approved => "✔",
            TransitionOutcome::Proposed => "⏳",
            TransitionOutcome::Rejected | TransitionOutcome::Refused => "✖",
        };
        write!(f, "{mark} {} -> {} [{}, {}]", self.from, self.to, self.actor, self.outcome)?;
        if !self.reason.is_empty() {
            write!(f, ": {}", self.reason)?;
        }
        Ok(())
    }
}

/// Восстанавливает путь задачи до `current` по принятым переходам `accepted`
/// (применённым или утверждённым, от старых к новым): последний переход в
/// `current`, перед ним — последний более ранний переход в его исходный этап,
/// и так до `planning`. Откаты и повторы отбрасываются сами: после пересмотра
/// плана (`execution -> planning -> execution`) в путь попадает только новый
/// план, на самом этапе пересмотра — только причина отката (старый план уже
/// не действует), а после доработки (`validation -> execution`) — и что было
/// сделано, и что велено исправить.
pub fn stage_path(accepted: &[TransitionRecord], current: Stage) -> Vec<TransitionRecord> {
    let mut path = Vec::new();
    let mut target = current;
    let mut end = accepted.len();
    while let Some(pos) = accepted[..end].iter().rposition(|r| r.to == target) {
        let record = &accepted[pos];
        path.push(record.clone());
        // Дошли до начала пути: выход из планирования — или возврат в него,
        // после которого всё более раннее пересмотрено и больше не действует.
        if record.from == Stage::Planning || record.to == Stage::Planning {
            break;
        }
        target = record.from;
        end = pos;
    }
    path.reverse();
    path
}

/// Сообщение агенту от имени человека сразу после снятия паузы (см.
/// `Agent::task_resume`): продолжить работу с незавершённого этапа, опираясь
/// на уже сделанное. `None`, когда агенту сейчас нечего делать самому —
/// задача завершена или ждёт решения человека по предложенному переходу.
pub fn resume_followup(task: &TaskState) -> Option<String> {
    if task.stage == Stage::Done || task.pending_stage.is_some() {
        return None;
    }
    let mut text = format!(
        "[Автомат задачи] Задача возобновлена после паузы. Продолжай с незавершённого этапа «{}» с того места, \
         где остановился",
        task.stage
    );
    if let Some(step) = task.current_step.as_deref().filter(|s| !s.trim().is_empty()) {
        text.push_str(&format!(" (текущий шаг: {step})"));
    }
    text.push('.');
    if let Some(expect) = task.expected_action.as_deref().filter(|s| !s.trim().is_empty()) {
        text.push_str(&format!(" Ожидалось: {expect}."));
    }
    if task.stage_path.is_empty() {
        text.push_str(" Опирайся на то, что уже собрано в диалоге, — не начинай заново.");
    } else {
        text.push_str(" Опирайся на итоги пройденных этапов и уже сделанное — не начинай заново.");
    }
    Some(text)
}

/// Служебное продолжение после того, как человек утвердил переход (см.
/// `Agent::task_approve`/`Agent::continue_task`) — модели, в чат не попадает.
pub fn approval_followup(from: Stage, to: Stage) -> String {
    match to {
        Stage::Done => format!(
            "[Автомат задачи] Человек утвердил переход «{from} -> {to}»: задача завершена. Кратко подведи итог."
        ),
        _ => format!(
            "[Автомат задачи] Человек утвердил переход «{from} -> {to}». Приступай к работе на этапе «{to}»."
        ),
    }
}

/// Служебное продолжение при снятии паузы, если пауза оборвала ответ на
/// запрос человека: сам запрос уже в истории диалога, модель должна ответить
/// на него сейчас (см. `Agent::task_resume`).
pub fn interrupted_request_followup(prompt: &str) -> String {
    format!(
        "[Автомат задачи] Задача возобновлена после паузы. Ответ на последний запрос человека был прерван \
         паузой — ответь на него сейчас: «{prompt}»"
    )
}

/// Последнее отклонение человеком перехода с ТЕКУЩЕГО этапа, если после него
/// модель ещё не предлагала переход заново (отказы автомата не в счёт — они
/// не отвечают на замечание). Пока оно есть, блок статуса задачи напоминает
/// модели причину (см. [`format_task_block`]); ручной переход человеком
/// тоже снимает его — запись «rejected» тогда относится к прежнему этапу.
pub fn active_rejection(task: &TaskState) -> Option<&TransitionRecord> {
    let last = task.transitions.iter().rev().find(|r| r.outcome != TransitionOutcome::Refused)?;
    (last.outcome == TransitionOutcome::Rejected && last.actor == TransitionActor::Human && last.from == task.stage)
        .then_some(last)
}

/// Сообщение агенту от имени человека сразу после отклонения перехода
/// (см. `Agent::task_reject`): запускает переработку текущего этапа с учётом
/// причины, не дожидаясь, пока человек сам напишет что-то вроде «переделай».
pub fn rejection_followup(stage: Stage, rejected: Stage, note: &str) -> String {
    match (stage, rejected) {
        (Stage::Planning, Stage::Execution) => format!(
            "План не утверждён. Замечание: {note}\nПереработай план с учётом замечания и предложи его снова."
        ),
        (Stage::Validation, Stage::Done) => format!(
            "Итог проверки не принят. Замечание: {note}\nПерепроверь результат с учётом замечания; если \
             нужна доработка — верни задачу в execution."
        ),
        _ => format!(
            "Переход «{stage} -> {rejected}» отклонён. Замечание: {note}\nДоработай текущий этап с учётом \
             замечания."
        ),
    }
}

/// Напоминание модели, как на самом деле утверждается переход: иначе она
/// склонна просить «напиши „утверждаю“», а такое сообщение ничего не
/// утверждает — автомат двигают только отдельные действия человека.
pub const APPROVAL_IS_NOT_A_CHAT_MESSAGE: &str =
    "Не проси человека написать «утверждаю»/«ок» в чате: сообщение в диалоге переход НЕ утверждает — у \
     человека для этого есть отдельные действия «утвердить» и «отклонить с причиной».";

/// Распознаёт в сообщении человека просьбу перескочить обязательный этап и
/// возвращает этап, в который его просят «прыгнуть»: из `planning` — сразу к
/// реализации (`execution`) без утверждённого плана, из `execution` — сразу к
/// финалу (`done`) без проверки. Такое сообщение перехватывается ДО обращения
/// к модели (см. `Agent::handle_request`): переход автомат и так не пропустит,
/// а здесь не даём модели и просто написать реализацию/финал текстом в обход
/// этапа. Распознавание — по фиксированному списку фраз (без отрицания «не»
/// прямо перед ними), поэтому намеренно консервативно: лучше пропустить
/// необычную формулировку (тогда сработает мягкая защита — директива этапа),
/// чем отказать в обычном вопросе.
pub fn skip_request_target(stage: Stage, prompt: &str) -> Option<Stage> {
    const SKIP_PLANNING: &[&str] = &[
        "пропусти план",
        "пропустим план",
        "пропускаем план",
        "пропустить план",
        "пропусти этап план",
        "пропусти planning",
        "без плана сразу",
        "сразу к реализ",
        "сразу к выполн",
        "сразу к исполн",
        "сразу к код",
        "сразу реализ",
        "сразу пиши",
        "сразу напиши",
        "сразу делай",
        "сразу в execution",
        "перейди к реализ",
        "переходи к реализ",
        "переходим к реализ",
        "перейди к выполн",
        "переходи к выполн",
        "переходим к выполн",
        "перейди в execution",
        "переходи в execution",
        "переходим в execution",
        "skip planning",
        "skip the plan",
        "skip plan",
        "start coding",
        "straight to implementation",
    ];
    const SKIP_VALIDATION: &[&str] = &[
        "пропусти провер",
        "пропустим провер",
        "пропускаем провер",
        "пропустить провер",
        "пропусти валидац",
        "пропусти этап провер",
        "пропусти validation",
        "без проверки заверш",
        "без проверок заверш",
        "сразу заверш",
        "сразу в done",
        "сразу финал",
        "skip validation",
        "skip testing",
        "skip the tests",
    ];
    let (phrases, target) = match stage {
        Stage::Planning => (SKIP_PLANNING, Stage::Execution),
        Stage::Execution => (SKIP_VALIDATION, Stage::Done),
        Stage::Validation | Stage::Done => return None,
    };
    let text = prompt.to_lowercase().replace('ё', "е");
    let asks = phrases.iter().any(|phrase| {
        text.match_indices(phrase).any(|(at, _)| {
            let before = text[..at].trim_end();
            !(before.ends_with(" не") || before == "не")
        })
    });
    asks.then_some(target)
}

impl TaskState {
    /// `true`, если переход `from -> to` дополнительно запрещён инвариантом
    /// ЭТОЙ задачи (см. `blocked_transitions`) — проверяется и в
    /// [`crate::agent::Agent::task_advance`], и в переходе, предложенном
    /// моделью, независимо от того, разрешён ли он общей картой автомата.
    pub fn transition_blocked(&self, from: Stage, to: Stage) -> bool {
        self.blocked_transitions.contains(&(from, to))
    }

    /// `true`, если переход `from -> to`, предложенный МОДЕЛЬЮ, требует
    /// подтверждения человеком — либо это один из двух переходов, гейтящихся
    /// всегда ([`Stage::requires_approval_to`]), либо дополнительный гейт,
    /// заданный инвариантом этой задачи (`extra_approval_transitions`).
    pub fn transition_requires_approval(&self, from: Stage, to: Stage) -> bool {
        from.requires_approval_to(to) || self.extra_approval_transitions.contains(&(from, to))
    }
}

/// Краткая сводка одной общей задачи для списков (см.
/// [`crate::agent::AgentManager::list_tasks`]) — имя, цель и имена всех
/// агентов, которые сейчас к ней присоединены, без самих данных задачи
/// (за ними — [`crate::agent::Agent::task_state`] у одного из участников).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedTaskSummary {
    pub name: String,
    pub goal: Option<String>,
    pub members: Vec<String>,
}

/// Форматирует долговременную память в системный блок для запроса к LLM.
/// Вызывающая сторона сама решает, добавлять ли этот блок (пустая память —
/// нет блока, см. [`crate::agent::Agent::handle_request`]).
pub fn format_long_term_block(memory: &LongTermMemory) -> String {
    let lines: Vec<String> =
        memory.iter().map(|(key, item)| format!("- [{}] {key} = {}", item.category, item.value)).collect();
    format!(
        "Долговременная память (принятые решения, знания — общая для ВСЕХ агентов, сохранена \
         явно человеком и не зависит ни от текущего диалога, ни от текущей задачи):\n\n{}",
        lines.join("\n")
    )
}

/// Форматирует рабочую память текущей задачи в системный блок для запроса к LLM.
pub fn format_task_block(task: &TaskState) -> String {
    let goal_suffix =
        task.goal.as_deref().map(|g| format!(" (цель: {g})")).unwrap_or_default();
    let body = if task.data.is_empty() {
        "(данных пока нет)".to_string()
    } else {
        task.data.iter().map(|(k, v)| format!("- {k} = {v}")).collect::<Vec<_>>().join("\n")
    };
    let invariants_suffix = format_task_invariants_block(task);
    let invariants_suffix =
        if invariants_suffix.is_empty() { String::new() } else { format!("\n\n{invariants_suffix}") };
    format!(
        "Рабочая память текущей задачи «{}»{goal_suffix} — общая для всех агентов, присоединившихся \
         к этой задаче (данные явно сохранены человеком, видны всем участникам и будут \
         удалены для всех сразу при завершении задачи):\n\n{}\n\n{body}{invariants_suffix}",
        task.name,
        format_stage_block(task),
    )
}

/// Форматирует текстовые инварианты, привязанные ТОЛЬКО к этой задаче (см.
/// [`TaskState::invariants`]) — пустая строка, если их нет. Как и глобальные
/// (см. [`crate::invariants::format_invariants_block`]), формулировка не
/// просит "иметь в виду", а прямо предписывает отказываться от решений,
/// нарушающих хотя бы один пункт, называя нарушенный и причину — с той же
/// силой, что и глобальные, но действует только пока идёт работа над этой
/// конкретной задачей (после `task finish` пункты пропадают вместе с задачей).
pub fn format_task_invariants_block(task: &TaskState) -> String {
    if task.invariants.is_empty() {
        return String::new();
    }
    let lines: Vec<String> = task.invariants.iter().map(|(id, text)| format!("- [{id}] {text}")).collect();
    format!(
        "ИНВАРИАНТЫ ЭТОЙ ЗАДАЧИ («{}») — обязательны наравне с жёсткими инвариантами проекта выше, но, \
         в отличие от них, действуют ТОЛЬКО пока идёт работа над этой задачей (заводит и снимает их \
         только человек). При конфликте \
         с запросом — откажись от нарушающей части, назови нарушенный пункт и причину, как и с глобальными \
         инвариантами:\n\n{}",
        task.name,
        lines.join("\n")
    )
}

/// Форматирует состояние конечного автомата задачи (этап/шаг/ожидаемое
/// действие/пауза) — отдельно от [`format_task_block`], чтобы этот же блок
/// можно было показать человеку (CLI/TUI/веб — `agent task <имя> show`) в
/// точности так же, как его видит модель. Именно этот блок — то, что делает
/// возобновление задачи после паузы возможным без повторных объяснений: он
/// целиком читается из БД заново при каждом запросе (см.
/// [`crate::agent::Agent::task_state`]), а не восстанавливается из истории
/// диалога.
pub fn format_stage_block(task: &TaskState) -> String {
    let mut lines = vec![format!("Состояние задачи (конечный автомат): этап = {}", task.stage)];
    // Структурные инварианты автомата ЭТОЙ задачи (см. TaskState::blocked_transitions/
    // extra_approval_transitions) — показываются всегда, независимо от паузы/ожидающего
    // перехода ниже, потому что это факт об устройстве автомата, а не о текущем моменте.
    if !task.blocked_transitions.is_empty() {
        let items: Vec<String> =
            task.blocked_transitions.iter().map(|(from, to)| format!("{from} → {to}")).collect();
        lines.push(format!(
            "🚫 ЗАПРЕЩЕНО инвариантом этой задачи (абсолютно, не обойти даже вручную): {}",
            items.join(", ")
        ));
    }
    if !task.extra_approval_transitions.is_empty() {
        let items: Vec<String> =
            task.extra_approval_transitions.iter().map(|(from, to)| format!("{from} → {to}")).collect();
        lines.push(format!(
            "🔒 Для этой задачи ДОПОЛНИТЕЛЬНО требуют подтверждения человеком (сверх обычных гейтов \
             планирования/итога), если предлагаешь их сама через move_stage: {}",
            items.join(", ")
        ));
    }
    if let Some(step) = task.current_step.as_deref().filter(|s| !s.trim().is_empty()) {
        lines.push(format!("Текущий шаг: {step}"));
    }
    if let Some(expect) = task.expected_action.as_deref().filter(|s| !s.trim().is_empty()) {
        lines.push(format!("Ожидаемое действие: {expect}"));
    }
    if !task.stage_path.is_empty() {
        lines.push(
            "Итоги пройденных этапов (это уже сделано и согласовано — опирайся на них, не начинай заново):"
                .to_string(),
        );
        for record in &task.stage_path {
            let reason = if record.actor == TransitionActor::Human && record.outcome == TransitionOutcome::Applied {
                "переведено человеком вручную, без итога"
            } else {
                record.reason.as_str()
            };
            lines.push(format!("- {} -> {}: {reason}", record.from, record.to));
        }
    }
    if task.paused {
        lines.push(
            "⏸ ЗАДАЧА НА ПАУЗЕ — не предпринимай новых действий по ней, пока человек явно её не \
             возобновит; при ответе учитывай «Ожидаемое действие» выше как то, чего \
             сейчас ждут. Инструменты автомата (move_stage/update_step) сейчас не предлагаются."
                .to_string(),
        );
    } else if let Some(rejection) = active_rejection(task) {
        lines.push(format!(
            "⚠ ЧЕЛОВЕК ОТКЛОНИЛ предложенный переход «{} -> {}». Причина: {}\n\
             Этап остаётся «{}». Переработай его результат с учётом этой причины — не повторяй отклонённый \
             вариант — и в этом же ответе снова вызови move_stage с переработанным результатом: без нового \
             предложения человеку нечего утверждать.",
            rejection.from, rejection.to, rejection.reason, task.stage
        ));
        lines.push(task.stage.directive().to_string());
    } else if let Some(pending) = task.pending_stage {
        let outcome = task.pending_outcome.as_deref().unwrap_or("(не указан)");
        lines.push(format!(
            "⏳ ПРЕДЛОЖЕН ПЕРЕХОД на этап «{pending}» (итог: {outcome}) — ЖДЁТ УТВЕРЖДЕНИЯ ЧЕЛОВЕКОМ \
             (он может утвердить переход или отклонить его с причиной). Не \
             веди себя так, будто переход уже произошёл, и не вызывай move_stage повторно — просто \
             сообщи, что ждёшь решения человека. {APPROVAL_IS_NOT_A_CHAT_MESSAGE}"
        ));
    } else {
        lines.push(task.stage.directive().to_string());
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_map_has_no_shortcuts() {
        let all = [Stage::Planning, Stage::Execution, Stage::Validation, Stage::Done];
        let legal = [
            (Stage::Planning, Stage::Execution),
            (Stage::Execution, Stage::Validation),
            (Stage::Execution, Stage::Planning),
            (Stage::Validation, Stage::Execution),
            (Stage::Validation, Stage::Done),
        ];
        for from in all {
            for to in all {
                assert_eq!(from.allowed_next().contains(&to), legal.contains(&(from, to)), "{from} -> {to}");
            }
        }
    }

    #[test]
    fn approval_required_for_leaving_planning_and_finishing() {
        assert!(Stage::Planning.requires_approval_to(Stage::Execution));
        assert!(Stage::Validation.requires_approval_to(Stage::Done));
        assert!(!Stage::Execution.requires_approval_to(Stage::Validation));
        assert!(!Stage::Validation.requires_approval_to(Stage::Execution));
        assert!(!Stage::Execution.requires_approval_to(Stage::Planning));
    }

    fn accepted(from: Stage, to: Stage, reason: &str) -> TransitionRecord {
        TransitionRecord {
            from,
            to,
            actor: TransitionActor::Model,
            outcome: TransitionOutcome::Applied,
            reason: reason.to_string(),
            at: String::new(),
        }
    }

    fn path_reasons(path: &[TransitionRecord]) -> Vec<&str> {
        path.iter().map(|r| r.reason.as_str()).collect()
    }

    #[test]
    fn stage_path_follows_the_way_to_current_stage() {
        use Stage::*;
        let log = [
            accepted(Planning, Execution, "план v1"),
            accepted(Execution, Planning, "план не годится"),
            accepted(Planning, Execution, "план v2"),
            accepted(Execution, Validation, "сделано"),
            accepted(Validation, Execution, "исправить граничные случаи"),
        ];
        // Доработка: план v2 (не v1), что сделано и что исправить.
        assert_eq!(path_reasons(&stage_path(&log, Execution)), ["план v2", "сделано", "исправить граничные случаи"]);
        assert_eq!(path_reasons(&stage_path(&log[..4], Validation)), ["план v2", "сделано"]);
        // После отката в планирование пройденных этапов нет, кроме самого отката.
        assert_eq!(path_reasons(&stage_path(&log[..2], Planning)), ["план не годится"]);
        assert!(stage_path(&[], Planning).is_empty());
    }

    #[test]
    fn resume_followup_continues_unfinished_stage() {
        let mut task = TaskState {
            name: "demo".to_string(),
            stage: Stage::Execution,
            current_step: Some("раздел 2 из 3".to_string()),
            expected_action: Some("дописать выводы".to_string()),
            stage_path: vec![accepted(Stage::Planning, Stage::Execution, "план")],
            ..Default::default()
        };
        let text = resume_followup(&task).unwrap();
        assert!(text.contains("незавершённого этапа «execution»"), "{text}");
        assert!(text.contains("раздел 2 из 3") && text.contains("дописать выводы"), "{text}");
        assert!(text.contains("итоги пройденных этапов"), "{text}");

        task.pending_stage = Some(Stage::Validation);
        assert_eq!(resume_followup(&task), None, "ждёт решения человека — агенту нечего делать");
        task.pending_stage = None;
        task.stage = Stage::Done;
        assert_eq!(resume_followup(&task), None);
    }

    #[test]
    fn detects_requests_to_skip_a_stage() {
        for prompt in [
            "Пропусти планирование, сразу пиши код",
            "давай сразу к реализации",
            "Переходим к выполнению!",
            "skip planning please",
        ] {
            assert_eq!(skip_request_target(Stage::Planning, prompt), Some(Stage::Execution), "{prompt}");
        }
        for prompt in ["пропусти проверку и закрывай", "Сразу завершай задачу", "skip validation"] {
            assert_eq!(skip_request_target(Stage::Execution, prompt), Some(Stage::Done), "{prompt}");
        }
    }

    #[test]
    fn ordinary_messages_are_not_treated_as_skip_requests() {
        for prompt in [
            "Не пропускай планирование, начнём с требований",
            "Какие этапы будут в плане?",
            "план выглядит разумно, но добавь сроки",
            "не переходи к реализации, пока не согласуем",
        ] {
            assert_eq!(skip_request_target(Stage::Planning, prompt), None, "{prompt}");
        }
        // Одна и та же фраза значима только на своём этапе.
        assert_eq!(skip_request_target(Stage::Validation, "сразу завершай"), None);
        assert_eq!(skip_request_target(Stage::Execution, "пропусти планирование"), None);
    }

    #[test]
    fn task_block_tells_model_what_it_may_do() {
        let mut task = TaskState { name: "demo".to_string(), ..Default::default() };
        assert!(format_task_block(&task).contains("Не решай задачу"));

        task.pending_stage = Some(Stage::Execution);
        let block = format_task_block(&task);
        assert!(block.contains("ЖДЁТ УТВЕРЖДЕНИЯ ЧЕЛОВЕКОМ"), "{block}");
        assert!(!block.contains("Не решай задачу"));

        task.paused = true;
        let block = format_task_block(&task);
        assert!(block.contains("ЗАДАЧА НА ПАУЗЕ"), "{block}");
        assert!(!block.contains("ЖДЁТ УТВЕРЖДЕНИЯ"));
    }
}
