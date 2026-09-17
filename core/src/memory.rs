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
/// happy path `planning → execution → validation → done`; из `validation`
/// дополнительно разрешён откат в `execution` (доработка после проваленной
/// проверки) — см. [`Stage::allowed_next`]. Прямых переходов через этап
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
    /// [`Stage`] за обоснованием отката `validation → execution`).
    pub fn allowed_next(self) -> &'static [Stage] {
        match self {
            Stage::Planning => &[Stage::Execution],
            Stage::Execution => &[Stage::Validation],
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
    /// (нужно подтвердить итог). Откат `validation → execution` (доработка) и
    /// переход `execution → validation` (запрос проверки) применяются сразу —
    /// это безопасное, обратимое направление, ворота там не нужны.
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
                 готовности), вызвав update_step. Когда план готов — вызови move_stage в execution с \
                 итогом (кратким описанием плана); этот переход требует подтверждения человека, так \
                 что после вызова остановись, покажи план и жди — не начинай выполнять его сам."
            }
            Stage::Execution => {
                "Сейчас этап EXECUTION. Выполняй задачу согласно согласованному плану (см. «Текущий \
                 шаг»/«Ожидаемое действие» ниже), отмечая прогресс через update_step. Когда работа \
                 сделана, вызови move_stage в validation с итогом (что именно сделано) — этот переход \
                 применяется сразу, без подтверждения."
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
         явно командой «agent remember» и не зависит ни от текущего диалога, ни от текущей задачи):\n\n{}",
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
         к этой задаче (данные явно сохранены командой «agent task set», видны всем участникам и будут \
         удалены для всех сразу при завершении задачи — «agent task finish»):\n\n{}\n\n{body}{invariants_suffix}",
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
         в отличие от них, действуют ТОЛЬКО пока идёт работа над этой задачей (заведены командой \
         «agent task <имя> invariant set», снимаются «agent task <имя> invariant remove»). При конфликте \
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
    if task.paused {
        lines.push(
            "⏸ ЗАДАЧА НА ПАУЗЕ — не предпринимай новых действий по ней, пока её явно не возобновят \
             («agent task <имя> resume»); при ответе учитывай «Ожидаемое действие» выше как то, чего \
             сейчас ждут. Инструменты автомата (move_stage/update_step) сейчас не предлагаются."
                .to_string(),
        );
    } else if let Some(pending) = task.pending_stage {
        let outcome = task.pending_outcome.as_deref().unwrap_or("(не указан)");
        lines.push(format!(
            "⏳ ПРЕДЛОЖЕН ПЕРЕХОД на этап «{pending}» (итог: {outcome}) — ЖДЁТ УТВЕРЖДЕНИЯ ЧЕЛОВЕКОМ \
             («agent task <имя> approve» применит, «agent task <имя> reject <причина>» отклонит). Не \
             веди себя так, будто переход уже произошёл, и не вызывай move_stage повторно — просто \
             сообщи, что ждёшь решения человека."
        ));
    } else {
        lines.push(task.stage.directive().to_string());
    }
    lines.join("\n")
}
