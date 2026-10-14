//! База знаний поддержки (RAG): FAQ и документация продукта из
//! `support/knowledge` (`SUPPORT_KNOWLEDGE`) в своём индексе
//! `rag/support.db` (`SUPPORT_RAG_DB`), отдельном от индексов агентов и
//! ассистента разработчика.
//!
//! Индекс целиком выводится из файлов: перед поиском он сверяется с ними по
//! SHA-256 ([`Knowledge::sync`]), и переиндексируется только изменившееся —
//! правка статьи базы знаний видна в следующем же ответе без перезапуска.

use anyhow::Result;
use llm_core::rag::{self, Embedder, RagConfig, RagContext, RagSettings, Rerank, Strategy};

pub const DEFAULT_CORPUS: &str = "support/knowledge";
pub const DEFAULT_DB: &str = "rag/support.db";
/// Статьи нарезаются по заголовкам Markdown: один раздел — одна инструкция.
pub const STRATEGY: Strategy = Strategy::Structure;
/// Сколько фрагментов уходит модели за один поиск.
const K: usize = 4;
/// Порог близости: у коротких статей поддержки ниже 0.45 — уже шум.
const MIN_SCORE: f32 = 0.45;
const CANDIDATES: usize = 12;

/// Индекс — один на процесс: два вопроса подряд не индексируют наперегонки.
static INDEX_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub fn config() -> RagConfig {
    let base = std::env::current_dir().unwrap_or_else(|_| ".".into());
    let corpus: Vec<String> = crate::env_or("SUPPORT_KNOWLEDGE", DEFAULT_CORPUS)
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    // «Папка загрузок» у RagConfig всегда часть корпуса — загрузок здесь нет,
    // ею служит первый путь самого корпуса.
    let uploads = base.join(corpus.first().map(String::as_str).unwrap_or(DEFAULT_CORPUS));
    RagConfig {
        db_path: base.join(crate::env_or("SUPPORT_RAG_DB", DEFAULT_DB)),
        corpus,
        eval_path: base.join("rag/eval.json"),
        report_path: base.join("rag/compare.md"),
        uploads,
        base,
    }
}

pub fn settings() -> RagSettings {
    RagSettings {
        strategy: Some(STRATEGY.name().to_string()),
        k: K,
        min_score: MIN_SCORE,
        candidates: CANDIDATES,
        rewrite: false,
        rerank: Rerank::Heuristic,
        ..RagSettings::default()
    }
}

pub struct Knowledge {
    pub cfg: RagConfig,
    pub embedder: Embedder,
}

impl Knowledge {
    pub fn new(embedder: Embedder) -> Self {
        Self {
            cfg: config(),
            embedder,
        }
    }

    /// Сверяет индекс с файлами. Возвращает, что изменилось (`None` — ничего).
    pub async fn sync(&self) -> Result<Option<String>> {
        let _guard = INDEX_LOCK.lock().await;
        let reset = rag::reset_if_model_changed(&self.cfg, &self.embedder)?;
        let report = rag::index(&self.cfg, &self.embedder, &[STRATEGY], &|_| {}).await?;
        let mut changes: Vec<String> = report
            .new_versions
            .iter()
            .map(|(source, version)| format!("{source} v{version}"))
            .collect();
        changes.extend(
            report
                .removed
                .iter()
                .map(|source| format!("{source} удалён")),
        );
        changes.extend(
            report
                .failed
                .iter()
                .map(|(source, err)| format!("{source}: {err}")),
        );
        let mut note = (!changes.is_empty())
            .then(|| format!("база знаний переиндексирована: {}", changes.join(", ")));
        if reset {
            note = Some(format!(
                "индекс пересобран моделью {}{}",
                self.embedder.model(),
                note.map(|n| format!(" — {n}")).unwrap_or_default()
            ));
        }
        Ok(note)
    }

    pub async fn search(&self, query: &str) -> Result<RagContext> {
        rag::retrieve::retrieve_in(&self.cfg, &self.embedder, query, &settings(), None, None).await
    }
}

/// Найденное за один ответ: несколько поисков с общей нумерацией фрагментов
/// (повторно найденный фрагмент сохраняет свой номер).
#[derive(Default)]
pub struct Found {
    pub rag: RagContext,
    pub queries: Vec<String>,
}

impl Found {
    /// Добавить результат поиска; возвращает текст фрагментов для модели.
    pub fn merge(&mut self, query: &str, ctx: RagContext) -> String {
        self.queries.push(query.to_string());
        let r = &mut self.rag;
        r.strategy = ctx.strategy.clone();
        r.candidates += ctx.candidates;
        r.below_threshold += ctx.below_threshold;
        r.below_rerank += ctx.below_rerank;
        r.min_score = ctx.min_score;
        r.rerank = ctx.rerank;
        r.best_score = match (r.best_score, ctx.best_score) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        r.warnings.extend(ctx.warnings);
        if ctx.sources.is_empty() {
            let best = ctx
                .best_score
                .map(|b| format!(": лучшая близость {b:.2} при пороге {:.2}", ctx.min_score))
                .unwrap_or_default();
            return format!("В базе знаний близких фрагментов нет{best}.");
        }
        let mut out = String::new();
        for mut src in ctx.sources {
            let n = match r.sources.iter().find(|s| s.chunk_id == src.chunk_id) {
                Some(seen) => seen.n,
                None => {
                    src.n = r.sources.len() + 1;
                    r.sources.push(src.clone());
                    src.n
                }
            };
            out.push_str(&format!("[{n}] {}\n{}\n\n", src.location(), src.text));
        }
        out.trim_end().to_string()
    }

    /// Ответ с блоком «Источники»: фрагменты, на которые сослалась модель.
    /// Номера берутся из данных поиска, не с её слов; проверка ссылок — в
    /// `rag.check`. Ни одной ссылки [n] — блока нет: ответ вроде «тикет не
    /// найден» на базу знаний не опирается, и список найденного вводил бы в
    /// заблуждение.
    /// Номера фрагментов, на которые сослался ответ (после [`Found::finalize`]).
    pub fn cited(&self) -> Vec<usize> {
        self.rag
            .check
            .as_ref()
            .map(|c| c.cited.clone())
            .unwrap_or_default()
    }

    pub fn finalize(&mut self, answer: &str) -> String {
        let mut text = answer.trim().to_string();
        if self.rag.sources.is_empty() {
            return text;
        }
        let check = rag::answer::check(answer, &self.rag);
        if !check.cited.is_empty() {
            text.push_str(&format!(
                "\n\nИсточники:{}",
                rag::answer::sources_block(&self.rag, &check.cited)
            ));
        }
        self.rag.check = Some(check);
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_core::rag::RagSource;

    fn source(n: usize, chunk: &str, text: &str) -> RagSource {
        RagSource {
            n,
            chunk_id: format!("support/knowledge/authentication.md@v1/structure/{chunk}"),
            source: "support/knowledge/authentication.md".into(),
            name: "support/knowledge/authentication.md".into(),
            version: 1,
            section: Some("Ошибка AUTH-LOCKED".into()),
            start_line: 3,
            end_line: 7,
            score: 0.7,
            rerank_score: None,
            rank_before: n,
            text: text.into(),
        }
    }

    #[test]
    fn config_points_to_support_knowledge() {
        let cfg = config();
        assert!(cfg.db_path.ends_with("rag/support.db"));
        assert_eq!(cfg.corpus, vec![DEFAULT_CORPUS.to_string()]);
        assert!(cfg.uploads.ends_with(DEFAULT_CORPUS));
    }

    #[test]
    fn searches_share_numbering_and_sources_are_cited_ones() {
        let mut found = Found::default();
        let first = RagContext {
            sources: vec![
                source(1, "0001", "блокировка 15 минут"),
                source(2, "0002", "2FA"),
            ],
            ..RagContext::default()
        };
        assert!(found
            .merge("AUTH-LOCKED", first)
            .starts_with("[1] support/knowledge/authentication.md › Ошибка AUTH-LOCKED"));
        let second = RagContext {
            sources: vec![
                source(1, "0003", "восстановление"),
                source(1, "0002", "2FA"),
            ],
            ..RagContext::default()
        };
        let out = found.merge("код не приходит", second);
        assert!(out.starts_with("[3] ") && out.contains("[2] "));
        assert_eq!(found.queries, vec!["AUTH-LOCKED", "код не приходит"]);

        let text = found.finalize("Подождите 15 минут [1].");
        assert!(text.contains("Источники:\n- [1]"));
        assert!(!text.contains("- [2]"));
        // Без ссылок [n] — без «Источников».
        assert_eq!(found.finalize("Тикет не найден."), "Тикет не найден.");

        let empty = RagContext {
            best_score: Some(0.3),
            min_score: 0.45,
            ..RagContext::default()
        };
        assert!(Found::default()
            .merge("q", empty)
            .contains("лучшая близость 0.30 при пороге 0.45"));
        assert_eq!(Found::default().finalize(" Нет данных. "), "Нет данных.");
    }
}
