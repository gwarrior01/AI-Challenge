//! Хранилище промежуточных результатов конвейера.
//!
//! Шаги передают друг другу не текст, а идентификатор: `pdf_to_markdown`
//! возвращает `document_id`, `summarize_markdown` принимает его и возвращает
//! `summary_id`, `save_summary` принимает `summary_id`. Если бы модель сама
//! переносила текст из ответа одного инструмента в аргументы другого, она
//! могла бы его обрезать или пересказать — и проверить это было бы нечем.
//!
//! Идентификатор выводится из SHA-256 содержимого (`doc-<16 hex>`,
//! `sum-<16 hex>`), содержимое лежит в `<root>/.artifacts/<id>.md`, метаданные —
//! в `<id>.json` рядом. При чтении хэш пересчитывается и сверяется с
//! метаданными и с самим идентификатором, так что следующий шаг получает ровно
//! то, что записал предыдущий, или ошибку. Хранилище на диске — поэтому
//! цепочку можно продолжить и после перезапуска сервера.
//!
//! Раскладка `<root>` (по умолчанию `documents/`, переменная
//! `DOCUMENTS_DIR`):
//! - `inbox/` — сюда удобно класть PDF: относительный путь ищется сначала здесь;
//! - `out/` — результаты для человека: `<имя>.md` и `<имя>.summary.md`;
//! - `.artifacts/` — промежуточные результаты по идентификаторам.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Сведения о Markdown-документе, полученном из PDF (шаг 1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMeta {
    pub id: String,
    /// Абсолютный путь к исходному PDF; пусто — документ из save_markdown.
    pub source_pdf: String,
    pub source_sha256: String,
    /// Копия Markdown для человека в `out/`.
    pub markdown_path: String,
    /// Имя документа без расширения — из него строятся имена файлов в `out/`.
    pub stem: String,
    pub title: Option<String>,
    pub pages: usize,
    pub chars: usize,
    pub sha256: String,
}

/// Сведения о кратком содержании (шаг 2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryMeta {
    pub id: String,
    pub document_id: String,
    /// Хэш Markdown, по которому строилось краткое содержание.
    pub document_sha256: String,
    /// Как построено: `llm (<модель>)` или `extractive`.
    pub method: String,
    pub chars: usize,
    pub sha256: String,
}

#[derive(Clone)]
pub struct Store {
    root: PathBuf,
}

impl Store {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        for dir in ["inbox", "out", ".artifacts"].map(|d| root.join(d)) {
            std::fs::create_dir_all(&dir).with_context(|| format!("не удалось создать {}", dir.display()))?;
        }
        // Абсолютный путь — чтобы пути в ответах инструментов не зависели от
        // того, из какого каталога запущен клиент.
        let root = root.canonicalize().unwrap_or(root);
        Ok(Self { root })
    }

    pub fn inbox(&self) -> PathBuf {
        self.root.join("inbox")
    }

    pub fn out(&self) -> PathBuf {
        self.root.join("out")
    }

    fn artifacts(&self) -> PathBuf {
        self.root.join(".artifacts")
    }

    /// Путь к PDF из аргумента инструмента: абсолютный, `~/…` или относительный —
    /// сначала относительно `inbox/`, затем текущего каталога.
    pub fn resolve_pdf(&self, path: &str) -> Result<PathBuf> {
        let path = path.trim();
        if path.is_empty() {
            bail!("путь к PDF пуст");
        }
        let candidate = if let Some(rest) = path.strip_prefix("~/") {
            PathBuf::from(std::env::var("HOME").context("не задана переменная HOME")?).join(rest)
        } else {
            PathBuf::from(path)
        };
        let resolved = if candidate.is_absolute() {
            candidate
        } else if self.inbox().join(&candidate).exists() {
            self.inbox().join(&candidate)
        } else {
            candidate
        };
        if !resolved.is_file() {
            bail!(
                "файл {} не найден (относительные пути ищутся в {} и в текущем каталоге; list_pdfs покажет, что лежит во входящих)",
                resolved.display(),
                self.inbox().display()
            );
        }
        let is_pdf = resolved.extension().is_some_and(|e| e.eq_ignore_ascii_case("pdf"));
        if !is_pdf {
            bail!("{} — не PDF (ожидается расширение .pdf)", resolved.display());
        }
        Ok(resolved.canonicalize().unwrap_or(resolved))
    }

    /// Сохраняет Markdown документа; возвращает его идентификатор.
    pub fn put_document(&self, markdown: &str, meta: impl FnOnce(String, String) -> DocumentMeta) -> Result<DocumentMeta> {
        let sha = sha256_hex(markdown.as_bytes());
        let id = format!("doc-{}", &sha[..16]);
        let meta = meta(id, sha);
        self.put(&meta.id, markdown, &meta)?;
        Ok(meta)
    }

    pub fn put_summary(&self, summary: &str, meta: impl FnOnce(String, String) -> SummaryMeta) -> Result<SummaryMeta> {
        let sha = sha256_hex(summary.as_bytes());
        let id = format!("sum-{}", &sha[..16]);
        let meta = meta(id, sha);
        self.put(&meta.id, summary, &meta)?;
        Ok(meta)
    }

    pub fn document(&self, id: &str) -> Result<(DocumentMeta, String)> {
        self.get(id, "doc-", "document_id из pdf_to_markdown или save_markdown")
            .and_then(|(meta, text): (DocumentMeta, String)| verify(&meta.id, &meta.sha256, &text).map(|_| (meta, text)))
    }

    pub fn summary(&self, id: &str) -> Result<(SummaryMeta, String)> {
        self.get(id, "sum-", "summary_id из summarize_markdown")
            .and_then(|(meta, text): (SummaryMeta, String)| verify(&meta.id, &meta.sha256, &text).map(|_| (meta, text)))
    }

    fn put<M: Serialize>(&self, id: &str, content: &str, meta: &M) -> Result<()> {
        let dir = self.artifacts();
        std::fs::write(dir.join(format!("{id}.md")), content).context("не удалось записать артефакт")?;
        let json = serde_json::to_string_pretty(meta)?;
        std::fs::write(dir.join(format!("{id}.json")), json).context("не удалось записать метаданные артефакта")?;
        Ok(())
    }

    fn get<M: DeserializeOwned>(&self, id: &str, prefix: &str, hint: &str) -> Result<(M, String)> {
        let id = id.trim();
        let well_formed = id.strip_prefix(prefix).is_some_and(|h| h.len() == 16 && h.chars().all(|c| c.is_ascii_hexdigit()));
        if !well_formed {
            bail!("некорректный идентификатор «{id}»: ожидается {prefix}<16 hex> — {hint}");
        }
        let dir = self.artifacts();
        let meta_path = dir.join(format!("{id}.json"));
        if !meta_path.exists() {
            bail!("артефакт {id} не найден — {hint}");
        }
        let meta = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)
            .with_context(|| format!("повреждены метаданные {id}"))?;
        let text = std::fs::read_to_string(dir.join(format!("{id}.md"))).with_context(|| format!("нет содержимого {id}"))?;
        Ok((meta, text))
    }
}

/// Проверка целостности при передаче между шагами: содержимое совпадает с
/// хэшем в метаданных, а хэш — с идентификатором.
fn verify(id: &str, expected_sha: &str, text: &str) -> Result<()> {
    let actual = sha256_hex(text.as_bytes());
    if actual != expected_sha || !id.ends_with(&actual[..16]) {
        bail!("содержимое {id} изменилось после записи (sha256 {actual}, ожидался {expected_sha}) — повторите предыдущий шаг");
    }
    Ok(())
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Безопасное имя файла в `out/`: только последний компонент пути, без
/// расширения `.md`/`.pdf`, без управляющих символов и разделителей.
pub fn safe_stem(name: &str) -> Result<String> {
    let base = Path::new(name.trim()).file_name().and_then(|n| n.to_str()).unwrap_or("");
    let base = base.strip_suffix(".md").unwrap_or(base);
    let base = base.strip_suffix(".summary").unwrap_or(base);
    let base = base.strip_suffix(".pdf").unwrap_or(base);
    let cleaned: String = base
        .chars()
        .map(|c| if c.is_control() || matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') { '_' } else { c })
        .collect();
    let cleaned = cleaned.trim().trim_start_matches('.').to_string();
    if cleaned.is_empty() {
        bail!("некорректное имя файла «{name}»");
    }
    Ok(cleaned)
}

/// Пустое хранилище во временном каталоге — для тестов.
#[cfg(test)]
pub fn temp_store(name: &str) -> Store {
    let dir = std::env::temp_dir().join(format!("documents-test-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    Store::open(dir).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tampered_artifact_is_rejected() {
        let store = temp_store("tamper");
        let meta = store
            .put_summary("исходный текст", |id, sha256| SummaryMeta {
                id,
                document_id: "doc-0000000000000000".into(),
                document_sha256: String::new(),
                method: "test".into(),
                chars: 14,
                sha256,
            })
            .unwrap();
        assert_eq!(store.summary(&meta.id).unwrap().1, "исходный текст");

        std::fs::write(store.artifacts().join(format!("{}.md", meta.id)), "подменённый текст").unwrap();
        let err = store.summary(&meta.id).unwrap_err().to_string();
        assert!(err.contains("изменилось"), "{err}");
    }

    #[test]
    fn bad_ids() {
        let store = temp_store("ids");
        assert!(store.summary("doc-0123456789abcdef").unwrap_err().to_string().contains("некорректный"));
        assert!(store.summary("sum-../../etc/passw").is_err());
        assert!(store.summary("sum-0123456789abcdef").unwrap_err().to_string().contains("не найден"));
    }

    #[test]
    fn safe_stems() {
        assert_eq!(safe_stem("../../etc/report.pdf").unwrap(), "report");
        assert_eq!(safe_stem("итог.summary.md").unwrap(), "итог");
        assert_eq!(safe_stem("a:b").unwrap(), "a_b");
        assert!(safe_stem("..").is_err());
        assert!(safe_stem("").is_err());
    }

    #[test]
    fn resolve_pdf_prefers_inbox() {
        let store = temp_store("resolve");
        std::fs::write(store.inbox().join("in.pdf"), b"%PDF").unwrap();
        assert!(store.resolve_pdf("in.pdf").unwrap().ends_with("inbox/in.pdf"));
        std::fs::write(store.inbox().join("notes.txt"), b"x").unwrap();
        assert!(store.resolve_pdf("notes.txt").unwrap_err().to_string().contains("не PDF"));
        assert!(store.resolve_pdf("missing.pdf").unwrap_err().to_string().contains("не найден"));
    }
}
