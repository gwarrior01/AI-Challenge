//! Три шага конвейера — три отдельных MCP-инструмента.
//!
//! Цепочку собирает не сервер, а модель: она планирует три вызова, а на
//! исполнении делает их по очереди, передавая идентификатор из ответа одного
//! шага в аргументы следующего. Каждый шаг возвращает `input_sha256` (хэш
//! того, что он реально прочитал) и `sha256` (хэш того, что он записал):
//! цепочка передала данные без искажений, если `input_sha256` шага N+1 равен
//! `sha256` шага N.

use std::path::Path;

use anyhow::{bail, Context, Result};
use rmcp::schemars::{self, JsonSchema};
use serde::Serialize;

use crate::markdown;
use crate::store::{safe_stem, sha256_hex, DocumentMeta, Store, SummaryMeta};
use crate::summarize::{Options, Summarizer};

/// Предел размера PDF — чтобы случайно не разбирать гигабайтный файл.
const MAX_PDF_BYTES: u64 = 100 * 1024 * 1024;
const PREVIEW_CHARS: usize = 600;

#[derive(Debug, Serialize, JsonSchema)]
pub struct Converted {
    /// Идентификатор Markdown-документа — его принимает summarize_markdown.
    pub document_id: String,
    pub source_pdf: String,
    /// SHA-256 исходного PDF.
    pub input_sha256: String,
    /// Markdown-файл для человека.
    pub markdown_path: String,
    /// SHA-256 полученного Markdown.
    pub sha256: String,
    pub title: Option<String>,
    pub pages: usize,
    pub chars: usize,
    /// Чем извлечён текст: `pdftotext` (poppler) или `pdf-extract`.
    pub extractor: String,
    /// Начало Markdown — чтобы видеть, что текст извлёкся.
    pub preview: String,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct Summarized {
    /// Идентификатор краткого содержания — его принимает save_summary.
    pub summary_id: String,
    pub document_id: String,
    /// SHA-256 Markdown, который был прочитан и сверен с шагом 1.
    pub input_sha256: String,
    /// SHA-256 краткого содержания.
    pub sha256: String,
    /// Как построено: `llm (<модель>)` или `extractive` (если LLM_API_* не заданы).
    pub method: String,
    pub source_chars: usize,
    pub chars: usize,
    pub summary: String,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct Saved {
    pub path: String,
    pub summary_id: String,
    /// SHA-256 краткого содержания, прочитанного и сверенного с шагом 2.
    pub input_sha256: String,
    /// SHA-256 текста краткого содержания, перечитанного из записанного файла.
    pub sha256: String,
    pub bytes: usize,
    /// Файл с таким именем уже был и перезаписан.
    pub overwritten: bool,
}

/// Шаг 1: PDF → Markdown.
pub async fn pdf_to_markdown(store: &Store, path: &str) -> Result<Converted> {
    let pdf_path = store.resolve_pdf(path)?;
    let size = std::fs::metadata(&pdf_path)?.len();
    if size > MAX_PDF_BYTES {
        bail!("PDF слишком большой: {} МБ (предел {} МБ)", size / 1024 / 1024, MAX_PDF_BYTES / 1024 / 1024);
    }
    let bytes = tokio::fs::read(&pdf_path).await.with_context(|| format!("не удалось прочитать {}", pdf_path.display()))?;
    let source_sha256 = sha256_hex(&bytes);

    let (pages, extractor) = extract_pages(&pdf_path, bytes).await?;
    if pages.iter().all(|p| p.trim().is_empty()) {
        bail!("в PDF нет текстового слоя (похоже на скан) — распознавание текста (OCR) не поддерживается");
    }

    let stem = safe_stem(&file_stem(&pdf_path))?;
    let converted = markdown::pages_to_markdown(&pages, &stem);
    let markdown_path = store.out().join(format!("{stem}.md"));
    tokio::fs::write(&markdown_path, &converted.markdown)
        .await
        .with_context(|| format!("не удалось записать {}", markdown_path.display()))?;

    let chars = converted.markdown.chars().count();
    let meta = store.put_document(&converted.markdown, |id, sha256| DocumentMeta {
        id,
        source_pdf: pdf_path.display().to_string(),
        source_sha256: source_sha256.clone(),
        markdown_path: markdown_path.display().to_string(),
        stem,
        title: converted.title.clone(),
        pages: pages.len(),
        chars,
        sha256,
    })?;
    Ok(Converted {
        document_id: meta.id,
        source_pdf: meta.source_pdf,
        input_sha256: meta.source_sha256,
        markdown_path: meta.markdown_path,
        sha256: meta.sha256,
        title: meta.title,
        pages: meta.pages,
        chars,
        extractor: extractor.to_string(),
        preview: preview(&converted.markdown),
    })
}

/// Шаг 2: Markdown → краткое содержание.
pub async fn summarize_markdown(
    store: &Store,
    summarizer: &Summarizer,
    document_id: &str,
    options: &Options,
) -> Result<Summarized> {
    let (document, markdown) = store.document(document_id)?;
    let summary = summarizer.summarize(&markdown, options).await?;
    let method = summarizer.describe();
    let chars = summary.chars().count();
    let meta = store.put_summary(&summary, |id, sha256| SummaryMeta {
        id,
        document_id: document.id.clone(),
        document_sha256: document.sha256.clone(),
        method: method.clone(),
        chars,
        sha256,
    })?;
    Ok(Summarized {
        summary_id: meta.id,
        document_id: document.id,
        input_sha256: document.sha256,
        sha256: meta.sha256,
        method,
        source_chars: document.chars,
        chars,
        summary,
    })
}

/// Шаг 3: краткое содержание → файл `out/<имя>.summary.md`.
///
/// В начале файла — YAML-шапка с происхождением (исходный PDF, Markdown и
/// хэши всех звеньев), после неё — краткое содержание байт в байт. После
/// записи файл перечитывается, и хэш текста после шапки сверяется с шагом 2.
pub async fn save_summary(store: &Store, summary_id: &str, file_name: Option<&str>) -> Result<Saved> {
    let (summary_meta, summary) = store.summary(summary_id)?;
    let (document, _) = store.document(&summary_meta.document_id)?;
    let stem = match file_name.map(str::trim).filter(|n| !n.is_empty()) {
        Some(name) => safe_stem(name)?,
        None => document.stem.clone(),
    };
    let path = store.out().join(format!("{stem}.summary.md"));
    let overwritten = path.exists();

    let header = format!(
        "---\nsource_pdf: {}\nsource_sha256: {}\nmarkdown: {}\nmarkdown_sha256: {}\nsummary_id: {}\nsummary_sha256: {}\nmethod: {}\n---\n\n",
        yaml_str(&document.source_pdf),
        document.source_sha256,
        yaml_str(&document.markdown_path),
        document.sha256,
        summary_meta.id,
        summary_meta.sha256,
        yaml_str(&summary_meta.method),
    );
    let content = format!("{header}{summary}");
    tokio::fs::write(&path, &content).await.with_context(|| format!("не удалось записать {}", path.display()))?;

    let written = tokio::fs::read_to_string(&path).await?;
    let body = written.strip_prefix(&header).context("записанный файл не начинается с шапки — его изменили во время записи")?;
    let sha256 = sha256_hex(body.as_bytes());
    if sha256 != summary_meta.sha256 {
        bail!("записанный текст не совпадает с кратким содержанием (sha256 {sha256}, ожидался {})", summary_meta.sha256);
    }
    Ok(Saved {
        path: path.display().to_string(),
        summary_id: summary_meta.id,
        input_sha256: summary_meta.sha256,
        sha256,
        bytes: written.len(),
        overwritten,
    })
}

/// Текст PDF по страницам. Основной способ — `pdftotext` из poppler, если он
/// есть в PATH: на свёрстанных документах (кернинг, разрядка) он собирает
/// слова правильно, а `pdf-extract` вставляет пробелы внутрь слов («К ратки й
/// обзор»). Без poppler — `pdf-extract`, чистый Rust.
async fn extract_pages(path: &Path, bytes: Vec<u8>) -> Result<(Vec<String>, &'static str)> {
    let output = tokio::process::Command::new("pdftotext")
        .args(["-enc", "UTF-8"])
        .arg(path)
        .arg("-")
        .kill_on_drop(true)
        .output()
        .await;
    match output {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            // Страницы разделены \f, после последней — тоже \f.
            let mut pages: Vec<String> = text.split('\x0C').map(str::to_string).collect();
            if pages.len() > 1 && pages.last().is_some_and(|p| p.trim().is_empty()) {
                pages.pop();
            }
            return Ok((pages, "pdftotext"));
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            bail!("не удалось разобрать PDF (pdftotext): {}", stderr.trim());
        }
        Err(_) => {} // pdftotext не установлен
    }
    // pdf-extract на повреждённых файлах иногда паникует — в отдельной задаче
    // паника становится обычной ошибкой, а не падением сервера.
    let pages = tokio::task::spawn_blocking(move || pdf_extract::extract_text_from_mem_by_pages(&bytes))
        .await
        .map_err(|e| anyhow::anyhow!("разбор PDF аварийно завершился ({e}) — файл повреждён или в неподдерживаемом формате"))?
        .map_err(|e| anyhow::anyhow!("не удалось разобрать PDF: {e}"))?;
    Ok((pages, "pdf-extract"))
}

fn file_stem(path: &Path) -> String {
    path.file_stem().and_then(|s| s.to_str()).unwrap_or("document").to_string()
}

fn preview(text: &str) -> String {
    match text.char_indices().nth(PREVIEW_CHARS) {
        Some((i, _)) => format!("{}…", &text[..i]),
        None => text.to_string(),
    }
}

/// Строка для YAML-шапки: в кавычках, если в ней есть что-то кроме простых символов.
fn yaml_str(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| format!("{value:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::temp_store;

    const DEMO_PDF: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/demo/mcp-report.pdf");

    fn options() -> Options {
        Options { max_words: 60, focus: None }
    }

    #[tokio::test]
    async fn chain_passes_data_between_steps() {
        let store = temp_store("pipeline");
        let converted = pdf_to_markdown(&store, DEMO_PDF).await.unwrap();
        let summarized = summarize_markdown(&store, &Summarizer::Extractive, &converted.document_id, &options())
            .await
            .unwrap();
        let saved = save_summary(&store, &summarized.summary_id, None).await.unwrap();

        // Выход каждого шага — вход следующего: и идентификатор, и хэш.
        assert_eq!(summarized.document_id, converted.document_id);
        assert_eq!(summarized.input_sha256, converted.sha256);
        assert_eq!(saved.summary_id, summarized.summary_id);
        assert_eq!(saved.input_sha256, summarized.sha256);
        assert_eq!(saved.sha256, summarized.sha256);

        let markdown = std::fs::read_to_string(&converted.markdown_path).unwrap();
        assert!(markdown.starts_with("# Отчёт о переходе на MCP-инструменты\n"), "{markdown}");
        assert!(markdown.contains("## 3. Результаты"));
        assert_eq!(sha256_hex(markdown.as_bytes()), converted.sha256);

        let file = std::fs::read_to_string(&saved.path).unwrap();
        assert!(saved.path.ends_with("out/mcp-report.summary.md"));
        assert!(file.ends_with(&summarized.summary));
        assert!(file.contains(&format!("markdown_sha256: {}", converted.sha256)));
        assert_eq!(summarized.method, "extractive");
    }

    #[tokio::test]
    async fn ids_file_names_and_mixed_up_ids() {
        let store = temp_store("chain");
        let converted = pdf_to_markdown(&store, DEMO_PDF).await.unwrap();
        assert_eq!(converted.pages, 1);
        let summarized = summarize_markdown(&store, &Summarizer::Extractive, &converted.document_id, &options())
            .await
            .unwrap();
        assert_eq!(summarized.input_sha256, converted.sha256);
        let saved = save_summary(&store, &summarized.summary_id, Some("../../итог.md")).await.unwrap();
        assert_eq!(saved.input_sha256, summarized.sha256);
        assert_eq!(saved.sha256, summarized.sha256);
        assert!(saved.path.ends_with("out/итог.summary.md"), "{}", saved.path);
        assert!(!saved.overwritten);
        assert!(save_summary(&store, &summarized.summary_id, Some("итог")).await.unwrap().overwritten);

        // Перепутанный идентификатор — понятная ошибка, а не пустой результат.
        let err = summarize_markdown(&store, &Summarizer::Extractive, &summarized.summary_id, &options())
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("document_id"), "{err}");
    }

    #[tokio::test]
    async fn broken_pdf_is_an_error_not_a_crash() {
        let store = temp_store("broken");
        std::fs::write(store.inbox().join("broken.pdf"), b"%PDF-1.4 not really a pdf").unwrap();
        let err = pdf_to_markdown(&store, "broken.pdf").await.unwrap_err();
        assert!(format!("{err:#}").contains("PDF"), "{err:#}");
        assert!(!store.out().join("broken.md").exists());
    }
}
