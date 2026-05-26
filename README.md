# Runbook Hybrid RAG API

Готовый API для поиска по runbook'ам и OpenAI-совместимого `/v1/chat/completions`.

## Что внутри

- **Гибридный retrieval**: BM25 + dense embeddings
- **RRF fusion**: объединяет lexical и semantic результаты
- **Reranker**: cross-encoder для повышения точности top-k
- **Нормализация технических токенов**: `billing-worker`, `billing worker`, `billing_worker`
- **Цитаты**: в ответ возвращаются использованные куски
- **OpenWebUI-friendly** API

## Структура

```text
runbook_rag/
  app.py
  requirements.txt
  README.md
  runbooks/         # сюда кладёшь .md/.txt/.yaml/.json
  index_data/       # здесь кэшируется индекс