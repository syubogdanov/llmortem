`llmortem` — это FastAPI-сервис для поиска по документации, runbook'ам и коду репозитория.

Сервис работает как RAG-прослойка между OpenWebUI и локальной Ollama-моделью.

## Возможности

- Индексация `docs/`, `docs/sre/`, `runbooks/` и комментариев/docstring'ов из кода.
- Поиск по документации и runbook'ам через BM25.
- Fallback к коду, если в документации нет точного ответа.
- Ответы на SRE/incident-вопросы вида: “Сломалось X, что делать?”.
- Генерация черновика недостающей документации.
- Генерация черновика postmortem по чатам, логам и метрикам.
- OpenAI-compatible endpoint `/v1/chat/completions` для OpenWebUI.
- Postgres + SQLAlchemy для хранения retrieval-конфигурации.
- In-memory cache конфигурации из Postgres.

## Архитектура

```text
OpenWebUI
   |
   | OpenAI-compatible API
   v
FastAPI app
   |
   | retrieval over docs/runbooks/code
   v
BM25 index
   |
   | prompt with grounded context
   v
Ollama

## Postgres используется для хранения конфигурационных параметров retrieval-логики:

stopwords;
synonyms;
technical keywords;
documentation QA patterns;
incident patterns.

При старте приложение создаёт таблицу config_items и заполняет её дефолтными значениями, если таблица пустая.

## Структура проекта

llmortem/
  app.py
  docker-compose.yml
  requirements.txt
  README.md

  docs/
    registration.md
    sre/
      queue-lag.md

  runbooks/
    api-5xx-errors.md
    auth-401-403.md
    database-connection-errors.md
    disk-space-full.md
    external-api-timeout.md
    high-cpu-service.md
    queue-lag.md
    release-rollback.md

  llmortem/
    __init__.py
    settings.py
    default_config.py
    config_store.py
    schemas.py
    text_processing.py
    document_loader.py
    code_scanner.py
    indexer.py
    retrieval.py
    prompts.py
    ollama_client.py


## Запуск
# 1. Установить зависимости
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Запустить Postgres
docker compose up -d postgres

По умолчанию используется база:

database: llmortem
user: llmortem
password: llmortem
host: localhost
port: 5432



## Основные endpoint'ы

# Healthcheck
  curl http://localhost:8000/health | python3 -m json.tool

# Пересобрать индекс
  curl -X POST http://localhost:8000/reindex | python3 -m json.tool

# Поиск по индексу
  curl -X POST http://localhost:8000/search \
    -H "Content-Type: application/json" \
    -d '{"query":"queue lag", "top_k": 5}' \
    | python3 -m json.tool

# Вопрос по документации
  curl -X POST http://localhost:8000/ask/docs \
    -H "Content-Type: application/json" \
    -d '{"query":"Как зарегистрироваться в системе?"}' \
    | python3 -m json.tool

# Incident-вопрос
  curl -X POST http://localhost:8000/ask/docs \
    -H "Content-Type: application/json" \
    -d '{"query":"Сломалась очередь, queue lag растет. Что делать?"}' \
    | python3 -m json.tool
# Черновик документации
  curl -X POST http://localhost:8000/draft-doc \
    -H "Content-Type: application/json" \
    -d '{
      "query": "Напиши документацию о том, как сбросить пароль",
      "target_path": "docs/password-reset.md"
    }' \
    | python3 -m json.tool

# Postmortem
  curl -X POST http://localhost:8000/postmortem \
    -H "Content-Type: application/json" \
    -d '{
      "title": "Queue lag after release",
      "incident_description": "После релиза начал расти queue lag.",
      "chat_transcript": "10:02 deploy started\n10:12 queue lag growing\n10:23 rollback started\n10:50 lag returned to normal",
      "logs": "10:16 ERROR worker failed to parse message: unknown field user_region",
      "metrics": "queue_lag peaked at 18000 messages at 10:30"
    }' \
    | python3 -m json.tool

## Подключение к OpenWebUI

В OpenWebUI нужно добавить endpoint, совместимый с OpenAI:

Base URL: http://host.docker.internal:8000/v1
API key: any value

Если OpenWebUI запущен в Docker на Linux, может потребоваться использовать IP хоста или настроить host.docker.internal.