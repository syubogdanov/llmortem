from __future__ import annotations

import time
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, HTTPException

from llmortem.config_store import init_db, seed_defaults
from llmortem.default_config import (
    DEFAULT_DOC_QA_PATTERNS,
    DEFAULT_INCIDENT_PATTERNS,
    DEFAULT_STOPWORDS,
    DEFAULT_SYNONYMS,
    DEFAULT_TECHNICAL_KEYWORDS,
)
from llmortem.indexer import RepoKnowledgeIndex
from llmortem.ollama_client import ollama_chat
from llmortem.prompts import (
    SYSTEM_DOC_DRAFT_PROMPT,
    SYSTEM_POSTMORTEM_PROMPT,
    SYSTEM_RAG_PROMPT,
)
from llmortem.retrieval import build_context, guess_intent, staged_search
from llmortem.schemas import (
    AskRequest,
    ChatCompletionRequest,
    DraftDocRequest,
    Message,
    PostmortemRequest,
    SearchRequest,
)
from llmortem.settings import (
    DEFAULT_MODEL,
    MAX_CONTEXT_CHARS,
    OLLAMA_BASE_URL,
    REPO_ROOT,
    DOCS_DIR,
    SRE_DOCS_DIR,
    RUNBOOK_DIR,
    INDEX_DIR,
)
from llmortem.text_processing import sha1_text


app = FastAPI(title="Repository Incident Assistant RAG API", version="2.0.0")
index = RepoKnowledgeIndex()


@app.on_event("startup")
def startup() -> None:
    init_db()
    seed_defaults(
        stopwords=DEFAULT_STOPWORDS,
        synonyms=DEFAULT_SYNONYMS,
        technical_keywords=DEFAULT_TECHNICAL_KEYWORDS,
        doc_qa_patterns=DEFAULT_DOC_QA_PATTERNS,
        incident_patterns=DEFAULT_INCIDENT_PATTERNS,
    )
    index.build_or_load(force=False)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "repo_root": str(REPO_ROOT),
        "docs_dir": str(DOCS_DIR),
        "sre_docs_dir": str(SRE_DOCS_DIR),
        "runbook_dir": str(RUNBOOK_DIR),
        "index_dir": str(INDEX_DIR),
        "chunks": len(index.chunks),
        "collections": index.collection_counts(),
        "default_model": DEFAULT_MODEL,
    }


@app.post("/reindex")
def reindex() -> Dict[str, Any]:
    index.build_or_load(force=True)
    return {"ok": True, "chunks": len(index.chunks), "meta": index.meta}


@app.post("/search")
def search(req: SearchRequest) -> Dict[str, Any]:
    intent = guess_intent(req.query)
    hits = index.search(
        req.query,
        top_k=req.top_k,
        collections=req.collections,
        intent=intent,
    )
    return {"query": req.query, "intent": intent, "hits": hits}


@app.post("/ask/docs")
async def ask_docs(req: AskRequest) -> Dict[str, Any]:
    model = req.model or DEFAULT_MODEL
    intent, hits, used_stage = staged_search(index, req.query, req.top_k_context)
    context, refs = build_context(hits)

    if not hits:
        return {
            "answer": "В документации и индексированных фрагментах кода не найден точный ответ на этот вопрос.",
            "intent": intent,
            "retrieval": {"used": True, "stage": used_stage, "hits": []},
            "sources": [],
        }

    messages = [
        {"role": "system", "content": SYSTEM_RAG_PROMPT},
        {
            "role": "system",
            "content": (
                f"Intent: {intent}\n"
                f"Retrieval stage used: {used_stage}\n\n"
                f"REPOSITORY CONTEXT:\n\n{context}"
            ),
        },
        {"role": "user", "content": req.query},
    ]

    answer = await ollama_chat(
        model=model,
        messages=messages,
        temperature=req.temperature,
    )

    return {
        "answer": answer,
        "intent": intent,
        "retrieval": {"used": True, "stage": used_stage, "hits": hits},
        "sources": refs,
    }


@app.post("/draft-doc")
async def draft_doc(req: DraftDocRequest) -> Dict[str, Any]:
    model = req.model or DEFAULT_MODEL

    hits = index.search(
        req.query,
        top_k=4,
        collections=["code", "docs"],
        intent=guess_intent(req.query),
    )
    context, refs = build_context(hits, max_chars=MAX_CONTEXT_CHARS)

    target_hint = req.target_path or "docs/TODO.md"

    messages = [
        {"role": "system", "content": SYSTEM_DOC_DRAFT_PROMPT},
        {
            "role": "system",
            "content": (
                f"Suggested target path: {target_hint}\n\n"
                f"REPOSITORY CONTEXT FOR STYLE AND FACTS:\n\n{context}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Documentation request: {req.query}\n\n"
                "Write a Markdown draft that could be added to the repository."
            ),
        },
    ]

    draft = await ollama_chat(
        model=model,
        messages=messages,
        temperature=req.temperature,
    )
    draft = clean_markdown_fence(draft)

    return {
        "target_path": target_hint,
        "draft_markdown": draft,
        "sources_used_for_grounding": refs,
        "note": "Проверьте TODO и факты перед коммитом. Endpoint не записывает файл автоматически.",
    }


@app.post("/postmortem")
async def postmortem(req: PostmortemRequest) -> Dict[str, Any]:
    model = req.model or DEFAULT_MODEL

    search_query = req.title or req.incident_description or req.logs[:300] or "incident postmortem"
    hits = index.search(
        search_query,
        top_k=8,
        collections=["sre", "runbooks", "docs"],
        intent="incident",
    )

    hits = [hit for hit in hits if float(hit.get("score", 0)) > 1.0][:4]
    context, refs = build_context(hits, max_chars=8000)

    evidence = (
        f"Title: {req.title or 'Unknown'}\n"
        f"Incident ID: {req.incident_id or 'Unknown'}\n"
        f"Date: {req.incident_date or 'Unknown'}\n"
        f"Severity: {req.severity or 'Unknown'}\n"
        f"Owner: {req.owner or 'Unknown'}\n"
        f"Affected service: {req.affected_service or 'Unknown'}\n\n"
        f"Incident description:\n{req.incident_description or 'Unknown'}\n\n"
        f"Chat transcript:\n{req.chat_transcript or 'Not provided'}\n\n"
        f"Logs:\n{req.logs or 'Not provided'}\n\n"
        f"Metrics:\n{req.metrics or 'Not provided'}\n"
    )

    messages = [
        {"role": "system", "content": SYSTEM_POSTMORTEM_PROMPT},
        {
            "role": "system",
            "content": (
                "Optional repository context for terminology/runbook references. "
                "Do not invent incident facts from this context.\n\n"
                f"{context}"
            ),
        },
        {"role": "user", "content": evidence},
    ]

    report = await ollama_chat(
        model=model,
        messages=messages,
        temperature=req.temperature,
    )

    return {
        "postmortem_markdown": report,
        "sources_used_for_context": refs,
        "note": "Постмортем построен по предоставленным чатам/логам/метрикам. Неизвестные факты должны быть подтверждены человеком.",
    }


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")

    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Ollama error: {response.text}")

    data = response.json()
    models = []

    for model in data.get("models", []):
        name = model.get("name") or model.get("model")
        if name:
            models.append({"id": name, "object": "model", "owned_by": "ollama"})

    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> Dict[str, Any]:
    try:
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        model = req.model or DEFAULT_MODEL
        user_query = get_last_user_message(req.messages)
        use_retrieval = should_use_retrieval(req.messages)

        if use_retrieval:
            intent, hits, used_stage = staged_search(index, user_query, req.top_k_context)
            context, refs = build_context(hits, MAX_CONTEXT_CHARS)

            if not hits:
                content = (
                    "В индексированной документации, SRE-разделах, runbook'ах и комментариях/докстрингах кода "
                    "не найден точный ответ на этот вопрос."
                )
            else:
                messages = build_rag_messages(req.messages, context, intent, used_stage)
                content = await ollama_chat(
                    model=model,
                    messages=messages,
                    temperature=req.temperature,
                )

            return chat_response(
                model=model,
                content=content,
                context=refs,
                retrieval={
                    "used": True,
                    "intent": intent,
                    "stage": used_stage,
                    "hits": hits,
                },
            )

        plain_messages = build_plain_messages(req.messages)
        content = await ollama_chat(
            model=model,
            messages=plain_messages,
            temperature=0.3,
        )

        return chat_response(
            model=model,
            content=content,
            context=[],
            retrieval={"used": False, "hits": []},
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Internal retrieval/generation error: {exc}") from exc


def get_last_user_message(messages: List[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content.strip()
    return ""


def should_use_retrieval(messages: List[Message]) -> bool:
    query = get_last_user_message(messages)
    return guess_intent(query) in {"docs_qa", "incident"}


def build_rag_messages(
    user_messages: List[Message],
    context: str,
    intent: str,
    used_stage: str,
) -> List[Dict[str, str]]:
    last_user = get_last_user_message(user_messages)
    history = [message for message in user_messages if message.role in {"user", "assistant"}][-6:]

    prompt_messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_RAG_PROMPT},
        {
            "role": "system",
            "content": (
                f"Intent: {intent}\n"
                f"Retrieval stage used: {used_stage}\n\n"
                f"REPOSITORY CONTEXT:\n\n{context}\n\n"
                "Answer the latest user question using primarily this context."
            ),
        },
    ]

    for message in history[:-1]:
        prompt_messages.append({"role": message.role, "content": message.content})

    prompt_messages.append({"role": "user", "content": last_user})
    return prompt_messages


def build_plain_messages(user_messages: List[Message]) -> List[Dict[str, str]]:
    last_user = get_last_user_message(user_messages)

    return [
        {
            "role": "system",
            "content": (
                "Ты дружелюбный и полезный помощник. Отвечай естественно, коротко и на языке пользователя. "
                "Если вопрос про документацию, код, инциденты или эксплуатацию сервиса, попроси пользователя "
                "сформулировать конкретный вопрос, чтобы можно было выполнить поиск по репозиторию."
            ),
        },
        {"role": "user", "content": last_user},
    ]


def chat_response(
    *,
    model: str,
    content: str,
    context: List[Dict[str, Any]],
    retrieval: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{sha1_text(str(time.time()))[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "context": context,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
        "retrieval": retrieval,
    }


def clean_markdown_fence(text: str) -> str:
    text = text.strip()

    if text.startswith("```markdown"):
        text = text.removeprefix("```markdown").strip()
    if text.startswith("```"):
        text = text.removeprefix("```").strip()
    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return text


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)