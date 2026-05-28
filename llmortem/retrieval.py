from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from llmortem.config_store import get_runtime_config
from llmortem.settings import MAX_CONTEXT_CHARS
from llmortem.text_processing import normalize_text, tokenize


def guess_intent(query: str) -> str:
    config = get_runtime_config()

    norm = normalize_text(query)
    tokens = set(tokenize(query))

    if any(pattern in norm for pattern in config.incident_patterns):
        return "incident"

    if re.search(r"\b(4\d\d|5\d\d)\b", norm) and (tokens & config.technical_keywords):
        return "incident"

    if len(tokens & config.technical_keywords) >= 2:
        return "incident"

    if any(pattern in norm for pattern in config.doc_qa_patterns):
        return "docs_qa"

    return "general"


def retrieval_plan_for_query(query: str) -> Tuple[str, List[List[str]]]:
    intent = guess_intent(query)

    if intent == "incident":
        return intent, [["sre", "runbooks"], ["docs"], ["code"]]

    if intent == "docs_qa":
        return intent, [["docs"], ["code"], ["sre", "runbooks"]]

    return intent, [["docs", "sre", "runbooks", "code"]]


def staged_search(index: Any, query: str, top_k: int) -> Tuple[str, List[Dict[str, Any]], str]:
    intent, stages = retrieval_plan_for_query(query)

    if intent == "docs_qa":
        docs_hits = index.search(query, top_k=top_k, collections=["docs"], intent=intent)
        code_hits = index.search(query, top_k=top_k, collections=["code"], intent=intent)

        docs_strong_score = 4.0

        if docs_hits and float(docs_hits[0].get("score", 0)) >= docs_strong_score:
            return intent, docs_hits[:top_k], "docs"

        combined = docs_hits[:2] + code_hits[:top_k]
        deduped = _dedupe_hits(combined)

        if deduped:
            return intent, deduped[:top_k], "docs:weak+code"

        return intent, [], "none"

    all_hits: List[Dict[str, Any]] = []
    used_stages: List[str] = []
    min_good_score = 1.5

    for collections in stages:
        hits = index.search(query, top_k=top_k, collections=collections, intent=intent)

        if not hits:
            continue

        stage_name = "+".join(collections)
        best_score = float(hits[0].get("score", 0))

        if best_score >= min_good_score:
            all_hits.extend(hits)
            used_stages.append(stage_name)
            break

        all_hits.extend(hits[:2])
        used_stages.append(stage_name + ":weak")

    return intent, _dedupe_hits(all_hits)[:top_k], "+".join(used_stages) if used_stages else "none"


def _dedupe_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for hit in sorted(hits, key=lambda item: float(item.get("score", 0)), reverse=True):
        chunk_id = hit.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(hit)

    return deduped


def build_context(
    hits: List[Dict[str, Any]],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> Tuple[str, List[Dict[str, Any]]]:
    blocks: List[str] = []
    refs: List[Dict[str, Any]] = []
    total = 0

    for index, hit in enumerate(hits, start=1):
        line_part = ""

        if hit.get("start_line"):
            line_part = f":L{hit['start_line']}"
            if hit.get("end_line") and hit["end_line"] != hit["start_line"]:
                line_part += f"-L{hit['end_line']}"

        block = (
            f"[SOURCE {index}]\n"
            f"collection: {hit['collection']}\n"
            f"file: {hit['source']}{line_part}\n"
            f"url: {hit['source_url']}\n"
            f"title: {hit['title']}\n"
            f"section: {hit['section']}\n"
            f"score: {hit['score']}\n"
            f"text:\n{hit['text']}\n"
        )

        if total + len(block) > max_chars:
            break

        blocks.append(block)
        refs.append(
            {
                "id": index,
                "collection": hit["collection"],
                "source": hit["source"],
                "source_url": hit["source_url"],
                "title": hit["title"],
                "section": hit["section"],
                "start_line": hit.get("start_line"),
                "end_line": hit.get("end_line"),
                "score": hit["score"],
            }
        )
        total += len(block)

    return "\n\n".join(blocks), refs