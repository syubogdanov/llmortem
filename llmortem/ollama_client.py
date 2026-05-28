from __future__ import annotations

from typing import Dict, List

import httpx
from fastapi import HTTPException

from llmortem.settings import OLLAMA_BASE_URL, OLLAMA_TIMEOUT


async def ollama_chat(
    *,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to connect to Ollama: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Ollama error: {response.text}")

    try:
        data = response.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from Ollama: {response.text}") from exc

    if "message" not in data or "content" not in data["message"]:
        raise HTTPException(status_code=502, detail=f"Unexpected Ollama response: {data}")

    return str(data["message"].get("content") or "")