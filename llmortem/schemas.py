from typing import List, Optional

from pydantic import BaseModel

from llmortem.settings import TOP_K_CONTEXT


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    temperature: float = 0.1
    top_k_context: int = TOP_K_CONTEXT
    stream: bool = False


class SearchRequest(BaseModel):
    query: str
    top_k: int = 8
    collections: Optional[List[str]] = None


class AskRequest(BaseModel):
    query: str
    model: Optional[str] = None
    temperature: float = 0.1
    top_k_context: int = TOP_K_CONTEXT


class DraftDocRequest(BaseModel):
    query: str
    target_path: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.2


class PostmortemRequest(BaseModel):
    title: Optional[str] = None
    incident_id: Optional[str] = None
    incident_date: Optional[str] = None
    severity: Optional[str] = None
    owner: Optional[str] = None
    affected_service: Optional[str] = None
    incident_description: str = ""
    chat_transcript: str = ""
    logs: str = ""
    metrics: str = ""
    model: Optional[str] = None
    temperature: float = 0.1