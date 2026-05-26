from __future__ import annotations

import ast
import hashlib
import json
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import yaml
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from unidecode import unidecode


# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT = Path(os.getenv("REPO_ROOT", ".")).resolve()

DOCS_DIR = Path(os.getenv("DOCS_DIR", str(REPO_ROOT / "docs")))
SRE_DOCS_DIR = Path(os.getenv("SRE_DOCS_DIR", str(DOCS_DIR / "sre")))
RUNBOOK_DIR = Path(os.getenv("RUNBOOK_DIR", str(REPO_ROOT / "runbooks")))
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(REPO_ROOT / "index_data")))

# If you later push the project to GitHub/GitLab, set:
# export REPO_BROWSER_BASE_URL="https://github.com/org/repo/blob/main"
# Then citations will include browser links like:
# https://github.com/org/repo/blob/main/docs/guide.md
REPO_BROWSER_BASE_URL = os.getenv("REPO_BROWSER_BASE_URL", "").rstrip("/")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://172.28.144.1:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))

MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "220"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "35"))
TOP_K_CONTEXT = int(os.getenv("TOP_K_CONTEXT", "6"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "14000"))

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".kt", ".rs", ".rb",
    ".php", ".cs", ".cpp", ".c", ".h", ".hpp", ".scala", ".sh", ".sql"
}
DOC_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".rst", ".yaml", ".yml", ".json", ".html",
    ".log", ".conf", ".ini", ".service", ".sh"
}
EXCLUDED_DIR_NAMES = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__", "node_modules",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache", "index_data",
}


# ============================================================================
# API models
# ============================================================================

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


@dataclass
class Chunk:
    chunk_id: str
    collection: str       # docs | sre | runbooks | code
    source: str           # repo-relative path
    title: str
    section: str
    text: str
    tokens: List[str]
    norm_text: str
    file_hash: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None


# ============================================================================
# Text processing
# ============================================================================

TOKEN_RE = re.compile(r"[\w\-./:]+", re.UNICODE)
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "at", "is", "are", "be",
    "как", "что", "где", "для", "при", "если", "и", "или", "в", "на", "по", "с", "к", "из",
    "это", "этот", "эта", "эти", "его", "ее", "их", "ли", "не", "но", "же", "бы",
}

SYNONYM_MAP = {
    "зарегистрироваться": ["register", "signup", "sign-up", "registration", "create-account"],
    "регистрация": ["register", "signup", "sign-up", "registration", "create-account"],
    "zaregistrirovatsia": ["register", "signup", "sign-up", "registration", "create-account"],
    "registratsiia": ["register", "signup", "sign-up", "registration", "create-account"],
    "sistema": ["system", "application", "app"],    "войти": ["login", "signin", "sign-in", "auth"],
    "авторизация": ["auth", "authorization", "login", "signin"],
    "перезапустить": ["restart", "reboot", "bounce"],
    "перезапуск": ["restart", "reboot"],
    "рестарт": ["restart", "reboot"],
    "сервис": ["service", "daemon", "worker", "job", "deployment", "pod"],
    "воркер": ["worker", "consumer", "job"],
    "ошибка": ["error", "failure", "incident"],
    "логи": ["logs", "log", "journal"],
    "очередь": ["queue", "consumer", "worker"],
    "база": ["database", "db", "postgres", "mysql"],
    "бд": ["database", "db", "postgres", "mysql"],
    "проверить": ["check", "verify", "status"],
    "документация": ["docs", "documentation", "guide", "manual"],
    "сбросить": ["reset"],
    "сброс": ["reset"],
    "пароль": ["password"],
    "sbrosit": ["reset"],
    "sbros": ["reset"],
    "parol": ["password"],
}

TECHNICAL_KEYWORDS = {
    "ошибка", "ошибки", "error", "errors", "incident", "outage", "timeout", "timed-out",
    "connection", "refused", "connection-refused", "database", "db", "postgres", "mysql",
    "redis", "база", "бд", "queue", "lag", "worker", "consumer", "очередь", "воркер",
    "api", "gateway", "upstream", "backend", "service", "сервис", "deployment", "deploy",
    "release", "rollback", "релиз", "откат", "pod", "kubernetes", "kubectl", "docker",
    "nginx", "cpu", "memory", "disk", "space", "latency", "лог", "логи", "logs", "metrics",
    "метрики", "401", "403", "404", "500", "502", "503", "504",
}

DOC_QA_PATTERNS = [
    # Russian original forms
    "как ",
    "где написано",
    "что написано",
    "документац",
    "зарегистрироваться",
    "регистрация",
    "настроить",
    "подключить",
    "создать",
    "сбросить",
    "сброс",
    "пароль",
    "sbrosit",
    "sbros",
    "parol",
    "password",
    "reset",

    # Russian after unidecode() transliteration
    "kak ",
    "gde napisano",
    "chto napisano",
    "dokumentats",
    "zaregistrirovatsia",
    "registratsiia",
    "nastroit",
    "podkliuchit",
    "sozdat",

    # English
    "how to",
    "how do i",
    "documentation",
    "docs",
    "register",
    "registration",
    "sign up",
    "signup",
]

INCIDENT_PATTERNS = [
    "сломалось", "не работает", "упал", "падает", "инцидент", "авария", "outage",
    "incident", "error rate", "после релиза", "rollback", "откат", "timeout",
    "ошибка 500", "ошибка 502", "ошибка 503", "ошибка 504", "high cpu", "queue lag",
    "disk space", "connection refused", "too many connections",
]


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("ё", "е")
    text = unidecode(text)
    text = text.replace("_", "-")
    text = re.sub(r"[`'\"“”‘’]", "", text)
    text = WS_RE.sub(" ", text)
    return text


def tokenize(text: str) -> List[str]:
    norm = normalize_text(text)
    raw = TOKEN_RE.findall(norm)
    out: List[str] = []

    for tok in raw:
        out.append(tok)

        for sep in ["-", "_", "/", "."]:
            if sep in tok and len(tok) > 3:
                parts = [p for p in tok.split(sep) if p]
                out.extend(parts)

        if "-" in tok:
            out.append(tok.replace("-", ""))

    return [t for t in out if t and t not in STOPWORDS]


def expand_query_tokens(query: str) -> List[str]:
    toks = tokenize(query)
    extra: List[str] = []
    for tok in toks:
        if tok in SYNONYM_MAP:
            extra.extend(SYNONYM_MAP[tok])
    return toks + extra


def strip_markdown(md: str) -> str:
    # Keep fenced code content, remove only the fence markers.
    md = CODE_FENCE_RE.sub(lambda m: "\n" + m.group(0).replace("```", "") + "\n", md)
    md = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", md)
    md = re.sub(r"^[>*-]\s?", "", md, flags=re.MULTILINE)
    md = re.sub(r"[*_~]", "", md)
    return md


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def make_source_url(source: str, start_line: Optional[int] = None) -> str:
    if not REPO_BROWSER_BASE_URL:
        if start_line:
            return f"repo://{source}#L{start_line}"
        return f"repo://{source}"

    url = f"{REPO_BROWSER_BASE_URL}/{source}"
    if start_line:
        url += f"#L{start_line}"
    return url


def load_document_file(path: Path) -> Tuple[str, str]:
    text = safe_read_text(path)
    suffix = path.suffix.lower()

    if suffix in {".md", ".markdown", ".rst"}:
        return path.stem, strip_markdown(text)

    if suffix in {".txt", ".log", ".conf", ".ini", ".service", ".sh"}:
        return path.stem, text

    if suffix in {".yaml", ".yml"}:
        try:
            obj = yaml.safe_load(text)
            pretty = yaml.safe_dump(obj, allow_unicode=True, sort_keys=False) if obj is not None else text
            return path.stem, pretty
        except Exception:
            return path.stem, text

    if suffix == ".json":
        try:
            obj = json.loads(text)
            return path.stem, json.dumps(obj, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return path.stem, text

    if suffix == ".html":
        soup = BeautifulSoup(text, "html.parser")
        return path.stem, soup.get_text("\n")

    return path.stem, HTML_TAG_RE.sub(" ", text)


def split_paragraphs(text: str) -> List[str]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
    return blocks or ([text.strip()] if text.strip() else [])


def line_span_for_text(full_text: str, fragment: str) -> Tuple[Optional[int], Optional[int]]:
    if not fragment.strip():
        return None, None
    pos = full_text.find(fragment[: min(80, len(fragment))])
    if pos < 0:
        return None, None
    start_line = full_text[:pos].count("\n") + 1
    end_line = start_line + fragment.count("\n")
    return start_line, end_line


def chunk_document(
    collection: str,
    source: str,
    title: str,
    text: str,
    file_hash: str,
) -> List[Chunk]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = split_paragraphs(text)

    chunks: List[Chunk] = []
    current: List[str] = []
    current_len = 0
    section = title
    idx = 0

    def emit(joined: str, section_name: str, local_idx: int) -> None:
        start_line, end_line = line_span_for_text(text, joined)
        chunk_id = sha1_text(f"{collection}:{source}:{local_idx}:{joined[:120]}")
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                collection=collection,
                source=source,
                title=title,
                section=section_name,
                text=joined,
                tokens=tokenize(joined),
                norm_text=normalize_text(joined),
                file_hash=file_hash,
                start_line=start_line,
                end_line=end_line,
            )
        )

    for para in paragraphs:
        m = re.match(r"^(#{1,6})\s+(.*)$", para)
        if m:
            section = m.group(2).strip()

        para_tokens = para.split()
        para_len = len(para_tokens)

        if para_len > MAX_CHUNK_TOKENS:
            step = max(1, MAX_CHUNK_TOKENS - CHUNK_OVERLAP)
            for start in range(0, para_len, step):
                sub = " ".join(para_tokens[start:start + MAX_CHUNK_TOKENS])
                emit(sub, section, idx)
                idx += 1
            continue

        if current_len + para_len > MAX_CHUNK_TOKENS and current:
            joined = "\n\n".join(current)
            emit(joined, section, idx)
            idx += 1

            overlap_words = " ".join(joined.split()[-CHUNK_OVERLAP:]) if CHUNK_OVERLAP else ""
            current = [overlap_words, para] if overlap_words else [para]
            current_len = len(overlap_words.split()) + para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        joined = "\n\n".join(current)
        emit(joined, section, idx)

    return chunks


# ============================================================================
# Code scanning
# ============================================================================

PY_COMMENT_RE = re.compile(r"^\s*#\s?(.*)$")
GENERIC_COMMENT_RE = re.compile(r"^\s*(#|//|--)\s?(.*)$")


def extract_python_knowledge(path: Path) -> str:
    """Extract useful comments, docstrings and function/class signatures from Python."""
    raw = safe_read_text(path)
    blocks: List[str] = []

    try:
        tree = ast.parse(raw)
    except SyntaxError:
        return extract_generic_code_knowledge(path)

    module_doc = ast.get_docstring(tree)
    if module_doc:
        blocks.append(f"Module docstring:\n{module_doc}")

    lines = raw.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            signature = ""
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                signature = f"{'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}def {node.name}({', '.join(args)})"
            elif isinstance(node, ast.ClassDef):
                signature = f"class {node.name}"

            if doc:
                blocks.append(f"{signature} at line {getattr(node, 'lineno', '?')}:\n{doc}")
            else:
                blocks.append(f"{signature} at line {getattr(node, 'lineno', '?')}")

    comment_runs: List[str] = []
    current: List[str] = []
    for line in lines:
        m = PY_COMMENT_RE.match(line)
        if m:
            current.append(m.group(1))
        else:
            if current:
                if len(" ".join(current).strip()) > 20:
                    comment_runs.append("\n".join(current))
                current = []
    if current and len(" ".join(current).strip()) > 20:
        comment_runs.append("\n".join(current))

    for i, comment in enumerate(comment_runs[:80], start=1):
        blocks.append(f"Comment block {i}:\n{comment}")

    return "\n\n".join(blocks)


def extract_generic_code_knowledge(path: Path) -> str:
    """Extract comments and nearby declarations from non-Python code."""
    raw = safe_read_text(path)
    lines = raw.splitlines()
    blocks: List[str] = []
    current: List[str] = []

    declaration_patterns = [
        re.compile(r"\bfunction\s+([A-Za-z0-9_]+)"),
        re.compile(r"\bclass\s+([A-Za-z0-9_]+)"),
        re.compile(r"\bfunc\s+([A-Za-z0-9_]+)"),
        re.compile(r"\bdef\s+([A-Za-z0-9_]+)"),
        re.compile(r"\bpublic\s+.*\s+([A-Za-z0-9_]+)\("),
    ]

    for i, line in enumerate(lines, start=1):
        cm = GENERIC_COMMENT_RE.match(line)
        if cm:
            current.append(cm.group(2))
            continue

        if current:
            text = "\n".join(current).strip()
            if len(text) > 20:
                blocks.append(f"Comment block near line {max(1, i - len(current))}:\n{text}")
            current = []

        for pat in declaration_patterns:
            m = pat.search(line)
            if m:
                blocks.append(f"Declaration near line {i}: {line.strip()}")
                break

    if current:
        text = "\n".join(current).strip()
        if len(text) > 20:
            blocks.append(f"Comment block near EOF:\n{text}")

    return "\n\n".join(blocks)


# ============================================================================
# Index
# ============================================================================

class RepoKnowledgeIndex:
    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.meta: Dict[str, Any] = {}

    def _iter_files(self, roots: Iterable[Path], extensions: set[str]) -> Iterable[Path]:
        seen: set[Path] = set()
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if path in seen:
                    continue
                seen.add(path)
                if not path.is_file():
                    continue
                if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
                    continue
                if path.suffix.lower() in extensions:
                    yield path

    def _compute_source_fingerprint(self) -> str:
        parts: List[str] = []
        roots = [DOCS_DIR, RUNBOOK_DIR, REPO_ROOT]
        extensions = DOC_EXTENSIONS | CODE_EXTENSIONS

        for path in self._iter_files(roots, extensions):
            rel = repo_relative(path)
            try:
                data_hash = hashlib.sha1(path.read_bytes()).hexdigest()
            except OSError:
                continue
            parts.append(f"{rel}:{data_hash}")

        return sha1_text("|".join(sorted(parts))) if parts else "empty"

    def build_or_load(self, force: bool = False) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        index_pkl = INDEX_DIR / "knowledge_index.pkl"
        meta_json = INDEX_DIR / "knowledge_meta.json"

        current_fp = self._compute_source_fingerprint()
        if not force and index_pkl.exists() and meta_json.exists():
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == current_fp:
                with index_pkl.open("rb") as f:
                    payload = pickle.load(f)
                self.chunks = payload["chunks"]
                self.bm25 = payload["bm25"]
                self.meta = meta
                return

        self._build_from_files(current_fp)

        with index_pkl.open("wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, f)

        meta_json.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _collection_for_document(self, path: Path) -> str:
        try:
            path.resolve().relative_to(SRE_DOCS_DIR.resolve())
            return "sre"
        except Exception:
            pass

        try:
            path.resolve().relative_to(RUNBOOK_DIR.resolve())
            return "runbooks"
        except Exception:
            pass

        return "docs"

    def _build_from_files(self, fingerprint: str) -> None:
        chunks: List[Chunk] = []

        # 1. Documentation and runbooks.
        doc_roots = [DOCS_DIR, RUNBOOK_DIR]
        for path in self._iter_files(doc_roots, DOC_EXTENSIONS):
            title, text = load_document_file(path)
            if not text.strip():
                continue
            source = repo_relative(path)
            collection = self._collection_for_document(path)
            file_hash = sha1_text(text)
            chunks.extend(chunk_document(collection, source, title, text, file_hash))

        # 2. Code comments / docstrings / declarations.
        for path in self._iter_files([REPO_ROOT], CODE_EXTENSIONS):
            # Avoid double-indexing shell scripts from docs/runbooks as code if already included as docs.
            if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
                continue

            source = repo_relative(path)
            if source.startswith("docs/") or source.startswith("runbooks/"):
                continue

            if path.suffix.lower() == ".py":
                extracted = extract_python_knowledge(path)
            else:
                extracted = extract_generic_code_knowledge(path)

            if not extracted.strip():
                continue

            file_hash = sha1_text(safe_read_text(path))
            chunks.extend(chunk_document("code", source, path.stem, extracted, file_hash))

        if not chunks:
            raise RuntimeError("No docs/runbooks/code knowledge found. Add docs/, runbooks/ or code comments/docstrings.")

        self.chunks = chunks
        self.bm25 = BM25Okapi([c.tokens for c in chunks])
        self.meta = {
            "fingerprint": fingerprint,
            "created_at": time.time(),
            "num_chunks": len(chunks),
            "collections": self.collection_counts(),
            "repo_root": str(REPO_ROOT),
        }

    def collection_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for c in self.chunks:
            out[c.collection] = out.get(c.collection, 0) + 1
        return out

    def _score_chunk_boost(self, query: str, chunk: Chunk) -> float:
        q_norm = normalize_text(query)
        q_tokens = set(tokenize(query))
        score = 0.0

        if q_norm and q_norm in chunk.norm_text:
            score += 3.0

        for tok in q_tokens:
            if tok in chunk.tokens:
                score += 0.2
            if tok in normalize_text(chunk.title):
                score += 0.5
            if tok in normalize_text(chunk.section):
                score += 0.35
            if tok in normalize_text(chunk.source):
                score += 0.25

        # Prefer docs for product/user-facing questions.
        if guess_intent(query) == "docs_qa":
            if chunk.collection == "docs":
                score += 0.7
            elif chunk.collection == "code":
                score -= 0.2

        # Prefer SRE/runbooks for incident questions.
        if guess_intent(query) == "incident":
            if chunk.collection in {"sre", "runbooks"}:
                score += 1.0
            elif chunk.collection == "code":
                score += 0.15

        # Operational command boost.
        if {"restart", "перезапустить", "перезапуск"} & q_tokens:
            if "systemctl restart" in chunk.norm_text or "kubectl rollout restart" in chunk.norm_text:
                score += 1.2

        return score

    def search(
        self,
        query: str,
        top_k: int = 8,
        collections: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.chunks or self.bm25 is None:
            self.build_or_load(force=False)

        allowed = set(collections or [])
        query_tokens = expand_query_tokens(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked: List[Tuple[int, float, float]] = []
        for idx, bm25_score in enumerate(scores):
            c = self.chunks[idx]
            if allowed and c.collection not in allowed:
                continue
            boost = self._score_chunk_boost(query, c)
            final = float(bm25_score) + boost
            if final > 0:
                ranked.append((idx, float(bm25_score), float(final)))

        ranked.sort(key=lambda x: x[2], reverse=True)

        out: List[Dict[str, Any]] = []
        for idx, bm25_score, final_score in ranked[:top_k]:
            c = self.chunks[idx]
            out.append(
                {
                    "chunk_id": c.chunk_id,
                    "collection": c.collection,
                    "source": c.source,
                    "source_url": make_source_url(c.source, c.start_line),
                    "title": c.title,
                    "section": c.section,
                    "text": c.text,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "score": round(final_score, 6),
                    "signals": {
                        "bm25": round(bm25_score, 6),
                        "boosted": round(final_score, 6),
                    },
                }
            )

        return out


# ============================================================================
# Retrieval strategies
# ============================================================================

def guess_intent(query: str) -> str:
    norm = normalize_text(query)
    toks = set(tokenize(query))

    if any(p in norm for p in INCIDENT_PATTERNS):
        return "incident"
    if re.search(r"\b(4\d\d|5\d\d)\b", norm) and (toks & TECHNICAL_KEYWORDS):
        return "incident"
    if len(toks & TECHNICAL_KEYWORDS) >= 2:
        return "incident"

    if any(p in norm for p in DOC_QA_PATTERNS):
        return "docs_qa"

    return "general"


def retrieval_plan_for_query(query: str) -> Tuple[str, List[List[str]]]:
    intent = guess_intent(query)

    if intent == "incident":
        # Scenario 4: docs/sre first, runbooks second, code as fallback.
        return intent, [["sre", "runbooks"], ["docs"], ["code"]]

    if intent == "docs_qa":
        # Scenarios 1 and 2: docs first, then code comments/docstrings/algorithms
        return intent, [["docs"], ["code"], ["sre", "runbooks"]]    
        
    # General fallback: do not force RAG for pure chat, but allow search endpoint to use all.
    return intent, [["docs", "sre", "runbooks", "code"]]


def staged_search(query: str, top_k: int) -> Tuple[str, List[Dict[str, Any]], str]:
    intent, stages = retrieval_plan_for_query(query)

    # Для вопросов по документации:
    # 1. сначала ищем в docs;
    # 2. если docs даёт сильный результат — используем docs;
    # 3. если docs слабый или пустой — ищем в code.
    if intent == "docs_qa":
        docs_hits = index.search(query, top_k=top_k, collections=["docs"])
        code_hits = index.search(query, top_k=top_k, collections=["code"])

        DOCS_STRONG_SCORE = 4.0

        if docs_hits and float(docs_hits[0].get("score", 0)) >= DOCS_STRONG_SCORE:
            return intent, docs_hits[:top_k], "docs"

        combined = docs_hits[:2] + code_hits[:top_k]

        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for hit in sorted(combined, key=lambda h: float(h.get("score", 0)), reverse=True):
            chunk_id = hit.get("chunk_id")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            deduped.append(hit)

        if deduped:
            return intent, deduped[:top_k], "docs:weak+code"

        return intent, [], "none"

    all_hits: List[Dict[str, Any]] = []
    used_stages: List[str] = []

    MIN_GOOD_SCORE = 1.5

    for collections in stages:
        hits = index.search(query, top_k=top_k, collections=collections)

        if not hits:
            continue

        best_score = float(hits[0].get("score", 0))
        stage_name = "+".join(collections)

        if best_score >= MIN_GOOD_SCORE:
            all_hits.extend(hits)
            used_stages.append(stage_name)
            break

        all_hits.extend(hits[:2])
        used_stages.append(stage_name + ":weak")

    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for hit in sorted(all_hits, key=lambda h: float(h.get("score", 0)), reverse=True):
        chunk_id = hit.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(hit)

    return intent, deduped[:top_k], "+".join(used_stages) if used_stages else "none"


def build_context(hits: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> Tuple[str, List[Dict[str, Any]]]:
    blocks: List[str] = []
    refs: List[Dict[str, Any]] = []
    total = 0

    for i, hit in enumerate(hits, start=1):
        line_part = ""
        if hit.get("start_line"):
            line_part = f":L{hit['start_line']}"
            if hit.get("end_line") and hit["end_line"] != hit["start_line"]:
                line_part += f"-L{hit['end_line']}"

        block = (
            f"[SOURCE {i}]\n"
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
                "id": i,
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


SYSTEM_RAG_PROMPT = """You are a repository-aware engineering assistant.

Rules:
1. Use the provided repository context as the primary source of truth.
2. Cite sources inline like [1], [2] after every factual claim based on context.
3. If documentation contains the answer, answer from documentation first.
4. If documentation does not contain the answer but code context does, clearly say that the answer was inferred from code, then cite code sources.
5. If the answer is absent from the provided sources, say that the repository context does not contain a precise answer. Do not invent details.
6. For incident questions, give concise operational steps, checks, and escalation points. Do not invent commands absent from sources.
7. Answer in the user's language.
"""

SYSTEM_DOC_DRAFT_PROMPT = """You write documentation patches for an existing repository.

Rules:
1. Use only facts explicitly present in the provided repository context.
2. Do not add UI labels, buttons, URLs, contacts, security requirements, timings, owners, or support instructions unless they are present in the context.
3. If a necessary detail is missing, write TODO instead of inventing it.
4. Keep the draft short and in the same simple style as the existing docs.
5. Return raw Markdown only. Do not wrap it in ```markdown fences.
"""


SYSTEM_POSTMORTEM_PROMPT = """You are an SRE postmortem assistant.

Create a blameless incident postmortem from the provided chats/logs/metrics.

Strict rules:
1. Do not invent incident ID, date, severity, owners, services, versions, timestamps, causes, or customer counts.
2. If incident ID, date, severity, owner, affected service, or exact root cause are not provided, write "Unknown" or "Needs confirmation".
3. Use only timestamps that appear in the provided incident evidence.
4. Separate "Facts" from "Hypotheses".
5. Root cause must be "Suspected" unless the evidence explicitly confirms it.
6. Follow-up actions must be phrased as recommendations, not completed facts.
7. Repository context may be used only for terminology and suggested remediation style. It must not override the incident evidence.
8. Answer in Russian unless the user content is mostly English.
"""



def get_last_user_message(messages: List[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content.strip()
    return ""


def should_use_retrieval(messages: List[Message]) -> bool:
    query = get_last_user_message(messages)
    intent = guess_intent(query)
    return intent in {"docs_qa", "incident"}


def build_rag_messages(user_messages: List[Message], context: str, intent: str, used_stage: str) -> List[Dict[str, str]]:
    last_user = get_last_user_message(user_messages)
    history = [m for m in user_messages if m.role in {"user", "assistant"}][-6:]

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

    for m in history[:-1]:
        prompt_messages.append({"role": m.role, "content": m.content})

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


async def ollama_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Failed to connect to Ollama: {e}") from e

    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Ollama error: {resp.text}")

    try:
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from Ollama: {resp.text}") from e

    if "message" not in data or "content" not in data["message"]:
        raise HTTPException(status_code=502, detail=f"Unexpected Ollama response: {data}")

    return str(data["message"].get("content") or "")


# ============================================================================
# FastAPI app
# ============================================================================

app = FastAPI(title="Repository Incident Assistant RAG API", version="2.0.0")
index = RepoKnowledgeIndex()


@app.on_event("startup")
def startup() -> None:
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
    hits = index.search(req.query, top_k=req.top_k, collections=req.collections)
    return {"query": req.query, "hits": hits}


@app.post("/ask/docs")
async def ask_docs(req: AskRequest) -> Dict[str, Any]:
    model = req.model or DEFAULT_MODEL
    intent, hits, used_stage = staged_search(req.query, req.top_k_context)
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
                f"Intent: {intent}\nRetrieval stage used: {used_stage}\n\n"
                f"REPOSITORY CONTEXT:\n\n{context}"
            ),
        },
        {"role": "user", "content": req.query},
    ]
    answer = await ollama_chat(model, messages, temperature=req.temperature)

    return {
        "answer": answer,
        "intent": intent,
        "retrieval": {"used": True, "stage": used_stage, "hits": hits},
        "sources": refs,
    }


@app.post("/draft-doc")
async def draft_doc(req: DraftDocRequest) -> Dict[str, Any]:
    model = req.model or DEFAULT_MODEL

    # Try docs and code so the generated doc patch can match existing style and use grounded facts.
    hits = index.search(req.query, top_k=4, collections=["code", "docs"])
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

    draft = await ollama_chat(model, messages, temperature=req.temperature)

    draft = draft.strip()
    if draft.startswith("```markdown"):
        draft = draft.removeprefix("```markdown").strip()
    if draft.startswith("```"):
        draft = draft.removeprefix("```").strip()
    if draft.endswith("```"):
        draft = draft.removesuffix("```").strip()


    return {
        "target_path": target_hint,
        "draft_markdown": draft,
        "sources_used_for_grounding": refs,
        "note": "Проверьте TODO и факты перед коммитом. Endpoint не записывает файл автоматически.",
    }


@app.post("/postmortem")
async def postmortem(req: PostmortemRequest) -> Dict[str, Any]:
    model = req.model or DEFAULT_MODEL

    # Use SRE docs/runbooks as style and remediation context, but do not let them override incident evidence.
    search_query = req.title or req.incident_description or req.logs[:300] or "incident postmortem"
    hits = index.search(search_query, top_k=8, collections=["sre", "runbooks", "docs"])
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

    report = await ollama_chat(model, messages, temperature=req.temperature)

    return {
        "postmortem_markdown": report,
        "sources_used_for_context": refs,
        "note": "Постмортем построен по предоставленным чатам/логам/метрикам. Неизвестные факты должны быть подтверждены человеком.",
    }


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")

    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Ollama error: {resp.text}")

    data = resp.json()
    models = []
    for m in data.get("models", []):
        name = m.get("name") or m.get("model")
        if name:
            models.append({"id": name, "object": "model", "owned_by": "ollama"})

    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> Dict[str, Any]:
    try:
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

#        if req.stream:
#            # OpenWebUI can work without streaming. Keeping explicit error is safer than pretending SSE works.
#            raise HTTPException(
#                status_code=400,
#                detail="stream=true is not implemented in this version. Set stream=false in OpenWebUI.",
#            )

        model = req.model or DEFAULT_MODEL
        user_query = get_last_user_message(req.messages)
        use_retrieval = should_use_retrieval(req.messages)

        if use_retrieval:
            intent, hits, used_stage = staged_search(user_query, req.top_k_context)
            context, refs = build_context(hits, MAX_CONTEXT_CHARS)

            if not hits:
                content = (
                    "В индексированной документации, SRE-разделах, runbook'ах и комментариях/докстрингах кода "
                    "не найден точный ответ на этот вопрос."
                )
            else:
                messages = build_rag_messages(req.messages, context, intent, used_stage)
                content = await ollama_chat(model, messages, temperature=req.temperature)

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
                            "context": refs,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                "retrieval": {"used": True, "intent": intent, "stage": used_stage, "hits": hits},
            }

        plain_messages = build_plain_messages(req.messages)
        content = await ollama_chat(model, plain_messages, temperature=0.3)

        return {
            "id": f"chatcmpl-{sha1_text(str(time.time()))[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "context": []},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            "retrieval": {"used": False, "hits": []},
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Internal retrieval/generation error: {e}") from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)