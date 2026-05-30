from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from llmortem.code_scanner import extract_generic_code_knowledge, extract_python_knowledge
from llmortem.document_loader import (
    file_hash_from_text,
    line_span_for_text,
    load_document_file,
    safe_read_text,
    split_paragraphs,
)
from llmortem.settings import (
    CHUNK_OVERLAP,
    CODE_EXTENSIONS,
    DOC_EXTENSIONS,
    DOCS_DIR,
    EXCLUDED_DIR_NAMES,
    INDEX_DIR,
    MAX_CHUNK_TOKENS,
    REPO_BROWSER_BASE_URL,
    REPO_ROOT,
    RUNBOOK_DIR,
    SRE_DOCS_DIR,
)
from llmortem.text_processing import expand_query_tokens, normalize_text, sha1_text, tokenize


@dataclass
class Chunk:
    chunk_id: str
    collection: str
    source: str
    title: str
    section: str
    text: str
    tokens: List[str]
    norm_text: str
    file_hash: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def make_source_url(source: str, start_line: Optional[int] = None) -> str:
    if not REPO_BROWSER_BASE_URL:
        return f"repo://{source}#L{start_line}" if start_line else f"repo://{source}"

    url = f"{REPO_BROWSER_BASE_URL}/{source}"
    if start_line:
        url += f"#L{start_line}"
    return url


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

    for paragraph in paragraphs:
        header = re.match(r"^(#{1,6})\s+(.*)$", paragraph)
        if header:
            section = header.group(2).strip()

        paragraph_tokens = paragraph.split()
        paragraph_len = len(paragraph_tokens)

        if paragraph_len > MAX_CHUNK_TOKENS:
            step = max(1, MAX_CHUNK_TOKENS - CHUNK_OVERLAP)
            for start in range(0, paragraph_len, step):
                sub = " ".join(paragraph_tokens[start:start + MAX_CHUNK_TOKENS])
                emit(sub, section, idx)
                idx += 1
            continue

        if current_len + paragraph_len > MAX_CHUNK_TOKENS and current:
            joined = "\n\n".join(current)
            emit(joined, section, idx)
            idx += 1

            overlap_words = " ".join(joined.split()[-CHUNK_OVERLAP:]) if CHUNK_OVERLAP else ""
            current = [overlap_words, paragraph] if overlap_words else [paragraph]
            current_len = len(overlap_words.split()) + paragraph_len
        else:
            current.append(paragraph)
            current_len += paragraph_len

    if current:
        emit("\n\n".join(current), section, idx)

    return chunks


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
            try:
                data_hash = hashlib.sha1(path.read_bytes()).hexdigest()
            except OSError:
                continue

            parts.append(f"{repo_relative(path)}:{data_hash}")

        return sha1_text("|".join(sorted(parts))) if parts else "empty"

    def build_or_load(self, force: bool = False) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        index_pkl = INDEX_DIR / "knowledge_index.pkl"
        meta_json = INDEX_DIR / "knowledge_meta.json"

        current_fp = self._compute_source_fingerprint()

        if not force and index_pkl.exists() and meta_json.exists():
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == current_fp:
                with index_pkl.open("rb") as file:
                    payload = pickle.load(file)

                self.chunks = payload["chunks"]
                self.bm25 = payload["bm25"]
                self.meta = meta
                return

        self._build_from_files(current_fp)

        with index_pkl.open("wb") as file:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, file)

        meta_json.write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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

        for path in self._iter_files([DOCS_DIR, RUNBOOK_DIR], DOC_EXTENSIONS):
            title, text = load_document_file(path)
            if not text.strip():
                continue

            source = repo_relative(path)
            collection = self._collection_for_document(path)
            chunks.extend(
                chunk_document(
                    collection=collection,
                    source=source,
                    title=title,
                    text=text,
                    file_hash=file_hash_from_text(text),
                )
            )

        for path in self._iter_files([REPO_ROOT], CODE_EXTENSIONS):
            source = repo_relative(path)

            if source.startswith("docs/") or source.startswith("runbooks/"):
                continue

            if path.suffix.lower() == ".py":
                extracted = extract_python_knowledge(path)
            else:
                extracted = extract_generic_code_knowledge(path)

            if not extracted.strip():
                continue

            chunks.extend(
                chunk_document(
                    collection="code",
                    source=source,
                    title=path.stem,
                    text=extracted,
                    file_hash=file_hash_from_text(safe_read_text(path)),
                )
            )

        if not chunks:
            raise RuntimeError("No docs/runbooks/code knowledge found.")

        self.chunks = chunks
        self.bm25 = BM25Okapi([chunk.tokens for chunk in chunks])
        self.meta = {
            "fingerprint": fingerprint,
            "created_at": time.time(),
            "num_chunks": len(chunks),
            "collections": self.collection_counts(),
            "repo_root": str(REPO_ROOT),
        }

    def collection_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for chunk in self.chunks:
            counts[chunk.collection] = counts.get(chunk.collection, 0) + 1
        return counts

    def _score_chunk_boost(self, query: str, chunk: Chunk, intent: Optional[str]) -> float:
        q_norm = normalize_text(query)
        q_tokens = set(tokenize(query))
        score = 0.0

        if q_norm and q_norm in chunk.norm_text:
            score += 3.0

        for token in q_tokens:
            if token in chunk.tokens:
                score += 0.2
            if token in normalize_text(chunk.title):
                score += 0.5
            if token in normalize_text(chunk.section):
                score += 0.35
            if token in normalize_text(chunk.source):
                score += 0.25

        if intent == "docs_qa":
            if chunk.collection == "docs":
                score += 0.7
            elif chunk.collection == "code":
                score -= 0.2

        if intent == "incident":
            if chunk.collection in {"sre", "runbooks"}:
                score += 1.0
            elif chunk.collection == "code":
                score += 0.15

        if {"restart", "перезапустить", "перезапуск"} & q_tokens:
            if "systemctl restart" in chunk.norm_text or "kubectl rollout restart" in chunk.norm_text:
                score += 1.2

        return score

    def search(
        self,
        query: str,
        top_k: int = 8,
        collections: Optional[List[str]] = None,
        intent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.chunks or self.bm25 is None:
            self.build_or_load(force=False)

        allowed = set(collections or [])
        query_tokens = expand_query_tokens(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked: List[Tuple[int, float, float]] = []

        for idx, bm25_score in enumerate(scores):
            chunk = self.chunks[idx]

            if allowed and chunk.collection not in allowed:
                continue

            boost = self._score_chunk_boost(query, chunk, intent)
            final_score = float(bm25_score) + boost

            if final_score > 0:
                ranked.append((idx, float(bm25_score), float(final_score)))

        ranked.sort(key=lambda item: item[2], reverse=True)

        results: List[Dict[str, Any]] = []

        for idx, bm25_score, final_score in ranked[:top_k]:
            chunk = self.chunks[idx]

            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "collection": chunk.collection,
                    "source": chunk.source,
                    "source_url": make_source_url(chunk.source, chunk.start_line),
                    "title": chunk.title,
                    "section": chunk.section,
                    "text": chunk.text,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "score": round(final_score, 6),
                    "signals": {
                        "bm25": round(bm25_score, 6),
                        "boosted": round(final_score, 6),
                    },
                }
            )

        return results