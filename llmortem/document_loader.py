from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple

import yaml
from bs4 import BeautifulSoup

from llmortem.text_processing import sha1_text


CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")


def strip_markdown(markdown: str) -> str:
    markdown = CODE_FENCE_RE.sub(
        lambda match: "\n" + match.group(0).replace("```", "") + "\n",
        markdown,
    )
    markdown = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", markdown)
    markdown = re.sub(r"^[>*-]\s?", "", markdown, flags=re.MULTILINE)
    markdown = re.sub(r"[*_~]", "", markdown)
    return markdown


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


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
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]
    return blocks or ([text.strip()] if text.strip() else [])


def line_span_for_text(full_text: str, fragment: str) -> Tuple[int | None, int | None]:
    if not fragment.strip():
        return None, None

    pos = full_text.find(fragment[: min(80, len(fragment))])
    if pos < 0:
        return None, None

    start_line = full_text[:pos].count("\n") + 1
    end_line = start_line + fragment.count("\n")
    return start_line, end_line


def file_hash_from_text(text: str) -> str:
    return sha1_text(text)