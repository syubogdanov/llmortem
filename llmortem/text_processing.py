from __future__ import annotations

import hashlib
import re
from typing import List

from unidecode import unidecode

from llmortem.config_store import get_runtime_config


TOKEN_RE = re.compile(r"[\w\-./:]+", re.UNICODE)
WS_RE = re.compile(r"\s+")


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
                out.extend([part for part in tok.split(sep) if part])

        if "-" in tok:
            out.append(tok.replace("-", ""))

    config = get_runtime_config()
    return [tok for tok in out if tok and tok not in config.stopwords]


def expand_query_tokens(query: str) -> List[str]:
    config = get_runtime_config()
    tokens = tokenize(query)
    extra: List[str] = []

    for token in tokens:
        if token in config.synonyms:
            extra.extend(config.synonyms[token])

    return tokens + extra