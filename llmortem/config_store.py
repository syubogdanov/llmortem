from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from sqlalchemy import Boolean, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://llmortem:llmortem@localhost:5432/llmortem",
)

CONFIG_CACHE_TTL_SECONDS = int(os.getenv("CONFIG_CACHE_TTL_SECONDS", "300"))


class Base(DeclarativeBase):
    pass


class ConfigItem(Base):
    __tablename__ = "config_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    key: Mapped[str] = mapped_column(String(255), index=True)
    value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)


@dataclass
class RuntimeConfig:
    stopwords: Set[str]
    synonyms: Dict[str, List[str]]
    technical_keywords: Set[str]
    doc_qa_patterns: List[str]
    incident_patterns: List[str]


engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

_cache: Optional[RuntimeConfig] = None
_cache_loaded_at: float = 0.0


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def seed_defaults(
    *,
    stopwords: Set[str],
    synonyms: Dict[str, List[str]],
    technical_keywords: Set[str],
    doc_qa_patterns: List[str],
    incident_patterns: List[str],
) -> None:
    with SessionLocal() as session:
        exists = session.scalar(select(ConfigItem.id).limit(1))
        if exists is not None:
            return

        items: List[ConfigItem] = []

        for word in sorted(stopwords):
            items.append(ConfigItem(kind="stopword", key=word, value=None, enabled=True))

        for key, values in sorted(synonyms.items()):
            for value in values:
                items.append(ConfigItem(kind="synonym", key=key, value=value, enabled=True))

        for word in sorted(technical_keywords):
            items.append(ConfigItem(kind="technical_keyword", key=word, value=None, enabled=True))

        for pattern in doc_qa_patterns:
            items.append(ConfigItem(kind="doc_qa_pattern", key=pattern, value=None, enabled=True))

        for pattern in incident_patterns:
            items.append(ConfigItem(kind="incident_pattern", key=pattern, value=None, enabled=True))

        session.add_all(items)
        session.commit()


def load_config_from_db() -> RuntimeConfig:
    with SessionLocal() as session:
        rows = session.scalars(
            select(ConfigItem).where(ConfigItem.enabled.is_(True))
        ).all()

    stopwords: Set[str] = set()
    synonyms: Dict[str, List[str]] = {}
    technical_keywords: Set[str] = set()
    doc_qa_patterns: List[str] = []
    incident_patterns: List[str] = []

    for row in rows:
        if row.kind == "stopword":
            stopwords.add(row.key)
        elif row.kind == "synonym" and row.value:
            synonyms.setdefault(row.key, []).append(row.value)
        elif row.kind == "technical_keyword":
            technical_keywords.add(row.key)
        elif row.kind == "doc_qa_pattern":
            doc_qa_patterns.append(row.key)
        elif row.kind == "incident_pattern":
            incident_patterns.append(row.key)

    return RuntimeConfig(
        stopwords=stopwords,
        synonyms=synonyms,
        technical_keywords=technical_keywords,
        doc_qa_patterns=doc_qa_patterns,
        incident_patterns=incident_patterns,
    )


def get_runtime_config() -> RuntimeConfig:
    global _cache, _cache_loaded_at

    now = time.time()
    if _cache is not None and now - _cache_loaded_at < CONFIG_CACHE_TTL_SECONDS:
        return _cache

    _cache = load_config_from_db()
    _cache_loaded_at = now
    return _cache


def clear_config_cache() -> None:
    global _cache, _cache_loaded_at
    _cache = None
    _cache_loaded_at = 0.0