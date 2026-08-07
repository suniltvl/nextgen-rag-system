"""Data access layer for the RAG dashboard.

Every function here reads from the local JSON sample dataset. The function
signatures are the contract the UI depends on — swap the bodies for
`requests.get(...)` / `requests.post(...)` calls against a future
`POST /api/rag/query` backend and nothing in `ui.py` or `callbacks.py` needs
to change.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from models import Domain, RAGResult

SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"
DOMAINS_FILE = SAMPLE_DATA_DIR / "domains.json"
QUESTIONS_FILE = SAMPLE_DATA_DIR / "questions.json"


@lru_cache(maxsize=1)
def _load_domains_raw() -> list[dict]:
    return json.loads(DOMAINS_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_questions_raw() -> list[dict]:
    return json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))


def load_domains() -> list[Domain]:
    """Return every domain available for selection."""
    return [Domain(**d) for d in _load_domains_raw()]


def get_domain_names() -> list[str]:
    """Return domain display names, in the order defined by domains.json."""
    return [d.name for d in load_domains()]


def get_domain_id_by_name(domain_name: str) -> str | None:
    for d in load_domains():
        if d.name == domain_name:
            return d.id
    return None


def load_questions_for_domain(domain_name: str) -> list[str]:
    """Return the question strings that belong to a given domain (by display name)."""
    domain_id = get_domain_id_by_name(domain_name)
    if domain_id is None:
        return []
    return [q["question"] for q in _load_questions_raw() if q["domain_id"] == domain_id]


def get_result_for_question(domain_name: str, question_text: str) -> RAGResult | None:
    """Run the (simulated) RAG pipeline for a question and return the full result."""
    domain_id = get_domain_id_by_name(domain_name)
    if domain_id is None:
        return None
    for q in _load_questions_raw():
        if q["domain_id"] == domain_id and q["question"] == question_text:
            return RAGResult.from_dict(q)
    return None


def total_question_count() -> int:
    return len(_load_questions_raw())
