from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class QuestionSample:
    """
    Minimal representation of a logged QA interaction.
    """

    question: str
    answer: Optional[str] = None
    status: Optional[str] = None
    similarity: Optional[float] = None
    model: Optional[str] = None
    cited_document_ids: Optional[List[str]] = None
    metadata: Optional[dict] = None


def _parse_similarity(raw: str | None) -> Optional[float]:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _parse_document_ids(raw: str | None) -> Optional[List[str]]:
    if not raw:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    if cleaned.startswith("[") and cleaned.endswith("]"):
        try:
            cleaned = cleaned.replace('""', '"')
            parsed = json.loads(cleaned)
            return [str(item) for item in parsed if item]
        except Exception:
            return None
    return None


def load_supabase_qa_export(path: Path | str, *, limit: Optional[int] = None) -> List[QuestionSample]:
    """
    Load rows from Supabase QA exports such as
    `data/Supabase Snippet Ad hoc queries.csv`.
    """

    path = Path(path)
    samples: List[QuestionSample] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            question = (row.get("question") or "").strip()
            if not question:
                continue

            sample = QuestionSample(
                question=question,
                answer=(row.get("answer") or "").strip() or None,
                status=(row.get("status") or "").strip() or None,
                similarity=_parse_similarity(row.get("similarity_score")),
                model=(row.get("llm_model") or "").strip() or None,
                cited_document_ids=_parse_document_ids(row.get("cited_document_ids")),
                metadata={
                    "id": row.get("id"),
                    "created_at": row.get("created_at"),
                    "guardian_id": row.get("guardian_id"),
                },
            )
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
    return samples


def load_question_prompts(path: Path | str, *, limit: Optional[int] = None) -> List[str]:
    """
    Read question-only CSV exports such as
    `data/Supabase Snippet Questions asked.csv`.
    """

    path = Path(path)
    questions: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            question = (row.get("question") or "").strip()
            if question:
                questions.append(question)
                if limit is not None and len(questions) >= limit:
                    break
    return questions


def merge_unique_questions(*sources: Iterable[str]) -> List[str]:
    """
    Merge multiple question lists while preserving the original order.
    """

    seen = set()
    merged: List[str] = []
    for source in sources:
        for question in source:
            key = question.strip()
            if not key or key in seen:
                continue
            merged.append(question)
            seen.add(key)
    return merged
