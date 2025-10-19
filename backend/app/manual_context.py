from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple


MANUAL_FACTS_PATH = Path(__file__).resolve().parents[2] / "data" / "manual" / "manual_facts.json"


def load_manual_facts() -> List[dict]:
    if not MANUAL_FACTS_PATH.exists():
        return []
    with MANUAL_FACTS_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def match_manual_facts(question: str) -> List[Tuple[str, str]]:
    question_lower = question.lower()
    matches: List[Tuple[str, str]] = []
    for fact in load_manual_facts():
        for keyword in fact.get("keywords", []):
            if keyword.lower() in question_lower:
                matches.append((fact.get("title", "Manual Note"), fact.get("content", "")))
                break
    return matches
