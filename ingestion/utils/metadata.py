from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _normalise_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text)
    return cleaned.lower()


def infer_title(path: Path) -> str:
    """Generate a human-friendly title from the filename."""
    stem = path.stem.replace("_", " ").replace("-", " ")
    collapsed = re.sub(r"\s+", " ", stem).strip()
    if not collapsed:
        return "Untitled Document"
    # Ensure acronyms stay uppercase (PDF, PTA, etc.)
    words = []
    for part in collapsed.split(" "):
        if part.isupper():
            words.append(part)
        else:
            words.append(part.capitalize())
    return " ".join(words)


GRADE_PATTERNS = [
    re.compile(r"\bgrade[ -]?(\d)\b", re.IGNORECASE),
    re.compile(r"\bgr?\s*(\d)\b", re.IGNORECASE),
    re.compile(r"\bclass[ -]?(\d)\b", re.IGNORECASE),
]

AUDIENCE_KEYWORDS = {
    "nursery": "Early Years",
    "sr kg": "Senior KG",
    "jr kg": "Junior KG",
    "primary": "Primary",
    "secondary": "Secondary",
    "whole school": "Whole School",
}

EVENT_KEYWORDS = {
    "orientation": "orientation",
    "trip": "trip",
    "field trip": "trip",
    "excursion": "trip",
    "calendar": "calendar",
    "schedule": "schedule",
    "timetable": "schedule",
    "menu": "cafeteria",
    "cafeteria": "cafeteria",
    "lunch": "cafeteria",
    "fee": "fees",
    "payment": "fees",
    "assessment": "assessment",
    "exam": "assessment",
    "test": "assessment",
    "holiday": "holiday",
    "break": "holiday",
    "diwali": "holiday",
    "dussehra": "holiday",
    "reopen": "holiday",
    "sports": "sports",
    "competition": "sports",
    "workshop": "workshop",
    "webinar": "workshop",
    "orientation": "orientation",
    "meeting": "meeting",
    "coffee morning": "meeting",
    "pta": "pta",
    "transport": "transport",
    "bus": "transport",
    "newsletter": "newsletter",
    "results": "results",
}

DOC_TYPE_KEYWORDS = {
    "calendar": "calendar",
    "schedule": "notice",
    "timetable": "notice",
    "menu": "notice",
    "newsletter": "newsletter",
    "newsletter": "newsletter",
    "form": "form",
    "consent": "form",
}


def infer_grade_tags(text: str) -> Optional[List[str]]:
    matches: Counter[str] = Counter()
    for pattern in GRADE_PATTERNS:
        for match in pattern.findall(text):
            grade = int(match)
            matches[f"Grade {grade}"] += 1

    if matches:
        top = [grade for grade, _ in matches.most_common()]
        return top

    # Broader audience tags (primary / secondary)
    fallback: List[str] = []
    if "primary" in text:
        fallback.append("Primary")
    if "secondary" in text:
        fallback.append("Secondary")
    if fallback:
        return fallback
    return None


def infer_event_tags(text: str) -> Optional[List[str]]:
    tags: List[str] = []
    for keyword, tag in EVENT_KEYWORDS.items():
        if keyword in text:
            tags.append(tag)
    if tags:
        return sorted(set(tags))
    return None


def infer_doc_type(text: str) -> Optional[str]:
    for keyword, doc_type in DOC_TYPE_KEYWORDS.items():
        if keyword in text:
            return doc_type
    return None


def infer_metadata(path: Path, text: str) -> Dict[str, Optional[Iterable[str]]]:
    combined = _normalise_text(path.stem + " " + text[:4000])
    title = infer_title(path)
    grade_tags = infer_grade_tags(combined)
    event_tags = infer_event_tags(combined)
    doc_type = infer_doc_type(combined)

    return {
        "title": title,
        "grade_tags": grade_tags,
        "event_tags": event_tags,
        "doc_type": doc_type,
    }
