from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .date_parser import extract_all_dates, infer_date_from_text
from .text_extraction import PageLayout

try:  # pragma: no cover - optional dependency
    import spacy  # type: ignore
    _NLP = spacy.load("en_core_web_sm")
except Exception:  # pragma: no cover - spaCy is optional
    _NLP = None


EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[- ]?)?(?:\d{3,4}[- ]?){2,3}\d{3,4}\b")
NAME_HINT_REGEX = re.compile(r"(?:Mr|Mrs|Ms|Dr)\.?\s+[A-Z][A-Za-z]+")
ENTITY_PHRASE_REGEX = re.compile(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b")

ACTION_KEYWORDS = {
    "submit",
    "bring",
    "return",
    "pay",
    "complete",
    "attend",
    "collect",
    "hand in",
    "register",
    "enrol",
    "enroll",
    "share",
    "fill",
}

DATE_KIND_HINTS: Sequence[tuple[str, str]] = (
    ("begins", "start"),
    ("starts", "start"),
    ("commence", "start"),
    ("opens", "start"),
    ("ends", "end"),
    ("concludes", "end"),
    ("resume", "resume"),
    ("reopens", "resume"),
    ("deadline", "deadline"),
    ("last date", "deadline"),
    ("due", "deadline"),
)


@dataclass
class StructuredDate:
    kind: str
    date: dt.date
    text: str
    confidence: float = 0.6


@dataclass
class StructuredAction:
    description: str
    due_date: Optional[dt.date]
    audience: Optional[str]
    confidence: float
    source_excerpt: str


@dataclass
class StructuredContact:
    name: Optional[str]
    role: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    notes: Optional[str]


@dataclass
class StructuredEntity:
    value: str
    entity_type: str
    confidence: float
    context: str


@dataclass
class PageSummary:
    page_number: int
    summary: str


@dataclass
class StructuredHeading:
    page_number: int
    text: str
    level: int


@dataclass
class StructuredTable:
    page_number: int
    rows: List[List[Optional[str]]]


@dataclass
class StructuredDocumentData:
    title_confidence: float
    issued_date: Optional[dt.date]
    effective_start: Optional[dt.date]
    effective_end: Optional[dt.date]
    audience: List[str]
    actions: List[StructuredAction]
    contacts: List[StructuredContact]
    entities: List[StructuredEntity]
    dates: List[StructuredDate]
    page_summaries: List[PageSummary]
    headings: List[StructuredHeading]
    tables: List[StructuredTable]


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _classify_date(raw_text: str, surrounding: str) -> str:
    lowered = surrounding.lower()
    for needle, label in DATE_KIND_HINTS:
        if needle in lowered:
            return label
    if "break" in lowered or "holiday" in lowered:
        if "start" in lowered or "begin" in lowered:
            return "start"
        if "end" in lowered or "resume" in lowered:
            return "end"
    return "generic"


def extract_structured_dates(text: str) -> List[StructuredDate]:
    dates: List[StructuredDate] = []
    for raw, date_value in extract_all_dates(text):
        start_idx = text.lower().find(raw.lower())
        window = text[max(0, start_idx - 60) : start_idx + len(raw) + 60] if start_idx >= 0 else raw
        kind = _classify_date(raw, window)
        confidence = 0.7 if kind != "generic" else 0.5
        dates.append(StructuredDate(kind=kind, date=date_value, text=raw, confidence=confidence))
    return dates


def extract_contacts(text: str) -> List[StructuredContact]:
    contacts: List[StructuredContact] = []
    for line in text.splitlines():
        emails = EMAIL_REGEX.findall(line)
        phones = PHONE_REGEX.findall(line)
        if not emails and not phones:
            continue
        name = None
        role = None
        name_match = NAME_HINT_REGEX.search(line)
        if name_match:
            name = name_match.group(0)
        else:
            tokens = line.split(",")[0].split(" - ")[0]
            if any(char.isalpha() for char in tokens):
                name = tokens.strip()
        if "transport" in line.lower():
            role = "Transport"
        elif "fee" in line.lower() or "accounts" in line.lower():
            role = "Accounts"
        elif "teacher" in line.lower() or "class" in line.lower():
            role = "Academics"
        contacts.append(
            StructuredContact(
                name=name,
                role=role,
                email=emails[0] if emails else None,
                phone=phones[0] if phones else None,
                notes=line.strip(),
            )
        )
    return contacts


def extract_actions(text: str, audience_fallback: Optional[str] = None) -> List[StructuredAction]:
    actions: List[StructuredAction] = []
    for sentence in _split_sentences(text):
        lowered = sentence.lower()
        if not any(keyword in lowered for keyword in ACTION_KEYWORDS):
            continue
        due_date = infer_date_from_text(sentence)
        audience = None
        if "parent" in lowered:
            audience = "Parents"
        elif "learner" in lowered or "students" in lowered:
            audience = "Learners"
        elif "teacher" in lowered:
            audience = "Teachers"
        confidence = 0.6
        if due_date:
            confidence += 0.2
        if audience is None and audience_fallback:
            audience = audience_fallback
        actions.append(
            StructuredAction(
                description=sentence[:300],
                due_date=due_date,
                audience=audience,
                confidence=min(confidence, 0.95),
                source_excerpt=sentence[:500],
            )
        )
    return actions


def extract_entities(text: str) -> List[StructuredEntity]:
    entities: List[StructuredEntity] = []
    seen: set[str] = set()

    if _NLP is not None:
        try:
            doc = _NLP(text)
            for ent in doc.ents:
                value = ent.text.strip()
                if not value:
                    continue
                key = value.lower()
                if key in seen:
                    continue
                seen.add(key)
                entities.append(
                    StructuredEntity(
                        value=value,
                        entity_type=ent.label_,
                        confidence=0.75,
                        context=ent.sent.text[:300] if ent.sent is not None else value,
                    )
                )
        except Exception:  # pragma: no cover - spaCy failures
            pass

    if not entities:  # fallback heuristic
        for sentence in _split_sentences(text):
            for match in ENTITY_PHRASE_REGEX.finditer(sentence):
                value = match.group(1).strip()
                if len(value.split()) == 1:
                    continue
                key = value.lower()
                if key in seen:
                    continue
                seen.add(key)
                entities.append(
                    StructuredEntity(
                        value=value,
                        entity_type="proper_noun",
                        confidence=0.5,
                        context=sentence[:300],
                    )
                )
    return entities


def summarise_pages(page_texts: Iterable[str]) -> List[PageSummary]:
    summaries: List[PageSummary] = []
    for idx, page_text in enumerate(page_texts, start=1):
        cleaned = page_text.strip()
        if not cleaned:
            continue
        first_sentence = _split_sentences(cleaned)
        if first_sentence:
            summary = first_sentence[0][:300]
        else:
            summary = cleaned[:300]
        summaries.append(PageSummary(page_number=idx, summary=summary))
    return summaries


def derive_effective_range(dates: Sequence[StructuredDate]) -> tuple[Optional[dt.date], Optional[dt.date]]:
    start_candidates = [item.date for item in dates if item.kind in {"start"}]
    end_candidates = [item.date for item in dates if item.kind in {"end"}]
    generic = [item.date for item in dates if item.kind == "generic"]

    start = min(start_candidates) if start_candidates else (min(generic) if generic else None)
    end = max(end_candidates) if end_candidates else (max(generic) if generic else None)
    if start and end and end < start:
        end = start
    return start, end


def extract_structured_data(
    *,
    text: str,
    page_texts: Iterable[str],
    published_on: Optional[dt.date],
    audience: Sequence[str] | None = None,
    layout: Sequence[PageLayout] | None = None,
) -> StructuredDocumentData:
    dates = extract_structured_dates(text)
    effective_start, effective_end = derive_effective_range(dates)

    audience_label = None
    if audience:
        audience_label = ", ".join(audience)
        audience_list = list(audience)
    else:
        audience_list = []

    actions = extract_actions(text, audience_fallback=audience_label)
    contacts = extract_contacts(text)
    entities = extract_entities(text)
    page_summaries = summarise_pages(page_texts)
    headings: List[StructuredHeading] = []
    tables: List[StructuredTable] = []

    if layout:
        for item in layout:
            for idx, heading in enumerate(item.headings, start=1):
                headings.append(
                    StructuredHeading(
                        page_number=item.page_number,
                        text=heading,
                        level=idx,
                    )
                )
            for table in item.tables:
                tables.append(
                    StructuredTable(
                        page_number=item.page_number,
                        rows=[[cell if cell is not None else "" for cell in row] for row in table],
                    )
                )

    issued_date = published_on

    return StructuredDocumentData(
        title_confidence=0.9,
        issued_date=issued_date,
        effective_start=effective_start,
        effective_end=effective_end,
        audience=audience_list,
        actions=actions,
        contacts=contacts,
        entities=entities,
        dates=dates,
        page_summaries=page_summaries,
        headings=headings,
        tables=tables,
    )
