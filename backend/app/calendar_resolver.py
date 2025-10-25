from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from rapidfuzz import fuzz

from .temporal_context import (
    format_date,
    infer_date_references,
    normalise_event_title,
    parse_date_range_from_text,
)

GENERIC_WORDS = {
    "what",
    "when",
    "where",
    "who",
    "how",
    "is",
    "the",
    "for",
    "on",
    "does",
    "do",
    "a",
    "an",
    "of",
    "to",
    "in",
    "next",
    "last",
    "previous",
    "coming",
    "upcoming",
    "was",
    "will",
    "be",
    "break",
    "holiday",
    "holidays",
    "vacation",
    "school",
    "calendar",
    "schedule",
    "start",
    "end",
    "resume",
    "reopens",
    "information",
    "details",
}

SYNONYM_MAP = {
    "xmas": "christmas",
    "christmas": "christmas",
    "vacation": "break",
    "holidays": "break",
    "holiday": "break",
}

NEXT_WORDS = {"next", "upcoming", "coming"}
PREVIOUS_WORDS = {"last", "previous", "earlier", "ago"}
DURATION_WORDS = {"howlong", "duration"}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass
class CalendarEventRecord:
    id: str
    title: str
    normalized_title: str
    description: str
    description_lower: str
    start: dt.date
    end: dt.date
    audience: tuple[str, ...]
    source: Optional[str]

    @property
    def duration_days(self) -> int:
        return max(1, (self.end - self.start).days + 1)

    def matches_keywords(self, keywords: Sequence[str]) -> bool:
        haystack = f"{self.normalized_title} {self.description_lower}"
        return all(keyword in haystack for keyword in keywords)


class CalendarCache:
    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl = ttl_seconds
        self._expires: Optional[dt.datetime] = None
        self._data: List[CalendarEventRecord] = []

    def get(self, client) -> List[CalendarEventRecord]:
        now = dt.datetime.utcnow()
        if self._expires and self._expires > now:
            return self._data

        response = (
            client.table("calendar_events")
            .select("id,title,event_date,end_date,audience,description,source")
            .execute()
        )

        events: List[CalendarEventRecord] = []
        for row in response.data or []:
            record = _build_event(row)
            if record:
                events.append(record)
        events.sort(key=lambda ev: (ev.start, ev.end, ev.title))
        self._data = events
        self._expires = now + dt.timedelta(seconds=self.ttl)
        return events


_CACHE = CalendarCache()


def reset_cache() -> None:
    _CACHE._expires = None
    _CACHE._data = []


@dataclass
class CalendarAnswer:
    event_id: str
    title: str
    start: dt.date
    end: dt.date
    description: str
    audience: tuple[str, ...]
    source: Optional[str]
    mode: str
    score: float

    def formatted_answer(self, include_duration: bool = True) -> str:
        if self.start == self.end:
            range_text = f"on {format_date(self.start)}"
        else:
            range_text = f"from {format_date(self.start)} to {format_date(self.end)}"

        if self.mode == "next":
            prefix = "The next event is"
            base = f"{prefix} {self.title} {range_text}".strip()
        elif self.mode == "previous":
            prefix = "The last event was"
            base = f"{prefix} {self.title} {range_text}".strip()
        elif self.mode == "date_lookup":
            base = f"On that date {self.title} {range_text}".strip()
        else:
            base = f"{self.title} runs {range_text}".strip()

        if include_duration:
            duration_days = max(1, (self.end - self.start).days + 1)
            if duration_days > 1:
                base = f"{base} ({duration_days} days)"

        if self.description:
            base = f"{base}. {self.description.strip()}"

        if self.audience:
            audience_label = ", ".join(sorted(self.audience))
            base = f"{base}. Audience: {audience_label}."

        return base.strip()


def _canonicalize_keyword(token: str) -> str:
    token = token.lower()
    return SYNONYM_MAP.get(token, token)


def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _clean_keywords(tokens: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for token in tokens:
        canonical = _canonicalize_keyword(token)
        if canonical and canonical not in GENERIC_WORDS:
            cleaned.append(canonical)
    return cleaned


def _parse_dates(row: dict) -> tuple[Optional[dt.date], Optional[dt.date]]:
    start_raw = row.get("event_date")
    end_raw = row.get("end_date") or start_raw
    start_date = None
    end_date = None
    try:
        if start_raw:
            start_date = dt.date.fromisoformat(start_raw)
    except Exception:
        start_date = None
    try:
        if end_raw:
            end_date = dt.date.fromisoformat(end_raw)
    except Exception:
        end_date = start_date

    description = row.get("description") or ""
    if description:
        reference_year = start_date.year if isinstance(start_date, dt.date) else dt.date.today().year
        parsed_start, parsed_end = parse_date_range_from_text(description, reference_year)
        if parsed_start:
            start_date = parsed_start
        if parsed_end:
            end_date = parsed_end

    if start_date and end_date and end_date < start_date:
        end_date = start_date
    return start_date, end_date


def _build_event(row: dict) -> Optional[CalendarEventRecord]:
    start, end = _parse_dates(row)
    if not start:
        return None
    end = end or start

    title = row.get("title") or "Calendar Event"
    normalized = normalise_event_title(title)
    description_raw = (row.get("description") or "").strip()
    description_lower = description_raw.lower()
    audience = tuple(row.get("audience") or [])
    source = row.get("source")

    return CalendarEventRecord(
        id=str(row.get("id") or title),
        title=title,
        normalized_title=normalized,
        description=description_raw,
        description_lower=description_lower,
        start=start,
        end=end,
        audience=audience,
        source=source,
    )


def _score_event(event: CalendarEventRecord, keywords: Sequence[str]) -> float:
    if not keywords:
        return 0.0
    target = " ".join(keywords)
    candidate = event.normalized_title
    desc = event.description_lower
    title_score = fuzz.token_set_ratio(target, candidate)
    desc_score = fuzz.partial_ratio(target, desc) if desc else 0
    return max(title_score, desc_score)


def _filter_by_keywords(events: Iterable[CalendarEventRecord], keywords: Sequence[str]) -> List[CalendarEventRecord]:
    if not keywords:
        return list(events)
    filtered = [event for event in events if event.matches_keywords(keywords)]
    return filtered or list(events)


def resolve_calendar_question(
    *,
    client,
    question: str,
    reference_date: dt.date,
) -> Optional[CalendarAnswer]:
    question_lower = question.lower()
    tokens = _tokenize(question_lower)
    keywords = _clean_keywords(tokens)

    mode = "specific"
    if any(word in question_lower for word in NEXT_WORDS):
        mode = "next"
    elif any(word in question_lower for word in PREVIOUS_WORDS):
        mode = "previous"

    has_duration_intent = any(word in question_lower.replace(" ", "") for word in DURATION_WORDS)

    events = _CACHE.get(client)
    if not events:
        return None

    date_refs = infer_date_references(question, dt.datetime.combine(reference_date, dt.time()))
    explicit_dates = [ref.target_date for ref in date_refs if ref.target_date]

    if explicit_dates and not keywords:
        target_date = explicit_dates[0]
        for event in events:
            if event.start <= target_date <= event.end:
                return CalendarAnswer(
                    event_id=event.id,
                    title=event.title,
                    start=event.start,
                    end=event.end,
                    description=event.description,
                    audience=event.audience,
                    source=event.source,
                    mode="date_lookup",
                    score=1.0,
                )

    canonical_keywords = keywords.copy()
    if not canonical_keywords and "break" in question_lower:
        canonical_keywords = ["break"]
    if not canonical_keywords and "holiday" in question_lower:
        canonical_keywords = ["break"]

    def _format_result(event: CalendarEventRecord, score: float, detected_mode: str) -> CalendarAnswer:
        description = event.description
        if has_duration_intent and not description:
            duration_days = event.duration_days
            description = f"Duration: {duration_days} day{'s' if duration_days != 1 else ''}."
        return CalendarAnswer(
            event_id=event.id,
            title=event.title,
            start=event.start,
            end=event.end,
            description=description,
            audience=event.audience,
            source=event.source,
            mode=detected_mode,
            score=score,
        )

    if mode == "next":
        future_events = [event for event in events if event.start >= reference_date]
        candidates = _filter_by_keywords(future_events, canonical_keywords)
        if candidates:
            best = min(candidates, key=lambda ev: (ev.start, ev.end, ev.title))
            score = _score_event(best, canonical_keywords)
            return _format_result(best, score, "next")

    if mode == "previous":
        past_events = [event for event in events if event.end <= reference_date]
        past_events.sort(key=lambda ev: (ev.end, ev.start), reverse=True)
        candidates = _filter_by_keywords(past_events, canonical_keywords)
        if candidates:
            best = candidates[0]
            score = _score_event(best, canonical_keywords)
            return _format_result(best, score, "previous")

    if canonical_keywords:
        best_event = None
        best_score = 0.0
        for event in events:
            score = _score_event(event, canonical_keywords)
            if explicit_dates:
                for target_date in explicit_dates:
                    if event.start <= target_date <= event.end:
                        score += 30
            if score > best_score:
                best_event = event
                best_score = score
        if best_event and best_score >= 45:
            return _format_result(best_event, best_score, "specific")

        try:
            response = client.rpc(
                "match_calendar_events_fuzzy",
                {"q": " ".join(canonical_keywords), "limit_count": 1},
            ).execute()
        except Exception:
            response = None
        if response and response.data:
            row = response.data[0]
            start_raw = row.get("event_date")
            end_raw = row.get("end_date") or start_raw
            try:
                start_date = dt.date.fromisoformat(start_raw) if start_raw else None
            except Exception:
                start_date = None
            try:
                end_date = dt.date.fromisoformat(end_raw) if end_raw else start_date
            except Exception:
                end_date = start_date
            if start_date:
                return CalendarAnswer(
                    event_id=row.get("id") or row.get("title") or "calendar",
                    title=row.get("title") or "Calendar event",
                    start=start_date,
                    end=end_date or start_date,
                    description=(row.get("description") or "").strip(),
                    audience=tuple(row.get("audience") or []),
                    source=row.get("source"),
                    mode="specific",
                    score=float(row.get("similarity") or 0.0),
                )

    if explicit_dates:
        target_date = explicit_dates[0]
        for event in events:
            if event.start <= target_date <= event.end:
                return _format_result(event, 0.0, "date_lookup")

    return None
