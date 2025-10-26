from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Set
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

HOLIDAY_KEYWORDS = {
    "holiday",
    "break",
    "vacation",
    "school closed",
    "no school",
    "closure",
}

TITLE_NORMALISE_REGEX = re.compile(r"\b(begins?|starts?|ends?|finishes|reopens|resume?s)\b", re.IGNORECASE)

PRE_PRIMARY_TOKENS = {
    "pre-primary",
    "pre primary",
    "jr kg",
    "junior kg",
    "sr kg",
    "senior kg",
    "kg",
    "kindergarten",
    "nursery",
}

SECONDARY_TOKENS = {
    "secondary",
    "grade 6",
    "grade 7",
    "grade 8",
    "grade 9",
    "grade 10",
    "grade 11",
    "grade 12",
}


def _normalise_grade_band(grade: Optional[str]) -> str:
    if not grade:
        return "primary"
    lowered = grade.lower()
    if any(token in lowered for token in PRE_PRIMARY_TOKENS):
        return "pre_primary"
    if any(token in lowered for token in SECONDARY_TOKENS):
        return "secondary"
    if "primary" in lowered:
        return "primary"
    if re.search(r"\bgrade\s*[1-5]\b", lowered):
        return "primary"
    return "primary"


def _allowed_audience_for_grade(grade: Optional[str]) -> Set[str]:
    band = _normalise_grade_band(grade)
    allowed: Set[str] = {"general", "whole_school", "whole_school_holiday"}
    if band == "pre_primary":
        allowed.update({"pre_primary"})
    elif band == "secondary":
        allowed.update({"secondary", "primary_secondary"})
    else:
        allowed.update({"primary", "primary_secondary"})
    return allowed


def audience_applies_to_grade(audience: Iterable[str], grade: Optional[str]) -> bool:
    audience_set = {entry.lower() for entry in audience if entry}
    if not audience_set:
        return True
    allowed = _allowed_audience_for_grade(grade)
    return bool(audience_set & allowed)


def normalise_event_title(title: str) -> str:
    base = TITLE_NORMALISE_REGEX.sub("", title or "").lower()
    base = re.sub(r"\s+", " ", base).strip()
    return base or (title or "").lower()


def prettify_event_title(title: str) -> str:
    cleaned = normalise_event_title(title)
    if not cleaned:
        return title or "Upcoming Break"
    return " ".join(word.capitalize() for word in cleaned.split())


@dataclass
class DateReference:
    phrase: str
    target_date: Optional[dt.date] = None
    range_start: Optional[dt.date] = None
    range_end: Optional[dt.date] = None
    description: Optional[str] = None


def current_ist() -> dt.datetime:
    return dt.datetime.now(IST)


def infer_date_references(question: str, now: dt.datetime) -> list[DateReference]:
    q = question.lower()
    references: list[DateReference] = []

    def add_ref(phrase: str, days_offset: int, description: Optional[str] = None):
        references.append(
            DateReference(
                phrase=phrase,
                target_date=(now + dt.timedelta(days=days_offset)).date(),
                description=description or phrase,
            )
        )

    if "day after tomorrow" in q:
        add_ref("day after tomorrow", 2, "Day after tomorrow")
    if "tomorrow" in q and "day after tomorrow" not in q:
        add_ref("tomorrow", 1, "Tomorrow")
    if "today" in q:
        add_ref("today", 0, "Today")
    if "yesterday" in q:
        add_ref("yesterday", -1, "Yesterday")
    if "day before yesterday" in q:
        add_ref("day before yesterday", -2, "Day before yesterday")

    if "next week" in q:
        # Define next week as the next Monday through Sunday window
        days_ahead = (7 - now.weekday()) % 7 or 7
        start = (now + dt.timedelta(days=days_ahead)).date()
        end = start + dt.timedelta(days=6)
        references.append(
            DateReference(
                phrase="next week",
                target_date=None,
                range_start=start,
                range_end=end,
                description="Next week",
            )
        )

    # Attempt to parse explicit dates like "14 November" or "14 Nov 2025"
    date_pattern = re.compile(
        r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+(?P<year>\d{2,4}))?\b",
        re.IGNORECASE,
    )
    for match in date_pattern.finditer(q):
        day = int(match.group("day"))
        month = MONTH_LOOKUP[match.group("month").lower()]
        year_str = match.group("year")
        current_year = now.year
        year = current_year
        if year_str:
            year = int(year_str)
            if year < 100:
                year += 2000
        else:
            candidate = dt.date(year, month, day)
            if candidate < now.date():
                year += 1
        try:
            target = dt.date(year, month, day)
            references.append(
                DateReference(
                    phrase=match.group(0),
                    target_date=target,
                    description=match.group(0),
                )
            )
        except ValueError:
            continue

    weekday_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
        "mon": 0,
        "tue": 1,
        "tues": 1,
        "wed": 2,
        "thu": 3,
        "thur": 3,
        "thurs": 3,
        "fri": 4,
        "sat": 5,
        "sun": 6,
    }
    dow_pattern = re.compile(
        r"\b(?:(next|coming|this|last)\s+)?(mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:rs|rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b",
        re.IGNORECASE,
    )
    for match in dow_pattern.finditer(q):
        modifier = (match.group(1) or "").lower()
        day_word = match.group(2).lower()
        target_weekday = weekday_map.get(day_word)
        if target_weekday is None:
            continue

        if modifier in {"next", "coming"}:
            days_ahead = (target_weekday - now.weekday() + 7) % 7
            if days_ahead == 0:
                days_ahead = 7
            target_date = (now + dt.timedelta(days=days_ahead)).date()
            description = f"Next {day_word.capitalize()}"
        elif modifier == "this":
            days_ahead = target_weekday - now.weekday()
            if days_ahead < 0:
                days_ahead += 7
            target_date = (now + dt.timedelta(days=days_ahead)).date()
            description = f"This {day_word.capitalize()}"
        elif modifier == "last":
            days_back = (now.weekday() - target_weekday + 7) % 7
            if days_back == 0:
                days_back = 7
            target_date = (now - dt.timedelta(days=days_back)).date()
            description = f"Last {day_word.capitalize()}"
        else:
            # plain "Monday" without modifier -> upcoming occurrence
            days_ahead = (target_weekday - now.weekday() + 7) % 7
            if days_ahead == 0:
                days_ahead = 7
            target_date = (now + dt.timedelta(days=days_ahead)).date()
            description = day_word.capitalize()

        references.append(
            DateReference(
                phrase=match.group(0),
                target_date=target_date,
                description=description,
            )
        )

    relative_pattern = re.compile(r"\b(in|after)\s+(\d{1,2})\s+day(s)?\b")
    for match in relative_pattern.finditer(q):
        count = int(match.group(2))
        target_date = (now + dt.timedelta(days=count)).date()
        references.append(
            DateReference(
                phrase=match.group(0),
                target_date=target_date,
                description=f"In {count} days",
            )
        )

    past_pattern = re.compile(r"\b(\d{1,2})\s+day(s)?\s+ago\b")
    for match in past_pattern.finditer(q):
        count = int(match.group(1))
        target_date = (now - dt.timedelta(days=count)).date()
        references.append(
            DateReference(
                phrase=match.group(0),
                target_date=target_date,
                description=f"{count} days ago",
            )
        )

    return references


def fetch_calendar_events_for_window(
    client,
    center: dt.date,
    window_days: int = 7,
    grade: Optional[str] = None,
) -> list[dict]:
    start = (center - dt.timedelta(days=window_days)).isoformat()
    end = (center + dt.timedelta(days=window_days)).isoformat()
    response = (
        client.table("calendar_events")
        .select("id,title,event_date,end_date,audience,description,source")
        .gte("event_date", start)
        .lte("event_date", end)
        .order("event_date")
        .execute()
    )
    rows = response.data or []
    if not rows:
        return []

    allowed = _allowed_audience_for_grade(grade)
    filtered: list[dict] = []
    for row in rows:
        audience = {entry.lower() for entry in row.get("audience") or [] if entry}
        if audience and not (audience & allowed):
            continue
        filtered.append(row)
    return filtered


def parse_date(value: Optional[str]) -> Optional[dt.date]:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def event_covers_date(event: dict, target: dt.date) -> bool:
    start = parse_date(event.get("event_date"))
    end = parse_date(event.get("end_date")) or start
    if start is None:
        return False
    return start <= target <= end


def format_date(value: dt.date) -> str:
    return value.strftime("%d %b %Y")


def format_date_range(start: dt.date, end: Optional[dt.date]) -> str:
    if not end or end == start:
        return format_date(start)
    return f"{format_date(start)} – {format_date(end)}"


DATE_TEXT_PATTERN = re.compile(
    r"\b(\d{1,2})\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*(\d{2,4})?\b",
    re.IGNORECASE,
)
ISO_DATE_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def parse_date_string(text: str, default_year: int) -> Optional[dt.date]:
    match = DATE_TEXT_PATTERN.search(text)
    if not match:
        return None
    day = int(match.group(1))
    month = MONTH_LOOKUP.get(match.group(2).lower())
    if not month:
        return None
    year_part = match.group(3)
    year = default_year
    if year_part:
        year = int(year_part)
        if year < 100:
            year += 2000
    try:
        candidate = dt.date(year, month, day)
    except ValueError:
        return None
    return candidate


def parse_date_range_from_text(text: str, reference_year: int) -> tuple[Optional[dt.date], Optional[dt.date]]:
    matches = list(DATE_TEXT_PATTERN.finditer(text))
    iso_matches = list(ISO_DATE_PATTERN.finditer(text))
    if not matches and not iso_matches:
        return (None, None)
    results: list[dt.date] = []
    for match in matches:
        day = int(match.group(1))
        month = MONTH_LOOKUP.get(match.group(2).lower())
        if not month:
            continue
        year_part = match.group(3)
        year = reference_year
        if year_part:
            year = int(year_part)
            if year < 100:
                year += 2000
        try:
            results.append(dt.date(year, month, day))
        except ValueError:
            continue
    for match in iso_matches:
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        try:
            results.append(dt.date(year, month, day))
        except ValueError:
            continue
    if not results:
        return (None, None)
    results.sort()
    if len(results) == 1:
        return (results[0], results[0])
    return (results[0], results[-1])


def is_probable_holiday(event: dict) -> bool:
    title = (event.get("title") or "").lower()
    description = (event.get("description") or "").lower()
    return any(keyword in title or keyword in description for keyword in HOLIDAY_KEYWORDS)


def format_event_summary(event: dict) -> str:
    start = parse_date(event.get("event_date"))
    end = parse_date(event.get("end_date")) or start
    if not start:
        return event.get("title") or "Calendar event"
    return f"{event.get('title', 'Calendar event')} ({format_date_range(start, end)})"


def determine_holiday_answer(
    reference: DateReference,
    events: Iterable[dict],
) -> Optional[dict]:
    if not reference.target_date:
        return None

    aggregated: dict[str, dict] = {}

    for event in events:
        base = normalise_event_title(event.get("title", ""))
        summary_text = event.get("summary") or event.get("description") or ""
        start = parse_date(event.get("event_date"))
        end = parse_date(event.get("end_date"))
        if not start:
            start, end = parse_date_range_from_text(summary_text, reference.target_date.year)
        if not start:
            continue
        end = end or start

        agg = aggregated.setdefault(
            base,
            {
                "title": event.get("title") or base.title(),
                "source": event.get("source"),
                "summary": summary_text,
                "start": start,
                "end": end,
                "holiday": is_probable_holiday(event) or any(
                    keyword in summary_text for keyword in HOLIDAY_KEYWORDS
                ),
            },
        )

        if start < agg["start"]:
            agg["start"] = start
        if end > agg["end"]:
            agg["end"] = end
        if not agg["holiday"]:
            agg["holiday"] = is_probable_holiday(event) or any(
                keyword in summary_text for keyword in HOLIDAY_KEYWORDS
            )

    for agg_event in aggregated.values():
        if not agg_event["holiday"]:
            continue
        if agg_event["start"] <= reference.target_date <= agg_event["end"]:
            resume_date = agg_event["end"] + dt.timedelta(days=1)
            return {
                "answer": (
                    f"Yes, {reference.description or 'the requested date'} "
                    f"({format_date(reference.target_date)}) falls during {agg_event['title']} "
                    f"({format_date_range(agg_event['start'], agg_event['end'])}). "
                    f"Classes resume on {format_date(resume_date)}."
                ),
                "event": {
                    "id": agg_event["title"],
                    "title": agg_event["title"],
                    "source": agg_event["source"],
                },
                "start": agg_event["start"],
                "end": agg_event["end"],
                "resume": resume_date,
            }
    return None


def upcoming_holiday_event(
    client,
    after_date: dt.date,
    lookahead_days: int = 180,
    grade: Optional[str] = None,
) -> dict[str, Optional[dict]]:
    window_end = (after_date + dt.timedelta(days=lookahead_days)).isoformat()
    response = (
        client.table("calendar_events")
        .select("id,title,event_date,end_date,audience,description,source")
        .gte("event_date", after_date.isoformat())
        .lte("event_date", window_end)
        .order("event_date")
        .execute()
    )
    rows = response.data or []
    aggregated: dict[str, dict] = {}
    allowed = _allowed_audience_for_grade(grade)

    for row in rows:
        audience = {entry.lower() for entry in row.get("audience") or [] if entry}
        if audience and not (audience & allowed):
            continue

        summary_text = (row.get("description") or "").lower()
        if not any(keyword in summary_text for keyword in HOLIDAY_KEYWORDS):
            title_lower = (row.get("title") or "").lower()
            if not any(keyword in title_lower for keyword in HOLIDAY_KEYWORDS):
                continue
        base = normalise_event_title(row.get("title", ""))
        start = parse_date(row.get("event_date"))
        end = parse_date(row.get("end_date")) or start
        if not start:
            continue
        agg = aggregated.setdefault(
            base,
            {
                "id": row.get("id") or row.get("title") or base,
                "title": prettify_event_title(row.get("title")),
                "source": row.get("source"),
                "start": start,
                "end": end,
            },
        )
        if start < agg["start"]:
            agg["start"] = start
        if end > agg["end"]:
            agg["end"] = end

    if not aggregated:
        return {"current": None, "next": None}

    ordered = sorted(aggregated.values(), key=lambda item: item["start"])
    current_event: Optional[dict] = None
    next_event: Optional[dict] = None

    for event in ordered:
        if event["start"] <= after_date <= event["end"]:
            current_event = event
        elif event["start"] > after_date and next_event is None:
            next_event = event

    result: dict[str, Optional[dict]] = {"current": None, "next": None}

    if current_event:
        result["current"] = {
            **current_event,
            "resume": current_event["end"] + dt.timedelta(days=1),
        }
    if next_event:
        result["next"] = {
            **next_event,
            "resume": next_event["end"] + dt.timedelta(days=1),
        }

    return result


def upcoming_break_from_matches(matches: Iterable[dict], reference_date: dt.date) -> dict[str, Optional[dict]]:
    current: Optional[dict] = None
    future: Optional[dict] = None
    best_future_start: Optional[dt.date] = None

    for match in matches:
        start = parse_date(match.get("event_date"))
        end = parse_date(match.get("end_date")) or start
        if not start:
            continue
        if start <= reference_date <= end:
            current = {
                "id": match.get("title"),
                "title": match.get("title"),
                "source": match.get("source"),
                "start": start,
                "end": end,
                "resume": end + dt.timedelta(days=1),
            }
        elif start > reference_date and (best_future_start is None or start < best_future_start):
            best_future_start = start
            future = {
                "id": match.get("title"),
                "title": match.get("title"),
                "source": match.get("source"),
                "start": start,
                "end": end,
                "resume": end + dt.timedelta(days=1),
            }

    return {"current": current, "next": future}


def build_reference_hint(ref: DateReference) -> str:
    if ref.target_date:
        return (
            f'The phrase "{ref.phrase}" refers to {format_date(ref.target_date)} (IST).'
        )
    if ref.range_start and ref.range_end:
        return (
            f'The phrase "{ref.phrase}" refers to the period '
            f"{format_date(ref.range_start)} to {format_date(ref.range_end)} (IST)."
        )
    return f'Consider the phrase "{ref.phrase}".'
MONTH_LOOKUP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
