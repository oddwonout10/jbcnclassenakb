from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Iterable, Optional

DATE_PATTERNS = [
    re.compile(
        r"(?P<day>\d{1,2})(st|nd|rd|th)?\s+(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+(?P<year>\d{4})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+(?P<day>\d{1,2})(st|nd|rd|th)?\s*,?\s*(?P<year>\d{4})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<day>\d{1,2})[-/](?P<month>\d{1,2})[-/](?P<year>\d{4})",
    ),
    re.compile(r"(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})"),
]

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
}


def infer_date_from_text(text: str) -> Optional[dt.date]:
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groupdict()
            try:
                year = int(groups["year"])
                month_raw = groups["month"]
                if month_raw.isdigit():
                    month = int(month_raw)
                else:
                    month = MONTH_LOOKUP[month_raw.lower()]
                day = int(groups["day"])
                return dt.date(year, month, day)
            except Exception:
                continue
    return None


def extract_all_dates(text: str) -> list[tuple[str, dt.date]]:
    """Return a list of (matched_text, date) tuples for every recognised date string."""
    results: list[tuple[str, dt.date]] = []
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(0)
            groups = match.groupdict()
            try:
                year = int(groups["year"])
                month_raw = groups["month"]
                if month_raw.isdigit():
                    month = int(month_raw)
                else:
                    month = MONTH_LOOKUP[month_raw.lower()]
                day = int(groups["day"])
                candidate = dt.date(year, month, day)
            except Exception:
                continue
            results.append((raw, candidate))
    return results


def infer_date_from_filename(path: Path) -> Optional[dt.date]:
    name = path.stem
    # simple YYYYMMDD pattern
    match = re.search(r"(20\d{2})(\d{2})(\d{2})", name)
    if match:
        year, month, day = match.groups()
        try:
            return dt.date(int(year), int(month), int(day))
        except ValueError:
            pass

    # look for year references e.g. 2025-26 -> assume first year
    match = re.search(r"(20\d{2})[^\d]+(20\d{2})", name)
    if match:
        year = int(match.group(1))
        return dt.date(year, 1, 1)
    return None


def determine_document_date(
    *, text: str, path: Path, default: Optional[dt.date] = None
) -> dt.date:
    candidates: list[dt.date] = []

    name_date = infer_date_from_filename(path)
    if name_date:
        candidates.append(name_date)

    text_date = infer_date_from_text(text)
    if text_date:
        candidates.append(text_date)

    if candidates:
        return max(candidates)

    if default:
        return default

    return dt.date.today()
