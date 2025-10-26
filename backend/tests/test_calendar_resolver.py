from __future__ import annotations

from pathlib import Path
import sys
import datetime as dt
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.calendar_resolver import resolve_calendar_question, reset_cache


class DummyCalendarClient:
    def __init__(self, rows, rpc_rows=None):
        self._rows = rows
        self._rpc_rows = rpc_rows
        self._mode = "table"

    def table(self, name):
        assert name == "calendar_events"
        self._mode = "table"
        return self

    def select(self, *_args, **_kwargs):
        return self

    def execute(self):
        if self._mode == "rpc":
            return SimpleNamespace(data=self._rpc_rows)
        return SimpleNamespace(data=self._rows)

    def rpc(self, name, params):
        assert name == "match_calendar_events_fuzzy"
        self._mode = "rpc"
        return self


def setup_function() -> None:
    reset_cache()


def test_specific_calendar_event_matches_synonyms():
    rows = [
        {
            "id": "evt-diwali",
            "title": "Diwali Break",
            "event_date": "2025-10-16",
            "end_date": "2025-10-24",
            "audience": ["whole_school"],
            "description": "Diwali holidays from 16 Oct 2025 to 24 Oct 2025",
            "source": "calendar.pdf",
        },
        {
            "id": "evt-christmas",
            "title": "Christmas Break",
            "event_date": "2025-12-22",
            "end_date": "2026-01-02",
            "audience": ["whole_school"],
            "description": "Winter holidays from 22 Dec 2025 to 2 Jan 2026",
            "source": "calendar.pdf",
        },
    ]
    client = DummyCalendarClient(rows)
    answer = resolve_calendar_question(
        client=client,
        question="When is the Xmas break?",
        reference_date=dt.date(2025, 9, 1),
        grade="Grade 3",
    )

    assert answer is not None
    assert answer.title == "Christmas Break"
    assert answer.start == dt.date(2025, 12, 22)
    assert answer.end == dt.date(2026, 1, 2)


def test_next_break_prefers_future_event():
    rows = [
        {
            "id": "evt-past",
            "title": "Sports Day",
            "event_date": "2025-08-10",
            "end_date": "2025-08-10",
            "audience": ["primary"],
            "description": "Sports day held on 10 August 2025",
            "source": "calendar.pdf",
        },
        {
            "id": "evt-future",
            "title": "Winter Concert",
            "event_date": "2025-12-15",
            "end_date": "2025-12-15",
            "audience": ["whole_school"],
            "description": "Concert on 15 December 2025",
            "source": "calendar.pdf",
        },
    ]
    client = DummyCalendarClient(rows)
    answer = resolve_calendar_question(
        client=client,
        question="When is the next concert?",
        reference_date=dt.date(2025, 9, 1),
        grade="Grade 3",
    )

    assert answer is not None
    assert answer.title == "Winter Concert"
    assert answer.start == dt.date(2025, 12, 15)


def test_break_question_prefers_break_event():
    rows = [
        {
            "id": "evt-party",
            "title": "Christmas Party",
            "event_date": "2025-12-16",
            "end_date": None,
            "audience": ["pre_primary"],
            "description": "Christmas Party celebration",
            "source": "calendar.pdf",
        },
        {
            "id": "evt-break-begins",
            "title": "Christmas break begins",
            "event_date": "2025-12-22",
            "end_date": None,
            "audience": ["whole_school_holiday"],
            "description": "Christmas break begins",
            "source": "calendar.pdf",
        },
        {
            "id": "evt-break-ends",
            "title": "Christmas break ends",
            "event_date": "2026-01-02",
            "end_date": None,
            "audience": ["whole_school_holiday"],
            "description": "Christmas break ends",
            "source": "calendar.pdf",
        },
    ]
    client = DummyCalendarClient(rows)
    answer = resolve_calendar_question(
        client=client,
        question="When is Christmas break?",
        reference_date=dt.date(2025, 11, 1),
        grade="Grade 3",
    )

    assert answer is not None
    assert "break" in answer.title.lower()
    assert answer.start == dt.date(2025, 12, 22)


def test_irrelevant_calendar_match_is_ignored():
    rows = [
        {
            "id": "evt-holiday",
            "title": "Ganesh Chaturthi",
            "event_date": "2025-08-27",
            "end_date": "2025-08-27",
            "audience": ["whole_school_holiday"],
            "description": "Ganesh Chaturthi holiday",
            "source": "calendar.pdf",
        }
    ]
    rpc_rows = [
        {
            "id": "evt-holiday",
            "title": "Ganesh Chaturthi",
            "event_date": "2025-08-27",
            "end_date": "2025-08-27",
            "audience": ["whole_school_holiday"],
            "description": "Ganesh Chaturthi holiday",
            "source": "calendar.pdf",
            "similarity": 0.12,
        }
    ]
    client = DummyCalendarClient(rows, rpc_rows=rpc_rows)
    answer = resolve_calendar_question(
        client=client,
        question="Tell me more about Nashik trip",
        reference_date=dt.date(2025, 8, 1),
        grade="Grade 3",
    )

    assert answer is None


def test_explicit_date_uses_aggregated_break():
    rows = [
        {
            "id": "evt-break-begins",
            "title": "Christmas break begins",
            "event_date": "2025-12-22",
            "end_date": None,
            "audience": ["whole_school_holiday"],
            "description": "Christmas break begins",
            "source": "calendar.pdf",
        },
        {
            "id": "evt-break-ends",
            "title": "Christmas break ends",
            "event_date": "2026-01-02",
            "end_date": None,
            "audience": ["whole_school_holiday"],
            "description": "Christmas break ends",
            "source": "calendar.pdf",
        },
    ]
    client = DummyCalendarClient(rows)
    answer = resolve_calendar_question(
        client=client,
        question="Is 31st December a holiday?",
        reference_date=dt.date(2025, 12, 1),
        grade="Grade 3",
    )

    assert answer is not None
    assert answer.title.lower() == "christmas break"
    assert answer.start == dt.date(2025, 12, 22)
    assert answer.end == dt.date(2026, 1, 2)
