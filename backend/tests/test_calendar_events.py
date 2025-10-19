from __future__ import annotations

from backend.app.calendar_events import fetch_calendar_context


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, data):
        self._data = data

    def select(self, *_args, **_kwargs):
        return self

    def ilike(self, column: str, pattern: str):
        token = pattern.strip("% ").lower()
        filtered = [row for row in self._data if token in row[column].lower()]
        return FakeQuery(filtered)

    def execute(self):
        return FakeResponse(self._data)


class FakeClient:
    def __init__(self, rows):
        self._rows = rows

    def table(self, _name):
        return FakeQuery(self._rows)


def test_fetch_calendar_context_groups_break_range():
    rows = [
        {
            "title": "Diwali Break Begins",
            "event_date": "2025-10-16",
            "end_date": None,
            "audience": ["whole_school"],
            "source": "calendar.pdf",
            "description": "Diwali Break Begins",
        },
        {
            "title": "Diwali Break Ends",
            "event_date": "2025-10-26",
            "end_date": None,
            "audience": ["whole_school"],
            "source": "calendar.pdf",
            "description": "Diwali Break Ends",
        },
    ]

    client = FakeClient(rows)
    summaries = fetch_calendar_context(client, "When is the Diwali break?", "Grade 3")

    assert len(summaries) == 1
    summary = summaries[0]["summary"]
    assert "2025-10-16" in summary and "2025-10-26" in summary
