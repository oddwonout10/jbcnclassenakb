from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app import qa_routes


def test_collect_schedule_terms_detects_uniform():
    terms = qa_routes._collect_schedule_terms("what uniform should we wear tomorrow?")
    assert "uniform" in terms


def test_infer_contact_role_transport():
    role = qa_routes._infer_contact_role("who do i contact for bus drop-off?".lower())
    assert role == "transport"


def test_detect_quick_link_type_cafeteria():
    link_type = qa_routes._detect_quick_link_type("Can you send the cafeteria menu please?".lower())
    assert link_type == "cafeteria_menu"


def test_answer_contains_explicit_date():
    assert qa_routes._answer_contains_explicit_date("Event is on 12 Oct 2025 at 10:00 am")
    assert not qa_routes._answer_contains_explicit_date("Schedule to be announced soon")


def test_extract_structured_keywords_filters_stopwords():
    keywords = qa_routes._extract_structured_keywords("When does Diwali break end?")
    assert "diwali" in keywords
    assert "when" not in keywords


def test_detect_date_intent_end():
    intent = qa_routes._detect_date_intent("when does term end?".lower())
    assert intent == "end"


class _DummyClient:
    def __init__(self, action_rows=None, date_rows=None):
        self._action_rows = action_rows or []
        self._date_rows = date_rows or []
        self._active_table = None
        self._grade_filter = None
        self._or_filters = None

    # Supabase-style query builder surface ---------------------------------
    def table(self, name):
        self._active_table = name
        return self

    def select(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def contains(self, column, value):
        self._grade_filter = (column, tuple(value))
        return self

    def or_(self, filters):
        self._or_filters = filters
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        if self._active_table == "document_actions":
            return SimpleNamespace(data=self._action_rows)
        if self._active_table == "document_dates":
            return SimpleNamespace(data=self._date_rows)
        return SimpleNamespace(data=[])


def test_structured_event_lookup_uses_action_rows(monkeypatch):
    rows = [
        {
            "document_id": "doc-123",
            "description": "Grade 3 Educational Field Trip to Nashik",
            "confidence": 0.82,
            "due_date": "2025-11-18",
            "documents": {
                "title": "C 165 (2025-26) - Educational Field Trip (1)",
                "published_on": "2025-09-20",
                "storage_path": "documents/c165.pdf",
                "original_filename": "c165.pdf",
                "grade_tags": ["Grade 3"],
            },
        }
    ]
    client = _DummyClient(action_rows=rows)
    monkeypatch.setattr(qa_routes, "_create_signed_url", lambda *_args, **_kwargs: None)

    result = qa_routes._structured_event_lookup(
        client=client,
        keywords=["field", "trip"],
        grade="Grade 3",
    )

    assert result is not None
    answer, sources, tag = result
    assert "field" in answer.lower()
    assert tag == "structured-event"
    assert sources and sources[0].document_id == "doc-123"


def test_structured_event_lookup_falls_back_to_dates(monkeypatch):
    rows = [
        {
            "document_id": "doc-777",
            "date_type": "start",
            "date_value": "2025-12-10",
            "raw_text": "Field trip starts 10 December 2025",
            "confidence": 0.7,
            "documents": {
                "title": "C 200 Grade 3 Field Trip",
                "published_on": "2025-10-01",
                "storage_path": "documents/c200.pdf",
                "original_filename": "c200.pdf",
                "grade_tags": ["Grade 3"],
            },
        }
    ]
    client = _DummyClient(date_rows=rows)
    monkeypatch.setattr(qa_routes, "_create_signed_url", lambda *_args, **_kwargs: None)

    result = qa_routes._structured_event_lookup(
        client=client,
        keywords=["field", "trip"],
        grade="Grade 3",
    )

    assert result is not None
    answer, sources, tag = result
    assert "field" in answer.lower()
    assert "10" in answer
    assert tag == "structured-event"
    assert sources and sources[0].document_id == "doc-777"
