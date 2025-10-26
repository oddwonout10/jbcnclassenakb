from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.calendar_resolver import CalendarAnswer
from app.config import Settings
from app.main import app
import app.qa_routes as qa_routes
from app.rag import ChunkHit


class DummyStorage:
    def from_(self, _bucket):
        return self

    def exists(self, _path):
        return True

    def create_signed_url(self, path: str, *_args, **_kwargs):
        return {"signedURL": f"https://example.com/{path}"}


class DummySupabase:
    def __init__(self):
        self.storage = DummyStorage()

    # Query builder chain support -------------------------------------------------
    def table(self, *_args, **_kwargs):
        return self

    def select(self, *_args, **_kwargs):
        return self

    def contains(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def gte(self, *_args, **_kwargs):
        return self

    def lte(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def ilike(self, *_args, **_kwargs):
        return self

    def or_(self, *_args, **_kwargs):
        return self

    def in_(self, *_args, **_kwargs):
        return self

    def delete(self, *_args, **_kwargs):
        return self

    def insert(self, *_args, **_kwargs):
        return self

    def upsert(self, *_args, **_kwargs):
        return self

    def execute(self):
        return SimpleNamespace(data=[])

    # RPC ------------------------------------------------------------------------
    def rpc(self, *_args, **_kwargs):
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=[]))


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    qa_routes._rate_limiter = None
    yield
    qa_routes._rate_limiter = None


@pytest.fixture
def qa_test_client(monkeypatch) -> TestClient:
    settings = Settings(
        supabase_url="https://example.supabase.co",
        supabase_anon_key="anon",
        supabase_service_role_key="service",
        app_env="test",
        cors_allow_origins=("http://localhost:3000",),
        escalation_email_to="alerts@example.com",
        escalation_email_from="noreply@example.com",
        smtp_username="smtp@example.com",
        smtp_password="password",
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        storage_bucket="class-circulars",
        llm_provider="openai",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        gemini_api_key=None,
        gemini_model="gemini-pro",
        groq_api_key=None,
        groq_model="llama3-70b-8192",
        anthropic_api_key=None,
        anthropic_model="claude-3-haiku-20240307",
        cohere_api_key=None,
        cohere_model="command",
        qa_similarity_threshold=0.72,
        qa_max_chunks=6,
        turnstile_secret_key=None,
        qa_rate_limit_per_minute=60,
    )

    dummy_supabase = DummySupabase()

    monkeypatch.setattr(qa_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(qa_routes, "get_supabase_client", lambda service_role=True: dummy_supabase)
    monkeypatch.setattr(qa_routes, "_log_interaction", lambda **_kwargs: None)
    monkeypatch.setattr(qa_routes, "_send_escalation_email", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(qa_routes, "_create_signed_url", lambda _client, _path: None)
    monkeypatch.setattr(qa_routes, "embed_text", lambda _text: [0.0])

    return TestClient(app)


def _calendar_answer() -> CalendarAnswer:
    return CalendarAnswer(
        event_id="evt-holiday",
        title="Christmas Break",
        start=dt.date(2025, 12, 22),
        end=dt.date(2026, 1, 2),
        description="Winter holidays for the whole school.",
        audience=("whole_school_holiday",),
        source="calendar.pdf",
        mode="specific",
        score=0.95,
    )


def test_answer_question_returns_calendar_when_available(qa_test_client: TestClient, monkeypatch):
    calendar_answer = _calendar_answer()

    monkeypatch.setattr(qa_routes, "fetch_calendar_context", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(qa_routes, "fetch_calendar_events_for_window", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(qa_routes, "upcoming_holiday_event", lambda *_args, **_kwargs: {"current": None, "next": None})
    monkeypatch.setattr(qa_routes, "_fetch_document_fuzzy", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(qa_routes, "resolve_calendar_question", lambda **_kwargs: calendar_answer)
    monkeypatch.setattr(qa_routes, "fetch_relevant_chunks", lambda **_kwargs: pytest.fail("Should not fetch documents when calendar answer exists"))
    monkeypatch.setattr(qa_routes, "generate_answer", lambda *_args, **_kwargs: pytest.fail("LLM should not be called for deterministic calendar answers"))

    response = qa_test_client.post(
        "/qa",
        json={"question": "When is the Christmas break?", "grade": "Grade 3"},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "answered"
    assert "Calendar: Christmas Break" in data["answer"]
    assert any(source["document_id"].startswith("calendar:") for source in data["sources"])


def test_answer_question_uses_circular_llm_flow(qa_test_client: TestClient, monkeypatch):
    monkeypatch.setattr(qa_routes, "fetch_calendar_context", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(qa_routes, "fetch_calendar_events_for_window", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(qa_routes, "upcoming_holiday_event", lambda *_args, **_kwargs: {"current": None, "next": None})
    monkeypatch.setattr(qa_routes, "resolve_calendar_question", lambda **_kwargs: None)
    monkeypatch.setattr(qa_routes, "_fetch_document_fuzzy", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(qa_routes, "_fetch_keyword_hits", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(qa_routes, "_fetch_document_metadata", lambda *_args, **_kwargs: {
        "doc-1": {
            "title": "Uniform Update",
            "published_on": "2025-08-10",
            "grade_tags": ["Grade 3"],
            "event_tags": ["uniform"],
        }
    })

    chunk = ChunkHit(
        document_id="doc-1",
        chunk_index=0,
        content="Learners must wear the sports uniform on Friday.",
        similarity=0.88,
        document_title="Uniform Update",
        original_filename="uniform.pdf",
        published_on=dt.date(2025, 8, 10),
        storage_path="documents/uniform.pdf",
        score=0.88,
    )
    monkeypatch.setattr(qa_routes, "fetch_relevant_chunks", lambda **_kwargs: [chunk])

    captured_prompt: dict[str, str] = {}

    def fake_generate_answer(prompt: str, _settings: Settings):
        captured_prompt["prompt"] = prompt
        return "Circular answer about uniforms.", "mock-llm"

    monkeypatch.setattr(qa_routes, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(
        qa_routes,
        "_group_sources",
        lambda _client, _hits: [
            qa_routes.SourceInfo(
                document_id="doc-1",
                title="Uniform Update",
                published_on="2025-08-10",
                original_filename="uniform.pdf",
                signed_url=None,
                storage_path="documents/uniform.pdf",
                similarity=0.88,
            )
        ],
    )

    response = qa_test_client.post(
        "/qa",
        json={"question": "What should learners wear on Friday?", "grade": "Grade 3"},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "answered"
    assert data["answer"].startswith("Circular answer about uniforms.")
    assert data["sources"][0]["document_id"] == "doc-1"
    assert "Circulars and documents" in captured_prompt["prompt"]


def test_calendar_answer_appends_circular_suggestions(qa_test_client: TestClient, monkeypatch):
    calendar_answer = _calendar_answer()

    suggestion = qa_routes.SourceInfo(
        document_id="doc-2",
        title="Holiday Logistics Circular",
        published_on="2025-12-18",
        original_filename="holiday-logistics.pdf",
        signed_url=None,
        storage_path="documents/holiday-logistics.pdf",
        similarity=0.55,
    )

    monkeypatch.setattr(qa_routes, "fetch_calendar_context", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(qa_routes, "fetch_calendar_events_for_window", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(qa_routes, "upcoming_holiday_event", lambda *_args, **_kwargs: {"current": None, "next": None})
    monkeypatch.setattr(qa_routes, "resolve_calendar_question", lambda **_kwargs: calendar_answer)
    monkeypatch.setattr(qa_routes, "_fetch_document_fuzzy", lambda *_args, **_kwargs: [suggestion])
    monkeypatch.setattr(qa_routes, "fetch_relevant_chunks", lambda **_kwargs: [])
    monkeypatch.setattr(qa_routes, "generate_answer", lambda *_args, **_kwargs: pytest.fail("LLM should not run in calendar shortcut path"))

    response = qa_test_client.post(
        "/qa",
        json={"question": "Tell me about the upcoming break.", "grade": "Grade 3"},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "answered"
    assert "Calendar: Christmas Break" in data["answer"]
    assert "Circulars:\n- Holiday Logistics Circular" in data["answer"]

    doc_ids = {source["document_id"] for source in data["sources"]}
    assert "calendar:evt-holiday" in doc_ids or any(doc.startswith("calendar:") for doc in doc_ids)
    assert "doc-2" in doc_ids
