from __future__ import annotations

import sys
import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.qa_routes as qa_routes
from app.config import Settings
from app.llm_client import LLMClientError
from app.qa_routes import SourceInfo
from app.rag import ChunkHit


def _settings() -> Settings:
    return Settings(
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
        openai_api_key="test",
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
        qa_vector_candidates=12,
        qa_keyword_candidates=12,
        qa_vector_weight=0.7,
        qa_keyword_weight=0.35,
        qa_recency_weight=0.12,
        turnstile_secret_key=None,
        qa_rate_limit_per_minute=60,
        enable_reranker=False,
        reranker_weight=0.0,
        reranker_model="BAAI/bge-reranker-base",
        reranker_max_passages=24,
    )


def test_sanitize_highlight_cleans_table_noise():
    raw = "Table: | OUTSTATION | EDUTRIP | GRADE - 3 | OUTSTATION | 14th- 16th January 2026 |"
    cleaned = qa_routes._sanitize_highlight(raw)
    assert cleaned is not None
    assert "Table" not in cleaned
    assert "|" not in cleaned
    # Ensure duplicate words are collapsed.
    assert cleaned.count("OUTSTATION") == 1


def test_append_circular_suggestions_dedupes_duplicates():
    existing = [
        SourceInfo(
            document_id="doc-1",
            title="Existing Doc",
            published_on="2025-01-01",
            original_filename="existing.pdf",
            signed_url=None,
            storage_path="existing.pdf",
            similarity=1.0,
        )
    ]
    suggestions = [
        SourceInfo(
            document_id="doc-2",
            title="Trip Details",
            published_on="2025-01-15",
            original_filename="trip.pdf",
            signed_url=None,
            storage_path="trip.pdf",
            similarity=0.9,
        ),
        SourceInfo(
            document_id="doc-2",
            title="Trip Details",
            published_on="2025-01-15",
            original_filename="trip.pdf",
            signed_url=None,
            storage_path="trip.pdf",
            similarity=0.9,
        ),
    ]

    parts: list[str] = []
    qa_routes._append_circular_suggestions(parts, existing, suggestions, limit=2)

    assert parts and parts[0].startswith("Circulars:")
    assert parts[0].count("Trip Details") == 1
    # Ensure only one suggestion added to sources.
    assert len(existing) == 2


def test_is_table_only_chunk_detects_table():
    table_chunk = "Table: | COLUMN A | COLUMN B |\nTable: 1 | 2"
    text_chunk = "Learners depart for Nashik at 6:30 AM on 14 January 2026 for the Grade 3 trip."

    assert qa_routes._is_table_only_chunk(table_chunk) is True
    assert qa_routes._is_table_only_chunk(text_chunk) is False


def test_fetch_chunks_ignores_duplicate_tables(monkeypatch):
    rows = [
        {
            "chunk_index": 0,
            "content": "Table: | OUTSTATION | EDUTRIP |",
            "published_on": "2026-01-16",
            "page_number": 1,
            "section_heading": "OUTSTATION",
        },
        {
            "chunk_index": 1,
            "content": "Learners depart 14 January 2026 at 6:30 AM from school.",
            "published_on": "2026-01-16",
            "page_number": 1,
            "section_heading": "Schedule",
        },
        {
            "chunk_index": 2,
            "content": "Return on 16 January 2026 by 6:00 PM.",
            "published_on": "2026-01-16",
            "page_number": 2,
            "section_heading": "Closing",
        },
    ]

    class DummyQuery:
        def __init__(self, rows):
            self.rows = rows
            self._limit = len(rows)

        def select(self, *_args, **_kwargs):
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, value):
            self._limit = value
            return self

        def execute(self):
            return SimpleNamespace(data=self.rows[: self._limit])

    class DummyClient:
        def __init__(self, rows):
            self.rows = rows

        def table(self, name):
            assert name == "document_chunks"
            return DummyQuery(self.rows)

    client = DummyClient(rows)
    source = SourceInfo(
        document_id="doc-nashik",
        title="Parent Orientation JBCN Parel G3 Nashik",
        published_on="2026-01-16",
        original_filename="nashik.pdf",
        signed_url=None,
        storage_path="nashik.pdf",
        similarity=0.95,
    )

    chunks = qa_routes._fetch_chunks_for_document(client, source, limit=2)

    assert len(chunks) == 2
    assert chunks[0].content.startswith("Learners depart 14 January 2026")
    assert chunks[1].content.startswith("Return on 16 January 2026")


def test_llm_summary_for_document_uses_cache(monkeypatch):
    qa_routes.SUMMARY_CACHE.clear()

    chunk = ChunkHit(
        document_id="doc-99",
        chunk_index=0,
        content="The Nashik trip runs from 14 January 2026 to 16 January 2026. Arrive by 6:30 AM.",
        similarity=0.9,
        document_title="Trip Plan",
        original_filename="trip.pdf",
        published_on=dt.date(2026, 1, 10),
        storage_path="trip.pdf",
        page_number=2,
        section_heading="Schedule",
    )

    call_count = 0

    def fake_fetch_chunks(*_args, **_kwargs):
        return [chunk]

    def fake_generate(prompt, _settings):
        nonlocal call_count
        call_count += 1
        assert "Schedule" in prompt
        return ("Trip runs 14-16 January 2026. Learners report by 6:30 AM.", "gpt-4o-mini")

    monkeypatch.setattr(qa_routes, "_fetch_chunks_for_document", fake_fetch_chunks)
    monkeypatch.setattr(qa_routes, "generate_answer", fake_generate)

    primary = SourceInfo(
        document_id="doc-99",
        title="Trip Plan",
        published_on="2026-01-10",
        original_filename="trip.pdf",
        signed_url=None,
        storage_path="trip.pdf",
        similarity=0.95,
    )

    summary_1 = qa_routes._llm_summary_for_document(
        client=None,
        question="When is the Nashik trip?",
        primary_source=primary,
        settings=_settings(),
        max_chunks=2,
    )

    summary_2 = qa_routes._llm_summary_for_document(
        client=None,
        question="When is the Nashik trip?",
        primary_source=primary,
        settings=_settings(),
        max_chunks=2,
    )

    assert summary_1 == summary_2
    assert call_count == 1
    assert summary_1.startswith("Trip runs 14-16 January 2026")


def test_llm_summary_for_document_handles_llm_errors(monkeypatch):
    qa_routes.SUMMARY_CACHE.clear()

    chunk = ChunkHit(
        document_id="doc-error",
        chunk_index=0,
        content="Trip departs on 14 January 2026.",
        similarity=0.9,
        document_title="Trip Plan",
        original_filename="trip.pdf",
        published_on=dt.date(2026, 1, 10),
        storage_path="trip.pdf",
        page_number=1,
        section_heading=None,
    )

    monkeypatch.setattr(qa_routes, "_fetch_chunks_for_document", lambda *_args, **_kwargs: [chunk])

    def failing_generate(*_args, **_kwargs):
        raise LLMClientError("boom")

    monkeypatch.setattr(qa_routes, "generate_answer", failing_generate)

    primary = SourceInfo(
        document_id="doc-error",
        title="Trip Plan",
        published_on="2026-01-10",
        original_filename="trip.pdf",
        signed_url=None,
        storage_path="trip.pdf",
        similarity=0.95,
    )

    summary = qa_routes._llm_summary_for_document(
        client=None,
        question="When is the Nashik trip?",
        primary_source=primary,
        settings=_settings(),
        max_chunks=2,
    )

    assert summary is None
