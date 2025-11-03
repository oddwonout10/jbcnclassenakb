import os
from dataclasses import replace

import pytest

from app.config import Settings
from app.llm_client import generate_answer, LLMClientError


@pytest.fixture
def settings_openai(monkeypatch) -> Settings:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    return Settings(
        supabase_url="https://example.supabase.co",
        supabase_anon_key="anon",
        supabase_service_role_key="service",
        app_env="test",
        cors_allow_origins=("http://localhost:3000",),
        escalation_email_to="test@example.com",
        escalation_email_from="test@example.com",
        smtp_username="test@example.com",
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
        enable_reranker=True,
        reranker_weight=0.35,
        reranker_model="BAAI/bge-reranker-base",
        reranker_max_passages=24,
    )


def test_generate_answer_with_missing_key(settings_openai):
    settings_missing_key = replace(settings_openai, openai_api_key=None)
    with pytest.raises(LLMClientError):
        generate_answer("test", settings_missing_key)
