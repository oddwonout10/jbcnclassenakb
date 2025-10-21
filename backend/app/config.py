from __future__ import annotations
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)  # Load project .env once at import time.
else:
    load_dotenv()  # Fallback: search upwards from CWD.


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Environment variable {name} must be an integer") from exc


def _env_optional(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str

    app_env: str
    cors_allow_origins: tuple[str, ...]

    escalation_email_to: str
    escalation_email_from: str
    smtp_username: str
    smtp_password: str
    smtp_host: str
    smtp_port: int
    storage_bucket: str
    llm_provider: str
    openai_api_key: Optional[str]
    openai_model: str
    gemini_api_key: Optional[str]
    gemini_model: str
    groq_api_key: Optional[str]
    groq_model: str
    anthropic_api_key: Optional[str]
    anthropic_model: str
    cohere_api_key: Optional[str]
    cohere_model: str
    qa_similarity_threshold: float
    qa_max_chunks: int
    turnstile_secret_key: Optional[str]
    qa_rate_limit_per_minute: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    cors_raw = os.getenv("CORS_ALLOW_ORIGINS")
    if cors_raw:
        origins = tuple(origin.strip() for origin in cors_raw.split(",") if origin.strip())
    else:
        origins = (
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        )

    return Settings(
        supabase_url=_env("SUPABASE_URL"),
        supabase_anon_key=_env("SUPABASE_ANON_KEY"),
        supabase_service_role_key=_env("SUPABASE_SERVICE_ROLE_KEY"),
        app_env=os.getenv("APP_ENV", "local"),
        cors_allow_origins=origins,
        escalation_email_to=_env("ESCALATION_EMAIL_TO"),
        escalation_email_from=_env("ESCALATION_EMAIL_FROM"),
        smtp_username=_env("SMTP_USERNAME"),
        smtp_password=_env("SMTP_PASSWORD"),
        smtp_host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
        smtp_port=_env_int("SMTP_PORT", 587),
        storage_bucket=os.getenv("SUPABASE_STORAGE_BUCKET", "class-circulars"),
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        openai_api_key=_env_optional("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        gemini_api_key=_env_optional("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-pro"),
        groq_api_key=_env_optional("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
        anthropic_api_key=_env_optional("ANTHROPIC_API_KEY"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
        cohere_api_key=_env_optional("COHERE_API_KEY"),
        cohere_model=os.getenv("COHERE_MODEL", "command"),
        qa_similarity_threshold=float(os.getenv("QA_SIMILARITY_THRESHOLD", "0.72")),
        qa_max_chunks=_env_int("QA_MAX_CHUNKS", 6),
        turnstile_secret_key=_env_optional("TURNSTILE_SECRET_KEY"),
        qa_rate_limit_per_minute=_env_int("QA_RATE_LIMIT_PER_MINUTE", 60),
    )
