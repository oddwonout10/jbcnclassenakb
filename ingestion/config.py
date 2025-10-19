from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()


def _env(name: str, *, default: str | None = None, required: bool = True) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    if value is None:
        return ""
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover
        raise RuntimeError(f"Environment variable {name} must be an integer") from exc


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_service_role_key: str
    storage_bucket: str
    ingestion_source_dir: Path
    chunk_size: int
    chunk_overlap: int
    metadata_path: Path | None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    metadata_env = os.getenv("INGESTION_METADATA_PATH")
    metadata_path = Path(metadata_env).resolve() if metadata_env else None

    return Settings(
        supabase_url=_env("SUPABASE_URL"),
        supabase_service_role_key=_env("SUPABASE_SERVICE_ROLE_KEY"),
        storage_bucket=os.getenv("SUPABASE_STORAGE_BUCKET", "class-circulars"),
        ingestion_source_dir=Path(
            os.getenv("INGESTION_SOURCE_DIR", "data/circulars")
        ).resolve(),
        chunk_size=_env_int("INGESTION_CHUNK_SIZE", 800),
        chunk_overlap=_env_int("INGESTION_CHUNK_OVERLAP", 200),
        metadata_path=metadata_path,
    )
