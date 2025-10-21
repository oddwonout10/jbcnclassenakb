from __future__ import annotations

import datetime as dt
import os
import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List

import typer
from tqdm import tqdm

from .config import get_settings
from .supabase_client import get_supabase_client
from .utils.chunks import TextChunk, chunk_text
from .utils.date_parser import determine_document_date
from .utils.text_extraction import extract_text
from .utils.metadata import infer_metadata
from openai import OpenAI


_EMBED_MODEL_NAME = "text-embedding-3-small"
_EMBED_DIMENSION = 768


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY must be set to generate embeddings during ingestion."
        )
    return OpenAI(api_key=api_key)


def load_metadata_overrides() -> dict[str, dict]:
    settings = get_settings()
    if not settings.metadata_path:
        return {}
    path = settings.metadata_path
    if not path.exists():
        LOGGER.warning("Metadata override file %s not found. Continuing without overrides.", path)
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("ingestion")

app = typer.Typer(help="Document ingestion pipeline for circulars.")


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def load_existing_hashes() -> dict[str, dict]:
    client = get_supabase_client()
    response = client.table("documents").select(
        "id, source_sha256, storage_path, published_on"
    ).execute()
    data = response.data or []
    return {row["source_sha256"]: row for row in data}


def sanitize_filename(filename: str) -> str:
    normalized = unicodedata.normalize("NFKD", filename)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.replace(" ", "_")
    ascii_name = re.sub(r"[^A-Za-z0-9._-]", "", ascii_name)
    ascii_name = re.sub(r"_+", "_", ascii_name)
    ascii_name = ascii_name.strip("._") or "document"
    return ascii_name


def upload_file(storage_path: str, local_path: Path) -> None:
    client = get_supabase_client()
    bucket = get_settings().storage_bucket
    with local_path.open("rb") as file:
        try:
            client.storage.from_(bucket).upload(
                storage_path,
                file,
                {"upsert": False},
            )
        except Exception as exc:
            message = str(exc)
            if "Duplicate" in message or "already exists" in message:
                LOGGER.info("File %s already exists in storage. Skipping upload.", storage_path)
                return
            raise


def insert_document_record(
    *,
    title: str,
    original_filename: str,
    storage_path: str,
    doc_type: str,
    published_on: dt.date,
    sha256: str,
    file_size: int,
    page_count: int | None,
    event_tags: list[str] | None = None,
    grade_tags: list[str] | None = None,
) -> str:
    client = get_supabase_client()
    payload = {
        "title": title,
        "original_filename": original_filename,
        "storage_path": storage_path,
        "doc_type": doc_type,
        "event_tags": event_tags or [],
        "grade_tags": grade_tags or ["Grade 3"],
        "published_on": published_on.isoformat(),
        "source_sha256": sha256,
        "file_size_bytes": file_size,
        "page_count": page_count,
    }
    try:
        response = client.table("documents").insert(payload).execute()
    except Exception as exc:
        message = str(exc)
        if "duplicate key value" in message or "already exists" in message:
            existing = (
                client.table("documents")
                .select("id")
                .eq("source_sha256", payload["source_sha256"])
                .execute()
            )
            if existing.data:
                document_id = existing.data[0]["id"]
                client.table("documents").update(payload).eq("id", document_id).execute()
                LOGGER.info(
                    "Updated metadata for existing document %s (id %s).",
                    original_filename,
                    document_id,
                )
                return document_id
        raise
    if not response.data:
        raise RuntimeError(f"Failed to insert document metadata for {original_filename}")
    return response.data[0]["id"]


def insert_chunks(
    document_id: str,
    published_on: dt.date,
    chunks: Iterable[TextChunk],
    embeddings: List[List[float]],
) -> None:
    client = get_supabase_client()
    rows = []
    for chunk, vector in zip(chunks, embeddings):
        rows.append(
            {
                "document_id": document_id,
                "chunk_index": chunk.index,
                "content": chunk.content,
                "published_on": published_on.isoformat(),
                "embedding": vector,
            }
        )
    client.table("document_chunks").insert(rows).execute()


def purge_document_chunks(document_id: str) -> None:
    client = get_supabase_client()
    client.table("document_chunks").delete().eq("document_id", document_id).execute()


def ensure_bucket_exists() -> None:
    client = get_supabase_client()
    bucket = get_settings().storage_bucket
    existing = client.storage.list_buckets()
    names = set()
    for item in existing or []:
        if hasattr(item, "name"):
            names.add(item.name)  # type: ignore[attr-defined]
        elif isinstance(item, dict) and "name" in item:
            names.add(item["name"])
    if bucket not in names:
        LOGGER.info("Creating storage bucket %s", bucket)
        client.storage.create_bucket(bucket, {"public": False})


@app.command()
def ingest(
    directory: Path = typer.Option(
        None,
        "--directory",
        "-d",
        help="Directory containing documents (defaults to INGESTION_SOURCE_DIR).",
    ),
    reembed: bool = typer.Option(
        False,
        "--reembed",
        help="Recompute embeddings and refresh chunks even if the document is already stored.",
    ),
) -> None:
    settings = get_settings()
    source_dir = directory.resolve() if directory else settings.ingestion_source_dir
    if not source_dir.exists():
        raise typer.BadParameter(f"Directory not found: {source_dir}")

    ensure_bucket_exists()

    existing = load_existing_hashes()
    LOGGER.info("Loaded %d existing documents from Supabase.", len(existing))

    allowed_suffixes = {".pdf", ".png", ".jpg", ".jpeg"}
    files = sorted(
        [
            p
            for p in source_dir.glob("**/*")
            if p.is_file() and p.suffix.lower() in allowed_suffixes
        ],
        key=lambda p: p.name.lower(),
    )
    if not files:
        LOGGER.warning("No files found in %s", source_dir)
        return

    openai_client = _get_openai_client()
    overrides = load_metadata_overrides()

    for path in tqdm(files, desc="Ingesting documents"):
        sha256 = compute_sha256(path)
        existing_record = existing.get(sha256)
        if existing_record and not reembed:
            LOGGER.info("Skipping %s (already ingested).", path.name)
            continue
        if existing_record and reembed:
            LOGGER.info("Reprocessing %s (refreshing embeddings).", path.name)

        LOGGER.info("Processing %s", path.name)
        text, page_count = extract_text(path)
        if not text.strip():
            LOGGER.warning("No text content extracted from %s. Skipping.", path)
            continue

        inferred = infer_metadata(path, text)
        override = overrides.get(path.name, {})

        default_date = dt.date.today()
        if "published_on" in override:
            try:
                default_date = dt.date.fromisoformat(override["published_on"])
            except ValueError:
                LOGGER.warning(
                    "Invalid published_on override for %s: %s",
                    path.name,
                    override["published_on"],
                )
        published_on = determine_document_date(text=text, path=path, default=default_date)

        doc_type = override.get("doc_type", inferred.get("doc_type")) or "circular"
        event_tags = override.get("event_tags", inferred.get("event_tags"))
        if event_tags:
            event_tags = sorted({tag for tag in event_tags if tag})

        grade_tags = override.get("grade_tags", inferred.get("grade_tags"))
        if grade_tags:
            grade_tags = sorted({tag for tag in grade_tags if tag})
        else:
            grade_tags = ["Grade 3"]

        if doc_type.lower() != "calendar":
            if "Grade 3" not in grade_tags:
                grade_tags = sorted({*grade_tags, "Grade 3"})

        title = override.get("title", inferred.get("title") or path.stem)

        LOGGER.debug(
            "Metadata for %s -> title=%r doc_type=%s grades=%s events=%s",
            path.name,
            title,
            doc_type,
            grade_tags,
            event_tags,
        )

        chunks = chunk_text(
            text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        if not chunks:
            LOGGER.warning("No chunks generated for %s. Skipping.", path)
            continue

        response = openai_client.embeddings.create(
            model=_EMBED_MODEL_NAME,
            input=[chunk.content for chunk in chunks],
            dimensions=_EMBED_DIMENSION,
        )
        embeddings = [item.embedding for item in response.data]

        if existing_record and existing_record.get("storage_path"):
            storage_path = existing_record["storage_path"]
        else:
            safe_filename = sanitize_filename(path.name)
            storage_path = f"circulars/{sha256[:12]}_{safe_filename}"
            upload_file(storage_path, path)

        document_id = insert_document_record(
            title=title,
            original_filename=path.name,
            storage_path=storage_path,
            doc_type=doc_type,
            published_on=published_on,
            sha256=sha256,
            file_size=path.stat().st_size,
            page_count=page_count,
            event_tags=event_tags,
            grade_tags=grade_tags,
        )

        if existing_record and reembed:
            purge_document_chunks(document_id)

        insert_chunks(document_id, published_on, chunks, embeddings)
        LOGGER.info("Stored %s with %d chunks.", path.name, len(chunks))


if __name__ == "__main__":
    app()
