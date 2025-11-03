from __future__ import annotations

import datetime as dt
import os
import hashlib
import json
import logging
import mimetypes
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Sequence

import typer
from tqdm import tqdm

from postgrest import APIError

from .config import get_settings
from .supabase_client import get_supabase_client
from .utils.chunks import TextChunk, chunk_pages
from .utils.date_parser import determine_document_date
from .utils.metadata import infer_metadata
from .utils.structured import extract_structured_data, StructuredDocumentData
from .utils.text_extraction import extract_text, extract_page_texts, extract_pdf_layout
from .utils.sections import (
    Highlight,
    Section,
    build_chunks_from_sections,
    build_sections,
    extract_highlights,
    extract_highlights_from_chunks,
)
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


def upload_file(storage_path: str, local_path: Path, *, force: bool = False) -> None:
    client = get_supabase_client()
    bucket = get_settings().storage_bucket
    bucket_client = client.storage.from_(bucket)

    content_type, _ = mimetypes.guess_type(str(local_path))
    if not content_type:
        # default to PDF for common circulars, otherwise generic binary
        content_type = "application/pdf" if local_path.suffix.lower() == ".pdf" else "application/octet-stream"

    upsert_flag = "true" if force else "false"
    upload_options = {"contentType": content_type, "upsert": upsert_flag}

    success = False
    try:
        with local_path.open("rb") as file_handle:
            bucket_client.upload(storage_path, file_handle, upload_options)
        success = True
    except Exception as exc:
        message = str(exc)
        if not force and ("Duplicate" in message or "already exists" in message):
            LOGGER.info("File %s already exists in storage. Skipping upload.", storage_path)
            _ensure_storage_mimetype(storage_path, content_type)
            return
        if force:
            LOGGER.warning(
                "Forced upload failed for %s (%s); attempting overwrite fallback.",
                storage_path,
                exc,
            )
            with local_path.open("rb") as file_handle:
                bucket_client.upload(
                    storage_path,
                    file_handle,
                    {"contentType": content_type, "upsert": "true"},
                )
            success = True
        else:
            raise
    if success:
        _ensure_storage_mimetype(storage_path, content_type)


def _ensure_storage_mimetype(storage_path: str, content_type: str) -> None:
    client = get_supabase_client()
    bucket = get_settings().storage_bucket
    storage_table = client.schema("storage").from_("objects")
    try:
        response = (
            storage_table.select("id,metadata")
            .eq("bucket_id", bucket)
            .eq("name", storage_path)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        LOGGER.warning("Failed to fetch storage metadata for %s: %s", storage_path, exc)
        return

    rows = response.data or []
    if not rows:
        LOGGER.warning("Storage object not found when setting mimetype: %s", storage_path)
        return

    record = rows[0]
    metadata = record.get("metadata") or {}
    if metadata.get("mimetype") == content_type:
        return

    metadata["mimetype"] = content_type
    try:
        storage_table.update({"metadata": metadata}).eq("id", record["id"]).execute()
    except Exception as exc:
        LOGGER.warning("Failed to set mimetype for %s: %s", storage_path, exc)

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
    now_iso = dt.datetime.utcnow().isoformat()
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
        "uploaded_at": now_iso,
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
                "page_number": chunk.page,
                "section_heading": chunk.heading,
            }
        )
    client.table("document_chunks").insert(rows).execute()


def purge_document_chunks(document_id: str) -> None:
    client = get_supabase_client()
    client.table("document_chunks").delete().eq("document_id", document_id).execute()


def _bulk_insert(table_name: str, rows: List[dict]) -> None:
    if not rows:
        return
    client = get_supabase_client()
    try:
        client.table(table_name).insert(rows).execute()
    except Exception as exc:  # pragma: no cover - depends on Supabase schema
        LOGGER.warning("Failed to insert structured rows into %s: %s", table_name, exc)


def _update_document_structured_fields(document_id: str, data: StructuredDocumentData) -> None:
    payload: dict[str, object] = {}
    if data.issued_date:
        payload["issued_on"] = data.issued_date.isoformat()
    if data.effective_start:
        payload["effective_start_on"] = data.effective_start.isoformat()
    if data.effective_end:
        payload["effective_end_on"] = data.effective_end.isoformat()
    if data.title_confidence is not None:
        payload["title_confidence"] = float(data.title_confidence)
    if data.audience:
        payload["audience"] = list({aud for aud in data.audience if aud})
    if payload:
        client = get_supabase_client()
        try:
            client.table("documents").update(payload).eq("id", document_id).execute()
        except Exception as exc:  # pragma: no cover - schema dependent
            LOGGER.warning("Failed to update structured fields on documents: %s", exc)


def store_structured_metadata(document_id: str, data: StructuredDocumentData) -> None:
    _update_document_structured_fields(document_id, data)

    client = get_supabase_client()
    for table in (
        "document_dates",
        "document_actions",
        "document_contacts",
        "document_entities",
        "document_page_summaries",
        "document_headings",
        "document_tables",
    ):
        try:
            client.table(table).delete().eq("document_id", document_id).execute()
        except Exception as exc:  # pragma: no cover - schema dependent
            LOGGER.warning("Failed to clear existing structured rows in %s: %s", table, exc)

    date_rows = [
        {
            "document_id": document_id,
            "date_type": item.kind,
            "date_value": item.date.isoformat(),
            "raw_text": item.text,
            "confidence": item.confidence,
        }
        for item in data.dates
    ]
    _bulk_insert("document_dates", date_rows)

    action_rows = [
        {
            "document_id": document_id,
            "description": action.description,
            "due_date": action.due_date.isoformat() if action.due_date else None,
            "audience": action.audience,
            "confidence": action.confidence,
            "source_excerpt": action.source_excerpt,
        }
        for action in data.actions
    ]
    _bulk_insert("document_actions", action_rows)

    contact_rows = [
        {
            "document_id": document_id,
            "contact_name": contact.name,
            "role": contact.role,
            "email": contact.email,
            "phone": contact.phone,
            "notes": contact.notes,
        }
        for contact in data.contacts
    ]
    _bulk_insert("document_contacts", contact_rows)

    entity_rows = [
        {
            "document_id": document_id,
            "entity_value": entity.value,
            "entity_type": entity.entity_type,
            "confidence": entity.confidence,
            "context": entity.context,
        }
        for entity in data.entities
    ]
    _bulk_insert("document_entities", entity_rows)

    summary_rows = [
        {
            "document_id": document_id,
            "page_number": summary.page_number,
            "summary": summary.summary,
        }
        for summary in data.page_summaries
    ]
    _bulk_insert("document_page_summaries", summary_rows)

    heading_rows = [
        {
            "document_id": document_id,
            "page_number": heading.page_number,
            "heading_text": heading.text,
            "heading_level": heading.level,
        }
        for heading in data.headings
    ]
    _bulk_insert("document_headings", heading_rows)

    table_rows = [
        {
            "document_id": document_id,
            "page_number": table.page_number,
            "table_data": json.dumps(table.rows),
        }
        for table in data.tables
    ]
    _bulk_insert("document_tables", table_rows)


def store_document_highlights(
    document_id: str,
    highlights: Sequence[Highlight],
    chunks: Sequence[TextChunk],
) -> None:
    if not highlights:
        return

    section_to_chunk: dict[str, int] = {}
    for chunk in chunks:
        if chunk.section_id and chunk.section_id not in section_to_chunk:
            section_to_chunk[chunk.section_id] = chunk.index

    rows: List[dict] = []
    for idx, highlight in enumerate(highlights, start=1):
        chunk_idx = None
        if highlight.chunk_index is not None:
            chunk_idx = highlight.chunk_index
        elif highlight.section_id:
            chunk_idx = section_to_chunk.get(highlight.section_id)
        rows.append(
            {
                "document_id": document_id,
                "highlight_index": idx,
                "text": highlight.text,
                "importance": float(highlight.score),
                "section_id": highlight.section_id,
                "chunk_index": chunk_idx,
            }
        )
    client = get_supabase_client()
    try:
        client.table("document_highlights").delete().eq("document_id", document_id).execute()
        client.table("document_highlights").insert(rows).execute()
    except Exception as exc:  # pragma: no cover - table may not exist yet
        LOGGER.debug("Skipping highlight persistence for %s: %s", document_id, exc)


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
    structured: bool | None = typer.Option(
        None,
        "--structured/--no-structured",
        help="Force enable or disable structured chunking for this run (defaults to settings).",
    ),
    structured_chunk_size: int | None = typer.Option(
        None,
        "--structured-chunk-size",
        help="Override structured chunk size for this run.",
    ),
    structured_chunk_overlap: int | None = typer.Option(
        None,
        "--structured-chunk-overlap",
        help="Override structured chunk overlap for this run.",
    ),
) -> None:
    settings = get_settings()
    structured_enabled = structured if structured is not None else settings.use_structured_chunking
    structured_size = structured_chunk_size or settings.structured_chunk_size
    structured_overlap = structured_chunk_overlap or settings.structured_chunk_overlap
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
        page_texts = extract_page_texts(path)
        layout_info = extract_pdf_layout(path) if path.suffix.lower() == ".pdf" else []
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

        structured_data = extract_structured_data(
            text=text,
            page_texts=page_texts,
            published_on=published_on,
            audience=grade_tags,
            layout=layout_info,
        )

        sections: list[Section] | None = None
        highlights: list[Highlight] = []

        if structured_enabled:
            section_tree = build_sections(page_texts, layout_info)
            sections = section_tree
            structured_chunks = build_chunks_from_sections(
                section_tree,
                chunk_size=structured_size,
                overlap=structured_overlap,
            )
            if structured_chunks:
                chunks = structured_chunks
                highlights = extract_highlights(section_tree)
                LOGGER.debug(
                    "Structured chunking produced %d chunks and %d highlights for %s",
                    len(chunks),
                    len(highlights),
                    path.name,
                )
            else:
                LOGGER.debug(
                    "Structured chunking yielded no chunks for %s; falling back to legacy chunking.",
                    path.name,
                )
                chunks = chunk_pages(
                    page_texts,
                    chunk_size=settings.chunk_size,
                    overlap=settings.chunk_overlap,
                )
        else:
            chunks = chunk_pages(
                page_texts,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )

        if sections and not highlights:
            highlights = extract_highlights(sections)
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
            if reembed:
                upload_file(storage_path, path, force=True)
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
        store_document_highlights(document_id, highlights, chunks)
        LOGGER.info("Stored %s with %d chunks.", path.name, len(chunks))

        store_structured_metadata(document_id, structured_data)


@app.command("refresh-highlights")
def refresh_highlights(
    document_id: str = typer.Option(
        None,
        "--document-id",
        help="Recompute highlights for a specific document UUID. When omitted, all non-calendar documents are processed.",
    ),
    limit: int = typer.Option(
        0,
        "--limit",
        help="Only refresh this many documents (useful for batches). 0 means no limit.",
    ),
    max_highlights: int = typer.Option(
        3,
        "--max-highlights",
        help="Maximum number of highlight bullets to store per document.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview the regenerated highlights without writing to Supabase.",
    ),
) -> None:
    """
    Recompute document_highlights entries from stored chunks to remove table noise.
    """

    client = get_supabase_client()

    if document_id:
        response = (
            client.table("documents")
            .select("id,title,doc_type,published_on")
            .eq("id", document_id)
            .limit(1)
            .execute()
        )
        documents = response.data or []
        if not documents:
            raise typer.Exit(f"Document {document_id} was not found.")
    else:
        query = (
            client.table("documents")
            .select("id,title,doc_type,published_on")
            .order("published_on", desc=True)
            .neq("doc_type", "calendar")
        )
        if limit > 0:
            query = query.limit(limit)
        documents = query.execute().data or []

    if not documents:
        typer.echo("No documents found for highlight refresh.")
        return

    total = len(documents)
    typer.echo(f"Refreshing highlights for {total} document(s)...")

    refreshed = 0
    for doc in tqdm(documents, disable=total < 5):
        doc_id = doc["id"]
        title = doc.get("title") or "Untitled"
        try:
            chunk_resp = (
                client.table("document_chunks")
                .select("chunk_index,content,page_number,section_heading,section_id")
                .eq("document_id", doc_id)
                .order("chunk_index")
                .execute()
            )
            chunk_rows = chunk_resp.data or []
            section_id_supported = True
        except APIError as exc:
            if "section_id" not in str(exc):
                raise
            chunk_resp = (
                client.table("document_chunks")
                .select("chunk_index,content,page_number,section_heading")
                .eq("document_id", doc_id)
                .order("chunk_index")
                .execute()
            )
            chunk_rows = chunk_resp.data or []
            section_id_supported = False
        if not chunk_rows:
            typer.echo(f"[skip] {title} ({doc_id}) has no chunks.")
            continue

        chunks: List[TextChunk] = []
        for row in chunk_rows:
            index = row.get("chunk_index")
            if index is None:
                continue
            chunks.append(
                TextChunk(
                    index=int(index),
                    content=row.get("content") or "",
                    page=row.get("page_number"),
                    heading=row.get("section_heading"),
                    section_id=row.get("section_id") if section_id_supported else None,
                )
            )

        highlights = extract_highlights_from_chunks(chunks, max_highlights=max_highlights)

        existing_resp = (
            client.table("document_highlights")
            .select("highlight_index,text")
            .eq("document_id", doc_id)
            .order("highlight_index")
            .execute()
        )
        existing = [row["text"] for row in existing_resp.data or []]

        if dry_run:
            typer.echo(f"\n{title} ({doc_id})")
            typer.echo("Existing:")
            if existing:
                for idx, text in enumerate(existing, start=1):
                    typer.echo(f"  {idx}. {text}")
            else:
                typer.echo("  (none)")
            typer.echo("Proposed:")
            if highlights:
                for idx, highlight in enumerate(highlights, start=1):
                    typer.echo(f"  {idx}. {highlight.text}")
            else:
                typer.echo("  (none)")
            continue

        if not highlights:
            client.table("document_highlights").delete().eq("document_id", doc_id).execute()
            typer.echo(f"[cleared] {title} ({doc_id}) – no highlight candidates found.")
            refreshed += 1
            continue

        store_document_highlights(doc_id, highlights, chunks)
        typer.echo(f"[updated] {title} ({doc_id})")
        refreshed += 1

    typer.echo(f"Completed highlight refresh for {refreshed} document(s).")


@app.command("refresh-mimetypes")
def refresh_mimetypes(
    document_id: str = typer.Option(
        None,
        "--document-id",
        help="Update storage metadata for a specific document UUID.",
    ),
    limit: int = typer.Option(
        0,
        "--limit",
        help="Only process this many documents (0 means all matching documents).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show planned metadata changes without writing to Supabase.",
    ),
) -> None:
    """
    Ensure storage objects have mimetype metadata aligned with their file extension.
    """

    client = get_supabase_client()
    settings = get_settings()
    bucket = settings.storage_bucket

    query = (
        client.table("documents")
        .select("id,title,original_filename,storage_path")
        .order("published_on", desc=True)
    )
    if document_id:
        query = query.eq("id", document_id)
    if limit > 0:
        query = query.limit(limit)

    docs = query.execute().data or []
    if not docs:
        typer.echo("No documents found for mimetype refresh.")
        return

    processed = 0
    for doc in tqdm(docs, disable=len(docs) < 5):
        storage_path = doc.get("storage_path")
        if not storage_path:
            continue
        filename = doc.get("original_filename") or storage_path.rsplit("/", 1)[-1]
        expected_type, _ = mimetypes.guess_type(filename)
        if not expected_type and filename.lower().endswith(".pdf"):
            expected_type = "application/pdf"
        if not expected_type:
            continue

        storage_table = client.schema("storage").from_("objects")
        resp = (
            storage_table.select("id,metadata")
            .eq("bucket_id", bucket)
            .eq("name", storage_path)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            typer.echo(f"[missing] {filename} ({storage_path}) not found in storage.")
            continue

        record = rows[0]
        metadata = record.get("metadata") or {}
        current_type = metadata.get("mimetype")
        if current_type == expected_type:
            continue

        typer.echo(f"[update] {filename}: {current_type or '(unset)'} -> {expected_type}")
        if dry_run:
            continue

        metadata["mimetype"] = expected_type
        storage_table.update({"metadata": metadata}).eq("id", record["id"]).execute()
        processed += 1

    typer.echo(f"Updated metadata for {processed} document(s).")


if __name__ == "__main__":
    app()
