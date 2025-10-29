from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from .config import get_settings
from .supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

TIMETABLE_FILENAME = "Grade_3_Ena_22072025_22072025143509_38.pdf"


class RecentDocument(BaseModel):
    document_id: str
    title: str
    display_title: str
    published_on: Optional[str]
    signed_url: Optional[str]
    is_timetable: bool = False


@router.get("/recent", response_model=List[RecentDocument])
def recent_circulars() -> List[RecentDocument]:
    settings = get_settings()
    client = get_supabase_client(service_role=True)

    documents: list[RecentDocument] = []

    try:
        timetable_resp = (
            client.table("documents")
            .select("id,title,original_filename,published_on,storage_path")
            .eq("original_filename", TIMETABLE_FILENAME)
            .order("published_on", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.exception("Failed to load timetable document", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to reach the document store.",
        ) from exc

    timetable_rows = timetable_resp.data or []

    def _signed_url(storage_path: str | None) -> Optional[str]:
        if not storage_path:
            return None
        try:
            result = (
                client.storage.from_(settings.storage_bucket)
                .create_signed_url(storage_path, 3600)
            )
        except Exception as storage_exc:  # pragma: no cover - network/storage failures
            logger.warning(
                "Failed to create signed URL for %s: %s", storage_path, storage_exc
            )
            return None

        payload = None
        if isinstance(result, dict):
            payload = result
        elif hasattr(result, "data"):
            payload = getattr(result, "data") or {}
        if isinstance(payload, dict):
            return payload.get("signedURL") or payload.get("signedUrl")
        return None

    if timetable_rows:
        doc = timetable_rows[0]
        documents.append(
            RecentDocument(
                document_id=doc["id"],
                title=doc.get("title") or "Grade 3 Ena time table",
                display_title="Grade 3 Ena time table",
                published_on=doc.get("published_on"),
                signed_url=_signed_url(doc.get("storage_path")),
                is_timetable=True,
            )
        )

    try:
        recent_resp = (
            client.table("documents")
            .select("id,title,original_filename,published_on,storage_path")
            .neq("original_filename", TIMETABLE_FILENAME)
            .order("published_on", desc=True)
            .limit(5)
            .execute()
        )
    except Exception as exc:
        logger.exception("Failed to load recent documents", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to reach the document store.",
        ) from exc

    for row in recent_resp.data or []:
        documents.append(
            RecentDocument(
                document_id=row["id"],
                title=row.get("title") or row.get("original_filename") or "Untitled circular",
                display_title=row.get("title") or row.get("original_filename") or "Untitled circular",
                published_on=row.get("published_on"),
                signed_url=_signed_url(row.get("storage_path")),
            )
        )

    # Cap at five entries total while ensuring timetable stays on top when present
    if len(documents) > 5:
        if documents and documents[0].is_timetable:
            documents = documents[:1] + documents[1:5]
        else:
            documents = documents[:5]

    return documents
