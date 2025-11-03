from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from .config import get_settings
from .supabase_client import get_supabase_client


def _guardian_from_token(token: str) -> dict:
    client = get_supabase_client(service_role=True)
    try:
        result = client.auth.get_user(token)
    except Exception as exc:  # pragma: no cover - auth errors
        logger.warning("Supabase get_user failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token validation failed.",
        ) from exc

    user = getattr(result, "user", None)
    user_id = getattr(user, "id", None)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token validation failed.",
        )

    guardian_resp = (
        client.table("guardians")
        .select("id,student_id,auth_user_id")
        .eq("auth_user_id", user_id)
        .limit(1)
        .execute()
    )
    guardian_rows = guardian_resp.data or []
    if not guardian_rows:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Guardian record not found.",
        )
    return guardian_rows[0]


def _require_guardian(authorization: str | None = Header(default=None)) -> dict:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header.",
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme.",
        )
    return _guardian_from_token(token)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

TIMETABLE_FILENAME = "Grade_3_Ena_22072025_22072025143509_38.pdf"


class RecentDocument(BaseModel):
    document_id: str
    title: str
    display_title: str
    published_on: Optional[str]
    download_path: Optional[str]
    is_timetable: bool = False


@router.get("/recent", response_model=List[RecentDocument])
def recent_circulars() -> List[RecentDocument]:
    settings = get_settings()
    client = get_supabase_client(service_role=True)

    documents: list[RecentDocument] = []

    try:
        timetable_resp = (
            client.table("documents")
            .select("id,title,original_filename,published_on,uploaded_at,storage_path")
            .eq("original_filename", TIMETABLE_FILENAME)
            .order("published_on", desc=True)
            .order("uploaded_at", desc=True)
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

    if timetable_rows:
        doc = timetable_rows[0]
        documents.append(
            RecentDocument(
                document_id=doc["id"],
                title=doc.get("title") or "Grade 3 Ena time table",
                display_title="Grade 3 Ena time table",
                published_on=doc.get("published_on"),
                download_path=f"/documents/{doc['id']}/file",
                is_timetable=True,
            )
        )

    try:
        recent_resp = (
            client.table("documents")
            .select("id,title,original_filename,published_on,uploaded_at,storage_path")
            .neq("original_filename", TIMETABLE_FILENAME)
            .order("published_on", desc=True)
            .order("uploaded_at", desc=True)
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
                download_path=f"/documents/{row['id']}/file",
            )
        )

    # Cap at five entries total while ensuring timetable stays on top when present
    if len(documents) > 5:
        if documents and documents[0].is_timetable:
            documents = documents[:1] + documents[1:5]
        else:
            documents = documents[:5]

    return documents


@router.get("/{document_id}/file")
def fetch_document_file(document_id: str):
    settings = get_settings()
    client = get_supabase_client(service_role=True)

    doc_resp = (
        client.table("documents")
        .select("storage_path,original_filename")
        .eq("id", document_id)
        .limit(1)
        .execute()
    )
    doc_rows = doc_resp.data or []
    if not doc_rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    storage_path = doc_rows[0].get("storage_path")
    if not storage_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    try:
        result = client.storage.from_(settings.storage_bucket).create_signed_url(
            storage_path, 3600, {"download": False}
        )
    except Exception as exc:  # pragma: no cover - storage errors
        logger.warning("Failed to create signed URL for %s: %s", storage_path, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to create download link.",
        ) from exc

    payload = None
    if isinstance(result, dict):
        payload = result
    elif hasattr(result, "data"):
        payload = getattr(result, "data") or {}

    signed_url = None
    if isinstance(payload, dict):
        signed_url = payload.get("signedURL") or payload.get("signedUrl")

    if not signed_url:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Download link was empty.",
        )

    if not signed_url.startswith("http"):
        base_url = settings.supabase_url.rstrip("/")
        path = signed_url if signed_url.startswith("/") else f"/{signed_url}"
        signed_url = f"{base_url}{path}"

    return RedirectResponse(url=signed_url)
