from __future__ import annotations

import datetime as dt
import logging
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, constr

from .config import Settings, get_settings
from .email import send_email
from .calendar_events import fetch_calendar_context
from .llm_client import LLMClientError, generate_answer
from .manual_context import match_manual_facts
from .rag import ChunkHit, embed_text, fetch_relevant_chunks
from .supabase_client import get_supabase_client
from .rate_limiter import RateLimiter


logger = logging.getLogger(__name__)


router = APIRouter(prefix="/qa", tags=["qa"])

_rate_limiter: RateLimiter | None = None


class QARequest(BaseModel):
    question: constr(min_length=5, max_length=2000)  # type: ignore[valid-type]
    grade: str | None = Field(default="Grade 3", max_length=50)
    guardian_name: str | None = Field(default=None, max_length=200)
    guardian_email: str | None = Field(default=None, max_length=200)
    captcha_token: str | None = Field(default=None, max_length=4096)


class SourceInfo(BaseModel):
    document_id: str
    title: str
    published_on: str | None
    original_filename: str
    signed_url: str | None
    storage_path: str
    similarity: float


class QAResponse(BaseModel):
    status: str
    answer: str
    sources: List[SourceInfo]


def _format_sources_section(sources: List[SourceInfo]) -> str:
    if not sources:
        return ""
    lines = ["Sources:"]
    for idx, src in enumerate(sources, start=1):
        if src.published_on:
            date_label = src.published_on
        elif src.document_id.startswith("manual:"):
            date_label = "Manual summary"
        else:
            date_label = "Date unavailable"
        lines.append(f"{idx}. {src.title} — {date_label}")
    return "\n".join(lines)


def _create_signed_url(client, storage_path: str) -> str | None:
    if not storage_path:
        return None
    bucket = get_settings().storage_bucket
    try:
        if not client.storage.from_(bucket).exists(storage_path):
            logger.warning("Storage object missing for path %s", storage_path)
            return None
    except Exception as exc:  # pragma: no cover - storage errors
        logger.warning("Error checking existence of %s: %s", storage_path, exc)
        return None
    try:
        result = client.storage.from_(bucket).create_signed_url(
            storage_path, 3600, {"download": True}
        )
    except Exception as exc:  # pragma: no cover - network/storage errors
        logger.warning("Failed to create signed URL for %s: %s", storage_path, exc)
        return None
    payload = None
    if isinstance(result, dict):
        payload = result
    elif hasattr(result, "data"):
        payload = getattr(result, "data") or {}

    if isinstance(payload, dict):
        signed = payload.get("signedURL") or payload.get("signed_url")
        if not signed:
            return None
        if signed.startswith("http"):
            return signed
        base_url = get_settings().supabase_url.rstrip("/")
        path = signed if signed.startswith("/") else f"/{signed}"
        return f"{base_url}{path}"
    return None


def _fetch_document_metadata(client, document_ids: List[str]) -> Dict[str, dict]:
    if not document_ids:
        return {}
    response = (
        client.table("documents")
        .select("id,title,published_on,grade_tags,event_tags,doc_type")
        .in_("id", document_ids)
        .execute()
    )
    metadata: Dict[str, dict] = {}
    for row in response.data or []:
        metadata[row["id"]] = row
    return metadata


def _group_sources(client, hits: List[ChunkHit]) -> List[SourceInfo]:
    seen: OrderedDict[str, ChunkHit] = OrderedDict()
    for hit in hits:
        if hit.document_id not in seen:
            seen[hit.document_id] = hit

    sources: list[SourceInfo] = []
    for hit in seen.values():
        signed_url = _create_signed_url(client, hit.storage_path)
        sources.append(
            SourceInfo(
                document_id=hit.document_id,
                title=hit.document_title,
                published_on=hit.published_on.isoformat() if hit.published_on else None,
                original_filename=hit.original_filename,
                signed_url=signed_url,
                storage_path=hit.storage_path,
                similarity=hit.similarity,
            )
        )
    return sources


def _verify_turnstile(token: str, secret: str, remote_ip: str | None) -> bool:
    payload = {
        "secret": secret,
        "response": token,
    }
    if remote_ip:
        payload["remoteip"] = remote_ip
    try:
        response = requests.post(
            "https://challenges.cloudflare.com/turnstile/v0/siteverify",
            data=payload,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return True
        errors = data.get("error-codes") or data.get("errorCodes")
        logger.warning("Turnstile verification failed: %s", errors)
        return False
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Turnstile verification error: %s", exc)
        return False


def _build_prompt(question: str, hits: List[ChunkHit], metadata: Dict[str, dict]) -> str:
    grouped: OrderedDict[str, List[ChunkHit]] = OrderedDict()
    for hit in hits:
        grouped.setdefault(hit.document_id, []).append(hit)

    context_blocks: List[str] = []
    for idx, (doc_id, doc_hits) in enumerate(grouped.items(), start=1):
        primary = doc_hits[0]
        meta = metadata.get(doc_id, {})
        title = meta.get("title") or primary.document_title

        published_on = meta.get("published_on")
        if isinstance(published_on, str):
            date_label = published_on
        elif isinstance(published_on, dt.date):
            date_label = published_on.isoformat()
        else:
            date_label = primary.published_on.isoformat() if primary.published_on else "Unknown date"

        info_lines: List[str] = []
        grade_tags = meta.get("grade_tags") or []
        event_tags = meta.get("event_tags") or []
        if grade_tags:
            info_lines.append("Grades: " + ", ".join(grade_tags))
        if event_tags:
            info_lines.append("Topics: " + ", ".join(event_tags))

        snippets: List[str] = []
        for chunk in doc_hits[:2]:
            text = chunk.content.strip()
            if len(text) > 600:
                text = text[:600].rsplit(" ", 1)[0] + "…"
            snippets.append(text)
        snippet_text = "\n---\n".join(snippets)

        block_lines = [f"Document {idx}: {title} ({date_label})"]
        if info_lines:
            block_lines.append("\n".join(info_lines))
        block_lines.append("Content:")
        block_lines.append(snippet_text)
        context_blocks.append("\n".join(block_lines))

    context = "\n\n".join(context_blocks)
    instructions = (
        "You are the class knowledge base assistant for Grade 3 families. "
        "Answer the question using ONLY the provided documents."
        " If the answer is not present, respond with 'I do not know based on the available circulars.'"
        " Keep tone warm, concise, and factual."
    )

    prompt = (
        f"{instructions}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in 3-6 sentences."
    )
    return prompt


def _log_interaction(
    *,
    client,
    question: str,
    answer: str,
    status_label: str,
    sources: List[SourceInfo],
    similarity: float | None,
    latency_ms: int,
    model_name: str,
) -> None:
    cited_ids: List[str] = []
    for src in sources:
        doc_id = src.document_id
        if doc_id.startswith("manual:") or doc_id.startswith("calendar:"):
            continue
        cited_ids.append(doc_id)
    payload = {
        "question": question,
        "answer": answer,
        "status": status_label,
        "similarity_score": similarity,
        "llm_model": model_name,
        "response_latency_ms": latency_ms,
        "cited_document_ids": cited_ids,
    }
    client.table("qa_logs").insert(payload).execute()


def _send_escalation_email(
    settings: Settings,
    payload: QARequest,
    reasons: str,
) -> None:
    question = payload.question.strip()
    body_lines = [
        "A guardian question was escalated.",
        "",
        f"Guardian: {payload.guardian_name or 'Unknown'}",
        f"Contact: {payload.guardian_email or 'Unknown'}",
        f"Grade: {payload.grade or 'Unknown'}",
        "",
        "Question:",
        question,
        "",
        f"Reason: {reasons}",
    ]
    send_email(
        subject="[Action Required] Guardian question escalation",
        body_text="\n".join(body_lines),
    )


@router.post("", response_model=QAResponse)
def answer_question(
    payload: QARequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> QAResponse:
    global _rate_limiter
    start_time = time.perf_counter()
    client = get_supabase_client(service_role=True)

    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            limit=settings.qa_rate_limit_per_minute,
            window_seconds=60,
        )

    requester_ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.allow(requester_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many questions from this device. Please wait a minute and try again.",
        )

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

    env_label = (settings.app_env or "").lower()
    captcha_required = bool(settings.turnstile_secret_key) and env_label not in {"local", "dev", "development"}

    if captcha_required:
        if not payload.captcha_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Verification is required before submitting a question.",
            )
        if not _verify_turnstile(payload.captcha_token, settings.turnstile_secret_key, requester_ip):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Verification failed. Please try again.",
            )

    embedding = embed_text(question)

    try:
        hits = fetch_relevant_chunks(
            client=client,
            question=question,
            query_embedding=embedding,
            max_chunks=settings.qa_max_chunks,
            similarity_threshold=settings.qa_similarity_threshold,
            grade_tag=payload.grade or "Grade 3",
        )
    except Exception as exc:  # pragma: no cover - network outages, Supabase errors
        logger.exception("Retrieval failed while querying Supabase", exc_info=exc)
        answer_text = (
            "I’m having trouble reaching the knowledge base right now. "
            "Your question has been sent to the class parent and they will get back to you shortly."
        )
        _log_interaction(
            client=client,
            question=question,
            answer=answer_text,
            status_label="escalated",
            sources=[],
            similarity=None,
            latency_ms=int((time.perf_counter() - start_time) * 1000),
            model_name=settings.llm_provider,
        )
        _send_escalation_email(settings, payload, "Knowledge base unreachable (network/Supabase error).")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Knowledge base temporarily unavailable") from exc

    calendar_matches = fetch_calendar_context(client, question, payload.grade)
    for entry in calendar_matches:
        doc_id = f"calendar:{entry['title'].lower().replace(' ', '-')[:40]}"
        summary_content = entry["summary"]
        event_date = dt.date.fromisoformat(entry["event_date"])
        end_date = entry.get("end_date")
        if end_date:
            try:
                published = dt.date.fromisoformat(end_date)
            except ValueError:
                published = event_date
        else:
            published = event_date

        hits.append(
            ChunkHit(
                document_id=doc_id,
                chunk_index=0,
                content=summary_content,
                similarity=0.99,
                document_title=entry["title"],
                original_filename=entry.get("source") or "calendar",
                published_on=published,
                storage_path="",
            )
        )

    manual_entries = match_manual_facts(question)
    for title, content in manual_entries:
        manual_id = f"manual:{title.lower().replace(' ', '-')[:40]}"
        hits.append(
            ChunkHit(
                document_id=manual_id,
                chunk_index=0,
                content=content,
                similarity=1.0,
                document_title=title,
                original_filename=title,
                published_on=None,
                storage_path="",
            )
        )

    doc_ids: List[str] = []
    for hit in hits:
        if hit.document_id.startswith("manual:") or hit.document_id.startswith("calendar:"):
            continue
        doc_ids.append(hit.document_id)
    doc_meta = _fetch_document_metadata(client, list({*doc_ids}))

    if hits:
        hits.sort(
            key=lambda h: (
                getattr(h, "score", h.similarity),
                h.similarity,
                h.published_on or dt.date.min,
            ),
            reverse=True,
        )

    if not hits:
        answer_text = (
            "I could not find this information in the current circulars. "
            "Your question has been escalated to the class parent."
        )
        _log_interaction(
            client=client,
            question=question,
            answer=answer_text,
            status_label="escalated",
            sources=[],
            similarity=None,
            latency_ms=int((time.perf_counter() - start_time) * 1000),
            model_name=settings.llm_provider,
        )
        _send_escalation_email(settings, payload, "No relevant documents met the similarity threshold.")
        return QAResponse(status="escalated", answer=answer_text, sources=[])

    prompt = _build_prompt(question, hits, doc_meta)

    try:
        model_answer, model_used = generate_answer(prompt, settings)
    except LLMClientError as exc:
        answer_text = (
            "The assistant is temporarily unavailable. "
            "Your question has been escalated to the class parent."
        )
        _log_interaction(
            client=client,
            question=question,
            answer=answer_text,
            status_label="escalated",
            sources=[],
            similarity=None,
            latency_ms=int((time.perf_counter() - start_time) * 1000),
            model_name=settings.llm_provider,
        )
        _send_escalation_email(
            settings,
            payload,
            f"LLM provider error: {exc}",
        )
        return QAResponse(status="escalated", answer=answer_text, sources=[])
    except Exception as exc:  # pragma: no cover - unexpected LLM issues
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    if not model_answer:
        answer_text = (
            "I couldn't confidently answer from the circulars I have. "
            "I've escalated this question for a manual response."
        )
        _log_interaction(
            client=client,
            question=question,
            answer=answer_text,
            status_label="escalated",
            sources=[],
            similarity=hits[0].similarity if hits else None,
            latency_ms=int((time.perf_counter() - start_time) * 1000),
            model_name=model_used,
        )
        _send_escalation_email(settings, payload, "LLM returned an empty answer.")
        return QAResponse(status="escalated", answer=answer_text, sources=[])

    sources = _group_sources(client, hits)
    sources_section = _format_sources_section(sources)
    if sources_section and sources_section not in model_answer:
        answer_text = f"{model_answer.strip()}\n\n{sources_section}"
    else:
        answer_text = model_answer.strip()

    _log_interaction(
        client=client,
        question=question,
        answer=answer_text,
        status_label="answered",
        sources=sources,
        similarity=hits[0].similarity if hits else None,
        latency_ms=int((time.perf_counter() - start_time) * 1000),
        model_name=model_used,
    )

    return QAResponse(status="answered", answer=answer_text, sources=sources)
