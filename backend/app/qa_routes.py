from __future__ import annotations

import datetime as dt
import logging
import re
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, constr

from .config import Settings, get_settings
from .email import send_email
from .calendar_events import fetch_calendar_context
from .calendar_resolver import CalendarAnswer, resolve_calendar_question
from .llm_client import LLMClientError, generate_answer
from .manual_context import match_manual_facts
from .rag import ChunkHit, embed_text, fetch_relevant_chunks, parse_date as rag_parse_date
from .supabase_client import get_supabase_client
from .rate_limiter import RateLimiter
from .temporal_context import (
    current_ist,
    infer_date_references,
    fetch_calendar_events_for_window,
    determine_holiday_answer,
    format_event_summary,
    build_reference_hint,
    format_date,
    HOLIDAY_KEYWORDS,
    parse_date_range_from_text,
    upcoming_holiday_event,
    upcoming_break_from_matches,
)


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


SCHEDULE_TERMS = {
    "uniform",
    "wear",
    "kit",
    "attire",
    "pickup",
    "dispersal",
    "dismissal",
    "bagless",
}

CONTACT_KEYWORDS = {
    "contact",
    "call",
    "phone",
    "reach",
    "email",
    "whatsapp",
}

CONTACT_ROLE_HINTS = {
    "transport": {"transport", "bus", "pickup", "drop"},
    "cafeteria": {"cafeteria", "canteen", "meal"},
    "academics": {"homework", "curriculum", "teacher"},
    "admin": {"fees", "payment", "accounts"},
}

QUICK_LINK_KEYWORD_MAP = {
    "cafeteria_menu": {"cafeteria", "canteen", "menu"},
    "bus_routes": {"bus", "transport"},
}

DATE_REGEX = re.compile(r"\b(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+|\d{4}-\d{2}-\d{2})\b")
TIME_REGEX = re.compile(r"\b\d{1,2}(:\d{2})?\s?(?:am|pm|hrs)\b", re.IGNORECASE)

STRUCTURED_STOPWORDS = {
    "what",
    "when",
    "where",
    "who",
    "how",
    "does",
    "do",
    "is",
    "the",
    "a",
    "an",
    "for",
    "with",
    "and",
    "or",
    "on",
    "in",
    "to",
    "of",
    "tell",
    "about",
    "please",
    "share",
    "info",
    "information",
    "details",
    "help",
    "me",
    "we",
    "need",
    "any",
}

DATE_INTENT_KEYWORDS = {
    "deadline": {"deadline", "due", "last date", "submit", "submission", "duedate"},
    "resume": {"resume", "reopen", "reopens", "restart"},
    "end": {"end", "ends", "finish", "finishes", "over", "closing"},
    "start": {"start", "starts", "begin", "begins", "commence", "opens"},
}


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


def _collect_schedule_terms(question: str) -> List[str]:
    lowered = question.lower()
    return [term for term in SCHEDULE_TERMS if term in lowered]


def _infer_contact_role(lowered: str) -> Optional[str]:
    for role, keywords in CONTACT_ROLE_HINTS.items():
        if any(keyword in lowered for keyword in keywords):
            return role
    return None


def _detect_quick_link_type(lowered: str) -> Optional[str]:
    for link_type, keywords in QUICK_LINK_KEYWORD_MAP.items():
        if any(keyword in lowered for keyword in keywords):
            return link_type
    return None


def _answer_contains_explicit_date(text: str) -> bool:
    return bool(DATE_REGEX.search(text) or TIME_REGEX.search(text))


def _extract_structured_keywords(question: str, limit: int = 6) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", question.lower())
    keywords: List[str] = []
    for token in tokens:
        if len(token) < 3:
            continue
        if token in STRUCTURED_STOPWORDS:
            continue
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def _detect_date_intent(lowered_question: str) -> Optional[str]:
    for intent, words in DATE_INTENT_KEYWORDS.items():
        if any(word in lowered_question for word in words):
            return intent
    return None


def _source_info_from_documents_row(client, row: dict, doc: dict, similarity: float = 1.0) -> SourceInfo:
    document_id = row.get("document_id") or doc.get("id")
    storage_path = doc.get("storage_path") or ""
    return SourceInfo(
        document_id=str(document_id),
        title=doc.get("title") or "Circular",
        published_on=doc.get("published_on"),
        original_filename=doc.get("original_filename") or "",
        signed_url=_create_signed_url(client, storage_path),
        storage_path=storage_path,
        similarity=similarity,
    )


def _structured_date_lookup(
    *,
    client,
    keywords: List[str],
    date_kind: str,
    grade: Optional[str],
) -> Optional[tuple[str, List[SourceInfo], str]]:
    try:
        query = (
            client.table("document_dates")
            .select(
                "document_id,date_type,date_value,raw_text,confidence,documents(title,published_on,storage_path,original_filename,grade_tags)"
            )
            .eq("date_type", date_kind)
            .order("confidence", desc=True)
        )
    except Exception as exc:  # pragma: no cover - Supabase errors
        logger.warning("Structured date lookup failed to build query: %s", exc)
        return None

    if grade:
        try:
            query = query.contains("documents.grade_tags", [grade])
        except Exception:
            # not all Supabase installs support contains on joined tables
            pass

    if keywords:
        or_filters = []
        for token in keywords:
            pattern = f"*{token}*"
            or_filters.append(f"raw_text.ilike.{pattern}")
            or_filters.append(f"documents.title.ilike.{pattern}")
        try:
            query = query.or_(",".join(or_filters))
        except Exception:
            pass

    try:
        response = query.limit(5).execute()
    except Exception as exc:  # pragma: no cover - Supabase errors
        logger.warning("Structured date lookup query error: %s", exc)
        return None

    rows = response.data or []
    if not rows:
        return None

    best_row = max(rows, key=lambda r: r.get("confidence") or 0.5)
    doc = best_row.get("documents") or {}
    date_value = best_row.get("date_value")
    if not date_value:
        return None
    try:
        if isinstance(date_value, dt.date):
            date_obj = date_value
        else:
            date_obj = dt.date.fromisoformat(str(date_value))
    except Exception:
        return None

    formatted_date = format_date(date_obj)
    doc_title = doc.get("title") or "This circular"
    action_phrase = {
        "deadline": "is due on",
        "resume": "resumes on",
        "end": "ends on",
        "start": "starts on",
    }.get(date_kind, "is on")
    answer = f"{doc_title} {action_phrase} {formatted_date}."

    sources = [_source_info_from_documents_row(client, best_row, doc)]
    return answer, sources, f"structured-date-{date_kind}"


def _structured_contact_lookup(
    *,
    client,
    keywords: List[str],
    role: Optional[str],
    grade: Optional[str],
) -> Optional[tuple[str, List[SourceInfo], str]]:
    try:
        query = (
            client.table("document_contacts")
            .select(
                "document_id,contact_name,role,email,phone,notes,documents(title,published_on,storage_path,original_filename,grade_tags)"
            )
            .order("created_at", desc=True)
        )
    except Exception as exc:
        logger.warning("Structured contact lookup failed to build query: %s", exc)
        return None

    if role:
        try:
            query = query.ilike("role", f"%{role}%")
        except Exception:
            pass

    if grade:
        try:
            query = query.contains("documents.grade_tags", [grade])
        except Exception:
            pass

    if keywords:
        or_filters = []
        for token in keywords:
            pattern = f"*{token}*"
            or_filters.append(f"notes.ilike.{pattern}")
            or_filters.append(f"documents.title.ilike.{pattern}")
        try:
            query = query.or_(",".join(or_filters))
        except Exception:
            pass

    try:
        response = query.limit(5).execute()
    except Exception as exc:
        logger.warning("Structured contact lookup query error: %s", exc)
        return None

    rows = response.data or []
    if not rows:
        return None

    best_row = rows[0]
    doc = best_row.get("documents") or {}
    contact_name = best_row.get("contact_name")
    contact_role = best_row.get("role") or role
    email = best_row.get("email")
    phone = best_row.get("phone")

    doc_title = doc.get("title") or "the circular"
    parts: List[str] = []
    if contact_name:
        parts.append(contact_name)
    if contact_role:
        parts.append(f"({contact_role})")
    contact_label = " ".join(parts) or "Contact"

    details: List[str] = []
    if phone:
        details.append(f"phone {phone}")
    if email:
        details.append(f"email {email}")
    if not details and best_row.get("notes"):
        details.append(best_row["notes"])

    if details:
        detail_text = " and ".join(details)
        answer = f"{contact_label} in {doc_title} can be reached via {detail_text}."
    else:
        answer = f"{contact_label} is listed in {doc_title}."

    sources = [_source_info_from_documents_row(client, best_row, doc, similarity=0.9)]
    return answer, sources, "structured-contact"


def _dedupe_sources(sources: List[SourceInfo]) -> List[SourceInfo]:
    seen: set[str] = set()
    unique: List[SourceInfo] = []
    for src in sources:
        if src.document_id in seen:
            continue
        seen.add(src.document_id)
        unique.append(src)
    return unique


def _lookup_structured_fact(
    *,
    client,
    question: str,
    grade: Optional[str],
) -> tuple[Optional[tuple[str, List[SourceInfo], str]], Optional[str]]:
    lowered = question.lower()
    keywords = _extract_structured_keywords(question)

    date_intent = _detect_date_intent(lowered)
    if date_intent:
        result = _structured_date_lookup(client=client, keywords=keywords, date_kind=date_intent, grade=grade)
        if result:
            return result, date_intent
        return None, date_intent

    contact_role = _infer_contact_role(lowered)
    if contact_role:
        result = _structured_contact_lookup(client=client, keywords=keywords, role=contact_role, grade=grade)
        if result:
            return result, "contact"
        return None, "contact"

    return None, None

def _structured_intent_fallback(question: str, intent_label: Optional[str] = None) -> Optional[tuple[str, List[SourceInfo], str]]:
    lowered = question.lower()
    if intent_label == "contact" or any(word in lowered for word in ("contact", "phone", "email")):
        answer = (
            "I couldn't find a contact for that in the current circulars. "
            "Please review the latest transport or administrative circular, or reach out to the class parent."
        )
        return answer, [], "no-structured-contact"
    if intent_label in {"end", "resume", "start", "deadline"} or any(word in lowered for word in ("deadline", "end", "resume")):
        answer = (
            "I couldn't find a specific date for that in the available circulars. "
            "Please check the latest circular or ask the class parent for confirmation."
        )
        return answer, [], "no-structured-date"
    return None

def _append_circular_suggestions(
    parts: List[str],
    sources: List[SourceInfo],
    document_sources: List[SourceInfo],
    *,
    limit: int = 2,
) -> None:
    if not document_sources:
        return

    limited_docs = document_sources[:limit]
    doc_lines = "\n".join(f"- {doc.title}" for doc in limited_docs)
    parts.append(f"Circulars:\n{doc_lines}")
    sources.extend(limited_docs)


def _fetch_document_metadata(client, document_ids: List[str]) -> Dict[str, dict]:
    if not document_ids:
        return {}
    response = (
        client.table("documents")
        .select("id,title,published_on,grade_tags,event_tags,doc_type,issued_on,effective_start_on,effective_end_on,audience,title_confidence")
        .in_("id", document_ids)
        .execute()
    )
    metadata: Dict[str, dict] = {}
    for row in response.data or []:
        metadata[row["id"]] = row
    return metadata


def _fetch_keyword_hits(client, question: str, limit: int) -> List[ChunkHit]:
    try:
        response = client.rpc(
            "match_document_chunks_fuzzy",
            {"q": question, "limit_count": limit},
        ).execute()
    except Exception as exc:  # pragma: no cover - Supabase RPC failure
        logger.warning("Keyword search RPC failed: %s", exc)
        return []

    rows = response.data or []
    hits: List[ChunkHit] = []
    for row in rows:
        document_id = row.get("document_id")
        if not document_id:
            continue
        chunk_index = row.get("chunk_index") or 0
        content = row.get("content") or ""
        similarity_raw = row.get("similarity")
        try:
            similarity = float(similarity_raw)
        except (TypeError, ValueError):
            similarity = 0.0
        published_on = rag_parse_date(row.get("published_on"))
        hits.append(
            ChunkHit(
                document_id=document_id,
                chunk_index=chunk_index,
                content=content,
                similarity=similarity,
                document_title=row.get("document_title") or row.get("title") or "Untitled",
                original_filename=row.get("original_filename") or "",
                published_on=published_on,
                storage_path=row.get("storage_path") or "",
                score=similarity,
            )
        )
    return hits


def _fetch_document_fuzzy(client, question: str, limit: int) -> List[SourceInfo]:
    try:
        response = client.rpc(
            "match_documents_fuzzy",
            {"q": question, "limit_count": limit},
        ).execute()
    except Exception as exc:  # pragma: no cover - Supabase RPC failure
        logger.warning("Document search RPC failed: %s", exc)
        return []

    sources: List[SourceInfo] = []
    for row in response.data or []:
        doc_id = row.get("id")
        if not doc_id:
            continue
        storage_path = row.get("storage_path") or ""
        sources.append(
            SourceInfo(
                document_id=doc_id,
                title=row.get("title") or "Untitled document",
                published_on=row.get("published_on"),
                original_filename=row.get("original_filename") or "",
                signed_url=_create_signed_url(client, storage_path),
                storage_path=storage_path,
                similarity=float(row.get("similarity") or 0.0),
            )
        )
    return sources


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


def _build_prompt(
    question: str,
    hits: List[ChunkHit],
    metadata: Dict[str, dict],
    temporal_hints: Optional[List[str]] = None,
) -> str:
    grouped: OrderedDict[str, List[ChunkHit]] = OrderedDict()
    for hit in hits:
        grouped.setdefault(hit.document_id, []).append(hit)

    calendar_blocks: List[str] = []
    document_blocks: List[str] = []
    manual_blocks: List[str] = []

    def _label_for_date(value) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dt.date):
            return value.isoformat()
        return "Unknown date"

    for doc_id, doc_hits in grouped.items():
        primary = doc_hits[0]
        meta = metadata.get(doc_id, {})
        title = meta.get("title") or primary.document_title

        published_on = meta.get("published_on")
        if not published_on and primary.published_on:
            published_on = primary.published_on
        date_label = _label_for_date(published_on)

        info_lines: List[str] = []
        grade_tags = meta.get("grade_tags") or []
        event_tags = meta.get("event_tags") or []
        audience_tags = meta.get("audience") or []
        issued_on = meta.get("issued_on")
        effective_start = meta.get("effective_start_on")
        effective_end = meta.get("effective_end_on")
        if grade_tags:
            info_lines.append("Grades: " + ", ".join(grade_tags))
        if event_tags:
            info_lines.append("Topics: " + ", ".join(event_tags))
        if audience_tags:
            info_lines.append("Audience: " + ", ".join(audience_tags))
        if issued_on:
            info_lines.append("Issued: " + _label_for_date(issued_on))
        range_parts: List[str] = []
        if effective_start:
            range_parts.append(_label_for_date(effective_start))
        if effective_end:
            range_parts.append(_label_for_date(effective_end))
        if range_parts:
            info_lines.append("Effective: " + " – ".join(range_parts))

        snippets: List[str] = []
        for chunk in doc_hits[:2]:
            text = chunk.content.strip()
            if len(text) > 600:
                text = text[:600].rsplit(" ", 1)[0] + "…"
            prefix_parts: List[str] = []
            if chunk.page_number:
                prefix_parts.append(f"Page {chunk.page_number}")
            if chunk.section_heading:
                prefix_parts.append(chunk.section_heading.strip())
            if prefix_parts:
                text = f"{' – '.join(prefix_parts)}\n{text}"
            snippets.append(text)
        snippet_text = "\n---\n".join(snippets)

        if doc_id.startswith("calendar:"):
            block_lines = [
                f"Calendar event: {title} ({date_label})",
                "Details:",
                snippet_text,
            ]
            calendar_blocks.append("\n".join(block_lines))
            continue

        if doc_id.startswith("manual:"):
            block_lines = [
                f"Manual note: {title}",
                "Details:",
                snippet_text,
            ]
            manual_blocks.append("\n".join(block_lines))
            continue

        block_lines = [f"Circular: {title} ({date_label})"]
        if info_lines:
            block_lines.append("\n".join(info_lines))
        block_lines.append("Content:")
        block_lines.append(snippet_text)
        document_blocks.append("\n".join(block_lines))

    context_sections: List[str] = []
    if calendar_blocks:
        context_sections.append("Calendar events:\n" + "\n\n".join(calendar_blocks))
    if document_blocks:
        context_sections.append("Circulars and documents:\n" + "\n\n".join(document_blocks))
    if manual_blocks:
        context_sections.append("Manual notes:\n" + "\n\n".join(manual_blocks))

    context = "\n\n".join(context_sections)

    instructions_parts = [
        "You are the class knowledge base assistant for Grade 3 families.",
        "Use the calendar events, circulars, and notes provided in the context to answer.",
        "Cite the calendar when giving dates about schedules or breaks, and cite circulars for policy or logistics details.",
        "If the answer is not present in any of the provided sources, respond with \"I do not know based on the available sources.\"",
        "Keep the tone warm, concise, and factual.",
    ]
    instructions = " ".join(instructions_parts)

    temporal_section = ""
    if temporal_hints:
        hint_lines = "\n".join(f"- {hint}" for hint in temporal_hints)
        temporal_section = f"Additional temporal context:\n{hint_lines}\n\n"

    prompt = (
        f"{instructions}\n\n"
        f"{temporal_section}"
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

    now_ist = current_ist()
    temporal_hints: List[str] = [f"Today is {format_date(now_ist.date())} (IST)."]
    date_references = infer_date_references(question, now_ist)
    event_hint_lines: List[str] = []
    holiday_result: Optional[dict] = None

    calendar_matches = fetch_calendar_context(client, question, payload.grade)

    for reference in date_references:
        hint = build_reference_hint(reference)
        if hint:
            temporal_hints.append(hint)
        if reference.target_date:
            events = fetch_calendar_events_for_window(
                client,
                reference.target_date,
                grade=payload.grade,
            )
            if events:
                for event in events:
                    event_hint_lines.append(format_event_summary(event))
                if holiday_result is None:
                    holiday_result = determine_holiday_answer(reference, events)
        elif reference.range_start and reference.range_end:
            range_length = (reference.range_end - reference.range_start).days
            midpoint = reference.range_start + dt.timedelta(days=range_length // 2)
            window = max(7, range_length + 1)
            events = fetch_calendar_events_for_window(
                client,
                midpoint,
                window,
                grade=payload.grade,
            )
            for event in events:
                event_hint_lines.append(format_event_summary(event))

    for match in calendar_matches or []:
        summary = match.get("summary")
        if summary:
            event_hint_lines.append(summary)

    if event_hint_lines:
        seen: set[str] = set()
        for line in event_hint_lines:
            if line in seen:
                continue
            seen.add(line)
            temporal_hints.append(f"Calendar note: {line}")
            if len(temporal_hints) >= 8:
                break

    grade_label = payload.grade or "Grade 3"

    calendar_answer = resolve_calendar_question(
        client=client,
        question=question,
        reference_date=now_ist.date(),
        grade=payload.grade,
    )
    doc_suggestions = _fetch_document_fuzzy(client, question, limit=3)

    structured_lookup = _lookup_structured_fact(
        client=client,
        question=question,
        grade=grade_label,
    )

    if holiday_result is None and date_references:
        for reference in date_references:
            if not reference.target_date:
                continue
            for match in calendar_matches or []:
                base_start = dt.date.fromisoformat(match["event_date"])
                end_value = match.get("end_date")
                base_end = dt.date.fromisoformat(end_value) if end_value else base_start
                summary_full = match.get("summary") or ""
                if not end_value and summary_full:
                    parsed_start, parsed_end = parse_date_range_from_text(
                        summary_full,
                        reference.target_date.year,
                    )
                    if parsed_start:
                        base_start = parsed_start
                    if parsed_end:
                        base_end = parsed_end
                summary_text = summary_full.lower()
                if base_start <= reference.target_date <= base_end and any(
                    keyword in summary_text for keyword in HOLIDAY_KEYWORDS
                ):
                    resume = base_end + dt.timedelta(days=1)
                    holiday_result = {
                        "answer": (
                            f"Yes, {reference.description or 'the requested date'} "
                            f"({format_date(reference.target_date)}) falls during {match.get('title', 'the listed break')} "
                            f"({format_date(base_start)} – {format_date(base_end)}). "
                            f"Classes resume on {format_date(resume)}."
                        ),
                        "event": {
                            "id": match.get("id") or match.get("title"),
                            "title": match.get("title"),
                            "source": match.get("source"),
                            "summary": match.get("summary"),
                            "description": match.get("summary"),
                        },
                        "start": base_start,
                        "end": base_end,
                        "resume": resume,
                    }
                    break
            if holiday_result:
                break

    lower_question = question.lower()
    upcoming_requested = any(
        phrase in lower_question
        for phrase in ["next holiday", "next break", "upcoming holiday", "upcoming break", "coming holiday", "coming break"]
    ) or ("next" in lower_question and "break" in lower_question)

    if holiday_result:
        event = holiday_result["event"]
        start = holiday_result["start"]
        answer_text = holiday_result["answer"]
        calendar_source = SourceInfo(
            document_id=f"calendar:{event.get('id') or event.get('title')}",
            title=event.get("title", "Calendar event"),
            published_on=start.isoformat(),
            original_filename=event.get("source") or "calendar",
            signed_url=None,
            storage_path="",
            similarity=1.0,
        )
        parts = [f"Calendar: {answer_text}"]
        sources = [calendar_source]
        _append_circular_suggestions(parts, sources, doc_suggestions)
        final_answer = "\n\n".join(parts)
        _log_interaction(
            client=client,
            question=question,
            answer=final_answer,
            status_label="answered",
            sources=sources,
            similarity=None,
            latency_ms=int((time.perf_counter() - start_time) * 1000),
            model_name="deterministic-calendar",
        )
        return QAResponse(status="answered", answer=final_answer, sources=sources)

    if holiday_result is None and upcoming_requested:
        upcoming = upcoming_holiday_event(client, now_ist.date(), grade=payload.grade)
        current_break = upcoming.get("current") if upcoming else None
        next_break = upcoming.get("next") if upcoming else None

        if (not current_break or not next_break) and calendar_matches:
            fallback = upcoming_break_from_matches(calendar_matches, now_ist.date())
            current_break = current_break or fallback.get("current")
            next_break = next_break or fallback.get("next")
        if current_break or next_break:
            parts: List[str] = []
            sources: List[SourceInfo] = []
            if current_break:
                parts.append(
                    f"Right now, {current_break['title']} runs from {format_date(current_break['start'])} "
                    f"to {format_date(current_break['end'])}. Classes resume on {format_date(current_break['resume'])}."
                )
                sources.append(
                    SourceInfo(
                        document_id=f"calendar:{current_break['title']}",
                        title=current_break["title"],
                        published_on=current_break["start"].isoformat(),
                        original_filename=current_break.get("source") or "calendar",
                        signed_url=None,
                        storage_path="",
                        similarity=1.0,
                    )
                )
            if next_break:
                parts.append(
                    f"The next break is {next_break['title']} from {format_date(next_break['start'])} "
                    f"to {format_date(next_break['end'])}. Classes resume on {format_date(next_break['resume'])}."
                )
                sources.append(
                    SourceInfo(
                        document_id=f"calendar:{next_break['title']}",
                        title=next_break["title"],
                        published_on=next_break["start"].isoformat(),
                        original_filename=next_break.get("source") or "calendar",
                        signed_url=None,
                        storage_path="",
                        similarity=1.0,
                    )
                )
            _append_circular_suggestions(parts, sources, doc_suggestions)
            answer_text = "\n\n".join(parts)
            _log_interaction(
                client=client,
                question=question,
                answer=answer_text,
                status_label="answered",
                sources=sources,
                similarity=None,
                latency_ms=int((time.perf_counter() - start_time) * 1000),
                model_name="deterministic-calendar",
            )
            return QAResponse(status="answered", answer=answer_text, sources=sources)

    structured_result, structured_intent = structured_lookup
    if structured_result:
        structured_answer, structured_sources, model_name = structured_result
        sources = _dedupe_sources(structured_sources[:])
        answer_text = structured_answer
        _log_interaction(
            client=client,
            question=question,
            answer=answer_text,
            status_label="answered",
            sources=sources,
            similarity=None,
            latency_ms=int((time.perf_counter() - start_time) * 1000),
            model_name=model_name,
        )
        return QAResponse(status="answered", answer=answer_text, sources=sources)

    if structured_intent:
        fallback = _structured_intent_fallback(question, intent_label=structured_intent)
        if fallback:
            answer_text, fallback_sources, status_label = fallback
            _log_interaction(
                client=client,
                question=question,
                answer=answer_text,
                status_label=status_label,
                sources=fallback_sources,
                similarity=None,
                latency_ms=int((time.perf_counter() - start_time) * 1000),
                model_name=status_label,
            )
            return QAResponse(status=status_label, answer=answer_text, sources=fallback_sources)

    if calendar_answer:
        calendar_text = calendar_answer.formatted_answer()
        calendar_source = SourceInfo(
            document_id=f"calendar:{calendar_answer.event_id}",
            title=calendar_answer.title,
            published_on=calendar_answer.start.isoformat(),
            original_filename=calendar_answer.source or "calendar",
            signed_url=None,
            storage_path="",
            similarity=calendar_answer.score,
        )
        parts = [f"Calendar: {calendar_text}"]
        sources = [calendar_source]
        _append_circular_suggestions(parts, sources, doc_suggestions)
        sources = _dedupe_sources(sources)
        answer_text = "\n\n".join(parts)
        _log_interaction(
            client=client,
            question=question,
            answer=answer_text,
            status_label="answered",
            sources=sources,
            similarity=None,
            latency_ms=int((time.perf_counter() - start_time) * 1000),
            model_name="calendar-resolver",
        )
        return QAResponse(status="answered", answer=answer_text, sources=sources)

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

    keyword_hits = _fetch_keyword_hits(client, question, settings.qa_max_chunks)
    if keyword_hits:
        keyed_hits: Dict[tuple[str, int], ChunkHit] = {
            (hit.document_id, hit.chunk_index): hit for hit in hits
        }
        for keyword_hit in keyword_hits:
            key = (keyword_hit.document_id, keyword_hit.chunk_index)
            existing = keyed_hits.get(key)
            if existing:
                if keyword_hit.similarity > existing.similarity:
                    existing.similarity = keyword_hit.similarity
                existing.score = max(existing.score, keyword_hit.score, keyword_hit.similarity)
                if len(keyword_hit.content) > len(existing.content):
                    existing.content = keyword_hit.content
            else:
                hits.append(keyword_hit)
                keyed_hits[key] = keyword_hit

    for entry in calendar_matches or []:
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
        doc_suggestions = _fetch_document_fuzzy(client, question, limit=3)
        if doc_suggestions:
            best = doc_suggestions[0]
            answer_text = (
                f"I couldn’t find a direct answer, but this circular might help: {best.title}."
            )
            _log_interaction(
                client=client,
                question=question,
                answer=answer_text,
                status_label="suggested",
                sources=doc_suggestions,
                similarity=None,
                latency_ms=int((time.perf_counter() - start_time) * 1000),
                model_name="fuzzy-doc-suggestion",
            )
            return QAResponse(status="suggested", answer=answer_text, sources=doc_suggestions)

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

    prompt = _build_prompt(question, hits, doc_meta, temporal_hints=temporal_hints)

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
