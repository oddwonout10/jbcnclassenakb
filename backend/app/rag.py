from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence

import re
import logging
from openai import OpenAI

try:  # pragma: no cover - optional dependency
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CrossEncoder = None  # type: ignore[assignment]

from .config import get_settings


_EMBED_MODEL_NAME = "text-embedding-3-small"
_EMBED_DIMENSION = 768
_OPENAI_CLIENT: OpenAI | None = None
_RERANKER_CACHE: tuple[str, CrossEncoder] | None = None  # type: ignore[type-arg]


logger = logging.getLogger(__name__)


def embed_text(text: str) -> List[float]:
    logger.info("Embedding single text (%d chars) first 120: %s", len(text), text[:120])
    embeddings = embed_texts([text])
    return embeddings[0] if embeddings else []


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    for idx, txt in enumerate(texts):
        logger.info(
            "Embedding batch item %d (%d chars) first 120: %s",
            idx + 1,
            len(txt),
            txt[:120],
        )
    client = _get_openai_client()
    response = client.embeddings.create(
        model=_EMBED_MODEL_NAME,
        input=texts,
        dimensions=_EMBED_DIMENSION,
    )
    return [item.embedding for item in response.data]


def _get_openai_client() -> OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OpenAI API key is not configured. Set OPENAI_API_KEY in the environment.")
        _OPENAI_CLIENT = OpenAI(api_key=settings.openai_api_key)
    return _OPENAI_CLIENT


def _get_reranker(model_name: str) -> CrossEncoder | None:  # type: ignore[name-defined]
    if CrossEncoder is None:  # pragma: no cover - optional dependency
        logger.debug("CrossEncoder dependency not installed; skipping reranking.")
        return None

    global _RERANKER_CACHE
    cached = _RERANKER_CACHE
    if cached and cached[0] == model_name:
        return cached[1]

    try:
        model = CrossEncoder(model_name, max_length=512)
    except Exception as exc:  # pragma: no cover - model load failures
        logger.warning("Failed to load reranker model %s: %s", model_name, exc)
        return None

    _RERANKER_CACHE = (model_name, model)
    logger.info("Loaded reranker model %s", model_name)
    return model


@dataclass
class ChunkHit:
    document_id: str
    chunk_index: int
    content: str
    similarity: float
    document_title: str
    original_filename: str
    published_on: dt.date | None
    storage_path: str
    page_number: int | None = None
    section_heading: str | None = None
    score: float = 0.0
    rerank_score: float | None = None
    vector_score: float | None = None
    keyword_score: float | None = None


STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "with",
    "about",
    "tell",
    "does",
    "what",
    "when",
    "where",
    "who",
    "how",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "in",
    "on",
    "at",
    "do",
    "does",
    "can",
    "you",
    "me",
}


def _compute_recency_boost(published_on: dt.date | None, window_days: int = 365) -> float:
    if not published_on:
        return 0.0
    today = dt.date.today()
    delta = (today - published_on).days
    if delta <= 0:
        return 1.0
    if delta >= window_days:
        return 0.0
    return 1.0 - (delta / window_days)


def _extract_keywords(question: str, limit: int = 5) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9']+", question.lower())
    keywords: List[str] = []
    for token in tokens:
        if len(token) < 3:
            continue
        if token in STOPWORDS:
            continue
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def parse_date(value) -> dt.date | None:
    if not value:
        return None
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError:
        return None


def _fetch_fuzzy_chunk_hits(
    *,
    client,
    question: str,
    limit: int,
) -> List[ChunkHit]:
    if limit <= 0:
        return []
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
        try:
            similarity = float(row.get("similarity") or 0.0)
        except (TypeError, ValueError):
            similarity = 0.0
        similarity = max(0.0, min(1.0, similarity))
        published_on = parse_date(row.get("published_on"))
        hits.append(
            ChunkHit(
                document_id=document_id,
                chunk_index=chunk_index,
                content=row.get("content") or "",
                similarity=similarity,
                document_title=row.get("document_title") or row.get("title") or "Untitled",
                original_filename=row.get("original_filename") or "",
                published_on=published_on,
                storage_path=row.get("storage_path") or "",
                score=similarity,
                keyword_score=similarity,
                page_number=row.get("page_number"),
                section_heading=row.get("section_heading"),
            )
        )
    return hits


def fetch_relevant_chunks(
    *,
    client,
    question: str,
    query_embedding: Iterable[float],
    max_chunks: int,
    similarity_threshold: float,
    grade_tag: str = "Grade 3",
    settings=None,
) -> List[ChunkHit]:
    config = settings or get_settings()
    vector_limit = max(max_chunks, config.qa_vector_candidates)
    params = {
        "query_embedding": list(query_embedding),
        "match_threshold": similarity_threshold,
        "match_count": vector_limit,
        "grade_tag": grade_tag,
    }

    response = client.rpc("match_document_chunks", params).execute()
    rows = response.data or []

    hits_by_key: dict[tuple[str, int], ChunkHit] = {}
    for row in rows:
        key = (row["document_id"], row.get("chunk_index", 0))
        vector_similarity = row.get("similarity") or 0.0
        try:
            vector_similarity = float(vector_similarity)
        except (TypeError, ValueError):
            vector_similarity = 0.0
        vector_similarity = max(0.0, min(1.0, vector_similarity))
        hit = ChunkHit(
            document_id=row["document_id"],
            chunk_index=row.get("chunk_index", 0),
            content=row["content"],
            similarity=vector_similarity,
            document_title=row.get("document_title", "Untitled"),
            original_filename=row.get("original_filename", ""),
            published_on=parse_date(row.get("document_published_on")),
            storage_path=row.get("storage_path", ""),
            page_number=row.get("page_number"),
            section_heading=row.get("section_heading"),
            vector_score=vector_similarity,
        )
        hits_by_key[key] = hit

    fuzzy_hits = _fetch_fuzzy_chunk_hits(
        client=client,
        question=question,
        limit=config.qa_keyword_candidates,
    )
    for fuzzy_hit in fuzzy_hits:
        key = (fuzzy_hit.document_id, fuzzy_hit.chunk_index)
        existing = hits_by_key.get(key)
        if existing:
            existing.keyword_score = max(existing.keyword_score or 0.0, fuzzy_hit.keyword_score or 0.0)
            if not existing.content or len(fuzzy_hit.content) > len(existing.content):
                existing.content = fuzzy_hit.content
            if (existing.similarity or 0.0) < (existing.keyword_score or 0.0):
                existing.similarity = existing.keyword_score or existing.similarity
        else:
            hits_by_key[key] = fuzzy_hit

    hits = list(hits_by_key.values())
    for hit in hits:
        recency = _compute_recency_boost(hit.published_on)
        vector_component = config.qa_vector_weight * (hit.vector_score or 0.0)
        lexical_component = config.qa_keyword_weight * (hit.keyword_score or 0.0)
        recency_component = config.qa_recency_weight * recency
        hit.score = vector_component + lexical_component + recency_component
        if not hit.similarity:
            hit.similarity = max(hit.vector_score or 0.0, hit.keyword_score or 0.0)
    hits.sort(
        key=lambda h: (h.score, h.similarity, h.published_on or dt.date.min),
        reverse=True,
    )
    return hits[:max_chunks]


def rerank_hits(
    question: str,
    hits: Sequence[ChunkHit],
    *,
    settings,
) -> tuple[List[ChunkHit], dict]:
    if not hits:
        return list(hits), {"applied": False}

    adjusted: List[ChunkHit] = [replace(hit) for hit in hits]
    reranker = _get_reranker(settings.reranker_model)
    rerank_weight = settings.reranker_weight
    max_passages = min(settings.reranker_max_passages, len(adjusted))
    metrics: dict = {"applied": False}

    scores: List[float] = []
    if reranker and max_passages:
        pairs = [[question, adjusted[idx].content] for idx in range(max_passages)]
        try:
            raw_scores = reranker.predict(pairs)  # type: ignore[call-arg]
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            scores = [float(score) for score in raw_scores]
        except Exception as exc:  # pragma: no cover - inference errors
            logger.warning("Reranker prediction failed: %s", exc)
            scores = []

    if scores:
        max_score = max(scores)
        min_score = min(scores)
        if max_score == min_score:
            normalized = [0.5] * len(scores)
        else:
            span = max_score - min_score
            normalized = [(score - min_score) / span for score in scores]

        for idx, norm_score in enumerate(normalized):
            hit = adjusted[idx]
            hit.rerank_score = hit.score + rerank_weight * norm_score

        for idx in range(len(normalized), len(adjusted)):
            hit = adjusted[idx]
            hit.rerank_score = hit.score

        metrics = {
            "applied": True,
            "model": settings.reranker_model,
            "passages_scored": len(scores),
        }
    else:
        keywords = set(_extract_keywords(question, limit=8))
        metrics = {
            "applied": False,
            "reason": "model_unavailable",
            "keywords": list(sorted(keywords)),
        }
        for idx, hit in enumerate(adjusted):
            keyword_hits = 0.0
            title_lower = hit.document_title.lower()
            content_lower = hit.content.lower()
            for kw in keywords:
                if kw in title_lower:
                    keyword_hits += 1.0
                elif kw in content_lower:
                    keyword_hits += 0.35
            position_penalty = 0.015 * idx
            hit.rerank_score = hit.score + rerank_weight * keyword_hits - position_penalty

    adjusted.sort(
        key=lambda hit: (hit.rerank_score or hit.score, hit.score),
        reverse=True,
    )
    metrics.setdefault("top_titles", [hit.document_title for hit in adjusted[:3]])
    logger.debug("Rerank metrics: %s", metrics)
    return adjusted, metrics
