from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import re
import logging
from openai import OpenAI

from .config import get_settings


_EMBED_MODEL_NAME = "text-embedding-3-small"
_EMBED_DIMENSION = 768
_OPENAI_CLIENT: OpenAI | None = None


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
    score: float = 0.0


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


def _keyword_variants(keyword: str) -> List[str]:
    variants = {keyword}
    if keyword.endswith("s") and len(keyword) > 3:
        variants.add(keyword[:-1])
    if keyword.endswith("es") and len(keyword) > 4:
        variants.add(keyword[:-2])
    if keyword.endswith("ed") and len(keyword) > 4:
        variants.add(keyword[:-2])
    if keyword.endswith("ing") and len(keyword) > 5:
        variants.add(keyword[:-3])
    if len(keyword) >= 7:
        variants.add(keyword[:5])
    return [variant for variant in variants if len(variant) >= 3]


def parse_date(value) -> dt.date | None:
    if not value:
        return None
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError:
        return None


def _keyword_document_matches(
    *,
    client,
    question: str,
    grade_tag: str,
    max_documents: int,
    chunks_per_document: int,
    question_embedding: np.ndarray,
) -> List[ChunkHit]:
    keywords = _extract_keywords(question)
    if not keywords:
        return []

    doc_query = (
        client.table("documents")
        .select(
            "id,title,original_filename,storage_path,published_on,grade_tags,event_tags"
        )
        .contains("grade_tags", [grade_tag])
    )

    expanded_terms: List[str] = []
    for kw in keywords:
        expanded_terms.extend(_keyword_variants(kw))
    unique_terms = []
    seen_terms = set()
    for term in expanded_terms:
        if term not in seen_terms:
            unique_terms.append(term)
            seen_terms.add(term)

    or_clauses = ",".join(f"title.ilike.%{term}%" for term in unique_terms)
    if or_clauses:
        doc_query = doc_query.or_(or_clauses)

    doc_resp = doc_query.limit(max_documents).execute()
    documents = doc_resp.data or []
    if not documents:
        return []

    hits: List[ChunkHit] = []

    for doc in documents:
        chunk_resp = (
            client.table("document_chunks")
            .select("document_id,chunk_index,content,published_on")
            .eq("document_id", doc["id"])
            .order("chunk_index")
            .limit(chunks_per_document)
            .execute()
        )
        chunk_rows = chunk_resp.data or []
        chunk_texts = [row["content"] for row in chunk_rows]
        embeddings = embed_texts(chunk_texts)
        for row, emb in zip(chunk_rows, embeddings):
            chunk_embedding = np.array(emb, dtype=float)
            similarity = float(np.dot(question_embedding, chunk_embedding))
            hit = ChunkHit(
                document_id=row["document_id"],
                chunk_index=row.get("chunk_index", 0),
                content=row["content"],
                similarity=similarity,
                document_title=doc.get("title", "Untitled"),
                original_filename=doc.get("original_filename", ""),
                published_on=parse_date(doc.get("published_on")),
                storage_path=doc.get("storage_path", ""),
            )
            recency = _compute_recency_boost(hit.published_on)
            hit.score = 0.85 * hit.similarity + 0.15 * recency
            hits.append(hit)
    return hits


def fetch_relevant_chunks(
    *,
    client,
    question: str,
    query_embedding: Iterable[float],
    max_chunks: int,
    similarity_threshold: float,
    grade_tag: str = "Grade 3",
) -> List[ChunkHit]:
    params = {
        "query_embedding": list(query_embedding),
        "match_threshold": similarity_threshold,
        "match_count": max_chunks,
        "grade_tag": grade_tag,
    }

    response = client.rpc("match_document_chunks", params).execute()
    rows = response.data or []

    q_embedding = np.array(list(query_embedding))

    hits_by_key: dict[tuple[str, int], ChunkHit] = {}
    for row in rows:
        key = (row["document_id"], row.get("chunk_index", 0))
        hit = ChunkHit(
            document_id=row["document_id"],
            chunk_index=row.get("chunk_index", 0),
            content=row["content"],
            similarity=row["similarity"],
            document_title=row.get("document_title", "Untitled"),
            original_filename=row.get("original_filename", ""),
            published_on=parse_date(row.get("document_published_on")),
            storage_path=row.get("storage_path", ""),
        )
        recency = _compute_recency_boost(hit.published_on)
        hit.score = 0.85 * hit.similarity + 0.15 * recency
        hits_by_key[key] = hit

    keyword_hits = _keyword_document_matches(
        client=client,
        question=question,
        grade_tag=grade_tag,
        max_documents=max_chunks,
        chunks_per_document=2,
        question_embedding=q_embedding,
    )

    for hit in keyword_hits:
        key = (hit.document_id, hit.chunk_index)
        existing = hits_by_key.get(key)
        if existing:
            # Keep the higher similarity score if keyword hit beats vector score
            if hit.similarity > existing.similarity:
                hits_by_key[key] = hit
        else:
            hits_by_key[key] = hit

    hits = list(hits_by_key.values())
    for hit in hits:
        recency = _compute_recency_boost(hit.published_on)
        hit.score = 0.85 * hit.similarity + 0.15 * recency
    hits.sort(
        key=lambda h: (h.score, h.similarity, h.published_on or dt.date.min),
        reverse=True,
    )
    return hits[:max_chunks]
