from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class TextChunk:
    index: int
    content: str
    page: Optional[int] = None
    heading: Optional[str] = None


def chunk_text(
    text: str,
    *,
    chunk_size: int = 800,
    overlap: int = 200,
) -> List[TextChunk]:
    words = text.split()
    if not words:
        return []

    chunks: list[TextChunk] = []
    step = max(chunk_size - overlap, 1)

    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        segment = " ".join(words[start:end])
        if segment.strip():
            chunks.append(TextChunk(index=len(chunks), content=segment))

        if end == len(words):
            break

    return chunks


def chunk_pages(
    page_texts: Iterable[str],
    *,
    chunk_size: int = 600,
    overlap: int = 120,
) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    chunk_index = 0
    for page_number, page_text in enumerate(page_texts, start=1):
        words = page_text.split()
        if not words:
            continue
        step = max(chunk_size - overlap, 1)
        for start in range(0, len(words), step):
            end = min(start + chunk_size, len(words))
            segment = " ".join(words[start:end])
            if not segment.strip():
                continue
            heading = None
            if start == 0:
                heading = segment.split("\n")[0][:120]
            chunks.append(
                TextChunk(
                    index=chunk_index,
                    content=segment,
                    page=page_number,
                    heading=heading,
                )
            )
            chunk_index += 1
            if end == len(words):
                break
    if not chunks:
        return chunk_text(" ".join(page_texts), chunk_size=chunk_size, overlap=overlap)
    return chunks
