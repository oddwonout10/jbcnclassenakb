from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class TextChunk:
    index: int
    content: str


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
