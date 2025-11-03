from __future__ import annotations

import itertools
import re
import uuid
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from .chunks import TextChunk
from .text_extraction import PageLayout

HEADING_MAX_LEN = 120
BULLET_PATTERN = re.compile(r"^\s*(?:[-*•]|[0-9]{1,2}[.)])\s+")
UPPER_HEADING_PATTERN = re.compile(r"^[A-Z0-9 ,./&'()]{4,}$")
DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?|"
    r"\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
ACTION_KEYWORDS = {
    "submit",
    "bring",
    "return",
    "pay",
    "complete",
    "attend",
    "collect",
    "register",
    "enrol",
    "enroll",
    "share",
    "fill",
    "deadline",
    "due",
    "visit",
    "meeting",
    "uniform",
    "kit",
    "logiqids",
    "competition",
    "trip",
}

TABLE_LINE_PATTERN = re.compile(r"^(?:table\s*:)?\s*[A-Z0-9\s\-\|:]+$", re.IGNORECASE)


@dataclass
class Section:
    id: str
    heading: Optional[str]
    level: int
    page_start: int
    page_end: int
    paragraphs: List[str] = field(default_factory=list)
    bullet_lists: List[List[str]] = field(default_factory=list)
    tables: List[List[List[Optional[str]]]] = field(default_factory=list)

    def append_paragraph(self, text: str) -> None:
        text = text.strip()
        if text:
            self.paragraphs.append(text)

    def append_bullet(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if not self.bullet_lists:
            self.bullet_lists.append([])
        if len(self.bullet_lists[-1]) > 0 and text == self.bullet_lists[-1][-1]:
            return
        self.bullet_lists[-1].append(text)

    def start_new_bullet_list(self) -> None:
        self.bullet_lists.append([])

    def iter_ordered_blocks(self) -> Iterable[Tuple[str, str]]:
        """
        Yield (role, text) for paragraphs, bullet lists, and table placeholders
        in the order they were added.
        """

        block_index = 0
        for paragraph in self.paragraphs:
            yield ("paragraph", paragraph)
            block_index += 1

        for bullet_list in self.bullet_lists:
            if not bullet_list:
                continue
            bullet_text = "\n".join(f"• {item}" for item in bullet_list)
            yield ("list", bullet_text)
            block_index += 1

        for table in self.tables:
            if not table:
                continue
            rows = [" | ".join(cell or "" for cell in row) for row in table]
            table_text = "Table:\n" + "\n".join(rows)
            yield ("table", table_text)
            block_index += 1

    def combined_text(self) -> str:
        parts = []
        for _, text in self.iter_ordered_blocks():
            parts.append(text)
        return "\n\n".join(parts)


@dataclass
class Highlight:
    text: str
    section_id: str
    score: float
    chunk_index: Optional[int] = None


def build_sections(
    page_texts: Sequence[str],
    layouts: Sequence[PageLayout] | None = None,
) -> List[Section]:
    sections: List[Section] = []
    current: Optional[Section] = None
    layout_map = {layout.page_number: layout for layout in (layouts or [])}
    heading_lookup = {
        (layout.page_number, heading.lower()): heading
        for layout in (layouts or [])
        for heading in layout.headings
    }

    def normalise(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def is_heading(line: str, page_number: int) -> bool:
        candidate = normalise(line)
        if not candidate:
            return False
        if len(candidate) > HEADING_MAX_LEN:
            return False
        if (page_number, candidate.lower()) in heading_lookup:
            return True
        return bool(UPPER_HEADING_PATTERN.match(candidate))

    section_counter = itertools.count(1)

    for page_index, raw_text in enumerate(page_texts, start=1):
        lines = raw_text.splitlines()
        page_layout = layout_map.get(page_index)

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if is_heading(stripped, page_index):
                heading_text = normalise(stripped)[:HEADING_MAX_LEN]
                section_id = f"sec-{page_index}-{next(section_counter)}"
                current = Section(
                    id=section_id,
                    heading=heading_text,
                    level=1,
                    page_start=page_index,
                    page_end=page_index,
                    tables=page_layout.tables if page_layout else [],
                )
                sections.append(current)
                continue

            if BULLET_PATTERN.match(stripped):
                if current is None:
                    section_id = f"sec-{page_index}-{next(section_counter)}"
                    current = Section(
                        id=section_id,
                        heading=None,
                        level=1,
                        page_start=page_index,
                        page_end=page_index,
                        tables=page_layout.tables if page_layout else [],
                    )
                    sections.append(current)
                if not current.bullet_lists:
                    current.start_new_bullet_list()
                current.append_bullet(BULLET_PATTERN.sub("", stripped, count=1))
                current.page_end = page_index
                continue

            if current is None:
                section_id = f"sec-{page_index}-{next(section_counter)}"
                current = Section(
                    id=section_id,
                    heading=None,
                    level=1,
                    page_start=page_index,
                    page_end=page_index,
                    tables=page_layout.tables if page_layout else [],
                )
                sections.append(current)

            current.append_paragraph(stripped)
            current.page_end = page_index

    return sections


def build_chunks_from_sections(
    sections: Sequence[Section],
    *,
    chunk_size: int = 220,
    overlap: int = 40,
) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    chunk_index = 0

    for order, section in enumerate(sections):
        combined = section.combined_text()
        if not combined.strip():
            continue

        words = combined.split()
        if not words:
            continue

        step = max(chunk_size - overlap, 1)
        for start in range(0, len(words), step):
            end = min(start + chunk_size, len(words))
            segment = " ".join(words[start:end]).strip()
            if not segment:
                continue
            chunk = TextChunk(
                index=chunk_index,
                content=segment,
                page=section.page_start,
                heading=section.heading,
                section_id=section.id,
                role="section",
                heading_level=section.level,
                page_end=section.page_end,
                order=order,
            )
            chunks.append(chunk)
            chunk_index += 1
            if end == len(words):
                break

    return chunks


def compute_highlight_score(sentence: str) -> float:
    score = 0.0
    if DATE_PATTERN.search(sentence):
        score += 1.5
    lowered = sentence.lower()
    if any(keyword in lowered for keyword in ACTION_KEYWORDS):
        score += 1.0
    score += min(len(sentence) / 120.0, 1.0) * 0.3
    return score


def is_table_like_text(text: str) -> bool:
    if not text:
        return True

    stripped = text.strip()
    if not stripped:
        return True

    pipe_count = stripped.count("|")
    total_chars = len(stripped)
    alpha_numeric = sum(1 for ch in stripped if ch.isalnum())

    if pipe_count >= 3 and alpha_numeric / max(total_chars, 1) < 0.35:
        return True

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return True

    if all(TABLE_LINE_PATTERN.match(line) for line in lines):
        return True

    return False


def extract_highlights(
    sections: Sequence[Section],
    *,
    max_highlights: int = 3,
) -> List[Highlight]:
    candidates: List[Highlight] = []

    for section in sections:
        text = section.combined_text()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            cleaned = sentence.strip()
            if not cleaned:
                continue
            score = compute_highlight_score(cleaned)
            if score <= 0:
                continue
            candidates.append(
                Highlight(
                    text=cleaned,
                    section_id=section.id,
                    score=score,
                )
            )

    # Deduplicate similar sentences
    unique: dict[str, Highlight] = {}
    for highlight in candidates:
        key = re.sub(r"\s+", " ", highlight.text.lower())
        if key not in unique or highlight.score > unique[key].score:
            unique[key] = highlight

    sorted_candidates = sorted(unique.values(), key=lambda h: h.score, reverse=True)
    return sorted_candidates[:max_highlights]


def extract_highlights_from_chunks(
    chunks: Sequence[TextChunk],
    *,
    max_highlights: int = 3,
) -> List[Highlight]:
    candidates: list[Highlight] = []

    for chunk in chunks:
        content = (chunk.content or "").strip()
        if not content:
            continue
        if is_table_like_text(content):
            continue
        sentences = re.split(r"(?<=[.!?])\s+", content)
        for sentence in sentences:
            cleaned = sentence.strip()
            if not cleaned:
                continue
            score = compute_highlight_score(cleaned)
            if score <= 0:
                continue
            candidates.append(
                Highlight(
                    text=cleaned,
                    section_id=chunk.section_id or f"chunk-{chunk.index}",
                    score=score,
                    chunk_index=chunk.index,
                )
            )

    unique: dict[str, Highlight] = {}
    for highlight in candidates:
        key = re.sub(r"\s+", " ", highlight.text.lower())
        existing = unique.get(key)
        if not existing or highlight.score > existing.score:
            unique[key] = highlight

    sorted_candidates = sorted(unique.values(), key=lambda h: h.score, reverse=True)
    return sorted_candidates[:max_highlights]
