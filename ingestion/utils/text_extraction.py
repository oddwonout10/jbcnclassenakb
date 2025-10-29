from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pdfplumber
from PIL import Image

try:
    import pytesseract
except ImportError:  # pragma: no cover - warn at runtime
    pytesseract = None

LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}


def extract_text(path: Path) -> tuple[str, Optional[int]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text, pages = _extract_text_from_pdf(path)
    elif suffix in IMAGE_EXTENSIONS:
        text, pages = _extract_text_from_image(path), 1
    elif suffix in TEXT_EXTENSIONS:
        text = path.read_text(encoding="utf-8")
        pages = 1
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    cleaned = _normalise_whitespace(text)
    return cleaned, pages


def _extract_text_from_pdf(path: Path) -> tuple[str, int]:
    with pdfplumber.open(path) as pdf:
        pages = len(pdf.pages)
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    if not text.strip():
        LOGGER.warning("No text extracted from %s. Attempting OCR fallback.", path)
        images = _convert_pdf_to_images(path)
        text = "\n".join(_extract_text_from_image(img) for img in images)
        pages = len(images)
    return text, pages


def _convert_pdf_to_images(path: Path) -> list[Image.Image]:
    try:
        from pdf2image import convert_from_path  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pdf2image is required for OCR fallback. Install poppler and pdf2image."
        ) from exc

    return convert_from_path(path)


def _extract_text_from_image(path_or_image) -> str:
    if pytesseract is None:
        raise RuntimeError(
            "pytesseract is required for image OCR. Install it with `pip install pytesseract` "
            "and ensure the Tesseract binary is available on your system."
        )
    if isinstance(path_or_image, Image.Image):
        image = path_or_image
    else:
        image = Image.open(path_or_image)
    return pytesseract.image_to_string(image)


WHITESPACE_REGEX = re.compile(r"[ \t]+")


def _normalise_whitespace(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = WHITESPACE_REGEX.sub(" ", raw_line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def extract_page_texts(path: Path) -> list[str]:
    """Return raw text per page for PDF files; otherwise treat the entire document as one page."""
    suffix = path.suffix.lower()
    if suffix != ".pdf":
        text, _ = extract_text(path)
        return [text]

    page_texts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text:
                page_texts.append(_normalise_whitespace(text))
            else:
                page_texts.append("")
    return page_texts


def extract_pdf_layout(path: Path) -> List[PageLayout]:
    layouts: List[PageLayout] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                headings = _extract_headings_from_page(page)
                raw_tables = page.extract_tables() or []
                tables_cleaned: List[List[List[Optional[str]]]] = []
                for table in raw_tables:
                    if not table:
                        continue
                    cleaned_rows = [
                        [cell.strip() if isinstance(cell, str) else cell for cell in row]
                        for row in table
                    ]
                    tables_cleaned.append(cleaned_rows)
                layouts.append(PageLayout(page_number=page_number, headings=headings, tables=tables_cleaned))
    except Exception as exc:  # pragma: no cover - pdf parsing may fail for corrupt files
        LOGGER.warning("Failed to extract layout information from %s: %s", path, exc)
    return layouts


def _extract_headings_from_page(page) -> List[str]:
    try:
        words = page.extract_words(extra_attrs=["size", "y0", "y1"]) or []
    except Exception:
        return []

    if not words:
        return []

    lines: dict[float, dict[str, list]] = {}
    for word in words:
        top = round(float(word.get("top", 0.0)), 1)
        entry = lines.setdefault(top, {"text": [], "sizes": []})
        entry["text"].append(word.get("text", ""))
        size = word.get("size")
        try:
            entry["sizes"].append(float(size))
        except (TypeError, ValueError):
            entry["sizes"].append(0.0)

    candidates: List[tuple[float, str]] = []
    for top, data in lines.items():
        if not data["text"]:
            continue
        text = " ".join(data["text"]).strip()
        if not text:
            continue
        avg_size = sum(data["sizes"]) / max(len(data["sizes"]), 1)
        candidates.append((avg_size, text))

    if not candidates:
        return []

    # Keep the top 5 largest font lines as headings
    candidates.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    headings: List[str] = []
    for _, text in candidates:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        headings.append(text[:200])
        if len(headings) >= 5:
            break
    return headings
@dataclass
class PageLayout:
    page_number: int
    headings: List[str]
    tables: List[List[List[Optional[str]]]]
