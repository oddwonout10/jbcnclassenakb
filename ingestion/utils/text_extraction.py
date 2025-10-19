from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

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
