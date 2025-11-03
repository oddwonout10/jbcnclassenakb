from __future__ import annotations

from ingestion.utils.sections import (
    build_chunks_from_sections,
    build_sections,
    extract_highlights,
)
from ingestion.utils.text_extraction import PageLayout


def test_build_sections_splits_on_headings():
    page_texts = [
        "\n".join(
            [
                "IMPORTANT UPDATE",
                "Learner Kit Distribution",
                "- Bring water bottle and snacks",
                "- Wear sports uniform",
                "Submission deadline 15 Oct 2025.",
                "Contact: Mr. Rao (transport@school.com)",
            ]
        )
    ]
    layouts = [PageLayout(page_number=1, headings=["IMPORTANT UPDATE", "Learner Kit Distribution"], tables=[])]

    sections = build_sections(page_texts, layouts)
    assert len(sections) >= 2
    assert sections[0].heading == "IMPORTANT UPDATE"
    wear_found = any(
        "wear sports uniform" in bullet.lower()
        for bullet_list in sections[1].bullet_lists
        for bullet in bullet_list
    )
    assert wear_found


def test_build_chunks_and_highlights():
    page_texts = [
        "\n".join(
            [
                "EVENT DETAILS",
                "Educational Field Trip to Nashik",
                "- Arrive by 7:30 am at school gate.",
                "- Carry ID cards and water.",
                "The trip is scheduled on 18 Nov 2025.",
                "Payment due 25 Oct 2025.",
            ]
        )
    ]
    layouts = [
        PageLayout(
            page_number=1,
            headings=["EVENT DETAILS", "Educational Field Trip to Nashik"],
            tables=[],
        )
    ]

    sections = build_sections(page_texts, layouts)
    chunks = build_chunks_from_sections(sections, chunk_size=80, overlap=10)
    assert chunks, "Expected non-empty chunks"
    assert all(chunk.section_id for chunk in chunks)
    assert chunks[0].heading in {"EVENT DETAILS", "Educational Field Trip to Nashik"}

    highlights = extract_highlights(sections, max_highlights=2)
    assert highlights, "Expected highlights to be extracted"
    assert any("Nov 2025" in highlight.text or "18" in highlight.text for highlight in highlights)
