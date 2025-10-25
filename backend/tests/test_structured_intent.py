from __future__ import annotations

from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app import qa_routes


def test_collect_schedule_terms_detects_uniform():
    terms = qa_routes._collect_schedule_terms("what uniform should we wear tomorrow?")
    assert "uniform" in terms


def test_infer_contact_role_transport():
    role = qa_routes._infer_contact_role("who do i contact for bus drop-off?".lower())
    assert role == "transport"


def test_detect_quick_link_type_cafeteria():
    link_type = qa_routes._detect_quick_link_type("Can you send the cafeteria menu please?".lower())
    assert link_type == "cafeteria_menu"


def test_answer_contains_explicit_date():
    assert qa_routes._answer_contains_explicit_date("Event is on 12 Oct 2025 at 10:00 am")
    assert not qa_routes._answer_contains_explicit_date("Schedule to be announced soon")
