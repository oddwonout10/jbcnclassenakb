from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class IntentLabel(str, Enum):
    GENERAL = "general"
    CALENDAR = "calendar"
    CONTACT = "contact"
    WEATHER = "unsupported_weather"
    WELLNESS = "unsupported_wellness"
    RANKING = "ranking"


@dataclass
class IntentResult:
    label: IntentLabel
    confidence: float
    reason: Optional[str] = None


_WEATHER_KEYWORDS = {
    "weather",
    "rain",
    "raining",
    "temperature",
    "humid",
    "humidity",
    "forecast",
    "umbrella",
}

_WELLNESS_KEYWORDS = {
    "feeling low",
    "feel low",
    "sad",
    "anxious",
    "anxiety",
    "depressed",
    "help me",
    "panic",
    "therapy",
    "counsellor",
    "counselor",
    "suicide",
    "hurt myself",
    "self harm",
    "self-harm",
}

_CALENDAR_KEYWORDS = {
    "holiday",
    "break",
    "vacation",
    "diwali",
    "christmas",
    "reopen",
    "resume",
    "starts",
    "ends",
    "when is school",
    "calendar",
}

_CONTACT_KEYWORDS = {
    "contact",
    "phone",
    "email",
    "reach",
    "pta",
    "teacher",
    "coordinator",
}

_RANKING_KEYWORDS = {
    "ranking",
    "rank",
    "ranked",
    "top school",
}


def _contains_any(text: str, keywords: set[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def classify_intent(question: str) -> IntentResult:
    normalized = re.sub(r"\s+", " ", question.strip().lower())
    if not normalized:
        return IntentResult(label=IntentLabel.GENERAL, confidence=0.0)

    if _contains_any(normalized, _WEATHER_KEYWORDS):
        return IntentResult(
            label=IntentLabel.WEATHER,
            confidence=0.95,
            reason="Weather/forecast keywords detected.",
        )

    if _contains_any(normalized, _WELLNESS_KEYWORDS):
        return IntentResult(
            label=IntentLabel.WELLNESS,
            confidence=0.9,
            reason="Emotional support keywords detected.",
        )

    if _contains_any(normalized, _CALENDAR_KEYWORDS):
        return IntentResult(
            label=IntentLabel.CALENDAR,
            confidence=0.6,
            reason="Calendar-related keywords detected.",
        )

    if _contains_any(normalized, _CONTACT_KEYWORDS):
        return IntentResult(
            label=IntentLabel.CONTACT,
            confidence=0.6,
            reason="Contact-related keywords detected.",
        )

    if _contains_any(normalized, _RANKING_KEYWORDS):
        return IntentResult(
            label=IntentLabel.RANKING,
            confidence=0.5,
            reason="Ranking keywords detected.",
        )

    return IntentResult(label=IntentLabel.GENERAL, confidence=0.3)
