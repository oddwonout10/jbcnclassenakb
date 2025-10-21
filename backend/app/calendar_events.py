from __future__ import annotations

import datetime as dt
import re
from typing import Dict, List


KEYWORD_MAP = {
    "diwali": {"diwali"},
    "break": {"break", "holiday", "vacation"},
    "holiday": {"holiday", "break", "vacation"},
    "reopen": {"reopen", "reopens", "resume"},
    "midterm": {"mid", "mid-term", "midterm"},
    "winter": {"winter"},
}


def _extract_keywords(question: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", question.lower())
    keywords: set[str] = set()
    token_set = set(tokens)
    for alias, terms in KEYWORD_MAP.items():
        if token_set.intersection({term.replace("-", "") for term in terms}):
            keywords.update(terms)
    # Always include base tokens for direct search
    for token in token_set:
        if len(token) >= 4 or token.isdigit():
            keywords.add(token)
    return list(keywords)


def _fetch_for_keyword(client, keyword: str) -> List[dict]:
    pattern = f"%{keyword}%"
    try:
        response = client.table("calendar_events").select("*").ilike("title", pattern).execute()
        records = response.data or []
        response_desc = client.table("calendar_events").select("*").ilike("description", pattern).execute()
        if response_desc.data:
            records.extend(response_desc.data)
        return records
    except Exception:  # pragma: no cover - Supabase table missing
        return []


def _normalise_base(title: str) -> str:
    base = title.lower()
    base = re.sub(r"\b(begins?|starts?|ends?|finishes|reopens|resume?s)\b", "", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base or title.lower()


def _format_range(record: dict) -> str:
    start = dt.date.fromisoformat(record["event_date"])
    end = record.get("end_date")
    end_display = None
    if end:
        end_display = dt.date.fromisoformat(end)

    if end_display:
        return f"{start:%Y-%m-%d} to {end_display:%Y-%m-%d}"
    return f"{start:%Y-%m-%d}"


def fetch_calendar_context(client, question: str, grade: str | None = None) -> List[dict]:
    keywords = _extract_keywords(question)
    if not keywords:
        return []

    results: Dict[str, dict] = {}
    for keyword in keywords:
        if len(keyword) < 3:
            continue
        for row in _fetch_for_keyword(client, keyword):
            base = _normalise_base(row["title"])
            key = f"{base}:{row['source']}"
            record = results.setdefault(
                key,
                {
                    "base": base,
                    "title": row["title"],
                    "event_date": row["event_date"],
                    "end_date": row.get("end_date") or row.get("event_date"),
                    "audience": set(row.get("audience") or []),
                    "source": row.get("source"),
                },
            )
            # Update event start/end bounds
            try:
                existing_start = dt.date.fromisoformat(record["event_date"])
                candidate_start = dt.date.fromisoformat(row["event_date"])
                if candidate_start < existing_start:
                    record["event_date"] = row["event_date"]
            except Exception:
                record["event_date"] = row["event_date"]

            existing_end_raw = record.get("end_date") or record.get("event_date")
            try:
                existing_end = dt.date.fromisoformat(existing_end_raw)
            except Exception:
                existing_end = None

            candidate_end_value = row.get("end_date") or row.get("event_date")
            try:
                candidate_end = dt.date.fromisoformat(candidate_end_value)
            except Exception:
                candidate_end = None

            if candidate_end and (existing_end is None or candidate_end > existing_end):
                record["end_date"] = candidate_end_value
            record["audience"].update(row.get("audience") or [])

    audience_whitelist = {"whole_school", "general"}
    grade_lower = (grade or "").lower()
    if "primary" in grade_lower or "grade 3" in grade_lower:
        audience_whitelist.update({"primary"})

    summaries: List[dict] = []
    for record in results.values():
        if record["audience"] and not (record["audience"] & audience_whitelist):
            continue
        audience_label = ", ".join(sorted(record["audience"])) if record["audience"] else "general"
        summaries.append(
            {
                "title": record["title"],
                "summary": f"{record['title']} ({audience_label}) — {_format_range(record)}",
                "event_date": record["event_date"],
                "end_date": record.get("end_date"),
                "audience": sorted(record["audience"]),
                "source": record["source"],
            }
        )

    return summaries
