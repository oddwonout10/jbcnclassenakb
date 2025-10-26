from __future__ import annotations

import datetime as dt
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional

import pdfplumber

from ingestion.supabase_client import get_supabase_client

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_INDEX = {
    name: idx
    for idx, name in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}
PDF_RELATIVE_PATH = Path("data/circulars/Parent_Yearly_Calendar_2025_26_22072025151110_14.pdf")

PRIMARY_COLORS = {
    "pre_primary": (0.02, 0.02, 0.95),
    "primary": (0.0, 0.69, 0.314),
    "secondary": (0.439, 0.188, 0.627),
    "primary_secondary": (1.0, 0.0, 1.0),
    "whole_school": (0.945, 0.761, 0.196),
    "whole_school_holiday": (1.0, 0.0, 0.0),
}
COLOR_VARIANTS = {
    "pre_primary": [(0.122, 0.286, 0.49)],
    "primary_secondary": [(0.839, 0.0, 0.576), (0.6, 0.0, 1.0)],
}
COLOR_LABELS = {
    "pre_primary": "pre_primary",
    "primary": "primary",
    "secondary": "secondary",
    "primary_secondary": "primary_secondary",
    "whole_school": "whole_school",
    "whole_school_holiday": "whole_school_holiday",
}
AUDIENCE_DISPLAY = {
    "pre-primary": "Pre-primary",
    "primary": "Primary",
    "secondary": "Secondary",
    "primary_secondary": "Primary & Secondary",
    "whole_school": "Whole School",
    "holiday": "Holiday",
    "general": "",
}
SKIP_PHRASES = [
    "the events in this calendar",
    "the events in this",
    "calendar are tentative",
    "tentative and subject to change",
    "any changes will be informed",
    "informed to the parent community",
    "announcements in toddle",
]


class CalendarParseError(RuntimeError):
    pass


def _normalise_color(color: Optional[tuple]) -> Optional[tuple[float, float, float]]:
    if color is None:
        return None
    if isinstance(color, (int, float)):
        return (round(float(color), 3),) * 3
    if len(color) < 3:
        return None
    return tuple(round(component, 3) for component in color[:3])


def _color_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def _classify_color(color: Optional[tuple]) -> Optional[str]:
    rgb = _normalise_color(color)
    if not rgb:
        return None

    best_key: Optional[str] = None
    best_score = float("inf")
    for key, base in PRIMARY_COLORS.items():
        score = _color_distance(rgb, base)
        if score < best_score:
            best_score = score
            best_key = key

    if best_key and best_score < 0.45:
        return COLOR_LABELS[best_key]

    for key, variants in COLOR_VARIANTS.items():
        for variant in variants:
            score = _color_distance(rgb, variant)
            if score < 0.45:
                return COLOR_LABELS[key]

    return None


def _build_column_bounds(header_words: List[dict]) -> List[float]:
    ordered = sorted(header_words, key=lambda word: word["x0"])
    centers = [(word["x0"] + word["x1"]) / 2 for word in ordered]
    if len(centers) < 2:
        raise CalendarParseError("Unable to detect weekday headers for the calendar grid.")

    gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    average_gap = statistics.mean(gaps)

    bounds: List[float] = [centers[0] - average_gap / 2]
    for index in range(len(centers) - 1):
        bounds.append((centers[index] + centers[index + 1]) / 2)
    bounds.append(centers[-1] + average_gap / 2)
    return bounds


def _assign_column(x_position: float, column_bounds: List[float]) -> Optional[int]:
    for index in range(len(column_bounds) - 1):
        if column_bounds[index] <= x_position < column_bounds[index + 1]:
            return index
    return None


def _cluster_rows(day_words: List[dict]) -> List[dict]:
    unique_tops = sorted({round(word["top"], 1) for word in day_words})
    if not unique_tops:
        return []

    gaps = [
        unique_tops[i + 1] - unique_tops[i]
        for i in range(len(unique_tops) - 1)
        if unique_tops[i + 1] - unique_tops[i] > 5
    ]
    default_height = statistics.median(gaps) if gaps else 90

    rows: List[dict] = []
    for index, top in enumerate(unique_tops):
        bottom = unique_tops[index + 1] if index + 1 < len(unique_tops) else top + default_height
        rows.append({"top": top - 6, "bottom": bottom - 4, "center": top, "days": {}})
    return rows


def _assign_row(y_position: float, rows: List[dict]) -> Optional[int]:
    for index, row in enumerate(rows):
        if row["top"] <= y_position < row["bottom"]:
            return index
    return None


def _clean_text(parts: List[str]) -> str:
    text = " ".join(parts)
    replacements = {
        " ,": ",",
        " .": ".",
        " ;": ";",
        " :": ":",
        " !": "!",
        " ?": "?",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("( ", "(").replace(" )", ")")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def _should_skip(content: str) -> bool:
    lowered = content.lower()
    return any(phrase in lowered for phrase in SKIP_PHRASES)


def _format_audience_suffix(audience: List[str]) -> str:
    labels = []
    for entry in audience:
        display = AUDIENCE_DISPLAY.get(entry, entry.replace("_", " ").title())
        if display:
            labels.append(display)
    if not labels:
        return ""
    return " (" + " & ".join(labels) + ")"


def _audience_from_labels(labels: Iterable[str]) -> List[str]:
    audience_tags: set[str] = set()
    for label in labels:
        label = label.replace("-", "_") if label else label
        if label == "whole_school_holiday":
            audience_tags.update({"whole_school", "holiday"})
        elif label == "primary_secondary":
            audience_tags.update({"primary", "secondary"})
        elif label:
            audience_tags.add(label)
    return sorted(audience_tags) if audience_tags else ["general"]


def _find_cell_color(rects, left: float, right: float, top: float, bottom: float) -> Optional[str]:
    for rect in rects:
        fill = rect.get("non_stroking_color")
        if not fill:
            continue
        rgb = _normalise_color(fill)
        if not rgb:
            continue
        if rect["x0"] <= right and rect["x1"] >= left and rect["top"] <= bottom and rect["bottom"] >= top:
            label = _classify_color(rgb)
            if label:
                return label
    return None


def _parse_page(page, *, source_name: str) -> List[dict]:
    words = page.extract_words(extra_attrs=["non_stroking_color"])
    rects = page.rects

    month_name: Optional[str] = None
    year_value: Optional[int] = None
    for word in words:
        if not month_name and word["text"] in MONTH_INDEX:
            month_name = word["text"]
        elif not year_value and word["text"].isdigit() and len(word["text"]) == 4:
            year_value = int(word["text"])
        if month_name and year_value:
            break

    if not month_name or not year_value:
        return []

    header_words = [word for word in words if word["text"] in WEEKDAYS]
    if len(header_words) < 7:
        return []

    column_bounds = _build_column_bounds(header_words)
    day_words = [
        word
        for word in words
        if word["text"].isdigit()
        and 1 <= int(word["text"]) <= 31
        and len(word.get("non_stroking_color", ())) == 1
    ]
    rows = _cluster_rows(day_words)
    if not rows:
        return []

    for word in day_words:
        column = _assign_column((word["x0"] + word["x1"]) / 2, column_bounds)
        row = _assign_row(word["top"], rows)
        if row is None or column is None:
            continue
        rows[row]["days"][column] = int(word["text"])

    table_top = rows[0]["top"]
    table_bottom = rows[-1]["bottom"]

    cell_words: defaultdict[tuple[int, int], List[dict]] = defaultdict(list)
    for word in words:

        y_position = word["top"]
        if y_position < table_top or y_position > table_bottom:
            continue

        column = _assign_column((word["x0"] + word["x1"]) / 2, column_bounds)
        row = _assign_row(word["top"], rows)
        if row is None or column is None:
            continue
        cell_words[(row, column)].append(word)

    events: List[dict] = []
    month_number = MONTH_INDEX[month_name]
    for row_index, row in enumerate(rows):
        for column_index, day in row["days"].items():
            words_in_cell = sorted(
                cell_words.get((row_index, column_index), []),
                key=lambda item: (item["top"], item["x0"]),
            )
            if not words_in_cell:
                continue

            text_parts: List[str] = []
            audience_counts: defaultdict[str, int] = defaultdict(int)
            segments: List[tuple[str, List[str]]] = []
            current_label: Optional[str] = None
            current_tokens: List[str] = []
            cell_left = column_bounds[column_index]
            cell_right = column_bounds[column_index + 1]
            cell_top = row["top"]
            cell_bottom = row["bottom"]
            cell_label = _find_cell_color(rects, cell_left, cell_right, cell_top, cell_bottom)

            line_clusters: dict[float, List[dict]] = {}
            for entry in words_in_cell:
                text = entry["text"]
                if text.isdigit() and int(text) == day:
                    continue
                key = round(entry["top"], 1)
                line_clusters.setdefault(key, []).append(entry)

            for key in sorted(line_clusters):
                entries = sorted(line_clusters[key], key=lambda item: (item["top"], item["x0"]))
                tokens = [e["text"] for e in entries]
                content = _clean_text(tokens)
                if not content or _should_skip(content):
                    continue

                labels = []
                for entry in entries:
                    label = _classify_color(entry.get("non_stroking_color"))
                    if label:
                        labels.append(label)

                if not labels and cell_label:
                    labels.append(cell_label)

                audience_list = _audience_from_labels(labels)
                event_date = dt.date(year_value, month_number, day)
                title_with_suffix = content + _format_audience_suffix(audience_list)
                events.append(
                    {
                        "event_date": event_date,
                        "title": title_with_suffix,
                        "description": content,
                        "audience": audience_list,
                        "source": source_name,
                    }
                )
    return events


def _split_event(text: str, labels: list[str]) -> list[dict]:
    lowercase = text.lower()
    if "halloween celebrations" in lowercase and "election day" in lowercase:
        segments = [
            ("pre-primary", "Halloween Celebrations"),
            ("primary", "Halloween Celebrations"),
            ("primary_secondary", "Election Day"),
            ("primary_secondary", "Result of the Election"),
        ]

        results: List[dict] = []
        for label, title in segments:
            results.append(
                {
                    "title": title,
                    "description": title,
                    "audience": [label],
                }
            )
        return results
    return []


def parse_calendar(pdf_path: Path = PDF_RELATIVE_PATH) -> List[dict]:
    if not pdf_path.exists():
        raise CalendarParseError(f"Calendar PDF not found: {pdf_path}")

    events: List[dict] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            events.extend(_parse_page(page, source_name=pdf_path.name))
    return events


def _group_ranges(events: Iterable[dict]) -> List[dict]:
    grouped: dict[str, dict] = {}

    for event in events:
        title = event["title"]
        base = " ".join(title.strip().split()).lower()
        audiences = tuple(sorted(event.get("audience", []) or ["general"]))
        date_key = event["event_date"].isoformat()
        key = f"{base}::{date_key}::{','.join(audiences)}::{event['source']}"
        record = grouped.setdefault(
            key,
            {
                "title": title,
                "event_date": event["event_date"],
                "end_date": None,
                "audience": set(event.get("audience", []) or ["general"]),
                "source": event["source"],
            },
        )
        record["audience"].update(event.get("audience", []))

    results: List[dict] = []
    for record in grouped.values():
        results.append(
            {
                "title": record["title"],
                "event_date": record["event_date"],
                "end_date": record["end_date"],
                "audience": sorted(record["audience"]),
                "source": record["source"],
            }
        )
    return results


def upsert_events(events: List[dict]) -> None:
    client = get_supabase_client()
    payload: List[dict] = []

    sources_to_reset = {event.get("source") for event in events if event.get("source")}
    for source in sources_to_reset:
        client.table("calendar_events").delete().eq("source", source).execute()

    for event in events:
        payload.append(
            {
                "event_date": event["event_date"].isoformat(),
                "end_date": event["end_date"].isoformat() if event.get("end_date") else None,
                "title": event["title"],
                "description": event["title"],
                "audience": event.get("audience") or ["general"],
                "source": event.get("source"),
            }
        )

    if not payload:
        return

    chunk_size = 200
    for start in range(0, len(payload), chunk_size):
        chunk = payload[start : start + chunk_size]
        client.table("calendar_events").insert(chunk).execute()


def main() -> None:
    events = parse_calendar()
    grouped = _group_ranges(events)
    upsert_events(grouped)
    print(f"Upserted {len(grouped)} calendar events")


if __name__ == "__main__":
    main()
