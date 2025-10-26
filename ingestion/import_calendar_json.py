from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from .supabase_client import get_supabase_client


def _chunked(iterable: List[dict], size: int) -> Iterable[List[dict]]:
    for index in range(0, len(iterable), size):
        yield iterable[index : index + size]


def load_calendar(path: Path, truncate: bool = True) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Calendar JSON not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError("Calendar JSON must be a list of event objects")

    dedup: dict[tuple[str, str], dict] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("Each calendar entry must be an object")

        event_date = entry.get("event_date")
        if not event_date:
            raise ValueError("Calendar entry missing required 'event_date'")

        audience = entry.get("audience") or ["general"]
        if isinstance(audience, str):
            audience = [audience]

        title = entry.get("title", "Untitled")
        key = (event_date, title)
        dedup[key] = {
            "event_date": event_date,
            "end_date": entry.get("end_date"),
            "title": title,
            "description": entry.get("description") or title,
            "audience": audience,
            "source": entry.get("source", path.name),
        }

    rows = list(dedup.values())

    client = get_supabase_client()

    if truncate:
        client.table("calendar_events").delete().gte("event_date", "0001-01-01").execute()

    for batch in _chunked(rows, 500):
        client.table("calendar_events").upsert(batch, on_conflict="event_date,title").execute()

    print(f"Imported {len(rows)} calendar events from {path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import calendar events from JSON")
    parser.add_argument("--json", required=True, help="Path to calendar JSON file")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append instead of truncating existing calendar events",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    load_calendar(json_path, truncate=not args.append)


if __name__ == "__main__":
    main()
