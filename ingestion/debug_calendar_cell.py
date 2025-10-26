from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pdfplumber

from .calendar_events import (
    _assign_column,
    _assign_row,
    _build_column_bounds,
    _cluster_rows,
    MONTH_INDEX,
    WEEKDAYS,
)


def inspect_cell(pdf_path: Path, month: int, day: int) -> list[dict[str, Any]]:
    with pdfplumber.open(str(pdf_path)) as pdf:
        events = []
        for page in pdf.pages:
            words = page.extract_words(extra_attrs=["non_stroking_color"])
            rects = page.rects

            month_name = None
            year_value = None
            for word in words:
                if word["text"] in MONTH_INDEX and month_name is None:
                    month_name = word["text"]
                elif word["text"].isdigit() and len(word["text"]) == 4 and year_value is None:
                    year_value = int(word["text"])
                if month_name and year_value:
                    break

            if not month_name or MONTH_INDEX[month_name] != month:
                continue

            header_words = [word for word in words if word["text"] in WEEKDAYS]
            if len(header_words) < 7:
                continue

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
                continue

            for word in day_words:
                column = _assign_column((word["x0"] + word["x1"]) / 2, column_bounds)
                row_index = _assign_row(word["top"], rows)
                if row_index is None or column is None:
                    continue
                if int(word["text"]) == day:
                    row_box = rows[row_index]
                    cell_left = column_bounds[column]
                    cell_right = column_bounds[column + 1]
                    cell_top = row_box["top"]
                    cell_bottom = row_box["bottom"]
                    cell_words = []
                    for entry in words:
                        column_check = _assign_column((entry["x0"] + entry["x1"]) / 2, column_bounds)
                        row_check = _assign_row(entry["top"], rows)
                        if row_check == row_index and column_check == column:
                            cell_words.append(entry)
                    result = []
                    rect_info = []
                    for rect in rects:
                        fill = rect.get("non_stroking_color")
                        if fill is None:
                            continue
                        if isinstance(fill, (int, float)):
                            normalized = (round(float(fill), 3),) * 3
                        elif len(fill) >= 3:
                            normalized = tuple(round(float(c), 3) for c in fill[:3])
                        else:
                            normalized = None
                        if not normalized:
                            continue
                        if (
                            rect["x0"] <= cell_right
                            and rect["x1"] >= cell_left
                            and rect["top"] <= cell_bottom
                            and rect["bottom"] >= cell_top
                        ):
                            rect_info.append(
                                {
                                    "x0": round(rect["x0"], 2),
                                    "x1": round(rect["x1"], 2),
                                    "top": round(rect["top"], 2),
                                    "bottom": round(rect["bottom"], 2),
                                    "color": normalized,
                                }
                            )
                    for entry in sorted(cell_words, key=lambda item: (item["top"], item["x0"])):
                        if entry == word:
                            continue
                        color = entry.get("non_stroking_color")
                        color_values = color[0] if color else None
                        if isinstance(color_values, (list, tuple)):
                            color_values = [round(c, 3) for c in color_values[:3]]
                        result.append(
                            {
                                "text": entry["text"],
                                "color": color_values,
                                "top": round(entry["top"], 2),
                                "x0": round(entry["x0"], 2),
                            }
                        )
                    return {"words": result, "rects": rect_info}
    return {"words": [], "rects": []}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect calendar cell tokens.")
    parser.add_argument("--pdf", default="data/circulars/Parent_Yearly_Calendar_2025_26_22072025151110_14.pdf")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--day", type=int, required=True)
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    tokens = inspect_cell(pdf_path, args.month, args.day)
    print(json.dumps(tokens, indent=2))


if __name__ == "__main__":
    main()
