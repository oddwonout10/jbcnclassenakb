from __future__ import annotations

import datetime as dt
import sys
from typing import Any

import typer

from ..supabase_client import get_supabase_client


app = typer.Typer(
    help="Inspect recently ingested documents to confirm structured chunks and highlights."
)


def _format_date(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, dt.date):
        return value.isoformat()
    try:
        return dt.date.fromisoformat(str(value)).isoformat()
    except Exception:  # pragma: no cover - defensive
        return str(value)


@app.command()
def recent(
    limit: int = typer.Option(
        5,
        "--limit",
        "-n",
        help="Number of documents to inspect (most recently uploaded first).",
    ),
    show_highlights: int = typer.Option(
        3,
        "--show-highlights",
        "-h",
        help="Number of highlight bullets to display per document (0 to skip).",
    ),
    sample_chunks: int = typer.Option(
        2,
        "--sample-chunks",
        "-c",
        help="Number of chunk summaries to show per document (0 to skip).",
    ),
) -> None:
    """
    Display high-level stats for recently ingested circulars, including
    chunk counts, highlight availability, and structured metadata coverage.
    """
    client = get_supabase_client()

    doc_response = (
        client.table("documents")
        .select("id,title,published_on,uploaded_at")
        .order("uploaded_at", desc=True)
        .limit(limit)
        .execute()
    )
    documents = doc_response.data or []
    if not documents:
        typer.echo("No documents found in Supabase.")
        raise typer.Exit(code=1)

    for doc in documents:
        doc_id = doc["id"]
        chunk_response = (
            client.table("document_chunks")
            .select("chunk_index,section_heading,page_number", count="exact")
            .eq("document_id", doc_id)
            .order("chunk_index")
            .limit(max(sample_chunks, 10) if sample_chunks else 10)
            .execute()
        )
        chunk_rows = chunk_response.data or []
        total_chunks = chunk_response.count or len(chunk_rows)
        structured_chunks = sum(
            1
            for row in chunk_rows
            if row.get("section_heading") or row.get("page_number") is not None
        )

        highlight_response = (
            client.table("document_highlights")
            .select("highlight_index,text,importance", count="exact")
            .eq("document_id", doc_id)
            .order("highlight_index")
            .execute()
        )
        highlight_rows = highlight_response.data or []
        total_highlights = highlight_response.count or len(highlight_rows)

        typer.echo(
            f"- {doc.get('title') or 'Untitled'} "
            f"(published { _format_date(doc.get('published_on')) })"
        )
        typer.echo(
            f"  chunks: {total_chunks} "
            f"(structured metadata on at least {structured_chunks})"
        )
        typer.echo(f"  highlights: {total_highlights}")

        if show_highlights and highlight_rows:
            for row in highlight_rows[:show_highlights]:
                text = row.get("text", "").strip()
                if len(text) > 160:
                    text = text[:157].rsplit(" ", 1)[0] + "…"
                typer.echo(f"    • [{row.get('highlight_index')}] {text}")

        if sample_chunks and chunk_rows:
            for row in chunk_rows[:sample_chunks]:
                heading = row.get("section_heading") or "—"
                page = row.get("page_number")
                typer.echo(
                    f"    chunk #{row.get('chunk_index')} heading={heading!r} page={page}"
                )

        typer.echo()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1].lower() == "recent":
        sys.argv.pop(1)
    app()


if __name__ == "__main__":
    main()
