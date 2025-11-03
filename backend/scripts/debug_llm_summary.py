#!/usr/bin/env python
"""Helper script to preview the per-request LLM summary for a circular."""

from __future__ import annotations

import argparse
import textwrap

from app.config import get_settings
from app.qa_routes import (
    SUMMARY_CACHE,
    SourceInfo,
    _fetch_chunks_for_document,
    _llm_summary_for_document,
    _summarise_snippet,
)
from app.supabase_client import get_supabase_client


def _fetch_document_row(client, document_id: str) -> dict | None:
    response = (
        client.table("documents")
        .select("id,title,published_on,original_filename,storage_path")
        .eq("id", document_id)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", required=True, help="UUID of the circular in the documents table.")
    parser.add_argument("--question", required=True, help="Question to steer the summary prompt.")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=4,
        help="Maximum number of chunks to include when prompting the LLM (default: 4).",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Print the chunk excerpts that will be sent to the LLM.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the cached summary for this process before running.",
    )
    args = parser.parse_args()

    settings = get_settings()
    client = get_supabase_client(service_role=True)

    row = _fetch_document_row(client, args.doc_id)
    if not row:
        raise SystemExit(f"Document {args.doc_id} was not found in Supabase.")

    source = SourceInfo(
        document_id=row["id"],
        title=row.get("title") or "Circular",
        published_on=row.get("published_on"),
        original_filename=row.get("original_filename") or "",
        signed_url=None,
        storage_path=row.get("storage_path") or "",
        similarity=1.0,
    )

    if args.clear_cache:
        SUMMARY_CACHE.clear()

    print(f"\nDocument: {source.title} ({source.document_id})")
    if source.published_on:
        print(f"Published on: {source.published_on}")
    print(f"Question: {args.question}\n")

    if args.show_chunks:
        chunk_hits = _fetch_chunks_for_document(client, source, limit=args.max_chunks)
        if not chunk_hits:
            print("No chunks found for this document.")
        else:
            print("Top excerpts:")
            for idx, chunk in enumerate(chunk_hits[: args.max_chunks], start=1):
                header_parts: list[str] = []
                if chunk.section_heading:
                    header_parts.append(chunk.section_heading)
                if chunk.page_number:
                    header_parts.append(f"page {chunk.page_number}")
                header = f"{idx}. {' — '.join(header_parts) if header_parts else 'Excerpt'}"
                print(header)
                print(textwrap.indent(textwrap.fill(chunk.content.strip(), width=90), prefix="   "))
                print()

    summary = _llm_summary_for_document(
        client=client,
        question=args.question,
        primary_source=source,
        settings=settings,
        max_chunks=args.max_chunks,
    )

    if summary:
        print("LLM summary:")
        print(textwrap.fill(summary, width=90))
    else:
        print("LLM summary: none (likely due to missing/blocked LLM provider).")
        snippet_row = client.table("document_chunks").select("content").eq("document_id", source.document_id).order(
            "chunk_index"
        ).limit(1).execute()
        snippet = snippet_row.data[0]["content"] if snippet_row.data else ""
        fallback = _summarise_snippet(snippet or "")
        if fallback:
            print("\nFallback snippet summary:")
            print(textwrap.fill(fallback, width=90))


if __name__ == "__main__":
    main()
