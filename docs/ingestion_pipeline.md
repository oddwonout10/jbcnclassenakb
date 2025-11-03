# Ingestion Pipeline

This document explains how circulars and calendar PDFs/screenshots are processed and stored so that the Q&A system can prioritise the latest documents and cite sources accurately.

## Overview

1. **Source drop** – Admin copies PDFs/images into `data/circulars/`.
2. **Ingestion run** – `python -m ingestion.ingest_documents`:
   - Computes SHA-256 checksum to deduplicate uploads (`documents.source_sha256` unique index).
   - Extracts text (PDF text layer, or OCR via Tesseract for images/scanned PDFs).
   - Builds layout-aware sections (headings, bullet lists, tables) and creates ~220-word chunks with 40-word overlaps, falling back to page-based windowing when structure is absent.
   - Generates 2–3 highlights per circular from the section tree so deterministic answers can surface key takeaways.
   - Uses `sentence-transformers/all-MiniLM-L6-v2` (via OpenAI `text-embedding-3-small`) to embed each chunk.
   - Derives `published_on` date by parsing filenames and in-document dates (latest found wins; defaults to today if none).
   - Uploads binary file to Supabase Storage (`class-circulars` bucket) and writes metadata to `documents`.
   - Inserts chunks + embeddings into `document_chunks`, storing the same `published_on` so retrieval can rank by recency.

3. **Worker metadata** – `documents` stores filename, presentation title, checksum, file size, page count, and tags (future enhancement for event types). `document_chunks` keeps embeddings alongside the chunk text and `published_on`.

## Latest-document preference

During retrieval, the backend selects relevant chunks via vector similarity, then:

- Sorts candidate chunks by `published_on` (descending) before handing them to the LLM.
- Adds document metadata (title, original filename, `published_on`) to the prompt so the OpenAI model can cite the correct circular.
- Returns answers with `(title — dd MMM yyyy)` citations and includes a signed URL for verification.

If multiple circulars address the same topic, the newest chunk automatically outranks older ones because:

- Ingestion stores the most recent recognised date for the document.
- Retrieval phases prefer higher dates; ties fall back to vector similarity.

## Operational checks

- A “dry run” mode (future enhancement) will show which files would be ingested without writing to Supabase.
- Logs surface any documents where no date could be inferred; the admin can re-run with an explicit override (CLI flag `--date` planned).
- If OCR dependencies (Tesseract, Poppler) are missing, the script raises an actionable error.

## Manual overrides (optional)

For edge cases where automatic date parsing fails, create a JSON/YAML file with overrides (not yet implemented). Planned format:

```jsonc
{
  "C 020 (2025-26) - Time Table.pdf": {
    "published_on": "2025-07-15",
    "doc_type": "calendar",
    "event_tags": ["schedule"]
  }
}
```

When present, the ingestion script will top-up inferred metadata with explicit values, ensuring accurate recency sorting.

## Structured chunking & highlights

- Structured chunking is enabled by default. Disable it with
  `INGESTION_USE_STRUCTURED_CHUNKING=false` (or `--no-structured` on the CLI) if
  a particular run needs the legacy word-window behaviour.
- Optional tuning knobs:
  - `INGESTION_STRUCTURED_CHUNK_SIZE` (default 220 words)
  - `INGESTION_STRUCTURED_CHUNK_OVERLAP` (default 40 words)
  - CLI flags `--structured-chunk-size` and `--structured-chunk-overlap` override
    these settings for a single run.
- Apply the migration in `docs/migrations/document_highlights.sql` so the
  generated summaries can be stored in Supabase.

### Refreshing highlights after OCR improvements

Legacy circulars that were ingested before the table-cleanup improvements may carry noisy highlight rows (e.g., repeated `Table: | ... |`). Run the refresh command to recompute them directly from the stored chunks:

```bash
cd backend
PYTHONPATH=. python ingestion/ingest_documents.py refresh-highlights --limit 20
```

Key options:
- `--document-id <uuid>` – refresh a specific circular (useful for spot fixes).
- `--dry-run` – print existing and proposed bullets without writing to Supabase.
- `--max-highlights N` – cap the number of bullets per document (default 3).

The command skips calendar documents automatically and overwrites `document_highlights` for each processed circular.
