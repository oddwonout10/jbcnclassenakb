# Ingestion Pipeline

This document explains how circulars and calendar PDFs/screenshots are processed and stored so that the Q&A system can prioritise the latest documents and cite sources accurately.

## Overview

1. **Source drop** – Admin copies PDFs/images into `data/circulars/`.
2. **Ingestion run** – `python -m ingestion.ingest_documents`:
   - Computes SHA-256 checksum to deduplicate uploads (`documents.source_sha256` unique index).
   - Extracts text (PDF text layer, or OCR via Tesseract for images/scanned PDFs).
   - Normalises whitespace and chunks text into ~800-word segments (200-word overlap).
   - Uses `sentence-transformers/all-MiniLM-L6-v2` to embed each chunk.
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
