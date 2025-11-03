# Ingestion Overhaul – Layout-Aware Chunking & Highlights

This note captures the design decisions for the next iteration of the ingestion
pipeline. The goal is to move away from uniform word-window chunks and
introduce layout-aware segments, highlight extraction, and dual embeddings so
retrieval can surface meaningfully scoped passages.

## Objectives

1. **Respect document structure** – preserve headings, lists, and tables so no
   chunk mixes unrelated contexts.
2. **Generate highlights** – capture 2–3 bullet summaries per circular during
   ingestion to power quick answers.
3. **Support multiple embeddings** – allow experimentation with open models
   (e.g., `bge-small-en`) alongside OpenAI embeddings without re-ingestion.
4. **Retain fallbacks** – continue supporting OCR and whole-document chunking
   when structural cues are missing.

## Chunking Strategy

- Parse per-page layout via `extract_pdf_layout` (existing helper) and derive a
  section tree:
  - Headings above a threshold font size open a new section.
  - Bullet/numbered lists form sub-sections.
  - Tables are stored separately (`document_tables`) with references from the
    parent section.
- Within each section, chunk by sentences (~180–220 words) with 15% overlap.
  Attach metadata:
  - `section_id`, `section_heading`, `page_start`, `page_end`
  - `chunk_role`: `"intro"`, `"body"`, `"action_list"`, `"table"`
  - `order_key`: preserves original ordering for prompt assembly.
- Fallback to existing `chunk_text` when no headings are detected.

## Highlights & Structured Data

- During section parsing, collect candidate highlight sentences. Priorities:
  1. Explicit dates/deadlines (`document_dates` table).
  2. Action verbs targeting parents/learners (`document_actions`).
  3. Contact details.
- Summaries:
  - Generate at most three bullet highlights using a heuristic scorer (date
    presence, action keywords) and optionally LLM polishing (`gpt-4o-mini`,
    max 120 tokens) behind a flag.
  - Store in new table `document_highlights`:
    ```
    document_id UUID
    highlight_index INT
    text TEXT
    importance REAL
    source_chunk_index INT
    ```
  - Expose as part of document metadata for retrieval prompts.

## Dual Embeddings

- Extend `document_chunks` with:
  - `embedding_openai vector(768)` (existing column renamed via migration).
  - `embedding_alt vector(768)` e.g., `bge-small-en`.
  - `embedding_model` ENUM/text to indicate active model.
- During ingestion:
  - Compute OpenAI embeddings if key present.
  - Optionally compute alt embeddings when `--alt-embeddings` flag is passed.
  - Store per-chunk tokens/cost estimates for monitoring (new columns
    `embedding_tokens_openai`, `embedding_tokens_alt`).

## Pipeline Integration

1. **Section extraction** – new module `ingestion/utils/sections.py` builds the
   section tree from page texts + layout.
2. **Chunk builder** – replaces `chunk_pages` with `build_chunks_from_sections`
   (feature flag `USE_STRUCTURED_CHUNKING` in settings to allow gradual rollout).
3. **Highlight writer** – after chunk insertion, insert rows into
   `document_highlights` and link to structured tables when applicable.
4. **Compatibility** – existing ingestion CLI gains options:
   - `--legacy-chunking` (fallback).
   - `--alt-embeddings` (compute secondary model).
   - `--highlight-llm` (enable LLM polish).

## Next Steps

1. Implement section detection + chunk builder utilities.
2. Define Supabase migrations for new columns/table (see `docs/migrations/document_highlights.sql` for the table definition).
3. Update ingestion flow to use the new builder under a feature flag.
4. Backfill existing documents and compare retrieval quality before fully
   switching.
