# Class Knowledge Base Platform

Centralized workspace for the Grade 3 JBCN Ena class knowledge base. The goal is to let guardians self-register, browse official circulars, ask natural-language questions, and trigger escalations to the class parent when answers are missing.

## Repository Layout

- `frontend/` – Next.js web app for guardians and admin dashboard.
- `backend/` – FastAPI service for authentication, retrieval, escalation, and analytics APIs.
- `ingestion/` – Background worker that extracts text from PDFs/images and stores embeddings.
- `data/` – Raw source documents such as student lists and school circulars.
- `docs/` – Specification notes, runbooks, and high-level design assets.

## Ingestion quick start

- Install requirements: `pip install -r ingestion/requirements.txt` (requires system Tesseract + Poppler for OCR fallbacks).
- Ensure `.env` contains Supabase credentials and `SUPABASE_STORAGE_BUCKET` if you customise the bucket name.
- Drop new circulars into `data/circulars/`; run `python -m ingestion.ingest_documents` to upload files, deduplicate via SHA-256, extract text, create embeddings, and populate Supabase.
- Parse the yearly calendar into structured events with `python -m ingestion.calendar_events`; this populates `calendar_events` so Q&A can reason about holiday ranges and reopen dates.

## Core Stack

- **Auth & DB:** Supabase (Postgres + pgvector + Auth).
- **LLM:** Configurable provider (default `openai` using `gpt-4o-mini`; optionally `gemini`, `groq`, `anthropic`, or `cohere`) with retrieval over Supabase embeddings.
- **Storage:** Supabase Storage for PDFs/screenshots.
- **Hosting:** Vercel (frontend) and Render (backend + worker) on free plans.
- **Email:** Gmail SMTP (app password) for escalation notifications.

## Immediate Milestones

1. Define Supabase schema (students, guardians, documents, embeddings, QA logs) with row-level security.
2. Create import routine to load the class list spreadsheet, generate invitation codes, and seed Supabase.
3. Stand up FastAPI backend with Supabase connectivity, Q&A retrieval, and email escalation.
4. Scaffold Next.js app with Supabase Auth integration and admin upload flow.

## Development Notes

- Default to Python for backend/worker code, TypeScript/React for the frontend.
- Keep documents under 100 MB total; prefer PDF or PNG/JPEG screenshots.
- Track system metrics (queries, escalations, top topics) via Supabase tables exposed in the admin UI.

See `docs/` for detailed architecture diagrams and operational runbooks as they are developed.
