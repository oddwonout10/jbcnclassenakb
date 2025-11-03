# Class Knowledge Base Platform

Centralized workspace for the Grade 3 JBCN Ena class knowledge base. The goal is to let guardians self-register, browse official circulars, ask natural-language questions, and trigger escalations to the class parent when answers are missing.

## Repository Layout

- `frontend/` – Next.js web app for guardians and admin dashboard.
- `backend/` – FastAPI service for authentication, retrieval, escalation, and analytics APIs.
- `ingestion/` – Background worker that extracts text from PDFs/images and stores embeddings.
- `data/` – Development fixtures. Real circular PDFs/CSVs live in Supabase Storage; only synthetic samples should remain in git (`data/circulars_sample/`). See `docs/data_policy.md`.
- `docs/` – Specification notes, runbooks, and high-level design assets.

## Ingestion quick start

- Install requirements: `pip install -r ingestion/requirements.txt` (requires system Tesseract + Poppler for OCR fallbacks).
- Copy `.env.example` to `.env` and fill in Supabase credentials (anon + service role) along with email/LLM secrets before running any ingestion or backend services. Never commit your real `.env`.
- Drop new circulars into `data/circulars/`; run `python -m ingestion.ingest_documents` to upload files, deduplicate via SHA-256, extract text, create embeddings, and populate Supabase. Layout-aware chunking and highlight extraction run by default—use `INGESTION_USE_STRUCTURED_CHUNKING=false` or the `--no-structured` flag to fall back to legacy windowing if required.
- Parse the yearly calendar into structured events with `python -m ingestion.calendar_events`; this populates `calendar_events` so Q&A can reason about holiday ranges and reopen dates.
- Run the regression harness with `python backend/scripts/evaluate_baseline.py --backend-url http://127.0.0.1:8000 --limit 25 --expectations backend/evaluation/expectations_baseline.json`. The script now adds manual high-value questions (contact, deadlines) and exits non-zero if mismatches or request errors occur; pass `--no-fail` to collect metrics without failing CI.

## Core Stack

- **Auth & DB:** Supabase (Postgres + pgvector + Auth).
- **LLM:** Configurable provider (default `openai` using `gpt-4o-mini`; optionally `gemini`, `groq`, `anthropic`, or `cohere`) with retrieval over Supabase embeddings.
- **Storage:** Supabase Storage for PDFs/screenshots.
- **Hosting:** Vercel (frontend) and Render (backend + worker) on free plans.
- **Build tooling:** Frontend build pins Python 3.11 (see `frontend/runtime.txt` or `vercel.json`) so `sentence-transformers` installs binary wheels during Vercel builds.
- **Caching:** The backend summaries keep a small in-process cache (256 entries, 24-hour TTL). Bump `SUMMARY_CACHE_VERSION` in the deployment environment after re-ingesting circulars to force cache refreshes if needed.
- **Optional reranker:** Backend defaults to the lightweight keyword/rule fallback. To enable the cross-encoder reranker, install `sentence-transformers` separately and set `QA_ENABLE_RERANKER=true` in the environment.
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
- The backend can load an optional cross-encoder reranker (`sentence-transformers`) for better document ordering; if the dependency is missing it falls back to keyword heuristics.

See `docs/` for detailed architecture diagrams and operational runbooks as they are developed.
