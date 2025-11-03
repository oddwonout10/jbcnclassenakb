# Retrieval & Answering Strategy

This overview explains how the FastAPI backend serves parent questions while ensuring responses cite the most recent circular. The language model provider is selectable via the `LLM_PROVIDER` environment variable (`openai`, `gemini`, `groq`, `anthropic`, or `cohere`).

## Retrieval flow

1. **Question intake** – `/qa` endpoint receives guardian question along with authenticated guardian/student context.
2. **Preprocessing** – Clean whitespace, expand abbreviations (planned), detect explicit date filters ("next week", "Diwali break") to use in vector metadata filtering.
3. **Hybrid candidate search** – Combine three signals:
   - **Vector recall** via `match_document_chunks` (pgvector), pulling ~12 candidates per query.
   - **Lexical recall** via `match_document_chunks_fuzzy` (pg_trgm + BM25) to catch paraphrases and keyword-heavy questions.
   - **Structured shortcuts** from tables such as `document_actions`, `document_dates`, and `document_contacts` when intents are explicit.
   Each chunk stores both vector and lexical scores; a weighted blend (configurable via `QA_VECTOR_WEIGHT`, `QA_KEYWORD_WEIGHT`, `QA_RECENCY_WEIGHT`) plus a recency boost trims the pool down to `QA_MAX_CHUNKS`.
4. **Cross-encoder rerank** – When `sentence-transformers` is installed, the top ~24 passages are rescored with `BAAI/bge-reranker-base`. The reranker output is fused with the blended score; a keyword heuristic kicks in if the model is unavailable.
5. **LLM prompt** – Provide the OpenAI model with structured context:
   ```
   You are the class assistant. Use only the provided circular excerpts. Always cite the document title and published_on date.
   <document id="1" title="C 081 - Clarification about Diwali Break" date="2025-09-10" path="...">
   ... chunk text ...
   </document>
   ```
6. **Post-processing** – Parse the OpenAI response; ensure citation markup `{title — dd MMM yyyy}` for each statement. If the response references unknown content, fallback to escalation.
7. **Escalation trigger** – If no chunk exceeds a similarity threshold (e.g., 0.75) or the OpenAI model replies "I don't know", call `send_email` to notify the class parent and return a polite escalation message to the guardian.

## Deterministic intents

Structured tables are consulted before retrieval. When a confident match is found we bypass the LLM and return a templated response:

- `document_dates` → `structured-date-*` answers (deadlines, start/resume/end dates) with highlight bullets.
- `document_actions` → `structured-event` summaries for competitions/trips, citing the originating circular.
- `document_contacts` → `structured-contact` replies that list phone/email details verbatim.
- Ranking queries use the stored `document_highlights` to enumerate key points from the latest ranking circular.

Each deterministic reply cites the relevant circular, appends up to three highlights, and logs the tag (e.g., `structured-contact`) so it is auditable.

### Per-request summariser

When a structured answer references a circular, the backend first asks the LLM to rewrite the top few chunks (normally the first 2–4 passages) into two natural sentences that explicitly call out dates and required actions. The prompt tells the model to ignore table scaffolding and headings. Summaries are cached per `(document_id, published_on, question)` for 24 hours so repeat questions stay fast and inexpensive. If the LLM is unavailable or returns an empty string the code falls back to the deterministic highlights (after cleaning table artefacts) or, as a last resort, a trimmed snippet pulled from the first chunk.

### Supabase function setup

Create the RPC used by the backend by running the SQL in `docs/retrieval_sql.sql` inside the Supabase SQL editor. The function `match_document_chunks` accepts the embedding, similarity threshold, maximum number of matches, and grade tag, returning the top chunks joined with document metadata. Grant execute permissions so both authenticated users and the service role can call it.

## Citation and honesty

- Responses must include explicit citations. Backend will append a verification section such as:
  `Sources: • C 081 - Clarification about Diwali Break (10 Sep 2025)`
- UI links to the Supabase Storage signed URL for each cited document.
- If multiple documents conflict, priority goes to the most recent `published_on`. Older references are still shown as secondary citations when relevant.

## Metrics capture

`qa_logs` records similarity score, model latency, escalation status, and `cited_document_ids`. This powers dashboards tracking unanswered questions and helps audit whether newer circulars override outdated ones.

## Configuration knobs

- `QA_VECTOR_CANDIDATES`, `QA_KEYWORD_CANDIDATES` – control how many vector and lexical hits are fused before truncation.
- `QA_VECTOR_WEIGHT`, `QA_KEYWORD_WEIGHT`, `QA_RECENCY_WEIGHT` – adjust the contribution of similarity vs. document recency in the blended score.
- `QA_RERANKER_MODEL`, `QA_RERANKER_WEIGHT`, `QA_RERANKER_MAX_PASSAGES` – select the cross-encoder model and influence how aggressively its score nudges the final ordering.
- Set `QA_ENABLE_RERANKER=false` to fall back to the lightweight keyword heuristic (useful for environments without `sentence-transformers`).

## Duplicate protection

Since ingestion sets a unique `source_sha256`, re-uploading the same circular is a no-op. If a truly updated circular has the same title but different content, it will generate a new hash and supersede old content automatically thanks to the newer `published_on` date.
