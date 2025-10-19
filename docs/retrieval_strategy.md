# Retrieval & Answering Strategy

This overview explains how the FastAPI backend serves parent questions while ensuring responses cite the most recent circular. The language model provider is selectable via the `LLM_PROVIDER` environment variable (`openai`, `gemini`, `groq`, `anthropic`, or `cohere`).

## Retrieval flow

1. **Question intake** – `/qa` endpoint receives guardian question along with authenticated guardian/student context.
2. **Preprocessing** – Clean whitespace, expand abbreviations (planned), detect explicit date filters ("next week", "Diwali break") to use in vector metadata filtering.
3. **Vector search** – Query `document_chunks` via pgvector using cosine similarity. Include filters:
   - `grade_tags` contains guardian’s grade (`Grade 3`).
   - Optional `doc_type`/`event_tags` derived from user intent.
   - Order by similarity DESC, then `published_on` DESC to surface the newest relevant chunks.
4. **Re-ranking** – Keep top N chunks (e.g., 6). Sort by `published_on` DESC before composing the prompt. Attach metadata (title, original filename, published_on, storage_path) to each chunk.
5. **LLM prompt** – Provide the OpenAI model with structured context:
   ```
   You are the class assistant. Use only the provided circular excerpts. Always cite the document title and published_on date.
   <document id="1" title="C 081 - Clarification about Diwali Break" date="2025-09-10" path="...">
   ... chunk text ...
   </document>
   ```
6. **Post-processing** – Parse the OpenAI response; ensure citation markup `{title — dd MMM yyyy}` for each statement. If the response references unknown content, fallback to escalation.
7. **Escalation trigger** – If no chunk exceeds a similarity threshold (e.g., 0.75) or the OpenAI model replies "I don't know", call `send_email` to notify the class parent and return a polite escalation message to the guardian.

### Supabase function setup

Create the RPC used by the backend by running the SQL in `docs/retrieval_sql.sql` inside the Supabase SQL editor. The function `match_document_chunks` accepts the embedding, similarity threshold, maximum number of matches, and grade tag, returning the top chunks joined with document metadata. Grant execute permissions so both authenticated users and the service role can call it.

## Citation and honesty

- Responses must include explicit citations. Backend will append a verification section such as:
  `Sources: • C 081 - Clarification about Diwali Break (10 Sep 2025)`
- UI links to the Supabase Storage signed URL for each cited document.
- If multiple documents conflict, priority goes to the most recent `published_on`. Older references are still shown as secondary citations when relevant.

## Metrics capture

`qa_logs` records similarity score, model latency, escalation status, and `cited_document_ids`. This powers dashboards tracking unanswered questions and helps audit whether newer circulars override outdated ones.

## Duplicate protection

Since ingestion sets a unique `source_sha256`, re-uploading the same circular is a no-op. If a truly updated circular has the same title but different content, it will generate a new hash and supersede old content automatically thanks to the newer `published_on` date.
