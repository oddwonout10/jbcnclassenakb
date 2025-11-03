# Data Handling Policy

- Real circular PDFs, CSV exports, and spreadsheets **must not** live in this repository. They belong in Supabase Storage or a private bucket.
- `data/circulars_sample` contains synthetic placeholder files for local development. Replace them with sanitized copies when running ingestion tests.
- The ingestion pipeline expects files in `data/circulars` during development. Populate that directory from Supabase (or the classroom share) **locally** and ensure the directory is empty before committing.
- Sensitive spreadsheets (class list, contact numbers, etc.) should stay outside git. Use environment-specific storage with row-level security.

## Deployment checklist

1. Sync the latest circulars from Supabase before running `python -m ingestion.ingest_documents`.
2. Verify `.env` contains sanitized credentials (never commit real secrets).
3. After ingestion, remove local copies of sensitive PDFs or overwrite them with placeholder files.
4. Rotate secrets if any sensitive data is accidentally committed.
