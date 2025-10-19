# Supabase Schema & Invitation Workflow

This document describes the database schema, row-level security (RLS), and data import flow for the Grade 3 knowledge base platform.

## Key Entities

### `students`
Stores the roster and one invitation code per child.

```sql
create table if not exists public.students (
  id uuid primary key default gen_random_uuid(),
  full_name text not null,
  grade text not null,
  section text,
  mother_name text,
  mother_phone text,
  father_name text,
  father_phone text,
  invitation_code_salt text not null,
  invitation_code_hash text not null,
  invitation_code_revealed boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create unique index students_full_name_grade_idx
  on public.students (lower(full_name), lower(grade), coalesce(lower(section), ''));
```

### `guardians`
Links Supabase auth users to students. Each student has exactly one active guardian record (representing the family account).

```sql
create table if not exists public.guardians (
  id uuid primary key default gen_random_uuid(),
  student_id uuid not null references public.students(id) on delete cascade,
  auth_user_id uuid not null, -- Supabase auth.users id
  email text not null,
  mother_name text,
  mother_phone text,
  father_name text,
  father_phone text,
  invited_at timestamptz not null default now(),
  registered_at timestamptz,
  last_login_at timestamptz,
  role text not null default 'guardian', -- future proofing
  constraint guardians_student_unique unique (student_id)
);
```

### `documents`
Metadata for uploaded circulars and calendar PDFs/images.

```sql
create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  original_filename text not null,
  storage_path text not null,
  doc_type text not null, -- 'circular' | 'calendar' | 'notice'
  event_tags text[] default '{}',
  grade_tags text[] default '{}',
  published_on date,
  source_sha256 text not null,
  file_size_bytes bigint,
  page_count integer,
  uploaded_by uuid references public.guardians(id), -- null for admin bot
  uploaded_at timestamptz not null default now(),
  status text not null default 'active'
);

create unique index documents_source_sha_idx on public.documents (source_sha256);
```

### `document_chunks`
Stores chunked text and embeddings for retrieval.

```sql
create extension if not exists vector with schema public;

create table if not exists public.document_chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents(id) on delete cascade,
  chunk_index integer not null,
  content text not null,
  published_on date,
  embedding vector(768) not null, -- sentence-transformers/all-MiniLM-L6-v2
  created_at timestamptz default now(),
  constraint document_chunks_unique unique (document_id, chunk_index)
);

create index document_chunks_embedding_idx
  on public.document_chunks using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);
```

### `calendar_events`
Stores structured events parsed from the yearly calendar so that holiday ranges and reopen dates can be queried programmatically.

```sql
create table if not exists public.calendar_events (
  id uuid primary key default gen_random_uuid(),
  event_date date not null,
  end_date date,
  title text not null,
  description text,
  audience text[] default '{general}',
  source text,
  created_at timestamptz not null default now(),
  slug text generated always as (md5(event_date::text || title)) stored,
  unique (slug)
);

create index calendar_events_event_date_idx
  on public.calendar_events (event_date);
```

#### Migration note

If you created tables before this update, run the following SQL to align columns:

```sql
alter table public.documents
  add column if not exists original_filename text,
  add column if not exists source_sha256 text,
  add column if not exists file_size_bytes bigint,
  add column if not exists page_count integer;

alter table public.document_chunks
  add column if not exists published_on date;

create unique index if not exists documents_source_sha_idx
  on public.documents (source_sha256);

alter table public.calendar_events
  enable row level security;

create policy if not exists "service role calendar" on public.calendar_events
  for all
  using (auth.role() = 'service_role')
  with check (auth.role() = 'service_role');

create policy if not exists "authenticated read calendar" on public.calendar_events
  for select
  using (auth.role() = 'authenticated' or auth.role() = 'anon');
```

### `document_ingest_jobs`
Tracks background processing of uploaded files.

```sql
create table if not exists public.document_ingest_jobs (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents(id) on delete cascade,
  status text not null default 'pending',
  error_message text,
  started_at timestamptz,
  finished_at timestamptz
);
```

### `qa_logs`
Analytics + audit trail for each guardian question.

```sql
create table if not exists public.qa_logs (
  id uuid primary key default gen_random_uuid(),
  guardian_id uuid references public.guardians(id),
  question text not null,
  answer text,
  status text not null default 'answered', -- 'answered' | 'escalated' | 'error'
  similarity_score numeric,
  llm_model text,
  response_latency_ms integer,
  cited_document_ids uuid[] default '{}',
  escalated_at timestamptz,
  created_at timestamptz not null default now()
);
```

### `manual_answers`
Allows the admin to resolve escalations and seed future responses.

```sql
create table if not exists public.manual_answers (
  id uuid primary key default gen_random_uuid(),
  qa_log_id uuid not null references public.qa_logs(id) on delete cascade,
  resolved_by uuid references public.guardians(id),
  answer text not null,
  created_at timestamptz not null default now()
);
```

## Row-Level Security (RLS)

Enable RLS on all tables and create policies:

- `students`: guardians can select only their own child (`exists` clause on guardians linking auth.uid). Admin role (`role = 'admin'`) can manage all rows.
- `guardians`: each auth user can select/update their own record. Admin can manage all.
- `documents`, `document_chunks`: read access for authenticated guardians; insert/update/delete restricted to admin. Supabase service role (used by backend/worker) bypasses RLS.
- `calendar_events`: read access for authenticated/anonymous parents; writes restricted to the service role or admin ingestion jobs.
- `document_ingest_jobs`: no direct guardian access; admin read; worker insert/update via service role.
- `qa_logs`: guardians can view their own history; admin view all; inserts handled by backend service role.
- `manual_answers`: admin/service role only.

Policies will be scripted after tables exist (see future migrations).

## Invitation Code Workflow

1. **Import**: Upload the Excel roster into `data/` and run the import script.
2. **Code generation**: For each student, generate a random 8-character alphanumeric invitation code (e.g., `F7K9-2QPL`). Create a random salt, compute `sha256(salt || code)`, and store both `invitation_code_salt` and `invitation_code_hash`.
3. **Distribution**: Export a CSV/PDF mapping of `Student Name → Invitation Code` for offline sharing with parents. The plaintext code is not stored after export (toggle `invitation_code_revealed` for audit).
4. **Signup flow**:
   - Guardian submits email, password, invitation code.
   - Backend validates code by hashing input and matching against `students`.
   - If unused, create guardian in Supabase Auth, insert row into `guardians`, and flip `invitation_code_revealed=true`.
   - Prevent duplicate signup by enforcing the unique constraint on `guardians.student_id`.
5. **Admin override**: Admin dashboard can reset a student’s invitation code (generates new hash and revokes existing guardian record if needed).

## Metrics & Analytics

- Use Supabase scheduled tasks or cron job in Render to aggregate daily metrics into a materialized view:
  - Daily question counts, escalations, average response latency.
  - Per-document view counts (derived from frontend events).
- Expose metrics through `/admin/metrics` API.

## Next Steps

- Translate the schema into Supabase migration SQL (stored in `backend/migrations`).
- Implement the roster import script (Python) to read `data/Grade 3 JBCN Ena class list.xlsx`.
- Create Supabase policies and test the full signup-to-query flow.
