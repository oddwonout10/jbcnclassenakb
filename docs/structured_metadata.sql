-- Structured metadata tables for enriched QA behaviour.
-- Run these statements in Supabase SQL editor before re-running ingestion.

alter table public.documents
    add column if not exists issued_on date,
    add column if not exists effective_start_on date,
    add column if not exists effective_end_on date,
    add column if not exists title_confidence double precision,
    add column if not exists audience text[] default array[]::text[];

alter table public.document_chunks
    add column if not exists page_number integer,
    add column if not exists section_heading text;

create index if not exists document_chunks_page_idx on public.document_chunks(document_id, page_number);

create table if not exists public.document_dates (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents(id) on delete cascade,
  date_type text not null,
  date_value date not null,
  raw_text text,
  confidence double precision default 0.5,
  created_at timestamptz not null default now()
);

create table if not exists public.document_actions (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents(id) on delete cascade,
  description text not null,
  due_date date,
  audience text,
  confidence double precision default 0.5,
  source_excerpt text,
  created_at timestamptz not null default now()
);

create table if not exists public.document_contacts (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents(id) on delete cascade,
  contact_name text,
  role text,
  email text,
  phone text,
  notes text,
  created_at timestamptz not null default now()
);

create table if not exists public.document_entities (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents(id) on delete cascade,
  entity_value text not null,
  entity_type text not null,
  confidence double precision default 0.5,
  context text,
  created_at timestamptz not null default now()
);

create table if not exists public.document_page_summaries (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents(id) on delete cascade,
  page_number integer not null,
  summary text not null,
  created_at timestamptz not null default now()
);

create table if not exists public.document_headings (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents(id) on delete cascade,
  page_number integer not null,
  heading_text text not null,
  heading_level integer,
  created_at timestamptz not null default now()
);

create table if not exists public.document_tables (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents(id) on delete cascade,
  page_number integer not null,
  table_data jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists document_dates_document_idx on public.document_dates(document_id);
create index if not exists document_actions_document_idx on public.document_actions(document_id);
create index if not exists document_contacts_document_idx on public.document_contacts(document_id);
create index if not exists document_entities_document_idx on public.document_entities(document_id);
create index if not exists document_page_summaries_document_idx on public.document_page_summaries(document_id);
create index if not exists document_headings_document_idx on public.document_headings(document_id);
create index if not exists document_tables_document_idx on public.document_tables(document_id);
