-- Creates the document_highlights table to store precomputed summary bullets
-- generated during ingestion. Run this in Supabase SQL editor or bundle it
-- with your migration workflow.

create table if not exists public.document_highlights (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents(id) on delete cascade,
  highlight_index integer not null,
  text text not null,
  importance double precision not null default 0,
  section_id text,
  chunk_index integer,
  created_at timestamp with time zone not null default timezone('utc', now())
);

create unique index if not exists document_highlights_document_idx
  on public.document_highlights(document_id, highlight_index);

create index if not exists document_highlights_document_id_idx
  on public.document_highlights(document_id);

grant select on public.document_highlights to authenticated, anon;
grant insert, update, delete on public.document_highlights to service_role;
