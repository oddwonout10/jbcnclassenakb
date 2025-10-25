-- Run this SQL in the Supabase SQL editor to enable vector search for the Q&A endpoint.
-- It assumes the `documents` and `document_chunks` tables already exist with pgvector.

create or replace function public.match_document_chunks(
  query_embedding vector(768),
  match_threshold double precision,
  match_count integer,
  grade_tag text default 'Grade 3'
)
returns table (
  document_id uuid,
  chunk_index integer,
  content text,
  similarity double precision,
  document_title text,
  document_published_on date,
  storage_path text,
  original_filename text
)
language plpgsql
security definer
set search_path = public
as $$
begin
  return query
  select
    dc.document_id,
    dc.chunk_index,
    dc.content,
    1 - (dc.embedding <=> query_embedding) as similarity,
    d.title as document_title,
    d.published_on as document_published_on,
    d.storage_path,
    d.original_filename
  from public.document_chunks dc
  join public.documents d on d.id = dc.document_id
  where
    (grade_tag is null or array_position(d.grade_tags, grade_tag) is not null)
    and 1 - (dc.embedding <=> query_embedding) >= match_threshold
  order by similarity desc, d.published_on desc nulls last
  limit match_count;
end;
$$;

grant execute on function public.match_document_chunks(vector(768), double precision, integer, text)
  to authenticated, service_role, anon;

-------------------------------------------------------------------------------
-- Typo tolerant fuzzy search helpers (pg_trgm + full-text search)
-------------------------------------------------------------------------------

create or replace function public.match_document_chunks_fuzzy(
  q text,
  limit_count int default 20
) returns table (
  document_id uuid,
  chunk_index int,
  content text,
  similarity real,
  published_on date,
  document_title text,
  original_filename text,
  storage_path text
)
language sql stable as $$
  select
    dc.document_id,
    dc.chunk_index,
    dc.content,
    greatest(
      word_similarity(q, dc.content),
      similarity(dc.content, q),
      word_similarity(q, d.title)
    ) as similarity,
    dc.published_on,
    d.title as document_title,
    d.original_filename,
    d.storage_path
  from public.document_chunks dc
  join public.documents d on d.id = dc.document_id
  where to_tsvector('english', dc.content) @@ plainto_tsquery('english', q)
     or dc.content % q
  order by similarity desc, dc.published_on desc
  limit limit_count;
$$;

create or replace function public.match_documents_fuzzy(
  q text,
  limit_count int default 20
) returns table (
  id uuid,
  title text,
  similarity real,
  published_on date,
  original_filename text,
  storage_path text
)
language sql stable as $$
  select
    d.id,
    d.title,
    greatest(
      word_similarity(q, d.title),
      similarity(d.title, q)
    ) as similarity,
    d.published_on,
    d.original_filename,
    d.storage_path
  from public.documents d
  where to_tsvector('english', d.title) @@ plainto_tsquery('english', q)
     or d.title % q
  order by similarity desc, d.published_on desc
  limit limit_count;
$$;

create or replace function public.match_calendar_events_fuzzy(
  q text,
  limit_count int default 20
) returns table (
  id text,
  title text,
  event_date date,
  end_date date,
  audience text[],
  description text,
  source text,
  similarity real
)
language sql stable as $$
  select
    ce.id::text,
    ce.title,
    ce.event_date,
    ce.end_date,
    ce.audience,
    ce.description,
    ce.source,
    greatest(
      word_similarity(q, ce.title),
      similarity(ce.title, q),
      word_similarity(q, coalesce(ce.description, ''))
    ) as similarity
  from public.calendar_events ce
  where to_tsvector('english', ce.title) @@ plainto_tsquery('english', q)
     or ce.title % q
     or (coalesce(ce.description, '') % q)
  order by similarity desc, coalesce(ce.event_date, now()) desc
  limit limit_count;
$$;
