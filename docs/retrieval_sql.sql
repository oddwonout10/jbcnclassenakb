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
