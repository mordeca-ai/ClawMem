-- 008_fts_separator_normalization.sql — align PG FTS token boundaries with
-- sqlite FTS5 unicode61 by replacing punctuation runs with spaces before the
-- english text-search parser runs.
--
-- LOCK/TIME NOTE: the idempotent backfill updates every documents and
-- origin_documents row. It takes row locks, fires the existing FTS triggers,
-- rewrites each stored tsvector, and can be long-running on a large vault.

CREATE OR REPLACE FUNCTION documents_fts_refresh() RETURNS trigger AS $$
BEGIN
  NEW.fts :=
      setweight(to_tsvector('english', regexp_replace(coalesce(NEW.title, ''), '[^[:alnum:][:space:]]+', ' ', 'g')), 'A')
    || setweight(to_tsvector('english', regexp_replace(coalesce(NEW.description, ''), '[^[:alnum:][:space:]]+', ' ', 'g')), 'B')
    || setweight(to_tsvector('english', regexp_replace(coalesce(array_to_string(NEW.tags, ' '), ''), '[^[:alnum:][:space:]]+', ' ', 'g')), 'B')
    || setweight(to_tsvector('english', regexp_replace(
         coalesce((SELECT doc FROM content WHERE content.hash = NEW.hash), ''),
         '[^[:alnum:][:space:]]+', ' ', 'g')), 'D');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SET search_path = pg_catalog, :CLAWMEM_SCHEMA;

CREATE OR REPLACE FUNCTION origin_documents_fts_refresh() RETURNS trigger AS $$
BEGIN
  NEW.fts :=
      setweight(to_tsvector('english', regexp_replace(coalesce(NEW.title, ''), '[^[:alnum:][:space:]]+', ' ', 'g')), 'A')
    || setweight(to_tsvector('english', regexp_replace(coalesce(NEW.description, ''), '[^[:alnum:][:space:]]+', ' ', 'g')), 'B')
    || setweight(to_tsvector('english', regexp_replace(coalesce(array_to_string(NEW.tags, ' '), ''), '[^[:alnum:][:space:]]+', ' ', 'g')), 'B')
    || setweight(to_tsvector('english', regexp_replace(
         coalesce((SELECT doc FROM content WHERE content.hash = NEW.hash), ''),
         '[^[:alnum:][:space:]]+', ' ', 'g')), 'D');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SET search_path = pg_catalog, :CLAWMEM_SCHEMA;

-- Re-running these statements is safe: each pass deterministically recomputes
-- fts from the current source columns and content row.
UPDATE documents SET hash = hash;
UPDATE origin_documents SET hash = hash;
