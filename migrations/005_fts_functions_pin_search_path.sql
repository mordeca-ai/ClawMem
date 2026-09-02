-- 005_fts_functions_pin_search_path.sql — pin `search_path` on the three FTS
-- trigger functions (master-harness-vn4rz.8).
--
-- ===========================================================================
-- FOUND BY A RESTORE DRILL, NOT BY A TEST. The backup was a hypothesis until
-- somebody restored it (postgres-engineer hard gate 4), and the first restore
-- of the vn4rz.8 post-load dump FAILED:
--
--   pg_restore: error: COPY failed for table "origin_documents_2026_06":
--     ERROR:  relation "content" does not exist
--     LINE 6:  coalesce((SELECT doc FROM content WHERE content.has...
--     QUERY:  NEW.fts := ...
--
-- ...five times, once per partition, and the restored database came up with
-- content=17,020, documents=3,305, content_vectors=154,353 and
-- **origin_documents = 0**. A whole tier silently absent behind a non-zero exit
-- code that reads like a warning.
--
-- ROOT CAUSE: none of the three FTS trigger functions pinned `search_path`, and
-- `pg_dump` emits `SELECT pg_catalog.set_config('search_path', '', false)` at
-- the top of its output — the CVE-2018-1058 hardening. Under an EMPTY
-- search_path the unqualified `content` in the function body resolves to
-- nothing, so every row the trigger fires on fails. This is the
-- search_path-breach class the postgres canon rates P1, arriving as data loss
-- on restore rather than as a privilege escalation.
--
-- WHY IT SHOWS ON THE ORIGIN TIER FIRST: a partitioned table's data restores
-- through its PARTITIONS, as separate TOC entries handled by parallel workers
-- that each apply the empty search_path. `documents` happened to survive the
-- same dump. That is luck, not a difference in correctness — so all THREE
-- functions are fixed here, not just the origin one.
--
-- THE FIX, two layers, both deliberate:
--   1. `SET search_path = pg_catalog, public` on each function, so the
--      resolution context is a property of the FUNCTION and not of whatever
--      session happens to call it.
--   2. Schema-qualify `public.content` in the body anyway. Belt and braces: (1)
--      alone is enough today, but a function whose body reads correctly under
--      any search_path is one that cannot be broken by a later ALTER FUNCTION.
--
-- Migration 001 cannot be edited in place — src/pg/migrate.ts checksums applied
-- migrations and refuses a changed one, which is the guard that keeps
-- environments from silently diverging. So the correction lives here.
--
-- SUPERSEDES the function bodies at 001_core_schema.sql (documents_fts_refresh,
-- content_fts_cascade) and 004_origin_documents_partitioned.sql
-- (origin_documents_fts_refresh). The trigger DEFINITIONS are unchanged and are
-- not re-created: CREATE OR REPLACE FUNCTION keeps every existing trigger
-- pointing at the new body.
--
-- NOT re-run over existing rows: the stored `fts` values were computed by a
-- session that COULD resolve `content`, so they are correct. This migration
-- changes how the function resolves names, not what it computes.
-- ===========================================================================

CREATE OR REPLACE FUNCTION documents_fts_refresh() RETURNS trigger AS $$
BEGIN
  NEW.fts :=
      setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A')
    || setweight(to_tsvector('english', coalesce(NEW.description, '')), 'B')
    || setweight(to_tsvector('english', coalesce(array_to_string(NEW.tags, ' '), '')), 'B')
    || setweight(to_tsvector('english',
         coalesce((SELECT doc FROM public.content WHERE public.content.hash = NEW.hash), '')), 'D');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SET search_path = pg_catalog, public;

CREATE OR REPLACE FUNCTION content_fts_cascade() RETURNS trigger AS $$
BEGIN
  IF NEW.doc IS DISTINCT FROM OLD.doc THEN
    UPDATE public.documents SET hash = hash WHERE hash = NEW.hash;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SET search_path = pg_catalog, public;

CREATE OR REPLACE FUNCTION origin_documents_fts_refresh() RETURNS trigger AS $$
BEGIN
  NEW.fts :=
      setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A')
    || setweight(to_tsvector('english', coalesce(NEW.description, '')), 'B')
    || setweight(to_tsvector('english', coalesce(array_to_string(NEW.tags, ' '), '')), 'B')
    || setweight(to_tsvector('english',
         coalesce((SELECT doc FROM public.content WHERE public.content.hash = NEW.hash), '')), 'D');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SET search_path = pg_catalog, public;

-- The partition-creation function builds and executes DDL by name. Same class of
-- exposure, same fix -- `to_regclass`, `format` and `date_trunc` are all
-- pg_catalog, and the CREATE TABLE it emits must land in `public`.
CREATE OR REPLACE FUNCTION origin_documents_ensure_partition(p_month date)
RETURNS text AS $$
DECLARE
  lo   date := date_trunc('month', p_month)::date;
  hi   date := (date_trunc('month', p_month) + interval '1 month')::date;
  name text := format('origin_documents_%s', to_char(lo, 'YYYY_MM'));
BEGIN
  IF to_regclass(format('public.%I', name)) IS NULL THEN
    EXECUTE format(
      'CREATE TABLE public.%I PARTITION OF public.origin_documents FOR VALUES FROM (%L) TO (%L)',
      name, lo::text || ' 00:00:00+00', hi::text || ' 00:00:00+00');
    EXECUTE format(
      'ALTER TABLE public.%I SET (autovacuum_vacuum_insert_threshold = 1000, '
      'autovacuum_vacuum_insert_scale_factor = 0)', name);
  END IF;
  RETURN name;
END;
$$ LANGUAGE plpgsql SET search_path = pg_catalog, public;
