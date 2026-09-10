-- 007_fts_functions_schema_portable.sql — make the FTS trigger functions
-- resolve their tables in the DEPLOYMENT'S OWN schema instead of the literal
-- schema named `public` (master-harness-vn4rz.46).
--
-- ===========================================================================
-- THE DEFECT 005 INTRODUCED. Migration 005 fixed a real P1 (a restore drill
-- lost the whole origin tier because pg_dump sets an EMPTY search_path and the
-- unqualified `content` in the trigger bodies resolved to nothing). Its fix had
-- two layers: pin `SET search_path = pg_catalog, public` on each function, AND
-- hard-qualify the body's table as `public.content`. Both layers named the
-- schema `public` LITERALLY.
--
-- That is correct only for a deployment that lives in `public`. clawmem also
-- runs against a caller-chosen schema: src/pg/config.ts::setPgSchema (and the
-- CLAWMEM_PG_SCHEMA env var) point every pooled connection at another schema,
-- which is exactly how the integration suite gets a throwaway schema per pass.
-- Under that configuration the tables live in `<schema>` while the trigger body
-- still read `public.content` — a DIFFERENT table (or none at all). The
-- subselect returned no row, `coalesce(..., '')` swallowed it, and the weight-D
-- body branch contributed NOTHING to the stored tsvector.
--
-- The failure mode is the dangerous one: no error, no warning, a tsvector that
-- is present and well-formed and TITLE-ONLY. Body-term search silently returns
-- nothing. Caught by tests/integration/pg-write-path.test.ts ("maintains the
-- tsvector by trigger, weighted, and matches an explicitly-configured query"),
-- which expected /fox/ in documents.fts and got "'chronicl':2A 'marmalad':1A".
--
-- THE FIX. Keep the CVE-2018-1058 hardening; drop the pin to one literal schema
-- NAME:
--   1. `SET search_path = pg_catalog, :CLAWMEM_SCHEMA` — still a property of
--      the FUNCTION, so a pg_dump restore under an empty session search_path
--      still resolves (the 005 restore-drill bug stays fixed), but the schema
--      is the one this deployment actually uses.
--   2. The body goes back to UNQUALIFIED table names, which now resolve through
--      that pinned search_path. Hard-qualifying is what pinned the function to
--      `public`; there is nothing left to hard-qualify it TO that is correct in
--      every deployment.
--
-- `:CLAWMEM_SCHEMA` is substituted by src/pg/migrate.ts::loadMigrations from
-- config.ts::pgSchema() (default `public`), already validated as a plain
-- identifier and double-quoted. It is the same substitution mechanism that
-- already carries `:EMBED_DIM`. Because the checksum is taken over the
-- SUBSTITUTED text, pointing a deployment at a different schema is recorded as
-- the schema change it is.
--
-- PRODUCTION IS UNCHANGED. With no override, `:CLAWMEM_SCHEMA` is `"public"`
-- and these definitions are semantically identical to 005's.
--
-- 005 cannot be edited in place — src/pg/migrate.ts checksums applied
-- migrations — so the correction lives here, as 005 itself did for 001/004.
-- SUPERSEDES the three function bodies in 005. Trigger DEFINITIONS are
-- unchanged and are not re-created: CREATE OR REPLACE FUNCTION keeps every
-- existing trigger pointing at the new body.
--
-- NOT re-run over existing rows in `public`: those `fts` values were computed
-- by a session that COULD resolve `public.content`, so they are correct. In a
-- non-public schema there are no pre-existing rows to repair (the schema is
-- created by the migration run itself).
--
-- origin_documents_ensure_partition is deliberately NOT changed here: it builds
-- DDL by name and its `public.%I` qualification is a separate question from the
-- FTS bodies. Tracked separately rather than folded into this fix.
-- ===========================================================================

CREATE OR REPLACE FUNCTION documents_fts_refresh() RETURNS trigger AS $$
BEGIN
  NEW.fts :=
      setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A')
    || setweight(to_tsvector('english', coalesce(NEW.description, '')), 'B')
    || setweight(to_tsvector('english', coalesce(array_to_string(NEW.tags, ' '), '')), 'B')
    || setweight(to_tsvector('english',
         coalesce((SELECT doc FROM content WHERE content.hash = NEW.hash), '')), 'D');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SET search_path = pg_catalog, :CLAWMEM_SCHEMA;

CREATE OR REPLACE FUNCTION content_fts_cascade() RETURNS trigger AS $$
BEGIN
  IF NEW.doc IS DISTINCT FROM OLD.doc THEN
    UPDATE documents SET hash = hash WHERE hash = NEW.hash;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SET search_path = pg_catalog, :CLAWMEM_SCHEMA;

CREATE OR REPLACE FUNCTION origin_documents_fts_refresh() RETURNS trigger AS $$
BEGIN
  NEW.fts :=
      setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A')
    || setweight(to_tsvector('english', coalesce(NEW.description, '')), 'B')
    || setweight(to_tsvector('english', coalesce(array_to_string(NEW.tags, ' '), '')), 'B')
    || setweight(to_tsvector('english',
         coalesce((SELECT doc FROM content WHERE content.hash = NEW.hash), '')), 'D');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SET search_path = pg_catalog, :CLAWMEM_SCHEMA;
