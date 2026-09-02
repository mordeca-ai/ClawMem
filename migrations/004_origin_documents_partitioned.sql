-- 004_origin_documents_partitioned.sql — the RANGE-partitioned ORIGIN tier
-- (master-harness-vn4rz.8). ADR-0162 §5.
--
-- ===========================================================================
-- WHAT THIS IS
--
-- ADR-0162 §6 splits the cutover in two. The file-authored collections are
-- DERIVATIVE and are REINDEXED (`documents`, migration 001). The ORIGIN data —
-- monchujo session transcripts, telemetry — has no authoring substrate to
-- rebuild from and is COPIED, landing in the monthly RANGE-partitioned tables
-- this migration creates.
--
-- This is the ONE sanctioned copy in the arc. Everywhere else the rule is
-- reindex-not-copy, and this migration does not license generalising it.
-- ===========================================================================
--
-- ===========================================================================
-- DESIGN DECISION (a) — TYPE DISCIPLINE: origin rows do NOT share the
-- ADR-0058 `content_type` enum. They carry their own closed vocabulary.
--
-- ADR-0058's enum exists to drive ADR-0143's decay / half-life machinery over
-- CURATED MEMORY: every admitted value has a half-life, and the enum's job is
-- to say which curve a document decays on. Origin transcripts have no such
-- curve. They are append-only, DB-authoritative, and their lifecycle is
-- partition-drop retention (§5, and the RETENTION section below) — not decay.
--
-- Admitting `session-transcript` into ADR-0058 would therefore hand the decay
-- engine 13,829 rows for which no half-life policy exists or is wanted, and
-- would grow a memory-lifecycle enum with a value that is not a memory
-- lifecycle. That is exactly what the ADR-0058 amendment of 2026-09-02
-- (operator verdict, master-harness-vn4rz.35) DECLINED to do when it extended
-- the enum by four values and deliberately left `session-transcript` out,
-- writing: "It belongs to the `monchujo` origin corpus, which is owned by
-- master-harness-vn4rz.8 and gets its own modelling."
--
-- This migration is that modelling. `origin_documents.record_type` is a
-- SEPARATE closed set with the SAME DISCIPLINE — closed CHECK, raw value
-- preserved in `record_type_raw`, an explicit `unknown` sink, and the unknown
-- count monitored rather than shrugged at (ADR-0162 §3). The symmetry is
-- deliberate: a reader who knows the `documents` rules already knows these.
--
-- TRADE-OFF, STATED: a cross-tier query ("everything about X, transcripts
-- included") must now UNION two tables with two type vocabularies instead of
-- filtering one column. That cost is real and is accepted, because the
-- alternative — one enum spanning two lifecycles — makes the decay engine's
-- input set ill-defined, which is a correctness cost rather than an ergonomic
-- one.
--
-- ARCHITECTURAL COMMITMENT BEYOND THIS BEAD: yes. "The substrate carries two
-- disjoint type vocabularies, keyed by table family" is a fact ADR-0162 §5
-- IMPLIES (by giving origin data its own tables) but never STATES. It is named
-- here for an ADR amendment rather than decided silently:
--   ADR-0162 §5 amendment — origin tables carry their own closed `record_type`
--   vocabulary, disjoint from the ADR-0058 `content_type` enum.
-- ===========================================================================
--
-- ===========================================================================
-- DESIGN DECISION (b) — PARTITION KEY = `authored_at` (the transcript's own
-- session timestamp), monthly RANGE, half-open [month, month+1) in UTC.
--
-- ADR-0162 §5 literally names `PARTITION BY RANGE (ingested_at)`. This
-- migration DEVIATES from the column name and KEEPS the rationale, and the
-- reason is measured on this very bead's own data rather than argued:
--
--   The 13,829 monchujo transcripts span 2026-02 .. 2026-09 by session date and
--   are ALL ingested on one day (2026-09-02) by this migration. Under
--   `ingested_at` every one of them lands in a single `2026_09` partition.
--   Retention-by-partition-drop would then be able to drop ALL of history at
--   once or NONE of it — the partition key would carry no retention signal at
--   all, which defeats the mechanism §5 introduces partitioning FOR.
--
-- §5's stated reasons for `ingested_at` were (i) high cardinality and (ii)
-- never mutates after insert — "a low-cardinality key is the wrong partition
-- key and a mutating key forces cross-partition row movement"
-- (postgresql-16-administration-cookbook#35). `authored_at` satisfies BOTH
-- identically: a session record's `ts:` frontmatter is written once by the
-- capture process and never rewritten. So the deviation honours §5's reasoning
-- while the literal column would defeat §5's own retention story.
--
-- `ingested_at` is KEPT as an ordinary column (it is genuinely useful — it is
-- how you ask "what did the 2026-09-02 backfill actually load") and is simply
-- not the partition key.
--
-- BOUNDARY RULE: partition `origin_documents_YYYY_MM` holds
--   authored_at >= 'YYYY-MM-01 00:00:00+00' AND authored_at < 'YYYY-(MM+1)-01 00:00:00+00'
-- Half-open, UTC, no gaps, no overlaps. Month is derived from the timestamp
-- normalised to UTC, so a session at 2026-03-01T00:30:00+09:00 (= 2026-02-28T
-- 15:30Z) files under 2026_02. Stated because "which month" is a real question
-- with a wrong answer available.
--
-- MISSING / UNPARSEABLE TIMESTAMP: the row lands in the DEFAULT partition
-- `origin_documents_unassigned` (§5 mandates a DEFAULT partition). It is NOT a
-- catch-all nobody revisits, and three mechanisms make that true rather than
-- aspirational:
--
--   1. `authored_at_source` records HOW the timestamp was obtained
--      ('frontmatter-ts' | 'frontmatter-authored-at' | 'none'), so the
--      placement DECISION stays retrievable (ADR-0162 §3: "the classification
--      decision itself stays retrievable ... so the judgment does not disappear
--      into black-boxed infrastructure").
--   2. `tools/clawmem-pg-parity` PRINTS the default partition's row count on
--      EVERY run, green or red, as a named line. A reader of a green run sees
--      the number.
--   3. Retention REFUSES to drop the default partition (see below). It has no
--      upper time bound, so dropping it would be unbounded data loss dressed as
--      a retention step.
-- ===========================================================================
--
-- ===========================================================================
-- RETENTION — DETACH CONCURRENTLY then DROP, NEVER a DELETE sweep.
--
-- ADR-0162 §5, from `the-art-of-postgresql#30` + `sql-performance-explained#37`
-- (DELETE only marks tuples dead and defers index reclamation to vacuum) and
-- `mastering-postgresql-17#25` (concurrent detach exists to cut maintenance
-- locking on a live table). The ADR flags honestly that "retention = partition
-- drop" is OUR INFERENCE assembled from those rules, not a single cited rule.
--
-- The mechanism is IMPLEMENTED, not merely documented: `src/pg/origin.ts`
-- `dropPartitionsBefore()`, exposed as `bun src/pg/cli.ts origin-retention
-- --before YYYY-MM [--apply]`, default DRY RUN.
--
-- It is NOT a plpgsql function, and that is forced rather than chosen: ALTER
-- TABLE ... DETACH PARTITION ... CONCURRENTLY cannot run inside a transaction
-- block, and every plpgsql function body IS a transaction block. A function
-- here could only offer plain DETACH (ACCESS EXCLUSIVE on the parent), which is
-- the lock cost §5 cites the concurrent form to avoid. So the verb lives in the
-- CLI, outside a transaction, where the concurrent form is actually reachable.
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- origin_documents — the append-only origin tier.
--
-- Content bodies and vectors are NOT duplicated here. `hash` references the
-- SAME content-addressed `content` table migration 001 created, and
-- `content_vectors` already keys on that hash. One content store, one vector
-- store, two document tiers pointing into them. This is why moving 13,723
-- already-loaded monchujo rows out of `documents` costs ZERO re-embedding: the
-- vectors were never keyed on the document row.
--
-- Per ADR-0162 §5: "do not partition the embedding table without a measured
-- reason" — content_vectors stays unpartitioned, deliberately.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS origin_documents (
  id                bigint      GENERATED ALWAYS AS IDENTITY,
  collection        text        NOT NULL,
  path              text        NOT NULL,
  title             text        NOT NULL,
  hash              text        NOT NULL REFERENCES content(hash) ON DELETE CASCADE,

  -- THE PARTITION KEY. Nullable on purpose: a NULL routes to the DEFAULT
  -- partition, which is the named, counted, never-dropped home for a record
  -- whose own timestamp could not be read. See DESIGN DECISION (b).
  authored_at       timestamptz,
  -- HOW authored_at was obtained. Provenance of the partition placement.
  authored_at_source text       NOT NULL DEFAULT 'none'
                      CHECK (authored_at_source IN
                        ('frontmatter-ts','frontmatter-authored-at','none')),
  -- When THIS substrate first saw the row. Useful, and deliberately NOT the
  -- partition key -- see DESIGN DECISION (b).
  ingested_at       timestamptz NOT NULL DEFAULT now(),

  -- ---- origin type discipline (DESIGN DECISION (a)) -------------------------
  -- A CLOSED set that is NOT the ADR-0058 content_type enum. Raw preserved,
  -- explicit unknown sink, count monitored.
  record_type       text        NOT NULL DEFAULT 'unknown'
                      CHECK (record_type IN ('session-transcript','unknown')),
  record_type_raw   text,

  -- ---- session identity (the shape monchujo records actually carry) ---------
  session_id        text,
  chunk_index       integer,
  chunk_count       integer,
  source_path       text,
  cwd               text,
  git_branch        text,
  agent_version     text,

  -- ---- the ADR-0162 §3 facets, same columns and same CHECKs as `documents` --
  -- Conformed once and reused, never re-derived per table (§3,
  -- data-warehouse-toolkit#24/#25): two attributes are the same facet only if
  -- they share both label and domain of values. These do.
  domain            text        NOT NULL DEFAULT 'unknown',
  audience          text        NOT NULL DEFAULT 'unknown'
                      CHECK (audience IN ('operator','household','external','agent-internal','unknown')),
  trust_tier        text        NOT NULL DEFAULT 'unknown'
                      CHECK (trust_tier IN ('authored','distilled','ingested-verbatim','derived','unknown')),
  provenance        jsonb       NOT NULL DEFAULT '{}'::jsonb,
  sensitivity       text        NOT NULL DEFAULT 'unknown'
                      CHECK (sensitivity IN ('public','private','crypt','nsfw','unknown')),
  source_ref        text,

  active            boolean     NOT NULL DEFAULT true,
  description       text,
  tags              text[],

  fts               tsvector,

  -- A partitioned table's UNIQUE/PK must contain the partition key. The natural
  -- key is (collection, path); authored_at joins it by force of that rule.
  --
  -- CONSEQUENCE, STATED: if a record's `ts` ever changed, the upsert would miss
  -- and produce a SECOND row in a different partition rather than updating the
  -- first. That is tolerable here ONLY because immutability-after-write is the
  -- premise the whole origin tier rests on (see DESIGN DECISION (b)); it is not
  -- a property to assume for any future origin corpus that rewrites history.
  -- `tools/clawmem-pg-parity` would catch it as a count overshoot.
  -- UNIQUE ... NULLS NOT DISTINCT, not PRIMARY KEY, and that is FORCED rather
  -- than stylistic: a PRIMARY KEY implies NOT NULL on every column, which would
  -- make `authored_at` non-nullable and leave a record whose timestamp cannot be
  -- read with NOWHERE TO GO -- the load would abort instead of filing it in the
  -- DEFAULT partition DESIGN DECISION (b) requires. NULLS NOT DISTINCT (PG15+)
  -- restores the half of PK semantics that actually matters here: two rows with
  -- the same (collection, path) and no timestamp collide rather than silently
  -- duplicating in the default partition.
  CONSTRAINT origin_documents_key
    UNIQUE NULLS NOT DISTINCT (collection, path, authored_at)
) PARTITION BY RANGE (authored_at);

-- The DEFAULT partition. Created FIRST so a load can never fail for want of a
-- home; §5 requires it unconditionally.
CREATE TABLE IF NOT EXISTS origin_documents_unassigned
  PARTITION OF origin_documents DEFAULT;

-- Indexes declared on the PARENT propagate to every partition, existing and
-- future, so a new month cannot come up under-indexed.
CREATE INDEX IF NOT EXISTS origin_documents_collection_active_idx
  ON origin_documents (collection, active);
CREATE INDEX IF NOT EXISTS origin_documents_hash_idx    ON origin_documents (hash);
CREATE INDEX IF NOT EXISTS origin_documents_session_idx ON origin_documents (session_id);
CREATE INDEX IF NOT EXISTS origin_documents_fts_idx     ON origin_documents USING gin (fts);
CREATE INDEX IF NOT EXISTS origin_documents_title_trgm_idx
  ON origin_documents USING gin (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS origin_documents_record_type_idx
  ON origin_documents (record_type) WHERE active;

-- FTS maintained by trigger with the configuration NAMED at write time, exactly
-- as migration 001 does for `documents` (ADR-0162 §2 — a bare to_tsquery at the
-- query end against a named write end fails SILENTLY). Same weights: title A,
-- description/tags B, body D.
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
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS origin_documents_fts_trg ON origin_documents;
CREATE TRIGGER origin_documents_fts_trg
  BEFORE INSERT OR UPDATE OF title, description, tags, hash ON origin_documents
  FOR EACH ROW EXECUTE FUNCTION origin_documents_fts_refresh();

-- ---------------------------------------------------------------------------
-- Partition creation, as a REPRODUCIBLE function rather than hand-typed DDL.
--
-- Idempotent: a second call for the same month is a no-op returning the same
-- name. Callers are the loader (which ensures a month before writing into it)
-- and anything that pre-creates the upcoming month.
--
-- LOCKING NOTE, stated rather than discovered: attaching a new range partition
-- while a DEFAULT partition exists takes ACCESS EXCLUSIVE on the default and
-- SCANS it to prove no existing default row belongs in the new range. On this
-- corpus the default holds ~0 rows so the scan is free; on a default that has
-- accumulated, it is not. That is a further reason the default is a monitored
-- number rather than a dumping ground.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION origin_documents_ensure_partition(p_month date)
RETURNS text AS $$
DECLARE
  lo   date := date_trunc('month', p_month)::date;
  hi   date := (date_trunc('month', p_month) + interval '1 month')::date;
  name text := format('origin_documents_%s', to_char(lo, 'YYYY_MM'));
BEGIN
  IF to_regclass(format('public.%I', name)) IS NULL THEN
    -- THE BOUNDS ARE PINNED TO UTC, EXPLICITLY.
    --
    -- `FOR VALUES FROM ('2026-03-01') TO ('2026-04-01')` on a timestamptz key
    -- casts each bare date using the SERVER's TimeZone GUC, not UTC. On this box
    -- (America/Chicago) that produced bounds of 2026-03-01 00:00 -06 .. 2026-04-01
    -- 00:00 -05 -- a partition that is not a calendar month in ANY timezone,
    -- because a DST transition falls inside it, and that silently disagrees with
    -- the UTC boundary rule declared in DESIGN DECISION (b). Caught on the
    -- one-month smoke, which is what the smoke is for. Append the offset.
    EXECUTE format(
      'CREATE TABLE %I PARTITION OF origin_documents FOR VALUES FROM (%L) TO (%L)',
      name, lo::text || ' 00:00:00+00', hi::text || ' 00:00:00+00');
    -- ADR-0162 §5: autovacuum_vacuum_insert_threshold is set PER TABLE on the
    -- append-only tables, not left at the global default -- it is the single
    -- most relevant autovacuum parameter for an insert-only table
    -- (mastering-postgresql-17#11). A month partition is written once during
    -- backfill then appended to; 1000 inserts is the trigger for the
    -- visibility-map/index-maintenance pass that keeps index-only scans
    -- reachable.
    EXECUTE format(
      'ALTER TABLE %I SET (autovacuum_vacuum_insert_threshold = 1000, '
      'autovacuum_vacuum_insert_scale_factor = 0)', name);
  END IF;
  RETURN name;
END;
$$ LANGUAGE plpgsql;

-- Same knob on the DEFAULT partition, which the ensure function never touches.
ALTER TABLE origin_documents_unassigned
  SET (autovacuum_vacuum_insert_threshold = 1000, autovacuum_vacuum_insert_scale_factor = 0);

-- ---------------------------------------------------------------------------
-- origin_partitions — the retention-facing view. One row per partition with its
-- real bounds read from the CATALOG (pg_get_expr on relpartbound), never parsed
-- back out of the table name. A name is a label; the bound is the truth, and
-- retention that trusted the label would drop the wrong month the first time a
-- partition was created by hand.
--
-- `is_default` is what `dropPartitionsBefore()` refuses on.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW origin_partitions AS
SELECT
  c.relname                                              AS partition_name,
  pg_get_expr(c.relpartbound, c.oid)                     AS bound_expr,
  pg_get_expr(c.relpartbound, c.oid) = 'DEFAULT'         AS is_default,
  c.reltuples::bigint                                    AS approx_rows,
  pg_total_relation_size(c.oid)                          AS total_bytes
FROM pg_class p
JOIN pg_inherits i ON i.inhparent = p.oid
JOIN pg_class c    ON c.oid = i.inhrelid
WHERE p.relname = 'origin_documents'
ORDER BY c.relname;
