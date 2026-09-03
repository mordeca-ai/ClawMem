-- 006_telemetry_partitioned.sql — the RANGE-partitioned TELEMETRY tables
-- `context_usage` + `recall_events` (master-harness-a2t8r). ADR-0162 §5.
--
-- ===========================================================================
-- WHAT THIS IS
--
-- Migration 001 DELIBERATELY did not port `recall_events` or `context_usage`;
-- its header lists them among the dormant sqlite subsystems. They are not
-- dormant any more: `tools/clawmem-recall-hook` writes one `context_usage` row
-- plus N `recall_events` rows on every injecting prompt, fleet-wide, and that
-- is the corpus ADR-0162 §7's 7-day harvest reads. This migration creates the
-- Postgres home for that write path.
--
-- ---------------------------------------------------------------------------
-- SEQUENCING — WHY THIS LANDS INERT, AND WHAT THAT MEANS
--
-- A telemetry WRITER pointed at Postgres while every READER still reads the
-- sqlite vault splits the measurement across two stores mid-window, and
-- ADR-0162 §7 names a 7-day harvest ending ~2026-09-06 that such a split would
-- destroy. `master-harness-vn4rz.11` (the consumer flip) is still OPEN.
--
-- So the whole mechanism ships here — schema, tier-aware resolution, writer,
-- tests — with the PRODUCTION DEFAULT LEFT ON SQLITE:
-- `infra/clawmem/clawmem-recall.toml` `[telemetry] backend = "sqlite"`.
-- Production behaviour after this lands is BYTE-IDENTICAL to before it. These
-- tables exist and stay EMPTY until vn4rz.11 flips one config line.
--
-- That is not a hedge, it is the sequencing constraint satisfied by
-- construction: "lands WITH or AFTER vn4rz.11" is true of a mechanism whose
-- activation IS vn4rz.11. A zero row-count in these tables before that flip is
-- the CORRECT observation, not a bug — do not "fix" it by flipping the default
-- here.
-- ===========================================================================
--
-- ===========================================================================
-- DESIGN DECISION (a) — `recall_events` KEYS DOCUMENTS BY NATURAL TEXT KEY,
-- NOT BY AN INTEGER FK, AND NOT BY A TIER DISCRIMINATOR ON ONE.
--
-- In sqlite the column was:
--     doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE
-- Under ADR-0162 the document population lives in TWO tables — `documents`
-- (derivative tier, migration 001) and `origin_documents` (origin tier,
-- migration 004) — and a single integer FK cannot span them. Measured on this
-- cluster 2026-09-02: `origin_documents` holds 13,829 `monchujo` rows and
-- `documents` holds 0. The largest recalled collection is entirely on the tier
-- the old FK cannot reference.
--
-- REJECTED — "keep doc_id, add a tier discriminator column beside it". That is
-- sql-antipatterns R19/R20 POLYMORPHIC ASSOCIATIONS verbatim: no foreign key is
-- declarable at all (a column cannot reference two tables), so nothing in the
-- engine validates either the id or the discriminator, and the integrity claim
-- becomes application folklore. Two further facts make it not merely inelegant
-- but impossible and wrong here:
--
--   1. THERE IS NO KEY TO REFERENCE. `origin_documents.id` is
--      `GENERATED ALWAYS AS IDENTITY` but is NOT a primary key and carries no
--      unique constraint of its own — migration 004 explains why (a PRIMARY KEY
--      implies NOT NULL on every column, which would make `authored_at`
--      non-nullable and leave a timestamp-less record with nowhere to go). The
--      only unique constraint is
--      `UNIQUE NULLS NOT DISTINCT (collection, path, authored_at)`, and a
--      partitioned table's referenceable key MUST contain the partition key. So
--      `REFERENCES origin_documents(id)` is not a design one may choose; the
--      engine rejects it.
--
--   2. `ON DELETE CASCADE` DESTROYS THE MEASUREMENT. A reindex deactivates and
--      replaces document rows as a matter of routine. Under the sqlite FK that
--      silently DELETES the recall history attached to them — i.e. the ordinary
--      operation of the system erases the very observations this telemetry
--      exists to produce. Telemetry is an IMMUTABLE HISTORICAL OBSERVATION. It
--      must outlive the thing it observed; "the document is gone" is a fact the
--      telemetry should be able to state, not a reason to forget the recall.
--
-- CHOSEN — the natural key plus the tier the resolution actually matched:
--     doc_collection text NOT NULL
--     doc_path       text NOT NULL
--     doc_tier       text NOT NULL CHECK (doc_tier IN ('derivative','origin'))
-- `doc_tier` records WHICH TABLE MATCHED at write time. That is the same
-- "provenance of the decision stays retrievable" discipline
-- `origin_documents.authored_at_source` encodes (ADR-0162 §3): the classifying
-- judgment is stored as data rather than left to be re-derived later against a
-- population that has since moved.
--
-- LICENSED DENORMALIZATION (effective-sql R10), stated explicitly. WHAT IS
-- DUPLICATED: (collection, path) is copied out of the document row into every
-- recall event. WHY: it is a POINT-IN-TIME FACT, not a live reference — "at
-- 14:02 on 2026-09-02 this prompt surfaced monchujo/records/x.md". If the
-- document is later renamed, the recall event must keep saying what was
-- actually surfaced. A normalized reference would rewrite history on rename and
-- lose it entirely on delete. The duplication IS the correctness property.
--
-- `resolved_doc_id bigint` is KEPT, nullable, with NO FOREIGN KEY, and is
-- documented in the column comment as ADVISORY / NON-AUTHORITATIVE: it makes a
-- derivative-tier join cheap, and it gives the 111 legacy sqlite `recall_events`
-- rows a landing column if anyone ever carries them across. It must never be
-- read as a guarantee that the row still exists.
-- ===========================================================================
--
-- ===========================================================================
-- DESIGN DECISION (b) — BOTH TABLES ARE ORIGIN/TELEMETRY TIER: MONTHLY RANGE
-- PARTITIONS ON THEIR OWN EVENT TIMESTAMP.
--
-- ADR-0162 §5. `context_usage` partitions on `timestamp`, `recall_events` on
-- `surfaced_at`. Both satisfy §5's two stated requirements identically
-- (postgresql-16-administration-cookbook#35): high cardinality, and never
-- mutated after insert — these are append-only observation rows, written once
-- by the hook and thereafter only ever read. (`was_referenced` is the one
-- column a later back-fill touches; it is not the partition key, so no
-- cross-partition row movement is possible.)
--
-- BOUNDARY RULE, identical to migration 004: partition `<parent>_YYYY_MM` holds
--     key >= 'YYYY-MM-01 00:00:00+00' AND key < 'YYYY-(MM+1)-01 00:00:00+00'
-- Half-open, UTC, no gaps, no overlaps. THE `+00` OFFSET IS APPENDED
-- EXPLICITLY. A bare `FROM ('2026-03-01')` casts using the SERVER's TimeZone
-- GUC; on this box (America/Chicago) that produced bounds spanning a DST
-- transition — a "month" that is not a calendar month in any timezone.
-- Migration 004 caught that on its one-month smoke; this migration inherits the
-- fix rather than rediscovering the bug.
--
-- DEFAULT PARTITION: created FIRST, unconditionally, per §5 — before any range
-- partition exists, so a write can never fail for want of a home.
--
-- HONEST NOTE ON THE DEFAULT PARTITIONS HERE. Unlike `origin_documents`, whose
-- partition key is nullable BY DESIGN (a transcript whose `ts` cannot be read
-- has to go somewhere), both keys here are `NOT NULL` and are supplied by the
-- writer from `now()`. So these default partitions SHOULD stay permanently
-- empty. They are created anyway because §5 requires it unconditionally and
-- because the cost of being wrong is asymmetric: without a default, a write
-- for a month whose partition was never ensured ERRORS and loses the row.
--
-- A NON-ZERO COUNT IN EITHER DEFAULT IS A MONITORED NUMBER, NOT A SHRUG. It can
-- only mean the ensure-function was not called for that month (a writer bug) or
-- that a row was inserted by something other than the hook. Read it with:
--     SELECT count(*) FROM context_usage_unassigned;
--     SELECT count(*) FROM recall_events_unassigned;
-- and treat any non-zero as a defect to explain, exactly as
-- `tools/clawmem-pg-parity` treats `origin_documents_unassigned`.
-- ===========================================================================
--
-- ===========================================================================
-- DESIGN DECISION (c) — NO FOREIGN KEY BETWEEN `recall_events` AND
-- `context_usage`. `usage_id` IS A CORRELATION KEY.
--
-- The sqlite schema had no FK here either (`usage_id INTEGER`, unconstrained);
-- this migration KEEPS that and states the reasoning rather than inheriting it
-- by accident.
--
--   1. THE ORPHAN IS UNREACHABLE BY CONSTRUCTION. Both rows are written by ONE
--      writer inside ONE transaction (`tools/clawmem-recall-hook`, a single
--      `psql` invocation wrapping resolve + both inserts in BEGIN/COMMIT), and
--      the `recall_events` insert selects its `usage_id` from the
--      `context_usage` insert's own RETURNING. There is no code path that can
--      produce a recall event whose usage row was never written. The only way
--      to orphan one is ASYMMETRIC RETENTION — dropping a `context_usage` month
--      while keeping the matching `recall_events` month.
--
--   2. AND AN FK WOULD NOT HELP IN EXACTLY THAT CASE. `ALTER TABLE ... DETACH
--      PARTITION` / `DROP TABLE <partition>` does not fire referential-action
--      checks against rows in another partitioned table. So the constraint
--      would give FALSE ASSURANCE precisely where the only real hazard lives —
--      the worst property a constraint can have.
--
--   3. postgresql-up-and-running R14/R15: on a partitioned table, CHECK is the
--      constraint type that propagates to partitions automatically; UNIQUE/PK
--      must contain the partition key and FKs against partitioned tables carry
--      their own maintenance and locking costs. Leaning on FKs here fights the
--      partition model instead of using it.
--
-- WHAT ACTUALLY GUARANTEES THE PAIRING: both tables partition on THE SAME
-- INSTANT (the hook writes one timestamp into both), so a given turn's
-- `context_usage` row and its `recall_events` rows always land in the
-- same-named month partition of each table. A month lives and dies together.
-- Any retention policy MUST drop the same month from both — that requirement is
-- stated here because it is now a documented invariant rather than an enforced
-- one.
-- ===========================================================================
--
-- ===========================================================================
-- DESIGN DECISION (d) — ONE GENERALIZED ENSURE-PARTITION FUNCTION.
--
-- Migration 004 shipped `origin_documents_ensure_partition(p_month date)`, hard
-- wired to one parent. Two more monthly-RANGE tables arrive here, and three
-- copies of the same body — including three copies of the `+00` DST fix and
-- three copies of the autovacuum knobs — is three places for the next fix to
-- miss one. So this migration adds ONE function,
-- `clawmem_ensure_month_partition(p_parent text, p_month date)`, used by both
-- new tables. THIS IS A DELIBERATE GENERALIZATION, named as such.
--
-- WHAT IS *NOT* DONE, AND WHY: migration 004 is NOT edited to call it.
-- `migrations/README.md` documents that `src/pg/migrate.ts` `loadMigrations()`
-- checksums the whole substituted file text and REFUSES to run when an applied
-- migration's on-disk checksum changes — any byte, comments included. 004 is
-- applied. `origin_documents_ensure_partition` therefore remains, unchanged and
-- still the origin tier's entry point; the generalized function is additive.
-- Corrections and cross-references for applied migrations go in that README.
--
-- SAFETY: the parent name is validated with `to_regclass` and interpolated with
-- `%I` (identifier quoting) — never `%s`. The function pins `search_path`, for
-- the CVE-2018-1058 reason migration 005 documents at length: a function whose
-- resolution context depends on the caller's session is a function that breaks
-- under `pg_dump`'s empty search_path.
--
-- CAUGHT ON THE SMOKE (and this is what the scratch-database smoke is FOR).
-- The first draft pinned `SET search_path = pg_catalog, public`, copying 005's
-- ordering verbatim. 005's three functions only READ; this one CREATES, and an
-- unqualified `CREATE TABLE` targets THE FIRST SCHEMA IN search_path. So the
-- first call died with:
--     ERROR:  permission denied to create "pg_catalog.context_usage_2026_03"
--     DETAIL:  System catalog modifications are currently disallowed.
-- A pinned search_path is necessary but not sufficient for a function that
-- creates objects. THE FIX: schema-qualify `public.` on every created and
-- altered object inside the function body, so the creation target is a property
-- of the STATEMENT and not of the search_path ordering at all. The pin stays for
-- the resolution reason 005 gives.
--
-- LOCKING NOTE, stated rather than discovered (inherited from 004): attaching a
-- new range partition while a DEFAULT partition exists takes ACCESS EXCLUSIVE on
-- the default and SCANS it to prove no existing default row belongs in the new
-- range. Both defaults here are expected to hold zero rows, so the once-a-month
-- scan is free — which is a further reason a non-zero default count is a
-- monitored number (DESIGN DECISION (b)).
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- clawmem_ensure_month_partition — idempotent monthly-partition creation for
-- ANY monthly-RANGE-partitioned clawmem parent. A second call for the same
-- (parent, month) is a no-op returning the same name.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION clawmem_ensure_month_partition(p_parent text, p_month date)
RETURNS text
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $$
DECLARE
  lo   date := date_trunc('month', p_month)::date;
  hi   date := (date_trunc('month', p_month) + interval '1 month')::date;
  name text := format('%s_%s', p_parent, to_char(lo, 'YYYY_MM'));
BEGIN
  IF pg_catalog.to_regclass(format('public.%I', p_parent)) IS NULL THEN
    RAISE EXCEPTION 'clawmem_ensure_month_partition: no such parent table: %', p_parent;
  END IF;
  IF pg_catalog.to_regclass(format('public.%I', name)) IS NULL THEN
    -- BOUNDS PINNED TO UTC, EXPLICITLY. See DESIGN DECISION (b).
    -- EVERY object is SCHEMA-QUALIFIED `public.`, and that is a SMOKE FINDING,
    -- not a stylistic flourish. See CAUGHT ON THE SMOKE in DESIGN DECISION (d).
    EXECUTE format(
      'CREATE TABLE public.%I PARTITION OF public.%I FOR VALUES FROM (%L) TO (%L)',
      name, p_parent, lo::text || ' 00:00:00+00', hi::text || ' 00:00:00+00');
    -- ADR-0162 §5 / mastering-postgresql-17#11: these are append-only tables, so
    -- autovacuum_vacuum_insert_threshold is the single most relevant knob and is
    -- set PER PARTITION rather than left at the global default.
    EXECUTE format(
      'ALTER TABLE public.%I SET (autovacuum_vacuum_insert_threshold = 1000, '
      'autovacuum_vacuum_insert_scale_factor = 0)', name);
  END IF;
  RETURN name;
END;
$$;

-- ---------------------------------------------------------------------------
-- context_usage — one row per INJECTING prompt-turn.
--
-- Column shape is conformed to the sqlite original so the ADR-0162 §7 harvest
-- queries port unchanged, with two type corrections the sqlite type system
-- could not express: `was_referenced` is a real boolean (sqlite: INTEGER 0/1),
-- and `injected_paths` is `jsonb` (sqlite: a TEXT column holding JSON, which
-- nothing could validate or index).
--
-- `timestamp` keeps its sqlite name for query portability even though it
-- shadows a type name; every reference quotes it.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS context_usage (
  id                bigint      GENERATED ALWAYS AS IDENTITY,
  session_id        text,
  -- THE PARTITION KEY. NOT NULL: supplied by the writer from now(); unlike
  -- origin_documents.authored_at there is no "unreadable timestamp" case here.
  "timestamp"       timestamptz NOT NULL,
  hook_name         text        NOT NULL,
  injected_paths    jsonb       NOT NULL DEFAULT '[]'::jsonb,
  estimated_tokens  integer     NOT NULL DEFAULT 0,
  was_referenced    boolean     NOT NULL DEFAULT false,
  turn_index        integer     NOT NULL DEFAULT 0,
  query_text        text,

  -- A partitioned table's PK must contain the partition key. Both columns are
  -- NOT NULL here, so unlike origin_documents a real PRIMARY KEY is declarable
  -- and is used: `id` alone identifies the row for a human, and (id, "timestamp")
  -- is what the engine can enforce.
  CONSTRAINT context_usage_pkey PRIMARY KEY (id, "timestamp")
) PARTITION BY RANGE ("timestamp");

-- The DEFAULT partition, created FIRST and unconditionally (ADR-0162 §5).
-- Expected to stay EMPTY -- see DESIGN DECISION (b).
CREATE TABLE IF NOT EXISTS context_usage_unassigned
  PARTITION OF context_usage DEFAULT;

ALTER TABLE context_usage_unassigned
  SET (autovacuum_vacuum_insert_threshold = 1000, autovacuum_vacuum_insert_scale_factor = 0);

-- Declared on the PARENT so every future month partition inherits them and a
-- new month cannot come up under-indexed.
--
-- (session_id, hook_name) is the writer's own hot read: turn_index is
-- COUNT(*) WHERE session_id = ? AND hook_name = ?, run once per injecting
-- prompt. Equality-only predicate, both columns equality — column order is free
-- (sql-performance-explained#1 constrains equality-before-range, and there is no
-- range here); session_id leads because it is by far the more selective.
CREATE INDEX IF NOT EXISTS context_usage_session_hook_idx
  ON context_usage (session_id, hook_name);

COMMENT ON COLUMN context_usage.injected_paths IS
  'jsonb array of the displayPaths actually injected this turn (post curation gate, '
  'post threshold, post top_k trim). Point-in-time record, never a live reference.';

-- ---------------------------------------------------------------------------
-- recall_events — one row per DOCUMENT injected, per turn.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recall_events (
  id                bigint      GENERATED ALWAYS AS IDENTITY,

  -- ---- the document, as a NATURAL TEXT KEY (DESIGN DECISION (a)) ------------
  doc_collection    text        NOT NULL,
  doc_path          text        NOT NULL,
  doc_tier          text        NOT NULL
                      CHECK (doc_tier IN ('derivative','origin')),
  -- ADVISORY / NON-AUTHORITATIVE. No FK, deliberately. See DESIGN DECISION (a).
  resolved_doc_id   bigint,

  query_hash        text        NOT NULL,
  search_score      double precision NOT NULL,
  session_id        text        NOT NULL,
  -- CORRELATION KEY, NOT A FOREIGN KEY. See DESIGN DECISION (c).
  usage_id          bigint,
  turn_index        integer     NOT NULL DEFAULT 0,
  -- THE PARTITION KEY.
  surfaced_at       timestamptz NOT NULL,
  was_referenced    boolean     NOT NULL DEFAULT false,

  CONSTRAINT recall_events_pkey PRIMARY KEY (id, surfaced_at)
) PARTITION BY RANGE (surfaced_at);

CREATE TABLE IF NOT EXISTS recall_events_unassigned
  PARTITION OF recall_events DEFAULT;

ALTER TABLE recall_events_unassigned
  SET (autovacuum_vacuum_insert_threshold = 1000, autovacuum_vacuum_insert_scale_factor = 0);

-- Index set kept DELIBERATELY SMALL. This is the highest-write-rate table in the
-- substrate (one row per injected doc per prompt, fleet-wide) and every index is
-- paid on every insert; the reads are analytical and run once a week, not once a
-- prompt. The partition key needs no index of its own -- partition pruning is the
-- range access path.
--
--   usage_id                  -- the attribution join (recall-attribution.ts)
--   session_id                -- per-session harvest
--   (doc_collection, doc_path)-- "how often was this document recalled"
CREATE INDEX IF NOT EXISTS recall_events_usage_idx      ON recall_events (usage_id);
CREATE INDEX IF NOT EXISTS recall_events_session_idx    ON recall_events (session_id);
CREATE INDEX IF NOT EXISTS recall_events_doc_idx        ON recall_events (doc_collection, doc_path);

COMMENT ON COLUMN recall_events.doc_tier IS
  'WHICH TABLE the (collection, path) resolution matched at write time: '
  'derivative = public.documents, origin = public.origin_documents. Provenance of '
  'the resolution decision, stored as data (ADR-0162 section 3).';
COMMENT ON COLUMN recall_events.resolved_doc_id IS
  'ADVISORY ONLY. The id observed at write time. NOT a foreign key and NOT '
  'authoritative: the row it named may have been reindexed away, and for the origin '
  'tier origin_documents.id is not even a unique key. Join on '
  '(doc_collection, doc_path) when correctness matters.';
COMMENT ON COLUMN recall_events.usage_id IS
  'Correlation key to context_usage.id, written in the same transaction. NOT a '
  'foreign key -- see migration 006 DESIGN DECISION (c).';

-- ---------------------------------------------------------------------------
-- telemetry_partitions — the retention-facing view, the same shape migration
-- 004 established for origin_documents. Bounds are read from the CATALOG
-- (pg_get_expr on relpartbound), never parsed back out of the partition name: a
-- name is a label, the bound is the truth, and retention that trusted the label
-- would drop the wrong month the first time a partition was created by hand.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW telemetry_partitions AS
SELECT
  p.relname                                              AS parent_name,
  c.relname                                              AS partition_name,
  pg_get_expr(c.relpartbound, c.oid)                     AS bound_expr,
  pg_get_expr(c.relpartbound, c.oid) = 'DEFAULT'         AS is_default,
  c.reltuples::bigint                                    AS approx_rows,
  pg_total_relation_size(c.oid)                          AS total_bytes
FROM pg_class p
JOIN pg_inherits i ON i.inhparent = p.oid
JOIN pg_class c    ON c.oid = i.inhrelid
WHERE p.relname IN ('context_usage', 'recall_events')
ORDER BY p.relname, c.relname;
