-- 003_content_type_enum_extend.sql — the ADR-0058 content_type enum EXTENDS
-- (master-harness-vn4rz.35).
--
-- ============================================================================
-- THIS MIGRATION SUPERSEDES THE COMMENT BLOCK AT 001_core_schema.sql:45-48.
--
-- That block says, verbatim:
--
--     "content_type is the CLOSED ADR-0058 enum (§5: 'gaps resolve by retag,
--      not extension')."
--
-- THAT IS NO LONGER TRUE, and 001 cannot be corrected in place: src/pg/migrate.ts
-- records a sha256 of each applied migration and REFUSES to re-apply an edited
-- one (that refusal is deliberate — it is what stops a schema silently diverging
-- between environments). So the correction lives here, in the migration that
-- actually performs the extension.
--
-- WHY: the ADR-0058 amendment of 2026-09-02 (operator verdict, BJ 2026-09-02,
-- Review Hub item 2940ea7c) REVERSES the ADR-0162 §5 "retag, not extension"
-- posture. The observed corpus does not have a retag backlog that a curator was
-- ever going to drain; it has (a) four genuinely missing types and (b) a set of
-- synonyms that mean an already-admitted type. The amendment therefore does two
-- things at once:
--
--   1. EXTENDS the admitted enum by four values:
--        eval-run, observation, plan, retro
--   2. CONFORMS ten observed synonyms onto one admitted value each (the map is
--      in the backfill below, and is mirrored by CONTENT_TYPE_CONFORM in
--      src/pg/write.ts; a test asserts the two never drift apart).
--
-- The admitted set is therefore 17 real types + the 'unknown' sink = 18 values.
--
-- WHAT IS DELIBERATELY *NOT* ADMITTED: `session-transcript`. It belongs to the
-- `monchujo` origin corpus, which is owned by master-harness-vn4rz.8 and gets
-- its own modelling. It must keep landing as 'unknown' with content_type_raw
-- set. Do not "helpfully" add it here or to the conform map.
--
-- LOCKING / VALIDATION — why plain ADD CONSTRAINT and not NOT VALID + VALIDATE:
--
-- The standard low-lock idiom for adding a CHECK to a live table is
-- `ADD CONSTRAINT ... NOT VALID` (catalog-only, brief lock) followed by a
-- separate `VALIDATE CONSTRAINT` (SHARE UPDATE EXCLUSIVE, concurrent with
-- DML). It does NOT earn its complexity here, for three independent reasons:
--
--   a. Replacing a CHECK requires DROP CONSTRAINT first, and DROP takes
--      ACCESS EXCLUSIVE unconditionally. The heavy lock is already paid; a
--      lighter ADD does not give it back.
--   b. The migration runner executes this file inside ONE transaction (see
--      src/pg/migrate.ts — the `-- clawmem:no-transaction` directive is for
--      CREATE INDEX CONCURRENTLY, which this file does not use, so the
--      directive is deliberately ABSENT). A VALIDATE issued inside that same
--      transaction runs while the transaction still holds the ACCESS EXCLUSIVE
--      lock from the DROP, so the weaker lock class is unobservable.
--   c. The new constraint is a strict SUPERSET of the old one, so validation
--      cannot fail on pre-existing rows; it is one sequential scan of a table
--      in the low tens of thousands of rows on a single-node WSL2 box.
--
-- The lock cost that DOES matter is wall-clock hold time, and the mitigation
-- that matters is not applying this while a bulk writer is mid-flight. That is
-- an operational sequencing constraint, not a DDL shape.
--
-- IDEMPOTENT / re-runnable in the same style as 001 and 002: DROP ... IF EXISTS
-- before the ADD, and a backfill whose WHERE clause selects nothing on a second
-- run.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. The widened CHECK.
--
-- Alphabetically ordered, and kept set-equal to CONTENT_TYPES in
-- src/pg/write.ts. tests/integration/pg-content-type-enum.test.ts parses the
-- value list straight out of THIS statement and asserts set-equality, so the
-- two cannot drift silently.
-- ---------------------------------------------------------------------------
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_content_type_check;

ALTER TABLE documents ADD CONSTRAINT documents_content_type_check
  CHECK (content_type IN (
    'antipattern','conversation','decision','deductive','eval-run','handoff','hub',
    'milestone','note','observation','plan','preference','problem','progress',
    'project','research','retro','unknown'
  ));

-- ---------------------------------------------------------------------------
-- 2. The backfill — repair rows the PRE-amendment narrowing landed as 'unknown'.
--
-- The reindex that ran under the narrow enum parked every extending and every
-- synonym value in the 'unknown' sink with the original string in
-- content_type_raw. Because the raw string was preserved, the repair is a
-- metadata UPDATE and NOT a re-embed: the body, the hash and the vectors are
-- all unchanged, so nothing downstream of the embedding needs to be recomputed.
--
-- content_type_raw IS DELIBERATELY NOT CLEARED. The conform is a *mapping
-- decision*, and a mapping decision has to stay auditable and reversible: with
-- the raw string retained, "which rows did we conform, from what, and can we
-- undo it" is a single query. Clearing it would make the repair
-- indistinguishable from a document that was authored with the admitted value
-- in the first place. The `content_type_raw = content_type_raw` below is a
-- deliberate no-op self-assignment, written out so the preservation is visible
-- in the DDL rather than inferred from an absence.
--
-- The `WHERE content_type = 'unknown'` guard is what makes this safe to re-run
-- and what stops it from ever overwriting a row that already carries a decided
-- type.
--
-- Rows whose content_type_raw is NOT in this map (notably 'session-transcript',
-- see the header) are LEFT ALONE, still 'unknown', still carrying their raw.
-- That residue is the real retag backlog, and it is now a small honest number
-- instead of a shrug.
-- ---------------------------------------------------------------------------
UPDATE documents AS d
   SET content_type     = m.admitted,
       content_type_raw = d.content_type_raw  -- deliberate no-op: the audit trail stays
  FROM (VALUES
          -- ---- the four newly-admitted values (identity mappings) ----------
          ('eval-run',          'eval-run'),
          ('observation',       'observation'),
          ('plan',              'plan'),
          ('retro',             'retro'),
          -- ---- the ten conformed synonyms ---------------------------------
          ('planning',          'plan'),
          ('queue-plan',        'plan'),
          ('run',               'eval-run'),
          ('run-report',        'eval-run'),
          ('synthesis',         'deductive'),
          ('research-synthesis','deductive'),
          ('memo',              'deductive'),
          ('reference',         'hub'),
          ('runbook',           'hub'),
          ('operations',        'handoff')
       ) AS m(raw, admitted)
 WHERE d.content_type = 'unknown'
   AND d.content_type_raw = m.raw;
