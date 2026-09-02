-- 001_core_schema.sql — clawmem PostgreSQL core schema (master-harness-vn4rz.7).
--
-- ADR-0162 §3 (faceted metadata), §5 (lifecycle), §7 (per-leg model identity).
--
-- SCOPE: the LIVE subset only — content, documents (+ the tsvector FTS column),
-- content_vectors, memory_evolution, entity_nodes. The dormant sqlite subsystems
-- (causal_*, beads_*, entity_triples, co_activations, llm_cache, hook_dedupe,
-- session_log, context_usage, judge_*, embed_canary, and the all-NULL observation
-- block on documents) are deliberately NOT ported.
--
-- NO `vault` COLUMN ANYWHERE. ADR-0162 §1 fixes vault topology as
-- database-per-vault (`clawmem` vs `clawmem_nsfw` are separate DATABASES), so a
-- tenant column here would re-open the cross-vault identity collision the
-- database boundary closes by construction. sqlite's entity_nodes.vault is
-- dropped on purpose.
--
-- :EMBED_DIM is substituted by the migration runner from the SINGLE named
-- constant in src/pg/config.ts (EMBED_DIM). Do not hardcode a dimension here.
--
-- Idempotent: every statement is IF NOT EXISTS / OR REPLACE / guarded DO block.

CREATE TABLE IF NOT EXISTS schema_migrations (
  version     text        PRIMARY KEY,
  applied_at  timestamptz NOT NULL DEFAULT now(),
  checksum    text        NOT NULL
);

-- ---------------------------------------------------------------------------
-- content — the deduplicated document body, keyed by sha256 of the raw file.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS content (
  hash        text        PRIMARY KEY,
  doc         text        NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- documents — the filesystem layer + the six ADR-0162 §3 facets.
--
-- Every facet carries an explicit 'unknown' value (§3: "No schema will ever be
-- complete"); the unknown counts are a MONITORED METRIC, not a shrug, and are
-- queryable per facet. `collection` survives as one more facet value, never as
-- the container.
--
-- content_type is the CLOSED ADR-0058 enum (§5: "gaps resolve by retag, not
-- extension"). A value outside it lands as 'unknown' with the raw string
-- preserved in content_type_raw so the retag backlog is a query, not an
-- archaeology exercise. NOTHING is silently dropped.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
  id                bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  collection        text        NOT NULL,
  path              text        NOT NULL,
  title             text        NOT NULL,
  hash              text        NOT NULL REFERENCES content(hash) ON DELETE CASCADE,
  created_at        timestamptz NOT NULL DEFAULT now(),
  modified_at       timestamptz NOT NULL DEFAULT now(),
  active            boolean     NOT NULL DEFAULT true,

  -- ---- the six facets (ADR-0162 §3) ----------------------------------------
  domain            text        NOT NULL DEFAULT 'unknown',
  audience          text        NOT NULL DEFAULT 'unknown'
                      CHECK (audience IN ('operator','household','external','agent-internal','unknown')),
  trust_tier        text        NOT NULL DEFAULT 'unknown'
                      CHECK (trust_tier IN ('authored','distilled','ingested-verbatim','derived','unknown')),
  -- provenance: WHO/WHAT assigned each facet value and WHY (§3 — "the
  -- classification decision itself stays retrievable"), written at load time,
  -- never attached retroactively.
  provenance        jsonb       NOT NULL DEFAULT '{}'::jsonb,
  content_type      text        NOT NULL DEFAULT 'unknown'
                      CHECK (content_type IN (
                        'antipattern','conversation','decision','deductive','handoff','hub',
                        'milestone','note','preference','problem','progress','project','research',
                        'unknown')),
  content_type_raw  text,
  sensitivity       text        NOT NULL DEFAULT 'unknown'
                      CHECK (sensitivity IN ('public','private','crypt','nsfw','unknown')),
  source_ref        text,

  -- ---- carried from the live sqlite schema ---------------------------------
  workstream        text,
  description       text,
  tags              text[],
  review_by         timestamptz,
  confidence        double precision NOT NULL DEFAULT 0.5,
  access_count      bigint      NOT NULL DEFAULT 0,
  content_hash      text,
  quality_score     double precision NOT NULL DEFAULT 0.5,
  pinned            boolean     NOT NULL DEFAULT false,
  snoozed_until     timestamptz,
  last_accessed_at  timestamptz,
  archived_at       timestamptz,
  memory_type       text        DEFAULT 'semantic',
  normalized_hash   text,
  duplicate_count   bigint      NOT NULL DEFAULT 1,
  last_seen_at      timestamptz,
  topic_key         text,
  revision_count    bigint      NOT NULL DEFAULT 1,
  embed_state       text        NOT NULL DEFAULT 'pending',
  embed_error       text,
  embed_attempts    integer     NOT NULL DEFAULT 0,
  amem_keywords     text,
  amem_tags         text,
  amem_context      text,
  invalidated_at    timestamptz,
  invalidated_by    bigint,
  superseded_by     bigint,
  authored_at       timestamptz,
  deactivated_reason text,
  origin            text,

  -- Materialized FTS (ADR-0162 §2). Maintained by trigger, weighted with
  -- setweight(), configuration named EXPLICITLY as 'english' at write time —
  -- the query side MUST name it too (a bare to_tsquery at one end fails
  -- SILENTLY, which is the whole reason §2 mandates naming it at both ends).
  fts               tsvector,

  CONSTRAINT documents_collection_path_key UNIQUE (collection, path)
);

CREATE INDEX IF NOT EXISTS documents_collection_active_idx ON documents (collection, active);
CREATE INDEX IF NOT EXISTS documents_hash_idx              ON documents (hash);
CREATE INDEX IF NOT EXISTS documents_fts_idx               ON documents USING gin (fts);
CREATE INDEX IF NOT EXISTS documents_title_trgm_idx        ON documents USING gin (title gin_trgm_ops);
-- facet indexes (§3): each facet is an independent selector, so each gets its
-- own index rather than one composite that only serves a single prefix order.
CREATE INDEX IF NOT EXISTS documents_domain_idx      ON documents (domain)       WHERE active;
CREATE INDEX IF NOT EXISTS documents_audience_idx    ON documents (audience)     WHERE active;
CREATE INDEX IF NOT EXISTS documents_trust_tier_idx  ON documents (trust_tier)   WHERE active;
CREATE INDEX IF NOT EXISTS documents_content_type_idx ON documents (content_type) WHERE active;
CREATE INDEX IF NOT EXISTS documents_sensitivity_idx ON documents (sensitivity)  WHERE active;
CREATE INDEX IF NOT EXISTS documents_source_ref_idx  ON documents (source_ref)   WHERE source_ref IS NOT NULL;
CREATE INDEX IF NOT EXISTS documents_effective_time_idx
  ON documents (COALESCE(authored_at, modified_at)) WHERE active;

-- ---------------------------------------------------------------------------
-- FTS maintenance. Two triggers, because the body lives in `content` and the
-- title/facets live in `documents` — a single trigger on one table would let
-- the other side drift silently.
-- ---------------------------------------------------------------------------
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
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS documents_fts_trg ON documents;
CREATE TRIGGER documents_fts_trg
  BEFORE INSERT OR UPDATE OF title, description, tags, hash ON documents
  FOR EACH ROW EXECUTE FUNCTION documents_fts_refresh();

-- A content body can be rewritten under a stable hash only by a bug, but an
-- UPDATE on content would otherwise leave every dependent document's fts stale
-- with no error. Close it.
CREATE OR REPLACE FUNCTION content_fts_cascade() RETURNS trigger AS $$
BEGIN
  IF NEW.doc IS DISTINCT FROM OLD.doc THEN
    UPDATE documents SET hash = hash WHERE hash = NEW.hash;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS content_fts_cascade_trg ON content;
CREATE TRIGGER content_fts_cascade_trg
  AFTER UPDATE OF doc ON content
  FOR EACH ROW EXECUTE FUNCTION content_fts_cascade();

-- ---------------------------------------------------------------------------
-- content_vectors — sqlite's content_vectors + vectors_vec collapsed into ONE
-- table. `model` is a first-class per-row column (ADR-0162 §7) and carries its
-- own index, which is what makes a future multi-space design additive.
--
-- The dimension is FIXED at CREATE TABLE (pgvector requirement) from the single
-- named constant EMBED_DIM in src/pg/config.ts. sqlite's vec0 table DISCOVERED
-- it at runtime from the first embed response; that is not available here, so
-- src/pg/write.ts asserts the live column's vector_dims against EMBED_DIM at
-- connect time and REFUSES on mismatch — never truncate, never pad.
--
-- NOT PARTITIONED, deliberately: ADR-0162 §5 — "do not partition the embedding
-- table without a measured reason." The append-only origin tables are vn4rz.8.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS content_vectors (
  hash            text          NOT NULL REFERENCES content(hash) ON DELETE CASCADE,
  seq             integer       NOT NULL DEFAULT 0,
  pos             integer       NOT NULL DEFAULT 0,
  model           text          NOT NULL,
  embedding       vector(:EMBED_DIM) NOT NULL,
  embedded_at     timestamptz   NOT NULL DEFAULT now(),
  fragment_type   text,
  fragment_label  text,
  canonical_id    text,
  embed_input_fp  text,
  CONSTRAINT content_vectors_pkey PRIMARY KEY (hash, seq)
);

-- Per-model index (§7). Every ANN probe and every geometry check filters on
-- model first, so this is an access predicate, not a filter.
CREATE INDEX IF NOT EXISTS content_vectors_model_idx ON content_vectors (model);

-- ---------------------------------------------------------------------------
-- memory_evolution
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memory_evolution (
  id                 bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  memory_id          bigint      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  triggered_by       bigint      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  version            integer     NOT NULL DEFAULT 1,
  previous_keywords  text,
  new_keywords       text,
  previous_context   text,
  new_context        text,
  reasoning          text,
  created_at         timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS memory_evolution_memory_id_idx    ON memory_evolution (memory_id);
CREATE INDEX IF NOT EXISTS memory_evolution_triggered_by_idx ON memory_evolution (triggered_by);
CREATE INDEX IF NOT EXISTS memory_evolution_created_at_idx   ON memory_evolution (created_at);

-- ---------------------------------------------------------------------------
-- entity_nodes — sqlite's `vault` column is DROPPED (see the header).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entity_nodes (
  entity_id      text        PRIMARY KEY,
  entity_type    text,
  name           text,
  description    text,
  created_at     timestamptz NOT NULL DEFAULT now(),
  mention_count  bigint      NOT NULL DEFAULT 0,
  last_seen      timestamptz,
  canonical_id   text
);

CREATE INDEX IF NOT EXISTS entity_nodes_type_idx      ON entity_nodes (entity_type);
CREATE INDEX IF NOT EXISTS entity_nodes_mentions_idx  ON entity_nodes (mention_count DESC);
CREATE INDEX IF NOT EXISTS entity_nodes_lower_name_idx ON entity_nodes (lower(name));
