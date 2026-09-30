-- 010_content_vectors_doc_tier.sql — mark which vectors the ANN reader can
-- ever return (master-harness-vn4rz.77). Pairs with 011, which builds the
-- partial HNSW index over the rows this column marks.
--
-- ===========================================================================
-- THE DEFECT. content_vectors holds the vectors of BOTH tiers: the curated
-- `documents` tier and the ADR-0162 origin tier (`origin_documents`, migration
-- 004), which shares the one content store so a tier move never re-embeds.
-- Measured on the live vault 2026-09-29: 239,805 vectors, one model; 172,929
-- (72%) belong to hashes referenced ONLY by origin_documents, 66,876 to
-- documents hashes, 0 hashes in both, 0 GC-eligible. They are not garbage.
--
-- But NO ANN reader ever searches origin vectors. The only ANN query,
-- src/pg/search.ts buildVecSearchQuery, JOINs `documents` and post-filters
-- through the one full HNSW index (content_vectors_embedding_hnsw_idx, 002)
-- under `hnsw.iterative_scan = strict_order`. A query landing in an
-- origin-dense region walks graph node after graph node whose join partner
-- does not exist: ~16.7k-20k tuples per probe (the 20,000 max_scan_tuples
-- cap), EXPLAIN ANALYZE 2.5-6.1 s against a 1,200 ms statement_timeout.
-- Live: 3 of 10 hybrid runs failed with PgVecSearchTimeoutError (ann-scan).
-- Documents-region probes: 30 ms warm. The index is 72% rows nobody asks for.
-- ===========================================================================
--
-- ===========================================================================
-- DESIGN DECISION — a MAINTAINED boolean + a PARTIAL index, not a second
-- table, not a partitioned content_vectors, not an origin-side delete.
--
-- A partial HNSW index needs an immutable row predicate. "Some documents row
-- references my hash" is a JOIN, which an index predicate cannot express, so
-- it is MATERIALISED here as `doc_tier` and kept current by triggers:
--
--   * content_vectors BEFORE INSERT: NEW.doc_tier := NEW.doc_tier OR
--     EXISTS(documents.hash = NEW.hash). The reindex writer commits its
--     documents row BEFORE it embeds (upsertDocument, then the batch), so the
--     normal write path lands true at insert time.
--   * documents AFTER INSERT, and AFTER UPDATE OF hash when the hash actually
--     changed: NEW.hash's vectors -> true. Covers the tier move origin ->
--     documents (the vectors already exist; nothing is re-embedded) and a
--     content change re-pointing documents.hash at already-embedded content.
--   * documents AFTER DELETE, and the same UPDATE OF hash: OLD.hash's vectors
--     -> false once NO documents row references OLD.hash any more. Covers the
--     tier move documents -> origin (src/pg/origin.ts drops the collection from
--     documents) and a superseded hash waiting for the content GC.
--
-- "ANY documents row" — active or not, invalidated or not — is the rule, on
-- purpose. It is the content GC's own protection rule (write.ts
-- ORPHAN_PREDICATE; the vn4rz.41 sweep soft-retires rows so a returning file
-- reactivates with its vectors intact), so a reactivation flips no flag and
-- needs no index maintenance. Inactive rows cost a little extra walk in the
-- partial index; the reader's `d.active` / `invalidated_at` fence still
-- excludes them. The column answers "could the reader EVER want this?", not
-- "does it want it right now?".
--
-- WHAT THE TRIGGERS CANNOT SEE. Two concurrent transactions — one inserting a
-- vector, the other the documents row for the same hash — each check under a
-- snapshot that cannot see the other's uncommitted row, so both leave
-- doc_tier false. The mirror race exists for delete-vs-insert. The flag is
-- therefore a MAINTAINED CACHE with a self-heal, not an invariant:
-- src/pg/write.ts reconcileDocTier() re-derives it at the end of every reindex
-- pass. A stale false only hides a vector until then; a stale true only costs
-- walk.
--
-- ALTERNATIVES REJECTED. (1) Moving origin vectors to their own table: the
-- ADR-0162 contract is ONE content store keyed by hash so a tier move is a
-- documents/origin_documents row move with zero re-embed; splitting the table
-- turns every tier move into a vector copy. (2) Partitioning content_vectors
-- by tier: ADR-0162 §5 forbids partitioning the embedding table without a
-- measured reason, and a tier move would then be a cross-partition row move
-- (delete + insert + two HNSW inserts) instead of a flag flip. (3) Raising
-- hnsw.max_scan_tuples / the timeout: pays the origin walk on every query
-- instead of removing it.
--
-- BACKFILL COST. `ADD COLUMN ... NOT NULL DEFAULT false` is metadata-only on
-- PG16. The backfill UPDATE writes a new heap tuple for every documents-tier
-- vector (66,876 live). The embedding is TOASTed and not modified, so the heap
-- tuple stays small, but it is a non-HOT update (doc_tier is in 011's index
-- predicate) and every existing index — including the full HNSW — receives
-- the new tuple. This migration is transactional, so ALTER TABLE's ACCESS
-- EXCLUSIVE lock is held for the whole backfill: apply it in the cutover
-- window, and VACUUM content_vectors afterwards to reclaim the dead tuples.
-- Measured on a scratch copy of the live vault: see the vn4rz.77 report.
--
-- ROLLBACK: DROP INDEX CONCURRENTLY content_vectors_embedding_doc_hnsw_idx
-- (011); DROP TRIGGER content_vectors_doc_tier_trg ON content_vectors;
-- DROP TRIGGER documents_doc_tier_ins_trg, documents_doc_tier_upd_trg,
-- documents_doc_tier_del_trg ON documents; DROP FUNCTION
-- content_vectors_doc_tier_on_insert(), documents_doc_tier_sync();
-- ALTER TABLE content_vectors DROP COLUMN doc_tier; DELETE the 010/011 rows
-- from schema_migrations. The code change in search.ts must be reverted
-- FIRST — it references cv.doc_tier.
--
-- Functions pin `search_path = pg_catalog, :CLAWMEM_SCHEMA` and use
-- unqualified table names, the 007 convention (CVE-2018-1058 hardening that
-- still follows setPgSchema / CLAWMEM_PG_SCHEMA deployments).
-- ===========================================================================

ALTER TABLE content_vectors
  ADD COLUMN IF NOT EXISTS doc_tier boolean NOT NULL DEFAULT false;

-- Backfill. `AND NOT doc_tier` keeps a re-run from rewriting rows that are
-- already correct.
UPDATE content_vectors cv
   SET doc_tier = true
 WHERE NOT cv.doc_tier
   AND EXISTS (SELECT 1 FROM documents d WHERE d.hash = cv.hash);

-- ---------------------------------------------------------------------------
-- content_vectors: a vector inserted after its documents row is born true.
-- ON CONFLICT DO UPDATE (the re-embed path) does not touch doc_tier, so the
-- existing row's value is kept.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION content_vectors_doc_tier_on_insert() RETURNS trigger AS $$
BEGIN
  IF NOT NEW.doc_tier THEN
    NEW.doc_tier := EXISTS (SELECT 1 FROM documents d WHERE d.hash = NEW.hash);
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SET search_path = pg_catalog, :CLAWMEM_SCHEMA;

DROP TRIGGER IF EXISTS content_vectors_doc_tier_trg ON content_vectors;
CREATE TRIGGER content_vectors_doc_tier_trg
  BEFORE INSERT ON content_vectors
  FOR EACH ROW EXECUTE FUNCTION content_vectors_doc_tier_on_insert();

-- ---------------------------------------------------------------------------
-- documents: keep the flag of the hash(es) a row change touches.
-- `AND doc_tier IS DISTINCT FROM <target>` keeps a no-op from writing a new
-- tuple (and a new HNSW entry) for every vector of the hash.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION documents_doc_tier_sync() RETURNS trigger AS $$
BEGIN
  IF TG_OP IN ('INSERT', 'UPDATE') THEN
    UPDATE content_vectors SET doc_tier = true
     WHERE hash = NEW.hash AND NOT doc_tier;
  END IF;
  IF TG_OP IN ('DELETE', 'UPDATE') THEN
    IF NOT EXISTS (SELECT 1 FROM documents d WHERE d.hash = OLD.hash) THEN
      UPDATE content_vectors SET doc_tier = false
       WHERE hash = OLD.hash AND doc_tier;
    END IF;
  END IF;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql SET search_path = pg_catalog, :CLAWMEM_SCHEMA;

-- Three triggers, not one: a WHEN clause that reads OLD is not allowed on an
-- INSERT trigger, and the UPDATE trigger must skip the hash-unchanged case
-- (content_fts_cascade's `SET hash = hash`, and every upsertDocument that
-- rewrites the row without a content change).
DROP TRIGGER IF EXISTS documents_doc_tier_ins_trg ON documents;
CREATE TRIGGER documents_doc_tier_ins_trg
  AFTER INSERT ON documents
  FOR EACH ROW EXECUTE FUNCTION documents_doc_tier_sync();

DROP TRIGGER IF EXISTS documents_doc_tier_upd_trg ON documents;
CREATE TRIGGER documents_doc_tier_upd_trg
  AFTER UPDATE OF hash ON documents
  FOR EACH ROW WHEN (OLD.hash IS DISTINCT FROM NEW.hash)
  EXECUTE FUNCTION documents_doc_tier_sync();

DROP TRIGGER IF EXISTS documents_doc_tier_del_trg ON documents;
CREATE TRIGGER documents_doc_tier_del_trg
  AFTER DELETE ON documents
  FOR EACH ROW EXECUTE FUNCTION documents_doc_tier_sync();
