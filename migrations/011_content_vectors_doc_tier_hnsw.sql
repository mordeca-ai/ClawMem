-- 011_content_vectors_doc_tier_hnsw.sql — the PARTIAL HNSW index the ANN
-- reader actually uses (master-harness-vn4rz.77).
--
-- ===========================================================================
-- WHAT THIS IS. An HNSW index over ONLY the documents-tier vectors
-- (`WHERE doc_tier`, the column migration 010 adds and maintains). The ANN
-- query in src/pg/search.ts carries `AND cv.doc_tier`, which is what lets the
-- planner prove the predicate and pick this index; a query without it can
-- only use the full index from 002.
--
-- WHY. On the live vault 72% of content_vectors are origin-tier vectors no ANN
-- reader ever returns (010's header has the measurement). Under the full
-- index, a query landing in an origin-dense region iteratively scans up to
-- the 20,000-tuple max_scan_tuples cap, discarding every origin row at the
-- documents JOIN: 2.5-6.1 s per probe against a 1,200 ms statement_timeout.
-- This index contains no origin rows, so there is nothing to discard.
--
-- PARAMETERS are 002's (m = 16, ef_construction = 128), unchanged. The family
-- and operating point are still vn4rz.32's to measure; this bead changes
-- which rows are indexed, not how.
--
-- THE FULL INDEX IS KEPT, deliberately, in this change. master-harness's
-- clawmem-pg-hnsw-reindex timer REINDEXes content_vectors_embedding_hnsw_idx
-- BY NAME, and dropping it here would turn that timer red on the next tick.
-- Nothing in src/ reads through it any more once this lands (the only ANN
-- query carries `cv.doc_tier`), so it is pure write amplification and ~835 MB
-- of disk. Dropping it — and repointing the timer at this index — is a
-- follow-up bead, not part of this migration.
--
-- CONCURRENTLY, for 002's reasons (no write lock on a live table; can fail
-- leaving an INVALID index, which src/pg/migrate.ts assertNoInvalidIndexes
-- refuses to record). Cannot run in a transaction block, hence the directive.
--
-- ROLLBACK: DROP INDEX CONCURRENTLY IF EXISTS
-- content_vectors_embedding_doc_hnsw_idx; DELETE FROM schema_migrations
-- WHERE version = '011_content_vectors_doc_tier_hnsw'. The search.ts
-- predicate may stay: without this index the planner serves it from the full
-- index with `doc_tier` as a filter (same results, the pre-vn4rz.77 latency).
-- ===========================================================================
-- clawmem:no-transaction

CREATE INDEX CONCURRENTLY IF NOT EXISTS content_vectors_embedding_doc_hnsw_idx
  ON content_vectors USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 128)
  WHERE doc_tier;
