-- 011_content_vectors_doc_tier_backfill.sql — set doc_tier on the vectors
-- that already exist (master-harness-vn4rz.77).
--
-- ===========================================================================
-- 010 added content_vectors.doc_tier (DEFAULT false) and the triggers that
-- keep it current for every write from then on. This file flips the vectors
-- that were already there: true wherever ANY documents row references the
-- hash (010's DESIGN DECISION has the rule and why it ignores `active`).
--
-- WHY A FILE OF ITS OWN. Measured on a scratch copy of the live vault
-- (239,677 vectors, 66,745 of them documents-tier, 2026-09-29): this UPDATE
-- took 2 min 55 s. The embedding is TOASTed and untouched, so each new heap
-- tuple is small, but the table is packed, so the update is non-HOT and the
-- 821 MB full HNSW index (002) re-inserts every one of the 66,745 tuples —
-- that is the cost. Here it holds ROW EXCLUSIVE on content_vectors, which
-- conflicts with no reader (ACCESS SHARE) and no INSERT; had it stayed in 010
-- it would have run under ADD COLUMN's ACCESS EXCLUSIVE and blocked every
-- search and write for the full three minutes.
--
-- WHILE IT RUNS, the flags flip atomically at COMMIT. Code that already reads
-- `cv.doc_tier` (src/pg/search.ts after vn4rz.77) sees no documents-tier
-- vectors until then, so apply 010-012 BEFORE deploying that code; code from
-- before vn4rz.77 ignores the column and is unaffected.
--
-- AFTERWARDS: `VACUUM (ANALYZE) content_vectors` to reclaim the 66,745 dead
-- tuples (autovacuum will get there, but the HNSW index scan pays for them
-- until it does). Measured: 61 s on the scratch copy.
--
-- IDEMPOTENT: `AND NOT doc_tier` makes a re-run rewrite nothing that is
-- already correct. It never sets false: right after 010 nothing can be
-- wrongly true (the column is born false). reconcileDocTier() in
-- src/pg/write.ts is the two-directional repair.
--
-- ROLLBACK: nothing to undo on its own — `UPDATE content_vectors SET
-- doc_tier = false` restores 010's state; see 010 for the full rollback.
-- ===========================================================================

UPDATE content_vectors cv
   SET doc_tier = true
 WHERE NOT cv.doc_tier
   AND EXISTS (SELECT 1 FROM documents d WHERE d.hash = cv.hash);
