-- 013_drop_full_content_vectors_hnsw.sql — retire the unused full ANN index.
--
-- Migration 012 created the doc-tier partial HNSW index used by every
-- documents-scoped vector reader. The master-harness reindex timer now targets
-- that partial index. On the SFW vault the full index's last recorded scan was
-- 2026-10-01 05:18 CDT; it had no scans for more than four days before this
-- migration was authored (2026-10-05). It occupied 968 MB at that check.
--
-- CONCURRENTLY avoids blocking writes to the live content_vectors table.
-- IF EXISTS permits a safe rerun after a drop succeeds but migration recording
-- fails. Rollback, if required: recreate the definition from migration 002.
-- clawmem:no-transaction

DROP INDEX CONCURRENTLY IF EXISTS content_vectors_embedding_hnsw_idx;
