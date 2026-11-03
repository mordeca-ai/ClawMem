-- SQLite lifecycle decisions are authoritative until the PG cutover.
-- Keep the archive timestamp so a later restore can clear it explicitly.
ALTER TABLE origin_documents ADD COLUMN IF NOT EXISTS archived_at timestamptz;
