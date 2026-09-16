-- 009_origin_documents_partition_schema_portable.sql — make origin partition
-- creation follow the deployment schema (master-harness-vn4rz.48).
--
-- Migration 005 correctly pinned this SECURITY-sensitive DDL helper's
-- search_path, but also hard-coded every partition reference to `public`.
-- That made setPgSchema()/CLAWMEM_PG_SCHEMA deployments call the function in
-- their own schema while the function created and inspected partitions in a
-- different one. Keep the pinned pg_catalog-first search_path, but substitute
-- the same validated, identifier-quoted deployment schema used by 007/008.
--
-- The placeholder also appears inside dynamic-SQL string literals. Migration
-- substitution happens before PostgreSQL parses the file, so a schema `foo`
-- produces `"foo".%I` in the format string: the schema remains quoted as an
-- identifier while `%I` independently quotes the generated partition name.
--
-- 005 is already applied and checksum-protected, so this correction must be a
-- forward migration. With the default schema, :CLAWMEM_SCHEMA is `"public"`
-- and production behavior is unchanged.

CREATE OR REPLACE FUNCTION origin_documents_ensure_partition(p_month date)
RETURNS text AS $$
DECLARE
  lo   date := date_trunc('month', p_month)::date;
  hi   date := (date_trunc('month', p_month) + interval '1 month')::date;
  name text := format('origin_documents_%s', to_char(lo, 'YYYY_MM'));
BEGIN
  IF to_regclass(format(':CLAWMEM_SCHEMA.%I', name)) IS NULL THEN
    EXECUTE format(
      'CREATE TABLE :CLAWMEM_SCHEMA.%I PARTITION OF :CLAWMEM_SCHEMA.origin_documents '
      'FOR VALUES FROM (%L) TO (%L)',
      name, lo::text || ' 00:00:00+00', hi::text || ' 00:00:00+00');
    EXECUTE format(
      'ALTER TABLE :CLAWMEM_SCHEMA.%I SET (autovacuum_vacuum_insert_threshold = 1000, '
      'autovacuum_vacuum_insert_scale_factor = 0)', name);
  END IF;
  RETURN name;
END;
$$ LANGUAGE plpgsql SET search_path = pg_catalog, :CLAWMEM_SCHEMA;
