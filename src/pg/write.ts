/**
 * The clawmem PostgreSQL write path (master-harness-vn4rz.7, ADR-0162).
 *
 * ADDITIVE. This module does not touch src/store.ts and does not change the
 * sqlite path's behavior. The two writers coexist until master-harness-vn4rz.11
 * retires sqlite — the transient window ADR-0162 §6 sanctions.
 *
 * ---------------------------------------------------------------------------
 * THE LOAD-BEARING INVARIANT: the write-geometry preflight runs INSIDE the same
 * transaction as every vector INSERT.
 *
 * sqlite's assertWriteEmbedModelConsistent (src/store.ts ~4812) memoized on
 * `PRAGMA data_version`. **PostgreSQL has no data_version equivalent, so the
 * memo is DROPPED and only the transactional atomicity is kept.** That is the
 * half that matters: doing the check outside the transaction, or after the
 * insert, re-opens the foreign-model vector poisoning that master-harness-p2ib3
 * and vn4rz.21 were written to close.
 *
 * The transaction is opened with an explicit `SELECT ... FOR UPDATE`-equivalent
 * advisory lock rather than relying on READ COMMITTED. Under READ COMMITTED two
 * concurrent writers can BOTH read an empty content_vectors, both see "nothing
 * to be inconsistent with", and both commit different models — a lost-update on
 * a set-level invariant that no row lock covers. A transaction-scoped advisory
 * lock serializes the check-then-write window; it is released at COMMIT/ROLLBACK
 * automatically.
 * ---------------------------------------------------------------------------
 */

import type { PoolClient } from "pg";
import { withTransaction, toVectorLiteral } from "./client.js";
import { embedDim } from "./config.js";
import {
  PgSchemaGeometryError,
  PgVecBatchModelMismatchError,
  PgVecDimensionMismatchError,
  PgVecWriteModelMismatchError,
} from "./errors.js";

/** Arbitrary but stable key for the write-geometry advisory lock. */
const GEOMETRY_LOCK_KEY = 0x1a2b3c4d;

function embedEndpointLabel(): string {
  return process.env.CLAWMEM_EMBED_URL || "the local in-process embedder";
}

// ===========================================================================
// Geometry
// ===========================================================================

/**
 * Assert the LIVE schema's vector column matches the compiled-in EMBED_DIM.
 * Called once per process before any vector write. sqlite's vec0 table
 * discovered the dimension from the first embed response; that discovery is
 * gone, so the two constants must be reconciled explicitly or every row is
 * written under an unverified assumption.
 */
export async function assertSchemaGeometry(c: PoolClient): Promise<number> {
  const { rows } = await c.query<{ dims: number | null }>(`
    SELECT atttypmod - 4 AS dims
      FROM pg_attribute
     WHERE attrelid = 'content_vectors'::regclass
       AND attname = 'embedding'
       AND NOT attisdropped
  `);
  const actual = rows[0]?.dims;
  if (actual == null || actual < 0) {
    throw new PgSchemaGeometryError(embedDim(), -1);
  }
  if (actual !== embedDim()) {
    throw new PgSchemaGeometryError(embedDim(), actual);
  }
  return actual;
}

/** DISTINCT set of models already embedded in this database. */
export async function getVecModels(c: PoolClient): Promise<string[]> {
  const { rows } = await c.query<{ model: string }>(
    "SELECT DISTINCT model FROM content_vectors ORDER BY model",
  );
  return rows.map(r => r.model);
}

/**
 * Write-path counterpart of the sqlite assertWriteEmbedModelConsistent
 * (master-harness-vn4rz.21). MUST be called inside the write transaction.
 *
 * Refuses when the database holds vectors under a DIFFERENT single model, or is
 * already heterogeneous. No-ops when the database has no vectors yet, or when
 * the write carries no model name (an endpoint that reports none cannot be
 * discriminated — mirrors sqlite's `probe.model &&` condition).
 *
 * No memo. `PRAGMA data_version` has no PG equivalent, and a memo keyed on
 * anything weaker would be a cache that can go stale in exactly the window the
 * fence exists to cover.
 */
export async function assertWriteEmbedModelConsistent(
  c: PoolClient,
  writeModel: string,
): Promise<void> {
  if (!writeModel) return;
  const stored = await getVecModels(c);
  if (stored.length === 0) return;
  if (!(stored.length === 1 && stored[0] === writeModel)) {
    throw new PgVecWriteModelMismatchError(stored, writeModel, embedEndpointLabel());
  }
}

export interface EmbeddingWrite {
  hash: string;
  seq: number;
  pos: number;
  embedding: ArrayLike<number>;
  model: string;
  embeddedAt?: Date | string;
  fragmentType?: string | null;
  fragmentLabel?: string | null;
  canonicalId?: string | null;
  embedInputFp?: string | null;
}

function assertDimension(w: EmbeddingWrite, expected: number): void {
  if (w.embedding.length !== expected) {
    throw new PgVecDimensionMismatchError(
      expected,
      w.embedding.length,
      `embedding for hash=${w.hash} seq=${w.seq}`,
    );
  }
}

/**
 * Insert a batch of embeddings. ONE transaction: advisory lock → schema geometry
 * → intra-batch heterogeneity → per-row dimension → per-row model fence → the
 * INSERTs. Any refusal rolls the WHOLE batch back; nothing is written.
 *
 * ON CONFLICT names its constraint EXPLICITLY. sqlite's `INSERT OR REPLACE`
 * infers the constraint; Postgres does not, and an unnamed/wrong target upserts
 * against the wrong key with no error.
 */
export async function insertEmbeddingsBatch(writes: EmbeddingWrite[]): Promise<void> {
  if (writes.length === 0) return;

  await withTransaction(async c => {
    await c.query("SELECT pg_advisory_xact_lock($1)", [GEOMETRY_LOCK_KEY]);

    const dim = await assertSchemaGeometry(c);

    // Intra-batch heterogeneity: a fresh/just-cleared database has no stored
    // model to compare against, so a batch from an endpoint that flapped
    // mid-flight would slip past the database-level comparison below. Two models
    // inside ONE batch is drift by definition.
    const batchModels = [...new Set(writes.map(w => w.model).filter(Boolean))].sort();
    if (batchModels.length > 1) {
      throw new PgVecBatchModelMismatchError(batchModels[0]!, batchModels[1]!);
    }

    for (const w of writes) assertDimension(w, dim);
    for (const w of writes) await assertWriteEmbedModelConsistent(c, w.model);

    for (const w of writes) {
      await c.query(
        `INSERT INTO content_vectors
           (hash, seq, pos, model, embedding, embedded_at,
            fragment_type, fragment_label, canonical_id, embed_input_fp)
         VALUES ($1, $2, $3, $4, $5::vector, COALESCE($6::timestamptz, now()), $7, $8, $9, $10)
         ON CONFLICT ON CONSTRAINT content_vectors_pkey DO UPDATE SET
           pos            = EXCLUDED.pos,
           model          = EXCLUDED.model,
           embedding      = EXCLUDED.embedding,
           embedded_at    = EXCLUDED.embedded_at,
           fragment_type  = EXCLUDED.fragment_type,
           fragment_label = EXCLUDED.fragment_label,
           canonical_id   = EXCLUDED.canonical_id,
           embed_input_fp = EXCLUDED.embed_input_fp`,
        [
          w.hash, w.seq, w.pos, w.model, toVectorLiteral(w.embedding),
          w.embeddedAt ?? null,
          w.fragmentType ?? null, w.fragmentLabel ?? null,
          w.canonicalId ?? null, w.embedInputFp ?? null,
        ],
      );
    }
  });
}

export async function insertEmbedding(w: EmbeddingWrite): Promise<void> {
  await insertEmbeddingsBatch([w]);
}

// ===========================================================================
// Content + documents
// ===========================================================================

/** The CLOSED ADR-0058 content_type enum (ADR-0162 §5), plus 'unknown'. */
export const CONTENT_TYPES = [
  "antipattern", "conversation", "decision", "deductive", "handoff", "hub",
  "milestone", "note", "preference", "problem", "progress", "project", "research",
  "unknown",
] as const;
export type ContentTypeFacet = (typeof CONTENT_TYPES)[number];

export const AUDIENCES = ["operator", "household", "external", "agent-internal", "unknown"] as const;
export const TRUST_TIERS = ["authored", "distilled", "ingested-verbatim", "derived", "unknown"] as const;
export const SENSITIVITIES = ["public", "private", "crypt", "nsfw", "unknown"] as const;

/**
 * Narrow a raw content_type onto the closed enum. Out-of-enum values become
 * 'unknown' AND the raw string is preserved in documents.content_type_raw, so
 * the retag backlog ADR-0162 §5 mandates is a query, not archaeology. Nothing is
 * silently discarded.
 */
export function narrowContentType(raw: string | null | undefined): {
  contentType: ContentTypeFacet;
  raw: string | null;
} {
  if (!raw) return { contentType: "unknown", raw: null };
  return (CONTENT_TYPES as readonly string[]).includes(raw)
    ? { contentType: raw as ContentTypeFacet, raw: null }
    : { contentType: "unknown", raw };
}

export interface Facets {
  domain?: string;
  audience?: (typeof AUDIENCES)[number];
  trustTier?: (typeof TRUST_TIERS)[number];
  sensitivity?: (typeof SENSITIVITIES)[number];
  sourceRef?: string | null;
  /** WHO/WHAT assigned these values and WHY (ADR-0162 §3). */
  provenance?: Record<string, unknown>;
}

export interface DocumentWrite {
  collection: string;
  path: string;
  title: string;
  hash: string;
  body: string;
  contentTypeRaw?: string | null;
  description?: string | null;
  tags?: string[] | null;
  workstream?: string | null;
  authoredAt?: Date | string | null;
  modifiedAt?: Date | string | null;
  qualityScore?: number;
  confidence?: number;
  contentHash?: string | null;
  origin?: string | null;
  memoryType?: string | null;
  facets?: Facets;
}

/**
 * Upsert content + document in ONE transaction.
 *
 * ON CONFLICT targets are NAMED: content on its primary key, documents on the
 * (collection, path) unique constraint. sqlite inferred these from INSERT OR
 * REPLACE; getting the target wrong in PG upserts against the wrong key with no
 * error at all.
 *
 * The fts column is NOT written here — the BEFORE trigger owns it, so write-time
 * and query-time configuration cannot drift apart across call sites.
 */
export async function upsertDocument(d: DocumentWrite): Promise<number> {
  const { contentType, raw } = narrowContentType(d.contentTypeRaw);
  const f = d.facets ?? {};

  return withTransaction(async c => {
    await c.query(
      `INSERT INTO content (hash, doc) VALUES ($1, $2)
       ON CONFLICT ON CONSTRAINT content_pkey DO UPDATE SET doc = EXCLUDED.doc`,
      [d.hash, d.body],
    );

    const { rows } = await c.query<{ id: string }>(
      `INSERT INTO documents (
         collection, path, title, hash, modified_at, active,
         domain, audience, trust_tier, provenance, content_type, content_type_raw,
         sensitivity, source_ref,
         workstream, description, tags, confidence, quality_score,
         content_hash, memory_type, authored_at, origin
       ) VALUES (
         $1, $2, $3, $4, COALESCE($5::timestamptz, now()), true,
         $6, $7, $8, $9::jsonb, $10, $11,
         $12, $13,
         $14, $15, $16::text[], $17, $18,
         $19, $20, $21::timestamptz, $22
       )
       ON CONFLICT ON CONSTRAINT documents_collection_path_key DO UPDATE SET
         title            = EXCLUDED.title,
         hash             = EXCLUDED.hash,
         modified_at      = EXCLUDED.modified_at,
         active           = true,
         domain           = EXCLUDED.domain,
         audience         = EXCLUDED.audience,
         trust_tier       = EXCLUDED.trust_tier,
         provenance       = EXCLUDED.provenance,
         content_type     = EXCLUDED.content_type,
         content_type_raw = EXCLUDED.content_type_raw,
         sensitivity      = EXCLUDED.sensitivity,
         source_ref       = EXCLUDED.source_ref,
         workstream       = EXCLUDED.workstream,
         description      = EXCLUDED.description,
         tags             = EXCLUDED.tags,
         confidence       = EXCLUDED.confidence,
         quality_score    = EXCLUDED.quality_score,
         content_hash     = EXCLUDED.content_hash,
         memory_type      = EXCLUDED.memory_type,
         authored_at      = EXCLUDED.authored_at,
         origin           = EXCLUDED.origin,
         revision_count   = documents.revision_count + 1
       RETURNING id`,
      [
        d.collection, d.path, d.title, d.hash, d.modifiedAt ?? null,
        f.domain ?? "unknown", f.audience ?? "unknown", f.trustTier ?? "unknown",
        JSON.stringify(f.provenance ?? {}), contentType, raw,
        f.sensitivity ?? "unknown", f.sourceRef ?? null,
        d.workstream ?? null, d.description ?? null, d.tags ?? null,
        d.confidence ?? 0.5, d.qualityScore ?? 0.5,
        d.contentHash ?? null, d.memoryType ?? "semantic",
        d.authoredAt ?? null, d.origin ?? null,
      ],
    );
    return Number(rows[0]!.id);
  });
}

// ===========================================================================
// entity_nodes + memory_evolution
// ===========================================================================

export interface EntityNodeWrite {
  entityId: string;
  entityType?: string | null;
  name?: string | null;
  description?: string | null;
  canonicalId?: string | null;
  /** Increment applied to mention_count on conflict. */
  mentionDelta?: number;
}

export async function upsertEntityNode(e: EntityNodeWrite): Promise<void> {
  await withTransaction(async c => {
    await c.query(
      `INSERT INTO entity_nodes (entity_id, entity_type, name, description, canonical_id, mention_count, last_seen)
       VALUES ($1, $2, $3, $4, $5, $6, now())
       ON CONFLICT ON CONSTRAINT entity_nodes_pkey DO UPDATE SET
         entity_type   = COALESCE(EXCLUDED.entity_type, entity_nodes.entity_type),
         name          = COALESCE(EXCLUDED.name, entity_nodes.name),
         description   = COALESCE(EXCLUDED.description, entity_nodes.description),
         canonical_id  = COALESCE(EXCLUDED.canonical_id, entity_nodes.canonical_id),
         mention_count = entity_nodes.mention_count + $6,
         last_seen     = now()`,
      [e.entityId, e.entityType ?? null, e.name ?? null, e.description ?? null,
       e.canonicalId ?? null, e.mentionDelta ?? 1],
    );
  });
}

export interface MemoryEvolutionWrite {
  memoryId: number;
  triggeredBy: number;
  version?: number;
  previousKeywords?: string | null;
  newKeywords?: string | null;
  previousContext?: string | null;
  newContext?: string | null;
  reasoning?: string | null;
}

export async function insertMemoryEvolution(m: MemoryEvolutionWrite): Promise<number> {
  return withTransaction(async c => {
    const { rows } = await c.query<{ id: string }>(
      `INSERT INTO memory_evolution
         (memory_id, triggered_by, version, previous_keywords, new_keywords,
          previous_context, new_context, reasoning)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
       RETURNING id`,
      [m.memoryId, m.triggeredBy, m.version ?? 1,
       m.previousKeywords ?? null, m.newKeywords ?? null,
       m.previousContext ?? null, m.newContext ?? null, m.reasoning ?? null],
    );
    return Number(rows[0]!.id);
  });
}
