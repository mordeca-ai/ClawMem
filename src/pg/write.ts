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
import { withTransaction, toVectorLiteral } from "./client.ts";
import { embedDim, resolvePgConfig } from "./config.ts";
import { resolveVault, type Vault } from "./vaults.ts";
import {
  ContentGcVaultRefusedError,
  PgSchemaGeometryError,
  PgVecBatchModelMismatchError,
  PgVecBatchVaultMismatchError,
  PgVecDimensionMismatchError,
  PgVecWriteModelMismatchError,
  PgWrongDatabaseError,
} from "./errors.ts";

/** Arbitrary but stable key for the write-geometry advisory lock. */
const GEOMETRY_LOCK_KEY = 0x1a2b3c4d;

function embedEndpointLabel(): string {
  return process.env.CLAWMEM_EMBED_URL || "the local in-process embedder";
}

// ===========================================================================
// The belt: prove the SESSION is where the routing thinks it is
// ===========================================================================

/**
 * Assert this transaction's session is attached to the database the vault's
 * configuration names — INSIDE the transaction, before any INSERT.
 *
 * Routing (src/pg/vaults.ts) decides which vault a write belongs to; the pool
 * (src/pg/client.ts) decides which connection string serves that vault. Neither
 * can see whether the connection string actually lands where it says: libpq
 * connects to the role-named database when the URL carries no path, a stale
 * shell export outranks the config, a copy-pasted URL keeps the wrong /dbname.
 * Every one of those routes a private write into the general database with the
 * routing layer reporting success. So the session is asked directly.
 *
 * Cost is one round trip per write transaction over a loopback socket. The
 * alternative is a privacy defect that no test of the routing layer can catch.
 */
export async function assertVaultDatabase(c: PoolClient, vault: Vault): Promise<string> {
  const expected = resolvePgConfig(vault).database;
  const { rows } = await c.query<{ db: string }>("SELECT current_database() AS db");
  const actual = rows[0]?.db ?? "";
  if (actual !== expected) {
    throw new PgWrongDatabaseError(vault, expected, actual);
  }
  return actual;
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
    -- pgvector stores the dimension DIRECTLY in atttypmod, with none of the
    -- varlena +4 offset that varchar/numeric use. Subtracting 4 here silently
    -- yields 764 for a vector(768) column and refuses every legitimate write.
    SELECT atttypmod AS dims
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
  /**
   * REQUIRED (master-harness-0ynkd). A fragment's vectors must land in the same
   * vault as the document they belong to, and the only thing that determines a
   * vault is the collection (+ path). Making this optional would let a caller
   * omit it and get "sfw" by accident — the exact defect this bead exists to
   * kill — so it is required and TypeScript refuses the call site instead.
   */
  collection: string;
  /** Collection-relative source path, when known. Feeds the PRIVATE_ROOTS tripwire. */
  path?: string | null;
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

  // Vault FIRST: resolved from the batch's own members, before a connection is
  // even checked out. A batch is written in ONE transaction against ONE
  // database, so a batch spanning two vaults is not a thing that can be
  // honoured — and splitting it silently would route half the caller's content
  // somewhere they never asked for. Refuse it, loudly, in the same shape and
  // for the same reason as the intra-batch model-mismatch check below.
  const vaults = new Map<Vault, string[]>();
  for (const w of writes) {
    const v = resolveVault(w.collection, w.path ?? undefined);
    const seen = vaults.get(v);
    if (seen) seen.push(w.collection);
    else vaults.set(v, [w.collection]);
  }
  const distinct = [...vaults.keys()];
  if (distinct.length > 1) {
    throw new PgVecBatchVaultMismatchError(
      distinct[0]!,
      distinct[1]!,
      [...new Set(writes.map(w => w.collection))].sort(),
    );
  }
  const vault = distinct[0]!;

  await withTransaction(vault, async c => {
    await assertVaultDatabase(c, vault);
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
/**
 * The admitted content_type enum — 17 real types + the 'unknown' sink.
 *
 * Kept ALPHABETICALLY ORDERED and set-equal to the CHECK constraint in
 * migrations/003_content_type_enum_extend.sql. That is not a convention, it is
 * a tested invariant: tests/integration/pg-content-type-enum.test.ts parses the
 * value list out of 003 and asserts set-equality, because a value admitted here
 * but not there is a runtime constraint violation on the write path, and a
 * value admitted there but not here is silently narrowed to 'unknown'.
 *
 * eval-run / observation / plan / retro were added by the ADR-0058 amendment of
 * 2026-09-02 (operator verdict, BJ 2026-09-02, Review Hub item 2940ea7c).
 */
export const CONTENT_TYPES = [
  "antipattern", "conversation", "decision", "deductive", "eval-run", "handoff", "hub",
  "milestone", "note", "observation", "plan", "preference", "problem", "progress",
  "project", "research", "retro", "unknown",
] as const;
export type ContentTypeFacet = (typeof CONTENT_TYPES)[number];

export const AUDIENCES = ["operator", "household", "external", "agent-internal", "unknown"] as const;
export const TRUST_TIERS = ["authored", "distilled", "ingested-verbatim", "derived", "unknown"] as const;
export const SENSITIVITIES = ["public", "private", "crypt", "nsfw", "unknown"] as const;

/**
 * Observed synonyms that CONFORM onto an admitted value (ADR-0058 amendment,
 * 2026-09-02; operator verdict BJ 2026-09-02, Review Hub item 2940ea7c).
 *
 * Mirrored by the backfill VALUES list in
 * migrations/003_content_type_enum_extend.sql. Two invariants are tested, not
 * merely intended:
 *
 *  - every VALUE here is a member of CONTENT_TYPES (a mapping onto a
 *    non-admitted value would produce a row the CHECK constraint rejects);
 *  - no KEY here is a member of CONTENT_TYPES (a key that is already admitted
 *    is unreachable — rule 2 of narrowContentType wins first — so its presence
 *    means someone made a mistake, not that the mapping is redundant).
 *
 * NOT PRESENT ON PURPOSE: `session-transcript`. It belongs to the `monchujo`
 * origin corpus owned by master-harness-vn4rz.8 and must keep landing as
 * 'unknown' with its raw preserved. Do not add it.
 */
export const CONTENT_TYPE_CONFORM: Readonly<Record<string, ContentTypeFacet>> = {
  "planning": "plan",
  "queue-plan": "plan",
  "run": "eval-run",
  "run-report": "eval-run",
  "synthesis": "deductive",
  "research-synthesis": "deductive",
  "memo": "deductive",
  "reference": "hub",
  "runbook": "hub",
  "operations": "handoff",
};

/**
 * Narrow a raw content_type onto the admitted enum, with a CONFORM layer.
 *
 * The enum is no longer closed-by-retag. ADR-0162 §5 said "gaps resolve by
 * retag, not extension"; the ADR-0058 amendment of 2026-09-02 (operator
 * verdict, BJ 2026-09-02, Review Hub item 2940ea7c) REVERSED that: the enum
 * extends, and observed synonyms conform onto one admitted value each.
 *
 * Precedence, in order:
 *
 *  1. falsy raw            -> 'unknown', raw null (nothing was ever declared)
 *  2. raw IS admitted      -> itself, raw null (no mapping decision was made,
 *                             so there is nothing to keep an audit trail of)
 *  3. raw is a conform key -> the MAPPED value, and the ORIGINAL string is
 *                             preserved in content_type_raw. The conform is a
 *                             mapping decision; it stays auditable and
 *                             reversible precisely because the raw survives.
 *  4. otherwise            -> 'unknown', raw preserved (the real retag backlog)
 *
 * Nothing is ever silently discarded in any branch.
 */
export function narrowContentType(raw: string | null | undefined): {
  contentType: ContentTypeFacet;
  raw: string | null;
} {
  if (!raw) return { contentType: "unknown", raw: null };
  if ((CONTENT_TYPES as readonly string[]).includes(raw)) {
    return { contentType: raw as ContentTypeFacet, raw: null };
  }
  const conformed = CONTENT_TYPE_CONFORM[raw];
  if (conformed !== undefined) return { contentType: conformed, raw };
  return { contentType: "unknown", raw };
}

/**
 * Is this narrowing result part of the retag backlog ADR-0162 §3 makes a MONITORED
 * METRIC?
 *
 * This predicate lives beside `narrowContentType` deliberately: it is the same
 * decision, and the ADR-0058 2026-09-02 amendment is exactly what pulled the two
 * apart. Before the conform layer, "carries a raw" and "needs a retag" were the same
 * condition, so the metric could read `.raw` directly. They are no longer the same —
 * a CONFORMED row also keeps its raw, as the audit trail of the mapping decision, but
 * it is resolved rather than backlog. A metric that kept reading `.raw` would count
 * every conformed row as backlog and start silently lying about the one number the
 * ADR asks us to watch. Keeping the predicate here means the next edit to the conform
 * rules has to walk past its own metric.
 *
 * The backlog is exactly the residue: landed in the `unknown` sink WITH a raw string,
 * i.e. neither admitted nor conformed (`session-transcript` today).
 */
export function isRetagBacklog(narrowed: { contentType: ContentTypeFacet; raw: string | null }): boolean {
  return narrowed.contentType === "unknown" && narrowed.raw !== null;
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

  // THE WRITE PATH ROUTES; the caller does not get a say (master-harness-0ynkd).
  // upsertDocument resolves the vault ITSELF rather than taking one, because a
  // vault parameter is a decision a caller can get wrong — and every caller
  // getting it right forever is not a property anyone can verify. Routing here
  // means there is exactly one place the decision is made, and exactly one place
  // to audit it.
  const vault = resolveVault(d.collection, d.path);

  return withTransaction(vault, async c => {
    await assertVaultDatabase(c, vault);
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
// The absent-from-walk sweep (master-harness-vn4rz.41)
// ===========================================================================

export interface DeactivateAbsentResult {
  collection: string;
  vault: Vault;
  /** Size of the DEDUPLICATED keep-set actually sent to the database. */
  keptPaths: number;
  deactivated: number;
  /** Sorted, so a diff of two runs is stable. */
  deactivatedPaths: string[];
}

/**
 * Deactivate every ACTIVE document row in `collection` whose path is not in
 * `keepPaths` (master-harness-vn4rz.41).
 *
 * WHY THIS EXISTS. reindexCollection walks the files that exist and upserts
 * them. Nothing ever retired a row whose source file had been DELETED, so
 * reindex converged in ONE direction only and PG monotonically accumulated
 * rows for files that no longer exist. Measured on the live vault: collection
 * `episodic-handoffs` held 123 active rows against 55 files on disk, and all 68
 * of the surplus were confirmed absent — pruned handoffs (master-harness
 * CLAUDE.md prunes a handoff when its work lands), which is why that collection
 * has the corpus's highest delete rate. Those 68 rows were live and retrievable
 * and indistinguishable at query time from content that still exists.
 *
 * SOFT, NEVER DESTRUCTIVE. The row is flipped to active=false with a
 * `deactivated_reason`; it is never DELETEd, and content_vectors is never
 * touched. The vector join already carries `AND d.active`, so retiring the
 * document row is sufficient to remove the content from retrieval, and keeping
 * the row means a file that comes back (a revert, a rename undone) upserts back
 * to active=true through the existing ON CONFLICT rather than losing history.
 *
 * THE EMPTY-KEEP-SET GUARD LIVES HERE, AT THE PRIMITIVE, NOT AT THE CALLER.
 * `keepPaths.length === 0` means "deactivate the entire collection", which is
 * exactly what a missing collection root, an unreadable mount or a glob that
 * matched nothing looks like from the walk's side — indistinguishable from a
 * genuine empty directory. reindex.ts refuses that case too (sweepDecision),
 * but a guard that lives only in one caller is a guard a second caller can
 * bypass by accident, so the refusal is duplicated at the primitive on purpose.
 */
export async function deactivateAbsentDocuments(
  collection: string,
  keepPaths: readonly string[],
  reason: string,
): Promise<DeactivateAbsentResult> {
  if (keepPaths.length === 0) {
    throw new Error(
      `deactivateAbsentDocuments refuses an EMPTY keep-set for collection ` +
      `${JSON.stringify(collection)}: a sweep with nothing to keep would ` +
      `deactivate the ENTIRE collection in one statement, and an empty walk is ` +
      `indistinguishable from a missing or unreadable collection root. That is ` +
      `a mass-deactivation hazard, not a legitimate convergence. Pass the walked ` +
      `file set, or do not sweep.`,
    );
  }
  // Deduplicate: a keep-set is a SET, and = ANY() over a list with repeats does
  // the same work twice for no benefit. Also makes keptPaths an honest number.
  const keep = [...new Set(keepPaths)];

  // Routed the same way upsertDocument routes, and for the same reason: the
  // write path decides, not the caller (master-harness-0ynkd). No relPath is
  // passed because the sweep is collection-scoped — resolveVault's relPath
  // argument only feeds the private-path tripwire and never changes which vault
  // is returned, so a collection resolves to exactly one vault.
  const vault = resolveVault(collection);

  return withTransaction(vault, async c => {
    await assertVaultDatabase(c, vault);
    const { rows } = await c.query<{ path: string }>(
      `UPDATE documents
          SET active = false,
              deactivated_reason = $3
        WHERE collection = $1
          AND active
          AND NOT (path = ANY($2::text[]))
        RETURNING path`,
      [collection, keep, reason],
    );
    const deactivatedPaths = rows.map(r => r.path).sort();
    return {
      collection,
      vault,
      keptPaths: keep.length,
      deactivated: deactivatedPaths.length,
      deactivatedPaths,
    };
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
  // Explicit "sfw": entity nodes carry no document body and today only the
  // general corpus produces them. Naming the vault rather than defaulting to it
  // is the point of withTransaction's required first parameter — when the nsfw
  // vault grows an entity graph, this line is where that decision gets made.
  await withTransaction("sfw", async c => {
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
  // Explicit "sfw" — see upsertEntityNode.
  return withTransaction("sfw", async c => {
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

// ===========================================================================
// Content GC (master-harness-vn4rz.49)
// ===========================================================================

/**
 * RETENTION RULE (master-harness-vn4rz.49) — read this before changing anything
 * below.
 *
 * A `content` row (and, through ON DELETE CASCADE, every `content_vectors` row
 * for its hash) is garbage ONLY when its hash is referenced by NO `documents`
 * row AND NO `origin_documents` row. Consequences, all deliberate:
 *
 *  - ANY documents row protects its content, INCLUDING active=false. The
 *    absent-from-walk sweep (vn4rz.41) soft-retires rows so a returning file
 *    reactivates with its vectors intact; GC honouring `active` would make that
 *    reactivation silently vectorless. This is also the sqlite parity rule
 *    (store.ts cleanupOrphanedContent: `hash NOT IN documents`, deliberately
 *    NOT active-scoped — active-scoping there cascade-destroyed archived docs).
 *  - origin_documents rows protect their content too: the origin tier shares
 *    the content table and cascades from it.
 *  - What IS collected: hashes superseded by a content change (upsertDocument
 *    re-points documents.hash and never deletes the old content row), which is
 *    how the live vault accumulated unreferenced vectors.
 *
 * SFW VAULT ONLY. A non-sfw vault is refused before any config is resolved or
 * any pool is opened (ContentGcVaultRefusedError).
 *
 * RACE GUARD. upsertDocument inserts content and documents in ONE transaction,
 * so a committed content row is never "not yet referenced" by its own writer.
 * The remaining windows are closed three ways:
 *   1. candidates are locked `FOR UPDATE OF c SKIP LOCKED` — a writer whose FK
 *      check already holds KEY SHARE on a content row (an in-flight documents /
 *      origin_documents / content_vectors insert) makes that row skipped, not
 *      waited on and not deleted;
 *   2. the DELETE re-checks both NOT EXISTS predicates under a fresh READ
 *      COMMITTED snapshot taken AFTER the row locks are held, so a reference
 *      committed between the candidate scan and the delete is seen and kept
 *      (a delete without the re-check would CASCADE-destroy that new document);
 *   3. a created_at grace window (default 15 min) keeps young content rows out
 *      of the candidate set, covering reindex's upsert -> embed gap where a
 *      concurrent writer could supersede a hash whose vectors are still pending.
 *
 * BOUNDED. Each batch is its own transaction of at most `batchSize` content
 * rows; a pass stops at `maxBatches` and reports `capped: true` so the next
 * pass continues. DELETE does not shrink the heap or the HNSW index — that
 * needs VACUUM / REINDEX CONCURRENTLY, which is out of scope here.
 */
export const CONTENT_GC_DEFAULTS = {
  batchSize: 500,
  maxBatches: 200,
  graceSeconds: 900,
} as const;

export interface ContentGcOptions {
  /** Must be "sfw". Anything else is refused before a connection exists. */
  vault?: Vault;
  batchSize?: number;
  maxBatches?: number;
  graceSeconds?: number;
  /** Count what WOULD be collected; delete nothing. */
  dryRun?: boolean;
}

export interface ContentGcResult {
  vault: Vault;
  dryRun: boolean;
  /** Eligible at the start of the pass (after the grace window). */
  eligibleContent: number;
  eligibleVectors: number;
  /** Unreferenced but younger than the grace window, so kept this pass. */
  withinGraceContent: number;
  contentDeleted: number;
  vectorsDeleted: number;
  batches: number;
  /** True when maxBatches stopped the pass before convergence. */
  capped: boolean;
}

const ORPHAN_PREDICATE = `
  NOT EXISTS (SELECT 1 FROM documents d WHERE d.hash = c.hash)
  AND NOT EXISTS (SELECT 1 FROM origin_documents o WHERE o.hash = c.hash)`;

function positiveInt(name: string, v: number, allowZero = false): number {
  if (!Number.isInteger(v) || v < 0 || (!allowZero && v === 0)) {
    throw new Error(`content GC: ${name} must be a ${allowZero ? "non-negative" : "positive"} integer, got ${v}`);
  }
  return v;
}

export async function gcOrphanedContent(opts: ContentGcOptions = {}): Promise<ContentGcResult> {
  const vault = opts.vault ?? "sfw";
  // Refuse FIRST — before resolvePgConfig, before a pool exists.
  if (vault !== "sfw") throw new ContentGcVaultRefusedError(vault);
  const batchSize = positiveInt("batchSize", opts.batchSize ?? CONTENT_GC_DEFAULTS.batchSize);
  const maxBatches = positiveInt("maxBatches", opts.maxBatches ?? CONTENT_GC_DEFAULTS.maxBatches);
  const graceSeconds = positiveInt(
    "graceSeconds", opts.graceSeconds ?? CONTENT_GC_DEFAULTS.graceSeconds, true,
  );
  const dryRun = opts.dryRun === true;

  const counts = await withTransaction(vault, async c => {
    await assertVaultDatabase(c, vault);
    const { rows } = await c.query<{ eligible: number; vectors: number; young: number }>(
      `SELECT
         count(*) FILTER (WHERE c.created_at <  now() - make_interval(secs => $1))::int AS eligible,
         count(*) FILTER (WHERE c.created_at >= now() - make_interval(secs => $1))::int AS young,
         COALESCE(sum((SELECT count(*) FROM content_vectors v WHERE v.hash = c.hash))
           FILTER (WHERE c.created_at < now() - make_interval(secs => $1)), 0)::int AS vectors
       FROM content c
      WHERE ${ORPHAN_PREDICATE}`,
      [graceSeconds],
    );
    return rows[0]!;
  });

  const result: ContentGcResult = {
    vault, dryRun,
    eligibleContent: counts.eligible,
    eligibleVectors: counts.vectors,
    withinGraceContent: counts.young,
    contentDeleted: 0, vectorsDeleted: 0, batches: 0, capped: false,
  };
  if (dryRun) return result;

  let converged = false;
  while (result.batches < maxBatches) {
    const batch = await withTransaction(vault, async c => {
      await assertVaultDatabase(c, vault);
      const locked = await c.query<{ hash: string }>(
        `SELECT c.hash FROM content c
          WHERE c.created_at < now() - make_interval(secs => $1)
            AND ${ORPHAN_PREDICATE}
          ORDER BY c.hash
          LIMIT $2
          FOR UPDATE OF c SKIP LOCKED`,
        [graceSeconds, batchSize],
      );
      if (locked.rows.length === 0) return { selected: 0, content: 0, vectors: 0 };
      const { rows } = await c.query<{ content_deleted: number; vectors_deleted: number }>(
        `WITH del AS (
           DELETE FROM content c
            WHERE c.hash = ANY($1::text[])
              AND ${ORPHAN_PREDICATE}
           RETURNING c.hash
         )
         SELECT (SELECT count(*) FROM del)::int AS content_deleted,
                (SELECT count(*) FROM content_vectors v
                  WHERE v.hash IN (SELECT hash FROM del))::int AS vectors_deleted`,
        [locked.rows.map(r => r.hash)],
      );
      return {
        selected: locked.rows.length,
        content: rows[0]!.content_deleted,
        vectors: rows[0]!.vectors_deleted,
      };
    });
    if (batch.selected === 0) { converged = true; break; }
    result.batches++;
    result.contentDeleted += batch.content;
    result.vectorsDeleted += batch.vectors;
    if (batch.selected < batchSize) { converged = true; break; }
  }
  result.capped = !converged;
  return result;
}
