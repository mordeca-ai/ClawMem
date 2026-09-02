/**
 * The ORIGIN tier loader + retention verb (master-harness-vn4rz.8, ADR-0162 §5/§6).
 *
 * ===========================================================================
 * THIS IS THE ONE SANCTIONED COPY. IT DOES NOT GENERALISE.
 *
 * ADR-0162 §6 splits the cutover: file-authored collections are REINDEXED
 * (src/pg/reindex.ts, and its header says reindex-not-copy is a HARD RULE);
 * ORIGIN data — monchujo transcripts, telemetry — is COPIED, because it has no
 * authoring substrate to rebuild from. The transcript record IS the artifact.
 *
 * Everything else in this arc stays reindex-not-copy. This file is the
 * exception §6 names, not a licence to widen it.
 * ===========================================================================
 *
 * WHY THIS IS NOT `reindex --collection monchujo`
 *
 * reindex.ts re-embeds EVERY document it walks, unconditionally. Pointed at
 * monchujo that is ~98,000 fragments against the yoshiee embed endpoint for a
 * corpus whose vectors ALREADY EXIST — content_vectors keys on the CONTENT
 * HASH, not on the document row, so moving a document between tiers costs no
 * re-embedding at all. This loader is therefore EMBED-INCREMENTAL: it embeds a
 * hash only when that hash has zero vector rows. On the migration of the 13,723
 * already-loaded rows that is exactly zero embed calls.
 *
 * It is also why the "undo" the clawmem-pg-parity docstring asks for is cheap:
 * dropping the monchujo rows out of `documents` does not touch `content` (the
 * FK runs documents -> content, not the other way) and therefore does not touch
 * `content_vectors` either.
 */

import { Glob } from "bun";
import { readFileSync } from "fs";
import { join } from "path";
import matter from "gray-matter";
import {
  extractTitle, hashContent, parseDocument,
} from "../indexer.ts";
import { getDefaultLlamaCpp, formatDocForEmbedding } from "../llm.ts";
import { splitDocument } from "../splitter.ts";
import { canonicalDocId } from "../store.ts";
import { withClient, withTransaction } from "./client.ts";
import { insertEmbeddingsBatch, type EmbeddingWrite } from "./write.ts";
import type { Vault } from "./vaults.ts";

/** How `authored_at` was obtained. Mirrors the CHECK in migration 004. */
export type AuthoredAtSource = "frontmatter-ts" | "frontmatter-authored-at" | "none";

/**
 * The origin tier's OWN closed record-type vocabulary — deliberately NOT the
 * ADR-0058 `content_type` enum. See migration 004's DESIGN DECISION (a) for the
 * full reasoning; the one-liner is that ADR-0058's enum selects a DECAY CURVE
 * for curated memory, and origin transcripts do not decay — they age out by
 * partition drop.
 *
 * Same discipline as `documents.content_type`, different vocabulary: closed set,
 * raw value preserved, explicit `unknown` sink, unknown count monitored.
 */
export const ORIGIN_RECORD_TYPES = ["session-transcript", "unknown"] as const;
export type OriginRecordType = (typeof ORIGIN_RECORD_TYPES)[number];

export function narrowRecordType(raw: string | null | undefined): {
  recordType: OriginRecordType;
  raw: string | null;
} {
  if (!raw) return { recordType: "unknown", raw: null };
  const t = raw.trim();
  return {
    recordType: (ORIGIN_RECORD_TYPES as readonly string[]).includes(t)
      ? (t as OriginRecordType)
      : "unknown",
    raw: t,
  };
}

/**
 * Resolve the partition key from a record's frontmatter.
 *
 * PRECEDENCE, and it is not arbitrary: `ts` first, because that is the key the
 * monchujo capture process actually writes and it names the moment the session
 * happened; `authored_at` second, so a future origin corpus using the house
 * frontmatter key is not silently unassigned; then nothing.
 *
 * A value that is present but does not parse resolves to `none` — it is NOT
 * coerced, NOT defaulted to now(), and NOT dropped. It lands in the DEFAULT
 * partition with `authored_at_source = 'none'`, which is a counted, printed,
 * never-dropped state rather than a silent one.
 */
/**
 * The RAW frontmatter map.
 *
 * `parseDocument` deliberately NARROWS frontmatter to the eight keys the
 * curated-document schema knows (title/description/tags/domain/workstream/
 * content_type/review_by/authored_at) and drops everything else. Origin records
 * carry a different key set — `ts`, `session_id`, `chunk_index`, `chunk_count`,
 * `source_path`, `cwd`, `git_branch`, `version` — so reading them through
 * parseDocument returns `undefined` for every one of them, which is exactly how
 * the first smoke pass put all five records in the DEFAULT partition.
 *
 * The fix is NOT to widen parseDocument: it is shared with the sqlite write path
 * and its narrowing is the contract that keeps the two parsers identical. Read
 * the raw map here instead, with the SAME `matter(content, {})` call shape —
 * the `{}` is load-bearing (see parseDocument's own comment: the no-options path
 * memoises a half-built empty result when the YAML parser throws).
 *
 * A frontmatter block that does not parse yields `{}`, so every origin key
 * resolves to absent and the record lands in the DEFAULT partition — counted and
 * printed, never silently dropped.
 */
export function rawFrontmatter(content: string): Record<string, unknown> {
  try {
    const { data } = matter(content, {});
    return (data ?? {}) as Record<string, unknown>;
  } catch {
    return {};
  }
}

export function resolveAuthoredAt(meta: Record<string, unknown>): {
  authoredAt: Date | null;
  source: AuthoredAtSource;
} {
  const attempts: [AuthoredAtSource, unknown][] = [
    ["frontmatter-ts", meta.ts],
    ["frontmatter-authored-at", meta.authored_at],
  ];
  for (const [source, v] of attempts) {
    if (v === undefined || v === null) continue;
    const d = v instanceof Date ? v : new Date(String(v));
    if (!isNaN(d.getTime())) return { authoredAt: d, source };
    // Present but unparseable: stop here rather than falling through to the
    // next key. Falling through would let a CORRUPT ts be silently replaced by
    // a different field's value, which is the "lands somewhere plausible and
    // wrong" failure the DEFAULT partition exists to make visible instead.
    return { authoredAt: null, source: "none" };
  }
  return { authoredAt: null, source: "none" };
}

/** UTC month key, `YYYY-MM-01`. The boundary rule of migration 004. */
export function utcMonthKey(d: Date): string {
  return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}-01`;
}

export interface OriginLoadOptions {
  collection: string;
  root: string;
  pattern: string;
  /** Restrict to ONE UTC month, `YYYY-MM`. This is the smoke-one-month-first arm. */
  month?: string;
  /** Stop after N records. */
  limit?: number;
  /** Do not embed at all (schema/count smoke with no yoshiee traffic). */
  skipEmbed?: boolean;
  embedBatchSize?: number;
  vault?: Vault;
  onProgress?: (m: string) => void;
}

export interface OriginLoadStats {
  collection: string;
  filesSeen: number;
  recordsWritten: number;
  /** Rows per UTC month partition, by partition name. */
  perMonth: Record<string, number>;
  /** Records that landed in the DEFAULT partition, and why. MONITORED, not ignored. */
  unassigned: number;
  unassignedPaths: string[];
  /** record_type values outside the closed origin vocabulary, with counts. */
  recordTypeBacklog: Record<string, number>;
  /** Hashes that already had vectors and were therefore NOT re-embedded. */
  hashesAlreadyEmbedded: number;
  fragmentsEmbedded: number;
  embedFailures: number;
  wallClockMs: number;
}

function expandBraces(pattern: string): string[] {
  const m = pattern.match(/^\{(.+)\}$/);
  return m ? m[1]!.split(",").map(s => s.trim()) : [pattern];
}

/**
 * Load one origin collection from its record files into `origin_documents`.
 *
 * Idempotent: the upsert targets `origin_documents_key` by name (never inferred
 * — an unnamed ON CONFLICT in Postgres binds to whatever index the planner
 * picks, which is how an upsert silently updates the wrong row).
 */
export async function loadOriginCollection(
  opts: OriginLoadOptions,
): Promise<OriginLoadStats> {
  const t0 = Date.now();
  const vault: Vault = opts.vault ?? "sfw";
  const log = opts.onProgress ?? (() => {});
  const stats: OriginLoadStats = {
    collection: opts.collection,
    filesSeen: 0, recordsWritten: 0, perMonth: {},
    unassigned: 0, unassignedPaths: [],
    recordTypeBacklog: {},
    hashesAlreadyEmbedded: 0, fragmentsEmbedded: 0, embedFailures: 0,
    wallClockMs: 0,
  };

  const files: string[] = [];
  for (const p of expandBraces(opts.pattern)) {
    for (const f of new Glob(p).scanSync({ cwd: opts.root, onlyFiles: true })) files.push(f);
  }
  files.sort();
  stats.filesSeen = files.length;

  // Which months already have a partition, so the ensure call is made once per
  // month rather than once per record.
  const ensured = new Set<string>();
  const ensureMonth = async (monthKey: string) => {
    if (ensured.has(monthKey)) return;
    await withClient(vault, c =>
      c.query("SELECT origin_documents_ensure_partition($1::date)", [monthKey]));
    ensured.add(monthKey);
  };

  interface PendingFragment {
    hash: string; seq: number; pos: number; text: string;
    fragmentType: string; fragmentLabel: string; canonicalId: string; relPath: string;
  }
  const pending: PendingFragment[] = [];
  const hashesDone = new Set<string>();
  const batchSize = opts.embedBatchSize ?? 50;
  const llm = opts.skipEmbed ? null : getDefaultLlamaCpp();

  const flush = async (force: boolean) => {
    if (!llm) { pending.length = 0; return; }
    while (pending.length >= batchSize || (force && pending.length > 0)) {
      const chunk = pending.splice(0, Math.min(batchSize, pending.length));
      const results = await llm.embedBatch(chunk.map(p => p.text));
      const writes: EmbeddingWrite[] = [];
      for (let i = 0; i < chunk.length; i++) {
        const r = results[i]; const frag = chunk[i]!;
        if (!r?.embedding) { stats.embedFailures++; continue; }
        writes.push({
          collection: opts.collection, path: frag.relPath,
          hash: frag.hash, seq: frag.seq, pos: frag.pos,
          embedding: r.embedding, model: r.model,
          fragmentType: frag.fragmentType, fragmentLabel: frag.fragmentLabel,
          canonicalId: frag.canonicalId, embedInputFp: hashContent(frag.text),
        });
      }
      if (writes.length) {
        // Dimension, model-consistency and vault checks all live inside
        // insertEmbeddingsBatch's transaction. Not re-checked here: one home.
        await insertEmbeddingsBatch(writes);
        stats.fragmentsEmbedded += writes.length;
      }
    }
  };

  let n = 0;
  for (const rel of files) {
    if (opts.limit !== undefined && n >= opts.limit) break;
    let raw: string;
    try { raw = readFileSync(join(opts.root, rel), "utf-8"); } catch { continue; }

    const { body, meta } = parseDocument(raw, rel);
    // Two reads of the same frontmatter, on purpose: `meta` for the narrowed
    // curated keys (title), `m` for the origin key set parseDocument drops.
    const m = rawFrontmatter(raw);
    const { authoredAt, source } = resolveAuthoredAt(m);

    // The one-month smoke arm. Filtering AFTER the timestamp resolve, so
    // `--month` selects on the same key the partition routes on.
    if (opts.month) {
      const want = `${opts.month}-01`;
      if (!authoredAt || utcMonthKey(authoredAt) !== want) continue;
    }

    const hash = hashContent(raw);
    const title = meta.title ?? extractTitle(raw, rel);
    const { recordType, raw: rtRaw } = narrowRecordType(m.content_type as string | undefined);
    if (recordType === "unknown" && rtRaw) {
      stats.recordTypeBacklog[rtRaw] = (stats.recordTypeBacklog[rtRaw] ?? 0) + 1;
    }

    if (authoredAt) {
      await ensureMonth(utcMonthKey(authoredAt));
      const pname = `origin_documents_${utcMonthKey(authoredAt).slice(0, 7).replace("-", "_")}`;
      stats.perMonth[pname] = (stats.perMonth[pname] ?? 0) + 1;
    } else {
      stats.unassigned++;
      if (stats.unassignedPaths.length < 50) stats.unassignedPaths.push(rel);
    }

    await withTransaction(vault, async c => {
      await c.query(
        `INSERT INTO content (hash, doc) VALUES ($1, $2)
         ON CONFLICT ON CONSTRAINT content_pkey DO UPDATE SET doc = EXCLUDED.doc`,
        [hash, raw],
      );
      await c.query(
        `INSERT INTO origin_documents (
           collection, path, title, hash, authored_at, authored_at_source,
           record_type, record_type_raw,
           session_id, chunk_index, chunk_count, source_path, cwd, git_branch, agent_version,
           domain, audience, trust_tier, provenance, sensitivity, active
         ) VALUES (
           $1,$2,$3,$4,$5::timestamptz,$6,
           $7,$8,
           $9,$10,$11,$12,$13,$14,$15,
           $16,$17,$18,$19::jsonb,$20,true
         )
         ON CONFLICT ON CONSTRAINT origin_documents_key DO UPDATE SET
           title              = EXCLUDED.title,
           hash               = EXCLUDED.hash,
           authored_at_source = EXCLUDED.authored_at_source,
           record_type        = EXCLUDED.record_type,
           record_type_raw    = EXCLUDED.record_type_raw,
           session_id         = EXCLUDED.session_id,
           chunk_index        = EXCLUDED.chunk_index,
           chunk_count        = EXCLUDED.chunk_count,
           source_path        = EXCLUDED.source_path,
           cwd                = EXCLUDED.cwd,
           git_branch         = EXCLUDED.git_branch,
           agent_version      = EXCLUDED.agent_version,
           provenance         = EXCLUDED.provenance,
           active             = true`,
        [
          opts.collection, rel, title, hash, authoredAt ?? null, source,
          recordType, rtRaw,
          (m.session_id as string) ?? null,
          m.chunk_index ?? null, m.chunk_count ?? null,
          (m.source_path as string) ?? null, (m.cwd as string) ?? null,
          (m.git_branch as string) ?? null, (m.version as string) ?? null,
          // Facets. `trust_tier` is 'ingested-verbatim' and that is a real
          // claim, not a default: a transcript is the captured artifact, neither
          // authored by us nor distilled from anything.
          "sessions", "agent-internal", "ingested-verbatim",
          JSON.stringify({
            assigned_by: "clawmem pg origin-load (master-harness-vn4rz.8)",
            assigned_at: new Date().toISOString(),
            collection: opts.collection,
            source_path: rel,
            rule:
              "origin tier: record_type from frontmatter content_type narrowed onto the " +
              "origin vocabulary (NOT the ADR-0058 enum); authored_at from " + source +
              "; partition = UTC month of authored_at, DEFAULT when unresolved",
            authored_at_source: source,
          }),
          "private",
        ],
      );
    });
    stats.recordsWritten++;
    n++;

    if (!hashesDone.has(hash)) {
      hashesDone.add(hash);
      // EMBED-INCREMENTAL. The vectors key on the content hash, so a hash the
      // vault already holds vectors for needs nothing. This is what makes the
      // 13,723-row migration cost ZERO embed calls -- and it is checked against
      // the database rather than assumed from "we reindexed it earlier".
      const { rows } = await withClient(vault, c =>
        c.query<{ n: string }>(
          "SELECT count(*)::text AS n FROM content_vectors WHERE hash = $1", [hash]));
      if (Number(rows[0]!.n) > 0) { stats.hashesAlreadyEmbedded++; continue; }
      if (opts.skipEmbed) continue;

      const fragments = splitDocument(body, { title });
      const canonicalId = canonicalDocId(opts.collection, rel);
      for (let seq = 0; seq < fragments.length; seq++) {
        const frag = fragments[seq]!;
        pending.push({
          hash, seq, pos: frag.startLine,
          text: formatDocForEmbedding(frag.content, frag.label || title),
          fragmentType: frag.type, fragmentLabel: frag.label, canonicalId, relPath: rel,
        });
      }
      if (pending.length >= batchSize) {
        await flush(false);
        log(`[${opts.collection}] ${stats.recordsWritten}/${files.length} records, ${stats.fragmentsEmbedded} fragments embedded`);
      }
    }
  }
  await flush(true);
  stats.wallClockMs = Date.now() - t0;
  return stats;
}

/**
 * THE UNDO the tools/clawmem-pg-parity docstring names: remove the origin rows
 * the vn4rz.7 reindex put into the UNPARTITIONED `documents` table.
 *
 * Guarded, not blind. It refuses unless every `documents` row for the
 * collection is already present in `origin_documents`, because the delete is
 * only safe once the copy is complete — and "I ran the loader first" is a claim
 * about my shell history, not about the database.
 *
 * `content` and `content_vectors` are untouched: the FK runs documents ->
 * content, so deleting a document row does not cascade into the body or its
 * vectors. That is what the origin rows now point at.
 */
export async function dropLegacyDocumentRows(
  collection: string,
  vault: Vault = "sfw",
  opts: { apply?: boolean } = {},
): Promise<{ inDocuments: number; inOrigin: number; missing: number; deleted: number }> {
  return withClient(vault, async c => {
    const { rows } = await c.query<{ ind: string; ino: string; missing: string }>(
      `SELECT
         (SELECT count(*) FROM documents WHERE collection = $1)::text        AS ind,
         (SELECT count(*) FROM origin_documents WHERE collection = $1)::text AS ino,
         (SELECT count(*) FROM documents d
            WHERE d.collection = $1
              AND NOT EXISTS (SELECT 1 FROM origin_documents o
                               WHERE o.collection = d.collection AND o.path = d.path))::text
                                                                             AS missing`,
      [collection],
    );
    const inDocuments = Number(rows[0]!.ind);
    const inOrigin = Number(rows[0]!.ino);
    const missing = Number(rows[0]!.missing);
    if (missing > 0) {
      throw new Error(
        `Refusing to drop ${collection} from documents: ${missing} of ${inDocuments} rows ` +
        `have no matching origin_documents row (by collection+path). The copy is not ` +
        `complete, so the delete would lose them. Run origin-load first.`,
      );
    }
    if (!opts.apply) return { inDocuments, inOrigin, missing, deleted: 0 };
    const del = await c.query("DELETE FROM documents WHERE collection = $1", [collection]);
    return { inDocuments, inOrigin, missing, deleted: del.rowCount ?? 0 };
  });
}

export interface PartitionRow {
  partition_name: string;
  bound_expr: string;
  is_default: boolean;
  approx_rows: number;
  total_bytes: number;
}

export async function listPartitions(vault: Vault = "sfw"): Promise<PartitionRow[]> {
  return withClient(vault, async c => (await c.query<PartitionRow>(
    "SELECT * FROM origin_partitions")).rows);
}

/**
 * RETENTION — ADR-0162 §5. DETACH CONCURRENTLY, then DROP. Never a DELETE sweep.
 *
 * Why this is a CLI verb and not a plpgsql function: ALTER TABLE ... DETACH
 * PARTITION ... CONCURRENTLY cannot run inside a transaction block, and every
 * plpgsql function body is one. A function could offer only the plain DETACH,
 * which takes ACCESS EXCLUSIVE on the parent — precisely the lock cost §5 cites
 * the concurrent form to avoid. So the verb runs here, outside a transaction.
 *
 * TWO REFUSALS, both deliberate:
 *  - The DEFAULT partition is NEVER dropped. It has no upper bound, so dropping
 *    it is unbounded data loss wearing a retention step's clothes.
 *  - A partition is dropped only when its UPPER bound is at or before the
 *    cutoff, read from the CATALOG (origin_partitions.bound_expr), never parsed
 *    back out of the partition NAME. A name is a label; the bound is the truth.
 *
 * DRY RUN BY DEFAULT. `apply` is opt-in.
 */
export async function dropPartitionsBefore(
  cutoffIsoDate: string,
  vault: Vault = "sfw",
  opts: { apply?: boolean } = {},
): Promise<{ dropped: string[]; kept: string[]; refusedDefault: string[] }> {
  const cutoff = new Date(`${cutoffIsoDate}T00:00:00Z`);
  if (isNaN(cutoff.getTime())) throw new Error(`unparseable cutoff: ${cutoffIsoDate}`);
  const parts = await listPartitions(vault);
  const dropped: string[] = [], kept: string[] = [], refusedDefault: string[] = [];

  for (const p of parts) {
    if (p.is_default) { refusedDefault.push(p.partition_name); continue; }
    // `FOR VALUES FROM ('...') TO ('...')` — the upper bound is the second
    // quoted literal. Read from pg_get_expr, i.e. from the catalog.
    const bounds = [...p.bound_expr.matchAll(/'([^']+)'/g)].map(x => x[1]!);
    if (bounds.length !== 2) { kept.push(p.partition_name); continue; }
    const upper = new Date(bounds[1]!);
    if (isNaN(upper.getTime()) || upper > cutoff) { kept.push(p.partition_name); continue; }
    dropped.push(p.partition_name);
    if (opts.apply) {
      await withClient(vault, async c => {
        await c.query(
          `ALTER TABLE origin_documents DETACH PARTITION ${quoteIdent(p.partition_name)} CONCURRENTLY`);
        await c.query(`DROP TABLE ${quoteIdent(p.partition_name)}`);
      });
    }
  }
  return { dropped, kept, refusedDefault };
}

/** Identifiers here come from pg_class, but quoting is not optional on the way back in. */
function quoteIdent(name: string): string {
  return `"${name.replaceAll('"', '""')}"`;
}
