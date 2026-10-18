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
import { basename, join } from "path";
import matter from "gray-matter";
import {
  extractTitle, hashContent, parseDocument, type DocumentMeta,
} from "../indexer.ts";
import type { FrontmatterMap } from "../frontmatter-map.ts";
import { getDefaultLlamaCpp, formatDocForEmbedding } from "../llm.ts";
import { splitDocument } from "../splitter.ts";
import { canonicalDocId } from "../store.ts";
import { withClient, withTransaction } from "./client.ts";
import {
  assertVaultDatabase, insertEmbeddingsBatch, ORPHAN_PREDICATE, type EmbeddingWrite,
} from "./write.ts";
import { resolveVault, type Vault } from "./vaults.ts";
import {
  classifyText, isContentClassified, SessionClassifier, sessionKeyFor, type SessionVerdict,
} from "./classify.ts";
import { embedInputFingerprint } from "../embed-fingerprint.ts";

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
  /**
   * master-harness-wzwh8.1: the collection's config frontmatter_map. On this
   * path only the TITLE mapping has an effect — record_type is read from the
   * raw frontmatter against the origin tier's own vocabulary, never from the
   * ADR-0058 content_type the map targets.
   */
  frontmatterMap?: FrontmatterMap;
}

/**
 * The SOLE parseDocument call on the origin-load path (master-harness-wzwh8.1),
 * exported so the frontmatter_map threading is testable without a Postgres.
 */
export function parseOriginDocument(
  raw: string,
  rel: string,
  frontmatterMap?: FrontmatterMap,
): { body: string; meta: DocumentMeta } {
  const { body, meta } = parseDocument(raw, rel, undefined, frontmatterMap);
  return { body, meta };
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
  // Index-time content classification (master-harness-vn4rz.66). All zero unless
  // the collection is in CONTENT_CLASSIFIED_COLLECTIONS and the requested vault
  // is sfw. Counted over the documents THIS run wrote (the `--month` / `--limit`
  // filters apply), while each session's verdict is taken over every file the
  // pattern matches.
  /** Documents written to the nsfw vault because their session was classified nsfw. */
  classifiedNsfwDocs: number;
  /** Distinct nsfw-classified sessions among the documents written. */
  classifiedNsfwSessions: number;
  /** Distinct sfw sessions with at least one AMBIGUOUS signal (reported, never routed). */
  ambiguousSessions: number;
  /** What the sfw residue removal deleted for nsfw-routed documents. */
  sfwResidueRemoved: SfwResidueRemoved;
  wallClockMs: number;
}

export interface SfwResidueRemoved {
  originRows: number;
  contentRows: number;
  vectorRows: number;
}

function expandBraces(pattern: string): string[] {
  const m = pattern.match(/^\{(.+)\}$/);
  return m ? m[1]!.split(",").map(s => s.trim()) : [pattern];
}

export interface ExpectedOriginWrite {
  path: string;
  title: string;
  hash: string;
  authoredAt: Date | null;
  sessionId: string | null;
}

export interface ContentWriteReadBack {
  hash?: string | null;
  doc_sha?: string | null;
}

export interface OriginWriteReadBack {
  path?: string | null;
  title?: string | null;
  hash?: string | null;
  authored_at?: Date | string | null;
  session_id?: string | null;
}

export interface OriginWriteReadBackDetails {
  expectedHash?: string | null;
  contentHash?: string | null;
  originHash?: string | null;
  docSha?: string | null;
}

function authoredAtMatches(
  expected: Date | null,
  got: Date | string | null | undefined,
): boolean {
  if (expected === null) {
    return got == null;
  }
  if (got == null) {
    return false;
  }
  const expMs = expected.getTime();
  const gotMs = got instanceof Date ? got.getTime() : new Date(got).getTime();
  if (Number.isNaN(expMs) || Number.isNaN(gotMs)) return false;
  return expMs === gotMs;
}

/**
 * Pure read-back check comparing expected write parameters against rows returned
 * from PostgreSQL RETURNING clauses (master-harness-vn4rz.109). Returns an empty array
 * if all columns match, or the list of mismatching column names.
 */
export function checkOriginWriteReadBack(
  expected: ExpectedOriginWrite,
  contentRow?: ContentWriteReadBack | null,
  originRow?: OriginWriteReadBack | null,
): string[] {
  const mismatches: string[] = [];

  if (originRow?.path !== expected.path) {
    mismatches.push("path");
  }

  if (originRow?.title !== expected.title) {
    mismatches.push("title");
  }

  if (contentRow?.hash !== expected.hash || originRow?.hash !== expected.hash) {
    mismatches.push("hash");
  }

  if (!authoredAtMatches(expected.authoredAt, originRow?.authored_at)) {
    mismatches.push("authored_at");
  }

  const expSession = expected.sessionId ?? null;
  const gotSession = originRow ? (originRow.session_id ?? null) : null;
  if (!originRow || expSession !== gotSession) {
    mismatches.push("session_id");
  }

  if (contentRow?.doc_sha !== expected.hash) {
    mismatches.push("doc_sha");
  }

  return mismatches;
}

/**
 * Thrown when PostgreSQL RETURNING verification detects stale or corrupted parameters
 * (master-harness-vn4rz.109). Rolling back prevents unverified writes from committing.
 * To preserve privacy, title text, session IDs, and doc bodies are NEVER included.
 */
export class OriginWriteReadBackError extends Error {
  readonly collection: string;
  readonly path: string;
  readonly mismatches: readonly string[];

  constructor(
    collection: string,
    path: string,
    mismatches: string[],
    details?: OriginWriteReadBackDetails,
  ) {
    const hexDetails: string[] = [];
    const exp8 = details?.expectedHash ? details.expectedHash.slice(0, 8) : undefined;
    if (mismatches.includes("hash")) {
      if (details?.contentHash && exp8 && details.contentHash.slice(0, 8) !== exp8) {
        hexDetails.push(`content.hash expected ${exp8} got ${details.contentHash.slice(0, 8)}`);
      }
      if (details?.originHash && exp8 && details.originHash.slice(0, 8) !== exp8) {
        hexDetails.push(`origin.hash expected ${exp8} got ${details.originHash.slice(0, 8)}`);
      }
      if (hexDetails.length === 0 && (exp8 || details?.originHash || details?.contentHash)) {
        const got8 = (details?.originHash ?? details?.contentHash)?.slice(0, 8) ?? "none";
        hexDetails.push(`hash expected ${exp8 ?? "none"} got ${got8}`);
      }
    }
    if (mismatches.includes("doc_sha")) {
      const gotDocSha8 = details?.docSha ? details.docSha.slice(0, 8) : "none";
      hexDetails.push(`doc_sha expected ${exp8 ?? "none"} got ${gotDocSha8}`);
    }

    const detailSuffix = hexDetails.length > 0 ? ` (${hexDetails.join(", ")})` : "";
    super(
      `Origin write read-back mismatch for collection "${collection}", path "${path}": ` +
        `columns [${mismatches.join(", ")}]${detailSuffix}`,
    );
    this.name = "OriginWriteReadBackError";
    this.collection = collection;
    this.path = path;
    this.mismatches = mismatches;
  }
}

/** Every file the collection pattern matches under `root`, sorted. */
export function listOriginFiles(root: string, pattern: string): string[] {
  const files: string[] = [];
  for (const p of expandBraces(pattern)) {
    for (const f of new Glob(p).scanSync({ cwd: root, onlyFiles: true })) files.push(f);
  }
  return files.sort();
}

/** A record's frontmatter `session_id` as a string, or null when absent. */
export function frontmatterSessionId(m: Record<string, unknown>): string | null {
  const v = m.session_id;
  if (typeof v === "string") return v;
  if (typeof v === "number") return String(v);
  return null;
}

/**
 * The classification PRE-PASS (master-harness-vn4rz.66): one verdict per session
 * over EVERY file the pattern matches, plus each file's session key. Streams the
 * files through SessionClassifier so no transcript text is retained.
 */
export function classifyOriginFiles(
  root: string,
  files: string[],
): { sessionOf: Map<string, string>; verdicts: Map<string, SessionVerdict> } {
  const sc = new SessionClassifier();
  const sessionOf = new Map<string, string>();
  for (const rel of files) {
    let raw: string;
    try { raw = readFileSync(join(root, rel), "utf-8"); } catch { continue; }
    sessionOf.set(rel, sc.add({ rel, raw, sessionId: frontmatterSessionId(rawFrontmatter(raw)) }));
  }
  return { sessionOf, verdicts: sc.verdicts() };
}

/** One session in the `origin-classify` report. Content-free by construction. */
export interface OriginClassifySession {
  session: string;
  vault: Vault;
  docs: number;
  private: string[];
  ambiguous: string[];
  /** Chunk basenames, aligned with `hashes`. */
  chunks: string[];
  /** hashContent of each chunk's raw file — the same key the loader writes. */
  hashes: string[];
}

export interface OriginClassifyReport {
  collection: string;
  sessions: OriginClassifySession[];
  totals: {
    files: number;
    sessions: number;
    nsfwSessions: number;
    sfwSessions: number;
    ambiguousSessions: number;
    nsfwDocs: number;
    sfwDocs: number;
  };
}

/**
 * The `origin-classify` dry run: what the loader WOULD route, per session,
 * touching no database. Output carries session keys, chunk basenames, signal ids,
 * counts and content hashes — never text, titles or directory paths.
 */
export function classifyOriginCollection(opts: {
  collection: string;
  root: string;
  pattern: string;
}): OriginClassifyReport {
  const files = listOriginFiles(opts.root, opts.pattern);
  const { verdicts } = classifyOriginFiles(opts.root, files);
  const bySession = new Map<string, { chunks: string[]; hashes: string[] }>();
  for (const rel of files) {
    let raw: string;
    try { raw = readFileSync(join(opts.root, rel), "utf-8"); } catch { continue; }
    const key = sessionKeyFor(rel, frontmatterSessionId(rawFrontmatter(raw)));
    let e = bySession.get(key);
    if (!e) { e = { chunks: [], hashes: [] }; bySession.set(key, e); }
    e.chunks.push(basename(rel));
    e.hashes.push(hashContent(raw));
  }
  const sessions: OriginClassifySession[] = [];
  for (const key of [...bySession.keys()].sort()) {
    const v = verdicts.get(key)!;
    const e = bySession.get(key)!;
    sessions.push({
      session: key, vault: v.vault, docs: e.chunks.length,
      private: v.private, ambiguous: v.ambiguous, chunks: e.chunks, hashes: e.hashes,
    });
  }
  const nsfw = sessions.filter(s => s.vault === "nsfw");
  const sfw = sessions.filter(s => s.vault === "sfw");
  return {
    collection: opts.collection,
    sessions,
    totals: {
      files: files.length,
      sessions: sessions.length,
      nsfwSessions: nsfw.length,
      sfwSessions: sfw.length,
      ambiguousSessions: sfw.filter(s => s.ambiguous.length > 0).length,
      nsfwDocs: nsfw.reduce((n, s) => n + s.docs, 0),
      sfwDocs: sfw.reduce((n, s) => n + s.docs, 0),
    },
  };
}

/**
 * SFW RESIDUE REMOVAL (master-harness-vn4rz.66) — what makes a later-flagged
 * session self-heal. For documents this run has just written to the nsfw vault,
 * in ONE sfw transaction:
 *
 *   1. delete the sfw origin_documents rows for (collection, path);
 *   2. take as candidates both the hashes those rows pointed at (an earlier
 *      revision of the chunk may have hashed differently) and the current ones;
 *   3. lock the candidates that NO remaining sfw documents / origin_documents
 *      row references (FOR UPDATE: a concurrent writer's FK check now waits
 *      rather than racing the delete), then delete their content_vectors and
 *      content rows. A hash still referenced by any sfw row is kept.
 *
 * Only ever called for documents already committed to nsfw, so the delete can
 * never be the only copy.
 */
export async function removeSfwResidue(
  collection: string,
  docs: { rel: string; hash: string }[],
): Promise<SfwResidueRemoved> {
  const out: SfwResidueRemoved = { originRows: 0, contentRows: 0, vectorRows: 0 };
  if (docs.length === 0) return out;
  return withTransaction("sfw", async c => {
    await assertVaultDatabase(c, "sfw");
    const del = await c.query<{ hash: string }>(
      `DELETE FROM origin_documents WHERE collection = $1 AND path = ANY($2::text[]) RETURNING hash`,
      [collection, docs.map(d => d.rel)],
    );
    out.originRows = del.rowCount ?? del.rows.length;
    const candidates = [...new Set([...del.rows.map(r => r.hash), ...docs.map(d => d.hash)])].sort();
    const locked = await c.query<{ hash: string }>(
      `SELECT c.hash FROM content c
        WHERE c.hash = ANY($1::text[])
          AND ${ORPHAN_PREDICATE}
        ORDER BY c.hash
        FOR UPDATE OF c`,
      [candidates],
    );
    const orphans = locked.rows.map(r => r.hash);
    if (orphans.length === 0) return out;
    const vec = await c.query(`DELETE FROM content_vectors WHERE hash = ANY($1::text[])`, [orphans]);
    out.vectorRows = vec.rowCount ?? 0;
    const con = await c.query(
      `DELETE FROM content c WHERE c.hash = ANY($1::text[]) AND ${ORPHAN_PREDICATE}`, [orphans]);
    out.contentRows = con.rowCount ?? 0;
    return out;
  });
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
    classifiedNsfwDocs: 0, classifiedNsfwSessions: 0, ambiguousSessions: 0,
    sfwResidueRemoved: { originRows: 0, contentRows: 0, vectorRows: 0 },
    wallClockMs: 0,
  };

  const files = listOriginFiles(opts.root, opts.pattern);
  stats.filesSeen = files.length;

  // INDEX-TIME CLASSIFICATION (master-harness-vn4rz.66; Review Hub a7981735).
  // Only for a content-classified collection loaded into the sfw vault: an
  // explicit `--vault nsfw`, or any other collection, takes the unchanged path.
  // The pre-pass walks EVERY matched file, ignoring --month/--limit, so a
  // session's verdict never depends on which of its chunks this run selected.
  const classified = isContentClassified(opts.collection) && vault === "sfw";
  const pre = classified
    ? classifyOriginFiles(opts.root, files)
    : { sessionOf: new Map<string, string>(), verdicts: new Map<string, SessionVerdict>() };
  const nsfwSessionsWritten = new Set<string>();
  const ambiguousSessionsWritten = new Set<string>();
  const residue: { rel: string; hash: string }[] = [];
  const RESIDUE_BATCH = 200;
  const flushResidue = async () => {
    if (residue.length === 0) return;
    const r = await removeSfwResidue(opts.collection, residue.splice(0, residue.length));
    stats.sfwResidueRemoved.originRows += r.originRows;
    stats.sfwResidueRemoved.contentRows += r.contentRows;
    stats.sfwResidueRemoved.vectorRows += r.vectorRows;
  };

  // Which (vault, month) pairs already have a partition, so the ensure call is
  // made once per month per vault rather than once per record.
  const ensured = new Set<string>();
  const ensureMonth = async (v: Vault, monthKey: string) => {
    const key = `${v}|${monthKey}`;
    if (ensured.has(key)) return;
    await withClient(v, c =>
      c.query("SELECT origin_documents_ensure_partition($1::date)", [monthKey]));
    ensured.add(key);
  };

  interface PendingFragment {
    hash: string; seq: number; pos: number; text: string;
    fragmentType: string; fragmentLabel: string | null; canonicalId: string; relPath: string;
    /** Content-routing hint for the vectors; set only for nsfw-classified documents. */
    vault?: Vault;
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
      // Grouped by routing hint: one embed batch can hold fragments of an
      // nsfw-classified document and of an sfw one, and insertEmbeddingsBatch
      // (rightly) refuses a batch that spans two vaults. Without classification
      // there is exactly one group, so the write is unchanged.
      const groups = new Map<string, EmbeddingWrite[]>();
      for (let i = 0; i < chunk.length; i++) {
        const r = results[i]; const frag = chunk[i]!;
        if (!r?.embedding) { stats.embedFailures++; continue; }
        const w: EmbeddingWrite = {
          collection: opts.collection, path: frag.relPath,
          hash: frag.hash, seq: frag.seq, pos: frag.pos,
          embedding: r.embedding, model: r.model, endpoint: r.endpoint,
          fragmentType: frag.fragmentType, fragmentLabel: frag.fragmentLabel,
          canonicalId: frag.canonicalId, embedInputFp: embedInputFingerprint(frag.text, r),
        };
        if (frag.vault) w.vault = frag.vault;
        const gk = frag.vault ?? "";
        const g = groups.get(gk);
        if (g) g.push(w); else groups.set(gk, [w]);
      }
      for (const writes of groups.values()) {
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

    const { body, meta } = parseOriginDocument(raw, rel, opts.frontmatterMap);
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

    // Where THIS document goes. Unclassified: the requested vault, as before.
    // Classified: the session verdict, through resolveVault's escalation-only
    // hint — the same function that routes the document's vectors, so row and
    // vectors cannot disagree, and the tripwire still runs.
    let docVault: Vault = vault;
    let verdict: SessionVerdict | undefined;
    let sessionKey = "";
    if (classified) {
      sessionKey = pre.sessionOf.get(rel) ?? sessionKeyFor(rel, frontmatterSessionId(m));
      verdict = pre.verdicts.get(sessionKey);
      if (!verdict) {
        // The file appeared between the pre-pass and now: judge it alone.
        const c = classifyText(raw);
        verdict = { vault: c.private.length ? "nsfw" : "sfw", private: c.private, ambiguous: c.ambiguous };
      }
      docVault = resolveVault(opts.collection, rel, verdict.vault);
    }
    const { recordType, raw: rtRaw } = narrowRecordType(m.content_type as string | undefined);
    if (recordType === "unknown" && rtRaw) {
      stats.recordTypeBacklog[rtRaw] = (stats.recordTypeBacklog[rtRaw] ?? 0) + 1;
    }

    if (authoredAt) {
      await ensureMonth(docVault, utcMonthKey(authoredAt));
      const pname = `origin_documents_${utcMonthKey(authoredAt).slice(0, 7).replace("-", "_")}`;
      stats.perMonth[pname] = (stats.perMonth[pname] ?? 0) + 1;
    } else {
      stats.unassigned++;
      if (stats.unassignedPaths.length < 50) stats.unassignedPaths.push(rel);
    }

    await withTransaction(docVault, async c => {
      // Belt for the new route: a classified document headed for nsfw verifies
      // the connection really is the nsfw database before anything is written.
      if (classified && docVault === "nsfw") await assertVaultDatabase(c, "nsfw");
      const { rows: contentRows } = await c.query<{ hash: string; doc_sha: string }>(
        `INSERT INTO content (hash, doc) VALUES ($1, $2)
         ON CONFLICT ON CONSTRAINT content_pkey DO UPDATE SET doc = EXCLUDED.doc
         RETURNING hash, encode(sha256(convert_to(doc, 'UTF8')), 'hex') AS doc_sha`,
        [hash, raw],
      );
      const { rows: originRows } = await c.query<{
        path: string;
        title: string;
        hash: string;
        authored_at: Date | string | null;
        session_id: string | null;
      }>(
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
           active             = true
         RETURNING path, title, hash, authored_at, session_id`,
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
            // Signal ids only — never the matched text.
            ...(verdict ? {
              content_classification: {
                rule: "per-session boundary signals (master-harness-vn4rz.66, Review Hub a7981735)",
                vault: docVault,
                private: verdict.private,
                ambiguous: verdict.ambiguous,
              },
            } : {}),
          }),
          "private",
        ],
      );

      // master-harness-vn4rz.109: read-back guard against stale params observed below
      // the JS layer (driver/wire/runtime). Fail-closed so stale writes never commit.
      const contentRow = contentRows[0];
      const originRow = originRows[0];
      const mismatches = checkOriginWriteReadBack(
        // session_id goes over the wire as text whatever YAML typed it (a bare
        // number parses as one), so compare its text form, not the raw value.
        {
          path: rel, title, hash, authoredAt: authoredAt ?? null,
          sessionId: m.session_id == null ? null : String(m.session_id),
        },
        contentRow,
        originRow,
      );
      if (mismatches.length > 0) {
        throw new OriginWriteReadBackError(
          opts.collection,
          rel,
          mismatches,
          {
            expectedHash: hash,
            contentHash: contentRow?.hash,
            originHash: originRow?.hash,
            docSha: contentRow?.doc_sha,
          },
        );
      }
    });
    stats.recordsWritten++;
    n++;

    if (classified && verdict) {
      if (docVault === "nsfw") {
        if (verdict.vault === "nsfw") {
          stats.classifiedNsfwDocs++;
          nsfwSessionsWritten.add(sessionKey);
        }
        residue.push({ rel, hash });
        if (residue.length >= RESIDUE_BATCH) await flushResidue();
      } else if (verdict.ambiguous.length > 0) {
        ambiguousSessionsWritten.add(sessionKey);
      }
    }

    // Keyed by vault as well as hash: a hash the sfw vault already covers says
    // nothing about the nsfw vault a classified document now routes to.
    const doneKey = `${docVault}|${hash}`;
    if (!hashesDone.has(doneKey)) {
      hashesDone.add(doneKey);
      // EMBED-INCREMENTAL. The vectors key on the content hash, so a hash the
      // vault already holds vectors for needs nothing. This is what makes the
      // 13,723-row migration cost ZERO embed calls -- and it is checked against
      // the database rather than assumed from "we reindexed it earlier". The
      // coverage read goes to the vault THIS document routes to.
      const { rows } = await withClient(docVault, c =>
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
          ...(classified && docVault === "nsfw" ? { vault: "nsfw" as const } : {}),
        });
      }
      if (pending.length >= batchSize) {
        await flush(false);
        log(`[${opts.collection}] ${stats.recordsWritten}/${files.length} records, ${stats.fragmentsEmbedded} fragments embedded`);
      }
    }
  }
  // Residue first: the nsfw rows are already committed, so the sfw copies can go
  // even if the final embed flush below fails.
  await flushResidue();
  await flush(true);
  stats.classifiedNsfwSessions = nsfwSessionsWritten.size;
  stats.ambiguousSessions = ambiguousSessionsWritten.size;
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
 * RETENTION — ADR-0162 §5. DETACH then DROP. NEVER a DELETE sweep.
 *
 * Why this is a CLI verb and not a plpgsql function: ALTER TABLE ... DETACH
 * PARTITION ... CONCURRENTLY cannot run inside a transaction block, and every
 * plpgsql function body is one. So the verb runs here, outside a transaction,
 * where the concurrent form is at least reachable.
 *
 * ===========================================================================
 * ADR-0162 §5 ASKS FOR TWO THINGS POSTGRES WILL NOT GIVE AT THE SAME TIME.
 *
 * §5 mandates, in the same list:
 *   - "Always a `DEFAULT` partition", and
 *   - "Retention is `DETACH … CONCURRENTLY` then `DROP`, never a `DELETE` sweep."
 *
 * PostgreSQL 16 refuses the combination outright:
 *
 *   ERROR: cannot detach partitions concurrently when a default partition exists
 *
 * (The concurrent detach runs in two transactions; in between, a row that
 * belongs to the detaching partition's range could legally be routed to the
 * default, so the server will not attempt it.) Observed on the vn4rz.8
 * retention drill against the live table.
 *
 * WHAT THIS FUNCTION DOES ABOUT IT, and it is a choice worth stating:
 *
 *   - It keeps the DEFAULT partition. §5's own reason for it — a record whose
 *     partition key cannot be resolved must have somewhere to land rather than
 *     aborting the load — is a CORRECTNESS requirement, and the concurrent
 *     detach is a LOCK-DURATION optimisation. Correctness wins.
 *   - It therefore falls back to a plain `DETACH PARTITION` + `DROP TABLE`,
 *     which takes ACCESS EXCLUSIVE on the parent for the duration of a catalog
 *     update — brief, but a full lock on a live table.
 *   - It PRINTS which mode it used and why, every time. A retention run that
 *     silently took a heavier lock than the ADR advertises is exactly the kind
 *     of quiet divergence this whole arc exists to stop.
 *   - It NEVER falls back to a DELETE sweep. That is the invariant §5 actually
 *     protects (dead tuples + deferred index reclamation), and it is untouched.
 *
 * NAMED FOR AN ADR AMENDMENT rather than decided silently: ADR-0162 §5's
 * "always a DEFAULT partition" and "retention is DETACH … CONCURRENTLY" are
 * mutually exclusive on PostgreSQL 16. One of them has to be relaxed, and this
 * code relaxes the second while saying so out loud.
 * ===========================================================================
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
export interface RetentionResult {
  dropped: string[];
  kept: string[];
  refusedDefault: string[];
  /** Which detach form is available, and the reason. Printed, never inferred. */
  detachMode: "concurrent" | "plain-access-exclusive";
  detachModeReason: string;
}

export async function dropPartitionsBefore(
  cutoffIsoDate: string,
  vault: Vault = "sfw",
  opts: { apply?: boolean } = {},
): Promise<RetentionResult> {
  const cutoff = new Date(`${cutoffIsoDate}T00:00:00Z`);
  if (isNaN(cutoff.getTime())) throw new Error(`unparseable cutoff: ${cutoffIsoDate}`);
  const parts = await listPartitions(vault);
  const dropped: string[] = [], kept: string[] = [], refusedDefault: string[] = [];

  // Decided UP FRONT from the catalog, not discovered from a thrown error. An
  // exception-driven fallback would only be exercised on the day it matters.
  const hasDefault = parts.some(p => p.is_default);
  const detachMode = hasDefault ? "plain-access-exclusive" : "concurrent";
  const detachModeReason = hasDefault
    ? "a DEFAULT partition exists, and PostgreSQL refuses DETACH ... CONCURRENTLY " +
      "in that case (\"cannot detach partitions concurrently when a default partition " +
      "exists\"). Using plain DETACH + DROP, which takes ACCESS EXCLUSIVE on the parent " +
      "for a brief catalog update. Still NEVER a DELETE sweep -- that is the invariant " +
      "ADR-0162 §5 actually protects."
    : "no DEFAULT partition present, so the concurrent detach ADR-0162 §5 prefers is " +
      "available and is used.";

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
          `ALTER TABLE origin_documents DETACH PARTITION ${quoteIdent(p.partition_name)}` +
          (detachMode === "concurrent" ? " CONCURRENTLY" : ""));
        await c.query(`DROP TABLE ${quoteIdent(p.partition_name)}`);
      });
    }
  }
  return { dropped, kept, refusedDefault, detachMode, detachModeReason };
}

/** Identifiers here come from pg_class, but quoting is not optional on the way back in. */
function quoteIdent(name: string): string {
  return `"${name.replaceAll('"', '""')}"`;
}
