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
import { closeSync, fstatSync, openSync, readFileSync, readSync } from "fs";
import { createRequire } from "module";
import { basename, join } from "path";
import matter from "gray-matter";
import {
  extractTitle, hashContent, parseDocument, type DocumentMeta,
} from "../indexer.ts";
import type { FrontmatterMap } from "../frontmatter-map.ts";
import { getDefaultLlamaCpp, formatDocForEmbedding } from "../llm.ts";
import { splitDocument } from "../splitter.ts";
import { canonicalDocId } from "../store.ts";
import type pg from "pg";
import {
  describePoolClient, withClient, withTransaction, type PoolClientIdentity,
} from "./client.ts";
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
  /**
   * TEST SEAM (master-harness-vn4rz.114): how the write loop reads a record's
   * text. Production never sets it (default: readFileSync utf-8). The read-back
   * guard's regression test passes a reader that returns ANOTHER file's text,
   * the 2026-10-18T08:53Z live failure shape, to prove the independent source
   * re-read catches it. The classification pre-pass always reads the real file.
   */
  readRaw?: (absPath: string) => string;
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
  /**
   * hashContent of an INDEPENDENT second read of the source file
   * (master-harness-vn4rz.114), or null when that read failed. Absent = not
   * checked. Every other expected value derives from the one `raw` string, so
   * without this a `raw` carrying another file's text passes every RETURNING
   * comparison: the 2026-10-18T08:53Z live run wrote 265 rows that way.
   */
  sourceHash?: string | null;
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
  contentHash?: unknown;
  originHash?: unknown;
  docSha?: unknown;
  /**
   * The full content-free picture (master-harness-vn4rz.114). When present the
   * message is built from it; the four hash fields above are the legacy shape.
   */
  diagnostics?: OriginReadBackDiagnostics;
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

  if (expected.sourceHash !== undefined && expected.sourceHash !== expected.hash) {
    mismatches.push("source");
  }

  return mismatches;
}

// ---------------------------------------------------------------------------
// READ-BACK DIAGNOSTICS (master-harness-vn4rz.114)
//
// The vn4rz.109 message printed one 8-char prefix per hash and fell back to a
// combined "hash expected X got Y" whose `got` was `origin ?? content`. On the
// 2026-10-18 recurrence that printed "expected 0458e203 got 0458e203" for both
// hash and doc_sha. That line cannot tell stale values from missing keys from a
// value of the wrong type. These diagnostics can, and they stay content-free:
//
//   - every guarded column, mismatched or not, with: is the key on the row,
//     typeof, constructor name, length, and the guard's own equality verdict;
//   - hash and doc_sha also get both 8-char prefixes (or "missing"/"null"/
//     "non-hex") and the first index where the text differs;
//   - authored_at gets the signed difference in milliseconds;
//   - each RETURNING result's row count, command, key names, and the
//     RowDescription names and type OIDs (names that are not plain SQL
//     identifiers print as their length only);
//   - the client: backend pid, database, which vault pool lists it, whether a
//     query is still in flight on it;
//   - an in-transaction re-read whose answers come back as COUNTS computed in
//     SQL, so "the DB holds the right row but RETURNING came back wrong" and
//     "the DB holds the wrong row" read differently;
//   - process.versions.bun and the pg package version.
//
// Title, session id and document text never appear: those fields report only
// type, length and equality.
// ---------------------------------------------------------------------------

/** One guarded column, described without its value. */
export interface ReadBackFieldDiagnostic {
  /** `content.hash`, `origin.title`, ... */
  field: string;
  /** Object.hasOwn(row, key). False when the row itself is missing. */
  present: boolean;
  /** typeof the value; "null" for null. */
  type: string;
  /** Constructor name of an object value (Date, Buffer, String, ...), else null. */
  ctor: string | null;
  /** Length of a string-like value (string, String, Uint8Array byteLength), else null. */
  len: number | null;
  /** Length of the expected value when it is a string, else null. */
  expLen: number | null;
  /** The guard's own equality verdict for this column. */
  equal: boolean;
  /** hash/doc_sha only: 8-char prefixes, or "missing" | "null" | "undefined" | "non-string" | "non-hex". */
  exp8?: string;
  got8?: string;
  /**
   * hash/doc_sha/path only: first index where the text differs; -1 when the text
   * is identical (so `equal: false` then means a type difference); null when the
   * value is not string-like.
   */
  firstDiff?: number | null;
  /** authored_at only: got minus expected in ms; null when either side has no instant. */
  deltaMs?: number | null;
}

/** The shape of one RETURNING result. Key and field names only, never values. */
export interface ReadBackResultShape {
  rows: number;
  rowCount: number | null;
  command: string | null;
  /** Object.keys(rows[0]), sanitised. */
  keys: string[];
  /** RowDescription as `name:typeOid`, sanitised. */
  fields: string[];
}

/**
 * An in-transaction re-read of what the database now holds for this write.
 * Every value is a count or an id computed server-side; no column text comes back.
 */
export interface StoredRowCheck {
  /** origin_documents rows for (collection, path), any authored_at. */
  pathRows: number;
  /** ... of which every guarded column equals the expected value. */
  exactRows: number;
  hashRows: number;
  titleRows: number;
  authoredRows: number;
  sessionRows: number;
  /** content rows whose hash and sha256(doc) both equal the expected hash. */
  contentOkRows: number;
  /** pg_backend_pid() as seen by the server on this client's connection. */
  backendPid: number | null;
  database: string | null;
}

export interface SourceReadFacts {
  /** 0-based position of this file among the records this run has written. */
  index: number;
  /** Bytes the independent re-read returned; null when it failed. */
  fileBytes: number | null;
  /** UTF-8 byte length of the `raw` string the loader derived everything from. */
  rawBytes: number;
  readError: string | null;
  /**
   * `raw` hashes the same as the FIRST record this run wrote, for a different
   * path: the exact vn4rz.109 / 2026-10-18T08:53Z signature. Null for record 0.
   */
  matchesRunFirst: boolean | null;
}

export interface OriginReadBackDiagnostics {
  fields: ReadBackFieldDiagnostic[];
  content: ReadBackResultShape | null;
  origin: ReadBackResultShape | null;
  client: PoolClientIdentity | null;
  /** The re-read, or the error class name when it could not run. */
  stored: StoredRowCheck | { error: string } | null;
  /** The source file: loop index, byte counts of both reads, re-read error class. */
  source: SourceReadFacts | null;
  runtime: { bun: string | null; node: string | null; pg: string | null };
}

const IDENT = /^[A-Za-z_][A-Za-z0-9_]{0,62}$/;
const HEX8 = /^[0-9a-f]{8}$/;

/** A key or field name that is a plain identifier prints as itself, anything else as its length. */
function safeName(n: string): string {
  return IDENT.test(n) ? n : `<non-ident len ${n.length}>`;
}

function stringLike(v: unknown): string | null {
  if (typeof v === "string") return v;
  if (v instanceof String) return v.valueOf();
  if (v instanceof Uint8Array) return Buffer.from(v.buffer, v.byteOffset, v.byteLength).toString("utf8");
  return null;
}

function lengthOf(v: unknown): number | null {
  if (typeof v === "string") return v.length;
  if (v instanceof String) return v.length;
  if (v instanceof Uint8Array) return v.byteLength;
  return null;
}

function typeName(v: unknown): string {
  return v === null ? "null" : typeof v;
}

function ctorName(v: unknown): string | null {
  if (v === null || typeof v !== "object") return null;
  const c = (v as { constructor?: { name?: unknown } }).constructor;
  return typeof c?.name === "string" ? c.name : "<none>";
}

/** 8-char hex prefix, or a word saying why there is none. Never prints non-hex text. */
function hex8(v: unknown, present: boolean): string {
  if (!present) return "missing";
  if (v === undefined) return "undefined";
  if (v === null) return "null";
  const s = stringLike(v);
  if (s === null) return "non-string";
  const p = s.slice(0, 8);
  return HEX8.test(p) ? p : "non-hex";
}

function firstDiffIndex(expected: string, got: unknown): number | null {
  const s = stringLike(got);
  if (s === null) return null;
  const n = Math.min(expected.length, s.length);
  for (let i = 0; i < n; i++) if (expected.charCodeAt(i) !== s.charCodeAt(i)) return i;
  return expected.length === s.length ? -1 : n;
}

function instantMs(v: unknown): number | null {
  if (v instanceof Date) return Number.isNaN(v.getTime()) ? null : v.getTime();
  if (typeof v === "string" || typeof v === "number") {
    const ms = new Date(v).getTime();
    return Number.isNaN(ms) ? null : ms;
  }
  return null;
}

function describeField(
  field: string,
  row: object | null | undefined,
  key: string,
  expected: unknown,
  equal: boolean,
  kind: "hex" | "path" | "text" | "instant",
): ReadBackFieldDiagnostic {
  const present = row != null && Object.hasOwn(row, key);
  const got = present ? (row as Record<string, unknown>)[key] : undefined;
  const d: ReadBackFieldDiagnostic = {
    field,
    present,
    type: typeName(got),
    ctor: ctorName(got),
    len: lengthOf(got),
    expLen: typeof expected === "string" ? expected.length : null,
    equal,
  };
  if (kind === "hex") {
    d.exp8 = hex8(expected, true);
    d.got8 = hex8(got, present);
  }
  if ((kind === "hex" || kind === "path") && typeof expected === "string") {
    d.firstDiff = present ? firstDiffIndex(expected, got) : null;
  }
  if (kind === "instant") {
    const e = instantMs(expected);
    const g = present ? instantMs(got) : null;
    d.deltaMs = e === null || g === null ? null : g - e;
  }
  return d;
}

/** Row count, command, key names and RowDescription of one pg result. Values never read. */
export function summarizeReadBackResult(res: unknown): ReadBackResultShape | null {
  if (res == null || typeof res !== "object") return null;
  const r = res as {
    rows?: unknown[];
    rowCount?: unknown;
    command?: unknown;
    fields?: { name?: unknown; dataTypeID?: unknown }[];
  };
  const rows = Array.isArray(r.rows) ? r.rows : [];
  const first = rows[0];
  return {
    rows: rows.length,
    rowCount: typeof r.rowCount === "number" ? r.rowCount : null,
    command: typeof r.command === "string" ? safeName(r.command) : null,
    keys: first != null && typeof first === "object" ? Object.keys(first).map(safeName) : [],
    fields: Array.isArray(r.fields)
      ? r.fields.map(f => `${safeName(String(f?.name ?? ""))}:${String(f?.dataTypeID ?? "?")}`)
      : [],
  };
}

let pgVersionMemo: string | null | undefined;
function pgPackageVersion(): string | null {
  if (pgVersionMemo !== undefined) return pgVersionMemo;
  try {
    const v = createRequire(import.meta.url)("pg/package.json") as { version?: unknown };
    pgVersionMemo = typeof v.version === "string" ? v.version : null;
  } catch {
    pgVersionMemo = null;
  }
  return pgVersionMemo;
}

/** The runtime facts a driver/runtime defect would depend on. */
export function readBackRuntime(): OriginReadBackDiagnostics["runtime"] {
  const versions = process.versions as Record<string, string | undefined>;
  return { bun: versions.bun ?? null, node: versions.node ?? null, pg: pgPackageVersion() };
}

/**
 * Describe every guarded column of one origin write against what was expected.
 * Pure; the caller supplies the result shapes, client identity and re-read.
 */
export function describeOriginWriteReadBack(
  expected: ExpectedOriginWrite,
  contentRow: ContentWriteReadBack | null | undefined,
  originRow: OriginWriteReadBack | null | undefined,
  extra: Partial<Omit<OriginReadBackDiagnostics, "fields" | "runtime">> = {},
): OriginReadBackDiagnostics {
  const expSession = expected.sessionId ?? null;
  const gotSession = originRow ? (originRow.session_id ?? null) : null;
  const fields: ReadBackFieldDiagnostic[] = [
    describeField("origin.path", originRow, "path", expected.path,
      originRow?.path === expected.path, "path"),
    describeField("origin.title", originRow, "title", expected.title,
      originRow?.title === expected.title, "text"),
    describeField("content.hash", contentRow, "hash", expected.hash,
      contentRow?.hash === expected.hash, "hex"),
    describeField("origin.hash", originRow, "hash", expected.hash,
      originRow?.hash === expected.hash, "hex"),
    describeField("origin.authored_at", originRow, "authored_at", expected.authoredAt,
      authoredAtMatches(expected.authoredAt, originRow?.authored_at), "instant"),
    describeField("origin.session_id", originRow, "session_id", expSession,
      !!originRow && expSession === gotSession, "text"),
    describeField("content.doc_sha", contentRow, "doc_sha", expected.hash,
      contentRow?.doc_sha === expected.hash, "hex"),
  ];
  if (expected.sourceHash !== undefined) {
    fields.push(describeField("source.sha", { sha: expected.sourceHash }, "sha", expected.hash,
      expected.sourceHash === expected.hash, "hex"));
  }
  return {
    fields,
    content: extra.content ?? null,
    origin: extra.origin ?? null,
    client: extra.client ?? null,
    stored: extra.stored ?? null,
    source: extra.source ?? null,
    runtime: readBackRuntime(),
  };
}

function fmtField(f: ReadBackFieldDiagnostic): string {
  const parts = [
    `has=${f.present ? 1 : 0}`,
    `type=${f.type}`,
    ...(f.ctor ? [`ctor=${f.ctor}`] : []),
    `len=${f.len ?? "-"}/${f.expLen ?? "-"}`,
    `eq=${f.equal ? 1 : 0}`,
  ];
  if (f.exp8 !== undefined) parts.push(`exp8=${f.exp8}`, `got8=${f.got8}`);
  if (f.firstDiff !== undefined) {
    parts.push(`diff=${f.firstDiff === null ? "n/a" : f.firstDiff === -1 ? "same-text" : f.firstDiff}`);
  }
  if (f.deltaMs !== undefined) parts.push(`dms=${f.deltaMs ?? "n/a"}`);
  return `${f.field}[${parts.join(" ")}]`;
}

function fmtShape(name: string, s: ReadBackResultShape | null): string {
  if (!s) return `${name}{none}`;
  return `${name}{rows=${s.rows} rowCount=${s.rowCount ?? "-"} cmd=${s.command ?? "-"} ` +
    `keys=[${s.keys.join(",")}] desc=[${s.fields.join(",")}]}`;
}

/** One compact, content-free line. Every field is printed, mismatched or not. */
export function formatOriginReadBackDiagnostics(d: OriginReadBackDiagnostics): string {
  const out = [
    d.fields.map(fmtField).join(" "),
    fmtShape("content", d.content),
    fmtShape("origin", d.origin),
  ];
  const c = d.client;
  if (c) {
    out.push(
      `client{vault=${c.vault} pid=${c.processID ?? "-"} db=${c.database ?? "-"} ` +
        `app=${c.applicationName ?? "-"} ownPool=${c.inOwnPool ?? "-"} ` +
        `otherPools=[${c.inOtherPools.join(",")}] total=${c.poolTotal ?? "-"} ` +
        `idle=${c.poolIdle ?? "-"} waiting=${c.poolWaiting ?? "-"} ` +
        `active=${c.activeQuery ?? "-"} queue=${c.queueDepth ?? "-"} ` +
        `ready=${c.readyForQuery ?? "-"}}`,
    );
  }
  const s = d.stored;
  if (s && "error" in s) out.push(`stored{error=${s.error}}`);
  else if (s) {
    out.push(
      `stored{pathRows=${s.pathRows} exact=${s.exactRows} hash=${s.hashRows} ` +
        `title=${s.titleRows} authored=${s.authoredRows} session=${s.sessionRows} ` +
        `contentOk=${s.contentOkRows} serverPid=${s.backendPid ?? "-"} db=${s.database ?? "-"}}`,
    );
  }
  const src = d.source;
  if (src) {
    out.push(
      `source{index=${src.index} fileBytes=${src.fileBytes ?? "-"} rawBytes=${src.rawBytes}` +
        `${src.readError ? ` readError=${src.readError}` : ""} ` +
        `matchesRunFirst=${src.matchesRunFirst ?? "-"}}`,
    );
  }
  out.push(`bun=${d.runtime.bun ?? "-"} node=${d.runtime.node ?? "-"} pg=${d.runtime.pg ?? "-"}`);
  return out.join(" | ");
}

/** Legacy details (four hashes) as diagnostics, so the message never prints a lossy fallback. */
function diagnosticsFromLegacy(d: OriginWriteReadBackDetails): OriginReadBackDiagnostics {
  const exp = d.expectedHash ?? "";
  const hexField = (field: string, v: unknown): ReadBackFieldDiagnostic => {
    const present = v !== undefined;
    return {
      field,
      present,
      type: typeName(v),
      ctor: ctorName(v),
      len: lengthOf(v),
      expLen: exp ? exp.length : null,
      equal: v === exp,
      exp8: exp ? hex8(exp, true) : "missing",
      got8: hex8(v, present),
      firstDiff: present && exp ? firstDiffIndex(exp, v) : null,
    };
  };
  return {
    fields: [
      hexField("content.hash", d.contentHash),
      hexField("origin.hash", d.originHash),
      hexField("content.doc_sha", d.docSha),
    ],
    content: null,
    origin: null,
    client: null,
    stored: null,
    source: null,
    runtime: readBackRuntime(),
  };
}

/**
 * An INDEPENDENT second read of a source file (master-harness-vn4rz.114): fd +
 * fstat + readSync into a fresh zero-filled Buffer, then the same UTF-8 decode
 * and hashContent the loader applies to `raw`, so a correct read always agrees
 * with it. It deliberately avoids readFileSync, the call that produced the
 * `raw` being checked. Never throws: a failed read is `hash: null` plus the
 * error class name, which the guard treats as a mismatch.
 */
export function rereadSource(path: string): { hash: string | null; bytes: number | null; error: string | null } {
  let fd: number | null = null;
  try {
    fd = openSync(path, "r");
    const size = fstatSync(fd).size;
    const buf = Buffer.alloc(size);
    let off = 0;
    while (off < size) {
      const got = readSync(fd, buf, off, size - off, off);
      if (got <= 0) break;
      off += got;
    }
    return { hash: hashContent(buf.subarray(0, off).toString("utf-8")), bytes: off, error: null };
  } catch (e) {
    return { hash: null, bytes: null, error: e instanceof Error ? safeName(e.name) : "unknown" };
  } finally {
    if (fd !== null) closeSync(fd);
  }
}

/**
 * Re-read, inside the still-open transaction, what the database now holds for
 * this write. Every answer is a count computed server-side, so no column text
 * comes back over the wire. Never throws: a failure is reported by class name.
 */
export async function probeStoredOriginWrite(
  c: pg.PoolClient,
  collection: string,
  expected: ExpectedOriginWrite,
  timeoutMs = 5_000,
): Promise<StoredRowCheck | { error: string }> {
  try {
    const { rows } = await c.query<{
      path_rows: number; exact_rows: number; hash_rows: number; title_rows: number;
      authored_rows: number; session_rows: number; content_ok_rows: number;
      backend_pid: number; db: string;
    }>({
      text: `
        WITH o AS (
          SELECT hash, title, authored_at, session_id FROM origin_documents
           WHERE collection = $1 AND path = $2
        )
        SELECT
          (SELECT count(*)::int FROM o) AS path_rows,
          (SELECT count(*)::int FROM o WHERE hash = $3 AND title = $4
              AND authored_at IS NOT DISTINCT FROM $5::timestamptz
              AND session_id IS NOT DISTINCT FROM $6::text) AS exact_rows,
          (SELECT count(*)::int FROM o WHERE hash = $3) AS hash_rows,
          (SELECT count(*)::int FROM o WHERE title = $4) AS title_rows,
          (SELECT count(*)::int FROM o
              WHERE authored_at IS NOT DISTINCT FROM $5::timestamptz) AS authored_rows,
          (SELECT count(*)::int FROM o
              WHERE session_id IS NOT DISTINCT FROM $6::text) AS session_rows,
          (SELECT count(*)::int FROM content WHERE hash = $3
              AND encode(sha256(convert_to(doc, 'UTF8')), 'hex') = $3) AS content_ok_rows,
          pg_backend_pid() AS backend_pid,
          current_database() AS db`,
      values: [
        collection, expected.path, expected.hash, expected.title,
        expected.authoredAt ?? null, expected.sessionId ?? null,
      ],
      query_timeout: timeoutMs,
    } as pg.QueryConfig);
    const r = rows[0];
    if (!r) return { error: "no-row" };
    const num = (v: unknown) => (typeof v === "number" ? v : Number(v));
    return {
      pathRows: num(r.path_rows),
      exactRows: num(r.exact_rows),
      hashRows: num(r.hash_rows),
      titleRows: num(r.title_rows),
      authoredRows: num(r.authored_rows),
      sessionRows: num(r.session_rows),
      contentOkRows: num(r.content_ok_rows),
      backendPid: typeof r.backend_pid === "number" ? r.backend_pid : null,
      database: typeof r.db === "string" ? safeName(r.db) : null,
    };
  } catch (e) {
    return { error: e instanceof Error ? safeName(e.name) : "unknown" };
  }
}

/**
 * Thrown when PostgreSQL RETURNING verification detects stale or corrupted parameters
 * (master-harness-vn4rz.109). Rolling back prevents unverified writes from committing.
 * To preserve privacy, title text, session IDs, and doc bodies are NEVER included.
 *
 * vn4rz.114: the message carries the full content-free diagnostics line (see
 * formatOriginReadBackDiagnostics), and `diagnostics` holds the same data
 * structured.
 */
export class OriginWriteReadBackError extends Error {
  readonly collection: string;
  readonly path: string;
  readonly mismatches: readonly string[];
  readonly diagnostics: OriginReadBackDiagnostics | null;

  constructor(
    collection: string,
    path: string,
    mismatches: string[],
    details?: OriginWriteReadBackDetails,
  ) {
    const diagnostics = details?.diagnostics
      ?? (details ? diagnosticsFromLegacy(details) : null);
    const suffix = diagnostics ? ` :: ${formatOriginReadBackDiagnostics(diagnostics)}` : "";
    super(
      `Origin write read-back mismatch for collection "${collection}", path "${path}": ` +
        `columns [${mismatches.join(", ")}]${suffix}`,
    );
    this.name = "OriginWriteReadBackError";
    this.collection = collection;
    this.path = path;
    this.mismatches = mismatches;
    this.diagnostics = diagnostics;
  }
}

/** The structured stderr line for a read-back mismatch. Same data as the message. */
export function originReadBackLogLine(
  collection: string,
  path: string,
  mismatches: readonly string[],
  diagnostics: OriginReadBackDiagnostics,
): string {
  return JSON.stringify({
    event: "origin_write_readback_mismatch",
    bead: "master-harness-vn4rz.114",
    ts: new Date().toISOString(),
    collection,
    path,
    mismatches,
    ...diagnostics,
  });
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
  // The first record written, for the read-back diagnostics' matchesRunFirst.
  let runFirst: { rel: string; hash: string } | null = null;
  for (const rel of files) {
    if (opts.limit !== undefined && n >= opts.limit) break;
    let raw: string;
    try {
      raw = opts.readRaw ? opts.readRaw(join(opts.root, rel)) : readFileSync(join(opts.root, rel), "utf-8");
    } catch { continue; }

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
    // vn4rz.114: an independent second read, through a different fs path
    // (fd + readSync into a fresh zero-filled Buffer), checked against `hash`
    // inside the transaction below. Taken now, next to the first read, so a
    // file the capture rewrites mid-run is the only expected disagreement (and
    // that fails closed too: the next run retries it).
    const reread = rereadSource(join(opts.root, rel));
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
      const contentRes = await c.query<{ hash: string; doc_sha: string }>(
        `INSERT INTO content (hash, doc) VALUES ($1, $2)
         ON CONFLICT ON CONSTRAINT content_pkey DO UPDATE SET doc = EXCLUDED.doc
         RETURNING hash, encode(sha256(convert_to(doc, 'UTF8')), 'hex') AS doc_sha`,
        [hash, raw],
      );
      const originRes = await c.query<{
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
      const contentRow = contentRes.rows[0];
      const originRow = originRes.rows[0];
      // session_id goes over the wire as text whatever YAML typed it (a bare
      // number parses as one), so compare its text form, not the raw value.
      const expected: ExpectedOriginWrite = {
        path: rel, title, hash, authoredAt: authoredAt ?? null,
        sessionId: m.session_id == null ? null : String(m.session_id),
        sourceHash: reread.hash,
      };
      const mismatches = checkOriginWriteReadBack(expected, contentRow, originRow);
      if (mismatches.length > 0) {
        // vn4rz.114: capture the client state BEFORE the re-read query runs on it.
        const client = describePoolClient(docVault, c);
        const diagnostics = describeOriginWriteReadBack(expected, contentRow, originRow, {
          content: summarizeReadBackResult(contentRes),
          origin: summarizeReadBackResult(originRes),
          client,
          stored: await probeStoredOriginWrite(c, opts.collection, expected),
          source: {
            index: n, fileBytes: reread.bytes, rawBytes: Buffer.byteLength(raw, "utf-8"),
            readError: reread.error,
            matchesRunFirst: runFirst && runFirst.rel !== rel ? runFirst.hash === hash : null,
          },
        });
        console.error(originReadBackLogLine(opts.collection, rel, mismatches, diagnostics));
        throw new OriginWriteReadBackError(opts.collection, rel, mismatches, { diagnostics });
      }
    });
    stats.recordsWritten++;
    n++;
    runFirst ??= { rel, hash };

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
