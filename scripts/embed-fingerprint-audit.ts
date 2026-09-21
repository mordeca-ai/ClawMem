/**
 * master-harness-5n0ew — READ-ONLY stored-vector audit. Opens the index read-only, never
 * writes it, and embeds serially against the configured remote endpoint.
 *
 * For each selected content_vectors row: rebuild the fragment through the production
 * pipeline (canonical doc → splitDocument → formatDocForEmbedding), embed it fresh
 * through LlamaCpp (same truncation the embed path applies, reported back as
 * EmbeddingResult.input), and classify it with canary.ts's classifyStoredVector — the
 * SAME verdict function `clawmem doctor` uses. Output:
 *   --out <file.jsonl>      one JSON line per audited row
 *   --requeue-out <file>    distinct content hashes needing a re-embed, in the format
 *                           `clawmem embed --requeue-hashes <file>` reads
 *   stdout                  cos distribution segmented oversized-vs-normal and by
 *                           >512-vs-<=512 tokens (the in-process ubatch boundary)
 *
 * Row selection (combine freely; with none, every eligible row is a candidate):
 *   --sample N          seeded random sample of N rows (deterministic for --seed)
 *   --windows <json>    only rows whose embedded_at falls inside one of the
 *                       [startUtc, endUtc] pairs (e.g. the cron runs that fell back to the
 *                       in-process arm)
 *   --truncated-only    only rows whose input the embedder truncates today (decided by the
 *                       embed itself, so pair with --min-chars to avoid embedding everything)
 *   --min-chars N       only rows whose formatted input is longer than N chars
 *   --hashes <file>     only rows of these content hashes (a --requeue-hashes file) — the
 *                       post-re-embed verification of exactly what was requeued
 *   --no-embed          do not embed at all: every selected row goes straight to the
 *                       requeue list (use when auditing would cost as much as re-embedding,
 *                       e.g. a whole in-process-fallback window)
 *
 * Usage:
 *   CLAWMEM_NO_LOCAL_MODELS=true bun scripts/embed-fingerprint-audit.ts \
 *     [--db ~/.cache/clawmem/index.sqlite] [--sample 240 --seed 5] \
 *     [--windows windows.json] [--truncated-only] [--min-chars 700] \
 *     --out audit.jsonl --requeue-out requeue.txt
 *
 * Exit codes: 0 audit ran (findings are data, not failure) · 2 infrastructure abort.
 */
import { parseArgs } from "util";
import { readFileSync, writeFileSync, appendFileSync } from "fs";
import { homedir } from "os";
import { join } from "path";
import { Database } from "bun:sqlite";
import * as sqliteVec from "sqlite-vec";
import { formatDocForEmbedding, LlamaCpp } from "../src/llm.ts";
import { buildEmbedFrontmatter } from "../src/embed-input.ts";
import { splitDocument } from "../src/splitter.ts";
import { canonicalDocId } from "../src/store.ts";
import { classifyStoredVector, type StoredVectorVerdict } from "../src/canary.ts";

const REQUEUE_KINDS = new Set<StoredVectorVerdict["kind"]>(["stale-input", "stale-policy", "arm-mismatch", "corruption/drift", "legacy-inconclusive"]);

const { values } = parseArgs({
  options: {
    db: { type: "string", default: join(homedir(), ".cache", "clawmem", "index.sqlite") },
    sample: { type: "string" },
    seed: { type: "string", default: "5" },
    windows: { type: "string" },
    hashes: { type: "string" },
    "truncated-only": { type: "boolean", default: false },
    "min-chars": { type: "string" },
    out: { type: "string" },
    "requeue-out": { type: "string" },
    "no-embed": { type: "boolean", default: false },
  },
});

const embedUrl = process.env.CLAWMEM_EMBED_URL ?? (values["no-embed"] ? "unused" : undefined);
if (!embedUrl) { console.error("CLAWMEM_EMBED_URL is required (the audit compares against the serving remote arm)"); process.exit(2); }
if (!values["no-embed"] && process.env.CLAWMEM_NO_LOCAL_MODELS !== "true") {
  console.error("set CLAWMEM_NO_LOCAL_MODELS=true — a silent in-process fallback would make every verdict meaningless");
  process.exit(2);
}

type Row = { hash: string; seq: number; fragment_label: string | null; embed_input_fp: string | null; canonical_id: string | null; embedded_at: string };
const db = new Database(values.db!, { readonly: true });
sqliteVec.load(db);
let rows = db.prepare(`
  SELECT cv.hash, cv.seq, cv.fragment_label, cv.embed_input_fp, cv.canonical_id, cv.embedded_at
  FROM content_vectors cv
  WHERE EXISTS (SELECT 1 FROM documents d WHERE d.hash = cv.hash AND d.active = 1 AND d.invalidated_at IS NULL AND d.embed_state = 'synced')
`).all() as Row[];
const eligible = rows.length;

if (values.windows) {
  const w = JSON.parse(readFileSync(values.windows, "utf8")) as [string, string, ...unknown[]][];
  const norm = (t: string) => t.slice(0, 19); // ISO seconds, UTC
  rows = rows.filter(r => { const t = norm(r.embedded_at); return w.some(([s, e]) => t >= norm(s) && t <= norm(e)); });
}
if (values.hashes) {
  const want = new Set(readFileSync(values.hashes, "utf8").split(/\r?\n/).map(l => l.replace(/#.*$/, "").trim()).filter(Boolean));
  rows = rows.filter(r => want.has(r.hash));
}
if (values.sample) {
  let seed = parseInt(values.seed!, 10) || 1;
  const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  for (let i = rows.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [rows[i], rows[j]] = [rows[j]!, rows[i]!]; }
}
const want = values.sample ? parseInt(values.sample, 10) : Infinity;
const minChars = values["min-chars"] ? parseInt(values["min-chars"], 10) : 0;

const llm = new LlamaCpp({ remoteEmbedUrl: embedUrl, remoteEmbedModel: process.env.CLAWMEM_EMBED_MODEL });
const aliasStmt = db.prepare(`SELECT collection, path, title, description, active FROM documents WHERE hash = ? ORDER BY active DESC, collection, path`);
const bodyStmt = db.prepare(`SELECT doc FROM content WHERE hash = ?`);
const vecStmt = db.prepare(`SELECT embedding FROM vectors_vec WHERE hash_seq = ?`);
if (values.out) writeFileSync(values.out, "");

type Audited = { hash: string; seq: number; where: string; chars: number; sentChars: number; truncated: boolean; tokens: number; kind: string; cos: number; embedded_at: string };
const audited: Audited[] = [];
let unreconstructable = 0;
let docCache: { hash: string; texts: string[]; labels: (string | null)[]; where: string } | null = null;

for (const row of rows) {
  if (audited.length >= want) break;
  if (!docCache || docCache.hash !== row.hash) {
    const aliases = aliasStmt.all(row.hash) as { collection: string; path: string; title: string; description: string | null; active: number }[];
    let canon = row.canonical_id ? aliases.find(a => canonicalDocId(a.collection, a.path) === row.canonical_id) : undefined;
    if (!canon && !row.canonical_id) { const act = aliases.filter(a => a.active === 1); if (act.length === 1) canon = act[0]; }
    const body = (bodyStmt.get(row.hash) as { doc: string } | undefined)?.doc;
    if (!canon || !body) docCache = { hash: row.hash, texts: [], labels: [], where: "" };
    else {
      const fm = buildEmbedFrontmatter(canon.title, canon.path, canon.description);
      const frags = splitDocument(body, fm);
      docCache = {
        hash: row.hash, where: `${canon.collection}/${canon.path}`,
        texts: frags.map(f => formatDocForEmbedding(f.content, f.label || fm.title)),
        labels: frags.map(f => f.label ?? null),
      };
    }
  }
  const text = docCache.texts[row.seq];
  // Label contract, as in doctor: a label mismatch is unreconstructable, never judged.
  if (text === undefined || docCache.labels[row.seq] !== (row.fragment_label ?? null)) { unreconstructable++; continue; }
  if (text.length <= minChars) continue;
  if (values["no-embed"]) {
    audited.push({ hash: row.hash, seq: row.seq, where: `${docCache.where}#${row.seq}`, chars: text.length, sentChars: NaN, truncated: false, tokens: NaN, kind: "selected-no-embed", cos: NaN, embedded_at: row.embedded_at });
    continue;
  }
  const sv = vecStmt.get(`${row.hash}_${row.seq}`) as { embedding: Uint8Array } | undefined;
  if (!sv) { unreconstructable++; continue; }
  const b = new Uint8Array(sv.embedding);
  const stored = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  const [fresh] = await llm.embedBatch([text]); // one input per request: usage = this row's tokens
  if (!fresh) { console.error(`fresh embed failed for ${docCache.where}#${row.seq} — aborting (endpoint down?)`); process.exit(2); }
  if (fresh.embedding.length !== stored.length) { unreconstructable++; continue; }
  const truncated = (fresh.input ?? text) !== text;
  if (values["truncated-only"] && !truncated) continue;
  const v = classifyStoredVector(row.embed_input_fp, text, stored, fresh);
  const a: Audited = {
    hash: row.hash, seq: row.seq, where: `${docCache.where}#${row.seq}`, chars: text.length,
    sentChars: (fresh.input ?? text).length, truncated, tokens: llm.lastBatchTokens,
    kind: v.kind, cos: v.sim ?? NaN, embedded_at: row.embedded_at,
  };
  audited.push(a);
  if (values.out) appendFileSync(values.out, JSON.stringify(a) + "\n");
}

const median = (xs: number[]) => { const s = [...xs].sort((p, q) => p - q); const m = s.length >> 1; return s.length ? (s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2) : NaN; };
const seg = (name: string, rs: Audited[]) => {
  const c = rs.map(r => r.cos);
  console.log(`${name.padEnd(30)} n=${String(rs.length).padStart(5)}  min=${rs.length ? Math.min(...c).toFixed(4) : "-"}  median=${rs.length ? median(c).toFixed(4) : "-"}  cos<0.98=${c.filter(x => x < 0.98).length}`);
};
console.log(`eligible=${eligible} selected=${rows.length} audited=${audited.length} unreconstructable=${unreconstructable}`);
seg("ALL", audited);
seg("oversized (truncated today)", audited.filter(r => r.truncated));
seg("normal (not truncated)", audited.filter(r => !r.truncated));
seg("tokens > 512", audited.filter(r => r.tokens > 512));
seg("tokens <= 512", audited.filter(r => r.tokens <= 512));
seg("seq-0", audited.filter(r => r.seq === 0));
seg("seq > 0", audited.filter(r => r.seq > 0));
const kinds: Record<string, number> = {};
for (const r of audited) kinds[r.kind] = (kinds[r.kind] ?? 0) + 1;
console.log(`verdicts: ${JSON.stringify(kinds)}`);
const requeue = [...new Set(audited.filter(r => r.kind === "selected-no-embed" || REQUEUE_KINDS.has(r.kind as StoredVectorVerdict["kind"])).map(r => r.hash))];
console.log(`hashes needing re-embed: ${requeue.length}`);
if (values["requeue-out"]) writeFileSync(values["requeue-out"], `# embed-fingerprint-audit ${new Date().toISOString()} — ${requeue.length} hash(es)\n${requeue.join("\n")}\n`);
