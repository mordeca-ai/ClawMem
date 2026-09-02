/**
 * REINDEX (not copy) into PostgreSQL — master-harness-vn4rz.7, ADR-0162 §6.
 *
 * ===========================================================================
 * REINDEX-NOT-COPY IS A HARD RULE.
 *
 * Every document is re-read and re-embedded FROM ITS FILE-AUTHORED SOURCE. No
 * row is ever copied out of sqlite. Two reasons, both load-bearing:
 *
 *  1. The sqlite rows carry a decade of accumulated ALTER TABLE drift and
 *     filename-inferred content_types (rvzn8.2). Copying them would import the
 *     drift into the schema built to end it.
 *  2. §3's facets can be assigned at near-zero marginal cost only at the moment
 *     the source is re-parsed. A copy has nothing to assign them from.
 *
 * The sqlite database is therefore READ ONLY as a COUNT baseline, by
 * tools/clawmem-pg-parity, and never as a data source.
 * ===========================================================================
 *
 * Parsing reuses src/indexer.ts's helpers verbatim (hashContent, parseDocument,
 * extractTitle, computeQualityScore, authoredAtFromFrontmatter) rather than
 * re-implementing them, so a PG-indexed document and a sqlite-indexed document
 * are parsed identically and any parity gap is a WRITE-path gap, not a parse
 * gap.
 */

import { Glob } from "bun";
import { readFileSync, statSync } from "fs";
import { join } from "path";
import {
  authoredAtFromFrontmatter, computeQualityScore, extractTitle, hashContent, parseDocument,
  type FrontmatterParseFailure,
} from "../indexer.ts";
import { listCollections } from "../collections.ts";
import { getDefaultLlamaCpp, formatDocForEmbedding } from "../llm.ts";
import { splitDocument } from "../splitter.ts";
import { canonicalDocId } from "../store.ts";
import { embedDim } from "./config.ts";
import { insertEmbeddingsBatch, upsertDocument, type EmbeddingWrite } from "./write.ts";

/** Mirrors indexer.ts's brace expansion — Bun.Glob has no brace support. */
function expandBraces(pattern: string): string[] {
  const m = pattern.match(/^\{(.+)\}$/);
  return m ? m[1]!.split(",").map(s => s.trim()) : [pattern];
}

export interface ReindexOptions {
  /** Restrict to these collection names. Empty/undefined = every collection. */
  collections?: string[];
  /** Stop after N documents. For the validate-small-before-batch smoke pass. */
  limit?: number;
  /** FRAGMENTS per embed round trip (sqlite's default is 50). */
  embedBatchSize?: number;
  /** Skip embedding entirely (schema/parity smoke without touching yoshiee). */
  skipEmbed?: boolean;
  onProgress?: (msg: string) => void;
}

export interface ReindexStats {
  collection: string;
  filesSeen: number;
  documentsWritten: number;
  fragmentsEmbedded: number;
  embedFailures: number;
  /** content_type values that fell outside the closed ADR-0058 enum, with counts. */
  contentTypeRetagBacklog: Record<string, number>;
  /**
   * Documents whose frontmatter block was present but UNPARSEABLE, path → the
   * YAML parser's own message (master-harness-vn4rz.34).
   *
   * Same shape and same reporting channel as contentTypeRetagBacklog above:
   * counted here so "N documents in this collection have unparseable
   * frontmatter, and here they are" is answerable from the reindex summary
   * instead of being swallowed. These documents lost title/description/tags/
   * domain/workstream, got a filename-INFERRED content_type (and so the wrong
   * decay curve), and had their raw YAML embedded as body prose.
   */
  frontmatterParseFailures: Record<string, string>;
  wallClockMs: number;
}

/**
 * Facet assignment. Deliberately CONSERVATIVE: a facet is set only when the
 * source declares it, otherwise 'unknown' (ADR-0162 §3 — "every facet carries an
 * explicit unknown value", and the unknown counts are a MONITORED METRIC, not a
 * shrug). Guessing audience/trust_tier from a path would manufacture a
 * classification nobody made and bury it in the data.
 *
 * `provenance` records WHO assigned the values and by WHAT rule, at load time,
 * so the classification decision stays retrievable (§3).
 */
function deriveFacets(collection: string, meta: { domain?: string }, relPath: string) {
  return {
    domain: meta.domain ?? "unknown",
    audience: "unknown" as const,
    trustTier: "unknown" as const,
    sensitivity: "unknown" as const,
    sourceRef: null,
    provenance: {
      assigned_by: "clawmem pg reindex (master-harness-vn4rz.7)",
      assigned_at: new Date().toISOString(),
      rule: "domain from frontmatter when declared, else unknown; audience/trust_tier/" +
            "sensitivity left unknown pending an explicit classification pass",
      collection,
      source_path: relPath,
    },
  };
}

/**
 * Fold one document's frontmatter-parse outcome into the collection summary
 * (master-harness-vn4rz.34).
 *
 * Extracted rather than inlined so the COUNTING is unit-testable without a
 * database — the guard this belongs to is worthless if nobody can prove the
 * number reaches the summary.
 *
 * A no-op when `err` is undefined, which covers BOTH "parsed cleanly" and
 * "declared no frontmatter". Only a document that declared a block the parser
 * refused is counted.
 */
export function noteFrontmatterFailure(
  stats: Pick<ReindexStats, "frontmatterParseFailures">,
  relPath: string,
  err: FrontmatterParseFailure | undefined,
): void {
  if (!err) return;
  stats.frontmatterParseFailures[relPath] = err.message;
}

export async function reindexCollection(
  name: string,
  root: string,
  pattern: string,
  opts: ReindexOptions = {},
): Promise<ReindexStats> {
  const t0 = Date.now();
  const log = opts.onProgress ?? (() => {});
  const batchSize = opts.embedBatchSize ?? Number(process.env.CLAWMEM_EMBED_BATCH_SIZE ?? 50);
  const dim = embedDim();
  const llm = getDefaultLlamaCpp();

  const seen = new Set<string>();
  const files: string[] = [];
  for (const p of expandBraces(pattern)) {
    for (const f of new Glob(p).scanSync({ cwd: root, followSymlinks: false, absolute: false })) {
      if (!seen.has(f)) { seen.add(f); files.push(f); }
    }
  }
  files.sort();

  const stats: ReindexStats = {
    collection: name, filesSeen: files.length, documentsWritten: 0,
    fragmentsEmbedded: 0, embedFailures: 0, contentTypeRetagBacklog: {},
    frontmatterParseFailures: {}, wallClockMs: 0,
  };

  /**
   * FRAGMENT-LEVEL EMBEDDING (delta amendment 1, master-harness-vn4rz.7 pass B).
   *
   * The first cut of this reindexer wrote ONE vector per document (seq=0 over the
   * whole body). sqlite does not: its embed path calls splitDocument() and writes
   * one row per semantic fragment, keyed (hash, seq), which is why the same 370
   * memory-topics documents hold 1,496 vectors there (max seq 17) and held only
   * 370 here. Document-count parity stayed GREEN across that gap — a doc-count
   * check structurally cannot see it — while within-document retrieval
   * granularity was silently destroyed. Exactly the silent-wrong-answer class.
   *
   * The fix reuses the EXISTING splitter through buildDocEmbedTask() rather than
   * re-implementing fragmentation, so the PG and sqlite paths cannot drift: same
   * fragment set, same seq ordering, same `pos` (fragment start line), same
   * fragment_type / fragment_label / canonical_id / embed_input_fp semantics, and
   * the same `label || title` rule feeding formatDocForEmbedding.
   */
  type PendingFragment = {
    hash: string; seq: number; pos: number; text: string;
    fragmentType: string; fragmentLabel: string | null; canonicalId: string;
  };
  const pending: PendingFragment[] = [];
  const hashesDone = new Set<string>();

  /**
   * Embed + write exactly ONE batch of fragments. sqlite chunks its flattened
   * fragment queue at a fixed batchSize; a single document can contribute up to
   * MAX_FRAGMENTS_PER_DOC fragments, so flushing "whatever accumulated" would
   * send ragged, sometimes oversized batches. Chunking here keeps the request
   * shape identical to the sqlite path's.
   */
  const embedAndWrite = async (chunk: PendingFragment[]) => {
    const pending = chunk;
    const texts = pending.map(p => p.text);
    const results = await llm.embedBatch(texts);
    const writes: EmbeddingWrite[] = [];
    for (let i = 0; i < pending.length; i++) {
      const r = results[i];
      const frag = pending[i]!;
      if (!r?.embedding) { stats.embedFailures++; continue; }
      if (r.embedding.length !== dim) {
        // Loud, immediate. Never truncate, never pad.
        throw new Error(
          `Embed endpoint returned ${r.embedding.length} dimensions for ` +
          `${frag.canonicalId} seq=${frag.seq}; the schema column is vector(${dim}). ` +
          `Refusing to write.`,
        );
      }
      writes.push({
        hash: frag.hash, seq: frag.seq, pos: frag.pos,
        embedding: r.embedding, model: r.model,
        fragmentType: frag.fragmentType,
        fragmentLabel: frag.fragmentLabel,
        canonicalId: frag.canonicalId,
        // SHA-256 over the UTF-8 bytes of the exact formatted embed input,
        // matching src/clawmem.ts's contract byte for byte.
        embedInputFp: hashContent(frag.text),
      });
    }
    if (writes.length > 0) {
      await insertEmbeddingsBatch(writes);
      stats.fragmentsEmbedded += writes.length;
    }
  };

  const flush = async (force: boolean) => {
    if (opts.skipEmbed) { pending.length = 0; return; }
    while (pending.length >= batchSize || (force && pending.length > 0)) {
      await embedAndWrite(pending.splice(0, Math.min(batchSize, pending.length)));
    }
  };

  let n = 0;
  for (const rel of files) {
    if (opts.limit !== undefined && n >= opts.limit) break;
    const abs = join(root, rel);
    let raw: string;
    try {
      raw = readFileSync(abs, "utf-8");
    } catch {
      continue;
    }
    const hash = hashContent(raw);
    const { body, meta, frontmatterError } = parseDocument(raw, rel);
    // parseDocument already emitted the per-file warning; this is the COUNT
    // that reaches the summary (vn4rz.34).
    noteFrontmatterFailure(stats, rel, frontmatterError);
    const title = meta.title ?? extractTitle(raw, rel);

    // The retag backlog: counted here so the enum decision has a number attached
    // to it rather than a shrug.
    //
    // What counts as backlog is `isRetagBacklog` in write.ts, beside the conform
    // map that defines it — see its doc comment for why a preserved raw stopped
    // meaning "needs a retag" at the ADR-0058 2026-09-02 amendment.
    const ct = meta.content_type as string | undefined;
    if (ct) {
      const { narrowContentType, isRetagBacklog } = await import("./write.ts");
      if (isRetagBacklog(narrowContentType(ct))) {
        stats.contentTypeRetagBacklog[ct] = (stats.contentTypeRetagBacklog[ct] ?? 0) + 1;
      }
    }

    await upsertDocument({
      collection: name,
      path: rel,
      title,
      hash,
      body: raw,
      contentTypeRaw: ct ?? null,
      description: meta.description ?? null,
      tags: meta.tags ?? null,
      workstream: meta.workstream ?? null,
      authoredAt: authoredAtFromFrontmatter((meta as { authored_at?: unknown }).authored_at) ?? null,
      modifiedAt: statSync(abs).mtime,
      qualityScore: computeQualityScore(body, meta),
      contentHash: hash,
      origin: "fs",
      facets: deriveFacets(name, meta, rel),
    });
    stats.documentsWritten++;
    n++;

    if (!hashesDone.has(hash)) {
      hashesDone.add(hash);
      // Inlines src/clawmem.ts::buildDocEmbedTask rather than importing it:
      // src/clawmem.ts calls main() at MODULE SCOPE, so importing it executes the
      // whole sqlite CLI against whatever argv this process happens to carry. The
      // shared thing that matters — splitDocument — is imported directly from
      // ../splitter.ts, so there is still exactly one splitter.
      //
      // `body` is the frontmatter-STRIPPED text, matching what the sqlite path
      // stores in content.doc — so the frontmatter fragment is SYNTHESIZED from
      // title/description (the master-harness-z7o4y fix) rather than re-parsed out
      // of a body that no longer has any frontmatter to find.
      const frontmatter: Record<string, unknown> = { title };
      if (meta.description) frontmatter.description = meta.description;
      const fragments = splitDocument(body, frontmatter);
      const canonicalId = canonicalDocId(name, rel);
      for (let seq = 0; seq < fragments.length; seq++) {
        const frag = fragments[seq]!;
        pending.push({
          hash, seq, pos: frag.startLine,
          text: formatDocForEmbedding(frag.content, frag.label || title),
          fragmentType: frag.type, fragmentLabel: frag.label, canonicalId,
        });
      }
      if (pending.length >= batchSize) {
        await flush(false);
        log(`[${name}] ${stats.documentsWritten}/${files.length} docs, ${stats.fragmentsEmbedded} fragments embedded`);
      }
    }
  }
  await flush(true);

  stats.wallClockMs = Date.now() - t0;
  return stats;
}

/** Reindex every configured collection (or the named subset). */
export async function reindex(opts: ReindexOptions = {}): Promise<ReindexStats[]> {
  const wanted = new Set(opts.collections ?? []);
  const out: ReindexStats[] = [];
  for (const c of listCollections()) {
    if (wanted.size > 0 && !wanted.has(c.name)) continue;
    out.push(await reindexCollection(c.name, c.path, c.pattern, opts));
  }
  return out;
}
