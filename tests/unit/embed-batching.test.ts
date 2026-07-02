import { afterEach, beforeEach, describe, expect, it, mock } from "bun:test";

/**
 * Tests for uo0c: batched embedding for the non-cloud remote GPU path.
 *
 * Before this fix, `clawmem embed` only used llm.embedBatch() (one HTTP
 * request per CLAWMEM_EMBED_BATCH_SIZE fragments) when CLAWMEM_EMBED_API_KEY
 * was set (cloud providers). Any other CLAWMEM_EMBED_URL — e.g. a self-hosted
 * GPU server like yoshiee's Ollama — fell into a per-fragment serial loop
 * (one HTTP round trip per fragment), which is round-trip-latency bound
 * rather than GPU-compute bound (measured 1.06 min/doc at 7% GPU util).
 *
 * runBatchedEmbed() is the extracted, store/llm-injectable core of the fix:
 * it flattens fragments across ALL documents into one queue and issues one
 * llm.embedBatch() call per CLAWMEM_EMBED_BATCH_SIZE-sized chunk, regardless
 * of whether the caller is a cloud API or a self-hosted remote endpoint —
 * cmdEmbed now dispatches into it whenever ANY CLAWMEM_EMBED_URL is set.
 */

import { runBatchedEmbed, type DocEmbedTask, type BatchedEmbedLLM } from "../../src/clawmem.ts";
import { createStore, type Store } from "../../src/store.ts";
import type { Fragment } from "../../src/splitter.ts";
import type { EmbeddingResult } from "../../src/llm.ts";

function frag(content: string, label: string | null = null): Fragment {
  return { type: "section", label, content, startLine: 0 };
}

function docTask(hash: string, path: string, fragments: Fragment[]): DocEmbedTask {
  return { hash, path, title: path, collection: "test", fragments };
}

function fakeEmbedding(dims = 4): number[] {
  return Array.from({ length: dims }, () => Math.random());
}

let store: Store;

beforeEach(() => {
  store = createStore(":memory:");
});

afterEach(() => {
  store.close();
});

describe("runBatchedEmbed — cross-document batching", () => {
  it("issues one embedBatch() call per batchSize-sized chunk, not per fragment or per document", async () => {
    // 3 docs × 2 fragments = 6 fragments total. batchSize=4 → 2 calls (4 + 2),
    // and the first call must span doc0 (both frags) + doc1 (first frag) —
    // proof that batching crosses document boundaries, not just within a doc.
    const docs: DocEmbedTask[] = [
      docTask("h0", "doc0.md", [frag("a"), frag("b")]),
      docTask("h1", "doc1.md", [frag("c"), frag("d")]),
      docTask("h2", "doc2.md", [frag("e"), frag("f")]),
    ];

    const seenChunkSizes: number[] = [];
    const embedBatch = mock(async (texts: string[]) => {
      seenChunkSizes.push(texts.length);
      return texts.map((): EmbeddingResult | null => ({ embedding: fakeEmbedding(), model: "test-model" }));
    });
    const llm: BatchedEmbedLLM = { embedBatch, lastBatchTokens: 0 };

    const result = await runBatchedEmbed(store, llm, docs, { batchSize: 4 });

    expect(embedBatch).toHaveBeenCalledTimes(2);
    expect(seenChunkSizes).toEqual([4, 2]);
    expect(result.requestCount).toBe(2);
    expect(result.totalFragments).toBe(6);
    expect(result.failedFragments).toBe(0);
    expect(result.embedded).toBe(3);
  });

  it("respects CLAWMEM_EMBED_BATCH_SIZE-equivalent batchSize of 50 by default", async () => {
    const docs: DocEmbedTask[] = Array.from({ length: 10 }, (_, i) =>
      docTask(`h${i}`, `doc${i}.md`, [frag("only fragment")])
    );
    const embedBatch = mock(async (texts: string[]) =>
      texts.map((): EmbeddingResult => ({ embedding: fakeEmbedding(), model: "test-model" }))
    );
    const llm: BatchedEmbedLLM = { embedBatch, lastBatchTokens: 0 };

    // 10 fragments, default batchSize 50 → everything fits in a single request.
    const result = await runBatchedEmbed(store, llm, docs, {});

    expect(embedBatch).toHaveBeenCalledTimes(1);
    expect(result.requestCount).toBe(1);
    expect(result.totalFragments).toBe(10);
  });

  it("writes embeddings to the store via a single batched transaction and marks docs synced", async () => {
    const docs: DocEmbedTask[] = [docTask("hA", "docA.md", [frag("only")])];
    const embedBatch = mock(async (texts: string[]) =>
      texts.map((): EmbeddingResult => ({ embedding: fakeEmbedding(4), model: "test-model" }))
    );
    const llm: BatchedEmbedLLM = { embedBatch, lastBatchTokens: 0 };

    await runBatchedEmbed(store, llm, docs, { batchSize: 50 });

    const rows = store.db.prepare(`SELECT hash, seq FROM content_vectors WHERE hash = ?`).all("hA") as { hash: string; seq: number }[];
    expect(rows).toHaveLength(1);
    expect(rows[0]!.seq).toBe(0);

    const vecRow = store.db.prepare(`SELECT hash_seq FROM vectors_vec WHERE hash_seq = ?`).get("hA_0");
    expect(vecRow).toBeTruthy();
  });
});

describe("runBatchedEmbed — failed-fragment handling (62xr.5: never write a null vector)", () => {
  it("does not write a vector for a null embedBatch result and marks that doc failed", async () => {
    const docs: DocEmbedTask[] = [
      docTask("hOk", "ok.md", [frag("fine")]),
      docTask("hBad", "bad.md", [frag("will fail")]),
    ];

    const embedBatch = mock(async (texts: string[]): Promise<(EmbeddingResult | null)[]> =>
      texts.map((t): EmbeddingResult | null =>
        t.includes("will fail") ? null : { embedding: fakeEmbedding(), model: "test-model" }
      )
    );
    const llm: BatchedEmbedLLM = { embedBatch, lastBatchTokens: 0 };

    const result = await runBatchedEmbed(store, llm, docs, { batchSize: 50 });

    expect(result.totalFragments).toBe(1);
    expect(result.failedFragments).toBe(1);
    expect(result.embedded).toBe(2); // both docs processed, one marked failed

    // The failed fragment must never appear as a vector row.
    const badRows = store.db.prepare(`SELECT * FROM content_vectors WHERE hash = ?`).all("hBad");
    expect(badRows).toHaveLength(0);

    const okRows = store.db.prepare(`SELECT * FROM content_vectors WHERE hash = ?`).all("hOk");
    expect(okRows).toHaveLength(1);
  });

  it("marks a document failed when its whole embedBatch chunk throws", async () => {
    const docs: DocEmbedTask[] = [docTask("hThrow", "throw.md", [frag("x")])];
    const embedBatch = mock(async (): Promise<(EmbeddingResult | null)[]> => {
      throw new Error("connection reset");
    });
    const llm: BatchedEmbedLLM = { embedBatch, lastBatchTokens: 0 };

    const result = await runBatchedEmbed(store, llm, docs, { batchSize: 50 });

    expect(result.failedFragments).toBe(1);
    expect(result.totalFragments).toBe(0);
    const rows = store.db.prepare(`SELECT * FROM content_vectors WHERE hash = ?`).all("hThrow");
    expect(rows).toHaveLength(0);
  });
});

describe("runBatchedEmbed — TPM pacing applies only to real rate-limited cloud APIs", () => {
  it("does not delay between batches when isCloudEmbed=false (self-hosted remote GPU)", async () => {
    const docs: DocEmbedTask[] = [
      docTask("h0", "doc0.md", [frag("a")]),
      docTask("h1", "doc1.md", [frag("b")]),
    ];
    const embedBatch = mock(async (texts: string[]) =>
      texts.map((): EmbeddingResult => ({ embedding: fakeEmbedding(), model: "test-model" }))
    );
    const llm: BatchedEmbedLLM = { embedBatch, lastBatchTokens: 0 };

    const start = Date.now();
    await runBatchedEmbed(store, llm, docs, {
      batchSize: 1, // force 2 separate batches, one per doc
      isCloudEmbed: false,
      tpmLimit: 1, // would force a huge delay if pacing were (incorrectly) applied
    });
    const elapsed = Date.now() - start;

    expect(elapsed).toBeLessThan(300);
  });

  it("applies the TPM pacing floor between batches when isCloudEmbed=true", async () => {
    const docs: DocEmbedTask[] = [
      docTask("h0", "doc0.md", [frag("a")]),
      docTask("h1", "doc1.md", [frag("b")]),
    ];
    const embedBatch = mock(async (texts: string[]) =>
      texts.map((): EmbeddingResult => ({ embedding: fakeEmbedding(), model: "test-model" }))
    );
    const llm: BatchedEmbedLLM = { embedBatch, lastBatchTokens: 0 };

    const start = Date.now();
    await runBatchedEmbed(store, llm, docs, {
      batchSize: 1, // force 2 separate batches
      isCloudEmbed: true,
      tpmLimit: 100000,
    });
    const elapsed = Date.now() - start;

    // requiredGapMs floors at 500ms; jitter is 0.85x-1.15x → expect at least ~400ms.
    expect(elapsed).toBeGreaterThanOrEqual(400);
  });
});
