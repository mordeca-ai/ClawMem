import { describe, it, expect, beforeEach, afterEach, afterAll } from "bun:test";

/**
 * master-harness-5n0ew — oversized fragments whose fingerprint matched but whose stored
 * vector did not (doctor: "fingerprint matches but cos(stored, fresh) = 0.7420 < 0.98").
 *
 * The v1 fingerprint digested the text clawmem ASKED to embed, before the embed arm
 * truncated it, and said nothing about which arm embedded it. So a stored vector produced
 * under one truncation policy (or by the in-process fallback arm) and a fresh embed under
 * another carried the SAME fingerprint — the validator had no way to tell "different input"
 * from "corrupt vector".
 *
 * These tests drive the REAL LlamaCpp remote arms (single + batch) against an in-process
 * fake /v1/embeddings endpoint whose vector is a pure function of the exact text it
 * RECEIVED, so any difference in what was sent shows up as a vector difference.
 */

import { LlamaCpp, resetEmbedContextCache, localEmbedContextOptions, formatDocForEmbedding, LOCAL_EMBED_ARM_LABEL } from "../../src/llm.ts";
import { runBatchedEmbed, buildDocEmbedTask, parseRequeueHashesFile } from "../../src/clawmem.ts";
import { runSampledVectorValidation, cosineSim, classifyStoredVector } from "../../src/canary.ts";
import { createStore, type Store } from "../../src/store.ts";
import { acquireWorkerLease } from "../../src/worker-lease.ts";
import { hashContent } from "../../src/indexer.ts";
import { buildEmbedFrontmatter } from "../../src/embed-input.ts";
import { splitDocument } from "../../src/splitter.ts";
import { embedInputFingerprint, parseEmbedInputFp, sha256Hex, embedArmOf } from "../../src/embed-fingerprint.ts";
import { fakeEmbedVector } from "../helpers/fake-embed-server.ts";

const DIM = 512;
const MODEL = "fake-embed";

/** In-process fake ollama: /api/show advertises `contextTokens`; /v1/embeddings is f(exact input). */
function startFakeEmbed(contextTokens: () => number) {
  const seen: string[] = [];
  const server = Bun.serve({
    port: 0,
    async fetch(req) {
      const url = new URL(req.url);
      if (url.pathname === "/api/show") {
        return Response.json({ model_info: { "fake.context_length": contextTokens() } });
      }
      if (url.pathname === "/v1/embeddings") {
        const body = await req.json() as { input: string | string[] };
        const inputs = Array.isArray(body.input) ? body.input : [body.input];
        seen.push(...inputs);
        return Response.json({
          model: MODEL,
          data: inputs.map((t, index) => ({ index, embedding: fakeEmbedVector(t, DIM) })),
        });
      }
      return new Response("not found", { status: 404 });
    },
  });
  return { url: `http://127.0.0.1:${server.port}`, seen, stop: () => server.stop(true) };
}

/** Real LlamaCpp with the token-aware refinement pinned to identity — no local tokenizer
 *  model is ever loaded in a unit test; the char budget is the policy under test. */
class CharPolicyLlama extends LlamaCpp {
  protected override async truncateToTokenCeiling(text: string): Promise<string> { return text; }
}

let advertised = 256; // tokens → 512-char budget at the default 2 chars/token
const fake = startFakeEmbed(() => advertised);
afterAll(() => fake.stop());

const ENV_KEYS = ["CLAWMEM_EMBED_MAX_TOKENS", "CLAWMEM_EMBED_MAX_CHARS", "CLAWMEM_EMBED_CHARS_PER_TOKEN", "CLAWMEM_NO_LOCAL_MODELS"] as const;
const savedEnv: Record<string, string | undefined> = {};
beforeEach(() => {
  for (const k of ENV_KEYS) { savedEnv[k] = process.env[k]; delete process.env[k]; }
  process.env.CLAWMEM_NO_LOCAL_MODELS = "true";
  advertised = 256;
  resetEmbedContextCache();
  fake.seen.length = 0;
});
afterEach(() => {
  for (const k of ENV_KEYS) { if (savedEnv[k] === undefined) delete process.env[k]; else process.env[k] = savedEnv[k]; }
});

function newLlm(): CharPolicyLlama {
  return new CharPolicyLlama({ remoteEmbedUrl: fake.url, remoteEmbedModel: MODEL });
}

const COLLECTION = "records";
const PATH = "session__c08.md";
const TITLE = "session oversized · chunk 9/22";
// ~1.4 KB of distinct words: the seq-0 (whole-document) fragment is far over the 512-char
// budget, so the arm truncates it — the exact shape of the live failing row.
const BODY = `# Session\n\n${Array.from({ length: 160 }, (_, i) => `w${i}`).join(" ")}\n\n## Tail\n\n${Array.from({ length: 60 }, (_, i) => `tail${i}`).join(" ")}\n`;

function seedDoc(store: Store): { hash: string; fragText: string } {
  const hash = hashContent(BODY + COLLECTION + PATH);
  const now = new Date().toISOString();
  store.insertContent(hash, BODY, now);
  store.insertDocument(COLLECTION, PATH, TITLE, hash, now, now);
  const fm = buildEmbedFrontmatter(TITLE, PATH, null);
  const f0 = splitDocument(BODY, fm)[0]!;
  return { hash, fragText: formatDocForEmbedding(f0.content, f0.label || fm.title) };
}

async function embedThroughBatchPath(store: Store, llm: LlamaCpp, hash: string) {
  const task = buildDocEmbedTask(hash, BODY, PATH, TITLE, null, COLLECTION);
  const r = await runBatchedEmbed(store, llm, [task], { batchSize: 50 });
  expect(r.failedFragments).toBe(0);
  return task;
}

function storedRow(store: Store, hash: string, seq: number) {
  const cv = store.db.prepare(`SELECT embed_input_fp FROM content_vectors WHERE hash = ? AND seq = ?`).get(hash, seq) as { embed_input_fp: string };
  const vv = store.db.prepare(`SELECT embedding FROM vectors_vec WHERE hash_seq = ?`).get(`${hash}_${seq}`) as { embedding: Uint8Array };
  const b = new Uint8Array(vv.embedding);
  return { fp: cv.embed_input_fp, vec: new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)) };
}

describe("oversized fragment round trip (master-harness-5n0ew)", () => {
  it("batch-embedded oversized seq-0: the validator's single-path reconstruction yields the identical sent text, identical fingerprint, cos 1.0", async () => {
    const store = createStore(":memory:");
    const { hash, fragText } = seedDoc(store);
    expect(fragText.length).toBeGreaterThan(512); // precondition: the fragment IS oversized
    const llm = newLlm();
    await embedThroughBatchPath(store, llm, hash);

    // What the batch arm actually put on the wire for seq 0: the 512-char prefix.
    const sentAtEmbed = fake.seen.find(t => fragText.startsWith(t) && t.length < fragText.length);
    expect(sentAtEmbed).toBe(fragText.slice(0, 512));

    const stored = storedRow(store, hash, 0);
    // The fingerprint attests the POST-truncation text + the arm — not the request.
    expect(stored.fp).toBe(`v2:remote:${sha256Hex(sentAtEmbed!)}`);
    expect(stored.fp).not.toContain(sha256Hex(fragText));

    // Validator path: single llm.embed of the reconstructed fragment.
    const fresh = await llm.embed(fragText);
    expect(fresh!.input).toBe(sentAtEmbed!);
    expect(embedInputFingerprint(fragText, fresh)).toBe(stored.fp);
    expect(cosineSim(stored.vec, new Float32Array(fresh!.embedding))).toBeCloseTo(1.0, 6);

    // And doctor's sampled validator, end to end, validates every row.
    const r = await runSampledVectorValidation(store, t => llm.embed(t));
    expect(r.definitiveFailures).toEqual([]);
    expect(r.stalePolicy).toBe(0);
    expect(r.validated).toBe(r.target);
    expect(r.validatedSeq0).toBe(1);
  });

  it("single and batch arms report the SAME sent text for the same input (one truncation policy)", async () => {
    const llm = newLlm();
    const text = "x ".repeat(700);
    const single = await llm.embed(text);
    const [batched] = await llm.embedBatch([text]);
    expect(single!.input).toBe(text.slice(0, 512));
    expect(batched!.input).toBe(single!.input);
    expect(embedInputFingerprint(text, batched)).toBe(embedInputFingerprint(text, single));
  });

  it("NEGATIVE CONTROL — truncation policy differs between embed and validate: v1 would read 'fingerprint matches' over a cos < 0.98 pair (the live symptom); v2 names it stale-input, not corruption", async () => {
    const store = createStore(":memory:");
    const { hash, fragText } = seedDoc(store);
    await embedThroughBatchPath(store, newLlm(), hash); // embedded under a 512-char budget

    advertised = 150; // the endpoint (or env) now yields a 300-char budget
    resetEmbedContextCache();
    const validatorLlm = newLlm();
    const fresh = await validatorLlm.embed(fragText);
    expect(fresh!.input!.length).toBe(300);
    const stored = storedRow(store, hash, 0);

    // RED under the old scheme: the v1 digest is identical on both sides...
    expect(sha256Hex(fragText)).toBe(sha256Hex(fragText));
    // ...while the vectors genuinely disagree — "fingerprint matches but cos < 0.98".
    expect(cosineSim(stored.vec, new Float32Array(fresh!.embedding))).toBeLessThan(0.98);

    // v2 sees it: the text embedded is not the text the embedder sends today.
    expect(embedInputFingerprint(fragText, fresh)).not.toBe(stored.fp);
    const r = await runSampledVectorValidation(store, t => validatorLlm.embed(t));
    expect(r.definitiveFailures.length).toBe(1);
    expect(r.definitiveFailures[0]).toStartWith("stale-input:");
    expect(r.definitiveFailures[0]).toContain("truncation policy");
    expect(r.definitiveFailures.some(f => f.startsWith("corruption/drift"))).toBe(false);
  });

  it("legacy v1 row on a TRUNCATED input is classified stale-policy (needs re-embed), not DEFINITIVE corruption", async () => {
    const store = createStore(":memory:");
    const { hash, fragText } = seedDoc(store);
    const llm = newLlm();
    await embedThroughBatchPath(store, llm, hash);
    // Simulate a pre-5n0ew row: v1 digest of the untruncated text, vector from some other
    // prefix (what an old policy / other arm produced).
    store.db.prepare(`UPDATE content_vectors SET embed_input_fp = ? WHERE hash = ? AND seq = 0`).run(sha256Hex(fragText), hash);
    const other = new Float32Array(fakeEmbedVector(fragText.slice(0, 200), DIM));
    store.db.prepare(`DELETE FROM vectors_vec WHERE hash_seq = ?`).run(`${hash}_0`);
    store.db.prepare(`INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)`).run(`${hash}_0`, other);

    const r = await runSampledVectorValidation(store, t => llm.embed(t));
    expect(r.definitiveFailures).toEqual([]);
    expect(r.stalePolicy).toBe(1);
    expect(r.stalePolicyRows[0]).toStartWith("stale-policy:");
    expect(r.stalePolicyRows[0]).toContain(`${COLLECTION}/${PATH}#0`);
    expect(r.validatedSeq0).toBe(0); // skipped rows never count toward the seq-0 quota
  });

  it("legacy v1 row on an UNTRUNCATED input keeps full validation (v1 and v2 agree when nothing is cut)", async () => {
    const store = createStore(":memory:");
    const { hash } = seedDoc(store);
    const llm = newLlm();
    await embedThroughBatchPath(store, llm, hash);
    // Rewrite every non-truncated row to its v1 digest: still validates at cos 1.0.
    const task = buildDocEmbedTask(hash, BODY, PATH, TITLE, null, COLLECTION);
    for (let seq = 1; seq < task.fragments.length; seq++) {
      const f = task.fragments[seq]!;
      const t = formatDocForEmbedding(f.content, f.label || task.title);
      if (t.length > 512) continue;
      store.db.prepare(`UPDATE content_vectors SET embed_input_fp = ? WHERE hash = ? AND seq = ?`).run(sha256Hex(t), hash, seq);
    }
    const r = await runSampledVectorValidation(store, t => llm.embed(t));
    expect(r.definitiveFailures).toEqual([]);
    expect(r.validated).toBeGreaterThan(0);
  });

  it("same bytes embedded by the in-process arm is an arm-mismatch (stale-policy), never a cross-arm cosine verdict", async () => {
    const store = createStore(":memory:");
    const { hash, fragText } = seedDoc(store);
    const llm = newLlm();
    await embedThroughBatchPath(store, llm, hash);
    const sent = fragText.slice(0, 512);
    store.db.prepare(`UPDATE content_vectors SET embed_input_fp = ? WHERE hash = ? AND seq = 0`)
      .run(embedInputFingerprint(fragText, { input: sent, endpoint: LOCAL_EMBED_ARM_LABEL }), hash);
    const r = await runSampledVectorValidation(store, t => llm.embed(t));
    expect(r.definitiveFailures).toEqual([]);
    expect(r.stalePolicy).toBe(1);
    expect(r.stalePolicyRows[0]).toStartWith("arm-mismatch:");
  });
});

describe("in-process embed context (master-harness-5n0ew)", () => {
  it("evaluates the whole input in ONE batch: batchSize === contextSize (node-llama-cpp defaults to 512)", () => {
    for (const train of [2048, 512, 8192, undefined, 0]) {
      const o = localEmbedContextOptions(train);
      expect(o.batchSize).toBe(o.contextSize);
    }
    expect(localEmbedContextOptions(2048)).toEqual({ contextSize: 2048, batchSize: 2048 });
    expect(localEmbedContextOptions(undefined).contextSize).toBe(2048);
  });

  it("ensureEmbedContext actually passes those options to createEmbeddingContext (wiring, skeptic F1)", async () => {
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    let received: unknown = "never-called";
    (llm as any).ensureEmbedModel = async () => ({
      trainContextSize: 2048,
      createEmbeddingContext: async (opts?: unknown) => {
        received = opts;
        return { dispose: async () => {} };
      },
    });
    await (llm as any).ensureEmbedContext();
    expect(received).toEqual({ contextSize: 2048, batchSize: 2048 });
  });
});

describe("classifyStoredVector cross-arm (skeptic F2)", () => {
  it("a v2:local row validated on the remote arm with a DIFFERENT sent prefix is arm-mismatch, not stale-input", () => {
    const text = "word ".repeat(1200);
    const fp = embedInputFingerprint(text, { endpoint: LOCAL_EMBED_ARM_LABEL, input: text.slice(0, 4096) });
    const v = new Float32Array([1, 0, 0]);
    const r = classifyStoredVector(fp, text, v, { embedding: [1, 0, 0], endpoint: "http://x:11434", input: text.slice(0, 3000) });
    expect(r.kind).toBe("arm-mismatch");
  });
  it("same arm, different sent text is still stale-input", () => {
    const text = "word ".repeat(1200);
    const fp = embedInputFingerprint(text, { endpoint: "http://x:11434", input: text.slice(0, 4096) });
    const r = classifyStoredVector(fp, text, new Float32Array([1, 0, 0]), { embedding: [1, 0, 0], endpoint: "http://x:11434", input: text.slice(0, 3000) });
    expect(r.kind).toBe("stale-input");
  });
});

describe("fingerprint format + targeted re-embed plumbing (master-harness-5n0ew)", () => {
  it("parses v2, reads bare/garbled values as opaque v1, absent as legacy", () => {
    const hex = "a".repeat(64);
    expect(parseEmbedInputFp(`v2:remote:${hex}`)).toEqual({ version: 2, arm: "remote", hex });
    expect(parseEmbedInputFp(`v2:local:${hex}`)).toEqual({ version: 2, arm: "local", hex });
    expect(parseEmbedInputFp(hex)).toEqual({ version: 1, hex });
    expect(parseEmbedInputFp("wrong")).toEqual({ version: 1, hex: "wrong" });
    expect(parseEmbedInputFp(null)).toBeNull();
    expect(embedArmOf({ endpoint: LOCAL_EMBED_ARM_LABEL })).toBe("local");
    expect(embedArmOf({ endpoint: "http://x:11434" })).toBe("remote");
    expect(embedArmOf({})).toBe("unreported");
  });

  it("parseRequeueHashesFile accepts hashes + comments, rejects anything else loudly", () => {
    const h1 = "b".repeat(64), h2 = "c".repeat(64);
    expect(parseRequeueHashesFile(`# stale\n${h1}\n\n${h2}  # seq0\n${h1}\n`)).toEqual([h1, h2]);
    expect(() => parseRequeueHashesFile("records/foo.md\n")).toThrow(/64-hex/);
    expect(() => parseRequeueHashesFile("# nothing\n")).toThrow(/no hashes/);
  });

  it("requeueEmbeds is lease-fenced and puts synced docs back on the worklist", () => {
    const store = createStore(":memory:");
    const { hash } = seedDoc(store);
    store.markEmbedSynced(hash);
    store.db.prepare(`INSERT INTO content_vectors (hash, seq, pos, model, embedded_at, fragment_type) VALUES (?, 0, 0, 'm', 'now', 'full')`).run(hash);
    expect(store.getHashesNeedingFragments().map(h => h.hash)).not.toContain(hash);
    const lease = acquireWorkerLease(store, "embedding", 60_000);
    expect(() => store.requeueEmbeds([hash], { workerName: "embedding", token: "not-the-token" })).toThrow();
    expect(store.requeueEmbeds([hash], { workerName: "embedding", token: lease.token! })).toBe(1);
    expect(store.getHashesNeedingFragments().map(h => h.hash)).toContain(hash);
  });
});
