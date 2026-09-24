import { describe, it, expect } from "bun:test";

/**
 * master-harness-vn4rz.59: `clawmem doctor --clear-taint` — the evidence-gated clear of
 * embed_geometry_taint. Clears ONLY under the embed lease, on a passing canary (incl.
 * drift vs baseline), and a sampled-vector run that reaches the target with no
 * definitive failure; every refusal leaves the taint in place.
 */

import { createHash } from "crypto";
import { canaryProbeInputs, persistCanaryBaselineIfFirst, runCanaryBattery } from "../../src/canary.ts";
import { createStore, canonicalDocId, type Store } from "../../src/store.ts";
import { acquireWorkerLease } from "../../src/worker-lease.ts";
import { hashContent } from "../../src/indexer.ts";
import { formatDocForEmbedding } from "../../src/llm.ts";
import {
  clearGeometryTaint,
  describeLastTaintClear,
  EMBED_LEASE_NAME,
  TAINT_CLEARED_FLAG,
  TAINT_FLAG,
  type ClearTaintEmbed,
} from "../../src/taint-clear.ts";

const TAINT_REASON = "no preflight validation at 2026-09-18T09:19:13Z";

/** Healthy canary geometry for probe inputs (same shape as canary-validation.test.ts). */
const LONG_BASES: Record<string, number[]> = { long_head: [0, 0, 0, 1], long_tail: [0, 0, 1, 0], long_doc: [0, 0, 0.35, 1] };
function healthyProbe(text: string): Float32Array | null {
  let bucket = 0;
  let longBase: number[] | undefined;
  for (const [id, t] of canaryProbeInputs()) {
    if (t === text) { longBase = LONG_BASES[id]; bucket = id === "unrel" ? 1 : id.startsWith("rel") ? 2 : 3; break; }
  }
  if (bucket === 0) return null;
  const jitter = (createHash("sha256").update(text).digest()[0]! / 255) * 0.05;
  const base = longBase ?? (bucket === 1 ? [0, 1, 0, 0] : bucket === 2 ? [1, 0, 0, 0] : [0.9, 0, 0.45, 0]);
  return new Float32Array([base[0]! + jitter, base[1]!, base[2]!, base[3]! + jitter * 0.5]);
}

/** Deterministic per-text vector for vault fragments. */
function textEmbed(text: string): Float32Array {
  const d = createHash("sha256").update(text).digest();
  return new Float32Array([d[0]! + 1, d[1]! + 1, d[2]! + 1, d[3]! + 1]);
}

const healthyEmbed: ClearTaintEmbed = async t => ({ embedding: healthyProbe(t) ?? textEmbed(t), model: "fake" });

function seedEmbedded(store: Store, path: string, body: string, opts?: { corruptVector?: boolean }) {
  const hash = hashContent(body + "user" + path);
  const now = new Date().toISOString();
  store.insertContent(hash, body, now);
  store.insertDocument("user", path, path, hash, now, now);
  store.markEmbedSynced(hash);
  store.ensureVecTable(4);
  const embedText = formatDocForEmbedding(body, path);
  const fp = createHash("sha256").update(embedText, "utf8").digest("hex");
  const vector = opts?.corruptVector ? new Float32Array([0, 0, 0, 1]) : textEmbed(embedText);
  store.insertEmbedding(hash, 0, 0, vector, "fake", now, "full", undefined, canonicalDocId("user", path), undefined, fp);
  return hash;
}

/** Tainted vault with `n` healthy rows and a stored canary baseline for the healthy geometry. */
async function taintedVault(n: number): Promise<Store> {
  const store = createStore(":memory:");
  for (let i = 0; i < n; i++) seedEmbedded(store, `d${i}.md`, `healthy document number ${i}`);
  const baseline = await runCanaryBattery(healthyEmbed, () => null);
  if ("unavailable" in baseline) throw new Error("unavailable");
  persistCanaryBaselineIfFirst(store, baseline, { recalibrate: false });
  store.setVaultFlag(TAINT_FLAG, TAINT_REASON);
  return store;
}

describe("doctor --clear-taint (evidence-gated taint clear)", () => {
  it("no taint → no-op, exit 0, no audit record", async () => {
    const store = createStore(":memory:");
    let embeds = 0;
    const out = await clearGeometryTaint(store, async t => { embeds++; return healthyEmbed(t); }, { sample: 5 });
    expect(out.status).toBe("no-taint");
    expect(out.exitCode).toBe(0);
    expect(embeds).toBe(0);
    expect(store.getVaultFlag(TAINT_CLEARED_FLAG)).toBeNull();
  });

  it("taint + canary pass + sampler all-validated → cleared, audit written, lease released", async () => {
    const store = await taintedVault(6);
    const out = await clearGeometryTaint(store, healthyEmbed, { sample: 5, now: () => new Date("2026-09-23T00:00:00Z") });
    expect(out.status).toBe("cleared");
    expect(out.exitCode).toBe(0);
    expect(store.getVaultFlag(TAINT_FLAG)).toBeNull();
    const audit = JSON.parse(store.getVaultFlag(TAINT_CLEARED_FLAG)!);
    expect(audit.prior_reason).toBe(TAINT_REASON);
    expect(audit.method).toBe("doctor --clear-taint");
    expect(audit.cleared_at).toBe("2026-09-23T00:00:00.000Z");
    expect(audit.sample_target).toBe(5);
    expect(audit.validated).toBe(5);
    expect(audit.canary.pass).toBe(true);
    expect(audit.canary.drift_checked).toBe(true);
    expect(describeLastTaintClear(store.getVaultFlag(TAINT_CLEARED_FLAG))).toContain("doctor --clear-taint");
    // Lease released: an embed can acquire it immediately.
    expect(acquireWorkerLease(store, EMBED_LEASE_NAME, 60_000).acquired).toBe(true);
  });

  it("NEGATIVE: sampler hits a corruption/drift row → refused (sampler gate), taint still set", async () => {
    const store = await taintedVault(3);
    seedEmbedded(store, "corrupt.md", "corrupted document", { corruptVector: true });
    const out = await clearGeometryTaint(store, healthyEmbed, { sample: 4 });
    expect(out.status).toBe("refused");
    expect(out.exitCode).toBe(1);
    if (out.status !== "refused") throw new Error("unreachable");
    expect(out.gate).toBe("sampler");
    expect(out.details.some(d => d.startsWith("corruption/drift:"))).toBe(true);
    expect(store.getVaultFlag(TAINT_FLAG)).toBe(TAINT_REASON);
    expect(store.getVaultFlag(TAINT_CLEARED_FLAG)).toBeNull();
  });

  it("NEGATIVE: target N not reachable (fewer eligible rows) → refused INCOMPLETE, taint still set", async () => {
    const store = await taintedVault(3);
    const out = await clearGeometryTaint(store, healthyEmbed, { sample: 200 });
    expect(out.status).toBe("refused");
    if (out.status !== "refused") throw new Error("unreachable");
    expect(out.gate).toBe("sampler");
    expect(out.message).toContain("INCOMPLETE");
    expect(store.getVaultFlag(TAINT_FLAG)).toBe(TAINT_REASON);
  });

  it("NEGATIVE: canary unavailable → refused (canary gate), taint still set", async () => {
    const store = await taintedVault(6);
    const out = await clearGeometryTaint(store, async () => null, { sample: 5 });
    expect(out.status).toBe("refused");
    if (out.status !== "refused") throw new Error("unreachable");
    expect(out.gate).toBe("canary");
    expect(out.message).toContain("unavailable");
    expect(store.getVaultFlag(TAINT_FLAG)).toBe(TAINT_REASON);
  });

  it("NEGATIVE: canary fails (collapsed geometry) → refused (canary gate), taint still set", async () => {
    const store = await taintedVault(6);
    const collapsed: ClearTaintEmbed = async t => {
      const j = (createHash("sha256").update(t).digest()[0]! / 255) * 0.0001;
      return { embedding: new Float32Array([1, j, 0, 0]), model: "fake" };
    };
    const out = await clearGeometryTaint(store, collapsed, { sample: 5 });
    expect(out.status).toBe("refused");
    if (out.status !== "refused") throw new Error("unreachable");
    expect(out.gate).toBe("canary");
    expect(out.message).toContain("FAILED");
    expect(store.getVaultFlag(TAINT_FLAG)).toBe(TAINT_REASON);
  });

  it("NEGATIVE: canary passes but no stored baseline (drift unchecked) → refused, taint still set", async () => {
    const store = createStore(":memory:");
    for (let i = 0; i < 6; i++) seedEmbedded(store, `d${i}.md`, `healthy document number ${i}`);
    store.setVaultFlag(TAINT_FLAG, TAINT_REASON);
    const out = await clearGeometryTaint(store, healthyEmbed, { sample: 5 });
    expect(out.status).toBe("refused");
    if (out.status !== "refused") throw new Error("unreachable");
    expect(out.gate).toBe("canary");
    expect(out.message).toContain("baseline");
    expect(store.getVaultFlag(TAINT_FLAG)).toBe(TAINT_REASON);
  });

  it("NEGATIVE: embed lease held by another writer → refused (lease gate), no embeds, taint still set", async () => {
    const store = await taintedVault(6);
    const other = acquireWorkerLease(store, EMBED_LEASE_NAME, 60_000);
    expect(other.acquired).toBe(true);
    let embeds = 0;
    const out = await clearGeometryTaint(store, async t => { embeds++; return healthyEmbed(t); }, { sample: 5 });
    expect(out.status).toBe("refused");
    if (out.status !== "refused") throw new Error("unreachable");
    expect(out.gate).toBe("lease");
    expect(embeds).toBe(0);
    expect(store.getVaultFlag(TAINT_FLAG)).toBe(TAINT_REASON);
    // The other writer's lease is untouched.
    const row = store.db.prepare(`SELECT lease_token FROM worker_leases WHERE worker_name = ?`).get(EMBED_LEASE_NAME) as { lease_token: string };
    expect(row.lease_token).toBe(other.token!);
  });
});
