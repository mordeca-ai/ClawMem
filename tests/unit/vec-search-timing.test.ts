import { afterEach, expect, test } from "bun:test";
import { createStore, searchVec, withVecSearchTiming, type VecSearchTimingPhase } from "../../src/store.ts";
import { setDefaultLlamaCpp } from "../../src/llm.ts";

afterEach(() => setDefaultLlamaCpp(null));

test("worker timing separates embedding and scan and does not leak into the next request", async () => {
  const store = createStore(":memory:");
  store.ensureVecTable(4);
  setDefaultLlamaCpp({ embed: async () => ({ embedding: [1, 0, 0, 0], model: "model-A" }) } as any);
  const phases: VecSearchTimingPhase[] = [];
  await withVecSearchTiming((phase, ms) => {
    expect(ms).toBeGreaterThanOrEqual(0);
    phases.push(phase);
  }, () => searchVec(store.db, "first", "model-A"));
  expect(phases).toEqual(["embed_ms", "scan_ms"]);

  await searchVec(store.db, "second", "model-A");
  expect(phases).toEqual(["embed_ms", "scan_ms"]);
  store.close();
});

test("worker timing observer clears after a failed embedding", async () => {
  const store = createStore(":memory:");
  store.ensureVecTable(4);
  setDefaultLlamaCpp({ embed: async () => { throw new Error("embed failed"); } } as any);
  const phases: VecSearchTimingPhase[] = [];
  await expect(withVecSearchTiming((phase) => phases.push(phase), () => searchVec(store.db, "first", "model-A")))
    .rejects.toThrow("embed failed");
  expect(phases).toEqual(["embed_ms"]);
  setDefaultLlamaCpp({ embed: async () => ({ embedding: [1, 0, 0, 0], model: "model-A" }) } as any);
  await searchVec(store.db, "second", "model-A");
  expect(phases).toEqual(["embed_ms"]);
  store.close();
});

test("overlapping in-process legs keep their timing observers separate", async () => {
  const store = createStore(":memory:");
  store.ensureVecTable(4);
  setDefaultLlamaCpp({ embed: async () => {
    await new Promise((resolve) => setTimeout(resolve, 5));
    return { embedding: [1, 0, 0, 0], model: "model-A" };
  } } as any);
  const first: VecSearchTimingPhase[] = [];
  const second: VecSearchTimingPhase[] = [];
  await Promise.all([
    withVecSearchTiming((phase) => first.push(phase), () => searchVec(store.db, "first", "model-A")),
    withVecSearchTiming((phase) => second.push(phase), () => searchVec(store.db, "second", "model-A")),
  ]);
  expect(first).toEqual(["embed_ms", "scan_ms"]);
  expect(second).toEqual(["embed_ms", "scan_ms"]);
  store.close();
});
