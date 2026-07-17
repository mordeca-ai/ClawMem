import { describe, it, expect } from "bun:test";
import { getCacheKey } from "../../src/store.ts";

// master-harness d0hz / ADR-0059 "cache-key collision" footgun: the rerank llm_cache key
// is getCacheKey("rerank", {query, file, model}). Pre-fix it omitted the backend URL, so the
// SAME (query,file,model) reranked once via the yoshiee-bge endpoint and once via the
// OpenRouter/Cohere cloud proxy collided on one cache row and could replay a stale
// cross-backend score. The fix threads the backend URL (CLAWMEM_RERANK_URL) into the key.
describe("d0hz: rerank cache key is namespaced by backend URL", () => {
  const base = { query: "what is the capital of france", file: "geo/france.md", model: "reranker-x" };

  it("distinct backend URLs produce distinct cache keys (no cross-backend collision)", () => {
    const yoshiee = getCacheKey("rerank", { ...base, rerankUrl: "http://192.168.2.15:8787" });
    const cloud = getCacheKey("rerank", { ...base, rerankUrl: "http://127.0.0.1:9099/openrouter" });
    expect(yoshiee).not.toBe(cloud);
  });

  it("the same backend URL yields a stable key (cache still hits within one backend)", () => {
    const a = getCacheKey("rerank", { ...base, rerankUrl: "http://192.168.2.15:8787" });
    const b = getCacheKey("rerank", { ...base, rerankUrl: "http://192.168.2.15:8787" });
    expect(a).toBe(b);
  });

  it("local (undefined rerankUrl) key is UNCHANGED from the pre-fix {query,file,model} shape — no cache invalidation for local-only users", () => {
    // JSON.stringify omits an undefined value, so {query,file,model,rerankUrl:undefined}
    // stringifies identically to the historic {query,file,model} body.
    const withUndefined = getCacheKey("rerank", { ...base, rerankUrl: undefined });
    const preFixShape = getCacheKey("rerank", base);
    expect(withUndefined).toBe(preFixShape);
  });

  it("a URL-backed key differs from the local (no-URL) key", () => {
    const local = getCacheKey("rerank", base);
    const remote = getCacheKey("rerank", { ...base, rerankUrl: "http://192.168.2.15:8787" });
    expect(local).not.toBe(remote);
  });
});
