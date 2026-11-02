import {
  createStore,
  enableVecScopeCache,
  FatalVectorError,
  prepareScopedVecScan,
  prewarmScopedVecScan,
  rethrowIfFatalVectorError,
  type SearchResult,
  type Store,
} from "./store.ts";
import { resetVecScopeScanStats, vecScopeScanStats, type VecScopeScanStats } from "./vec-scope-scan.ts";

import { postWorkerLoaded } from "./worker-load-guard.ts";

declare var self: Worker;

// One read-only connection per database path. A long-lived pool (MCP server) can serve
// several vaults; each worker opens a path at most once and keeps the handle warm.
const stores = new Map<string, Store>();

function storeFor(dbPath: string): Store {
  let st = stores.get(dbPath);
  if (!st) {
    st = createStore(dbPath, { readonly: true });
    stores.set(dbPath, st);
  }
  return st;
}

// The current query's scoped store (deadline path, vn4rz.131). It holds the scope's vectors, so
// it lives only as long as that query: the next scope id or a release closes it.
let scoped: { scopeId: string; store: Store | null } | null = null;

function releaseScoped(): void {
  try { scoped?.store?.close(); } catch {}
  scoped = null;
}

function scopedStoreFor(scopeId: string, dbPath: string, collections: string[], mmapBytes: number): Store | null {
  if (scoped?.scopeId === scopeId) return scoped.store;
  releaseScoped();
  resetVecScopeScanStats();
  let ro: Store | null = null;
  try {
    ro = createStore(dbPath, { readonly: true, mmapBytes });
    if (enableVecScopeCache(ro.db) && prepareScopedVecScan(ro.db, { collections }, false)) {
      scoped = { scopeId, store: ro };
      return ro;
    }
  } catch {
    // Fall through: the plain leg covers any failure to open or plan.
  }
  try { ro?.close(); } catch {}
  scoped = { scopeId, store: null };
  return null;
}

/**
 * One scoped leg, reading the scope's vectors while the embed is in flight as the CLI's in-process
 * scoped path does. The load stays on this thread: a nested read worker outlives terminate()
 * (measured on Bun 1.3.13), so it would escape the deadline.
 */
async function runScopedLeg(
  scopeId: string, dbPath: string, model: string, query: string, limit: number, collections: string[], mmapBytes: number,
): Promise<{ results: SearchResult[]; scopeStats: VecScopeScanStats }> {
  const ro = scopedStoreFor(scopeId, dbPath, collections, mmapBytes);
  let results: SearchResult[];
  if (ro) {
    try {
      const leg = ro.searchVec(query, model, limit, undefined, collections);
      leg.catch(() => {}); // Observed below; stops an early rejection reading as unhandled.
      await new Promise(resolve => setTimeout(resolve, 0));
      await prewarmScopedVecScan(ro.db, ro.dbPath, { collections }, 1);
      results = await leg;
    } catch (e) {
      rethrowIfFatalVectorError(e);
      results = await storeFor(dbPath).searchVec(query, model, limit, undefined, collections);
    }
  } else {
    results = await storeFor(dbPath).searchVec(query, model, limit, undefined, collections);
  }
  return { results, scopeStats: vecScopeScanStats() };
}

self.onmessage = async (ev: MessageEvent) => {
  const { id, dbPath, model, query, limit, collections, detailed, dateRange, excludeCollections, warm, scopeId, mmapBytes } = ev.data;
  if (ev.data.releaseScoped) {
    releaseScoped();
    return;
  }
  try {
    if (scopeId) {
      const results = await runScopedLeg(scopeId, dbPath, model, query, limit, collections, mmapBytes);
      self.postMessage({ id, ok: true, results });
      return;
    }
    const st = storeFor(dbPath);
    if (warm) {
      // Pre-warm (vn4rz.97): the worker is up and the vault is open; no search runs.
      self.postMessage({ id, ok: true, results: null });
      return;
    }
    const results = detailed
      ? await st.searchVecDetailed(query, model, limit, { collections, dateRange, excludeCollections })
      : await st.searchVec(query, model, limit, undefined, collections);
    self.postMessage({ id, ok: true, results });
  } catch (e) {
    self.postMessage({
      id,
      ok: false,
      error: e instanceof Error ? e.message : String(e),
      fatalName: e instanceof FatalVectorError ? e.name : undefined,
    });
  }
};

// Last statement of module evaluation: from here a terminate is safe (src/worker-load-guard.ts).
postWorkerLoaded();
