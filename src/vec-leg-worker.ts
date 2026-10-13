import { createStore, type Store } from "./store.ts";

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

self.onmessage = async (ev: MessageEvent) => {
  const { id, dbPath, model, query, limit, collections, detailed, dateRange, excludeCollections, warm } = ev.data;
  try {
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
    });
  }
};
