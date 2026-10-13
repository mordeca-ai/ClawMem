import { createStore, type Store } from "./store.ts";

declare var self: Worker;

let st: Store | undefined;

self.onmessage = async (ev: MessageEvent) => {
  const { id, dbPath, model, query, limit, collections } = ev.data;
  try {
    if (!st) {
      st = createStore(dbPath, { readonly: true });
    }
    const results = await st.searchVec(query, model, limit, undefined, collections);
    self.postMessage({ id, ok: true, results });
  } catch (e) {
    self.postMessage({
      id,
      ok: false,
      error: e instanceof Error ? e.message : String(e),
    });
  }
};
