import { createStore, type Store } from "./store.ts";
import { queryBudgetTestFtsFail, queryBudgetTestStall } from "./query-budget.ts";

import { postWorkerLoaded } from "./worker-load-guard.ts";

declare var self: Worker;

// One read-only connection per database path, opened on the first message that names it and kept
// warm. Same open options as the vector-leg worker: query_only, WAL, 5 s busy timeout.
const stores = new Map<string, Store>();

function storeFor(dbPath: string): Store {
  let st = stores.get(dbPath);
  if (!st) {
    st = createStore(dbPath, { readonly: true });
    stores.set(dbPath, st);
  }
  return st;
}

/**
 * The query budget's FTS worker (master-harness-vn4rz.135). It runs `Store.searchFTS` unchanged,
 * so a result is the one the main thread would have computed for the same arguments.
 */
self.onmessage = (ev: MessageEvent) => {
  const { id, dbPath, warm, query, limit, collections } = ev.data;
  try {
    const st = storeFor(dbPath);
    if (warm) {
      self.postMessage({ id, ok: true, results: null });
      return;
    }
    // The test seam stalls HERE, inside the worker, so a test can only pass when FTS is off the
    // main thread. A no-op unless CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MS is set.
    queryBudgetTestStall(query);
    queryBudgetTestFtsFail(query);
    self.postMessage({ id, ok: true, results: st.searchFTS(query, limit, undefined, collections) });
  } catch (e) {
    self.postMessage({ id, ok: false, error: e instanceof Error ? e.message : String(e) });
  }
};

// Last statement of module evaluation: from here a terminate is safe (src/worker-load-guard.ts).
postWorkerLoaded();
