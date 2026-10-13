import type { SearchResult } from "./store.ts";

export function vecLegParallelism(env: Record<string, string | undefined> = process.env): number {
  const raw = env.CLAWMEM_QUERY_VEC_PARALLEL?.trim();
  if (!raw) return 4;
  if (!/^-?\d+$/.test(raw)) return 4;
  const n = Number.parseInt(raw, 10);
  if (Number.isNaN(n)) return 4;
  if (n <= 1) return 1;
  if (n > 8) return 8;
  return n;
}

export async function runVecLegs<T>(
  queries: string[],
  runLeg: (q: string) => Promise<T>,
  concurrency: number
): Promise<T[]> {
  if (queries.length === 0) return [];
  const limit = Math.max(1, Math.floor(concurrency));
  if (limit <= 1) {
    const results: T[] = [];
    for (const q of queries) {
      results.push(await runLeg(q));
    }
    return results;
  }

  const results: T[] = new Array(queries.length);
  let nextIndex = 0;
  let inFlight = 0;
  let rejected = false;

  return new Promise<T[]>((resolve, reject) => {
    const launchNext = () => {
      if (rejected) return;
      if (nextIndex >= queries.length && inFlight === 0) {
        resolve(results);
        return;
      }

      while (inFlight < limit && nextIndex < queries.length) {
        const idx = nextIndex++;
        inFlight++;
        runLeg(queries[idx]!)
          .then((res) => {
            inFlight--;
            if (!rejected) {
              results[idx] = res;
              launchNext();
            }
          })
          .catch((err) => {
            inFlight--;
            if (!rejected) {
              rejected = true;
              reject(err);
            }
          });
      }
    };

    launchNext();
  });
}

export interface WorkerLegPoolOpts {
  dbPath: string;
  model: string;
  limit: number;
  collections?: string[];
}

export interface WorkerLegPool {
  size: number;
  runLeg(query: string): Promise<SearchResult[]>;
  close(): void;
}

interface WorkerSlot {
  worker: Worker;
  busy: boolean;
  currentPending?: {
    id: number;
    resolve: (res: SearchResult[]) => void;
    reject: (err: Error) => void;
  };
}

interface QueuedRequest {
  query: string;
  resolve: (res: SearchResult[]) => void;
  reject: (err: Error) => void;
}

export function createWorkerLegPool(size: number, opts: WorkerLegPoolOpts): WorkerLegPool {
  const poolSize = Math.max(1, size);
  const slots: WorkerSlot[] = [];
  const idleSlots: WorkerSlot[] = [];
  const queue: QueuedRequest[] = [];
  let nextId = 1;

  const processQueue = () => {
    while (queue.length > 0 && idleSlots.length > 0) {
      const req = queue.shift()!;
      const slot = idleSlots.pop()!;
      slot.busy = true;
      const reqId = nextId++;
      slot.currentPending = {
        id: reqId,
        resolve: req.resolve,
        reject: req.reject,
      };
      slot.worker.postMessage({
        id: reqId,
        dbPath: opts.dbPath,
        model: opts.model,
        query: req.query,
        limit: opts.limit,
        collections: opts.collections,
      });
    }
  };

  for (let i = 0; i < poolSize; i++) {
    const worker = new Worker(new URL("./vec-leg-worker.ts", import.meta.url).href);
    const slot: WorkerSlot = {
      worker,
      busy: false,
    };

    worker.onmessage = (ev: MessageEvent) => {
      const msg = ev.data;
      if (slot.currentPending && slot.currentPending.id === msg.id) {
        const { resolve, reject } = slot.currentPending;
        slot.currentPending = undefined;
        slot.busy = false;
        idleSlots.push(slot);
        if (msg.ok) {
          resolve(msg.results);
        } else {
          reject(new Error(msg.error));
        }
        processQueue();
      }
    };

    worker.onerror = (ev: ErrorEvent) => {
      if (slot.currentPending) {
        const { reject } = slot.currentPending;
        slot.currentPending = undefined;
        slot.busy = false;
        const err = ev.error instanceof Error ? ev.error : new Error(ev.message || "Worker error");
        reject(err);
        processQueue();
      }
    };

    slots.push(slot);
    idleSlots.push(slot);
  }

  const runLeg = (query: string): Promise<SearchResult[]> => {
    return new Promise<SearchResult[]>((resolve, reject) => {
      if (idleSlots.length > 0) {
        const slot = idleSlots.pop()!;
        slot.busy = true;
        const reqId = nextId++;
        slot.currentPending = {
          id: reqId,
          resolve,
          reject,
        };
        slot.worker.postMessage({
          id: reqId,
          dbPath: opts.dbPath,
          model: opts.model,
          query,
          limit: opts.limit,
          collections: opts.collections,
        });
      } else {
        queue.push({ query, resolve, reject });
      }
    });
  };

  const close = () => {
    for (const slot of slots) {
      if (slot.currentPending) {
        slot.currentPending.reject(new Error("Worker pool closed"));
        slot.currentPending = undefined;
      }
      slot.worker.terminate();
    }
    for (const req of queue) {
      req.reject(new Error("Worker pool closed"));
    }
    queue.length = 0;
    idleSlots.length = 0;
  };

  return {
    size: poolSize,
    runLeg,
    close,
  };
}
