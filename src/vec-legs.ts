import type { SearchResult, VecSearchDetailedResult } from "./store.ts";

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

/** Full per-call configuration for a detailed leg (the MCP `query` path). */
export interface DetailedLegOpts {
  dbPath: string;
  model: string;
  limit: number;
  collections?: string[];
  dateRange?: { start: string; end: string };
  excludeCollections?: string[];
}

export interface WorkerLegPool {
  size: number;
  /** Plain `searchVec` leg with the pool's construction-time options (CLI `query`). */
  runLeg(query: string): Promise<SearchResult[]>;
  /** `searchVecDetailed` leg with per-call options (MCP `query`); carries the degraded marker. */
  runLegDetailed(query: string, opts: DetailedLegOpts): Promise<VecSearchDetailedResult>;
  close(): void;
}

interface Pending {
  id: number;
  resolve: (res: any) => void;
  reject: (err: Error) => void;
}

interface WorkerSlot {
  worker: Worker;
  pending?: Pending;
}

interface QueuedRequest {
  body: Record<string, unknown>;
  resolve: (res: any) => void;
  reject: (err: Error) => void;
}

/**
 * A bounded pool of read-only vector-leg workers. Workers are spawned lazily (never more
 * than `size`), so a long-lived owner (the MCP server) keeps them warm across calls and a
 * short-lived one (CLI) only pays for the legs it actually runs. A worker that errors is
 * dropped and its request rejected; the next queued request lazily spawns a replacement.
 */
export function createWorkerLegPool(size: number, opts: WorkerLegPoolOpts): WorkerLegPool {
  const poolSize = Math.max(1, Math.floor(size));
  const slots = new Set<WorkerSlot>();
  const idleSlots: WorkerSlot[] = [];
  const queue: QueuedRequest[] = [];
  let nextId = 1;
  let closed = false;

  const dropSlot = (slot: WorkerSlot) => {
    slots.delete(slot);
    const at = idleSlots.indexOf(slot);
    if (at >= 0) idleSlots.splice(at, 1);
    try { slot.worker.terminate(); } catch {}
  };

  const spawn = (): WorkerSlot | undefined => {
    let worker: Worker;
    try {
      worker = new Worker(new URL("./vec-leg-worker.ts", import.meta.url).href);
    } catch {
      return undefined;
    }
    const slot: WorkerSlot = { worker };

    worker.onmessage = (ev: MessageEvent) => {
      const msg = ev.data;
      const pending = slot.pending;
      if (!pending || pending.id !== msg.id) return;
      slot.pending = undefined;
      if (closed) return;
      idleSlots.push(slot);
      if (msg.ok) pending.resolve(msg.results);
      else pending.reject(new Error(msg.error));
      dispatch();
    };

    worker.onerror = (ev: ErrorEvent) => {
      const pending = slot.pending;
      slot.pending = undefined;
      dropSlot(slot);
      if (pending) {
        pending.reject(ev.error instanceof Error ? ev.error : new Error(ev.message || "Worker error"));
      }
      if (!closed) dispatch();
    };

    slots.add(slot);
    return slot;
  };

  const dispatch = () => {
    while (queue.length > 0 && !closed) {
      let slot = idleSlots.pop();
      if (!slot && slots.size < poolSize) slot = spawn();
      if (!slot) {
        // Nothing idle. Wait for a busy slot to free up, unless none exists at all
        // (the spawn failed): then reject rather than hang the caller.
        if (slots.size === 0) {
          const req = queue.shift()!;
          req.reject(new Error("Worker spawn failed"));
          continue;
        }
        break;
      }
      const req = queue.shift()!;
      const id = nextId++;
      slot.pending = { id, resolve: req.resolve, reject: req.reject };
      slot.worker.postMessage({ id, ...req.body });
    }
  };

  const submit = <R>(body: Record<string, unknown>): Promise<R> =>
    new Promise<R>((resolve, reject) => {
      if (closed) {
        reject(new Error("Worker pool closed"));
        return;
      }
      queue.push({ body, resolve, reject });
      dispatch();
    });

  const runLeg = (query: string): Promise<SearchResult[]> =>
    submit<SearchResult[]>({
      dbPath: opts.dbPath,
      model: opts.model,
      query,
      limit: opts.limit,
      collections: opts.collections,
    });

  const runLegDetailed = (query: string, o: DetailedLegOpts): Promise<VecSearchDetailedResult> =>
    submit<VecSearchDetailedResult>({
      detailed: true,
      dbPath: o.dbPath,
      model: o.model,
      query,
      limit: o.limit,
      collections: o.collections,
      dateRange: o.dateRange,
      excludeCollections: o.excludeCollections,
    });

  const close = () => {
    closed = true;
    for (const slot of [...slots]) {
      const pending = slot.pending;
      slot.pending = undefined;
      dropSlot(slot);
      pending?.reject(new Error("Worker pool closed"));
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
    runLegDetailed,
    close,
  };
}
