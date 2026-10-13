// vn4rz.100: reads a share of a scope's vec0 chunks for loadScopedVectorsParallel.
// Plain bun:sqlite on a read-only connection: the shadow tables are ordinary tables, so the
// sqlite-vec extension is not loaded and the worker stays cheap to start.
import { Database } from "bun:sqlite";
import { readChunkTasks, type ChunkReadRequest, type ChunkReadResponse } from "./vec-scope-scan.ts";

declare var self: Worker;

self.onmessage = (ev: MessageEvent) => {
  const req = ev.data as ChunkReadRequest;
  let db: Database | undefined;
  let reply: ChunkReadResponse;
  const vectors = new Float32Array(req.rows * req.dim);
  const mags = new Float32Array(req.rows);
  try {
    db = new Database(req.dbPath, { readonly: true });
    db.exec("PRAGMA busy_timeout = 5000");
    if (req.mmapBytes > 0) db.exec(`PRAGMA mmap_size = ${req.mmapBytes}`);
    db.exec("BEGIN");
    const bytesRead = readChunkTasks(db, req.dim, req.tasks, vectors, mags);
    db.exec("COMMIT");
    reply =
      bytesRead < 0
        ? { ok: false, error: "chunk verification failed" }
        : { ok: true, vectors, mags, bytesRead };
  } catch (e) {
    reply = { ok: false, error: e instanceof Error ? e.message : String(e) };
  } finally {
    try {
      db?.close();
    } catch {
      /* closing a failed handle */
    }
  }
  if (reply.ok) {
    self.postMessage(reply, {
      transfer: [reply.vectors.buffer as ArrayBuffer, reply.mags.buffer as ArrayBuffer],
    });
  } else {
    self.postMessage(reply);
  }
};
