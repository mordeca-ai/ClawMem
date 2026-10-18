/**
 * ADR-0085 session-focus scope for the PG read path (master-harness-vn4rz.111).
 *
 * Parity with the sqlite CLI's resolveCollectionScope (src/clawmem.ts): the
 * precedence decision is the SAME pure function, computeCollectionScope —
 * explicit -c > session focus naming a real collection > unscoped. This module
 * only does the IO the sqlite wrapper does (session id, focus topic).
 *
 * An explicit --collection is passed through untouched (the PG path never
 * validated names; this change does not alter that). Fail-open: any missing
 * session id / focus file / unknown focus topic yields undefined (unscoped).
 */

import { computeCollectionScope } from "../collection-scope.ts";
import { resolveEnvSessionId, resolveSessionTopic } from "../session-focus.ts";

export function resolvePgCollectionScope(
  explicit: string[] | undefined,
  knownNames: string[],
  sessionId?: string,
): string[] | undefined {
  if (explicit !== undefined) return explicit;
  const sid = sessionId?.trim() || resolveEnvSessionId();
  const focus = resolveSessionTopic(sid, process.env.CLAWMEM_SESSION_FOCUS);
  const r = computeCollectionScope(undefined, focus, knownNames);
  return r && !("error" in r) ? r.collections : undefined;
}
