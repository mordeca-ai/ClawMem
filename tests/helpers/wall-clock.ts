import { cpus, loadavg } from "os";

/**
 * Upper bound for a wall-clock assertion that has to survive a loaded host (vn4rz.139).
 *
 * A budget test proves "this was cut short", and what it cuts short is a known stall: a 5 s
 * sleep, a 4 s busy_timeout. A bound of "budget + small epsilon" also measures how busy the
 * machine is, so it fails on load alone. This stretches the bound with the host's run-queue
 * pressure, the same load/cpus factor tools/clawmem-lander applies to bun's --timeout, but never
 * past `ceilingMs`, which is the required second argument: name the stall, and pass a ceiling
 * just under it, so a run that really waited it out stays red however busy the host is.
 */
export function loadScaledBound(ms: number, ceilingMs: number): number {
  const pressure = loadavg()[0]! / Math.max(1, cpus().length);
  return Math.min(ms * Math.max(1, pressure), ceilingMs);
}
