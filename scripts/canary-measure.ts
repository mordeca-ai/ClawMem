/**
 * master-harness-vn4rz.60 — run the geometry canary battery once against the CONFIGURED
 * embed arm and print every margin. Read-only: no vault is opened and no baseline is read
 * or written (intrinsic floors only, as on a fresh profile).
 *
 * The arm is whatever getDefaultLlamaCpp() resolves from the environment, exactly as
 * `clawmem embed` would:
 *   remote:     CLAWMEM_EMBED_URL=http://host:11434 CLAWMEM_EMBED_MODEL=embeddinggemma \
 *                 bun scripts/canary-measure.ts
 *   in-process: env -u CLAWMEM_EMBED_URL nice -n 19 bun scripts/canary-measure.ts
 *
 * Prints one JSON line: { arm, profileKey, pass, margins, failures }.
 * Exit codes: 0 canary PASS · 1 canary FAIL · 2 unavailable (nothing embedded).
 */
import { getDefaultLlamaCpp } from "../src/llm.ts";
import { runCanaryBattery } from "../src/canary.ts";

const llm = getDefaultLlamaCpp();
const arms = new Set<string>();
const outcome = await runCanaryBattery(
  async (t) => {
    const r = await llm.embed(t);
    if (r?.endpoint) arms.add(r.endpoint);
    return r;
  },
  () => null,
);

if ("unavailable" in outcome) {
  console.log(JSON.stringify({ arm: [...arms], unavailable: outcome.reason }));
  process.exit(2);
}
const margins = Object.fromEntries(
  Object.entries(outcome.margins).map(([k, v]) => [k, +v.toFixed(4)]),
);
console.log(
  JSON.stringify({
    arm: [...arms],
    profileKey: outcome.profileKey,
    pass: outcome.pass,
    margins,
    failures: outcome.failures,
  }),
);
process.exit(outcome.pass ? 0 : 1);
