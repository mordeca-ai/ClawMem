import { describe, expect, it } from "bun:test";
import { loadScaledBound } from "../helpers/wall-clock.ts";

describe("loadScaledBound (vn4rz.139)", () => {
  it("never reads below the bound the test asked for", () => {
    expect(loadScaledBound(100, 10_000)).toBeGreaterThanOrEqual(100);
  });

  it("never passes the ceiling, however busy the host, so a run that waited out the stall stays red", () => {
    expect(loadScaledBound(1_000_000, 4900)).toBe(4900);
  });
});
