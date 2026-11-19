/** The confidence threshold: what it keeps, how the tile list counts it, how it is remembered. */

import { describe, expect, it } from "vitest";

import type { MarkerScore } from "./api";
import {
  formatKept,
  keptCount,
  keptTitle,
  loadThreshold,
  normaliseThreshold,
  passesThreshold,
  saveThreshold,
  stepThreshold,
} from "./threshold";

/** A `Storage` stand-in, optionally refusing every access as a private window may. */
function memoryStorage(refuse = false) {
  const items = new Map<string, string>();
  return {
    items,
    getItem: (k: string) => {
      if (refuse) throw new Error("SecurityError");
      return items.get(k) ?? null;
    },
    setItem: (k: string, v: string) => {
      if (refuse) throw new Error("QuotaExceededError");
      items.set(k, v);
    },
  };
}

describe("passesThreshold", () => {
  it("keeps a marker at or above the threshold, and one without a confidence", () => {
    expect(passesThreshold(0.4, 0.4)).toBe(true);
    expect(passesThreshold(0.39, 0.4)).toBe(false);
    expect(passesThreshold(null, 1)).toBe(true);
    expect(passesThreshold(0, 0)).toBe(true);
  });
});

describe("normaliseThreshold and stepThreshold", () => {
  it("clamps to [0, 1] on the 0.01 grid without float residue", () => {
    expect(normaliseThreshold(0.57)).toBe(0.57);
    expect(normaliseThreshold(0.41000000000000003)).toBe(0.41);
    expect(normaliseThreshold(0.123)).toBe(0.12);
    expect(normaliseThreshold(-0.2)).toBe(0);
    expect(normaliseThreshold(3)).toBe(1);
    expect(normaliseThreshold(Number.NaN)).toBe(0);
  });

  it("steps and stops at the ends", () => {
    expect(stepThreshold(0.1, 0.05)).toBe(0.15);
    expect(stepThreshold(0.15, 0.05)).toBe(0.2);
    expect(stepThreshold(0.98, 0.05)).toBe(1);
    expect(stepThreshold(0.02, -0.05)).toBe(0);
  });
});

describe("keptCount and formatKept", () => {
  const scores: MarkerScore[] = [
    [0.2, true],
    [0.5, true],
    [0.9, false],
    [null, true],
  ];

  it("counts the kept markers and the decoded ones among them", () => {
    expect(keptCount(scores, 0)).toEqual({ kept: 4, keptDecoded: 3, total: 4 });
    expect(keptCount(scores, 0.5)).toEqual({ kept: 3, keptDecoded: 2, total: 4 });
    expect(keptCount(scores, 1)).toEqual({ kept: 1, keptDecoded: 1, total: 4 });
    expect(keptCount([], 0.5)).toEqual({ kept: 0, keptDecoded: 0, total: 0 });
    expect(keptCount(null, 0.5)).toBeNull();
    expect(keptCount(undefined, 0.5)).toBeNull();
  });

  it("says how many the threshold cut, else the decoded ones when not all are", () => {
    expect(formatKept({ kept: 63, keptDecoded: 63, total: 63 })).toBe("63");
    expect(formatKept({ kept: 63, keptDecoded: 60, total: 63 })).toBe("63 / 60 id");
    expect(formatKept({ kept: 56, keptDecoded: 56, total: 63 })).toBe("56 of 63");
    expect(formatKept({ kept: 56, keptDecoded: 50, total: 63 })).toBe("56 of 63");
    expect(formatKept(null)).toBe("–");
    expect(keptTitle({ kept: 56, keptDecoded: 50, total: 63 })).toBe("56 of 63 markers kept, 50 of them decoded");
    expect(keptTitle(null)).toBe("No output for this tile");
  });
});

describe("loadThreshold and saveThreshold", () => {
  it("round-trips through storage, normalised", () => {
    const storage = memoryStorage();
    expect(loadThreshold(storage)).toBe(0);
    saveThreshold(0.35, storage);
    expect(loadThreshold(storage)).toBe(0.35);
    saveThreshold(7, storage);
    expect(loadThreshold(storage)).toBe(1);
  });

  it("falls back to 0 on garbage or a storage that refuses, and never throws", () => {
    const storage = memoryStorage();
    storage.items.set("ringgrid-review:min-confidence", "not a number");
    expect(loadThreshold(storage)).toBe(0);
    const refusing = memoryStorage(true);
    expect(() => saveThreshold(0.5, refusing)).not.toThrow();
    expect(loadThreshold(refusing)).toBe(0);
    expect(loadThreshold(null)).toBe(0);
  });
});
