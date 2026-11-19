/**
 * The confidence threshold: a view filter that shows what cutting the detections at a minimum
 * confidence would remove. It is the reviewer's view state, kept per viewer in `localStorage`,
 * never in the feedback file.
 *
 * One rule decides what a threshold keeps (`passesThreshold`); the server applies the same rule
 * before it compares two runs (`passes_confidence` in tools/review/server.py).
 */

import type { MarkerScore } from "./api";

/** Steps of the slider and the field: hundredths. */
export const THRESHOLD_STEP = 0.01;
const STEPS_PER_UNIT = 100;
/** Step of the `[` and `]` keys. */
export const THRESHOLD_KEY_STEP = 0.05;

const STORAGE_KEY = "ringgrid-review:min-confidence";

/**
 * Whether a marker is kept at `threshold`: its confidence is at least the threshold, or it has
 * none (there is nothing to cut on). At 0 every marker is kept.
 */
export function passesThreshold(confidence: number | null, threshold: number): boolean {
  return confidence === null || confidence >= threshold;
}

/** A threshold in [0, 1] on the 0.01 grid; anything that is not a finite number is 0. */
export function normaliseThreshold(value: number): number {
  if (!Number.isFinite(value)) return 0;
  // Divided, not multiplied by the step, so 0.57 is 0.57 and not 0.5700000000000001.
  return Math.round(Math.min(1, Math.max(0, value)) * STEPS_PER_UNIT) / STEPS_PER_UNIT;
}

/** `threshold` moved by `delta`, normalised. */
export const stepThreshold = (threshold: number, delta: number) => normaliseThreshold(threshold + delta);

export interface KeptCount {
  /** Markers kept at the threshold, and how many of those are decoded. */
  kept: number;
  keptDecoded: number;
  /** All markers of the run on the tile. */
  total: number;
}

/** What a run's markers on one tile come to at `threshold`, or null when it has no output there. */
export function keptCount(scores: readonly MarkerScore[] | null | undefined, threshold: number): KeptCount | null {
  if (!scores) return null;
  let kept = 0;
  let keptDecoded = 0;
  for (const [confidence, decoded] of scores) {
    if (!passesThreshold(confidence, threshold)) continue;
    kept += 1;
    if (decoded) keptDecoded += 1;
  }
  return { kept, keptDecoded, total: scores.length };
}

/**
 * A tile-list cell, short enough for the narrow list: when the threshold cuts nothing, the
 * markers and the decoded ones when not all are ("63", "63 / 60 id"); when it cuts some, what
 * it keeps of the total ("56 of 63"), the decoded count moving to `keptTitle`.
 */
export function formatKept(c: KeptCount | null): string {
  if (!c) return "–";
  if (c.kept !== c.total) return `${c.kept} of ${c.total}`;
  return c.keptDecoded === c.kept ? `${c.kept}` : `${c.kept} / ${c.keptDecoded} id`;
}

/** The cell's tooltip: the whole count in words. */
export function keptTitle(c: KeptCount | null): string {
  if (!c) return "No output for this tile";
  return `${c.kept} of ${c.total} markers kept, ${c.keptDecoded} of them decoded`;
}

/** The stored threshold, or 0 when there is none or storage cannot be read. */
export function loadThreshold(storage: Pick<Storage, "getItem"> | null = safeLocalStorage()): number {
  try {
    const raw = storage?.getItem(STORAGE_KEY);
    return raw == null ? 0 : normaliseThreshold(Number(raw));
  } catch {
    return 0;
  }
}

/** Remembers the threshold for this viewer; a storage that refuses it is ignored. */
export function saveThreshold(threshold: number, storage: Pick<Storage, "setItem"> | null = safeLocalStorage()): void {
  try {
    storage?.setItem(STORAGE_KEY, String(normaliseThreshold(threshold)));
  } catch {
    // Private window or blocked site data: the threshold just is not remembered.
  }
}

/** `localStorage`, or null where reading the accessor itself throws. */
function safeLocalStorage(): Storage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}
