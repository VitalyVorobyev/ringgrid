/**
 * The review overlay as data: which markers of which runs are drawn, in what colour and dash,
 * with what label, dimmed or hidden. Pure, so the comparison and flag rules are tested without
 * a stage; `ReviewStage` only hands the result to the stage2d layers.
 *
 * Colour follows the vitavision overlay grammar:
 * - one run: the `feature` role (outer ring) and `model` (inner ring);
 * - two runs compared: run A and B in the first two categorical series colours, B dashed;
 *   a pair whose id or position differs in a verdict colour (the app compared it against the
 *   declared tolerances), pairs that agree dimmed;
 * - confidence (one run): viridis, a magnitude, with its legend;
 * - a reviewer's flag is a verdict and wins: defect for a wrong id or a false positive, warn for a
 *   wrong centre, normal for confirmed. Every class also has a label, so colour is never the
 *   only cue.
 */

import { colormapRgb } from "@vitavision/charts";

import type { Comparison, Detection, FlagKind, Marker, MarkerFlag, MatchClass, Pin } from "./api";

export type Side = "A" | "B";

/** How a marker relates to the other run, in compare mode. */
export type PairClass = MatchClass | "only";

export interface OverlayOptions {
  /** Show run B beside run A and classify the pairs. */
  compare: boolean;
  /** In compare mode, hide pairs that agree (flagged and selected markers stay). */
  differencesOnly: boolean;
  /** Colour markers by confidence (one run only). */
  confidence: boolean;
  /** Label markers with their id. */
  ids: boolean;
  /** Label markers with their grid coordinate. */
  grid: boolean;
}

export interface MarkerView {
  /** Unique across both runs: `A:<index>` or `B:<index>`. */
  key: string;
  side: Side;
  run: string;
  marker: Marker;
  /** Compare mode only. */
  cls: PairClass | null;
  /** The paired marker of the other run, when there is one. */
  other: Marker | null;
  flag: MarkerFlag | undefined;
  /** CSS colour, or undefined for the role colours. */
  colour: string | undefined;
  dashed: boolean;
  dimmed: boolean;
  label: string | undefined;
  /** Outer radius in image px: the fitted ellipse's semi-major axis, or the run's hint. */
  radius: number;
}

export const markerKey = (side: Side, index: number) => `${side}:${index}`;
export const pinKey = (uid: string) => `pin:${uid}`;

export const FLAG_LABEL: Record<FlagKind, string> = {
  wrong_id: "wrong id",
  wrong_center: "wrong centre",
  false_positive: "false positive",
  ok_confirmed: "ok",
};

export const FLAG_COLOUR: Record<FlagKind, string> = {
  wrong_id: "var(--defect)",
  false_positive: "var(--defect)",
  wrong_center: "var(--warn)",
  ok_confirmed: "var(--normal)",
};

export const CLASS_COLOUR: Record<"id_differs" | "position_differs", string> = {
  id_differs: "var(--defect)",
  position_differs: "var(--warn)",
};

export const RUN_COLOUR: Record<Side, string> = { A: "var(--series-1)", B: "var(--series-2)" };

/** Steps of the confidence colour map: markers sharing a colour share a batched path. */
const CONFIDENCE_STEPS = 16;

/** Viridis at `t` in [0, 1], quantised to `CONFIDENCE_STEPS`. */
export function confidenceColour(t: number): string {
  const q = Math.round(Math.min(1, Math.max(0, t)) * (CONFIDENCE_STEPS - 1)) / (CONFIDENCE_STEPS - 1);
  const [r, g, b] = colormapRgb("viridis", q);
  return `rgb(${Math.round(r)} ${Math.round(g)} ${Math.round(b)})`;
}

const fmtId = (id: number | null) => (id === null ? "?" : String(id));

export interface OverlayInput {
  a: { run: string; detection: Detection } | null;
  b: { run: string; detection: Detection } | null;
  comparison: Comparison | null;
  flags: readonly MarkerFlag[];
  /** The selected key (`markerKey` or `pinKey`), which is never dimmed or hidden. */
  selected: string | null;
  options: OverlayOptions;
}

/** Every marker to draw, in A-then-B order. */
export function markerViews({ a, b, comparison, flags, selected, options }: OverlayInput): MarkerView[] {
  const comparing = options.compare && a !== null && b !== null && comparison !== null;
  const pairOf: Record<Side, Map<number, Comparison["pairs"][number]>> = { A: new Map(), B: new Map() };
  if (comparing) {
    for (const p of comparison.pairs) {
      pairOf.A.set(p.a, p);
      pairOf.B.set(p.b, p);
    }
  }
  const byIndex = (d: Detection | undefined) => new Map((d?.markers ?? []).map((m) => [m.index, m]));
  const markersOf: Record<Side, Map<number, Marker>> = { A: byIndex(a?.detection), B: byIndex(b?.detection) };

  const views: MarkerView[] = [];
  const sides: [Side, OverlayInput["a"]][] = [["A", a]];
  if (comparing) sides.push(["B", b]);
  for (const [side, entry] of sides) {
    if (!entry) continue;
    const otherSide: Side = side === "A" ? "B" : "A";
    for (const marker of entry.detection.markers) {
      const key = markerKey(side, marker.index);
      const flag = flags.find((f) => f.run === entry.run && f.marker.index === marker.index);
      const pair = comparing ? pairOf[side].get(marker.index) : undefined;
      const cls: PairClass | null = comparing ? (pair?.cls ?? "only") : null;
      const other = pair ? (markersOf[otherSide].get(side === "A" ? pair.b : pair.a) ?? null) : null;
      const keep = flag !== undefined || key === selected;
      if (comparing && options.differencesOnly && cls === "same" && !keep) continue;

      let colour: string | undefined;
      if (flag) colour = FLAG_COLOUR[flag.kind];
      else if (cls === "id_differs" || cls === "position_differs") colour = CLASS_COLOUR[cls];
      else if (comparing) colour = RUN_COLOUR[side];
      else if (options.confidence && marker.confidence !== null) colour = confidenceColour(marker.confidence);

      const parts: string[] = [];
      if (options.ids) {
        // A differing id is written once, on A's marker, as "A ≠ B".
        if (cls === "id_differs") {
          if (side === "A") parts.push(`${fmtId(marker.id)} ≠ ${other ? fmtId(other.id) : "?"}`);
        } else {
          parts.push(fmtId(marker.id));
        }
      }
      if (options.grid && marker.grid_coord) parts.push(`(${marker.grid_coord[0]},${marker.grid_coord[1]})`);
      if (cls === "only") parts.push(`only ${side}`);
      if (flag) parts.push(FLAG_LABEL[flag.kind] + (flag.kind === "wrong_id" && flag.correct_id != null ? ` → ${flag.correct_id}` : ""));

      const outer = marker.ellipse_outer;
      views.push({
        key,
        side,
        run: entry.run,
        marker,
        cls,
        other,
        flag,
        colour,
        dashed: comparing && side === "B",
        dimmed: cls === "same" && !keep,
        label: parts.length > 0 ? parts.join(" · ") : undefined,
        radius: outer ? Math.max(outer.a, outer.b) : entry.detection.radius_hint,
      });
    }
  }
  return views;
}

/** The text drawn beside a pin. */
export function pinLabel(pin: Pin): string {
  if (pin.type === "missed_marker") return pin.true_id != null ? `missed ${pin.true_id}` : "missed";
  const text = (pin.text ?? "").trim();
  return text === "" ? "note" : text.length > 40 ? `${text.slice(0, 39)}…` : text;
}

/** The comparison counts in words, for the legend. */
export function comparisonSummary(c: Comparison): string {
  const n = c.counts;
  return `same ${n.same} · id ≠ ${n.id_differs} · position ≠ ${n.position_differs} · only A ${n.only_a} · only B ${n.only_b}`;
}
