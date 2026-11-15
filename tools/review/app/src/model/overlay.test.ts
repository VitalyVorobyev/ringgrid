/** The run comparison as it is drawn: classes, colours, dashes, labels, dimming, hiding. */

import { describe, expect, it } from "vitest";

import type { Comparison, Detection, Marker, MarkerFlag } from "./api";
import { CLASS_COLOUR, confidenceColour, FLAG_COLOUR, markerViews, pinLabel, RUN_COLOUR, type OverlayOptions } from "./overlay";

function marker(index: number, id: number | null, x: number, extra: Partial<Marker> = {}): Marker {
  return {
    index,
    id,
    confidence: 0.5,
    center: [x, 10],
    grid_coord: [index, 0],
    board_xy_mm: null,
    ellipse_outer: { cx: x, cy: 10, a: 8, b: 6, angle: 0 },
    ellipse_inner: null,
    center_source: null,
    details: {},
    ...extra,
  };
}

function detection(run: string, markers: Marker[]): Detection {
  return {
    tile: "t",
    run,
    image_size: [100, 100],
    frame: null,
    markers,
    n_markers: markers.length,
    n_decoded: 0,
    radius_hint: 12,
    warnings: [],
  };
}

// A: 0 same as B0, 1 id differs from B1, 2 moved vs B2, 3 only in A. B: 3 only in B.
const A = detection("a", [marker(0, 1, 10), marker(1, 2, 30), marker(2, 3, 50), marker(3, 4, 70, { ellipse_outer: null })]);
const B = detection("b", [marker(0, 1, 10), marker(1, 9, 30), marker(2, 3, 52), marker(3, 5, 90)]);
const CMP: Comparison = {
  pairs: [
    { a: 0, b: 0, dist: 0, cls: "same" },
    { a: 1, b: 1, dist: 0, cls: "id_differs" },
    { a: 2, b: 2, dist: 2, cls: "position_differs" },
  ],
  only_a: [3],
  only_b: [3],
  counts: { same: 1, id_differs: 1, position_differs: 1, only_a: 1, only_b: 1 },
};
const OPTS: OverlayOptions = { compare: true, differencesOnly: false, confidence: false, ids: true, grid: false };
const input = (o: Partial<OverlayOptions> = {}, flags: MarkerFlag[] = [], selected: string | null = null) => ({
  a: { run: "a", detection: A },
  b: { run: "b", detection: B },
  comparison: CMP,
  flags,
  selected,
  options: { ...OPTS, ...o },
});

describe("markerViews", () => {
  it("draws one run in the role colours with id labels, ignoring B and the comparison", () => {
    const views = markerViews(input({ compare: false }));
    expect(views.map((v) => v.key)).toEqual(["A:0", "A:1", "A:2", "A:3"]);
    expect(views.every((v) => v.colour === undefined && !v.dashed && !v.dimmed && v.cls === null)).toBe(true);
    expect(views.map((v) => v.label)).toEqual(["1", "2", "3", "4"]);
    // Without a fitted ellipse the radius is the run's hint.
    expect(views[3]?.radius).toBe(12);
    expect(views[0]?.radius).toBe(8);
  });

  it("classifies both runs in compare mode: differences in verdict colours, B dashed, agreement dimmed", () => {
    const views = markerViews(input());
    const by = Object.fromEntries(views.map((v) => [v.key, v]));
    expect(views).toHaveLength(8);
    expect(by["A:0"]).toMatchObject({ cls: "same", colour: RUN_COLOUR.A, dimmed: true, dashed: false });
    expect(by["B:0"]).toMatchObject({ cls: "same", colour: RUN_COLOUR.B, dimmed: true, dashed: true });
    expect(by["A:1"]).toMatchObject({ cls: "id_differs", colour: CLASS_COLOUR.id_differs, label: "2 ≠ 9" });
    expect(by["A:1"]?.other?.id).toBe(9);
    // The differing id is written once, on A.
    expect(by["B:1"]?.label).toBeUndefined();
    expect(by["A:2"]).toMatchObject({ cls: "position_differs", colour: CLASS_COLOUR.position_differs, dimmed: false });
    expect(by["A:3"]).toMatchObject({ cls: "only", label: "4 · only A", other: null });
    expect(by["B:3"]).toMatchObject({ cls: "only", label: "5 · only B" });
  });

  it("hides agreeing pairs with differences only, except flagged and selected markers", () => {
    const flag: MarkerFlag = { run: "b", kind: "wrong_center", marker: { index: 0, id: 1, center: [10, 10] } };
    expect(markerViews(input({ differencesOnly: true })).map((v) => v.key)).toEqual(["A:1", "A:2", "A:3", "B:1", "B:2", "B:3"]);
    const kept = markerViews(input({ differencesOnly: true }, [flag], "A:0"));
    expect(kept.map((v) => v.key)).toEqual(["A:0", "A:1", "A:2", "A:3", "B:0", "B:1", "B:2", "B:3"]);
    const b0 = kept.find((v) => v.key === "B:0");
    expect(b0).toMatchObject({ colour: FLAG_COLOUR.wrong_center, dimmed: false, label: "1 · wrong centre" });
  });

  it("lets a flag win over every other colour and writes the corrected id", () => {
    const flag: MarkerFlag = { run: "a", kind: "wrong_id", marker: { index: 1, id: 2, center: [30, 10] }, correct_id: 7 };
    const v = markerViews(input({}, [flag])).find((x) => x.key === "A:1");
    expect(v?.colour).toBe(FLAG_COLOUR.wrong_id);
    expect(v?.label).toBe("2 ≠ 9 · wrong id → 7");
  });

  it("colours by confidence only for a single run, and adds grid labels on request", () => {
    const single = markerViews(input({ compare: false, confidence: true, grid: true }));
    expect(single[0]?.colour).toBe(confidenceColour(0.5));
    expect(single[0]?.label).toBe("1 · (0,0)");
    expect(markerViews(input({ confidence: true }))[1]?.colour).toBe(CLASS_COLOUR.id_differs);
  });

  it("falls back to one run when the comparison or run B is missing", () => {
    expect(markerViews({ ...input(), comparison: null }).every((v) => v.side === "A" && v.cls === null)).toBe(true);
    expect(markerViews({ ...input(), b: null })).toHaveLength(4);
  });
});

describe("confidenceColour and pinLabel", () => {
  it("quantises viridis and clamps", () => {
    expect(confidenceColour(0.5)).toBe(confidenceColour(0.52));
    expect(confidenceColour(-1)).toBe(confidenceColour(0));
    expect(confidenceColour(2)).toBe(confidenceColour(1));
    expect(confidenceColour(0)).not.toBe(confidenceColour(1));
    expect(confidenceColour(1)).toMatch(/^rgb\(\d+ \d+ \d+\)$/);
  });

  it("names pins by their content", () => {
    expect(pinLabel({ uid: "u", type: "missed_marker", x: 0, y: 0 })).toBe("missed");
    expect(pinLabel({ uid: "u", type: "missed_marker", x: 0, y: 0, true_id: 12 })).toBe("missed 12");
    expect(pinLabel({ uid: "u", type: "note", x: 0, y: 0, text: "  " })).toBe("note");
    expect(pinLabel({ uid: "u", type: "note", x: 0, y: 0, text: "x".repeat(50) })).toHaveLength(40);
  });
});
