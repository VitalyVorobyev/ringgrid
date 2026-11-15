/**
 * The edits produce exactly the records the server validates (`validate_tile_record`) and the
 * earlier page wrote, so existing feedback files keep reading back.
 */

import { describe, expect, it } from "vitest";

import { FEEDBACK_SCHEMA, type Feedback } from "./api";
import {
  dirtyKey,
  feedbackReducer,
  flagFor,
  isReviewed,
  issueCount,
  newPin,
  savePayload,
  tileRecord,
  type FeedbackAction,
} from "./feedback";

const empty = (): Feedback => ({ schema: FEEDBACK_SCHEMA, dataset: "d", general_remarks: [], tiles: {} });
const seen = { tile: "t_00", now: "2026-11-15T10:00:00Z", runs: ["a"] };
const marker = { index: 3, id: 17, center: [10.5, 20.25] as [number, number] };
const apply = (fb: Feedback, ...actions: FeedbackAction[]) => actions.reduce(feedbackReducer, fb);

describe("feedbackReducer", () => {
  it("toggles a verdict and clears reviewed_at when it is cleared", () => {
    let fb = apply(empty(), { type: "verdict", verdict: "ok", ...seen });
    expect(tileRecord(fb, "t_00")).toMatchObject({ verdict: "ok", reviewed_at: seen.now, runs_seen: ["a"] });
    expect(isReviewed(fb, "t_00")).toBe(true);
    fb = apply(fb, { type: "verdict", verdict: "has_issues", ...seen, runs: ["a", "b"] });
    expect(fb.tiles["t_00"]?.verdict).toBe("has_issues");
    expect(fb.tiles["t_00"]?.runs_seen).toEqual(["a", "b"]);
    fb = apply(fb, { type: "verdict", verdict: "has_issues", ...seen });
    expect(fb.tiles["t_00"]).toMatchObject({ verdict: null, reviewed_at: null });
    expect(isReviewed(fb, "t_00")).toBe(false);
  });

  it("flags a marker with a snapshot of it, re-flags in place and drops correct_id off wrong_id", () => {
    let fb = apply(empty(), { type: "flag", run: "a", marker, kind: "wrong_id", ...seen });
    const flags = fb.tiles["t_00"]?.marker_flags;
    expect(flags).toEqual([{ run: "a", kind: "wrong_id", marker: { index: 3, id: 17, center: [10.5, 20.25] }, at: seen.now }]);
    fb = apply(fb, { type: "flagField", tile: "t_00", run: "a", index: 3, field: "correct_id", value: 18 });
    expect(flagFor(fb, "t_00", "a", 3)?.correct_id).toBe(18);
    fb = apply(fb, { type: "flagField", tile: "t_00", run: "a", index: 3, field: "remark", value: "neighbour's id" });
    fb = apply(fb, { type: "flag", run: "a", marker, kind: "wrong_center", ...seen });
    expect(fb.tiles["t_00"]?.marker_flags).toHaveLength(1);
    expect(flagFor(fb, "t_00", "a", 3)).toMatchObject({ kind: "wrong_center", remark: "neighbour's id" });
    expect(flagFor(fb, "t_00", "a", 3)).not.toHaveProperty("correct_id");
    // The same marker index in another run is another flag.
    fb = apply(fb, { type: "flag", run: "b", marker, kind: "ok_confirmed", ...seen });
    expect(fb.tiles["t_00"]?.marker_flags).toHaveLength(2);
    expect(issueCount(fb, "t_00")).toBe(1);
    fb = apply(fb, { type: "clearFlag", run: "a", index: 3, ...seen });
    expect(flagFor(fb, "t_00", "a", 3)).toBeUndefined();
    expect(flagFor(fb, "t_00", "b", 3)?.kind).toBe("ok_confirmed");
  });

  it("clearing the correct id removes the key rather than writing null", () => {
    let fb = apply(empty(), { type: "flag", run: "a", marker, kind: "wrong_id", ...seen });
    fb = apply(fb, { type: "flagField", tile: "t_00", run: "a", index: 3, field: "correct_id", value: 5 });
    fb = apply(fb, { type: "flagField", tile: "t_00", run: "a", index: 3, field: "correct_id", value: null });
    expect(flagFor(fb, "t_00", "a", 3)).not.toHaveProperty("correct_id");
  });

  it("adds, edits and deletes pins of both kinds with only their own fields", () => {
    const missed = newPin("missed_marker", 1.234, 5.678, seen.now, "u1");
    const note = newPin("note", 9, 9, seen.now, "u2");
    expect(missed).toEqual({ uid: "u1", type: "missed_marker", x: 1.23, y: 5.68, at: seen.now });
    let fb = apply(empty(), { type: "addPin", pin: missed, ...seen }, { type: "addPin", pin: note, ...seen });
    fb = apply(
      fb,
      { type: "updatePin", tile: "t_00", uid: "u1", patch: { true_id: 9, remark: "under glare" } },
      { type: "updatePin", tile: "t_00", uid: "u2", patch: { text: "blurred corner", true_id: 4 } },
    );
    const pins = tileRecord(fb, "t_00").pins;
    expect(pins[0]).toEqual({ ...missed, true_id: 9, remark: "under glare" });
    // A note never takes a true id (the server rejects one).
    expect(pins[1]).toEqual({ ...note, text: "blurred corner" });
    expect(issueCount(fb, "t_00")).toBe(2);
    fb = apply(fb, { type: "deletePin", uid: "u1", ...seen });
    expect(tileRecord(fb, "t_00").pins.map((p) => p.uid)).toEqual(["u2"]);
  });

  it("keeps the other tiles and the original object untouched", () => {
    const before = apply(empty(), { type: "remark", text: "x", ...seen, tile: "other" });
    const snapshot = JSON.stringify(before);
    const after = apply(before, { type: "remark", text: "y", ...seen });
    expect(JSON.stringify(before)).toBe(snapshot);
    expect(after.tiles["other"]).toBe(before.tiles["other"]);
    expect(after.tiles["t_00"]?.remark).toBe("y");
  });

  it("appends general remarks and saves only what is dirty", () => {
    let fb = apply(empty(), { type: "generalRemark", text: "lighting changes", now: seen.now });
    fb = apply(fb, { type: "verdict", verdict: "ok", ...seen });
    expect(fb.general_remarks).toEqual([{ at: seen.now, text: "lighting changes" }]);
    expect(dirtyKey({ type: "generalRemark", text: "", now: "" })).toBe("");
    expect(savePayload(fb, ["t_00"])).toEqual({ tiles: { t_00: fb.tiles["t_00"] } });
    expect(savePayload(fb, [""])).toEqual({ general_remarks: fb.general_remarks });
    // A tile with no record (all cleared) is sent as an empty record, which the server drops.
    expect(savePayload(fb, ["gone"])).toEqual({ tiles: { gone: { verdict: null, remark: "", marker_flags: [], pins: [] } } });
  });

  it("reads records written without the optional lists", () => {
    const fb: Feedback = { ...empty(), tiles: { t_00: { verdict: "ok" } } };
    expect(tileRecord(fb, "t_00")).toEqual({ verdict: "ok", remark: "", reviewed_at: null, runs_seen: [], marker_flags: [], pins: [] });
    expect(issueCount(fb, "t_00")).toBe(0);
    const next = apply(fb, { type: "flag", run: "a", marker, kind: "false_positive", ...seen });
    expect(next.tiles["t_00"]?.verdict).toBe("ok");
  });
});
