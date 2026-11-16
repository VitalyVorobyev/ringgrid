/**
 * The feedback document (`ringgrid.review.v1`) and every edit the reviewer can make to it, as a
 * pure reducer. The records it writes have exactly the shape the server validates and the
 * earlier page wrote, so a feedback file recorded with either reads back unchanged.
 */

import type { Feedback, FlagKind, Marker, MarkerFlag, Pin, TileRecord, Verdict } from "./api";

/** Edits that mark the tile as looked at carry the moment and the runs on screen. */
interface Seen {
  tile: string;
  /** ISO time of the edit. */
  now: string;
  /** The runs visible while editing, added to `runs_seen`. */
  runs: readonly string[];
}

export type FeedbackAction =
  | ({ type: "verdict"; verdict: Verdict } & Seen)
  | ({ type: "remark"; text: string } & Seen)
  | ({ type: "flag"; run: string; marker: Pick<Marker, "index" | "id" | "center">; kind: FlagKind } & Seen)
  | ({ type: "clearFlag"; run: string; index: number } & Seen)
  | { type: "flagField"; tile: string; run: string; index: number; field: "correct_id"; value: number | null }
  | { type: "flagField"; tile: string; run: string; index: number; field: "remark"; value: string }
  | ({ type: "addPin"; pin: Pin } & Seen)
  | ({ type: "deletePin"; uid: string } & Seen)
  | { type: "updatePin"; tile: string; uid: string; patch: { true_id?: number | null; remark?: string; text?: string } }
  | { type: "generalRemark"; text: string; now: string };

/** The key an action dirties for saving: its tile, or `""` for the general remarks. */
export function dirtyKey(action: FeedbackAction): string {
  return action.type === "generalRemark" ? "" : action.tile;
}

/** A tile's record with every list present (records on disk may omit them). */
export function tileRecord(feedback: Feedback, tile: string): Required<TileRecord> {
  const rec = feedback.tiles[tile];
  return {
    verdict: rec?.verdict ?? null,
    remark: rec?.remark ?? "",
    reviewed_at: rec?.reviewed_at ?? null,
    runs_seen: rec?.runs_seen ?? [],
    marker_flags: rec?.marker_flags ?? [],
    pins: rec?.pins ?? [],
  };
}

/** The flag on a marker of a run, if any. */
export function flagFor(feedback: Feedback, tile: string, run: string, index: number): MarkerFlag | undefined {
  return feedback.tiles[tile]?.marker_flags?.find((f) => f.run === run && f.marker.index === index);
}

function withTile(feedback: Feedback, tile: string, edit: (rec: Required<TileRecord>) => Required<TileRecord>): Feedback {
  return { ...feedback, tiles: { ...feedback.tiles, [tile]: edit(tileRecord(feedback, tile)) } };
}

function seen(rec: Required<TileRecord>, s: Seen, reviewed: boolean): Required<TileRecord> {
  return {
    ...rec,
    reviewed_at: reviewed ? s.now : rec.reviewed_at,
    runs_seen: [...new Set([...rec.runs_seen, ...s.runs])],
  };
}

const sameMarker = (f: MarkerFlag, run: string, index: number) => f.run === run && f.marker.index === index;

export function feedbackReducer(feedback: Feedback, action: FeedbackAction): Feedback {
  switch (action.type) {
    case "verdict":
      return withTile(feedback, action.tile, (rec) => {
        // Pressing the current verdict again clears it, and the tile counts as not reviewed.
        const verdict = rec.verdict === action.verdict ? null : action.verdict;
        const next = seen({ ...rec, verdict }, action, verdict !== null);
        return verdict === null ? { ...next, reviewed_at: null } : next;
      });
    case "remark":
      return withTile(feedback, action.tile, (rec) => seen({ ...rec, remark: action.text }, action, true));
    case "flag":
      return withTile(feedback, action.tile, (rec) => {
        const { run, marker, kind, now } = action;
        const existing = rec.marker_flags.find((f) => sameMarker(f, run, marker.index));
        let flag: MarkerFlag;
        if (existing) {
          flag = { ...existing, kind, at: now };
          if (kind !== "wrong_id") delete flag.correct_id;
        } else {
          flag = { run, kind, marker: { index: marker.index, id: marker.id, center: [marker.center[0], marker.center[1]] }, at: now };
        }
        const flags = existing ? rec.marker_flags.map((f) => (f === existing ? flag : f)) : [...rec.marker_flags, flag];
        return seen({ ...rec, marker_flags: flags }, action, true);
      });
    case "clearFlag":
      return withTile(feedback, action.tile, (rec) =>
        seen({ ...rec, marker_flags: rec.marker_flags.filter((f) => !sameMarker(f, action.run, action.index)) }, action, true),
      );
    case "flagField":
      return withTile(feedback, action.tile, (rec) => ({
        ...rec,
        marker_flags: rec.marker_flags.map((f) => {
          if (!sameMarker(f, action.run, action.index)) return f;
          const next = { ...f };
          if (action.field === "correct_id") {
            if (action.value === null) delete next.correct_id;
            else next.correct_id = action.value;
          } else {
            next.remark = action.value;
          }
          return next;
        }),
      }));
    case "addPin":
      return withTile(feedback, action.tile, (rec) => seen({ ...rec, pins: [...rec.pins, action.pin] }, action, true));
    case "deletePin":
      return withTile(feedback, action.tile, (rec) => seen({ ...rec, pins: rec.pins.filter((p) => p.uid !== action.uid) }, action, true));
    case "updatePin":
      return withTile(feedback, action.tile, (rec) => ({
        ...rec,
        pins: rec.pins.map((p): Pin => {
          if (p.uid !== action.uid) return p;
          const { true_id, remark, text } = action.patch;
          if (p.type === "missed_marker") {
            return { ...p, ...(true_id !== undefined ? { true_id } : {}), ...(remark !== undefined ? { remark } : {}) };
          }
          return { ...p, ...(text !== undefined ? { text } : {}) };
        }),
      }));
    case "generalRemark":
      return { ...feedback, general_remarks: [...feedback.general_remarks, { at: action.now, text: action.text }] };
  }
}

/** A new pin at image point (x, y), rounded to 1/100 px as the server stores it. */
export function newPin(type: Pin["type"], x: number, y: number, now: string, uid: string): Pin {
  const round = (v: number) => Math.round(v * 100) / 100;
  return { uid, type, x: round(x), y: round(y), at: now };
}

/** A pin id unlikely to collide within a tile (the server checks uniqueness). */
export function pinUid(): string {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
}

/** The body of a save: the dirty tiles' records and, when dirty, the general remarks. */
export function savePayload(feedback: Feedback, keys: Iterable<string>) {
  const tiles: Record<string, TileRecord> = {};
  let general = false;
  for (const key of keys) {
    if (key === "") general = true;
    else tiles[key] = feedback.tiles[key] ?? { verdict: null, remark: "", marker_flags: [], pins: [] };
  }
  return {
    ...(Object.keys(tiles).length > 0 ? { tiles } : {}),
    ...(general ? { general_remarks: feedback.general_remarks } : {}),
  };
}

/** Whether a tile has a verdict: what counts as reviewed in the list and the progress. */
export const isReviewed = (feedback: Feedback, tile: string) => Boolean(feedback.tiles[tile]?.verdict);

/** Flags that say something is wrong, plus pins: the count shown in the tile list. */
export function issueCount(feedback: Feedback, tile: string): number {
  const rec = feedback.tiles[tile];
  if (!rec) return 0;
  return (rec.marker_flags ?? []).filter((f) => f.kind !== "ok_confirmed").length + (rec.pins ?? []).length;
}
