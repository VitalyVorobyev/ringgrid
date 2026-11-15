/**
 * The review server's JSON API (tools/review/server.py), typed. The shapes are the server's:
 * detections are normalised there (`normalise_detection`), the feedback file is
 * `ringgrid.review.v1` (`validate_feedback`).
 */

export const FEEDBACK_SCHEMA = "ringgrid.review.v1";

export type Verdict = "ok" | "has_issues" | "unusable_image";
export type FlagKind = "wrong_id" | "wrong_center" | "false_positive" | "ok_confirmed";
export type PinType = "missed_marker" | "note";

export interface Ellipse {
  cx: number;
  cy: number;
  a: number;
  b: number;
  /** Radians, clockwise on screen (image y down). */
  angle: number;
}

export interface Marker {
  index: number;
  id: number | null;
  confidence: number | null;
  /** Image pixels, the centre of pixel i at i (ringgrid's convention, and stage2d's). */
  center: [number, number];
  grid_coord: [number, number] | null;
  board_xy_mm: [number, number] | null;
  ellipse_outer: Ellipse | null;
  ellipse_inner: Ellipse | null;
  center_source: string | null;
  details: Record<string, string | number | boolean>;
}

export interface Detection {
  tile: string;
  run: string;
  image_size: [number, number] | null;
  frame: string | null;
  markers: Marker[];
  n_markers: number;
  n_decoded: number;
  radius_hint: number;
  warnings: string[];
}

export type MatchClass = "same" | "id_differs" | "position_differs";

export interface Comparison {
  pairs: { a: number; b: number; dist: number; cls: MatchClass }[];
  only_a: number[];
  only_b: number[];
  counts: Record<MatchClass | "only_a" | "only_b", number>;
}

export interface Tile {
  name: string;
  source: string;
  index: number | null;
  width: number;
  height: number;
  eval_skip: boolean;
}

export interface RunStatus {
  state: "running" | "done" | "failed" | "incomplete";
  done: number;
  total: number;
  errors: string[];
}

export interface Run {
  name: string;
  manifest: Record<string, unknown>;
  status: RunStatus;
  /** Per tile: `[markers, decoded]`, or null when the run has no output for it. */
  counts: Record<string, [number, number] | null>;
}

export interface MarkerFlag {
  run: string;
  kind: FlagKind;
  /** The marker as it was when flagged. */
  marker: { index: number; id: number | null; center: [number, number] };
  correct_id?: number | null;
  remark?: string | null;
  at?: string | null;
}

export interface MissedPin {
  uid: string;
  type: "missed_marker";
  x: number;
  y: number;
  true_id?: number | null;
  remark?: string | null;
  at?: string | null;
}

export interface NotePin {
  uid: string;
  type: "note";
  x: number;
  y: number;
  text?: string | null;
  at?: string | null;
}

export type Pin = MissedPin | NotePin;

export interface TileRecord {
  verdict: Verdict | null;
  remark?: string | null;
  reviewed_at?: string | null;
  runs_seen?: string[];
  marker_flags?: MarkerFlag[];
  pins?: Pin[];
}

export interface GeneralRemark {
  at: string;
  text: string;
}

export interface Feedback {
  schema: string;
  dataset: string;
  updated_at?: string;
  general_remarks: GeneralRemark[];
  tiles: Record<string, TileRecord>;
}

export interface ServerState {
  dataset: { name: string; split: number; tiles: Tile[] };
  runs: Run[];
  differing_tiles: string[];
  binaries: { label: string; path: string }[];
  defaults: { match_px: number; pos_px: number };
  feedback: Feedback;
}

/** An error the server answered with (`{ error }`), or a transport failure. */
export class ApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  const init: RequestInit =
    body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch (error) {
    throw new ApiError(error instanceof Error ? error.message : "network error", 0);
  }
  let data: unknown = null;
  try {
    data = await response.json();
  } catch {
    // A non-JSON body: reported by status below.
  }
  if (!response.ok) {
    const message = data !== null && typeof data === "object" && "error" in data ? String(data.error) : `HTTP ${response.status}`;
    throw new ApiError(message, response.status);
  }
  return data as T;
}

const enc = encodeURIComponent;

export const api = {
  state: () => request<ServerState>("/api/state"),
  detection: (run: string, tile: string) => request<Detection>(`/api/run/${enc(run)}/tile/${enc(tile)}`),
  compare: (tile: string, a: string, b: string, matchPx: number, posPx: number) =>
    request<Comparison>(`/api/compare/${enc(tile)}?a=${enc(a)}&b=${enc(b)}&match=${matchPx}&pos=${posPx}`),
  saveFeedback: (patch: { tiles?: Record<string, TileRecord>; general_remarks?: GeneralRemark[] }) =>
    request<{ ok: true; updated_at: string }>("/api/feedback", patch),
  createRun: (name: string, binary: string, args: string) => request<{ ok: true; run: string }>("/api/runs", { name, binary, args }),
};

/** URL of a tile's image. */
export const tileImageUrl = (tile: string) => `/api/tile/${enc(tile)}/image`;
