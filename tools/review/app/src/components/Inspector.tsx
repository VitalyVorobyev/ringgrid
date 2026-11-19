/**
 * The inspector: the selected marker or pin (its fields, its flag, the flag's details), the
 * tile's verdict and remark, and the general remarks. Every control is a `@vitavision/ui`
 * primitive; every edit is a `FeedbackAction`.
 */

import { Button, Field, Input, NumberInput, Panel, Textarea, ToggleChip } from "@vitavision/ui";
import { useState } from "react";

import type { Feedback, FlagKind, Pin, Verdict } from "../model/api";
import { tileRecord } from "../model/feedback";
import { FLAG_LABEL, type MarkerView } from "../model/overlay";

const FLAG_KEYS: Record<FlagKind, string> = { wrong_id: "W", wrong_center: "C", false_positive: "F", ok_confirmed: "O" };
const VERDICTS: { value: Verdict; label: string; key: string }[] = [
  { value: "ok", label: "ok", key: "1" },
  { value: "has_issues", label: "has issues", key: "2" },
  { value: "unusable_image", label: "unusable image", key: "3" },
];

const fmt = (v: number, digits = 2) => v.toFixed(digits);
const deg = (rad: number) => `${((rad * 180) / Math.PI).toFixed(1)}°`;

function Facts({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-fg-muted">{k}</dt>
          <dd className="font-mono break-all text-fg">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export interface SelectionPanelProps {
  view: MarkerView | null;
  pin: Pin | null;
  comparing: boolean;
  onFlag: (kind: FlagKind) => void;
  onClearFlag: () => void;
  onCorrectId: (id: number | null) => void;
  onFlagRemark: (text: string) => void;
  onPinChange: (patch: { true_id?: number | null; remark?: string; text?: string }) => void;
  onDeletePin: () => void;
}

export function SelectionPanel(p: SelectionPanelProps) {
  if (p.pin) {
    const pin = p.pin;
    return (
      <Panel title={pin.type === "missed_marker" ? "Missed marker" : "Note"}>
        <div className="grid gap-2.5">
          <span className="font-mono text-xs text-fg-muted">
            at {fmt(pin.x, 1)}, {fmt(pin.y, 1)} px
          </span>
          {pin.type === "missed_marker" ? (
            <>
              <Field label="True id">
                <NumberInput
                  data-autofocus="pin"
                  min={0}
                  precision={0}
                  value={pin.true_id ?? null}
                  onValueChange={(v) => p.onPinChange({ true_id: Math.max(0, Math.round(v)) })}
                  onClear={() => p.onPinChange({ true_id: null })}
                />
              </Field>
              <Field label="Remark">
                <Input value={pin.remark ?? ""} onChange={(e) => p.onPinChange({ remark: e.target.value })} />
              </Field>
            </>
          ) : (
            <Field label="Note">
              <Textarea
                data-autofocus="pin"
                minRows={2}
                maxRows={8}
                value={pin.text ?? ""}
                onValueChange={(text) => p.onPinChange({ text })}
              />
            </Field>
          )}
          <Button size="sm" variant="danger" onClick={p.onDeletePin}>
            Delete pin
          </Button>
        </div>
      </Panel>
    );
  }
  const v = p.view;
  if (!v) {
    return (
      <Panel title="Selection">
        <p className="text-xs text-fg-muted">Click a marker or a pin. Shift-click the image to pin a missed marker.</p>
      </Panel>
    );
  }
  const m = v.marker;
  const rows: [string, string][] = [
    ["run", `${v.run} (${v.side})`],
    ["id", m.id === null ? "–" : String(m.id)],
    ["confidence", m.confidence === null ? "–" : `${fmt(m.confidence, 3)}${v.cut ? " (below the threshold)" : ""}`],
    ["centre", `${fmt(m.center[0])}, ${fmt(m.center[1])}`],
    ["grid", m.grid_coord ? `(${m.grid_coord.join(", ")})` : "–"],
    ["board mm", m.board_xy_mm ? m.board_xy_mm.map((x) => fmt(x)).join(", ") : "–"],
    ["centre source", m.center_source ?? "–"],
  ];
  if (m.ellipse_outer) rows.push(["outer a, b, θ", `${fmt(m.ellipse_outer.a)}, ${fmt(m.ellipse_outer.b)}, ${deg(m.ellipse_outer.angle)}`]);
  if (m.ellipse_inner) rows.push(["inner a, b", `${fmt(m.ellipse_inner.a)}, ${fmt(m.ellipse_inner.b)}`]);
  if (p.comparing && v.cls !== null) {
    rows.push(["other run", v.cls === "only" ? "no match (only here)" : `${v.cls.replace("_", " ")}; id ${v.other?.id ?? "?"}`]);
  }
  for (const [k, val] of Object.entries(m.details)) {
    rows.push([k, typeof val === "number" ? (Number.isInteger(val) ? String(val) : val.toFixed(4)) : String(val)]);
  }
  const flag = v.flag;
  return (
    <Panel title={`Marker ${m.id ?? "?"}`}>
      <div className="grid gap-3">
        <Facts rows={rows} />
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Flag">
          {(Object.keys(FLAG_LABEL) as FlagKind[]).map((kind) => (
            <ToggleChip
              key={kind}
              size="sm"
              checked={flag?.kind === kind}
              onCheckedChange={(on) => (on ? p.onFlag(kind) : p.onClearFlag())}
              title={`${FLAG_LABEL[kind]} (${FLAG_KEYS[kind]})`}
            >
              {FLAG_LABEL[kind]}
            </ToggleChip>
          ))}
        </div>
        {flag?.kind === "wrong_id" && (
          <Field label="Correct id">
            <NumberInput
              data-autofocus="correct-id"
              min={0}
              precision={0}
              value={flag.correct_id ?? null}
              onValueChange={(id) => p.onCorrectId(Math.max(0, Math.round(id)))}
              onClear={() => p.onCorrectId(null)}
            />
          </Field>
        )}
        {flag && (
          <Field label="Remark on this marker">
            <Input value={flag.remark ?? ""} onChange={(e) => p.onFlagRemark(e.target.value)} />
          </Field>
        )}
      </div>
    </Panel>
  );
}

export interface TilePanelProps {
  tile: string;
  feedback: Feedback;
  onVerdict: (v: Verdict) => void;
  onRemark: (text: string) => void;
}

export function TilePanel({ tile, feedback, onVerdict, onRemark }: TilePanelProps) {
  const rec = tileRecord(feedback, tile);
  const flagCounts = (Object.keys(FLAG_LABEL) as FlagKind[])
    .map((k) => [k, rec.marker_flags.filter((f) => f.kind === k).length] as const)
    .filter(([, n]) => n > 0)
    .map(([k, n]) => `${n} ${FLAG_LABEL[k]}`);
  const missed = rec.pins.filter((p) => p.type === "missed_marker").length;
  const notes = rec.pins.filter((p) => p.type === "note").length;
  const summary = [...flagCounts, missed ? `${missed} missed` : "", notes ? `${notes} notes` : ""].filter(Boolean).join(" · ");
  return (
    <Panel title={`Tile ${tile}`}>
      <div className="grid gap-2.5">
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Verdict">
          {VERDICTS.map((v) => (
            <ToggleChip
              key={v.value}
              size="sm"
              checked={rec.verdict === v.value}
              onCheckedChange={() => onVerdict(v.value)}
              title={`${v.label} (${v.key})`}
            >
              {v.label}
            </ToggleChip>
          ))}
        </div>
        <Field label="Remark">
          <Textarea
            id="tile-remark"
            minRows={3}
            maxRows={10}
            value={rec.remark ?? ""}
            onValueChange={onRemark}
            placeholder="What is wrong on this tile ( / )"
          />
        </Field>
        <span className="text-xs text-fg-muted">{summary || "No flags or pins."}</span>
      </div>
    </Panel>
  );
}

export function GeneralPanel({ feedback, onAdd }: { feedback: Feedback; onAdd: (text: string) => void }) {
  const [text, setText] = useState("");
  const add = () => {
    const t = text.trim();
    if (!t) return;
    onAdd(t);
    setText("");
  };
  return (
    <Panel title="General remarks">
      <div className="grid gap-2">
        <Textarea
          aria-label="General remark"
          minRows={2}
          maxRows={6}
          value={text}
          onValueChange={setText}
          placeholder="A remark not tied to a tile"
        />
        <Button size="sm" variant="secondary" onClick={add} disabled={!text.trim()}>
          Add remark
        </Button>
        <ul className="grid gap-2 text-xs">
          {[...feedback.general_remarks].reverse().map((r) => (
            <li key={`${r.at}-${r.text}`}>
              <span className="text-fg-muted">{r.at.replace("T", " ").slice(0, 16)}</span>
              <p className="whitespace-pre-wrap text-fg">{r.text}</p>
            </li>
          ))}
        </ul>
      </div>
    </Panel>
  );
}
