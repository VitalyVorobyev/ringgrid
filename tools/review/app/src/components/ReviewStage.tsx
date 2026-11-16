/**
 * The tile image and its overlay on the stage2d `ImageStage`: fitted rings (`EllipseSet`),
 * marker centres (`PointSet`, the pick targets), the reviewer's pins (`PointSet`), with the
 * stage's own toolbar, readout and banner. Everything drawn comes from the pure view model
 * (`markerViews`); this component only maps it onto layers.
 */

import { Popover, Slider, Button } from "@vitavision/ui";
import {
  EllipseSet,
  ImageStage,
  PointSet,
  STAGE_HIT_PRIORITY,
  StageButton,
  StageLayersMenu,
  StageReadout,
  StageToolbar,
  StageToolbarDivider,
  type EllipseSetItem,
  type MarkerShape,
  type PointSetItem,
  type StageHandle,
  type StageView,
  type StageViewChange,
  type Point,
} from "@vitavision/stage2d";
import { Diamond, SlidersHorizontal, StickyNote } from "lucide-react";
import { useMemo, useState, type Ref } from "react";

import type { Pin, PinType, Tile } from "../model/api";
import { tileImageUrl } from "../model/api";
import { pinKey, pinLabel, type MarkerView } from "../model/overlay";
import { ImageLayer } from "../shims/stage2d";

export interface LayerVisibility {
  centres: boolean;
  ellipses: boolean;
  ids: boolean;
  grid: boolean;
  confidence: boolean;
}

export const LAYER_NAMES: Record<keyof LayerVisibility, { label: string; shortcut?: string }> = {
  centres: { label: "Centres" },
  ellipses: { label: "Ellipses", shortcut: "E" },
  ids: { label: "Ids", shortcut: "I" },
  grid: { label: "Grid coordinates", shortcut: "G" },
  confidence: { label: "Colour by confidence", shortcut: "P" },
};

/** A pin for a missed marker: a diamond, 6 px to each point. */
const PIN_MARKERS: Readonly<Record<string, MarkerShape>> = {
  missed: {
    paint: "line",
    size: 6,
    path: (x, y, u) => `M${x} ${y - 6 * u}L${x + 6 * u} ${y}L${x} ${y + 6 * u}L${x - 6 * u} ${y}Z`,
  },
  /** Centres off: the marker is still picked at its centre, but nothing is drawn. */
  none: { paint: "line", size: 0, path: () => "" },
};

/** The stage2d layer items of the markers: outer and inner rings, and the centres that are picked. */
function markerItems(markers: readonly MarkerView[]) {
  const outer: EllipseSetItem[] = [];
  const inner: EllipseSetItem[] = [];
  const centres: PointSetItem[] = [];
  const dimmed = new Set<string>();
  for (const v of markers) {
    const m = v.marker;
    if (v.dimmed) dimmed.add(v.key);
    const o = m.ellipse_outer;
    outer.push(
      o
        ? { id: v.key, x: o.cx, y: o.cy, rx: o.a, ry: o.b, angle: o.angle, color: v.colour, dashed: v.dashed }
        : // No fitted ellipse in this run's output: a dashed circle of the run's typical radius.
          { id: v.key, x: m.center[0], y: m.center[1], rx: v.radius, ry: v.radius, color: v.colour, dashed: true },
    );
    const i = m.ellipse_inner;
    if (i) inner.push({ id: v.key, x: i.cx, y: i.cy, rx: i.a, ry: i.b, angle: i.angle, role: "model", color: v.colour, dashed: v.dashed });
    centres.push({ id: v.key, x: m.center[0], y: m.center[1], pickRadius: 0.9 * v.radius, color: v.colour, label: v.label });
  }
  return { outer, inner, centres, dimmed };
}

/** The pins as points: a diamond for a missed marker, a square for a note. */
function pinPointItems(pins: readonly Pin[]): PointSetItem[] {
  return pins.map((p) =>
    p.type === "missed_marker"
      ? { id: pinKey(p.uid), x: p.x, y: p.y, kind: "missed", color: "var(--warn)", label: pinLabel(p), pickRadius: 4 }
      : { id: pinKey(p.uid), x: p.x, y: p.y, kind: "square", role: "label", label: pinLabel(p), pickRadius: 4 },
  );
}

export interface ReviewStageProps {
  tile: Tile;
  markers: readonly MarkerView[];
  pins: readonly Pin[];
  layers: LayerVisibility;
  onLayerChange: (id: keyof LayerVisibility, visible: boolean) => void;
  selected: string | null;
  onSelect: (key: string | null) => void;
  pinMode: PinType | null;
  onPinModeChange: (mode: PinType | null) => void;
  onDropPin: (type: PinType, point: Point) => void;
  gain: number;
  gamma: number;
  onAdjust: (gain: number, gamma: number) => void;
  view: StageView | null;
  onView: (view: StageView, change: StageViewChange) => void;
  ref: Ref<StageHandle>;
}

export function ReviewStage({ ref, tile, markers, pins, layers, selected, onSelect, pinMode, onDropPin, ...props }: ReviewStageProps) {
  const [cursor, setCursor] = useState<Point | null>(null);
  const [hovered, setHovered] = useState<string | number | null>(null);

  const { outer, inner, centres, dimmed } = useMemo(() => markerItems(markers), [markers]);
  const pinItems = useMemo(() => pinPointItems(pins), [pins]);

  const selectedIds = useMemo(() => (selected === null ? [] : [selected]), [selected]);
  const press = (id: string | number) => onSelect(String(id));

  const toolbar = (
    <StageToolbar>
      <StageLayersMenu
        layers={(Object.keys(LAYER_NAMES) as (keyof LayerVisibility)[]).map((id) => ({
          id,
          label: LAYER_NAMES[id].label,
          shortcut: LAYER_NAMES[id].shortcut,
          visible: layers[id],
        }))}
        onVisibleChange={(id, visible) => props.onLayerChange(id as keyof LayerVisibility, visible)}
      />
      <Popover
        aria-label="Image adjustment"
        side="top"
        trigger={
          <StageButton label="Image adjustment" pressed={props.gain !== 1 || props.gamma !== 1}>
            <SlidersHorizontal className="size-4" aria-hidden />
          </StageButton>
        }
      >
        <div className="grid w-60 gap-3 p-1">
          <label className="grid gap-1 text-xs font-medium">
            Gain
            <Slider
              aria-label="Gain"
              min={0.5}
              max={6}
              step={0.05}
              value={props.gain}
              onValueChange={(g) => props.onAdjust(g, props.gamma)}
              readout={`×${props.gain.toFixed(2)}`}
            />
          </label>
          <label className="grid gap-1 text-xs font-medium">
            Gamma
            <Slider
              aria-label="Gamma"
              min={0.4}
              max={3}
              step={0.05}
              value={props.gamma}
              onValueChange={(g) => props.onAdjust(props.gain, g)}
              readout={props.gamma.toFixed(2)}
            />
          </label>
          <p className="text-xs text-fg-muted">On screen only: detections and pins use the image as recorded.</p>
          <Button size="sm" variant="secondary" onClick={() => props.onAdjust(1, 1)}>
            Reset
          </Button>
        </div>
      </Popover>
      <StageToolbarDivider />
      <StageButton
        label="Pin a missed marker"
        hint="Then click the image; or shift-click"
        shortcut="M"
        pressed={pinMode === "missed_marker"}
        onClick={() => props.onPinModeChange(pinMode === "missed_marker" ? null : "missed_marker")}
      >
        <Diamond className="size-4" aria-hidden />
      </StageButton>
      <StageButton
        label="Pin a note"
        hint="Then click the image"
        shortcut="N"
        pressed={pinMode === "note"}
        onClick={() => props.onPinModeChange(pinMode === "note" ? null : "note")}
      >
        <StickyNote className="size-4" aria-hidden />
      </StageButton>
    </StageToolbar>
  );

  const banner = pinMode ? (
    <span className="rounded-control bg-overlay/90 px-2 py-1 text-xs font-medium text-fg shadow-lg">
      Click the image to pin {pinMode === "missed_marker" ? "a missed marker" : "a note"} · Esc cancels
    </span>
  ) : null;

  return (
    <ImageStage
      ref={ref}
      key={tile.name}
      image={{ width: tile.width, height: tile.height }}
      view={props.view}
      onView={props.onView}
      initialView="fit"
      shortcuts={false}
      label={`Tile ${tile.name}`}
      toolbar={toolbar}
      readout={<StageReadout cursor={cursor} />}
      banner={banner}
      onHover={setCursor}
      onBackgroundClick={(event, point) => {
        if (pinMode) onDropPin(pinMode, point);
        else if (event.shiftKey) onDropPin("missed_marker", point);
        else onSelect(null);
      }}
      className={pinMode ? "cursor-crosshair" : ""}
    >
      <ImageLayer src={tileImageUrl(tile.name)} alt={`Tile ${tile.name}`} gain={props.gain} gamma={props.gamma} />
      {layers.ellipses && (
        <>
          <EllipseSet
            items={outer}
            selectedIds={selectedIds}
            hoveredId={hovered}
            dimmed={dimmed}
            pickable={false}
            label="Fitted outer rings"
          />
          {inner.length > 0 && (
            <EllipseSet items={inner} role="model" selectedIds={selectedIds} hoveredId={hovered} dimmed={dimmed} pickable={false} />
          )}
        </>
      )}
      <PointSet
        items={centres}
        kind={layers.centres ? "plus" : "none"}
        markers={PIN_MARKERS}
        selectedIds={selectedIds}
        dimmed={dimmed}
        labels
        onHoverChange={setHovered}
        onItemPress={press}
        layerId="markers"
        label="Marker centres"
      />
      <PointSet
        items={pinItems}
        markers={PIN_MARKERS}
        selectedIds={selectedIds}
        onItemPress={press}
        layerId="pins"
        priority={STAGE_HIT_PRIORITY.point + 10}
        label="Pins"
      />
    </ImageStage>
  );
}
