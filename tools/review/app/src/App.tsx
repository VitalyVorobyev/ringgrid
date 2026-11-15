/**
 * The review app: a workbench `AppShell` with the tile navigator on the left, the stage in the
 * middle, the inspector on the right and the save state in the status bar. State that is the
 * reviewer's work (the feedback document) lives in `useFeedback`; everything else here is view
 * state.
 */

import { ColormapLegend } from "@vitavision/charts";
import { scaleRange, steppedScale, type Box, type Point, type StageHandle, type StageView } from "@vitavision/stage2d";
import {
  Button,
  DensityProvider,
  ErrorBox,
  NumberInput,
  Popover,
  Select,
  SkeletonRows,
  StatusDot,
  Switch,
  ThemeToggle,
  Toaster,
  TooltipProvider,
  toast,
} from "@vitavision/ui";
import { AppShell, StatusBar } from "@vitavision/workbench";
import { ArrowLeftRight, Keyboard, Plus, Settings2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { GeneralPanel, SelectionPanel, TilePanel } from "./components/Inspector";
import { HelpDialog, RunDialog } from "./components/Dialogs";
import { ReviewStage, type LayerVisibility } from "./components/ReviewStage";
import { passesFilter, TileList, type TileFilter } from "./components/TileList";
import { api, type Comparison, type Detection, type FlagKind, type PinType, type ServerState, type Verdict } from "./model/api";
import { flagFor, newPin, pinUid, tileRecord } from "./model/feedback";
import { commandFor, isTyping, type Command } from "./model/keymap";
import { comparisonSummary, markerViews, pinKey } from "./model/overlay";
import { useFeedback, type SaveState } from "./model/useFeedback";

const nowIso = () => new Date().toISOString();

export function App() {
  const [state, setState] = useState<ServerState | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.state().then(setState, (e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);
  if (error) return <ErrorBox className="m-6">Cannot load /api/state: {error}</ErrorBox>;
  if (!state) return <SkeletonRows rows={6} className="m-6" />;
  return (
    <TooltipProvider>
      <Review initial={state} />
      <Toaster />
    </TooltipProvider>
  );
}

function saveReadout(save: SaveState) {
  switch (save.kind) {
    case "saved":
      return <StatusDot tone="normal">{save.at ? `saved ${save.at.toLocaleTimeString()}` : "saved"}</StatusDot>;
    case "pending":
      return <StatusDot tone="neutral">unsaved changes</StatusDot>;
    case "saving":
      return <StatusDot tone="neutral">saving…</StatusDot>;
    case "failed":
      return <StatusDot tone="defect">save failed: {save.error} (kept in the page, retrying)</StatusDot>;
  }
}

function Review({ initial }: { initial: ServerState }) {
  const [server, setServer] = useState(initial);
  const { feedback, dispatch, save } = useFeedback(initial.feedback);
  const tiles = server.dataset.tiles;
  const differing = useMemo(() => new Set(server.differing_tiles), [server.differing_tiles]);

  const [filter, setFilter] = useState<TileFilter>("all");
  const [tile, setTile] = useState<string | null>(() => {
    const want = decodeURIComponent(window.location.hash.slice(1));
    return (tiles.find((t) => t.name === want) ?? tiles.find((t) => !initial.feedback.tiles[t.name]?.verdict) ?? tiles[0])?.name ?? null;
  });
  const [runA, setRunA] = useState<string | null>(initial.runs[0]?.name ?? null);
  const [runB, setRunB] = useState<string | null>(initial.runs[1]?.name ?? null);
  const [compare, setCompare] = useState(false);
  const [differencesOnly, setDifferencesOnly] = useState(false);
  const [tolerance, setTolerance] = useState(initial.defaults);
  const [layers, setLayers] = useState<LayerVisibility>({ centres: true, ellipses: true, ids: true, grid: false, confidence: false });
  const [adjust, setAdjust] = useState({ gain: 1, gamma: 1 });
  const [pinMode, setPinMode] = useState<PinType | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [view, setView] = useState<StageView | null>(null);
  const box = useRef<Box>({ width: 0, height: 0 });
  const stage = useRef<StageHandle>(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [runDialogOpen, setRunDialogOpen] = useState(false);

  const comparing = compare && runB !== null && runB !== runA;
  const visibleTiles = useMemo(
    () => tiles.filter((t) => passesFilter(filter, t.name, feedback, differing)),
    [tiles, filter, feedback, differing],
  );
  const currentTile = tiles.find((t) => t.name === tile) ?? null;
  const runByName = (name: string | null) => server.runs.find((r) => r.name === name);

  // -- detections of the current tile ------------------------------------------------------
  // Fetched results are kept with the request they answer, so a result for another tile or
  // run pair is never shown while the next one loads.
  const [fetched, setFetched] = useState<{ key: string; detections: Record<string, Detection | null> } | null>(null);
  const [compared, setCompared] = useState<{ key: string; comparison: Comparison | null } | null>(null);
  const [reload, setReload] = useState(0);
  const runsKey = [...new Set([runA, runB].filter((r): r is string => r !== null))].join("\n");
  const detectionsKey = tile ? `${tile}\n${runsKey}\n${reload}` : null;
  const comparisonKey =
    tile && runA && runB && runA !== runB ? `${tile}\n${runA}\n${runB}\n${tolerance.match_px}\n${tolerance.pos_px}\n${reload}` : null;
  useEffect(() => {
    if (!tile || detectionsKey === null) return;
    let live = true;
    const runs = runsKey === "" ? [] : runsKey.split("\n");
    void Promise.all(
      runs.map((r) =>
        api.detection(r, tile).then(
          (d) => [r, d] as const,
          () => [r, null] as const,
        ),
      ),
    ).then((entries) => {
      if (live) setFetched({ key: detectionsKey, detections: Object.fromEntries(entries) });
    });
    return () => {
      live = false;
    };
  }, [tile, runsKey, detectionsKey]);
  useEffect(() => {
    if (!tile || !runA || !runB || comparisonKey === null) return;
    let live = true;
    api.compare(tile, runA, runB, tolerance.match_px, tolerance.pos_px).then(
      (c) => live && setCompared({ key: comparisonKey, comparison: c }),
      () => live && setCompared({ key: comparisonKey, comparison: null }),
    );
    return () => {
      live = false;
    };
  }, [tile, runA, runB, tolerance, comparisonKey]);
  const detections = fetched?.key === detectionsKey ? fetched.detections : {};
  const comparison = compared?.key === comparisonKey ? compared.comparison : null;

  // -- polling while a run is being computed ------------------------------------------------
  const running = server.runs.some((r) => r.status.state === "running");
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      api.state().then(
        (s) => {
          setServer(s);
          if (!s.runs.some((r) => r.status.state === "running")) {
            setReload((n) => n + 1);
            const failed = s.runs.find((r) => r.status.errors.length > 0);
            if (failed)
              toast({
                title: `Run ${failed.name}: ${failed.status.errors.length} tile error(s)`,
                description: failed.status.errors[0],
                tone: "error",
              });
          }
        },
        () => {},
      );
    }, 2000);
    return () => clearInterval(id);
  }, [running]);

  // -- the overlay -----------------------------------------------------------------------
  const rec = tile ? tileRecord(feedback, tile) : null;
  const detA = runA ? detections[runA] : undefined;
  const detB = runB ? detections[runB] : undefined;
  const markers = useMemo(
    () =>
      markerViews({
        a: runA && detA ? { run: runA, detection: detA } : null,
        b: comparing && runB && detB ? { run: runB, detection: detB } : null,
        comparison: comparing ? comparison : null,
        flags: rec?.marker_flags ?? [],
        selected,
        options: { compare: comparing, differencesOnly, confidence: layers.confidence, ids: layers.ids, grid: layers.grid },
      }),
    [runA, runB, detA, detB, comparing, comparison, rec?.marker_flags, selected, differencesOnly, layers],
  );
  const selectedView = markers.find((m) => m.key === selected) ?? null;
  const selectedPin = rec?.pins.find((p) => pinKey(p.uid) === selected) ?? null;
  const visibleRuns = [runA, comparing ? runB : null].filter((r): r is string => r !== null);

  // -- actions -----------------------------------------------------------------------------
  const goTo = useCallback((name: string) => {
    setTile(name);
    setSelected(null);
    setPinMode(null);
    setView(null);
    window.history.replaceState(null, "", `#${encodeURIComponent(name)}`);
  }, []);
  const step = (delta: 1 | -1) => {
    if (visibleTiles.length === 0) return;
    let i = visibleTiles.findIndex((t) => t.name === tile);
    if (i < 0) {
      // The current tile is filtered out: continue from where it sits in the full list.
      const at = tiles.findIndex((t) => t.name === tile);
      const after = visibleTiles.findIndex((t) => tiles.indexOf(t) > at);
      i = after < 0 ? visibleTiles.length : after;
      if (delta > 0) i -= 1;
    }
    const next = visibleTiles[Math.min(visibleTiles.length - 1, Math.max(0, i + delta))];
    if (next) goTo(next.name);
  };
  const seen = () => ({ tile: tile!, now: nowIso(), runs: visibleRuns });
  const setVerdict = (verdict: Verdict) => tile && dispatch({ type: "verdict", verdict, ...seen() });
  const setFlag = (kind: FlagKind) => {
    if (!tile || !selectedView) return;
    dispatch({ type: "flag", run: selectedView.run, marker: selectedView.marker, kind, ...seen() });
    if (kind === "wrong_id") focusSoon('[data-autofocus="correct-id"]');
  };
  const clear = () => {
    if (!tile) return;
    if (selectedPin) {
      dispatch({ type: "deletePin", uid: selectedPin.uid, ...seen() });
      setSelected(null);
    } else if (selectedView && flagFor(feedback, tile, selectedView.run, selectedView.marker.index)) {
      dispatch({ type: "clearFlag", run: selectedView.run, index: selectedView.marker.index, ...seen() });
    }
  };
  const dropPin = (type: PinType, p: Point) => {
    if (!tile) return;
    const pin = newPin(type, p.x, p.y, nowIso(), pinUid());
    dispatch({ type: "addPin", pin, ...seen() });
    setSelected(pinKey(pin.uid));
    setPinMode(null);
    focusSoon('[data-autofocus="pin"]');
  };
  const swapRuns = () => {
    if (!runB) return;
    setRunA(runB);
    setRunB(runA);
    setSelected(null);
  };
  const toggleCompare = () => {
    if (!compare && (runB === null || runB === runA)) {
      const other = server.runs.find((r) => r.name !== runA);
      if (!other) {
        toast({ title: "Only one run: nothing to compare", tone: "info" });
        return;
      }
      setRunB(other.name);
    }
    setCompare(!compare);
  };
  const zoom = (direction: 1 | -1) => {
    if (!currentTile || !view) return;
    const [min, max] = scaleRange(box.current, { width: currentTile.width, height: currentTile.height });
    stage.current?.zoomTo(steppedScale(view.scale, direction, min, max));
  };
  const toggleLayer = (id: keyof LayerVisibility) => setLayers((l) => ({ ...l, [id]: !l[id] }));

  const run: Record<Command, () => void> = {
    nextTile: () => step(1),
    prevTile: () => step(-1),
    verdictOk: () => setVerdict("ok"),
    verdictIssues: () => setVerdict("has_issues"),
    verdictUnusable: () => setVerdict("unusable_image"),
    focusRemark: () => document.getElementById("tile-remark")?.focus(),
    swapRuns,
    toggleCompare,
    toggleDifferences: () => setDifferencesOnly((d) => !d),
    toggleEllipses: () => toggleLayer("ellipses"),
    toggleIds: () => toggleLayer("ids"),
    toggleGrid: () => toggleLayer("grid"),
    toggleConfidence: () => toggleLayer("confidence"),
    fit: () => stage.current?.fit(),
    actualSize: () => stage.current?.zoomTo(1),
    zoomIn: () => zoom(1),
    zoomOut: () => zoom(-1),
    flagWrongId: () => setFlag("wrong_id"),
    flagWrongCentre: () => setFlag("wrong_center"),
    flagFalsePositive: () => setFlag("false_positive"),
    flagOk: () => setFlag("ok_confirmed"),
    clear,
    missedMode: () => setPinMode((m) => (m === "missed_marker" ? null : "missed_marker")),
    noteMode: () => setPinMode((m) => (m === "note" ? null : "note")),
    escape: () => {
      setPinMode(null);
      setSelected(null);
    },
    help: () => setHelpOpen((o) => !o),
  };
  // The key handler is installed once and reads the latest commands through a ref.
  const commands = useRef(run);
  useEffect(() => {
    commands.current = run;
  });
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if (isTyping(event.target)) {
        if (event.key === "Escape" && event.target instanceof HTMLElement && !event.target.closest("[role=dialog]")) event.target.blur();
        return;
      }
      const command = commandFor(event);
      if (!command) return;
      event.preventDefault();
      commands.current[command]();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // -- layout ----------------------------------------------------------------------------
  const runOptions = server.runs.map((r) => ({
    value: r.name,
    label: r.status.state === "running" ? `${r.name} (${r.status.done}/${r.status.total})` : r.name,
  }));
  const header = (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 px-3 py-1.5">
      <span className="text-sm font-semibold tracking-tight">{server.dataset.name}</span>
      <div className="flex items-center gap-1.5">
        <span className="text-xs font-medium text-fg-muted">A</span>
        <Select aria-label="Run A" value={runA ?? ""} options={runOptions} onValueChange={(v) => setRunA(v)} className="w-40" />
        <Button
          size="sm"
          variant="ghost"
          aria-label="Swap runs A and B (X)"
          title="Swap runs A and B (X)"
          onClick={swapRuns}
          icon={<ArrowLeftRight className="size-4" />}
        />
        <span className="text-xs font-medium text-fg-muted">B</span>
        <Select
          aria-label="Run B"
          value={runB ?? ""}
          options={runOptions}
          unsetLabel="(none)"
          onValueChange={(v) => setRunB(v === "" ? null : v)}
          className="w-40"
        />
      </div>
      <Switch checked={comparing} onCheckedChange={toggleCompare} label="Compare (V)" />
      <Switch checked={differencesOnly} onCheckedChange={setDifferencesOnly} label="Differences only (D)" disabled={!comparing} />
      <Popover
        aria-label="Matching tolerances"
        trigger={
          <Button size="sm" variant="ghost" icon={<Settings2 className="size-4" />}>
            Tolerances
          </Button>
        }
      >
        <div className="grid w-56 gap-2 p-1 text-xs">
          <label className="grid gap-1 font-medium">
            Match radius
            <NumberInput
              min={1}
              max={100}
              precision={1}
              unit="px"
              value={tolerance.match_px}
              onValueChange={(v) => v > 0 && setTolerance((t) => ({ ...t, match_px: v }))}
            />
          </label>
          <label className="grid gap-1 font-medium">
            Same position within
            <NumberInput
              min={0}
              max={20}
              precision={2}
              unit="px"
              value={tolerance.pos_px}
              onValueChange={(v) => v >= 0 && setTolerance((t) => ({ ...t, pos_px: v }))}
            />
          </label>
          <p className="text-fg-muted">
            Markers of A and B closer than the match radius are paired; a pair further apart than the position tolerance differs.
          </p>
        </div>
      </Popover>
      <div className="ml-auto flex items-center gap-1.5">
        <Button size="sm" variant="secondary" icon={<Plus className="size-4" />} onClick={() => setRunDialogOpen(true)}>
          New run
        </Button>
        <Button
          size="sm"
          variant="ghost"
          aria-label="Keyboard shortcuts (?)"
          title="Keyboard shortcuts (?)"
          onClick={() => setHelpOpen(true)}
          icon={<Keyboard className="size-4" />}
        />
        <ThemeToggle />
      </div>
    </div>
  );

  const legend = comparing ? (comparison ? `A solid · B dashed · ${comparisonSummary(comparison)}` : "no comparison for this tile") : null;
  const warnings = [...(detA?.warnings ?? []), ...(comparing ? (detB?.warnings ?? []) : [])];

  const main = currentTile ? (
    <div className="flex h-full min-h-0 flex-col">
      <div className="relative min-h-0 flex-1 bg-canvas">
        <ReviewStage
          tile={currentTile}
          markers={markers}
          pins={rec?.pins ?? []}
          layers={layers}
          onLayerChange={(id, visible) => setLayers((l) => ({ ...l, [id]: visible }))}
          selected={selected}
          onSelect={setSelected}
          pinMode={pinMode}
          onPinModeChange={setPinMode}
          onDropPin={dropPin}
          gain={adjust.gain}
          gamma={adjust.gamma}
          onAdjust={(gain, gamma) => setAdjust({ gain, gamma })}
          view={view}
          onView={(v, change) => {
            box.current = change.box;
            setView(v);
          }}
          ref={stage}
        />
      </div>
      {(layers.confidence && !comparing) || warnings.length > 0 || (runA && detA === null) ? (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-line bg-surface px-3 py-1 text-xs text-fg-muted">
          {layers.confidence && !comparing && <ColormapLegend map="viridis" domain={[0, 1]} label="Confidence" />}
          {runA && detA === null && <span>Run {runA} has no detection for this tile.</span>}
          {warnings.length > 0 && <span title={warnings.join("\n")}>{warnings.length} normalisation warning(s)</span>}
        </div>
      ) : null}
    </div>
  ) : (
    <ErrorBox className="m-6">No tiles in this dataset.</ErrorBox>
  );

  const right = tile ? (
    <DensityProvider value="compact">
      <div className="grid gap-2.5 p-2.5">
        <SelectionPanel
          view={selectedView}
          pin={selectedPin}
          comparing={comparing}
          onFlag={setFlag}
          onClearFlag={clear}
          onCorrectId={(value) =>
            selectedView &&
            dispatch({ type: "flagField", tile, run: selectedView.run, index: selectedView.marker.index, field: "correct_id", value })
          }
          onFlagRemark={(value) =>
            selectedView &&
            dispatch({ type: "flagField", tile, run: selectedView.run, index: selectedView.marker.index, field: "remark", value })
          }
          onPinChange={(patch) => selectedPin && dispatch({ type: "updatePin", tile, uid: selectedPin.uid, patch })}
          onDeletePin={clear}
        />
        <TilePanel
          tile={tile}
          feedback={feedback}
          onVerdict={setVerdict}
          onRemark={(text) => dispatch({ type: "remark", text, ...seen() })}
        />
        <GeneralPanel feedback={feedback} onAdd={(text) => dispatch({ type: "generalRemark", text, now: nowIso() })} />
      </div>
    </DensityProvider>
  ) : null;

  return (
    <>
      <AppShell
        header={header}
        left={
          <DensityProvider value="compact">
            <TileList
              tiles={visibleTiles}
              allTiles={tiles}
              current={tile}
              onSelect={goTo}
              feedback={feedback}
              differing={differing}
              runA={runByName(runA)}
              runB={comparing ? runByName(runB) : undefined}
              filter={filter}
              onFilterChange={setFilter}
            />
          </DensityProvider>
        }
        leftLabel="Tiles"
        leftSize={{ defaultSize: 300, minSize: 220, maxSize: "40%" }}
        main={main}
        right={right}
        rightSize={{ defaultSize: 330, minSize: 260, maxSize: "45%" }}
        storageKey="ringgrid-review"
        bottom={
          <StatusBar
            live
            start={[{ value: saveReadout(save) }]}
            end={[
              { label: "tile", value: tile ?? "–" },
              ...(legend ? [{ value: legend }] : []),
              ...(running ? [{ value: "computing a run…" }] : []),
            ]}
          />
        }
      />
      <HelpDialog open={helpOpen} onOpenChange={setHelpOpen} />
      <RunDialog
        open={runDialogOpen}
        onOpenChange={setRunDialogOpen}
        binaries={server.binaries}
        onCreated={(name) => {
          toast({ title: `Run ${name} started`, tone: "info" });
          void api.state().then(setServer);
        }}
      />
    </>
  );
}

/** Focus a field the next render brings (the correct id after a wrong-id flag, a new pin's field). */
function focusSoon(selector: string) {
  setTimeout(() => {
    const el = document.querySelector<HTMLElement>(selector);
    el?.focus();
    if (el instanceof HTMLInputElement) el.select();
  }, 0);
}
