/**
 * The navigator: tiles as a `Table` (counts per run, badges for differences, flags and the
 * verdict), a filter, and the review progress as a `Breakdown`.
 */

import { Badge, Breakdown, Select, Table, type Column } from "@vitavision/ui";
import { useEffect, useRef } from "react";

import type { Feedback, Run, Tile, Verdict } from "../model/api";
import { issueCount } from "../model/feedback";

export type TileFilter = "all" | "unreviewed" | "has_issues" | "flagged" | "differs";

export const FILTER_OPTIONS: { value: TileFilter; label: string }[] = [
  { value: "all", label: "All tiles" },
  { value: "unreviewed", label: "Unreviewed" },
  { value: "has_issues", label: "Has issues" },
  { value: "flagged", label: "Has flags or pins" },
  { value: "differs", label: "Differs between runs" },
];

export const VERDICT_LABEL: Record<Verdict, string> = { ok: "ok", has_issues: "issues", unusable_image: "unusable" };
const VERDICT_TONE = { ok: "normal", has_issues: "defect", unusable_image: "warning" } as const;

export function passesFilter(filter: TileFilter, tile: string, feedback: Feedback, differing: ReadonlySet<string>): boolean {
  switch (filter) {
    case "unreviewed":
      return !feedback.tiles[tile]?.verdict;
    case "has_issues":
      return feedback.tiles[tile]?.verdict === "has_issues";
    case "flagged":
      return issueCount(feedback, tile) > 0;
    case "differs":
      return differing.has(tile);
    case "all":
      return true;
  }
}

const counts = (c: [number, number] | null | undefined) => (c ? (c[1] !== c[0] ? `${c[0]} / ${c[1]} id` : `${c[0]}`) : "–");

/** The list's columns: name, counts per run, and the review state as badges. */
function tileColumns(feedback: Feedback, differing: ReadonlySet<string>, runA: Run | undefined, runB: Run | undefined): Column<Tile>[] {
  return [
    {
      key: "name",
      header: "Tile",
      cell: (t) => (
        <span className="flex items-center gap-1.5 font-mono text-xs">
          {t.name}
          {t.eval_skip && (
            <span title="Skipped by the rtv3d evaluation">
              <Badge tone="neutral">skip</Badge>
            </span>
          )}
        </span>
      ),
    },
    { key: "a", header: "A", numeric: true, cell: (t) => counts(runA?.counts[t.name]) },
    ...(runB ? [{ key: "b", header: "B", numeric: true, cell: (t: Tile) => counts(runB.counts[t.name]) }] : []),
    {
      key: "state",
      header: <span className="sr-only">Review state</span>,
      cell: (t) => {
        const verdict = feedback.tiles[t.name]?.verdict ?? null;
        const issues = issueCount(feedback, t.name);
        return (
          <span className="flex justify-end gap-1">
            {differing.has(t.name) && (
              <span title="Differs between runs">
                <Badge tone="warning">Δ</Badge>
              </span>
            )}
            {issues > 0 && (
              <span title="Flags and pins">
                <Badge tone="info">⚑ {issues}</Badge>
              </span>
            )}
            {verdict && <Badge tone={VERDICT_TONE[verdict]}>{VERDICT_LABEL[verdict]}</Badge>}
          </span>
        );
      },
    },
  ];
}

export interface TileListProps {
  tiles: readonly Tile[];
  allTiles: readonly Tile[];
  current: string | null;
  onSelect: (tile: string) => void;
  feedback: Feedback;
  differing: ReadonlySet<string>;
  runA: Run | undefined;
  runB: Run | undefined;
  filter: TileFilter;
  onFilterChange: (filter: TileFilter) => void;
}

export function TileList({ tiles, allTiles, current, onSelect, feedback, differing, runA, runB, filter, onFilterChange }: TileListProps) {
  const scroller = useRef<HTMLDivElement>(null);
  useEffect(() => {
    scroller.current?.querySelector('tr[aria-current="true"]')?.scrollIntoView({ block: "nearest" });
  }, [current, filter]);

  const verdicts = { ok: 0, has_issues: 0, unusable_image: 0 };
  for (const t of allTiles) {
    const v = feedback.tiles[t.name]?.verdict;
    if (v) verdicts[v] += 1;
  }

  const columns = tileColumns(feedback, differing, runA, runB);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="grid gap-2 border-b border-line p-2.5">
        <Select aria-label="Filter tiles" value={filter} options={FILTER_OPTIONS} onValueChange={(v) => onFilterChange(v as TileFilter)} />
        <Breakdown
          segments={[
            { label: "ok", tone: "normal", value: verdicts.ok },
            { label: "with issues", tone: "defect", value: verdicts.has_issues },
            { label: "unusable", tone: "warn", value: verdicts.unusable_image },
          ]}
          total={allTiles.length}
          remainderLabel="to review"
        />
        <span className="text-xs text-fg-muted">
          {verdicts.ok + verdicts.has_issues + verdicts.unusable_image} of {allTiles.length} reviewed · {tiles.length} shown
        </span>
      </div>
      <div ref={scroller} className="min-h-0 flex-1 overflow-auto px-2.5">
        <Table
          caption="Tiles"
          columns={columns}
          rows={[...tiles]}
          rowKey={(t) => t.name}
          onRowClick={(t) => onSelect(t.name)}
          isRowActive={(t) => t.name === current}
          empty="No tiles match the filter."
        />
      </div>
    </div>
  );
}
