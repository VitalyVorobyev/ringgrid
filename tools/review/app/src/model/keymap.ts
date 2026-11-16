/**
 * The keyboard map: one table that both binds the keys (`commandFor`) and feeds the `?` help
 * (`helpGroups`), so the two cannot disagree.
 */

import type { ShortcutGroup } from "../shims/ui";

export type Command =
  | "nextTile"
  | "prevTile"
  | "verdictOk"
  | "verdictIssues"
  | "verdictUnusable"
  | "focusRemark"
  | "swapRuns"
  | "toggleCompare"
  | "toggleDifferences"
  | "toggleEllipses"
  | "toggleIds"
  | "toggleGrid"
  | "toggleConfidence"
  | "fit"
  | "actualSize"
  | "zoomIn"
  | "zoomOut"
  | "flagWrongId"
  | "flagWrongCentre"
  | "flagFalsePositive"
  | "flagOk"
  | "clear"
  | "missedMode"
  | "noteMode"
  | "escape"
  | "help";

interface Binding {
  /** `KeyboardEvent.key` values. */
  keys: readonly string[];
  /** How the keys are shown in the help. */
  shown: readonly string[];
  description: string;
  /** Null for a row that documents a pointer gesture. */
  command: Command | null;
}

export const KEYMAP: readonly { title: string; bindings: readonly Binding[] }[] = [
  {
    title: "Tiles",
    bindings: [
      { keys: ["j", "ArrowDown", "ArrowRight"], shown: ["J", "↓", "→"], description: "Next tile (in the filter)", command: "nextTile" },
      { keys: ["k", "ArrowUp", "ArrowLeft"], shown: ["K", "↑", "←"], description: "Previous tile", command: "prevTile" },
      { keys: ["1"], shown: ["1"], description: "Verdict: ok (again to clear)", command: "verdictOk" },
      { keys: ["2"], shown: ["2"], description: "Verdict: has issues", command: "verdictIssues" },
      { keys: ["3"], shown: ["3"], description: "Verdict: unusable image", command: "verdictUnusable" },
      { keys: ["/"], shown: ["/"], description: "Write the tile remark (Esc leaves)", command: "focusRemark" },
    ],
  },
  {
    title: "Markers and pins",
    bindings: [
      { keys: [], shown: ["click"], description: "Select a marker or pin", command: null },
      { keys: ["w"], shown: ["W"], description: "Flag: wrong id (then type the right one)", command: "flagWrongId" },
      { keys: ["c"], shown: ["C"], description: "Flag: wrong centre", command: "flagWrongCentre" },
      { keys: ["f"], shown: ["F"], description: "Flag: false positive", command: "flagFalsePositive" },
      { keys: ["o"], shown: ["O"], description: "Confirm ok", command: "flagOk" },
      { keys: ["u", "Backspace", "Delete"], shown: ["U", "⌫"], description: "Clear the flag, or delete the pin", command: "clear" },
      { keys: ["m"], shown: ["M", "⇧ click"], description: "Pin a missed marker", command: "missedMode" },
      { keys: ["n"], shown: ["N"], description: "Pin a note", command: "noteMode" },
      { keys: ["Escape"], shown: ["Esc"], description: "Leave pin mode, deselect", command: "escape" },
    ],
  },
  {
    title: "Runs",
    bindings: [
      { keys: ["v"], shown: ["V"], description: "Compare run A with run B", command: "toggleCompare" },
      { keys: ["d"], shown: ["D"], description: "Differences only", command: "toggleDifferences" },
      { keys: ["x"], shown: ["X"], description: "Swap runs A and B", command: "swapRuns" },
    ],
  },
  {
    title: "View",
    bindings: [
      { keys: ["e"], shown: ["E"], description: "Ellipses", command: "toggleEllipses" },
      { keys: ["i"], shown: ["I"], description: "Ids", command: "toggleIds" },
      { keys: ["g"], shown: ["G"], description: "Grid coordinates", command: "toggleGrid" },
      { keys: ["p"], shown: ["P"], description: "Colour by confidence", command: "toggleConfidence" },
      { keys: ["0"], shown: ["0"], description: "Fit to window", command: "fit" },
      { keys: ["."], shown: ["."], description: "Actual size (100%)", command: "actualSize" },
      { keys: ["+", "="], shown: ["+", "−"], description: "Zoom in, out (or the wheel; drag to pan)", command: "zoomIn" },
      { keys: ["-", "_"], shown: [], description: "", command: "zoomOut" },
      { keys: ["?"], shown: ["?"], description: "This help", command: "help" },
    ],
  },
];

const BY_KEY = new Map<string, Command>();
for (const group of KEYMAP) {
  for (const b of group.bindings) {
    if (b.command) for (const key of b.keys) BY_KEY.set(key, b.command);
  }
}

/** The command a key press runs, or null. Presses with ⌘, Ctrl or Alt are the browser's. */
export function commandFor(event: Pick<KeyboardEvent, "key" | "metaKey" | "ctrlKey" | "altKey">): Command | null {
  if (event.metaKey || event.ctrlKey || event.altKey) return null;
  return BY_KEY.get(event.key) ?? null;
}

/** Whether a key press belongs to a text field (or a menu or dialog), not to the review keys. */
export function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  if (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return true;
  return target.closest("[role=dialog], [role=menu], [role=listbox]") !== null;
}

/** The help, as `ShortcutList` groups. */
export function helpGroups(): ShortcutGroup[] {
  return KEYMAP.map((group) => ({
    title: group.title,
    shortcuts: group.bindings.filter((b) => b.shown.length > 0).map((b) => ({ keys: b.shown, description: b.description })),
  }));
}
