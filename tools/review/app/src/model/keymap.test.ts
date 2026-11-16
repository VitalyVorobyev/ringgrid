import { describe, expect, it } from "vitest";

import { commandFor, helpGroups, isTyping, KEYMAP } from "./keymap";

const key = (k: string, mods: Partial<{ metaKey: boolean; ctrlKey: boolean; altKey: boolean }> = {}) => ({
  key: k,
  metaKey: false,
  ctrlKey: false,
  altKey: false,
  ...mods,
});

describe("keymap", () => {
  it("binds every key to one command, and none twice", () => {
    const seen = new Map<string, string>();
    for (const group of KEYMAP) {
      for (const b of group.bindings) {
        for (const k of b.keys) {
          expect(seen.has(k), `key ${k} bound twice`).toBe(false);
          seen.set(k, String(b.command));
        }
      }
    }
    expect(commandFor(key("j"))).toBe("nextTile");
    expect(commandFor(key("ArrowLeft"))).toBe("prevTile");
    expect(commandFor(key("w"))).toBe("flagWrongId");
    expect(commandFor(key("Delete"))).toBe("clear");
    expect(commandFor(key("?"))).toBe("help");
    expect(commandFor(key("q"))).toBeNull();
  });

  it("leaves modified presses (copy, reload) to the browser", () => {
    expect(commandFor(key("c", { metaKey: true }))).toBeNull();
    expect(commandFor(key("r", { ctrlKey: true }))).toBeNull();
    expect(commandFor(key("1", { altKey: true }))).toBeNull();
  });

  it("does not take keys typed into fields or dialogs", () => {
    const input = document.createElement("input");
    const dialog = document.createElement("div");
    dialog.setAttribute("role", "dialog");
    const inside = document.createElement("button");
    dialog.append(inside);
    document.body.append(input, dialog);
    expect(isTyping(input)).toBe(true);
    expect(isTyping(inside)).toBe(true);
    expect(isTyping(document.body)).toBe(false);
    expect(isTyping(null)).toBe(false);
  });

  it("documents every bound command in the help, pointer gestures included", () => {
    const help = helpGroups();
    const descriptions = help.flatMap((g) => g.shortcuts.map((s) => s.description));
    expect(descriptions).toContain("Select a marker or pin");
    expect(descriptions).toContain("Swap runs A and B");
    // The zoom-out row is folded into the zoom-in row's keys.
    expect(descriptions).not.toContain("");
    expect(help.every((g) => g.shortcuts.every((s) => s.keys.length > 0))).toBe(true);
  });
});
