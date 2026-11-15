/**
 * The feedback document in the page, saved as the reviewer edits: each edit marks its tile (or
 * the general remarks) dirty, and a debounced save sends those records to the server, which
 * writes feedback.json atomically. A failed save keeps the edits in the page and retries.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { api, type Feedback } from "./api";
import { dirtyKey, feedbackReducer, savePayload, type FeedbackAction } from "./feedback";

export type SaveState = { kind: "saved"; at: Date | null } | { kind: "pending" } | { kind: "saving" } | { kind: "failed"; error: string };

const DEBOUNCE_MS = 400;
const RETRY_MS = 4000;

export function useFeedback(initial: Feedback) {
  const [feedback, setFeedback] = useState(initial);
  const [save, setSave] = useState<SaveState>({ kind: "saved", at: null });
  const current = useRef(initial);
  /** Dirty key → version of its last edit. */
  const dirty = useRef(new Map<string, number>());
  const version = useRef(0);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const saving = useRef(false);

  const flush = useCallback(async function run(): Promise<void> {
    if (saving.current || dirty.current.size === 0) return;
    saving.current = true;
    setSave({ kind: "saving" });
    const sent = new Map(dirty.current);
    let failed: string | null = null;
    try {
      await api.saveFeedback(savePayload(current.current, sent.keys()));
      // Keys edited again while the request was in flight stay dirty.
      for (const [key, v] of sent) if (dirty.current.get(key) === v) dirty.current.delete(key);
    } catch (error) {
      failed = error instanceof Error ? error.message : String(error);
    }
    saving.current = false;
    if (failed !== null) setSave({ kind: "failed", error: failed });
    else setSave(dirty.current.size > 0 ? { kind: "pending" } : { kind: "saved", at: new Date() });
    if (dirty.current.size > 0) {
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => void run(), failed !== null ? RETRY_MS : 50);
    }
  }, []);

  const dispatch = useCallback(
    (action: FeedbackAction) => {
      const next = feedbackReducer(current.current, action);
      current.current = next;
      setFeedback(next);
      dirty.current.set(dirtyKey(action), ++version.current);
      setSave((s) => (s.kind === "failed" ? s : { kind: "pending" }));
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => void flush(), action.type === "generalRemark" ? 100 : DEBOUNCE_MS);
    },
    [flush],
  );

  // Leaving with unsaved edits asks first.
  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => {
      if (dirty.current.size > 0) event.preventDefault();
    };
    window.addEventListener("beforeunload", warn);
    return () => {
      window.removeEventListener("beforeunload", warn);
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  return { feedback, dispatch, save };
}
