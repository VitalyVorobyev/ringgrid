/** The two dialogs: a new detection run, and the keyboard help. */

import { Button, Callout, Dialog, Field, Input, Select } from "@vitavision/ui";
import { useState } from "react";

import { api, type ServerState } from "../model/api";
import { helpGroups } from "../model/keymap";
import { ShortcutList } from "../shims/ui";

export function RunDialog({
  open,
  onOpenChange,
  binaries,
  onCreated,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  binaries: ServerState["binaries"];
  onCreated: (run: string) => void;
}) {
  const [name, setName] = useState("");
  const [binary, setBinary] = useState(binaries[0]?.path ?? "");
  const [args, setArgs] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const valid = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(name) && binary !== "";

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await api.createRun(name, binary, args);
      onCreated(res.run);
      onOpenChange(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="New detection run"
      description="Runs an allow-listed detector on every tile; results are cached under the output directory."
      footer={
        <>
          <Button variant="secondary" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={() => void start()} disabled={!valid || busy} loading={busy}>
            Start
          </Button>
        </>
      }
    >
      {binaries.length === 0 ? (
        <Callout tone="warning">
          No detector is allowed: start the server with --detect NAME=BINARY, or build target/release/ringgrid-dev.
        </Callout>
      ) : (
        <div className="grid gap-3">
          <Field label="Run name" description="Letters, digits, . _ -">
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. current-20-56" />
          </Field>
          <Field label="Detector">
            <Select
              value={binary}
              onValueChange={setBinary}
              options={binaries.map((b) => ({ value: b.path, label: b.label, note: b.path }))}
            />
          </Field>
          <Field label="Extra detect arguments">
            <Input value={args} onChange={(e) => setArgs(e.target.value)} placeholder="--marker-diameter-min 20 --marker-diameter-max 56" />
          </Field>
          {error && <Callout tone="error">{error}</Callout>}
        </div>
      )}
    </Dialog>
  );
}

export function HelpDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange} title="Keyboard shortcuts" className="max-w-3xl">
      <ShortcutList groups={helpGroups()} />
    </Dialog>
  );
}
