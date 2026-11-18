import { Button, Callout, Dialog, Field, NumberInput, Select } from "@vitavision/ui";
import { useState } from "react";

import { api, type ScreenTargetExport, type ScreenTargetOptions } from "../model/api";

const INKPAD_LITE: ScreenTargetOptions = {
  width_px: 825,
  height_px: 1200,
  pixel_pitch_mm: 25.4 / 150,
  marker_diameter_mm: 10.752,
  margin_px: 8,
};

const TEMPLATES = [
  { value: "coded_hex", label: "Coded hex" },
  { value: "plain_rect", label: "Plain rectangular with origin dots" },
];

function download(filename: string, data: BlobPart, type: string) {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function Downloads({ generated }: { generated: ScreenTargetExport }) {
  const { metadata } = generated;
  return (
    <div className="grid gap-2">
      <div className="text-xs text-fg-muted" role="status">
        {metadata.screen.width_px} × {metadata.screen.height_px} px image. {metadata.marker_count} markers, {metadata.rows} rows and up to{" "}
        {metadata.columns} columns. Drawn diameter {metadata.drawn_marker_diameter_mm.toFixed(3)} mm (
        {metadata.drawn_marker_diameter_px.toFixed(2)} px).
        {metadata.code_band_width_px !== null && <> Code band {metadata.code_band_width_px.toFixed(2)} px wide.</>}
      </div>
      <div className="flex flex-wrap gap-2">
        <Button
          size="sm"
          variant="secondary"
          onClick={() =>
            download(
              "target_screen.png",
              Uint8Array.from(atob(generated.png_base64), (c) => c.charCodeAt(0)),
              "image/png",
            )
          }
        >
          PNG
        </Button>
        <Button size="sm" variant="secondary" onClick={() => download("target_spec.json", generated.json_text, "application/json")}>
          Target JSON
        </Button>
        <Button
          size="sm"
          variant="secondary"
          onClick={() => download("screen.json", JSON.stringify(metadata, null, 2), "application/json")}
        >
          Screen JSON
        </Button>
      </div>
    </div>
  );
}

export function ScreenTargetDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (value: boolean) => void }) {
  const [template, setTemplate] = useState("coded_hex");
  const [options, setOptions] = useState(INKPAD_LITE);
  const [generated, setGenerated] = useState<ScreenTargetExport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  function setNumber(field: keyof ScreenTargetOptions, value: number) {
    setOptions((previous) => ({ ...previous, [field]: value }));
    setGenerated(null);
    setError(null);
  }

  async function generate() {
    setError(null);
    setGenerated(null);
    setBusy(true);
    try {
      setGenerated(await api.generateScreenTarget(template, options));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="Generate screen target"
      description="Fit the largest grid at the requested marker diameter. Display the downloaded PNG at one image pixel per screen pixel."
      footer={
        <Button disabled={busy} loading={busy} onClick={() => void generate()}>
          Generate
        </Button>
      }
    >
      <fieldset disabled={busy} className="grid gap-3">
        <div>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => {
              setTemplate("coded_hex");
              setOptions(INKPAD_LITE);
              setGenerated(null);
              setError(null);
            }}
          >
            InkPad Lite preset
          </Button>
        </div>
        <Field label="Target template">
          <Select
            aria-label="Target template"
            options={TEMPLATES}
            value={template}
            onValueChange={(value) => {
              setTemplate(value);
              setGenerated(null);
              setError(null);
            }}
          />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Native width">
            <NumberInput min={1} unit="px" value={options.width_px} onValueChange={(v) => setNumber("width_px", Math.round(v))} />
          </Field>
          <Field label="Native height">
            <NumberInput min={1} unit="px" value={options.height_px} onValueChange={(v) => setNumber("height_px", Math.round(v))} />
          </Field>
          <Field label="Physical pixel pitch">
            <NumberInput min={0.000001} unit="mm" value={options.pixel_pitch_mm} onValueChange={(v) => setNumber("pixel_pitch_mm", v)} />
          </Field>
          <Field label="Drawn marker diameter">
            <NumberInput
              min={0.001}
              unit="mm"
              value={options.marker_diameter_mm}
              onValueChange={(v) => setNumber("marker_diameter_mm", v)}
            />
          </Field>
          <Field label="Minimum margin">
            <NumberInput min={0} unit="px" value={options.margin_px} onValueChange={(v) => setNumber("margin_px", Math.round(v))} />
          </Field>
        </div>
      </fieldset>
      {error && <Callout tone="error">{error}</Callout>}
      {generated && <Downloads generated={generated} />}
    </Dialog>
  );
}
