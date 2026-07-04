# Target JSON (schema v5)

`ringgrid.target.v5` is the canonical, compositional target schema. It mirrors
the [target model](target-model.md) one-to-one: a top-level object with
`lattice`, `marker`, `coding`, and optional `fiducials` sections. Loaders also
accept the legacy flat `ringgrid.target.v4` schema and migrate it on the fly;
writers always emit v5.

## Annotated example — plain rect with origin dots

Produced by `TargetLayout::rect_24x24()` (abbreviated to a 4×4 lattice with
two dots so the shape is easy to read):

```jsonc
{
  "schema": "ringgrid.target.v5",         // schema tag; dispatched before full parse
  "name": "ringgrid_rect_r4_c4_p14.000_o5.600_i2.800",  // human-readable name
  "lattice": {
    "kind": "rect",                       // "rect" | "hex"
    "rows": 4,
    "cols": 4,
    "pitch_mm": 14.0                       // center-to-center spacing
  },
  "marker": {
    "outer_radius_mm": 5.6,               // annulus bounds (plain) / centerline (coded)
    "inner_radius_mm": 2.8
  },
  "coding": {
    "kind": "plain"                       // plain annulus, no identity code
  },
  "fiducials": {                          // optional; present only when defined
    "dot_radius_mm": 1.4,
    "dots_mm": [                          // dark dots in board mm, break lattice symmetry
      [21.0, 21.0],
      [7.0, 21.0]
    ]
  }
}
```

## Annotated example — coded hex

Produced by `TargetLayout::default_hex()` (shown here for a small 3-row lattice):

```jsonc
{
  "schema": "ringgrid.target.v5",
  "name": "ringgrid_hex_r3_c4_p8.000_o4.800_i3.200_w1.152",
  "lattice": {
    "kind": "hex",
    "rows": 3,
    "long_row_cols": 4,                   // markers in the long (even-offset) rows
    "pitch_mm": 8.0
  },
  "marker": {
    "outer_radius_mm": 4.8,               // stroked-ring centerline radii
    "inner_radius_mm": 3.2
  },
  "coding": {
    "kind": "coded16",
    "ring_width_mm": 1.152                // stroke width of the inner and outer rings
    // "id_assignment": [ ... ]           // optional; omitted ⇒ sequential 0,1,2,...
  }
  // no "fiducials" — coded markers anchor themselves via decoded IDs
}
```

## Field reference

### Top level

| Field | Type | Notes |
|---|---|---|
| `schema` | string | `"ringgrid.target.v5"` (or legacy `"ringgrid.target.v4"` on input). |
| `name` | string | Non-empty. Presets and CLI use a deterministic geometry-derived name. |
| `lattice` | object | Tagged by `kind`. |
| `marker` | object | Ring radii. |
| `coding` | object | Tagged by `kind`. |
| `fiducials` | object? | Omitted when the target defines no origin dots. |

Unknown top-level fields are rejected (`deny_unknown_fields`).

### `lattice`

| `kind` | Fields |
|---|---|
| `"hex"` | `rows`, `long_row_cols`, `pitch_mm` |
| `"rect"` | `rows`, `cols`, `pitch_mm` |

### `marker`

`outer_radius_mm`, `inner_radius_mm` (both mm, `inner < outer`).

### `coding`

| `kind` | Fields |
|---|---|
| `"coded16"` | `ring_width_mm`, optional `id_assignment` (array of codebook IDs, one per cell in generation order) |
| `"plain"` | none |

A sequential `id_assignment` (`0, 1, 2, …`) is normalized back to the implicit
form and omitted on write.

### `fiducials`

`dot_radius_mm`, `dots_mm` (array of `[x_mm, y_mm]` dot centers). See
[Origin Fiducials](origin-fiducials.md).

## v4 auto-migration

The pre-0.8 flat schema described a hex coded target with top-level
`pitch_mm`, `rows`, `long_row_cols`, `marker_outer_radius_mm`,
`marker_inner_radius_mm`, `marker_ring_width_mm`, and optional `id_assignment`:

```json
{
  "schema": "ringgrid.target.v4",
  "name": "legacy",
  "pitch_mm": 8.0,
  "rows": 15,
  "long_row_cols": 14,
  "marker_outer_radius_mm": 4.8,
  "marker_inner_radius_mm": 3.2,
  "marker_ring_width_mm": 1.152
}
```

Every loader — `TargetLayout::from_json_str` / `from_json_file`, the CLI
`--target` flag, the CLI `gen-target from-spec`, and the Python / WASM detector
constructors — accepts this and migrates it to a `Hex` + `Coded16` layout.
Writers only ever emit v5, so re-serializing a migrated target upgrades it:

```rust
let target = ringgrid::TargetLayout::from_json_str(v4_json)?; // accepts v4
let v5_json = target.to_json_string();                        // emits v5
```

The checked-in `tools/board/board_spec*.json` fixtures are still v4 and load
unchanged, including their optimized `id_assignment`. An unknown or unsupported
schema tag is rejected with `TargetValidationError::UnsupportedSchema`.

## Generating target JSON from the CLI

`gen-target` writes `target_spec.json` (v5) alongside printable SVG/PNG. It is a
subcommand family:

```bash
# Classic hex coded target
ringgrid gen-target hex \
  --pitch_mm 8 --rows 15 --long_row_cols 14 \
  --marker_outer_radius_mm 4.8 --marker_inner_radius_mm 3.2 \
  --marker_ring_width_mm 1.152 \
  --out_dir tools/out/target

# Rect plain target with origin dots
ringgrid gen-target rect \
  --pitch_mm 14 --rows 24 --cols 24 \
  --marker_outer_radius_mm 5.6 --marker_inner_radius_mm 2.8 \
  --dot_radius_mm 1.4 --dot_mm 161,161 --dot_mm 147,161 --dot_mm 161,175 \
  --out_dir tools/out/target

# A built-in preset
ringgrid gen-target preset default-hex --out_dir tools/out/target
ringgrid gen-target preset rect24x24   --out_dir tools/out/target

# Re-render (and upgrade) an existing spec, v5 or legacy v4
ringgrid gen-target from-spec --spec path/to/target_spec.json --out_dir tools/out/target
```

See [Target Generation](../target-generation.md) for the full flag reference and
the equivalent Rust/Python paths.

**Source:** `crates/ringgrid/src/target/schema.rs`
