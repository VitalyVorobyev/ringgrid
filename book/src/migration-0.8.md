# Migrating from 0.7 to 0.8

Release 0.8 introduces the [compositional target model](targets/target-model.md)
(hex/rect lattices, plain/coded markers, origin fiducials), retiers the public
API, and renames a handful of fields. This page lists the breaking changes per
interface with concrete before/after snippets. The pre-1.0 policy favors the
cleaner design over source stability, but most changes are shims or renames with
compatibility aliases, so migration is mechanical.

## At a glance

| Change | Rust | Python | CLI | WASM |
|---|---|---|---|---|
| `DetectConfig.board` → `.target` (`TargetLayout`) | ✓ | — (Python keeps `board`) | — | — |
| `detect*` return `Result<_, DetectError>` | ✓ | — (raises on error) | — | — |
| Diagnostics/codebook moved off the crate root | ✓ | — | — | — |
| `max_center_shift_px` → `max_correction_shift_px` (proj. center) | ✓ | ✓ | via `--config` | via overlay |
| `BoardLayout` deprecated (use `TargetLayout`) | ✓ | still v4-shaped | — | — |
| Target JSON v5 (v4 still accepted on input) | ✓ | ✓ | ✓ | ✓ |
| New result fields `grid_coord`, `board_frame` | ✓ | ✓ | ✓ | ✓ |
| `gen-target` is now a subcommand family | — | — | ✓ | — |

## Rust

### `DetectConfig.board` → `DetectConfig.target`

The config field and its builder are renamed, and constructors now take
`impl Into<TargetLayout>`. `From<BoardLayout> for TargetLayout` means existing
`BoardLayout` callers still compile:

```rust
// 0.7
let board = BoardLayout::from_json_file(Path::new("board_spec.json"))?;
let mut cfg = DetectConfig::from_target(board);
let target_ref = &cfg.board;

// 0.8
let target = TargetLayout::from_json_file(Path::new("target_spec.json"))?; // or a BoardLayout
let mut cfg = DetectConfig::from_target(target);
let target_ref = &cfg.target;
```

`DetectConfig::with_board` is renamed to `with_target`; the free proposal helpers
(`propose_with_marker_scale`, `propose_with_heatmap_and_marker_scale`) now take
`&TargetLayout`.

### `detect*` now return `Result<_, DetectError>`

Every `Detector::detect*` method returns `Result<DetectionResult, DetectError>`
(the mapper/diagnostics variants return the tuple wrapped in `Result`). It is
currently infallible for all built-in targets, so `?` or `.unwrap()` suffices:

```rust
// 0.7
let result = detector.detect(&image);

// 0.8
let result = detector.detect(&image)?;
```

### Public surface tiering

Opt-in diagnostics and codebook helpers moved off the crate root; the stable
`DetectionResult` and its field types stay at the root.

```rust
// 0.7
use ringgrid::{FitMetrics, DecodeMetrics, RansacStats, codebook_info, decode_word};

// 0.8
use ringgrid::diagnostics::{FitMetrics, DecodeMetrics, RansacStats, StageTimings};
use ringgrid::codebook::{codebook_info, decode_word, CodebookInfo, CodewordMatch};
```

Moved to `ringgrid::diagnostics`: `DetectionDiagnostics`, `MarkerDiagnostics`,
`FitMetrics`, `DecodeMetrics`, `DetectionSource`, `InnerFitReason`,
`InnerFitStatus`, `RansacStats`, `StageTimings`. Moved to `ringgrid::codebook`:
`CodebookInfo`, `CodewordMatch`, `codebook_info`, `decode_word`.

### `max_center_shift_px` → `max_correction_shift_px`

`ProjectiveCenterConfig`'s shift gate is renamed to disambiguate it from the
unrelated inner-fit gate `InnerFitConfig::max_center_shift_px` (which keeps its
name). The type is now `Option<f64>`, and **`None` means "auto"** — the gate uses
the nominal marker diameter derived from the active `MarkerScalePrior` at the
point of use, so an explicit value now survives target re-derivation instead of
being silently clobbered:

```rust
// 0.7
cfg.advanced.projective_center.max_center_shift_px = 12.0;

// 0.8
cfg.advanced.projective_center.max_correction_shift_px = Some(12.0); // explicit
cfg.advanced.projective_center.max_correction_shift_px = None;       // "auto" (default)
```

A `serde` alias keeps 0.7.x JSON configs loading (the old key deserializes into
the new field).

### `BoardLayout` deprecation

`BoardLayout`, `BoardMarker`, `BoardLayoutValidationError`, and
`BoardLayoutLoadError` are deprecated (a thin facade over the target module) and
will be removed after 0.8. Prefer `TargetLayout` and the target error types
(`TargetValidationError`, `TargetLoadError`). Until then the facade is fully
functional and geometry-identical to `TargetLayout::default_hex()`.

### New result fields

`DetectedMarker` gains `grid_coord: Option<[i32; 2]>` (the lattice cell — the
only key for plain targets) and `DetectionResult` gains
`board_frame: Option<BoardFrame>` (`Absolute` | `RelativeCanonical`). Both types
are `#[non_exhaustive]`, so match/construct through their public fields as before.

### Target JSON v4 → v5

Loaders accept both schemas; writers emit v5. Existing v4 files (including
`tools/board/board_spec*.json`) keep loading and upgrade on re-serialize. See
[Target JSON (schema v5)](targets/target-json-v5.md).

## Python

The Python `BoardLayout` class and the `Detector(board)` / `Detector.from_board`
construction path are **unchanged** — they stay v4-shaped (hex coded targets)
for the 0.8 cycle, so existing Python code needs no construction changes.

What changed:

- **Results.** `DetectedMarker` gains `grid_coord: list[int] | None`;
  `DetectionResult` gains `board_frame: str | None` (`"absolute"` /
  `"relative_canonical"`).
- **Projective-center rename.** The `ProjectiveCenterConfig` dataclass field is
  `max_correction_shift_px: float | None` (`None` = auto), matching Rust. Configs
  round-tripped through `to_dict()` / config overlays keep loading the old
  `max_center_shift_px` key. The separate `InnerFitConfig.max_center_shift_px`
  is unchanged.

```python
# 0.8 — read the new fields
result = detector.detect(image)
print(result.board_frame)                 # "absolute" | "relative_canonical" | None
for m in result.detected_markers:
    print(m.id, m.grid_coord)             # id is None for plain targets
```

`Detector.detect` still returns a `DetectionResult` (Rust's `DetectError` surfaces
as an exception; detection is currently infallible for built-in targets).

## CLI

### `gen-target` is now a subcommand family

The old flat `gen-target` flags are replaced by four subcommands. Output is
`target_spec.json` (v5):

```bash
# 0.7
ringgrid gen-target --pitch_mm 8 --rows 15 --long_row_cols 14 \
  --marker_outer_radius_mm 4.8 --marker_inner_radius_mm 3.2 --out_dir out

# 0.8 — classic hex now lives under the `hex` subcommand
ringgrid gen-target hex --pitch_mm 8 --rows 15 --long_row_cols 14 \
  --marker_outer_radius_mm 4.8 --marker_inner_radius_mm 3.2 \
  --marker_ring_width_mm 1.152 --out_dir out
```

New subcommands: `rect` (plain rings, optional `--dot_mm`/`--dot_radius_mm`),
`preset` (`default-hex`, `rect24x24`), and `from-spec` (render any target JSON).

### `detect`

`--target` accepts v5 and legacy v4 JSON. The `detect.json` output gains
per-marker `grid_coord` and a top-level `board_frame` for grid-labeled runs.
Pre-0.8 `--config` overlays that name `max_center_shift_px` still load — the key
is normalized to `max_correction_shift_px` before merging.

## WASM

`RinggridDetector.new(board_json)` / `.with_config(board_json, config_json)`
keep their signatures; the JSON argument accepts both v5 and legacy v4, so no
call-site change is needed. `default_board_json()` now emits v5 (consumers
already accept both). Detection result JSON gains `grid_coord` and `board_frame`
like the CLI. Config overlays via `update_config` accept the pre-0.8
`max_center_shift_px` key.

## See also

- [The Compositional Target Model](targets/target-model.md)
- [Target JSON (schema v5)](targets/target-json-v5.md)
- [Origin Fiducials](targets/origin-fiducials.md)
- [Full 0.8 changelog](https://github.com/VitalyVorobyev/ringgrid/blob/main/CHANGELOG.md)
