# CLI Guide

The `ringgrid` command-line tool provides access to the ring marker detection pipeline
from the terminal. It is built from the `ringgrid-cli` crate.

## Installation

Install from the workspace:

```bash
cargo install ringgrid-cli
```

Or build from source:

```bash
cargo build --release -p ringgrid-cli
```

The binary is named `ringgrid`.

## Commands

### `ringgrid gen-target` -- Generate canonical target JSON + printable SVG/PNG

`gen-target` is a subcommand family. Each subcommand builds a `TargetLayout` and
writes `target_spec.json` (schema `ringgrid.target.v5`), `<basename>.svg`, and
`<basename>.png` to `--out_dir`.

| Subcommand | Target |
|---|---|
| `hex` | Hex lattice of 16-sector coded rings (the classic ringgrid target). |
| `rect` | Rectangular lattice of plain (uncoded) rings, optionally with origin dots. |
| `preset` | A built-in preset: `default-hex` or `isra24x24`. |
| `from-spec` | Render from an existing target spec JSON (v5, or legacy v4). |

Common examples:

```bash
# Classic hex coded target
ringgrid gen-target hex \
    --pitch_mm 8 --rows 15 --long_row_cols 14 \
    --marker_outer_radius_mm 4.8 --marker_inner_radius_mm 3.2 \
    --marker_ring_width_mm 1.152 \
    --out_dir tools/out/target

# Rectangular plain target with three origin dots
ringgrid gen-target rect \
    --pitch_mm 14 --rows 24 --cols 24 \
    --marker_outer_radius_mm 5.6 --marker_inner_radius_mm 2.8 \
    --dot_radius_mm 1.4 --dot_mm 161,161 --dot_mm 147,161 --dot_mm 161,175 \
    --out_dir tools/out/target_rect

# A built-in preset
ringgrid gen-target preset isra24x24 --out_dir tools/out/target

# Re-render (and upgrade to v5) an existing spec
ringgrid gen-target from-spec --spec tools/out/target/target_spec.json --out_dir tools/out/target2
```

Underscore flag names are primary; hyphenated aliases such as `--pitch-mm`,
`--long-row-cols`, and `--out-dir` are also accepted. For the full
per-subcommand flag reference and the equivalent Rust API, see
[Target Generation](target-generation.md).

### `ringgrid detect` -- Detect markers in an image

This is the primary command. It loads an image, runs the detection pipeline, and writes
results as JSON.

**Required arguments:**

| Flag | Description |
|---|---|
| `--image <path>` | Path to the input image file. |
| `--out <path>` | Path to write detection results (JSON). |

**Output and diagnostics:**

| Flag | Default | Description |
|---|---|---|
| `--include-proposals` | false | Add pass-1 proposal diagnostics to the output JSON as top-level `proposal_frame`, `proposal_count`, and `proposals` fields. |

**Board target:**

| Flag | Default | Description |
|---|---|---|
| `--target <path>` | built-in board | Path to a target spec JSON file (`ringgrid.target.v5`, or legacy `v4`). When omitted, uses the built-in default 203-marker hex board. |

**Marker scale:**

| Flag | Default | Description |
|---|---|---|
| `--marker-diameter <px>` | -- | Fixed marker outer diameter in pixels (legacy mode). Overrides min/max range. |
| `--marker-diameter-min <px>` | unset | Minimum marker outer diameter for scale search. |
| `--marker-diameter-max <px>` | unset | Maximum marker outer diameter for scale search. |

When `--marker-diameter` is set, it locks the detector to a single scale instead of
searching a range. This is a legacy compatibility path; prefer the min/max range for
new workflows.

When both min/max are omitted, the detector uses the library default prior
(`14-66 px`).
If only one bound is provided, the missing side uses the legacy compatibility
fallback (`20` for min, `56` for max).

**RANSAC homography:**

| Flag | Default | Description |
|---|---|---|
| `--ransac-thresh-px <px>` | 5.0 | Inlier threshold in pixels. |
| `--ransac-iters <n>` | 2000 | Maximum iterations. |
| `--no-global-filter` | false | Disable the global homography filter entirely. |

**Homography-guided completion:**

| Flag | Default | Description |
|---|---|---|
| `--no-complete` | false | Disable completion (fitting at H-projected missing IDs). |
| `--complete-reproj-gate <px>` | 3.0 | Reprojection error gate for accepting completed markers. |
| `--complete-min-conf <f>` | 0.45 | Minimum fit confidence in [0, 1] for completed markers. |
| `--complete-roi-radius <px>` | auto | ROI radius for edge sampling during completion. Defaults to a value derived from the nominal marker diameter. |

**Center refinement (projective center):**

| Flag | Default | Description |
|---|---|---|
| `--circle-refine-method <m>` | projective-center | Center correction method: `none` or `projective-center`. |
| `--proj-center-max-shift-px <px>` | auto | Maximum allowed correction shift. Defaults to a value derived from nominal marker diameter. |
| `--proj-center-max-residual <f>` | 0.25 | Reject corrections with residual above this. |
| `--proj-center-min-eig-sep <f>` | 1e-6 | Reject corrections with eigenvalue separation below this. |

**Self-undistort:**

| Flag | Default | Description |
|---|---|---|
| `--self-undistort` | false | Estimate a 1-parameter division-model distortion from detected markers, then re-run detection. |
| `--self-undistort-lambda-min <f>` | -8e-7 | Lambda search lower bound. |
| `--self-undistort-lambda-max <f>` | 8e-7 | Lambda search upper bound. |
| `--self-undistort-min-markers <n>` | 6 | Minimum number of markers with inner+outer edge points required for estimation. |

**Camera intrinsics:**

You can provide an external camera model either inline via `--cam-*` or from a
JSON file via `--calibration <file.json>`.

| Flag | Default | Description |
|---|---|---|
| `--calibration <file.json>` | -- | Load a Brown-Conrady `CameraModel` from JSON. Accepts either direct `{ "intrinsics": ..., "distortion": ... }` or wrapped `{ "camera": { ... } }` shapes. |
| `--cam-fx <f>` | -- | Focal length x (pixels). |
| `--cam-fy <f>` | -- | Focal length y (pixels). |
| `--cam-cx <f>` | -- | Principal point x (pixels). |
| `--cam-cy <f>` | -- | Principal point y (pixels). |
| `--cam-k1 <f>` | 0.0 | Radial distortion k1. |
| `--cam-k2 <f>` | 0.0 | Radial distortion k2. |
| `--cam-p1 <f>` | 0.0 | Tangential distortion p1. |
| `--cam-p2 <f>` | 0.0 | Tangential distortion p2. |
| `--cam-k3 <f>` | 0.0 | Radial distortion k3. |

For inline parameters, all four intrinsic parameters (`fx`, `fy`, `cx`, `cy`)
must be provided together. The distortion coefficients are optional and default
to zero.

`--calibration` and inline `--cam-*` parameters are **mutually exclusive**.
Any external camera model and `--self-undistort` are also **mutually exclusive**.
Providing both will produce an error.

When a camera model is provided, the detector runs a two-pass pipeline: pass 1
without distortion mapping to find initial markers, then pass 2 with the camera
model applied.

### Usage Examples

Basic detection with default settings:

```bash
ringgrid detect --image photo.png --out result.json
```

Specifying the expected marker size range:

```bash
ringgrid detect \
    --image photo.png \
    --out result.json \
    --marker-diameter-min 20 \
    --marker-diameter-max 56
```

Using a custom target specification:

```bash
ringgrid detect \
    --target target_spec.json \
    --image photo.png \
    --out result.json
```

With camera intrinsics and distortion:

```bash
ringgrid detect \
    --image photo.png \
    --out result.json \
    --cam-fx 900 --cam-fy 900 --cam-cx 640 --cam-cy 480 \
    --cam-k1 -0.15 --cam-k2 0.05
```

With a calibration JSON file:

```bash
ringgrid detect \
    --image photo.png \
    --out result.json \
    --calibration camera_model.json
```

With self-undistort estimation:

```bash
ringgrid detect \
    --image photo.png \
    --out result.json \
    --self-undistort
```

Disabling completion and global filter (raw fit-decode output only):

```bash
ringgrid detect \
    --image photo.png \
    --out result.json \
    --no-global-filter \
    --no-complete
```

## Adaptive Scale Status

Adaptive multi-scale entry points are currently exposed via Rust and Python APIs:

- `Detector::detect_adaptive`
- `Detector::detect_adaptive_with_hint`
- `Detector::detect_multiscale`

Python bindings expose the same concepts on `ringgrid.Detector`:

- `detect_adaptive(image, nominal_diameter_px=None)` (canonical)
- `detect_adaptive_with_hint(image, nominal_diameter_px=...)` (compatibility alias)
- `detect_multiscale(image, tiers)`

CLI `ringgrid detect` uses the regular config-driven detect flow.

### `ringgrid codebook-info` -- Print codebook statistics

Prints information about the embedded 16-bit codebook profiles. The output shows
the shipped default `base` profile plus the opt-in `extended` profile.

`ringgrid detect` continues to use `base` unless a config file sets
`decode.codebook_profile` to `extended`.

```bash
ringgrid codebook-info
```

Example output:

```text
ringgrid embedded codebook profiles
  default profile:      base
  base:
    bits per codeword:    16
    number of codewords:  893
    min cyclic Hamming:   2
    generator seed:       1
    first codeword:       0x035D
    last codeword:        0x0E63
  extended:
    bits per codeword:    16
    number of codewords:  2180
    min cyclic Hamming:   1
    generator seed:       1
    first codeword:       0x035D
    last codeword:        0x2CD3
```

### `ringgrid board-info` -- Print default board specification

Prints summary information about the built-in default board layout: marker count,
pitch, rows, columns, and spatial extent.

```bash
ringgrid board-info
```

### `ringgrid decode-test` -- Decode a 16-bit word

Tests a hex word against the selected embedded codebook profile and prints the
best match with confidence metrics. Useful for debugging code sampling issues.

```bash
ringgrid decode-test --word 0x035D
# or:
ringgrid decode-test --word 0x0001 --profile extended
```

Example output:

```text
Input word:   0x035D (binary: 0000001101011101)
Profile:      base
Best match:
  id:         0
  codeword:   0x035D
  rotation:   0 sectors
  distance:   0 bits
  margin:     2 bits
  confidence: 1.000
```

## Logging

ringgrid uses the `tracing` crate for structured logging. Control verbosity with the
`RUST_LOG` environment variable:

```bash
# Default level (info) -- shows summary statistics
ringgrid detect --image photo.png --out result.json

# Debug level -- shows per-stage diagnostics
RUST_LOG=debug ringgrid detect --image photo.png --out result.json

# Trace level -- shows detailed per-marker information
RUST_LOG=trace ringgrid detect --image photo.png --out result.json
```

At the default `info` level, the detector logs:

- Image dimensions
- Board layout loaded (name, marker count)
- Number of detected markers (total and with decoded IDs)
- Homography statistics (inlier count, mean and p95 reprojection error)
- Self-undistort results when enabled (lambda, objective improvement, marker count)
- Output file path

## Output Format

`ringgrid detect` writes the serialized `DetectionResult` fields at the top
level:

- `detected_markers`
- `center_frame`
- `homography_frame`
- `image_size`
- optional `homography` and `self_undistort`
- a nested `diagnostics` object carrying per-marker algorithm internals
  (`diagnostics.markers`) and homography RANSAC statistics (`diagnostics.ransac`)

The CLI may add extra top-level fields:

- `camera` when a camera model was supplied
- `proposal_frame`, `proposal_count`, and `proposals` when
  `--include-proposals` is enabled

The full file shape, nested marker fields, and frame semantics are documented in
[Output Format](./output-format.md).

## Source Files

- CLI implementation: `crates/ringgrid-cli/src/main.rs`
