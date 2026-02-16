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

### `ringgrid detect` -- Detect markers in an image

This is the primary command. It loads an image, runs the detection pipeline, and writes
results as JSON.

**Required arguments:**

| Flag | Description |
|---|---|
| `--image <path>` | Path to the input image file. |
| `--out <path>` | Path to write detection results (JSON). |

**Board target:**

| Flag | Default | Description |
|---|---|---|
| `--target <path>` | built-in board | Path to a board layout JSON file. When omitted, uses the built-in default 203-marker hex board. |

**Marker scale:**

| Flag | Default | Description |
|---|---|---|
| `--marker-diameter <px>` | -- | Fixed marker outer diameter in pixels (legacy mode). Overrides min/max range. |
| `--marker-diameter-min <px>` | 20.0 | Minimum marker outer diameter for scale search. |
| `--marker-diameter-max <px>` | 56.0 | Maximum marker outer diameter for scale search. |

When `--marker-diameter` is set, it locks the detector to a single scale instead of
searching a range. This is a legacy compatibility path; prefer the min/max range for
new workflows.

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

All four intrinsic parameters (`fx`, `fy`, `cx`, `cy`) must be provided together. The
distortion coefficients are optional and default to zero.

| Flag | Default | Description |
|---|---|---|
| `--cam-fx <f>` | -- | Focal length x (pixels). |
| `--cam-fy <f>` | -- | Focal length y (pixels). |
| `--cam-cx <f>` | -- | Principal point x (pixels). |
| `--cam-cy <f>` | -- | Principal point y (pixels). |
| `--cam-k1 <f>` | 0.0 | Radial distortion k1. |
| `--cam-k2 <f>` | 0.0 | Radial distortion k2. |
| `--cam-p1 <f>` | 0.0 | Tangential distortion p1. |
| `--cam-p2 <f>` | 0.0 | Tangential distortion p2. |
| `--cam-k3 <f>` | 0.0 | Radial distortion k3. |

Camera intrinsics and `--self-undistort` are **mutually exclusive**. Providing both
will produce an error.

When camera intrinsics are provided, the detector runs a two-pass pipeline: pass 1
without distortion mapping to find initial markers, then pass 2 with the camera model
applied.

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

Using a custom board specification:

```bash
ringgrid detect \
    --target board_spec.json \
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

### `ringgrid codebook-info` -- Print codebook statistics

Prints information about the embedded 16-bit codebook: number of codewords, minimum
cyclic Hamming distance, and generator seed.

```bash
ringgrid codebook-info
```

Example output:

```
ringgrid embedded codebook
  bits per codeword:    16
  number of codewords:  893
  min cyclic Hamming:   3
  generator seed:       1
  first codeword:       0x0007
  last codeword:        0xFFE0
```

### `ringgrid board-info` -- Print default board specification

Prints summary information about the built-in default board layout: marker count,
pitch, rows, columns, and spatial extent.

```bash
ringgrid board-info
```

### `ringgrid decode-test` -- Decode a 16-bit word

Tests a hex word against the embedded codebook and prints the best match with
confidence metrics. Useful for debugging code sampling issues.

```bash
ringgrid decode-test --word 0xABCD
```

Example output:

```
Input word:   0xABCD (binary: 1010101111001101)
Best match:
  id:         42
  codeword:   0xABC5
  rotation:   0 sectors
  distance:   1 bits
  margin:     2 bits
  confidence: 0.875
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

The detection result JSON contains the detected markers, homography statistics, and
optionally the camera model used. See the [Output Format](./output-format.md) chapter
for the full schema.

## Source Files

- CLI implementation: `crates/ringgrid-cli/src/main.rs`
