# Detection Quality & Rejection

`ringgrid` is built for **sensor calibration**, where one wrong correspondence can
poison an entire bundle adjustment. Detection is therefore **precision-first**: it
would rather drop a marginal marker than emit a geometrically impossible one. The
contract for downstream consumers is simple — **every `DetectedMarker` with
`Some(id)` is a trusted, lattice-consistent correspondence** carrying a
`board_xy_mm` ↔ `center` pair you can hand straight to a calibrator.

## The geometric verification gate

The decisive precision stage runs after the final homography, over **all** decoded
markers (including those filled in by completion). It removes any marker the hex
lattice judges geometrically inconsistent, using two complementary, hex-aware
tests and rejecting on their **union**:

1. **Local lattice-midpoint test (primary).** Each marker's *deviation from the
   final homography* (its center minus its cell's projection) is compared with
   the average midpoint of its opposite lattice neighbors' deviations. The
   homography's own curvature cancels — a raw neighbor midpoint is exact only
   under an affine map and misses by 2–3 px for large rings under strong
   perspective — so what remains is the curvature of smooth lens distortion,
   a few tenths of a pixel, while a marker in the wrong board cell sits a full
   pitch (tens of pixels) away. The test runs on the markers the global test
   keeps, so a gross blunder does not drag its correct neighbors over the
   threshold. A marker without an opposite neighbor pair (a corner of the
   visible patch) is tested one-sided, against `2·f(c+d) − f(c+2d)`. Without a
   homography it compares the centers themselves. A labeled marker neither
   variant reaches (no labeled neighbor) is removed when the global threshold
   has grown past half a cell pitch — under strong uncorrected distortion the
   global test alone cannot tell its cell from the next.

2. **Global final-H reprojection test (backstop).** Each marker's board position
   is projected through the final homography and compared with its center. This
   catches **boundary** markers that lack a complete neighbor pair for the local
   test (a false detection on the board edge is invisible to the local test but
   projects far from its claimed board cell). An ellipse-shape test runs with
   it: a marker whose outer ellipse's axis ratio departs from the one the
   homography predicts at its cell (a ring cut by the border or merged with a
   neighbor) is removed.

With photometric center refinement the gate runs after the photometric stage,
on the refined centers, against the homography that best fits the photometric
centers (an edge-fitted center of a blurred, eccentric ring is biased by
2–3 px; its photometric center is not).

### Recall-safe adaptive thresholds

A *fixed* pixel threshold cannot serve both regimes: true-marker residuals are
~0.1 px on a clean board but rise to ~1.4 px (peripherally higher) on a
lens-distorted board imaged without a camera model, because a single global
homography is a poor model for distortion. So each test's threshold **adapts** to
the observed inlier-residual distribution:

```text
threshold = max(floor, median + k · 1.4826 · MAD)
```

The robust `median + k·MAD` term only ever *raises* the threshold, so the floor
dominates on clean boards while distorted boards auto-loosen. The local floor is a
couple of pixels (far below a wrong-cell residual); the global floor is a small
multiple of the RANSAC inlier band. The result is recall-safe in the clean,
distorted-without-camera-model, and external-camera regimes alike, while a
wrong-cell or gross-blunder marker exceeds *both* thresholds by a wide margin and
is removed.

### Configuration

The gate is **on by default**. It exposes a single switch:

```rust
use ringgrid::{DetectConfig, Detector};

let mut config = DetectConfig::default();
// Precision-first (default): inconsistent markers are removed.
assert!(config.advanced.geometric_verify);

// Opt out to receive every decoded marker and apply your own filtering.
config.advanced.geometric_verify = false;
let detector = Detector::with_config(config);
```

There are no per-scene thresholds to tune — the adaptive design removes that
burden. If you disable the gate, inspect `MarkerDiagnostics::fit.h_reproj_err_px`
(the working-frame reprojection residual the gate computes for every decoded
marker) to filter on your own terms.

## Other rejection stages

The gate is the final guarantee, layered on top of earlier per-marker gates:

- **Decode gate** — Hamming distance, margin, contrast, and confidence thresholds
  on the 16-sector code.
- **Fit gates** — RANSAC inlier ratios, RMS residuals, and angular-coverage
  requirements on the outer and inner ellipse fits.
- **Global RANSAC homography filter** — drops markers that do not agree with the
  dominant board-to-image homography.
- **ID correction** — hex-neighbor consensus that clears or corrects IDs that
  contradict their neighborhood (see [ID Correction](detection-pipeline/id-correction.md)).
- **Axis-ratio consistency** — removes markers whose inner/outer ellipse ratio is
  a strong outlier (a sign the inner ring was fitted as the outer).

## Using ringgrid for calibration

A typical calibration consumer needs only the trusted correspondences:

```rust
use ringgrid::{TargetLayout, Detector};
use std::path::Path;

let target = TargetLayout::from_json_file(Path::new("target.json"))?;
let image = image::open("photo.png")?.to_luma8();

let detector = Detector::new(target);
let result = detector.detect(&image).unwrap(); // 0.8: detect* return Result<_, DetectError>

// Every marker with an id is lattice-consistent: a board↔image correspondence.
// Feed these (board_mm, image_px) pairs straight to your calibrator.
let correspondences: Vec<([f64; 2], [f64; 2])> = result
    .detected_markers
    .iter()
    .filter_map(|m| Some((m.board_xy_mm?, m.center)))
    .collect();
```

Because the gate runs by default, you do **not** need to post-filter on confidence
or reprojection error for precision — the markers that survive are the ones the
lattice vouches for. Keep the gate on for calibration; turn it off only when you
want raw detections for your own analysis.
