# Adaptive Scale Detection

Adaptive scale detection handles images where marker diameters vary widely (for
example near/far perspective or mixed focal lengths).

## Why Use It

A single marker-size prior can under-detect:

- very small markers (proposal/search window too large or weak)
- very large markers (proposal/search window too small)

Adaptive mode runs multiple scale tiers and merges results with size-aware dedup.

## API Entry Points

```rust
use ringgrid::{BoardLayout, Detector, ScaleTiers};
use std::path::Path;

let board = BoardLayout::from_json_file(Path::new("target.json"))?;
let detector = Detector::new(board);
let image = image::open("photo.png")?.to_luma8();

// 1) Automatic probe + auto-tier selection
let r1 = detector.detect_adaptive(&image);

// 2) Optional nominal size hint (px) -> 2-tier bracket around hint
let r2 = detector.detect_adaptive_with_hint(&image, Some(32.0));

// 3) Explicit manual tiers
let tiers = ScaleTiers::four_tier_wide();
let r3 = detector.detect_multiscale(&image, &tiers);
# let _ = (r1, r2, r3);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Tier Presets

| Preset | Tiers | Diameter range | Typical use |
|---|---|---:|---|
| `ScaleTiers::four_tier_wide()` | 4 | 8–220 px | Unknown or extreme scale variation |
| `ScaleTiers::two_tier_standard()` | 2 | 14–100 px | Moderate variation, lower runtime |
| `ScaleTiers::single(prior)` | 1 | custom | Single-pass equivalent |

## How `detect_adaptive` Chooses Tiers

1. Runs a lightweight scale probe to estimate dominant code-band radii.
2. Builds one or more tiers from probe clusters (`ScaleTiers::from_detected_radii`).
3. Falls back to `ScaleTiers::four_tier_wide()` if probe signal is unavailable.
4. Runs one full detect pass per tier.
5. Merges all markers with size-consistency-aware dedup.
6. Runs global filter + completion + final homography refit once on merged results.

## When To Prefer Each Method

- `detect()`:
  - fastest and simplest
  - best when marker size range is relatively tight
- `detect_adaptive()`:
  - good default when scale is unknown
  - robust across mixed near/far markers
- `detect_adaptive_with_hint(..., Some(d))`:
  - use when you have an approximate diameter
  - skips probe and narrows search to a focused two-tier bracket
- `detect_multiscale(...)`:
  - use when you need explicit control over tiers
  - useful for reproducible experiments and benchmarks

## CLI Status

Adaptive entry points are currently Rust API methods. The CLI `ringgrid detect`
uses the regular config-driven detection flow (`detect` / `detect_with_mapper`).

## Source

- `crates/ringgrid/src/api.rs`
- `crates/ringgrid/src/pipeline/run.rs`
- `crates/ringgrid/src/detector/config.rs`
