# Proposal Diagnostics

The proposal stage can be run standalone to inspect candidate centers and the
vote accumulator heatmap without running the full detection pipeline.

## Python API

### Raw mode (no scale prior)

```python
import ringgrid
from ringgrid import viz

# Default ProposalConfig
proposals = ringgrid.propose("photo.png")
result = ringgrid.propose_with_heatmap("photo.png")

print(len(result.proposals))
print(result.heatmap.shape)  # (H, W), float32

viz.plot_proposal_diagnostics(
    image="photo.png",
    diagnostics=result,
    out="proposal_overlay.png",
)
```

### Detector-aware mode (scale-tuned)

When a board layout and marker diameter are available, the proposal config is
derived from `MarkerScalePrior` for tighter search windows:

```python
board = ringgrid.BoardLayout.default()
cfg = ringgrid.DetectConfig(board)
detector = ringgrid.Detector(cfg)

result = detector.propose_with_heatmap("photo.png")
```

Or via the module-level convenience function:

```python
result = ringgrid.propose_with_heatmap(
    "photo.png",
    target=board,
    marker_diameter=32.0,
)
```

### Custom ProposalConfig

```python
config = ringgrid.ProposalConfig(
    r_min=5.0,
    r_max=40.0,
    min_distance=15.0,
    edge_thinning=True,
)
result = ringgrid.propose_with_heatmap("photo.png", config=config)
```

## Rust API

The standalone `proposal` module provides entry points that work on any
grayscale image, independent of the ringgrid detection pipeline:

```rust
use ringgrid::proposal::{find_ellipse_centers, find_ellipse_centers_with_heatmap, ProposalConfig};

let config = ProposalConfig {
    r_min: 5.0,
    r_max: 30.0,
    min_distance: 15.0,
    ..Default::default()
};

// Proposals only
let proposals = find_ellipse_centers(&gray, &config);

// Proposals + heatmap
let result = find_ellipse_centers_with_heatmap(&gray, &config);
println!("heatmap size: {:?}", result.image_size);
```

## ProposalResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `proposals` | `list[Proposal]` | Detected center candidates with `(x, y, score)` |
| `heatmap` | `np.ndarray` (H, W), float32 | Post-smoothed vote accumulator used for NMS |
| `image_size` | `[int, int]` | `[width, height]` of the input image |

The `heatmap` is the Gaussian-smoothed vote accumulator that the proposal stage
uses for thresholding and NMS. It is useful for understanding where the detector
sees radial symmetry evidence.

## ProposalConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r_min` | 3.0 | Minimum voting radius (pixels) |
| `r_max` | 12.0 | Maximum voting radius (pixels) |
| `min_distance` | 10.0 | Minimum distance between output proposals (pixels) |
| `grad_threshold` | 0.05 | Gradient magnitude threshold (fraction of max) |
| `min_vote_frac` | 0.1 | Minimum accumulator peak (fraction of max) |
| `accum_sigma` | 2.0 | Gaussian smoothing sigma |
| `edge_thinning` | true | Apply Canny-style gradient NMS before voting |
| `max_candidates` | None | Optional hard cap on proposals |

## Visualization Tool

The repo includes a CLI tool for proposal visualization with optional
ground-truth recall overlay:

```bash
python tools/plot_proposal.py \
    --image tools/out/synth_001/img_0000.png \
    --gt tools/out/synth_001/gt_0000.json \
    --out tools/out/synth_001/proposals_0000.png
```

Detector-aware mode (with marker scale prior):

```bash
python tools/plot_proposal.py \
    --image testdata/target_3_split_00.png \
    --target tools/out/target_faststart/board_spec.json \
    --marker-diameter 32.0 \
    --out proposals_overlay.png
```

## Backward Compatibility

The Python class `ProposalDiagnostics` is a deprecated alias for
`ProposalResult`. Existing code using `ProposalDiagnostics` will continue to
work.
