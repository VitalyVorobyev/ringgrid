# ringgrid-wasm

WebAssembly bindings for [ringgrid](https://github.com/VitalyVorobyev/ringgrid) — a pure-Rust detector for dense coded ring calibration targets.

## Building

```bash
wasm-pack build crates/ringgrid-wasm --target web --release
```

## Demo

Build the WASM package and serve from the repository root:

```bash
# From the repository root
wasm-pack build crates/ringgrid-wasm --target web --release
python3 -m http.server 8080
```

Open [http://localhost:8080/crates/ringgrid-wasm/demo/](http://localhost:8080/crates/ringgrid-wasm/demo/) in your browser.

> **Important:** The server must be started from the repository root (not from the `demo/` directory), so that the WASM package at `pkg/` and the test images at `testdata/` are both accessible.

The demo supports:

- **Image loading** — drag-and-drop or file picker, plus a built-in test image
- **Five detection modes** — `detect`, `detect_adaptive`, `detect_adaptive` with diameter hint, `detect_multiscale` with tier presets, and `propose_with_heatmap`
- **Config editing** — quick controls for marker scale, completion, center refinement, and proposal downscale, plus a full JSON config editor
- **Visualization** — outer/inner ellipses (confidence-colored), center dots, ID labels, optional edge point overlay, and proposal heatmap

## Usage

```js
import init, { RinggridDetector, default_board_json, default_config_json } from './pkg/ringgrid_wasm.js';

await init();

const boardJson = default_board_json();
const detector = new RinggridDetector(boardJson);

// From canvas ImageData (RGBA)
const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
const resultJson = detector.detect_rgba(imageData.data, canvas.width, canvas.height);
const result = JSON.parse(resultJson);

console.log(`Found ${result.detected_markers.length} markers`);
for (const m of result.detected_markers) {
    console.log(`  Marker ${m.id} at (${m.center[0].toFixed(1)}, ${m.center[1].toFixed(1)})`);
}
```

### Custom config

```js
// Get the default config, modify it, create detector with full config control
const configJson = default_config_json(boardJson);
const config = JSON.parse(configJson);
config.completion.enable = false;
config.marker_scale = { diameter_min_px: 20, diameter_max_px: 80 };
const detector = RinggridDetector.with_config(boardJson, JSON.stringify(config));

// Or apply a partial overlay to an existing detector
detector.update_config(JSON.stringify({ completion: { enable: true } }));
```

## API

### `RinggridDetector`

#### Constructors

| Method | Input | Output |
|--------|-------|--------|
| `new(board_json)` | Board layout JSON string | `RinggridDetector` |
| `with_marker_scale(board_json, min_px, max_px)` | Board + scale range | `RinggridDetector` |
| `with_marker_diameter(board_json, diameter_px)` | Board + diameter hint | `RinggridDetector` |
| `with_config(board_json, config_json)` | Board + full config JSON | `RinggridDetector` |

#### Config access

| Method | Input | Output |
|--------|-------|--------|
| `config_json()` | — | Full config JSON string |
| `update_config(overlay_json)` | Partial config JSON | — (mutates detector) |

#### Detection (grayscale)

| Method | Input | Output |
|--------|-------|--------|
| `detect(pixels, w, h)` | Grayscale `Uint8Array` | JSON `DetectionResult` |
| `detect_adaptive(pixels, w, h)` | Grayscale `Uint8Array` | JSON `DetectionResult` |
| `detect_adaptive_with_hint(pixels, w, h, diameter_px)` | Grayscale + hint | JSON `DetectionResult` |
| `detect_multiscale(pixels, w, h, tiers_json)` | Grayscale + tiers | JSON `DetectionResult` |

#### Detection (RGBA)

| Method | Input | Output |
|--------|-------|--------|
| `detect_rgba(pixels, w, h)` | RGBA `Uint8Array` | JSON `DetectionResult` |
| `detect_adaptive_rgba(pixels, w, h)` | RGBA `Uint8Array` | JSON `DetectionResult` |
| `detect_adaptive_with_hint_rgba(pixels, w, h, diameter_px)` | RGBA + hint | JSON `DetectionResult` |
| `detect_multiscale_rgba(pixels, w, h, tiers_json)` | RGBA + tiers | JSON `DetectionResult` |

#### Proposals

| Method | Input | Output |
|--------|-------|--------|
| `propose_with_heatmap(pixels, w, h)` | Grayscale `Uint8Array` | JSON proposals |
| `propose_with_heatmap_rgba(pixels, w, h)` | RGBA `Uint8Array` | JSON proposals |
| `heatmap_f32()` | — | `Float32Array` (row-major) |
| `heatmap_width()` / `heatmap_height()` | — | `number` |

### Free functions

| Function | Output |
|----------|--------|
| `default_board_json()` | Default board layout JSON string |
| `default_config_json(board_json)` | Default detection config for a board |
| `scale_tiers_four_tier_wide_json()` | Four-tier preset (8-220 px) |
| `scale_tiers_two_tier_standard_json()` | Two-tier preset (14-100 px) |
| `version()` | Package version string |

## Output format

Detection results are returned as JSON strings matching the Rust `DetectionResult` type.
Key fields:
- `detected_markers[].id` — codebook index (0-892)
- `detected_markers[].center` — `[x, y]` pixel coordinates
- `detected_markers[].confidence` — detection confidence (0-1)
- `detected_markers[].ellipse_outer` / `ellipse_inner` — fitted ellipse parameters (`{cx, cy, a, b, angle}`)
- `detected_markers[].edge_points_outer` / `edge_points_inner` — sampled edge points as `[[x, y], ...]`
- `detected_markers[].fit` — fit quality metrics (inlier ratios, RMS residuals, angular gaps)
- `detected_markers[].decode` — decode metrics (observed word, best ID, distance, margin)
- `homography` — 3x3 board-to-image homography matrix (if available)
- `ransac` — RANSAC stats (inlier count, threshold, mean/p95 error)

Heatmap is a row-major `Float32Array` of size `width * height`, representing the RSD vote accumulator.

## Config format

The config JSON contains all tunable detection parameters. Key fields for common tuning:

```json
{
  "marker_scale": { "diameter_min_px": 14.0, "diameter_max_px": 66.0 },
  "circle_refinement": "ProjectiveCenter",
  "completion": { "enable": true, "reproj_gate_px": 3.0 },
  "proposal_downscale": "off",
  "self_undistort": { "enable": false },
  "id_correction": { "enable": true }
}
```

Use `default_config_json(board_json)` to get the full config with all defaults, then modify the fields you need. Pass partial JSON to `update_config()` to change only specific fields.

## Scale tiers format

For `detect_multiscale`, pass a JSON object with a `tiers` array:

```json
{
  "tiers": [
    { "diameter_min_px": 14.0, "diameter_max_px": 42.0 },
    { "diameter_min_px": 36.0, "diameter_max_px": 100.0 }
  ]
}
```

Use `scale_tiers_two_tier_standard_json()` or `scale_tiers_four_tier_wide_json()` for presets.
