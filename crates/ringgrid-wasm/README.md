# ringgrid-wasm

WebAssembly bindings for [ringgrid](https://github.com/VitalyVorobyev/ringgrid) — a pure-Rust detector for dense ring calibration targets on hex or rectangular lattices (coded 16-sector or plain rings).

## Install

The published package is [`@vitavision/ringgrid`](https://www.npmjs.com/package/@vitavision/ringgrid):

```bash
npm install @vitavision/ringgrid
```

```js
import init, { RinggridDetector, default_board_json } from "@vitavision/ringgrid";

await init();                                   // load the wasm module
const detector = new RinggridDetector(default_board_json());
const json = detector.detect_rgba(imageData.data, imageData.width, imageData.height);
const result = JSON.parse(json);                // a DetectionResult
```

## Building from source

```bash
wasm-pack build crates/ringgrid-wasm --target web --release
```

## Demo

The interactive demo now lives at `book/demo/` (single canonical source) and
is embedded in the mdBook user guide. Build and serve it from the repository
root:

```bash
bash book/build.sh
python3 -m http.server -d book/book
```

Open [http://localhost:8000/demo/index.html](http://localhost:8000/demo/index.html) in your browser, or visit the
live demo at <https://vitalyvorobyev.github.io/ringgrid/demo/>.

The demo supports:

- **All six target combinations** — one bundled sample per `{hex, rect}` ×
  `{coded, plain}` × `{origin dots, no dots}` layout, each carrying its own
  inline target spec; plus your own uploads via the file picker
- **Adaptive multi-scale detection** — runs `detect_adaptive`, so markers are
  found without a known pixel size
- **Visualization** — confidence-colored outer ellipses, center crosshairs, ID
  or grid-coordinate labels, and the resolved origin (dots + origin cell)
- **Result chips** — marker count, decoded IDs / labeled cells, homography,
  origin frame (absolute vs relative), and the `board_complete` signal
- **Hover inspector** — per-marker id/grid coord, center, confidence, and axes

## Usage

```js
import init, {
  RinggridDetector,
  default_board_json,
  rect_24x24_target_json,
  default_config_json,
} from './pkg/ringgrid_wasm.js';

await init();

// The constructor accepts any `ringgrid.target.v5` JSON: default_board_json()
// for the classic coded hex board, rect_24x24_target_json() for the 24×24
// plain rect target, or your own TargetLayout JSON.
const targetJson = default_board_json();
const detector = new RinggridDetector(targetJson);

// From canvas ImageData (RGBA)
const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
const resultJson = detector.detect_rgba(imageData.data, canvas.width, canvas.height);
const result = JSON.parse(resultJson);

console.log(`Found ${result.detected_markers.length} markers`);
for (const m of result.detected_markers) {
    // Coded targets key markers by `id`; plain targets by `grid_coord`.
    const key = m.id ?? `cell ${m.grid_coord}`;
    console.log(`  Marker ${key} at (${m.center[0].toFixed(1)}, ${m.center[1].toFixed(1)})`);
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
| `default_board_json()` | Classic coded hex target JSON string |
| `rect_24x24_target_json()` | 24×24 plain rect target JSON string (with origin dots) |
| `default_config_json(board_json)` | Default detection config for a target |
| `scale_tiers_four_tier_wide_json()` | Four-tier preset (8-220 px) |
| `scale_tiers_two_tier_standard_json()` | Two-tier preset (14-100 px) |
| `version()` | Package version string |

## Output format

Detection results are returned as JSON strings matching the Rust `DetectionResult` type.
Key fields:
- `detected_markers[].id` — codebook index (0-892) on coded targets; `null` on plain targets
- `detected_markers[].grid_coord` — lattice coordinate (`[q, r]` hex, `[col, row]` rect); the marker key on plain targets
- `board_frame` — `absolute` (origin-anchored) or `relative_canonical` (plain target with no resolved origin)
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
