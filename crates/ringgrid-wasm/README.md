# ringgrid-wasm

WebAssembly bindings for [ringgrid](https://github.com/VitalyVorobyev/ringgrid) — a pure-Rust detector for dense coded ring calibration targets.

## Building

```bash
wasm-pack build crates/ringgrid-wasm --target web --release
```

## Usage

```js
import init, { RinggridDetector, default_board_json } from './pkg/ringgrid_wasm.js';

await init();

const detector = new RinggridDetector(default_board_json());

// From canvas ImageData (RGBA)
const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
const resultJson = detector.detect_rgba(imageData.data, canvas.width, canvas.height);
const result = JSON.parse(resultJson);

console.log(`Found ${result.detected_markers.length} markers`);
for (const m of result.detected_markers) {
    console.log(`  Marker ${m.id} at (${m.center[0].toFixed(1)}, ${m.center[1].toFixed(1)})`);
}
```

## API

### `RinggridDetector`

| Method | Input | Output |
|--------|-------|--------|
| `new(board_json)` | Board layout JSON string | `RinggridDetector` |
| `with_marker_scale(board_json, min_px, max_px)` | Board + scale range | `RinggridDetector` |
| `with_marker_diameter(board_json, diameter_px)` | Board + diameter hint | `RinggridDetector` |
| `detect(pixels, w, h)` | Grayscale `Uint8Array` | JSON `DetectionResult` |
| `detect_rgba(pixels, w, h)` | RGBA `Uint8Array` | JSON `DetectionResult` |
| `detect_adaptive(pixels, w, h)` | Grayscale `Uint8Array` | JSON `DetectionResult` |
| `detect_adaptive_rgba(pixels, w, h)` | RGBA `Uint8Array` | JSON `DetectionResult` |
| `propose_with_heatmap(pixels, w, h)` | Grayscale `Uint8Array` | JSON proposals |
| `propose_with_heatmap_rgba(pixels, w, h)` | RGBA `Uint8Array` | JSON proposals |
| `heatmap_f32()` | — | `Float32Array` (row-major) |
| `heatmap_width()` / `heatmap_height()` | — | `number` |

### Free functions

| Function | Output |
|----------|--------|
| `default_board_json()` | Default board layout JSON string |
| `version()` | Package version string |

## Output format

Detection results are returned as JSON strings matching the Rust `DetectionResult` type.
Key fields:
- `detected_markers[].id` — codebook index (0–892)
- `detected_markers[].center` — `[x, y]` pixel coordinates
- `detected_markers[].confidence` — detection confidence (0–1)
- `detected_markers[].ellipse_outer` / `ellipse_inner` — fitted ellipse parameters
- `homography` — 3x3 board-to-image homography matrix (if available)

Heatmap is a row-major `Float32Array` of size `width * height`, representing the RSD vote accumulator.
