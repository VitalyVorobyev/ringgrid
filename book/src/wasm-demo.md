# WASM & Web Demo

ringgrid compiles to WebAssembly, enabling browser-based marker detection with the same algorithms as the native Rust library.

## Prerequisites

- Rust toolchain (stable, 1.88+)
- `wasm-pack` — install with `cargo install wasm-pack`
- Python 3 (for local HTTP server) or any static file server

## Building the WASM Package

From the repository root:

```bash
wasm-pack build crates/ringgrid-wasm --target web --release
```

This produces `crates/ringgrid-wasm/pkg/` containing the `.wasm` binary, JavaScript glue module, and TypeScript definitions.

## Running the Interactive Demo

1. Build the WASM package (see above).
2. Start a local server from the repository root:

   ```bash
   python3 -m http.server 8080
   ```

3. Open [http://localhost:8080/crates/ringgrid-wasm/demo/](http://localhost:8080/crates/ringgrid-wasm/demo/) in a browser.

## Using the Demo

- **Load Test Image** — fetches the bundled test image (`testdata/target_3_split_00.png`).
- **File input** — load any image from disk.
- **Mode selector**:
  - `detect` — single-pass detection with default scale prior.
  - `detect_adaptive` — multi-scale adaptive detection (slower, finds more markers at varying scales).
  - `propose_with_heatmap` — proposal generation with vote accumulator heatmap visualization.
- **Run Detection** — executes the selected mode and overlays results on the image canvas.
- **Board JSON** — editable board layout (pre-filled with the default board). Expand to view or modify.
- **JSON Result** — collapsible panel with the full detection output as JSON.

Detection overlays show outer ellipses (solid), inner ellipses (dashed), center points, and decoded marker IDs. Colors indicate confidence: green (>0.7), yellow (0.4--0.7), red (<0.4).

## Running WASM Tests

The WASM crate includes native Rust tests that verify parity with the core `ringgrid` library:

```bash
cd crates/ringgrid-wasm
cargo test
```

Tests cover: input validation, RGBA-to-grayscale conversion, board JSON parsing, detection result parity (grayscale and RGBA), adaptive detection parity, proposal/heatmap parity, constructor variants, and version consistency.

## API Quick Reference

### `RinggridDetector` (class)

| Method | Input | Output |
|--------|-------|--------|
| `new(board_json)` | Board layout JSON string | `RinggridDetector` |
| `with_marker_scale(board_json, min_px, max_px)` | Board JSON + scale range | `RinggridDetector` |
| `with_marker_diameter(board_json, diameter_px)` | Board JSON + diameter hint | `RinggridDetector` |
| `detect(pixels, width, height)` | Grayscale `Uint8Array` | JSON string (`DetectionResult`) |
| `detect_rgba(pixels, width, height)` | RGBA `Uint8Array` | JSON string (`DetectionResult`) |
| `detect_adaptive(pixels, width, height)` | Grayscale `Uint8Array` | JSON string (`DetectionResult`) |
| `detect_adaptive_rgba(pixels, width, height)` | RGBA `Uint8Array` | JSON string (`DetectionResult`) |
| `propose_with_heatmap(pixels, width, height)` | Grayscale `Uint8Array` | JSON string (proposals) |
| `propose_with_heatmap_rgba(pixels, width, height)` | RGBA `Uint8Array` | JSON string (proposals) |
| `heatmap_f32()` | — | `Float32Array` (row-major) |
| `heatmap_width()` / `heatmap_height()` | — | `number` |

### Free functions

| Function | Output |
|----------|--------|
| `default_board_json()` | Default board layout as JSON string |
| `version()` | Package version string |
