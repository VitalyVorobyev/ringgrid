# ringgrid WASM demo

Interactive browser demo for the [ringgrid](https://github.com/VitalyVorobyev/ringgrid)
dense coded ring calibration target detector. It runs the full detection
pipeline entirely in the browser via WebAssembly — no image ever leaves the
device.

`book/demo/` is the **single canonical source** for the demo. Both the
GitHub Pages standalone copy (`/demo/`) and the embedded book page
(`/book/demo/`) are staged from here by [`book/build.sh`](../build.sh) into
`book/src/demo/` (generated, git-ignored), together with the freshly built
`crates/ringgrid-wasm/pkg/`.

## Files

| File | Role |
|------|------|
| `index.html` | Layout and controls |
| `styles.css` | Visual system |
| `app.js` | WASM driver + overlay rendering (ellipses, centers, decoded IDs) |
| `samples/`, `samples.json` | Sample images + captions, each carrying its own inline target spec |
| `favicon.svg` | Browser-tab icon |
| `pkg/` | Built WASM package (generated, git-ignored) |

Targets are **data-driven**: every entry in `samples.json` carries its own
`target` spec, so the gallery covers all six valid target combinations —
`{hex, rect}` × `{coded, plain}` × `{origin dots, no dots}` — without any
per-target WASM helper. The target `<select>` (used for uploads) is populated
from those specs at load time.

## Build & run locally

```bash
# From the repository root: builds the WASM package, stages book/demo/ into
# book/src/demo/, and renders the mdBook.
bash book/build.sh

# Serve the built book
python3 -m http.server -d book/book

# Open the demo directly, or via the embedded book page at /demo.html
open http://localhost:8000/demo/index.html
```

## What it shows

- **Sample gallery** — one bundled image per valid target combination (coded
  and plain, hex and rect, with and without origin dots), plus your own uploads.
- **Detection overlay** — outer/inner ellipse fits, decoded marker IDs or grid
  coordinates, and the recovered board origin/frame, drawn on a pixel-accurate
  canvas.
- **Result chips** — marker count, decoded IDs / labeled cells, homography,
  resolved origin frame, and the `board_complete` signal (the success criterion
  for plain, no-dots boards).

## Notes

- Single-page static app — no JS build step or bundler.
- The demo imports the WASM package via relative paths (`./pkg/...`) so it
  works identically whether served standalone or staged into the book.
