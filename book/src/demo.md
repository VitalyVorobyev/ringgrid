# Interactive Demo

The interactive demo runs the full ringgrid detection pipeline in your browser
via WebAssembly — no image ever leaves your device. Choose from bundled samples
covering all six valid target combinations — `{hex, rect}` lattices × `{coded,
plain}` rings × `{origin dots, no dots}` — or upload your own image, then inspect
the decoded marker IDs (or grid coordinates), fitted ellipses, recovered origin,
and the `board_complete` signal overlaid on the source image.

<iframe
  id="ringgrid-demo"
  src="demo/index.html"
  style="width: 100%; height: 900px; border: 1px solid #ddd; border-radius: 4px;"
  loading="lazy">
</iframe>

<noscript>
The interactive demo requires JavaScript and WebAssembly support.
</noscript>

## Running locally

If the embedded demo above does not load, you can build and serve it locally:

```bash
# Build the WASM package and stage the demo into the book
bash book/build.sh

# Serve the built book
python3 -m http.server -d book/book

# Open the demo
open http://localhost:8000/demo/index.html
```

The demo source lives in `book/demo/` (see `book/demo/README.md`) — it is the
single canonical copy, staged by `book/build.sh` into the gitignored
`book/src/demo/` for both this embedded page and the standalone `/demo/`
deployment on GitHub Pages.

## Using the WASM package directly

The demo is a thin UI over the `ringgrid-wasm` npm package. To drive detection
from your own JavaScript:

```js
import init, { RinggridDetector, default_board_json } from './pkg/ringgrid_wasm.js';

await init();

const boardJson = default_board_json();
const detector = new RinggridDetector(boardJson);

const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
const result = JSON.parse(detector.detect_rgba(imageData.data, canvas.width, canvas.height));

console.log(`Found ${result.detected_markers.length} markers`);
```

For the full constructor, detection, config, and scale-tier API, plus output
format details, see `crates/ringgrid-wasm/README.md`.
