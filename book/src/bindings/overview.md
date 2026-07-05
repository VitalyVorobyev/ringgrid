# Language Bindings

ringgrid is one detection engine with several surfaces. The Rust crate is the
reference; the bindings wrap it with the same detection results and the same
JSON-at-the-boundary convention (targets, configs, and results cross as JSON;
pixels as raw buffers), so switching languages does not change behavior.

| Language | Install | Import |
|---|---|---|
| **Rust** | `cargo add ringgrid` | `use ringgrid::Detector;` |
| **Python** | `pip install ringgrid` | `import ringgrid` |
| **C / C++** | `vcpkg install ringgrid` (or CMake from source) | `#include "ringgrid.h"` / `"ringgrid.hpp"` |
| **JavaScript / WASM** | `npm install @vitavision/ringgrid` | `import init, { RinggridDetector } from "@vitavision/ringgrid";` |

There is also a command-line tool — see the [CLI Guide](../cli-guide.md) — and an
in-browser [interactive demo](../demo.md).

## Python

```bash
pip install ringgrid          # detector + typed TargetLayout API
pip install "ringgrid[viz]"   # + optional matplotlib/Pillow visualization helpers
```

```python
import ringgrid

detector = ringgrid.Detector.from_target(ringgrid.TargetLayout.coded_hex())
result = detector.detect_path("photo.png")
for m in result.detected_markers:
    print(m.id, m.center)
```

The Python package ships type stubs (`py.typed`) and a typed `TargetLayout` API
that can also render printable SVG/PNG/DXF targets. See the
[`ringgrid-py` README](https://pypi.org/project/ringgrid/) for the full
`DetectConfig` field guide.

## C / C++

A stable C ABI plus a header-only C++ RAII wrapper, distributed through vcpkg and
CMake `find_package`. See [C & C++](cpp.md) for the full guide.

## JavaScript / WebAssembly

```bash
npm install @vitavision/ringgrid
```

```js
import init, { RinggridDetector, default_board_json } from "@vitavision/ringgrid";

await init();
const detector = new RinggridDetector(default_board_json());
const json = detector.detect_rgba(imageData.data, imageData.width, imageData.height);
const result = JSON.parse(json);
```

The WASM package powers the [interactive demo](../demo.md); it accepts grayscale
or RGBA (canvas `ImageData`) buffers and returns the same `DetectionResult` JSON.
