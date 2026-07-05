# C & C++

ringgrid exposes a stable **C ABI** and a thin, header-only **C++** convenience
wrapper. Both come from the `ringgrid-c` crate and are distributed through vcpkg
and CMake. Targets, configs, and results cross the boundary as JSON strings, and
pixel buffers as raw pointers — the same convention as the other bindings.

Building the library requires a Rust toolchain (`cargo`) on `PATH`, because the
implementation is compiled from Rust source.

## Install

### vcpkg

```bash
vcpkg install ringgrid --overlay-ports=crates/ringgrid-c/vcpkg
```

Then, in your `CMakeLists.txt`:

```cmake
find_package(ringgrid CONFIG REQUIRED)
target_link_libraries(app PRIVATE ringgrid::ringgrid)
```

### CMake, from source

```bash
cmake -S crates/ringgrid-c -B build -DCMAKE_INSTALL_PREFIX=/your/prefix
cmake --build build
cmake --install build
```

`-DRINGGRID_BUILD_SHARED=ON` selects the shared library (static by default; vcpkg
maps `VCPKG_LIBRARY_LINKAGE`). A `pkg-config` file (`ringgrid.pc`) is installed
for non-CMake build systems.

## C++

The RAII wrapper (`ringgrid.hpp`, C++17) owns the detector handle, returns
`std::string` JSON, and throws `ringgrid::Error` on failure:

```cpp
#include "ringgrid.hpp"

// Build a detector from a target spec (from ringgrid_default_target_json(),
// the CLI's target_spec.json, or your own).
ringgrid::Detector detector(ringgrid::default_target_json());

// pixels: width*height grayscale bytes (or use detect_rgba for RGBA).
std::string result_json = detector.detect(pixels, width, height);
// result_json is a DetectionResult; parse it with your JSON library of choice.
```

Adaptive, multi-scale, diagnostics, external-mapper, and proposal entry points
mirror the Rust `Detector` (`detect_adaptive`, `detect_multiscale`,
`detect_with_diagnostics`, `detect_with_mapper`, `propose_with_heatmap`, …).

## C

The C ABI (`ringgrid.h`) returns a `RinggridStatus` from every fallible call and
writes the result to an out-parameter:

```c
#include "ringgrid.h"

char *target = NULL;
ringgrid_default_target_json(&target);

RinggridDetector *det = NULL;
ringgrid_detector_new(target, &det);
ringgrid_string_free(target);

char *result = NULL;
if (ringgrid_detect(det, pixels, width, height, &result) == RINGGRID_STATUS_OK) {
    /* result is a DetectionResult JSON string */
    ringgrid_string_free(result);
}
ringgrid_detector_free(det);
```

### Ownership rules

- Every `char*` written to a `char**` out-parameter is **owned by you** — free it
  with `ringgrid_string_free`.
- `ringgrid_status_str` returns a **static** string — do not free it.
- `ringgrid_heatmap_data` returns a pointer **borrowed** from the handle — do not
  free it; it is invalidated by the next `propose`/`free`.
- Free a detector handle with `ringgrid_detector_free`. Never mix allocators.

### Errors and ABI version

Non-zero statuses map to `RINGGRID_STATUS_ERR_*`; `ringgrid_status_str` describes
them. Panics are caught at the boundary and reported as
`RINGGRID_STATUS_ERR_PANIC` (never unwound across FFI). Check
`ringgrid_abi_version()` against the header's `RINGGRID_ABI_VERSION` — the C++
wrapper does this automatically.
