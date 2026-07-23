# ringgrid-c

C ABI for [ringgrid](https://github.com/VitalyVorobyev/ringgrid) — a pure-Rust
detector for dense ring calibration targets. A flat, stable C surface plus a
header-only C++ convenience wrapper, distributed via CMake and vcpkg.

## Design

A flat C ABI following the JSON-at-the-boundary convention of the Python and
WASM bindings: targets, configs, and results cross as JSON strings, and pixel
buffers as raw pointers. The surface is isomorphic to the WASM binding — one
long-lived detector handle, reused across images.

- **Handle:** create a `RinggridDetector*` from a target JSON (optionally with a
  config, marker scale, or diameter hint) and release it with
  `ringgrid_detector_free`.
- **Errors:** every fallible call returns a `RinggridStatus`
  (`RINGGRID_STATUS_OK == 0`) and writes its payload to an out-parameter.
  `ringgrid_status_str` describes a code. Panics are caught at the boundary and
  reported as `RINGGRID_STATUS_ERR_PANIC` (never unwound across FFI).
- **Ownership:** every `char*` written to a `char**` out-parameter is heap-owned
  by the caller — free it with `ringgrid_string_free`. `ringgrid_status_str`
  returns a static string (never free it). `ringgrid_heatmap_data` returns a
  pointer borrowed from the handle (never free it; invalidated by the next
  `propose`/`free`).
- **ABI guard:** `ringgrid_abi_version()` should match the header's
  `RINGGRID_ABI_VERSION`; the C++ wrapper checks this automatically.

The C ABI in `include/ringgrid.h` is the source of truth. `include/ringgrid.hpp`
is a thin, header-only C++17 RAII layer over it (move-only `ringgrid::Detector`,
`std::string` results, exceptions).

## Surface

Lifecycle/config: `ringgrid_detector_new` / `_with_marker_scale` /
`_with_marker_diameter` / `_with_config` / `_config_json` / `_update_config` /
`_free`.

Detection (each with an `_rgba` variant): `ringgrid_detect`,
`ringgrid_detect_with_diagnostics`, `ringgrid_detect_adaptive`,
`ringgrid_detect_adaptive_with_hint`, `ringgrid_detect_multiscale`,
`ringgrid_detect_with_mapper`, `ringgrid_detect_with_mapper_diagnostics`.

Proposals: `ringgrid_propose_with_heatmap`, `ringgrid_heatmap_data` /
`_heatmap_width` / `_heatmap_height`.

Introspection: `ringgrid_version`, `ringgrid_abi_version`,
`ringgrid_default_target_json`, `ringgrid_rect_24x24_target_json`,
`ringgrid_default_config_json`, `ringgrid_scale_tiers_{four_tier_wide,two_tier_standard}_json`.

**Targets are consumed, not authored, through this ABI.** Every detector
constructor takes target JSON, and the two preset accessors above cover the
common cases; there is deliberately no C entry point for composing a custom
target. Author targets with the `ringgrid` CLI (`ringgrid gen <recipe.toml>`) or
the Rust/Python `TargetLayout` constructors, then ship the emitted
`target_spec.json` alongside your application and pass its contents here.

## Using it

### vcpkg (overlay port)

Until ringgrid is in the upstream vcpkg registry, install from a local checkout
by pointing the port at this repository with `RINGGRID_SOURCE_DIR` (no published
tag or archive hash required):

```bash
RINGGRID_SOURCE_DIR="$(git rev-parse --show-toplevel)" \
  vcpkg install ringgrid --overlay-ports=crates/ringgrid-c/vcpkg
```

Then, in `CMakeLists.txt`:

```cmake
find_package(ringgrid CONFIG REQUIRED)
target_link_libraries(app PRIVATE ringgrid::ringgrid)
```

Building the port requires a Rust toolchain (`cargo`) on `PATH`. Without
`RINGGRID_SOURCE_DIR`, the port instead fetches the released source tarball for
tag `v<version>` from GitHub, verified against the `SHA512` committed in
`vcpkg/portfile.cmake` (kept current for the released version; regenerated on
each version bump). The overlay port is verified in CI (which sets
`RINGGRID_SOURCE_DIR`); upstream vcpkg-registry submission is tracked separately.

### CMake, from source

```bash
cmake -S crates/ringgrid-c -B build -DCMAKE_INSTALL_PREFIX=/your/prefix
cmake --build build
cmake --install build
# then: find_package(ringgrid CONFIG REQUIRED); link ringgrid::ringgrid
```

`RINGGRID_BUILD_SHARED=ON` selects the shared library (default is static; vcpkg
maps `VCPKG_LIBRARY_LINKAGE`). A `pkg-config` file (`ringgrid.pc`) is installed
for non-CMake consumers.

### C example

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

### C++ example

```cpp
#include "ringgrid.hpp"

ringgrid::Detector det(ringgrid::default_target_json());
std::string result = det.detect(pixels, width, height);  // throws ringgrid::Error on failure
```

See `examples/smoke.c` and `examples/example.cpp` for complete programs.

## Regenerating the header

`include/ringgrid.h` is generated by cbindgen and committed; CI regenerates it
and fails on any diff.

```bash
cbindgen --config crates/ringgrid-c/cbindgen.toml \
         --output crates/ringgrid-c/include/ringgrid.h crates/ringgrid-c
```
