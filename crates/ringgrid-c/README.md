# ringgrid-c

C ABI for [ringgrid](https://github.com/VitalyVorobyev/ringgrid) — a pure-Rust
detector for dense ring calibration targets.

> **Status: scaffold.** This crate currently exposes a minimal, compiling C ABI
> (version, target presets, a grayscale detect entry, string free). The full
> surface and the C++/CMake/vcpkg packaging are a tracked follow-up — see
> [ADR-018](../../docs/decisions/018-c-cpp-vcpkg-api.md).

## Design

A flat C ABI, following the JSON-at-the-boundary convention of the Python and
WASM bindings: targets and results cross as JSON strings, pixel buffers as raw
pointers. Every returned string is heap-owned by the caller and freed with
`ringgrid_string_free`; a `NULL` return signals an error.

## Current surface

```c
char *ringgrid_version(void);
char *ringgrid_default_target_json(void);
char *ringgrid_rect_24x24_target_json(void);
char *ringgrid_detect_gray(const char *target_json,
                           const uint8_t *pixels, uint32_t width, uint32_t height);
void  ringgrid_string_free(char *s);
```

## Building the library

```bash
# Shared + static library (target/release/{libringgrid_c.dylib,libringgrid_c.a})
cargo build --release --manifest-path crates/ringgrid-c/Cargo.toml

# Generate the C header (requires cbindgen)
cbindgen --config crates/ringgrid-c/cbindgen.toml \
         --output crates/ringgrid-c/include/ringgrid.h \
         crates/ringgrid-c
```

## Roadmap (deferred)

- Full surface: config, `detect_adaptive` / `detect_multiscale`, diagnostics,
  camera intrinsics / self-undistort.
- `ringgrid.hpp` — a thin RAII C++ convenience header over the C ABI.
- CMake package config (`find_package(ringgrid)`).
- vcpkg port (build via cargo; install header + lib + CMake config).
- CI job exercising the cbindgen + CMake build; add this crate to the release
  version-sync guards (the 5th version location).
