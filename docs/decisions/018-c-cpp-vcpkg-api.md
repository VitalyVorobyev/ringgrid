# ADR-018: C / C++ / vcpkg API

- **Status:** proposed
- **Date:** 2026-07-04
- **Author role:** pipeline-architect
- **Supersedes:** none

## Context

ringgrid ships Rust, Python (PyO3), and WASM (wasm-bindgen) surfaces, but C and
C++ consumers — the dominant language for machine-vision and metrology
integrations — have no supported path. We want a stable C ABI that C++ projects
can consume idiomatically and that is distributable through the package managers
those projects already use (CMake `find_package`, vcpkg).

Two ABI strategies exist: the `cxx` crate (C++-first, tightly coupled bindings)
or a hand-authored **C ABI** with a cbindgen-generated header plus a thin C++
convenience wrapper. The C-ABI route is chosen for portability: a flat `extern
"C"` surface is the most stable, most widely consumable contract and packages
cleanly for both vcpkg and Conan.

## Decision

Add a new crate `crates/ringgrid-c` exposing a **flat C ABI** over the
`Detector`, following the JSON-at-the-boundary convention the Python and WASM
crates already use (targets and results cross as JSON strings; pixel buffers
cross as raw pointers). Distribution layers on top:

- **cbindgen** generates `ringgrid.h` from the `#[unsafe(no_mangle)] pub extern
  "C"` surface.
- A thin, hand-written **C++ convenience header** (`ringgrid.hpp`) wraps the C
  functions in RAII types (owning-string / detector handles).
- A **CMake package config** (`ringgrid-config.cmake`) so consumers
  `find_package(ringgrid)` and link the `cdylib`/`staticlib`.
- A **vcpkg port** builds the crate via cargo and installs the header + library
  + CMake config.

Crate conventions mirror the other bindings: `crate-type = ["cdylib",
"staticlib"]`, **excluded** from the root workspace with its own empty
`[workspace]` marker so cargo does not fold it into the main workspace, and a
`ringgrid` dependency by `path` + `version`. It becomes the **fifth**
version-sync location; the `tomllib` version guards in `publish-crates.yml` /
`release-pypi.yml` must learn about it, and a CI build job must exercise the
cbindgen + CMake path.

**Scope of this ADR / current session:** design + a compiling **scaffold** only
— the crate, `cbindgen.toml`, and a minimal `extern "C"` surface (version,
target presets, a grayscale detect entry, string free). The full surface
(config, adaptive/multiscale, diagnostics, camera/undistort), the C++ header,
the CMake package, the vcpkg port, and CI wiring are a tracked follow-up phase
(use the `rust-cpp-bindings` skill).

## Consequences

**Positive:**
- C and C++ integrations get a first-class, package-manager-distributable path.
- A flat C ABI is the most stable contract and the easiest to keep semver-safe.
- Reuses the proven JSON-boundary convention — one serialization contract across
  Python/WASM/C.

**Negative:**
- A fifth version-sync location and a new CI job to maintain.
- Raw-pointer/`unsafe` surface: ownership rules (who frees returned strings) must
  be documented precisely and covered by tests.

**Neutral:**
- The C++ header is convenience only; the C ABI remains the source of truth.
- staticlib output is new (neither py nor wasm produces one).

## Affected Modules

- `crates/ringgrid-c/` (new: `Cargo.toml`, `cbindgen.toml`, `src/lib.rs`,
  `include/ringgrid.h`, later `ringgrid.hpp` + CMake + vcpkg port)
- `Cargo.toml` (root `[workspace] exclude`)
- `.github/workflows/publish-crates.yml`, `release-pypi.yml` (version guards)
