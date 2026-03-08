# DEC-001: Pure Rust — No OpenCV Bindings

**Status:** Active
**Date:** 2025 (project inception)

## Decision

All image processing, ellipse fitting, homography estimation, and geometric
computation is implemented in pure Rust. No OpenCV (or any C/C++ vision library)
bindings are permitted anywhere in the workspace.

## Rationale

- Zero native build dependencies beyond the Rust toolchain.
- Reproducible builds across platforms without system-level library management.
- Full control over algorithm numerics (important for subpixel precision).

## Enforcement

- `Cargo.toml` files must never depend on `opencv`, `cv-*`, or similar binding crates.
- CI runs `cargo build` and `cargo test` without any system library prerequisites
  beyond standard C/libc.
