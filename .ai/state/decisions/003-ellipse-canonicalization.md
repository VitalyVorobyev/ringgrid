# DEC-003: Ellipse Canonicalization Invariants

**Status:** Active
**Date:** 2025

## Decision

The `Ellipse` struct uses the following canonical form:

```
Ellipse { cx, cy, a, b, angle }
```

### Invariants

1. **`a >= b`** — semi-major axis `a` is always ≥ semi-minor axis `b`.
   The `conic_to_ellipse` conversion enforces this by swapping and rotating
   when needed.

2. **`angle ∈ (−π/2, π/2]`** — rotation of the major axis from +x in radians.
   Normalized by `normalize_angle()` after every conversion.

3. **`a > 0` and `b > 0`** — both semi-axes are strictly positive.
   Checked by `Ellipse::is_valid()`.

4. **All fields are finite** — enforced by `is_valid()`.

5. **`aspect_ratio()` returns `a / b`**, always ≥ 1 when canonicalized.
   The method is defensive and returns `max(a,b) / min(a,b)` to handle
   edge cases.

## Conic representation

- General conic: `A x² + B xy + C y² + D x + E y + F = 0`, stored as
  `ConicCoeffs([A, B, C, D, E, F])`.
- Matrix form `Conic2D.mat` is symmetric: off-diagonals are halved
  (`B/2`, `D/2`, `E/2`).
- Normalization: `ConicCoeffs::normalized()` sets `A + C = 1` (trace-1).
- `Conic2D::normalize_frobenius()` normalizes to unit Frobenius norm.

## Roundtrip guarantee

`Ellipse → ConicCoeffs → Ellipse` is tested to be lossless within
floating-point epsilon (see `conic/fit.rs` tests).
