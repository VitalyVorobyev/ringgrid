//! Platform-compatible `Instant` for pipeline timing, plus the one
//! `Duration` → milliseconds conversion the pipeline's timing logs use.
//!
//! On native targets, re-exports `std::time::Instant`.
//! On WASM, uses `web_time::Instant` which delegates to `performance.now()`.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;

#[cfg(target_arch = "wasm32")]
pub(crate) use web_time::Instant;

/// Convert a [`std::time::Duration`] to fractional milliseconds.
#[inline]
pub(crate) fn duration_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
