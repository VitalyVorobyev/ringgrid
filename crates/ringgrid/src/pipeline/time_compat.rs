//! Platform-compatible `Instant` for pipeline timing.
//!
//! On native targets, re-exports `std::time::Instant`.
//! On WASM, uses `web_time::Instant` which delegates to `performance.now()`.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;

#[cfg(target_arch = "wasm32")]
pub(crate) use web_time::Instant;
