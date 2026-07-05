//! C ABI for [ringgrid](https://github.com/VitalyVorobyev/ringgrid) — a
//! pure-Rust detector for dense ring calibration targets.
//!
//! A flat, stable C ABI over the ringgrid [`Detector`](ringgrid::Detector),
//! isomorphic to the WASM binding: targets, configs, and results cross the
//! boundary as JSON strings, and pixel buffers as raw pointers.
//!
//! # Calling convention
//!
//! Every fallible function returns a [`RinggridStatus`] and writes its payload
//! to an out-parameter; `RINGGRID_STATUS_OK` (0) is success. A
//! [`RinggridDetector`] handle is created once (from a target JSON, optionally
//! with a config or scale) and reused across images.
//!
//! # Ownership
//!
//! - Every `char*` written to a `char**` out-parameter (JSON results,
//!   `ringgrid_version`) is **heap-owned by the caller** — release it with
//!   [`ringgrid_string_free`].
//! - [`ringgrid_status_str`](crate::status::ringgrid_status_str) returns a
//!   **static** string — never free it.
//! - The `const float*` from
//!   [`ringgrid_heatmap_data`](crate::propose::ringgrid_heatmap_data) is
//!   **borrowed** from the handle — never free it; it is invalidated by the
//!   next `propose`/`free`.
//! - A [`RinggridDetector`] handle is released with `ringgrid_detector_free`.
//! - Never free a library-allocated pointer with `free()` or any other allocator.

mod detect;
mod detector;
mod introspect;
mod mapper;
mod propose;
pub mod status;
mod util;
mod wire;

#[cfg(test)]
mod tests;

pub use detector::RinggridDetector;
pub use status::RinggridStatus;

use std::ffi::{CString, c_char};

/// The ABI version of this library, bumped on any breaking change to the C
/// surface. Consumers compiled against a header should check it matches
/// [`ringgrid_abi_version`](crate::introspect::ringgrid_abi_version).
pub const RINGGRID_ABI_VERSION: u32 = 1;

/// Free a string previously written by any `ringgrid_*` function. NULL is a
/// no-op.
///
/// # Safety
/// `s` must be NULL or a pointer written to a `char**` out-parameter by this
/// library, freed exactly once. Passing any other pointer, or freeing twice, is
/// undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}
