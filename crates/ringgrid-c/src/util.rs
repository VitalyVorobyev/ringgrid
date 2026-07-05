//! Boundary primitives shared by every entry point: panic guarding, pointer
//! validation, string/image marshalling, and JSON (de)serialization. No public
//! ABI symbols live here — this is the internal marshalling layer.

use std::ffi::{CStr, CString, c_char};

use image::GrayImage;

use crate::status::RinggridStatus;

/// Run `f` with a panic firewall, mapping any unwind to
/// [`RinggridStatus::ErrPanic`]. Edition-2024 `extern "C"` aborts on unwind, so
/// this is what converts a Rust panic into an error code instead.
///
/// The closure captures raw pointers (`AssertUnwindSafe`); it only ever reads
/// through them under the documented safety preconditions of the entry point,
/// and returns before any observable state is left half-updated.
pub(crate) fn guard(f: impl FnOnce() -> Result<(), RinggridStatus>) -> RinggridStatus {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(())) => RinggridStatus::Ok,
        Ok(Err(status)) => status,
        Err(_) => RinggridStatus::ErrPanic,
    }
}

/// Borrow a NUL-terminated C string as `&str`, or fail with `ErrNullArg`
/// (NULL) / `ErrBadJson` (not valid UTF-8).
///
/// # Safety
/// `ptr` must be NULL or point to a valid NUL-terminated C string.
pub(crate) unsafe fn read_cstr<'a>(ptr: *const c_char) -> Result<&'a str, RinggridStatus> {
    if ptr.is_null() {
        return Err(RinggridStatus::ErrNullArg);
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|_| RinggridStatus::ErrBadJson)
}

/// Move an owned `String` across the ABI as a heap `*mut c_char`. Fails with
/// `ErrSerialize` only if the string contains an interior NUL byte.
fn into_c_string(s: String) -> Result<*mut c_char, RinggridStatus> {
    CString::new(s)
        .map(|c| c.into_raw())
        .map_err(|_| RinggridStatus::ErrSerialize)
}

/// Write an owned string to a `char**` out-parameter as a heap-owned C string
/// (freed by [`crate::ringgrid_string_free`]).
///
/// # Safety
/// `out` must be NULL or point to a writable `*mut c_char`.
pub(crate) unsafe fn write_out(out: *mut *mut c_char, s: String) -> Result<(), RinggridStatus> {
    let slot = unsafe { out.as_mut() }.ok_or(RinggridStatus::ErrNullArg)?;
    *slot = into_c_string(s)?;
    Ok(())
}

/// Serialize a value to a JSON string, mapping failure to `ErrSerialize`.
pub(crate) fn to_json<T: serde::Serialize>(value: &T) -> Result<String, RinggridStatus> {
    serde_json::to_string(value).map_err(|_| RinggridStatus::ErrSerialize)
}

/// Parse a target spec (compositional `ringgrid.target.v5`, or legacy
/// `ringgrid.target.v4` auto-migrated).
pub(crate) fn parse_target(json: &str) -> Result<ringgrid::TargetLayout, RinggridStatus> {
    ringgrid::TargetLayout::from_json_str(json).map_err(|_| RinggridStatus::ErrBadJson)
}

/// Build a grayscale image from `width * height` row-major bytes.
///
/// # Safety
/// `pixels` must be NULL or point to at least `width * height` readable bytes.
pub(crate) unsafe fn gray_from_raw(
    pixels: *const u8,
    width: u32,
    height: u32,
) -> Result<GrayImage, RinggridStatus> {
    if pixels.is_null() {
        return Err(RinggridStatus::ErrNullArg);
    }
    let n = pixel_count(width, height)?;
    let buf = unsafe { std::slice::from_raw_parts(pixels, n) }.to_vec();
    GrayImage::from_raw(width, height, buf).ok_or(RinggridStatus::ErrBadImage)
}

/// Build a grayscale image from `width * height` row-major RGBA bytes via the
/// BT.601 luma `Y = (77R + 150G + 29B + 128) >> 8`, matching the WASM binding.
///
/// # Safety
/// `pixels` must be NULL or point to at least `width * height * 4` readable bytes.
pub(crate) unsafe fn gray_from_rgba(
    pixels: *const u8,
    width: u32,
    height: u32,
) -> Result<GrayImage, RinggridStatus> {
    if pixels.is_null() {
        return Err(RinggridStatus::ErrNullArg);
    }
    let n = pixel_count(width, height)?;
    let total = n.checked_mul(4).ok_or(RinggridStatus::ErrBadImage)?;
    let rgba = unsafe { std::slice::from_raw_parts(pixels, total) };
    let mut gray = Vec::with_capacity(n);
    for i in 0..n {
        let r = rgba[4 * i] as u32;
        let g = rgba[4 * i + 1] as u32;
        let b = rgba[4 * i + 2] as u32;
        gray.push(((77 * r + 150 * g + 29 * b + 128) >> 8) as u8);
    }
    GrayImage::from_raw(width, height, gray).ok_or(RinggridStatus::ErrBadImage)
}

/// Checked `width * height` with non-zero dimensions, or `ErrBadImage`.
fn pixel_count(width: u32, height: u32) -> Result<usize, RinggridStatus> {
    if width == 0 || height == 0 {
        return Err(RinggridStatus::ErrBadImage);
    }
    (width as usize)
        .checked_mul(height as usize)
        .ok_or(RinggridStatus::ErrBadImage)
}
