//! C ABI for ringgrid — **scaffold**.
//!
//! A stable, flat C ABI over the ringgrid detector for C and C++ consumers,
//! distributed via a cbindgen-generated header, a CMake package, and a vcpkg
//! port (see `README.md` and `docs/decisions/018-c-cpp-vcpkg-api.md`).
//!
//! Targets and results cross the boundary as JSON strings, and pixel buffers as
//! raw pointers — the same convention the Python and WASM bindings use. This is
//! the scaffold surface: version, target presets, a grayscale detect entry, and
//! a string deallocator. The full surface (config, adaptive/multiscale,
//! diagnostics, camera/undistort) and the C++/CMake/vcpkg packaging are a
//! tracked follow-up.
//!
//! **Ownership:** every `*mut c_char` returned by this library is heap-owned by
//! the caller and must be released with [`ringgrid_string_free`]. A NULL return
//! signals an error (invalid argument, bad JSON, or size mismatch).

use std::ffi::{CStr, CString, c_char};

use ringgrid::{Detector, TargetLayout};

/// ringgrid version as a NUL-terminated string. Free with [`ringgrid_string_free`].
#[unsafe(no_mangle)]
pub extern "C" fn ringgrid_version() -> *mut c_char {
    into_c_string(env!("CARGO_PKG_VERSION").to_string())
}

/// The default coded-hex target as `ringgrid.target.v5` JSON.
/// Free with [`ringgrid_string_free`].
#[unsafe(no_mangle)]
pub extern "C" fn ringgrid_default_target_json() -> *mut c_char {
    into_c_string(TargetLayout::default_hex().to_json_string())
}

/// The 24x24 plain-rect target (with origin dots) as `ringgrid.target.v5` JSON.
/// Free with [`ringgrid_string_free`].
#[unsafe(no_mangle)]
pub extern "C" fn ringgrid_rect_24x24_target_json() -> *mut c_char {
    into_c_string(TargetLayout::rect_24x24().to_json_string())
}

/// Detect markers in an 8-bit grayscale image.
///
/// `target_json` is a NUL-terminated `ringgrid.target.v5` (or legacy v4) JSON
/// string. `pixels` points to `width * height` row-major grayscale bytes.
/// Returns a NUL-terminated `DetectionResult` JSON string on success, or NULL on
/// error. Free the result with [`ringgrid_string_free`].
///
/// # Safety
/// `target_json` must be a valid NUL-terminated C string, and `pixels` must
/// point to at least `width * height` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detect_gray(
    target_json: *const c_char,
    pixels: *const u8,
    width: u32,
    height: u32,
) -> *mut c_char {
    if target_json.is_null() || pixels.is_null() {
        return std::ptr::null_mut();
    }
    let Ok(json) = (unsafe { CStr::from_ptr(target_json) }).to_str() else {
        return std::ptr::null_mut();
    };
    let Ok(target) = TargetLayout::from_json_str(json) else {
        return std::ptr::null_mut();
    };
    let n = (width as usize).saturating_mul(height as usize);
    let buf = unsafe { std::slice::from_raw_parts(pixels, n) }.to_vec();
    let Some(image) = image::GrayImage::from_raw(width, height, buf) else {
        return std::ptr::null_mut();
    };
    match Detector::new(target).detect(&image) {
        Ok(result) => match serde_json::to_string(&result) {
            Ok(s) => into_c_string(s),
            Err(_) => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a string previously returned by a `ringgrid_*` function.
///
/// # Safety
/// `s` must be a pointer returned by this library, or NULL. Passing any other
/// pointer, or freeing twice, is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}

/// Move an owned `String` across the ABI as a heap `*mut c_char` (or NULL if it
/// contains an interior NUL byte).
fn into_c_string(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn take(ptr: *mut c_char) -> String {
        assert!(!ptr.is_null());
        let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_owned();
        unsafe { ringgrid_string_free(ptr) };
        s
    }

    #[test]
    fn version_matches_crate() {
        assert_eq!(take(ringgrid_version()), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn presets_are_valid_v5_json() {
        for json in [take(ringgrid_default_target_json()),
                     take(ringgrid_rect_24x24_target_json())] {
            assert!(TargetLayout::from_json_str(&json).is_ok());
        }
    }

    #[test]
    fn detect_blank_image_returns_json() {
        let target = CString::new(TargetLayout::default_hex().to_json_string()).unwrap();
        let pixels = vec![0u8; 64 * 48];
        let out = unsafe {
            ringgrid_detect_gray(target.as_ptr(), pixels.as_ptr(), 64, 48)
        };
        let json = take(out);
        let val: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(val.get("detected_markers").is_some());
    }

    #[test]
    fn null_arguments_return_null() {
        let out = unsafe {
            ringgrid_detect_gray(std::ptr::null(), std::ptr::null(), 8, 8)
        };
        assert!(out.is_null());
    }
}
