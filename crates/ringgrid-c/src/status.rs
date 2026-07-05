//! ABI status codes.

use std::ffi::c_char;

/// Result status for every fallible `ringgrid_*` call.
///
/// `RINGGRID_STATUS_OK` (0) is success; any other value is an error and the
/// call's out-parameter is left untouched.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RinggridStatus {
    /// Success.
    Ok = 0,
    /// A required pointer argument was NULL.
    ErrNullArg = 1,
    /// A JSON argument (target, config, tiers, or mapper) failed to parse.
    ErrBadJson = 2,
    /// Image dimensions were zero, overflowed, or the buffer did not match.
    ErrBadImage = 3,
    /// Detection failed (e.g. an unsupported target or a strict-mode gate).
    ErrDetect = 4,
    /// Serializing the result to JSON failed (should not happen in practice).
    ErrSerialize = 5,
    /// A heatmap accessor was called before `ringgrid_propose_with_heatmap`.
    ErrNoHeatmap = 6,
    /// A Rust panic was caught at the FFI boundary.
    ErrPanic = 7,
}

/// Human-readable description of a status code.
///
/// The returned pointer is a static, NUL-terminated string owned by the
/// library — the caller must **not** free it (unlike the owned strings the
/// other entry points return).
#[unsafe(no_mangle)]
pub extern "C" fn ringgrid_status_str(status: RinggridStatus) -> *const c_char {
    match status {
        RinggridStatus::Ok => c"ok".as_ptr(),
        RinggridStatus::ErrNullArg => c"null argument".as_ptr(),
        RinggridStatus::ErrBadJson => c"invalid JSON argument".as_ptr(),
        RinggridStatus::ErrBadImage => c"invalid image dimensions or buffer".as_ptr(),
        RinggridStatus::ErrDetect => c"detection failed".as_ptr(),
        RinggridStatus::ErrSerialize => c"result serialization failed".as_ptr(),
        RinggridStatus::ErrNoHeatmap => {
            c"no heatmap available; call ringgrid_propose_with_heatmap first".as_ptr()
        }
        RinggridStatus::ErrPanic => c"internal panic".as_ptr(),
    }
}
