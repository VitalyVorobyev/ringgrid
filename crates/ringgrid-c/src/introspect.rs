//! Version, ABI, target presets, config, and scale-tier preset accessors.

use std::ffi::c_char;

use crate::RINGGRID_ABI_VERSION;
use crate::status::RinggridStatus;
use crate::util::{guard, parse_target, read_cstr, to_json, write_out};
use crate::wire::scale_tiers_to_json;

/// ringgrid version as a NUL-terminated string written to `out` (owned; free
/// with `ringgrid_string_free`).
///
/// # Safety
/// `out` must point to a writable `char*` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_version(out: *mut *mut c_char) -> RinggridStatus {
    guard(|| unsafe { write_out(out, env!("CARGO_PKG_VERSION").to_string()) })
}

/// The ABI version of this library. A C++ (or C) consumer compiled against a
/// given header should check this equals the header's `RINGGRID_ABI_VERSION`.
#[unsafe(no_mangle)]
pub extern "C" fn ringgrid_abi_version() -> u32 {
    RINGGRID_ABI_VERSION
}

/// The default coded-hex target as `ringgrid.target.v6` JSON.
///
/// # Safety
/// `out` must point to a writable `char*` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_default_target_json(out: *mut *mut c_char) -> RinggridStatus {
    guard(|| unsafe { write_out(out, ringgrid::TargetLayout::default_hex().to_json_string()) })
}

/// The 24×24 plain-rect target (with origin dots) as `ringgrid.target.v6` JSON.
///
/// # Safety
/// `out` must point to a writable `char*` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_rect_24x24_target_json(out: *mut *mut c_char) -> RinggridStatus {
    guard(|| unsafe { write_out(out, ringgrid::TargetLayout::rect_24x24().to_json_string()) })
}

/// The default detection config for a target layout, as a JSON string.
///
/// # Safety
/// `target_json` must be a valid NUL-terminated C string; `out` must point to a
/// writable `char*` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_default_config_json(
    target_json: *const c_char,
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| {
        let target = parse_target(unsafe { read_cstr(target_json) }?)?;
        let config = ringgrid::DetectConfig::from_target(target);
        unsafe { write_out(out, to_json(&config)?) }
    })
}

/// The four-tier-wide scale-tier preset (≈8–220 px) as JSON.
///
/// # Safety
/// `out` must point to a writable `char*` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_scale_tiers_four_tier_wide_json(
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| unsafe {
        write_out(
            out,
            scale_tiers_to_json(&ringgrid::ScaleTiers::four_tier_wide())?,
        )
    })
}

/// The two-tier-standard scale-tier preset (≈14–100 px) as JSON.
///
/// # Safety
/// `out` must point to a writable `char*` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_scale_tiers_two_tier_standard_json(
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| unsafe {
        write_out(
            out,
            scale_tiers_to_json(&ringgrid::ScaleTiers::two_tier_standard())?,
        )
    })
}
