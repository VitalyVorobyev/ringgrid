//! Detector handle lifecycle and configuration entry points.

use std::ffi::c_char;

use crate::status::RinggridStatus;
use crate::util::{guard, parse_target, read_cstr, to_json, write_out};

/// Opaque, heap-owned detector handle.
///
/// Holds the target layout, detection config, and the last proposal heatmap.
/// Create with `ringgrid_detector_new` (or a `_with_*` variant) and release
/// with `ringgrid_detector_free`. Not thread-safe: use one handle per thread,
/// or synchronize externally.
pub struct RinggridDetector {
    pub(crate) detector: ringgrid::Detector,
    pub(crate) last_heatmap: Option<Vec<f32>>,
    pub(crate) last_heatmap_size: [u32; 2],
}

impl RinggridDetector {
    pub(crate) fn wrap(detector: ringgrid::Detector) -> Box<Self> {
        Box::new(Self {
            detector,
            last_heatmap: None,
            last_heatmap_size: [0, 0],
        })
    }
}

/// Borrow a handle immutably, or `ErrNullArg`.
///
/// # Safety
/// `handle` must be NULL or a live handle from a `ringgrid_detector_*` constructor.
pub(crate) unsafe fn as_ref_handle<'a>(
    handle: *const RinggridDetector,
) -> Result<&'a RinggridDetector, RinggridStatus> {
    unsafe { handle.as_ref() }.ok_or(RinggridStatus::ErrNullArg)
}

/// Borrow a handle mutably, or `ErrNullArg`.
///
/// # Safety
/// `handle` must be NULL or a live handle from a `ringgrid_detector_*` constructor.
pub(crate) unsafe fn as_mut_handle<'a>(
    handle: *mut RinggridDetector,
) -> Result<&'a mut RinggridDetector, RinggridStatus> {
    unsafe { handle.as_mut() }.ok_or(RinggridStatus::ErrNullArg)
}

/// # Safety
/// `out` must be NULL or point to a writable `*mut RinggridDetector`.
unsafe fn write_handle(
    out: *mut *mut RinggridDetector,
    handle: Box<RinggridDetector>,
) -> Result<(), RinggridStatus> {
    let slot = unsafe { out.as_mut() }.ok_or(RinggridStatus::ErrNullArg)?;
    *slot = Box::into_raw(handle);
    Ok(())
}

/// Create a detector from a target layout JSON string (`v5`, or legacy `v4`).
///
/// # Safety
/// `target_json` must be a valid NUL-terminated C string; `out` must point to a
/// writable handle slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detector_new(
    target_json: *const c_char,
    out: *mut *mut RinggridDetector,
) -> RinggridStatus {
    guard(|| {
        let target = parse_target(unsafe { read_cstr(target_json) }?)?;
        let handle = RinggridDetector::wrap(ringgrid::Detector::new(target));
        unsafe { write_handle(out, handle) }
    })
}

/// Create a detector with an explicit min/max marker diameter (pixels).
///
/// # Safety
/// See [`ringgrid_detector_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detector_with_marker_scale(
    target_json: *const c_char,
    min_px: f32,
    max_px: f32,
    out: *mut *mut RinggridDetector,
) -> RinggridStatus {
    guard(|| {
        let target = parse_target(unsafe { read_cstr(target_json) }?)?;
        let scale = ringgrid::MarkerScalePrior {
            diameter_min_px: min_px,
            diameter_max_px: max_px,
        };
        let handle = RinggridDetector::wrap(ringgrid::Detector::with_marker_scale(target, scale));
        unsafe { write_handle(out, handle) }
    })
}

/// Create a detector with a nominal marker diameter hint (pixels).
///
/// # Safety
/// See [`ringgrid_detector_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detector_with_marker_diameter(
    target_json: *const c_char,
    diameter_px: f32,
    out: *mut *mut RinggridDetector,
) -> RinggridStatus {
    guard(|| {
        let target = parse_target(unsafe { read_cstr(target_json) }?)?;
        let handle = RinggridDetector::wrap(ringgrid::Detector::with_marker_diameter_hint(
            target,
            diameter_px,
        ));
        unsafe { write_handle(out, handle) }
    })
}

/// Create a detector from a target plus a full config snapshot (as returned by
/// `ringgrid_detector_config_json`).
///
/// # Safety
/// `target_json` and `config_json` must be valid NUL-terminated C strings;
/// `out` must point to a writable handle slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detector_with_config(
    target_json: *const c_char,
    config_json: *const c_char,
    out: *mut *mut RinggridDetector,
) -> RinggridStatus {
    guard(|| {
        let target = parse_target(unsafe { read_cstr(target_json) }?)?;
        let config_value: serde_json::Value =
            serde_json::from_str(unsafe { read_cstr(config_json) }?)
                .map_err(|_| RinggridStatus::ErrBadJson)?;
        let config: ringgrid::DetectConfig =
            serde_json::from_value(config_value).map_err(|_| RinggridStatus::ErrBadJson)?;
        let config = config.with_target(target);
        let handle = RinggridDetector::wrap(ringgrid::Detector::with_config(config));
        unsafe { write_handle(out, handle) }
    })
}

/// Write the current detection config as a JSON string to `out` (owned; free
/// with `ringgrid_string_free`).
///
/// # Safety
/// `handle` must be a live handle; `out` must point to a writable `char*` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detector_config_json(
    handle: *const RinggridDetector,
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| {
        let det = unsafe { as_ref_handle(handle) }?;
        let json = to_json(det.detector.config())?;
        unsafe { write_out(out, json) }
    })
}

/// Apply a partial config overlay (a JSON object with any subset of config
/// fields; stage tuning nests under `"advanced"`). Only the named leaves change.
///
/// # Safety
/// `handle` must be a live handle; `overlay_json` must be a valid
/// NUL-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detector_update_config(
    handle: *mut RinggridDetector,
    overlay_json: *const c_char,
) -> RinggridStatus {
    guard(|| {
        let det = unsafe { as_mut_handle(handle) }?;
        let overlay: serde_json::Value = serde_json::from_str(unsafe { read_cstr(overlay_json) }?)
            .map_err(|_| RinggridStatus::ErrBadJson)?;
        if !overlay.is_object() {
            return Err(RinggridStatus::ErrBadJson);
        }
        let merged = det
            .detector
            .config()
            .with_json_overlay(overlay)
            .map_err(|_| RinggridStatus::ErrBadJson)?;
        *det.detector.config_mut() = merged;
        Ok(())
    })
}

/// Free a detector handle. NULL is a no-op.
///
/// # Safety
/// `handle` must be NULL or a handle returned by a `ringgrid_detector_*`
/// constructor, freed exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detector_free(handle: *mut RinggridDetector) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}
