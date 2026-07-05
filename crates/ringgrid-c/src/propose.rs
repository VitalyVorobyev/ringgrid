//! Proposal + heatmap entry points. `propose_with_heatmap` caches the heatmap
//! in the handle; the accessors expose it as a borrowed buffer.

use std::ffi::c_char;

use image::GrayImage;

use crate::detector::{RinggridDetector, as_mut_handle, as_ref_handle};
use crate::status::RinggridStatus;
use crate::util::{gray_from_raw, gray_from_rgba, guard, write_out};
use crate::wire::ProposalPayload;

/// Run proposal generation on `gray`, cache the heatmap, and write the
/// `{"image_size": ..., "proposals": ...}` payload to `out`.
fn run_propose(
    det: &mut RinggridDetector,
    gray: &GrayImage,
    out: *mut *mut c_char,
) -> Result<(), RinggridStatus> {
    let result = det.detector.propose_with_heatmap(gray);
    det.last_heatmap_size = result.image_size;
    det.last_heatmap = Some(result.heatmap);
    let payload = ProposalPayload {
        image_size: det.last_heatmap_size,
        proposals: &result.proposals,
    };
    let json = crate::util::to_json(&payload)?;
    unsafe { write_out(out, json) }
}

/// Generate proposals with a heatmap (grayscale). The heatmap is cached; read
/// it with `ringgrid_heatmap_data`. Writes proposals + image size to `out`.
///
/// # Safety
/// `handle` must be a live handle; `pixels` must point to `width*height` bytes;
/// `out` must be a writable `char*` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_propose_with_heatmap(
    handle: *mut RinggridDetector,
    pixels: *const u8,
    width: u32,
    height: u32,
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| {
        let det = unsafe { as_mut_handle(handle) }?;
        let gray = unsafe { gray_from_raw(pixels, width, height) }?;
        run_propose(det, &gray, out)
    })
}

/// Generate proposals with a heatmap (RGBA, BT.601-converted).
///
/// # Safety
/// As [`ringgrid_propose_with_heatmap`], but `pixels` must point to
/// `width*height*4` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_propose_with_heatmap_rgba(
    handle: *mut RinggridDetector,
    pixels: *const u8,
    width: u32,
    height: u32,
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| {
        let det = unsafe { as_mut_handle(handle) }?;
        let gray = unsafe { gray_from_rgba(pixels, width, height) }?;
        run_propose(det, &gray, out)
    })
}

/// Borrow the last heatmap as a `float` buffer of `width*height` values.
///
/// The pointer written to `out_ptr` is **borrowed** from the handle: it is
/// valid until the next `ringgrid_propose_with_heatmap*` call or
/// `ringgrid_detector_free`, and must **not** be freed. Returns
/// `RINGGRID_STATUS_ERR_NO_HEATMAP` if no proposal call has run yet.
///
/// # Safety
/// `handle` must be a live handle; `out_ptr` and `out_len` must point to
/// writable slots.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_heatmap_data(
    handle: *const RinggridDetector,
    out_ptr: *mut *const f32,
    out_len: *mut usize,
) -> RinggridStatus {
    guard(|| {
        let det = unsafe { as_ref_handle(handle) }?;
        let ptr_slot = unsafe { out_ptr.as_mut() }.ok_or(RinggridStatus::ErrNullArg)?;
        let len_slot = unsafe { out_len.as_mut() }.ok_or(RinggridStatus::ErrNullArg)?;
        match &det.last_heatmap {
            Some(data) => {
                *ptr_slot = data.as_ptr();
                *len_slot = data.len();
                Ok(())
            }
            None => Err(RinggridStatus::ErrNoHeatmap),
        }
    })
}

/// Width of the last heatmap (0 if none, or on NULL handle).
///
/// # Safety
/// `handle` must be NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_heatmap_width(handle: *const RinggridDetector) -> u32 {
    match unsafe { handle.as_ref() } {
        Some(det) => det.last_heatmap_size[0],
        None => 0,
    }
}

/// Height of the last heatmap (0 if none, or on NULL handle).
///
/// # Safety
/// `handle` must be NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_heatmap_height(handle: *const RinggridDetector) -> u32 {
    match unsafe { handle.as_ref() } {
        Some(det) => det.last_heatmap_size[1],
        None => 0,
    }
}
