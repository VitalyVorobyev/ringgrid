//! Tools related to pixel mapping for undistortion

mod cameramodel;
mod distortion;
mod self_undistort;

pub use cameramodel::{CameraIntrinsics, CameraModel};
pub use distortion::{DivisionModel, RadialTangentialDistortion, UndistortConfig};
pub use self_undistort::{estimate_self_undistort, SelfUndistortConfig, SelfUndistortResult};

/// Mapping between raw image pixels and detector working-frame pixels.
///
/// The working frame is the coordinate system used by sampling/fitting stages.
/// For distortion-aware processing this is typically an undistorted pixel frame.
pub trait PixelMapper {
    /// Map from image (distorted) pixel coordinates to working coordinates.
    fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]>;
    /// Map from working coordinates back to image (distorted) pixel coordinates.
    fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]>;
}
