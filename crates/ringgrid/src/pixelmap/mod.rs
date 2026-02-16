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
///
/// Implement this trait to plug in a custom distortion model. Both methods
/// must be approximate inverses of each other. Return `None` if a point
/// cannot be mapped (e.g. it falls outside the valid distortion domain).
///
/// Built-in implementations: [`CameraModel`] (Brown-Conrady radial-tangential)
/// and [`DivisionModel`] (single-parameter division model).
///
/// # Example
///
/// ```
/// use ringgrid::PixelMapper;
///
/// struct Identity;
///
/// impl PixelMapper for Identity {
///     fn image_to_working_pixel(&self, p: [f64; 2]) -> Option<[f64; 2]> {
///         Some(p)
///     }
///     fn working_to_image_pixel(&self, p: [f64; 2]) -> Option<[f64; 2]> {
///         Some(p)
///     }
/// }
/// ```
pub trait PixelMapper {
    /// Map from image (distorted) pixel coordinates to working coordinates.
    fn image_to_working_pixel(&self, image_xy: [f64; 2]) -> Option<[f64; 2]>;
    /// Map from working coordinates back to image (distorted) pixel coordinates.
    fn working_to_image_pixel(&self, working_xy: [f64; 2]) -> Option<[f64; 2]>;
}
