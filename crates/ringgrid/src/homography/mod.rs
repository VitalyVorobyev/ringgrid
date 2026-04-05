//! Homography estimation, projection, and related utilities.

mod core;
mod correspondence;
pub(crate) mod utils;

pub use core::{
    RansacHomographyConfig, RansacHomographyResult, RansacStats, fit_homography_ransac,
    homography_project,
};

pub(crate) use correspondence::{
    CorrespondenceDestinationFrame, DuplicateIdPolicy, collect_marker_correspondences,
    collect_masked_inlier_errors, mean_and_p95, mean_finite_masked_inlier_error,
    reprojection_errors,
};
pub(crate) use utils::{compute_h_stats, matrix3_to_array, mean_reproj_error_px, refit_homography};
