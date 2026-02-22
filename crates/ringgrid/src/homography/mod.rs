//! Homography estimation, projection, and related utilities.

mod core;
mod correspondence;
pub(crate) mod utils;

pub use core::{
    fit_homography_ransac, homography_project, RansacHomographyConfig, RansacHomographyResult,
    RansacStats,
};

pub(crate) use correspondence::{
    collect_marker_correspondences, collect_masked_inlier_errors, mean_and_p95,
    mean_finite_masked_inlier_error, reprojection_errors, CorrespondenceDestinationFrame,
    DuplicateIdPolicy,
};
pub(crate) use utils::{compute_h_stats, matrix3_to_array, mean_reproj_error_px, refit_homography};
