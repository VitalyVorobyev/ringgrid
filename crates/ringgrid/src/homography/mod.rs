//! Homography estimation, projection, and related utilities.

mod core;
mod correspondence;
pub(crate) mod utils;

pub use core::{RansacHomographyResult, RansacStats, fit_homography_ransac};
pub(crate) use core::{homography_project, homography_project_rows};

pub(crate) use core::estimate_homography_dlt;

pub(crate) use correspondence::{
    CorrespondenceDestinationFrame, DuplicateIdPolicy, collect_marker_correspondences,
    mean_finite_masked_inlier_error, reprojection_errors,
};
pub(crate) use utils::{compute_h_stats, matrix3_to_array, mean_reproj_error_px, refit_homography};
