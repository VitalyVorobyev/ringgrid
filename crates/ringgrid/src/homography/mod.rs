//! Homography estimation, projection, and related utilities.

mod core;
pub(crate) mod utils;

pub use core::{
    fit_homography_ransac, homography_project, RansacHomographyConfig, RansacHomographyResult,
    RansacStats,
};

pub(crate) use utils::{
    compute_h_stats, matrix3_to_array, mean_reproj_error_px, refit_homography_matrix,
};
