//! Inner and outer ellipse fitting configuration.

/// Configuration for robust inner ellipse fitting from outer-fit hints.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct InnerFitConfig {
    /// Minimum number of sampled points required to attempt a fit.
    pub min_points: usize,
    /// Minimum accepted inlier ratio when RANSAC is used.
    pub min_inlier_ratio: f32,
    /// Maximum accepted RMS Sampson residual (px) of the fitted inner ellipse.
    pub max_rms_residual: f64,
    /// Maximum allowed center shift from outer to inner fit center (px).
    pub max_center_shift_px: f64,
    /// Maximum allowed absolute error in recovered scale ratio vs radial hint.
    pub max_ratio_abs_error: f64,
    /// Local half-width (in radius-sample indices) around the radial hint.
    pub local_peak_halfwidth_idx: usize,
    /// RANSAC configuration for robust inner ellipse fitting.
    pub ransac: crate::conic::RansacConfig,
    /// Confidence multiplier applied when inner ellipse fit fails or is absent.
    ///
    /// Inner fit failure is a reliable signal of poor image quality (heavy blur,
    /// distortion, or edge contamination). Setting this below 1.0 discounts the
    /// decode confidence when the inner ring cannot be fitted, making true markers
    /// in clear regions easier to separate from false detections.
    ///
    /// Default: 0.7 (30 % confidence reduction on inner-fit miss).
    #[serde(default = "InnerFitConfig::default_miss_confidence_factor")]
    pub miss_confidence_factor: f32,
    /// Maximum allowed angular gap (radians) between consecutive inner edge
    /// points. Fits where the largest gap exceeds this are rejected.
    ///
    /// Default: π/2 (90 degrees).
    #[serde(default = "InnerFitConfig::default_max_angular_gap_rad")]
    pub max_angular_gap_rad: f64,
    /// When true, markers are hard-rejected (not just penalized) if the inner
    /// ellipse cannot be fitted. Requires two good ellipses per marker.
    ///
    /// Default: false (backward-compatible).
    #[serde(default = "InnerFitConfig::default_require_inner_fit")]
    pub require_inner_fit: bool,
}

impl InnerFitConfig {
    fn default_miss_confidence_factor() -> f32 {
        0.7
    }
    fn default_max_angular_gap_rad() -> f64 {
        std::f64::consts::FRAC_PI_2
    }
    fn default_require_inner_fit() -> bool {
        false
    }
}

impl Default for InnerFitConfig {
    fn default() -> Self {
        Self {
            min_points: 20,
            min_inlier_ratio: 0.5,
            max_rms_residual: 1.0,
            max_center_shift_px: 12.0,
            max_ratio_abs_error: 0.15,
            local_peak_halfwidth_idx: 3,
            ransac: crate::conic::RansacConfig {
                max_iters: 200,
                inlier_threshold: 1.5,
                min_inliers: 8,
                seed: 43,
            },
            miss_confidence_factor: 0.7,
            max_angular_gap_rad: Self::default_max_angular_gap_rad(),
            require_inner_fit: false,
        }
    }
}

/// Configuration for robust outer ellipse fitting from sampled edge points.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct OuterFitConfig {
    /// Minimum number of sampled points required to attempt direct LS fit.
    pub min_direct_fit_points: usize,
    /// Minimum sampled points required before attempting RANSAC.
    pub min_ransac_points: usize,
    /// RANSAC configuration for robust outer ellipse fitting.
    pub ransac: crate::conic::RansacConfig,
    /// Relative weight of size agreement in outer-hypothesis scoring.
    ///
    /// The score combines decode quality, fit support, size agreement, and
    /// residual quality. This weight controls the size-agreement term and is
    /// normalized with the other terms at runtime.
    ///
    /// Default: `0.15` (preserves legacy behavior).
    #[serde(default = "OuterFitConfig::default_size_score_weight")]
    pub size_score_weight: f32,
    /// Maximum allowed angular gap (radians) between consecutive outer edge
    /// points. Fits where the largest gap exceeds this are rejected.
    ///
    /// Default: π/2 (90 degrees).
    #[serde(default = "OuterFitConfig::default_max_angular_gap_rad")]
    pub max_angular_gap_rad: f64,
}

impl OuterFitConfig {
    fn default_size_score_weight() -> f32 {
        0.15
    }

    fn default_max_angular_gap_rad() -> f64 {
        std::f64::consts::FRAC_PI_2
    }
}

impl Default for OuterFitConfig {
    fn default() -> Self {
        Self {
            min_direct_fit_points: 6,
            min_ransac_points: 8,
            ransac: crate::conic::RansacConfig {
                max_iters: 200,
                inlier_threshold: 1.5,
                min_inliers: 6,
                seed: 42,
            },
            size_score_weight: Self::default_size_score_weight(),
            max_angular_gap_rad: Self::default_max_angular_gap_rad(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inner_fit_config_defaults_are_stable() {
        let core = InnerFitConfig::default();
        assert_eq!(core.min_points, 20);
        assert!((core.min_inlier_ratio - 0.5).abs() < 1e-6);
        assert!((core.max_rms_residual - 1.0).abs() < 1e-9);
        assert!((core.max_center_shift_px - 12.0).abs() < 1e-9);
        assert!((core.max_ratio_abs_error - 0.15).abs() < 1e-9);
        assert_eq!(core.local_peak_halfwidth_idx, 3);
        assert_eq!(core.ransac.max_iters, 200);
        assert!((core.ransac.inlier_threshold - 1.5).abs() < 1e-9);
        assert_eq!(core.ransac.min_inliers, 8);
        assert_eq!(core.ransac.seed, 43);
        assert!((core.miss_confidence_factor - 0.7).abs() < 1e-6);
        assert!(
            (core.max_angular_gap_rad - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
            "inner max_angular_gap_rad"
        );
        assert!(!core.require_inner_fit);
    }

    #[test]
    fn outer_fit_config_defaults_are_stable() {
        let core = OuterFitConfig::default();
        assert_eq!(core.min_direct_fit_points, 6);
        assert_eq!(core.min_ransac_points, 8);
        assert_eq!(core.ransac.max_iters, 200);
        assert!((core.ransac.inlier_threshold - 1.5).abs() < 1e-9);
        assert_eq!(core.ransac.min_inliers, 6);
        assert_eq!(core.ransac.seed, 42);
        assert!((core.size_score_weight - 0.15).abs() < 1e-6);
        assert!(
            (core.max_angular_gap_rad - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
            "outer max_angular_gap_rad"
        );
    }

    #[test]
    fn outer_fit_config_deserialize_missing_size_weight_uses_default() {
        let json = r#"{
            "min_direct_fit_points": 6,
            "min_ransac_points": 8,
            "ransac": {
                "max_iters": 200,
                "inlier_threshold": 1.5,
                "min_inliers": 6,
                "seed": 42
            }
        }"#;
        let cfg: OuterFitConfig = serde_json::from_str(json).expect("deserialize outer fit config");
        assert!((cfg.size_score_weight - 0.15).abs() < 1e-6);
    }
}
