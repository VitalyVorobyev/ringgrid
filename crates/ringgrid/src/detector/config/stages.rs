//! Per-stage pipeline configuration: seeding, completion, projective center,
//! ID correction, and inner-as-outer recovery.

/// Seed-injection controls for proposal generation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct SeedProposalConfig {
    /// Radius (pixels) used to merge seed centers with detector proposals.
    pub merge_radius_px: f32,
    /// Score assigned to injected seed proposals.
    pub seed_score: f32,
    /// Maximum number of seeds consumed in one run.
    pub max_seeds: Option<usize>,
}

impl Default for SeedProposalConfig {
    fn default() -> Self {
        Self {
            merge_radius_px: 3.0,
            seed_score: 1.0e12,
            max_seeds: Some(512),
        }
    }
}

/// Configuration for homography-guided completion: attempt local fits for
/// missing IDs at H-projected board locations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct CompletionConfig {
    /// Enable completion (runs only when a valid homography is available).
    pub enable: bool,
    /// Radial sampling extent (pixels) used for edge sampling around the prior center.
    pub roi_radius_px: f32,
    /// Maximum allowed reprojection error (pixels) between the fitted center and
    /// the H-projected board center.
    pub reproj_gate_px: f32,
    /// Minimum fit confidence in [0, 1].
    pub min_fit_confidence: f32,
    /// Minimum arc coverage (fraction of rays with both edges found).
    pub min_arc_coverage: f32,
    /// Optional cap on how many completion fits to attempt (in ID order).
    pub max_attempts: Option<usize>,
    /// Skip attempts whose projected center is too close to the image boundary.
    pub image_margin_px: f32,
    /// Require a perfect decode (dist=0 and margin ≥ the active profile's
    /// minimum cyclic Hamming distance) for a completion marker to be accepted.
    ///
    /// When homography prediction accuracy is low (e.g. significant lens distortion
    /// without a calibrated mapper), the H-projected seed can be several pixels off.
    /// Under those conditions, the geometry gates alone are insufficient; requiring a
    /// perfect decode provides an independent quality signal that does not depend on H.
    ///
    /// Default: `false` (backward-compatible). Set to `true` for Scheimpflug / high-
    /// distortion setups where no calibrated camera model is available.
    #[serde(default = "CompletionConfig::default_require_perfect_decode")]
    pub require_perfect_decode: bool,
    /// Maximum allowed coefficient of variation (std_dev / mean) of per-ray outer
    /// radii. High scatter indicates inner/outer edge contamination — rays landing on
    /// the inner ring inflate the apparent outer radius for some angles. Values above
    /// this threshold cause the completion candidate to be rejected.
    ///
    /// The gate is skipped when fewer than 2 rays have valid outer radii or when the
    /// mean outer radius is below 1 px (degenerate fit).
    ///
    /// Default: `0.35` (35% coefficient of variation).
    #[serde(default = "CompletionConfig::default_max_radii_std_ratio")]
    pub max_radii_std_ratio: f32,
}

impl CompletionConfig {
    fn default_require_perfect_decode() -> bool {
        false
    }

    fn default_max_radii_std_ratio() -> f32 {
        0.35
    }
}

impl Default for CompletionConfig {
    fn default() -> Self {
        Self {
            enable: true,
            roi_radius_px: 24.0,
            reproj_gate_px: 3.0,
            min_fit_confidence: 0.45,
            min_arc_coverage: 0.35,
            max_attempts: None,
            image_margin_px: 10.0,
            require_perfect_decode: false,
            max_radii_std_ratio: 0.35,
        }
    }
}

/// Projective-only unbiased center recovery from inner/outer conics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct ProjectiveCenterConfig {
    /// Use `marker_spec.r_inner_expected` as an optional eigenvalue prior.
    pub use_expected_ratio: bool,
    /// Weight of the eigenvalue-vs-ratio penalty term.
    pub ratio_penalty_weight: f64,
    /// Maximum allowed shift (pixels) from the pre-correction center.
    ///
    /// Corrections jumping further than this are rejected and the original
    /// center is kept. `None` means "auto": the gate uses the nominal marker
    /// diameter derived from the active [`MarkerScalePrior`](super::MarkerScalePrior).
    /// Explicit values are honored as-is and survive target re-derivation.
    /// (Renamed from `max_center_shift_px` in 0.8.0 to disambiguate from the
    /// unrelated inner-fit field of the same name; the old JSON key is still
    /// accepted as an alias.)
    #[serde(alias = "max_center_shift_px")]
    pub max_correction_shift_px: Option<f64>,
    /// Optional maximum accepted projective-selection residual.
    ///
    /// Higher values are less strict; `None` disables this gate.
    pub max_selected_residual: Option<f64>,
    /// Optional minimum accepted eigenvalue separation used by the selector.
    ///
    /// Low separation indicates unstable conic-pencil eigenpairs.
    pub min_eig_separation: Option<f64>,
}

impl Default for ProjectiveCenterConfig {
    fn default() -> Self {
        Self {
            use_expected_ratio: true,
            ratio_penalty_weight: 1.0,
            max_correction_shift_px: None,
            max_selected_residual: Some(0.25),
            min_eig_separation: Some(1e-6),
        }
    }
}

/// Structural ID verification and correction using hex neighborhood consensus.
///
/// Runs after fit-decode and deduplication, before the global RANSAC filter.
/// Uses the board's hex lattice geometry to detect wrong IDs (misidentified by
/// the codebook decoder) and recover missing ones. Each marker's correct ID is
/// implied by its decoded neighbors' positions: neighbors vote on the expected
/// board position of the query marker using a local affine transform (or scale
/// estimate when fewer than 3 neighbors are available).
///
/// Markers that cannot be verified or corrected have their IDs cleared
/// (`id = None`) or are removed entirely depending on `remove_unverified`.
/// This guarantees no wrong IDs reach the global filter or completion stages.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct IdCorrectionConfig {
    /// Enable structural ID verification and correction.
    pub enable: bool,
    /// Local-scale staged search multipliers (one per recovery pass, sorted ascending).
    ///
    /// The neighbor gate for pair `(i, j)` in each pass is:
    /// `dist_px(i,j) <= mul * 0.5 * (outer_radius_px_i + outer_radius_px_j)`.
    ///
    /// Multiple multipliers produce a staged sweep from tight to loose. A single-element
    /// Vec produces one pass (equivalent to the old `search_radius_outer_mul`).
    pub auto_search_radius_outer_muls: Vec<f64>,
    /// Local-scale neighborhood multiplier for consistency checks.
    pub consistency_outer_mul: f64,
    /// Minimum number of local neighbors required to run consistency checks.
    /// Default: 1 (any single neighbor provides enough evidence).
    pub consistency_min_neighbors: usize,
    /// Minimum number of one-hop board-neighbor support edges required for a
    /// non-soft-locked ID to remain assigned. Default: 1.
    pub consistency_min_support_edges: usize,
    /// Maximum allowed contradiction fraction in local consistency checks.
    pub consistency_max_contradiction_frac: f32,
    /// Promote a decoded ID to trusted when its local neighborhood structurally
    /// confirms it (support edges ≥ `consistency_min_support_edges`, zero
    /// contradiction edges, and no confident vote against it), even when the
    /// voting stages produced no candidate. Recovers correct non-exact decodes
    /// in sparse/partial/blurry views that would otherwise be cleared by
    /// cleanup. Default: true.
    pub confirm_by_consistency: bool,
    /// When enabled, exact decodes (`best_dist=0, margin>=2`) are soft-locked:
    /// they are not overridden during normal recovery and only cleared on
    /// strict structural contradiction.
    pub soft_lock_exact_decode: bool,
    /// Minimum number of independent neighbor votes required to accept a
    /// candidate ID for a marker that already has an id. Default: 2.
    pub min_votes: usize,
    /// Minimum votes to assign an ID to a marker that currently has `id = None`.
    ///
    /// A single high-confidence trusted neighbor is sufficient evidence when
    /// there is no existing wrong ID to protect. Default: 1.
    pub min_votes_recover: usize,
    /// Minimum fraction of total weighted votes the winning candidate must
    /// receive. Default: 0.55 (slight majority).
    pub min_vote_weight_frac: f32,
    /// H-reprojection gate (pixels) used by rough-homography fallback
    /// assignments. Intentionally loose to tolerate significant distortion.
    pub h_reproj_gate_px: f64,
    /// Enable rough-homography fallback for unresolved markers.
    pub homography_fallback_enable: bool,
    /// Minimum trusted markers required before attempting homography fallback.
    pub homography_min_trusted: usize,
    /// Minimum inliers required for fallback homography RANSAC acceptance.
    pub homography_min_inliers: usize,
    /// Maximum number of iterative correction passes. Default: 5.
    pub max_iters: usize,
    /// When `true`, remove markers that cannot be verified or corrected.
    /// When `false` (default), clear their ID (set to `None`) and keep the
    /// detection so its geometry is available for debugging.
    pub remove_unverified: bool,
    /// Minimum decode confidence for bootstrapping trusted seeds when no
    /// homography is available. Default: 0.7.
    pub seed_min_decode_confidence: f32,
}

impl IdCorrectionConfig {
    /// Returns the minimum vote threshold appropriate for a marker at `i`:
    /// `min_votes_recover` if the marker has no current ID, `min_votes` otherwise.
    pub(crate) fn effective_min_votes(&self, has_id: bool) -> usize {
        if has_id {
            self.min_votes
        } else {
            self.min_votes_recover
        }
    }
}

impl Default for IdCorrectionConfig {
    fn default() -> Self {
        Self {
            enable: true,
            auto_search_radius_outer_muls: vec![2.4, 2.9, 3.5, 4.2, 5.0],
            consistency_outer_mul: 3.2,
            consistency_min_neighbors: 1,
            consistency_min_support_edges: 1,
            consistency_max_contradiction_frac: 0.5,
            confirm_by_consistency: true,
            soft_lock_exact_decode: true,
            min_votes: 2,
            min_votes_recover: 1,
            min_vote_weight_frac: 0.55,
            h_reproj_gate_px: 30.0,
            homography_fallback_enable: true,
            homography_min_trusted: 24,
            homography_min_inliers: 12,
            max_iters: 5,
            remove_unverified: false,
            seed_min_decode_confidence: 0.7,
        }
    }
}

/// Configuration for automatic recovery of markers where the inner edge was
/// incorrectly fitted as the outer ellipse.
///
/// After all markers are finalized, each marker's outer radius is compared to
/// the median outer radius of its k nearest neighbors. A ratio well below 1.0
/// (see `ratio_threshold`) indicates the outer fit locked onto the inner ring
/// edge. When enabled, the detector re-attempts the outer fit for flagged
/// markers using the neighbor median radius as the corrected expected radius.
///
/// The recovery re-fit uses a tight 4 px search window (to exclude the inner
/// ring) combined with relaxed quality gates (`min_theta_consistency`,
/// `min_ring_depth`, `refine_halfwidth_px`) suited to the blurry/soft edges
/// that typically cause inner-as-outer confusion. A post-fit size gate
/// (`size_gate_tolerance`) prevents the relaxed estimator from re-locking onto
/// the inner ring even under the relaxed thresholds.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct InnerAsOuterRecoveryConfig {
    /// Enable inner-as-outer recovery (default: `true`).
    pub enable: bool,
    /// Neighbor-radius ratio below which a marker is considered anomalous and
    /// a re-fit is attempted.
    ///
    /// Default: `0.75`. Markers with `own_radius / neighbor_median < 0.75` are
    /// flagged. A value of ~0.64 is expected for an inner-as-outer confusion
    /// (inner radius ≈ 0.49 × outer radius → ratio = 0.49 / 0.77 ≈ 0.64 when
    /// inner accounts for the inner/outer ratio of the ring marker geometry).
    pub ratio_threshold: f32,
    /// Number of nearest neighbors used to compute the median outer radius.
    ///
    /// Default: `6` (matching the hex-lattice neighbor count). Self is always
    /// excluded by passing k+1 to the neighbor function.
    pub k_neighbors: usize,
    /// Minimum fraction of angular samples (rays) whose radial peak must agree
    /// with the selected hypothesis radius during the recovery re-estimation.
    ///
    /// Lower than the production default (0.35) because blurry outer edges
    /// scatter per-θ peaks more widely. Default: `0.18`.
    pub min_theta_consistency: f32,
    /// Minimum fraction of angular rays with valid (in-bounds) samples during
    /// the recovery re-estimation. Default: `0.40`.
    pub min_theta_coverage: f32,
    /// Minimum signed intensity depth at a candidate outer edge point during
    /// recovery edge collection. Lower than production (0.05) to tolerate
    /// blur-smeared intensity gradients. Default: `0.02`.
    pub min_ring_depth: f32,
    /// Per-ray radius refinement half-width (pixels) during recovery. Wider
    /// than production (1.0 px) to catch the flat-topped derivative peaks that
    /// occur under blur. Default: `2.5`.
    pub refine_halfwidth_px: f32,
    /// Maximum allowed fractional deviation of the recovered outer radius from
    /// the neighbor-median corrected radius: `|r_recovered - r_corrected| /
    /// r_corrected ≤ size_gate_tolerance`. Prevents the relaxed estimator from
    /// accepting a re-locked inner-ring fit. Default: `0.25`.
    pub size_gate_tolerance: f32,
}

impl Default for InnerAsOuterRecoveryConfig {
    fn default() -> Self {
        Self {
            enable: true,
            ratio_threshold: 0.75,
            k_neighbors: 6,
            min_theta_consistency: 0.18,
            min_theta_coverage: 0.40,
            min_ring_depth: 0.02,
            refine_halfwidth_px: 2.5,
            size_gate_tolerance: 0.25,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_correction_config_defaults_are_stable() {
        let cfg = IdCorrectionConfig::default();
        assert!(cfg.enable);
        assert_eq!(
            cfg.auto_search_radius_outer_muls,
            vec![2.4, 2.9, 3.5, 4.2, 5.0]
        );
        assert!((cfg.consistency_outer_mul - 3.2).abs() < 1e-9);
        assert_eq!(cfg.consistency_min_neighbors, 1);
        assert_eq!(cfg.consistency_min_support_edges, 1);
        assert!((cfg.consistency_max_contradiction_frac - 0.5).abs() < 1e-6);
        assert!(cfg.soft_lock_exact_decode);
        assert_eq!(cfg.min_votes, 2);
        assert_eq!(cfg.min_votes_recover, 1);
        assert!((cfg.min_vote_weight_frac - 0.55).abs() < 1e-6);
        assert!((cfg.h_reproj_gate_px - 30.0).abs() < 1e-9);
        assert!(cfg.homography_fallback_enable);
        assert_eq!(cfg.homography_min_trusted, 24);
        assert_eq!(cfg.homography_min_inliers, 12);
        assert_eq!(cfg.max_iters, 5);
        assert!(!cfg.remove_unverified);
        assert!((cfg.seed_min_decode_confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn id_correction_config_unknown_fields_are_silently_ignored() {
        // Old config JSON with fields that no longer exist — serde should ignore them
        // and fill in current defaults for the missing new fields.
        let json = r#"{
            "enable": true,
            "neighbor_search_radius_px": null,
            "homography_ransac_max_iters": 1200,
            "homography_use_recovered_seeds": false,
            "homography_candidate_top_k": 19,
            "min_votes": 2,
            "min_votes_recover": 1,
            "min_vote_weight_frac": 0.55,
            "h_reproj_gate_px": 30.0,
            "max_iters": 5,
            "remove_unverified": false,
            "seed_min_decode_confidence": 0.7
        }"#;
        let cfg: IdCorrectionConfig =
            serde_json::from_str(json).expect("deserialize old id correction config");
        assert_eq!(
            cfg.auto_search_radius_outer_muls,
            vec![2.4, 2.9, 3.5, 4.2, 5.0]
        );
        assert!((cfg.consistency_outer_mul - 3.2).abs() < 1e-9);
        assert_eq!(cfg.consistency_min_neighbors, 1);
        assert_eq!(cfg.consistency_min_support_edges, 1);
        assert!(cfg.homography_fallback_enable);
        assert_eq!(cfg.homography_min_trusted, 24);
        assert_eq!(cfg.homography_min_inliers, 12);
    }

    #[test]
    fn projective_center_accepts_pre_0_8_shift_key() {
        // 0.7.x configs used `max_center_shift_px`; the alias must keep them
        // loading into the renamed field.
        let cfg: ProjectiveCenterConfig =
            serde_json::from_str(r#"{"max_center_shift_px": 7.5}"#).expect("deserialize");
        assert_eq!(cfg.max_correction_shift_px, Some(7.5));

        let cfg: ProjectiveCenterConfig =
            serde_json::from_str(r#"{"max_correction_shift_px": 3.0}"#).expect("deserialize");
        assert_eq!(cfg.max_correction_shift_px, Some(3.0));
    }
}
