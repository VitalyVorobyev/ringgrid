//! Versioned debug dump schema for ringgrid detection.
//!
//! This module defines a stable JSON schema intended for manual inspection and
//! tooling. The normal detector output (`DetectionResult`) must remain stable
//! and lightweight; this debug dump is optionally large and is produced only
//! when explicitly requested by the CLI.

use crate::{DetectedMarker, FitMetrics};
use serde::{Deserialize, Serialize};

pub const DEBUG_SCHEMA_V1: &str = "ringgrid.debug.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugDumpV1 {
    pub schema_version: String,
    pub image: ImageDebugV1,
    pub board: BoardDebugV1,
    pub params: ParamsDebugV1,
    pub stages: StagesDebugV1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageDebugV1 {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardDebugV1 {
    pub pitch_mm: f32,
    /// Convenience scalar for square boards (typically equals board_size_mm\[0\]).
    pub board_mm: f32,
    pub board_size_mm: [f32; 2],
    pub marker_count: usize,
    pub codebook_bits: usize,
    pub codebook_n: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamsDebugV1 {
    pub marker_diameter_px: f64,

    pub proposal: ProposalParamsV1,
    pub edge_sample: EdgeSampleParamsV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outer_estimation: Option<OuterEstimationParamsV1>,
    pub decode: DecodeParamsV1,
    pub marker_spec: crate::marker_spec::MarkerSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projective_center: Option<ProjectiveCenterParamsV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nl_refine: Option<NlRefineParamsV1>,

    pub min_semi_axis: f64,
    pub max_semi_axis: f64,
    pub max_aspect_ratio: f64,
    pub dedup_radius: f64,

    pub use_global_filter: bool,
    pub ransac_homography: RansacHomographyParamsV1,
    pub refine_with_h: bool,

    pub debug: DebugOptionsV1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugOptionsV1 {
    pub max_candidates: usize,
    pub store_points: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalParamsV1 {
    pub r_min: f32,
    pub r_max: f32,
    pub grad_threshold: f32,
    pub nms_radius: f32,
    pub min_vote_frac: f32,
    pub accum_sigma: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeSampleParamsV1 {
    pub n_rays: usize,
    pub r_max: f32,
    pub r_min: f32,
    pub r_step: f32,
    pub min_ring_depth: f32,
    pub min_rays_with_ring: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OuterGradPolarityParamsV1 {
    DarkToLight,
    LightToDark,
    Auto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OuterEstimationParamsV1 {
    pub search_halfwidth_px: f32,
    pub radial_samples: usize,
    pub theta_samples: usize,
    pub aggregator: crate::marker_spec::AngularAggregator,
    pub grad_polarity: OuterGradPolarityParamsV1,
    pub min_theta_coverage: f32,
    pub min_theta_consistency: f32,
    pub allow_two_hypotheses: bool,
    pub second_peak_min_rel: f32,
    pub refine_halfwidth_px: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeParamsV1 {
    pub code_band_ratio: f32,
    pub samples_per_sector: usize,
    pub n_radial_rings: usize,
    pub max_decode_dist: u8,
    pub min_decode_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RansacHomographyParamsV1 {
    pub max_iters: usize,
    pub inlier_threshold: f64,
    pub min_inliers: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectiveCenterParamsV1 {
    pub enabled: bool,
    pub use_expected_ratio: bool,
    pub ratio_penalty_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagesDebugV1 {
    pub stage0_proposals: StageDebugV1,
    pub stage1_fit_decode: StageDebugV1,
    pub stage2_dedup: DedupDebugV1,
    pub stage3_ransac: RansacDebugV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage4_refine: Option<RefineDebugV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage5_completion: Option<CompletionDebugV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage6_nl_refine: Option<NlRefineDebugV1>,
    #[serde(rename = "final")]
    pub final_: FinalDebugV1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NlRefineParamsV1 {
    pub enabled: bool,
    pub max_iters: usize,
    pub huber_delta_mm: f64,
    pub min_points: usize,
    pub reject_shift_mm: f64,
    pub enable_h_refit: bool,
    pub h_refit_iters: usize,
    pub marker_outer_radius_mm: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NlRefineDebugV1 {
    pub enabled: bool,
    pub params: NlRefineParamsV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub h_used: Option<[[f64; 3]; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub h_refit: Option<[[f64; 3]; 3]>,
    pub stats: NlRefineStatsDebugV1,
    pub refined_markers: Vec<NlRefinedMarkerDebugV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NlRefineStatsDebugV1 {
    pub n_inliers: usize,
    pub n_refined: usize,
    pub n_failed: usize,
    pub mean_before_mm: f64,
    pub mean_after_mm: f64,
    pub p95_before_mm: f64,
    pub p95_after_mm: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NlRefineStatusDebugV1 {
    Ok,
    Rejected,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NlRefinedMarkerDebugV1 {
    pub id: usize,
    pub n_points: usize,
    pub init_center_board_mm: [f64; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refined_center_board_mm: Option<[f64; 2]>,
    pub center_img_before: [f64; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub center_img_after: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before_rms_mm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_rms_mm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta_center_mm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_points_img: Option<Vec<[f32; 2]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_points_board_mm: Option<Vec<[f32; 2]>>,
    pub status: NlRefineStatusDebugV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageDebugV1 {
    pub n_total: usize,
    pub n_recorded: usize,
    pub candidates: Vec<CandidateDebugV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateDebugV1 {
    pub cand_idx: usize,
    pub proposal: ProposalDebugV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ring_fit: Option<RingFitDebugV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode: Option<DecodeDebugV1>,
    pub decision: DecisionDebugV1,
    pub derived: DerivedDebugV1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalDebugV1 {
    pub center_xy: [f32; 2],
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionDebugV1 {
    pub status: DecisionStatusV1,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DecisionStatusV1 {
    Accepted,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedDebugV1 {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub center_xy: Option<[f32; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingFitDebugV1 {
    pub center_xy_fit: [f32; 2],
    pub edges: RingEdgesDebugV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outer_estimation: Option<OuterEstimationDebugV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ellipse_outer: Option<EllipseParamsDebugV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ellipse_inner: Option<EllipseParamsDebugV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inner_estimation: Option<InnerEstimationDebugV1>,
    pub metrics: RingFitMetricsDebugV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub points_outer: Option<Vec<[f32; 2]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub points_inner: Option<Vec<[f32; 2]>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingEdgesDebugV1 {
    pub n_angles_total: usize,
    pub n_angles_with_both: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inner_peak_r: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outer_peak_r: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EllipseParamsDebugV1 {
    pub center_xy: [f32; 2],
    pub semi_axes: [f32; 2],
    pub angle: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingFitMetricsDebugV1 {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inlier_ratio_inner: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inlier_ratio_outer: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_resid_inner: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_resid_outer: Option<f32>,
    pub arc_coverage: f32,
    pub valid_inner: bool,
    pub valid_outer: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InnerPolarityDebugV1 {
    Pos,
    Neg,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InnerEstimationStatusDebugV1 {
    Ok,
    Rejected,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnerEstimationDebugV1 {
    pub r_inner_expected: f32,
    pub search_window: [f32; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r_inner_found: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub polarity: Option<InnerPolarityDebugV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_strength: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub theta_consistency: Option<f32>,
    pub status: InnerEstimationStatusDebugV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub radial_response_agg: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r_samples: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OuterEstimationStatusDebugV1 {
    Ok,
    Rejected,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OuterHypothesisDebugV1 {
    pub r_outer_px: f32,
    pub peak_strength: f32,
    pub theta_consistency: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OuterEstimationDebugV1 {
    pub r_outer_expected_px: f32,
    pub search_window_px: [f32; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r_outer_found_px: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub polarity: Option<InnerPolarityDebugV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_strength: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub theta_consistency: Option<f32>,
    pub status: OuterEstimationStatusDebugV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hypotheses: Vec<OuterHypothesisDebugV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_hypothesis: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub radial_response_agg: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r_samples: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeDebugV1 {
    pub sector_means: [f32; 16],
    pub threshold: f32,
    pub observed_word_hex: String,
    pub inverted_used: bool,
    pub r#match: DecodeMatchDebugV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reject_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeMatchDebugV1 {
    pub best_id: usize,
    pub best_rotation: u8,
    pub best_dist: u8,
    pub margin: u8,
    pub decode_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupDebugV1 {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub kept_by_proximity: Vec<KeptByProximityDebugV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub kept_by_id: Vec<KeptByIdDebugV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeptByProximityDebugV1 {
    pub kept_cand_idx: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dropped_cand_indices: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeptByIdDebugV1 {
    pub id: usize,
    pub kept_cand_idx: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dropped_cand_indices: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RansacDebugV1 {
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub h_best: Option<[[f64; 3]; 3]>,
    pub correspondences_used: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inlier_ids: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outlier_ids: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_id_error_px: Option<Vec<PerIdErrorDebugV1>>,
    pub stats: RansacStatsDebugV1,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerIdErrorDebugV1 {
    pub id: usize,
    pub reproj_err_px: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RansacStatsDebugV1 {
    pub iters: usize,
    pub thresh_px: f64,
    pub n_corr: usize,
    pub n_inliers: usize,
    pub mean_err_inliers: f64,
    pub p95_err_inliers: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefineDebugV1 {
    pub h_prior: [[f64; 3]; 3],
    pub refined_markers: Vec<RefinedMarkerDebugV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub h_refit: Option<[[f64; 3]; 3]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinedMarkerDebugV1 {
    pub id: usize,
    pub prior_center_xy: [f32; 2],
    pub refined_center_xy: [f32; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ellipse_outer: Option<EllipseParamsDebugV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ellipse_inner: Option<EllipseParamsDebugV1>,
    pub fit: FitMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionDebugV1 {
    pub enabled: bool,
    pub params: CompletionParamsDebugV1,
    pub attempted: Vec<CompletionAttemptDebugV1>,
    pub stats: CompletionStatsDebugV1,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionParamsDebugV1 {
    pub roi_radius_px: f32,
    pub reproj_gate_px: f32,
    pub min_fit_confidence: f32,
    pub min_arc_coverage: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_attempts: Option<usize>,
    pub image_margin_px: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionAttemptStatusDebugV1 {
    Added,
    SkippedPresent,
    SkippedOob,
    FailedFit,
    FailedGate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionAttemptDebugV1 {
    pub id: usize,
    pub projected_center_xy: [f32; 2],
    pub status: CompletionAttemptStatusDebugV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reproj_err_px: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fit_confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fit: Option<RingFitDebugV1>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionStatsDebugV1 {
    pub n_candidates_total: usize,
    pub n_in_image: usize,
    pub n_attempted: usize,
    pub n_added: usize,
    pub n_failed_fit: usize,
    pub n_failed_gate: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalDebugV1 {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub h_final: Option<[[f64; 3]; 3]>,
    pub detections: Vec<DetectedMarker>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_dump_v1_json_roundtrip_minimal() {
        let dd = DebugDumpV1 {
            schema_version: DEBUG_SCHEMA_V1.to_string(),
            image: ImageDebugV1 {
                path: None,
                width: 640,
                height: 480,
            },
            board: BoardDebugV1 {
                pitch_mm: 8.0,
                board_mm: 200.0,
                board_size_mm: [200.0, 200.0],
                marker_count: 0,
                codebook_bits: 16,
                codebook_n: 0,
            },
            params: ParamsDebugV1 {
                marker_diameter_px: 32.0,
                proposal: ProposalParamsV1 {
                    r_min: 2.0,
                    r_max: 12.0,
                    grad_threshold: 0.05,
                    nms_radius: 7.0,
                    min_vote_frac: 0.1,
                    accum_sigma: 2.0,
                },
                edge_sample: EdgeSampleParamsV1 {
                    n_rays: 48,
                    r_max: 14.0,
                    r_min: 1.5,
                    r_step: 0.5,
                    min_ring_depth: 0.08,
                    min_rays_with_ring: 16,
                },
                outer_estimation: None,
                decode: DecodeParamsV1 {
                    code_band_ratio: 0.76,
                    samples_per_sector: 5,
                    n_radial_rings: 3,
                    max_decode_dist: 3,
                    min_decode_confidence: 0.15,
                },
                marker_spec: crate::marker_spec::MarkerSpec::default(),
                projective_center: None,
                nl_refine: None,
                min_semi_axis: 3.0,
                max_semi_axis: 15.0,
                max_aspect_ratio: 3.0,
                dedup_radius: 6.0,
                use_global_filter: true,
                ransac_homography: RansacHomographyParamsV1 {
                    max_iters: 2000,
                    inlier_threshold: 5.0,
                    min_inliers: 6,
                    seed: 0,
                },
                refine_with_h: true,
                debug: DebugOptionsV1 {
                    max_candidates: 300,
                    store_points: false,
                },
            },
            stages: StagesDebugV1 {
                stage0_proposals: StageDebugV1 {
                    n_total: 0,
                    n_recorded: 0,
                    candidates: Vec::new(),
                    notes: Vec::new(),
                },
                stage1_fit_decode: StageDebugV1 {
                    n_total: 0,
                    n_recorded: 0,
                    candidates: Vec::new(),
                    notes: Vec::new(),
                },
                stage2_dedup: DedupDebugV1 {
                    kept_by_proximity: Vec::new(),
                    kept_by_id: Vec::new(),
                    notes: Vec::new(),
                },
                stage3_ransac: RansacDebugV1 {
                    enabled: false,
                    h_best: None,
                    correspondences_used: 0,
                    inlier_ids: Vec::new(),
                    outlier_ids: Vec::new(),
                    per_id_error_px: None,
                    stats: RansacStatsDebugV1 {
                        iters: 0,
                        thresh_px: 0.0,
                        n_corr: 0,
                        n_inliers: 0,
                        mean_err_inliers: 0.0,
                        p95_err_inliers: 0.0,
                    },
                    notes: Vec::new(),
                },
                stage4_refine: None,
                stage5_completion: None,
                stage6_nl_refine: None,
                final_: FinalDebugV1 {
                    h_final: None,
                    detections: Vec::new(),
                    notes: Vec::new(),
                },
            },
        };

        let s = serde_json::to_string_pretty(&dd).unwrap();
        let dd2: DebugDumpV1 = serde_json::from_str(&s).unwrap();
        assert_eq!(dd2.schema_version, DEBUG_SCHEMA_V1);
        assert_eq!(dd2.image.width, 640);
        assert_eq!(dd2.params.debug.max_candidates, 300);
    }
}
