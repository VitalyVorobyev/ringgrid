//! Versioned debug dump schema for ringgrid detection.
//!
//! This schema intentionally reuses production detector structures wherever
//! possible. Debug-only structs are limited to stage orchestration metadata
//! (candidate decisions, dedup actions, stage notes).

use serde::{Deserialize, Serialize};

use crate::board_layout::BoardLayout;
use crate::conic::Ellipse;
use crate::detector::proposal::{Proposal, ProposalConfig};
use crate::detector::{
    CircleRefinementMethod, CompletionAttemptRecord, CompletionParams, CompletionStats,
    DebugCollectConfig, DetectConfig, MarkerScalePrior, ProjectiveCenterParams,
};
use crate::homography::RansacHomographyConfig;
use crate::marker::decode::{DecodeDiagnostics, DecodeResult};
use crate::marker::{DecodeConfig, MarkerSpec};
use crate::pixelmap::SelfUndistortConfig;
use crate::ring::edge_sample::{EdgeSampleConfig, EdgeSampleResult};
use crate::ring::inner_estimate::InnerEstimate;
use crate::ring::outer_estimate::{OuterEstimate, OuterEstimationConfig};
use crate::{DecodeMetrics, DetectedMarker, FitMetrics, RansacStats};

pub const DEBUG_SCHEMA_V7: &str = "ringgrid.debug.v7";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugDump {
    pub schema_version: String,
    pub image: ImageDebug,
    pub detect_config: DetectConfigSnapshot,
    pub stages: StagesDebug,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageDebug {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardSummary {
    pub name: String,
    pub pitch_mm: f32,
    pub rows: usize,
    pub long_row_cols: usize,
    pub marker_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub marker_span_mm: Option<[f32; 2]>,
    pub marker_outer_radius_mm: f32,
    pub marker_inner_radius_mm: f32,
}

impl From<&BoardLayout> for BoardSummary {
    fn from(board: &BoardLayout) -> Self {
        Self {
            name: board.name.clone(),
            pitch_mm: board.pitch_mm,
            rows: board.rows,
            long_row_cols: board.long_row_cols,
            marker_count: board.n_markers(),
            marker_span_mm: board.marker_span_mm(),
            marker_outer_radius_mm: board.marker_outer_radius_mm(),
            marker_inner_radius_mm: board.marker_inner_radius_mm(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectConfigSnapshot {
    pub marker_scale: MarkerScalePrior,
    pub proposal: ProposalConfig,
    pub edge_sample: EdgeSampleConfig,
    pub outer_estimation: OuterEstimationConfig,
    pub decode: DecodeConfig,
    pub marker_spec: MarkerSpec,
    pub circle_refinement: CircleRefinementMethod,
    pub projective_center: ProjectiveCenterParams,
    pub completion: CompletionParams,
    pub min_semi_axis: f64,
    pub max_semi_axis: f64,
    pub max_aspect_ratio: f64,
    pub dedup_radius: f64,
    pub use_global_filter: bool,
    pub ransac_homography: RansacHomographyConfig,
    pub refine_with_h: bool,
    pub self_undistort: SelfUndistortConfig,
    pub board: BoardSummary,
    pub debug_collect: DebugCollectConfig,
}

impl DetectConfigSnapshot {
    pub fn from_config(config: &DetectConfig, debug_collect: &DebugCollectConfig) -> Self {
        Self {
            marker_scale: config.marker_scale,
            proposal: config.proposal.clone(),
            edge_sample: config.edge_sample.clone(),
            outer_estimation: config.outer_estimation.clone(),
            decode: config.decode.clone(),
            marker_spec: config.marker_spec.clone(),
            circle_refinement: config.circle_refinement,
            projective_center: config.projective_center.clone(),
            completion: config.completion.clone(),
            min_semi_axis: config.min_semi_axis,
            max_semi_axis: config.max_semi_axis,
            max_aspect_ratio: config.max_aspect_ratio,
            dedup_radius: config.dedup_radius,
            use_global_filter: config.use_global_filter,
            ransac_homography: config.ransac_homography.clone(),
            refine_with_h: config.refine_with_h,
            self_undistort: config.self_undistort.clone(),
            board: BoardSummary::from(&config.board),
            debug_collect: debug_collect.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagesDebug {
    pub stage0_proposals: StageDebug,
    pub stage1_fit_decode: StageDebug,
    pub stage2_dedup: DedupDebug,
    pub stage3_ransac: RansacDebug,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage4_refine: Option<RefineDebug>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage5_completion: Option<CompletionDebug>,
    #[serde(rename = "final")]
    pub final_: FinalDebug,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageDebug {
    pub n_total: usize,
    pub n_recorded: usize,
    pub candidates: Vec<CandidateDebug>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateDebug {
    pub cand_idx: usize,
    pub proposal: Proposal,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ring_fit: Option<RingFitDebug>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode: Option<DecodeDebug>,
    pub decision: DecisionDebug,
    pub derived: DerivedDebug,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionDebug {
    pub status: DecisionStatus,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DecisionStatus {
    Accepted,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedDebug {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub center_xy: Option<[f64; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingFitDebug {
    pub center_xy_fit: [f64; 2],
    pub edge: EdgeSampleResult,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outer_estimation: Option<OuterEstimate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_outer_hypothesis: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ellipse_outer: Option<Ellipse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ellipse_inner: Option<Ellipse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inner_estimation: Option<InnerEstimate>,
    pub fit: FitMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inner_points_fit: Option<Vec<[f64; 2]>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeDebug {
    pub diagnostics: DecodeDiagnostics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<DecodeResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_metrics: Option<DecodeMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupDebug {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub kept_by_proximity: Vec<KeptByProximityDebug>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub kept_by_id: Vec<KeptByIdDebug>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeptByProximityDebug {
    pub kept_cand_idx: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dropped_cand_indices: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeptByIdDebug {
    pub id: usize,
    pub kept_cand_idx: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dropped_cand_indices: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RansacDebug {
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<crate::homography::RansacHomographyResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<RansacStats>,
    pub correspondences_used: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inlier_ids: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outlier_ids: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_id_error_px: Vec<PerIdErrorDebug>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerIdErrorDebug {
    pub id: usize,
    pub reproj_err_px: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefineDebug {
    pub h_prior: [[f64; 3]; 3],
    pub refined_markers: Vec<RefinedMarkerDebug>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub h_refit: Option<[[f64; 3]; 3]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinedMarkerDebug {
    pub id: usize,
    pub prior_center_xy: [f64; 2],
    pub refined_marker: DetectedMarker,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionDebug {
    pub enabled: bool,
    pub params: CompletionParams,
    pub attempted: Vec<CompletionAttemptRecord>,
    pub stats: CompletionStats,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalDebug {
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
    fn debug_dump_json_roundtrip_minimal() {
        let cfg =
            DetectConfig::from_target_and_marker_diameter(crate::BoardLayout::default(), 32.0);
        let dbg_cfg = DebugCollectConfig {
            image_path: None,
            marker_diameter_px: 32.0,
            max_candidates: 10,
            store_points: false,
        };
        let dd = DebugDump {
            schema_version: DEBUG_SCHEMA_V7.to_string(),
            image: ImageDebug {
                path: None,
                width: 640,
                height: 480,
            },
            detect_config: DetectConfigSnapshot::from_config(&cfg, &dbg_cfg),
            stages: StagesDebug {
                stage0_proposals: StageDebug {
                    n_total: 0,
                    n_recorded: 0,
                    candidates: vec![],
                    notes: vec![],
                },
                stage1_fit_decode: StageDebug {
                    n_total: 0,
                    n_recorded: 0,
                    candidates: vec![],
                    notes: vec![],
                },
                stage2_dedup: DedupDebug {
                    kept_by_proximity: vec![],
                    kept_by_id: vec![],
                    notes: vec![],
                },
                stage3_ransac: RansacDebug {
                    enabled: false,
                    result: None,
                    stats: None,
                    correspondences_used: 0,
                    inlier_ids: vec![],
                    outlier_ids: vec![],
                    per_id_error_px: vec![],
                    notes: vec![],
                },
                stage4_refine: None,
                stage5_completion: None,
                final_: FinalDebug {
                    h_final: None,
                    detections: vec![],
                    notes: vec![],
                },
            },
        };

        let s = serde_json::to_string_pretty(&dd).unwrap();
        let dd2: DebugDump = serde_json::from_str(&s).unwrap();
        assert_eq!(dd2.schema_version, DEBUG_SCHEMA_V7);
        assert_eq!(dd2.image.width, 640);
    }
}
