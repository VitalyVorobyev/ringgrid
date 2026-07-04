//! Detection configuration: top-level [`DetectConfig`] / [`AdvancedDetectConfig`]
//! plus per-stage and scale-prior config types in submodules.

mod fit;
mod scale;
mod stages;

pub use fit::{InnerFitConfig, OuterFitConfig};
pub use scale::{
    CircleRefinementMethod, MarkerScalePrior, ProposalDownscale, ScaleTier, ScaleTiers,
};
pub use stages::{
    CompletionConfig, IdCorrectionConfig, InnerAsOuterRecoveryConfig, ProjectiveCenterConfig,
    SeedProposalConfig,
};

use crate::conic::RansacConfig;
use crate::marker::{DecodeConfig, MarkerSpecConfig};
use crate::pixelmap::SelfUndistortConfig;
use crate::ring::{EdgeSampleConfig, OuterEstimationConfig};
use crate::target::{MarkerCoding, TargetLayout};

use crate::proposal::ProposalConfig;

fn proposal_spacing_ratio_for_target(target: &TargetLayout) -> f32 {
    let outer_diameter_mm = 2.0 * target.ring().outer_radius_mm;
    if !(outer_diameter_mm.is_finite() && outer_diameter_mm > 0.0) {
        return 1.0;
    }

    let spacing_mm = target.min_center_spacing_mm();
    if !(spacing_mm.is_finite() && spacing_mm > 0.0) {
        return 1.0;
    }

    spacing_mm / outer_diameter_mm
}

pub(crate) fn derive_proposal_config(
    target: &TargetLayout,
    marker_scale: MarkerScalePrior,
    base: &ProposalConfig,
) -> ProposalConfig {
    let [d_min, d_max] = marker_scale.diameter_range_px();
    let outer_radius_max_px = d_max * 0.5;
    let spacing_ratio = proposal_spacing_ratio_for_target(target);
    let spacing_min_px = spacing_ratio * d_min;
    let spacing_max_px = spacing_ratio * d_max;

    let nms_radius = (0.16 * d_min).max(4.0);

    let mut proposal = base.clone();
    proposal.r_min = (0.15 * spacing_min_px).max(2.0);
    proposal.r_max = (0.45 * spacing_max_px).min(1.35 * outer_radius_max_px);
    proposal.min_distance = nms_radius.max(0.85 * spacing_min_px);

    proposal
}

/// Advanced per-stage tuning parameters for the detection pipeline.
///
/// These fields control the internal behavior of individual pipeline stages
/// (proposal generation, edge sampling, ellipse fitting, decoding, ID
/// correction, completion). Most users never need to touch them — the
/// [`DetectConfig`] constructors derive sensible scale-dependent values.
/// Override individual fields for fine-grained tuning of difficult scenes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct AdvancedDetectConfig {
    /// Outer edge estimation configuration (anchored on `marker_scale`).
    pub outer_estimation: OuterEstimationConfig,
    /// Proposal generation configuration.
    pub proposal: ProposalConfig,
    /// Seed-injection controls for multi-pass proposal generation.
    pub seed_proposals: SeedProposalConfig,
    /// Radial edge sampling configuration.
    pub edge_sample: EdgeSampleConfig,
    /// Marker decode configuration.
    pub decode: DecodeConfig,
    /// Marker geometry specification and estimator controls.
    pub marker_spec: MarkerSpecConfig,
    /// Robust inner ellipse fitting controls shared across pipeline stages.
    pub inner_fit: InnerFitConfig,
    /// Robust outer ellipse fitting controls shared across pipeline stages.
    pub outer_fit: OuterFitConfig,
    /// Projective-center recovery controls.
    pub projective_center: ProjectiveCenterConfig,
    /// Homography-guided completion controls.
    pub completion: CompletionConfig,
    /// Minimum semi-axis for a valid outer ellipse.
    /// Derived from `marker_scale` by the config constructors; do not set directly.
    #[serde(skip)]
    pub(crate) min_semi_axis: f64,
    /// Maximum semi-axis for a valid outer ellipse.
    /// Derived from `marker_scale` by the config constructors; do not set directly.
    #[serde(skip)]
    pub(crate) max_semi_axis: f64,
    /// Maximum aspect ratio (a/b) for a valid ellipse.
    pub max_aspect_ratio: f64,
    /// NMS dedup radius for final markers (pixels).
    pub dedup_radius: f64,
    /// Enable global homography filtering (requires board spec).
    pub use_global_filter: bool,
    /// Final precision-first geometric verification gate.
    ///
    /// After the final homography, each decoded marker is checked against its
    /// hex-neighbor midpoint prediction (locally affine, so distortion-robust)
    /// and, as a gross-blunder backstop, against its final-H reprojection
    /// residual. Both thresholds adapt to the observed inlier residual
    /// distribution, so the gate stays recall-safe on clean and lens-distorted
    /// boards alike. Markers the lattice judges geometrically inconsistent are
    /// removed, so only trusted board correspondences reach the output.
    ///
    /// Disable (`false`) to keep every decoded marker and apply your own
    /// filtering. Default: `true`.
    pub geometric_verify: bool,
    /// RANSAC homography configuration.
    pub ransac_homography: RansacConfig,
    /// Structural ID verification and correction using hex neighborhood consensus.
    pub id_correction: IdCorrectionConfig,
    /// Automatic recovery for markers where the inner edge was incorrectly
    /// fitted as the outer ellipse.
    pub inner_as_outer_recovery: InnerAsOuterRecoveryConfig,
    /// Optional image downscaling before proposal generation.
    ///
    /// When markers are large, running proposals on a smaller image is much
    /// faster while proposal coordinates are approximate anyway (downstream
    /// stages refine at full resolution). Default: `Off`.
    pub proposal_downscale: ProposalDownscale,
}

impl Default for AdvancedDetectConfig {
    fn default() -> Self {
        Self {
            outer_estimation: OuterEstimationConfig::default(),
            proposal: ProposalConfig::default(),
            seed_proposals: SeedProposalConfig::default(),
            edge_sample: EdgeSampleConfig::default(),
            decode: DecodeConfig::default(),
            marker_spec: MarkerSpecConfig::default(),
            inner_fit: InnerFitConfig::default(),
            outer_fit: OuterFitConfig::default(),
            projective_center: ProjectiveCenterConfig::default(),
            completion: CompletionConfig::default(),
            min_semi_axis: 3.0,
            max_semi_axis: 15.0,
            max_aspect_ratio: 3.0,
            dedup_radius: 6.0,
            use_global_filter: true,
            geometric_verify: true,
            ransac_homography: RansacConfig {
                max_iters: 2000,
                inlier_threshold: 5.0,
                min_inliers: 6,
                seed: 0,
            },
            id_correction: IdCorrectionConfig::default(),
            inner_as_outer_recovery: InnerAsOuterRecoveryConfig::default(),
            proposal_downscale: ProposalDownscale::default(),
        }
    }
}

/// Top-level detection configuration.
///
/// Holds the durable user choices that shape detection: target layout, marker
/// scale prior, center-refinement method, and self-undistort policy. All
/// per-stage tuning lives under [`AdvancedDetectConfig`] in the `advanced`
/// field.
///
/// Use one of the recommended constructors rather than constructing directly:
///
/// - [`DetectConfig::from_target`] — default scale prior
/// - [`DetectConfig::from_target_and_scale_prior`] — explicit scale range
/// - [`DetectConfig::from_target_and_marker_diameter`] — fixed diameter hint
///
/// These constructors auto-derive scale-dependent parameters (proposal radii,
/// edge search windows, validation bounds) from the target geometry and marker
/// scale prior. Individual fields can be tuned after construction.
///
/// A `DetectConfig` deserialized from JSON has a default `target` (the target
/// layout is not serialized) and zeroed derived bounds. Call
/// [`DetectConfig::with_target`] to attach the real target layout and
/// re-derive all scale- and geometry-coupled parameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct DetectConfig {
    /// Target layout: marker cell positions and geometry.
    ///
    /// Not serialized — supply it via a constructor or [`DetectConfig::with_target`].
    #[serde(skip)]
    pub target: TargetLayout,
    /// Marker diameter prior (range) in working-frame pixels.
    pub marker_scale: MarkerScalePrior,
    /// Post-fit circle refinement method selector.
    pub circle_refinement: CircleRefinementMethod,
    /// Self-undistort estimation controls.
    pub self_undistort: SelfUndistortConfig,
    /// Advanced per-stage tuning parameters.
    pub advanced: AdvancedDetectConfig,
}

impl DetectConfig {
    /// Build a configuration with scale-dependent parameters derived from a
    /// marker diameter range and a runtime target layout.
    ///
    /// This is the recommended constructor for library users. After calling it,
    /// individual fields can be overridden as needed.
    pub fn from_target_and_scale_prior(
        target: impl Into<TargetLayout>,
        marker_scale: MarkerScalePrior,
    ) -> Self {
        let mut cfg = Self {
            target: target.into(),
            marker_scale: marker_scale.normalized(),
            ..Default::default()
        };
        apply_target_geometry_priors(&mut cfg);
        apply_marker_scale_prior(&mut cfg);
        cfg
    }

    /// Build a configuration from target layout and default marker scale prior.
    pub fn from_target(target: impl Into<TargetLayout>) -> Self {
        Self::from_target_and_scale_prior(target, MarkerScalePrior::default())
    }

    /// Build a configuration from target layout and a fixed marker diameter hint.
    pub fn from_target_and_marker_diameter(
        target: impl Into<TargetLayout>,
        diameter_px: f32,
    ) -> Self {
        Self::from_target_and_scale_prior(
            target,
            MarkerScalePrior::from_nominal_diameter_px(diameter_px),
        )
    }

    /// Attach a target layout and re-derive all scale- and geometry-coupled
    /// parameters.
    ///
    /// `target` is `#[serde(skip)]`, so a `DetectConfig` deserialized from
    /// JSON carries a default target layout and zeroed derived ellipse bounds.
    /// This is the single entry point for consumers that load a config from
    /// JSON: it sets the real target, then re-runs the geometry- and
    /// scale-prior derivation (proposal radii, edge search windows, validation
    /// bounds) so callers do not duplicate that logic.
    pub fn with_target(mut self, target: impl Into<TargetLayout>) -> Self {
        self.target = target.into();
        apply_target_geometry_priors(&mut self);
        apply_marker_scale_prior(&mut self);
        self
    }

    /// Apply a partial JSON overlay and return the resulting config.
    ///
    /// The overlay is a (possibly partial) `DetectConfig` JSON object; stage
    /// tuning nests under `"advanced"`. It is merged recursively onto this
    /// config's JSON view (objects merge key-by-key, other values replace),
    /// so a deeply nested section like
    /// `{"advanced": {"completion": {"enable": false}}}` overrides only the
    /// named leaves. The result is deserialized and the target re-attached
    /// via [`Self::with_target`], re-deriving all target- and scale-coupled
    /// fields.
    ///
    /// Legacy pre-0.8 key names in the overlay are accepted and normalized
    /// (serde aliases only cover whole-document deserialization; an overlay
    /// merged onto a serialized base would otherwise put both the old and
    /// new spelling in one object, which serde rejects as a duplicate
    /// field).
    pub fn with_json_overlay(
        &self,
        mut overlay: serde_json::Value,
    ) -> Result<Self, serde_json::Error> {
        normalize_legacy_overlay_keys(&mut overlay);
        let mut merged = serde_json::to_value(self)?;
        merge_json_value(&mut merged, overlay);
        let config: Self = serde_json::from_value(merged)?;
        Ok(config.with_target(self.target.clone()))
    }

    /// Update marker scale prior and re-derive all scale-coupled parameters.
    pub fn set_marker_scale_prior(&mut self, marker_scale: MarkerScalePrior) {
        self.marker_scale = marker_scale.normalized();
        apply_marker_scale_prior(self);
    }

    /// Update marker scale prior from a fixed marker diameter hint.
    pub fn set_marker_diameter_hint_px(&mut self, diameter_px: f32) {
        self.set_marker_scale_prior(MarkerScalePrior::from_nominal_diameter_px(diameter_px));
    }

    #[cfg(test)]
    fn proposal_spacing_ratio(&self) -> f32 {
        proposal_spacing_ratio_for_target(&self.target)
    }

    #[cfg(test)]
    fn proposal_spacing_min_px(&self) -> f32 {
        let [d_min, _] = self.marker_scale.diameter_range_px();
        self.proposal_spacing_ratio() * d_min
    }

    #[cfg(test)]
    fn proposal_spacing_max_px(&self) -> f32 {
        let [_, d_max] = self.marker_scale.diameter_range_px();
        self.proposal_spacing_ratio() * d_max
    }
}

/// Recursively merge `overlay` into `base`: objects merge key-by-key, all
/// other values replace.
fn merge_json_value(base: &mut serde_json::Value, overlay: serde_json::Value) {
    match (base, overlay) {
        (serde_json::Value::Object(base_obj), serde_json::Value::Object(overlay_obj)) => {
            for (key, overlay_value) in overlay_obj {
                match base_obj.get_mut(&key) {
                    Some(base_value) => merge_json_value(base_value, overlay_value),
                    None => {
                        base_obj.insert(key, overlay_value);
                    }
                }
            }
        }
        (base_slot, overlay_value) => *base_slot = overlay_value,
    }
}

/// Rename pre-0.8 config keys in an overlay to their current spellings.
///
/// Currently: `advanced.projective_center.max_center_shift_px` →
/// `max_correction_shift_px` (renamed in 0.8.0). The current spelling wins
/// when an overlay carries both.
fn normalize_legacy_overlay_keys(overlay: &mut serde_json::Value) {
    let Some(projective_center) = overlay
        .get_mut("advanced")
        .and_then(|advanced| advanced.get_mut("projective_center"))
        .and_then(|value| value.as_object_mut())
    else {
        return;
    };
    if let Some(value) = projective_center.remove("max_center_shift_px")
        && !projective_center.contains_key("max_correction_shift_px")
    {
        projective_center.insert("max_correction_shift_px".to_string(), value);
    }
}

impl Default for DetectConfig {
    fn default() -> Self {
        let mut cfg = Self {
            target: TargetLayout::default_hex(),
            marker_scale: MarkerScalePrior::default(),
            circle_refinement: CircleRefinementMethod::default(),
            self_undistort: SelfUndistortConfig::default(),
            advanced: AdvancedDetectConfig::default(),
        };
        apply_target_geometry_priors(&mut cfg);
        apply_marker_scale_prior(&mut cfg);
        cfg
    }
}

fn apply_marker_scale_prior(config: &mut DetectConfig) {
    config.marker_scale = config.marker_scale.normalized();
    let [d_min, d_max] = config.marker_scale.diameter_range_px();
    let d_nom = config.marker_scale.nominal_diameter_px();
    let outer_radius_min_px = d_min * 0.5;
    let outer_radius_max_px = d_max * 0.5;
    let adv = &mut config.advanced;
    adv.proposal = derive_proposal_config(&config.target, config.marker_scale, &adv.proposal);

    // Edge sampling range
    adv.edge_sample.r_max = outer_radius_max_px * 2.0;
    adv.edge_sample.r_min = 1.5;
    let desired_halfwidth = ((outer_radius_max_px - outer_radius_min_px) * 0.5).max(2.0);
    let base_halfwidth = OuterEstimationConfig::default().search_halfwidth_px;
    adv.outer_estimation.search_halfwidth_px = desired_halfwidth.max(base_halfwidth);

    // Ellipse validation bounds
    adv.min_semi_axis = (outer_radius_min_px as f64 * 0.3).max(2.0);
    adv.max_semi_axis = (outer_radius_max_px as f64 * 2.5).max(adv.min_semi_axis);

    // Completion ROI
    adv.completion.roi_radius_px = ((d_nom as f64 * 0.75).clamp(24.0, 80.0)) as f32;

    // `projective_center.max_correction_shift_px` is deliberately NOT derived
    // here: `None` means "auto" and the nominal-diameter fallback is applied at
    // the use site, so explicit config values survive target re-derivation.
}

fn apply_target_geometry_priors(config: &mut DetectConfig) {
    let ring = config.target.ring();
    let outer = ring.outer_radius_mm;
    let inner = ring.inner_radius_mm;
    // Stroked (coded) rings put the detected edges half a stroke width past
    // the centerline radii; plain annuli expose the radii directly.
    let edge_pad = match config.target.coding() {
        MarkerCoding::Coded16(spec) => {
            if !(spec.ring_width_mm.is_finite() && spec.ring_width_mm > 0.0) {
                return;
            }
            0.5 * spec.ring_width_mm
        }
        MarkerCoding::Plain => 0.0,
    };
    if !(outer.is_finite() && inner.is_finite()) || outer <= 0.0 || inner <= 0.0 || inner >= outer {
        return;
    }
    let inner_edge = (inner - edge_pad).max(outer * 0.05);
    let outer_edge = outer + edge_pad;
    if inner_edge > 0.0 && inner_edge < outer_edge {
        let r_inner_expected = (inner_edge / outer_edge).clamp(0.1, 0.95);
        config.advanced.marker_spec.r_inner_expected = r_inner_expected;
        config.advanced.decode.code_band_ratio = (0.5 * (1.0 + r_inner_expected)).clamp(0.2, 0.98);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_config_includes_fit_configs() {
        let cfg = DetectConfig::default();
        assert_eq!(cfg.advanced.inner_fit.min_points, 20);
        assert_eq!(cfg.advanced.inner_fit.ransac.min_inliers, 8);
        assert_eq!(cfg.advanced.outer_fit.min_direct_fit_points, 6);
        assert_eq!(cfg.advanced.outer_fit.ransac.min_inliers, 6);
    }

    #[test]
    fn marker_scale_prior_derives_spacing_aware_proposal_geometry() {
        let cfg = DetectConfig::from_target(TargetLayout::default_hex());
        let spacing_ratio =
            cfg.target.min_center_spacing_mm() / (2.0 * cfg.target.ring().outer_radius_mm);
        let [d_min, d_max] = cfg.marker_scale.diameter_range_px();
        let spacing_min_px = spacing_ratio * d_min;
        let spacing_max_px = spacing_ratio * d_max;
        let outer_radius_max_px = 0.5 * d_max;

        assert!((cfg.proposal_spacing_ratio() - spacing_ratio).abs() < 1.0e-6);
        assert!((cfg.proposal_spacing_min_px() - spacing_min_px).abs() < 1.0e-6);
        assert!((cfg.proposal_spacing_max_px() - spacing_max_px).abs() < 1.0e-6);
        assert!((cfg.advanced.proposal.r_min - (0.15 * spacing_min_px).max(2.0)).abs() < 1.0e-6);
        assert!(
            (cfg.advanced.proposal.r_max - (0.45 * spacing_max_px).min(1.35 * outer_radius_max_px))
                .abs()
                < 1.0e-6
        );
        let expected_nms = (0.16 * d_min).max(4.0);
        let expected_min_dist = expected_nms.max(0.85 * spacing_min_px);
        assert!((cfg.advanced.proposal.min_distance - expected_min_dist).abs() < 1.0e-6);
    }

    #[test]
    fn fixed_marker_hint_keeps_spacing_aware_seed_distance() {
        let cfg = DetectConfig::from_target_and_marker_diameter(TargetLayout::default_hex(), 32.0);
        assert!((cfg.advanced.proposal.r_min - 6.928203).abs() < 1.0e-5);
        assert!((cfg.advanced.proposal.r_max - 20.784609).abs() < 1.0e-5);
        assert!((cfg.advanced.proposal.min_distance - 39.259_815).abs() < 1.0e-5);
    }

    #[test]
    fn detect_config_json_roundtrip_preserves_effective_config() {
        let target = TargetLayout::default_hex();
        let mut original = DetectConfig::from_target_and_marker_diameter(target.clone(), 32.0);
        original.circle_refinement = CircleRefinementMethod::None;
        original.advanced.completion.enable = false;
        original.advanced.id_correction.max_iters = 9;
        original.self_undistort.enable = true;

        let json = serde_json::to_string(&original).expect("serialize DetectConfig");
        let deserialized: DetectConfig =
            serde_json::from_str(&json).expect("deserialize DetectConfig");
        // target is #[serde(skip)] — reattach it to re-derive geometry/scale fields.
        let restored = deserialized.with_target(target);

        assert_eq!(restored.circle_refinement, original.circle_refinement);
        assert_eq!(
            restored.advanced.completion.enable,
            original.advanced.completion.enable
        );
        assert_eq!(
            restored.advanced.id_correction.max_iters,
            original.advanced.id_correction.max_iters
        );
        assert_eq!(
            restored.self_undistort.enable,
            original.self_undistort.enable
        );
        // Scale-derived fields match because with_target re-runs derivation.
        assert!(
            (restored.advanced.proposal.r_min - original.advanced.proposal.r_min).abs() < 1.0e-6
        );
        assert!((restored.advanced.min_semi_axis - original.advanced.min_semi_axis).abs() < 1.0e-9);
        assert!((restored.advanced.max_semi_axis - original.advanced.max_semi_axis).abs() < 1.0e-9);
        assert_eq!(restored.target.n_cells(), original.target.n_cells());
    }

    #[test]
    fn json_overlay_merges_partial_sections() {
        let base = DetectConfig::default();
        let overlay = serde_json::json!({
            "advanced": { "completion": { "enable": false } }
        });
        let merged = base.with_json_overlay(overlay).expect("overlay applies");
        assert!(!merged.advanced.completion.enable);
        // Sibling fields keep their base values.
        assert_eq!(
            merged.advanced.completion.reproj_gate_px,
            base.advanced.completion.reproj_gate_px
        );
        assert_eq!(merged.target.n_cells(), base.target.n_cells());
    }

    #[test]
    fn json_overlay_accepts_pre_0_8_shift_key() {
        // The serialized base already carries `max_correction_shift_px`; a
        // 0.7.x overlay uses the old name. Without normalization the merged
        // object holds both spellings and serde rejects the whole config as
        // a duplicate field (regression: codex review on PR #54).
        let overlay = serde_json::json!({
            "advanced": { "projective_center": {
                "max_center_shift_px": 7.5,
                "use_expected_ratio": false,
            } }
        });
        let merged = DetectConfig::default()
            .with_json_overlay(overlay)
            .expect("legacy overlay key must keep loading");
        assert!(!merged.advanced.projective_center.use_expected_ratio);
        // Explicit shift values survive target re-derivation since 0.8.0
        // (`None` means "auto"; `with_target` no longer clobbers the field).
        assert_eq!(
            merged.advanced.projective_center.max_correction_shift_px,
            Some(7.5)
        );

        // Mixed spellings in one overlay must also load (current name wins
        // pre-merge; the merged document never carries both keys).
        let overlay = serde_json::json!({
            "advanced": { "projective_center": {
                "max_center_shift_px": 7.5,
                "max_correction_shift_px": 3.0,
            } }
        });
        DetectConfig::default()
            .with_json_overlay(overlay)
            .expect("mixed-spelling overlay must load");
    }
}
