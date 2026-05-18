//! Public facade guard.
//!
//! This integration test exists purely to pin the set of names re-exported by
//! `crates/ringgrid/src/lib.rs`. It `use`s every type, function, and module
//! that the crate root exports. If a public facade name is ever removed or
//! renamed, this file fails to compile — turning an accidental breaking change
//! into an immediate, local build failure.
//!
//! It is an interim, hand-maintained public-API snapshot guard. The proper
//! tool, `cargo public-api`, does not currently run in this repository because
//! rustdoc fails while processing the `fixed` dependency on the local
//! toolchain (see `API_REVISION.md`). When that is resolved, a real
//! `cargo public-api` snapshot check should supersede this file.
//!
//! When the facade in `lib.rs` changes intentionally, update the imports below
//! to match.

// ── Detector facade and proposal-only convenience helpers ───────────────────
use ringgrid::{
    Detector, propose_with_heatmap_and_marker_diameter, propose_with_heatmap_and_marker_scale,
    propose_with_marker_diameter, propose_with_marker_scale,
};

// ── Proposal module (standalone ellipse center detection) ───────────────────
use ringgrid::proposal;
use ringgrid::{Proposal, ProposalConfig, ProposalResult};
use ringgrid::{find_ellipse_centers, find_ellipse_centers_with_heatmap};

// ── Slim primary result types ───────────────────────────────────────────────
use ringgrid::{DetectedMarker, DetectionFrame, DetectionResult};

// ── Diagnostics channel ─────────────────────────────────────────────────────
use ringgrid::{DecodeMetrics, DetectionDiagnostics, MarkerDiagnostics, RansacStats};
use ringgrid::{DetectionSource, FitMetrics, InnerFitReason, InnerFitStatus};

// ── Configuration ───────────────────────────────────────────────────────────
use ringgrid::{
    AdvancedDetectConfig, CircleRefinementMethod, CompletionParams, DetectConfig,
    IdCorrectionConfig, InnerAsOuterRecoveryConfig, InnerFitConfig, MarkerScalePrior,
    OuterFitConfig, ProjectiveCenterParams, ProposalDownscale, RansacHomographyConfig, ScaleTier,
    ScaleTiers, SeedProposalParams,
};

// ── Sub-configs ─────────────────────────────────────────────────────────────
use ringgrid::{AngularAggregator, CodebookProfile, DecodeConfig, GradPolarity};
use ringgrid::{EdgeSampleConfig, OuterEstimationConfig};

// ── Codebook diagnostics ────────────────────────────────────────────────────
use ringgrid::{CodebookInfo, CodewordMatch, codebook_info, decode_word};

// ── Geometry ────────────────────────────────────────────────────────────────
use ringgrid::{BoardLayout, BoardLayoutLoadError, BoardLayoutValidationError, BoardMarker};
use ringgrid::{Ellipse, RansacConfig};
use ringgrid::{MarkerSpec, PngTargetOptions, SvgTargetOptions, TargetGenerationError};

// ── Camera / distortion ─────────────────────────────────────────────────────
use ringgrid::{
    CameraIntrinsics, CameraModel, DivisionModel, PixelMapper, RadialTangentialDistortion,
    SelfUndistortConfig, SelfUndistortResult, UndistortConfig,
};

/// Reference every imported name once so an unused-import warning cannot mask a
/// missing facade export, and exercise a couple of them at runtime.
#[test]
fn facade_names_resolve() {
    // Types referenced in type position — proves the path still resolves.
    fn _assert_named<T>() {}
    _assert_named::<Detector>();
    _assert_named::<Proposal>();
    _assert_named::<ProposalConfig>();
    _assert_named::<ProposalResult>();
    _assert_named::<DetectedMarker>();
    _assert_named::<DetectionFrame>();
    _assert_named::<DetectionResult>();
    _assert_named::<DetectionSource>();
    _assert_named::<FitMetrics>();
    _assert_named::<InnerFitReason>();
    _assert_named::<InnerFitStatus>();
    _assert_named::<DecodeMetrics>();
    _assert_named::<DetectionDiagnostics>();
    _assert_named::<MarkerDiagnostics>();
    _assert_named::<RansacStats>();
    _assert_named::<AdvancedDetectConfig>();
    _assert_named::<CircleRefinementMethod>();
    _assert_named::<CompletionParams>();
    _assert_named::<DetectConfig>();
    _assert_named::<IdCorrectionConfig>();
    _assert_named::<InnerAsOuterRecoveryConfig>();
    _assert_named::<InnerFitConfig>();
    _assert_named::<MarkerScalePrior>();
    _assert_named::<OuterFitConfig>();
    _assert_named::<ProjectiveCenterParams>();
    _assert_named::<ProposalDownscale>();
    _assert_named::<RansacHomographyConfig>();
    _assert_named::<ScaleTier>();
    _assert_named::<ScaleTiers>();
    _assert_named::<SeedProposalParams>();
    _assert_named::<AngularAggregator>();
    _assert_named::<CodebookProfile>();
    _assert_named::<DecodeConfig>();
    _assert_named::<GradPolarity>();
    _assert_named::<EdgeSampleConfig>();
    _assert_named::<OuterEstimationConfig>();
    _assert_named::<CodebookInfo>();
    _assert_named::<CodewordMatch>();
    _assert_named::<BoardLayout>();
    _assert_named::<BoardLayoutLoadError>();
    _assert_named::<BoardLayoutValidationError>();
    _assert_named::<BoardMarker>();
    _assert_named::<Ellipse>();
    _assert_named::<RansacConfig>();
    _assert_named::<MarkerSpec>();
    _assert_named::<PngTargetOptions>();
    _assert_named::<SvgTargetOptions>();
    _assert_named::<TargetGenerationError>();
    _assert_named::<CameraIntrinsics>();
    _assert_named::<CameraModel>();
    _assert_named::<DivisionModel>();
    _assert_named::<RadialTangentialDistortion>();
    _assert_named::<SelfUndistortConfig>();
    _assert_named::<SelfUndistortResult>();
    _assert_named::<UndistortConfig>();

    // Free functions referenced as values — proves the fn paths still resolve.
    // Each is bound to an explicitly typed `fn` item so the path is checked
    // without relying on type inference.
    let _: fn(&image::GrayImage, &BoardLayout, f32) -> ProposalResult =
        propose_with_heatmap_and_marker_diameter;
    let _: fn(&image::GrayImage, &BoardLayout, MarkerScalePrior) -> ProposalResult =
        propose_with_heatmap_and_marker_scale;
    let _: fn(&image::GrayImage, &BoardLayout, f32) -> Vec<Proposal> = propose_with_marker_diameter;
    let _: fn(&image::GrayImage, &BoardLayout, MarkerScalePrior) -> Vec<Proposal> =
        propose_with_marker_scale;
    let _: fn(&image::GrayImage, &ProposalConfig) -> Vec<Proposal> = find_ellipse_centers;
    let _: fn(&image::GrayImage, &ProposalConfig) -> ProposalResult =
        find_ellipse_centers_with_heatmap;

    // Trait used as a bound — proves the trait path still resolves.
    fn _accepts_mapper<M: PixelMapper>(_m: &M) {}

    // The `proposal` module path must stay public.
    let _: proposal::ProposalConfig = proposal::ProposalConfig::default();

    // Exercise a couple of facade items at runtime.
    let info: CodebookInfo = codebook_info(CodebookProfile::Base);
    assert!(info.len > 0, "base codebook profile must be non-empty");

    let decoded: CodewordMatch = decode_word(0, CodebookProfile::Base);
    assert_eq!(decoded.profile, CodebookProfile::Base);
}
