use std::collections::BTreeMap;

/// Trust state for per-marker structural ID verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Trust {
    Untrusted,
    AnchorWeak,
    AnchorStrong,
    RecoveredLocal,
    RecoveredHomography,
}

impl Trust {
    #[inline]
    pub(super) fn is_trusted(self) -> bool {
        self != Self::Untrusted
    }

    #[inline]
    pub(super) fn is_anchor(self) -> bool {
        matches!(self, Self::AnchorWeak | Self::AnchorStrong)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum RecoverySource {
    Local,
    Homography,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum ScrubStage {
    Pre,
    Post,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ConsistencyEvidence {
    pub(super) n_neighbors: usize,
    pub(super) support_edges: usize,
    pub(super) contradiction_edges: usize,
    pub(super) contradiction_frac: f64,
    pub(super) vote_mismatch: bool,
    pub(super) vote_winner_frac: f64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct HomographyAssignment {
    pub(super) marker_index: usize,
    pub(super) id: usize,
    pub(super) reproj_err_px: f64,
}

#[derive(Debug, Clone)]
pub(super) struct HomographyFallbackModel {
    pub(super) trusted_by_id: BTreeMap<usize, usize>,
    pub(super) h: nalgebra::Matrix3<f64>,
    pub(super) h_inv: nalgebra::Matrix3<f64>,
    pub(super) n_inliers: usize,
}

/// Statistics produced by the ID verification and correction stage.
#[derive(Debug, Clone, Default)]
pub(crate) struct IdCorrectionStats {
    /// Markers whose decoded ID was replaced with a different, verified ID.
    pub n_ids_corrected: usize,
    /// Markers whose id was `None` and received a new ID.
    pub n_ids_recovered: usize,
    /// Markers assigned by rough-homography fallback.
    pub n_homography_seeded: usize,
    /// Markers whose ID was cleared (`id = None`) after failed verification.
    pub n_ids_cleared: usize,
    /// Markers removed entirely (only when `remove_unverified = true`).
    pub n_markers_removed: usize,
    /// Markers confirmed as structurally consistent with the board layout.
    pub n_verified: usize,
    /// Count of unresolved markers with no trusted neighbors in final diagnosis.
    pub n_unverified_no_neighbors: usize,
    /// Count of unresolved markers with no usable votes in final diagnosis.
    pub n_unverified_no_votes: usize,
    /// Count of unresolved markers blocked by vote-fraction gate in diagnosis.
    pub n_unverified_gate_rejects: usize,
    /// Number of local iterative passes executed across all local stages.
    pub n_iterations: usize,
    /// Estimated board pitch in image pixels (legacy diagnostic field).
    pub pitch_px_estimated: Option<f64>,
    /// IDs cleared by pre-recovery consistency scrub.
    pub n_ids_cleared_inconsistent_pre: usize,
    /// IDs cleared by post-recovery consistency sweep.
    pub n_ids_cleared_inconsistent_post: usize,
    /// Soft-locked exact decodes cleared on strict contradiction.
    pub n_soft_locked_cleared: usize,
    /// IDs recovered by local iterative stage.
    pub n_recovered_local: usize,
    /// IDs recovered by homography fallback stage.
    pub n_recovered_homography: usize,
    /// Remaining IDs that still violate consistency rules after full pipeline.
    pub n_inconsistent_remaining: usize,
}
