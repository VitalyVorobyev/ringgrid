use super::diagnostics::{count_inconsistent_remaining, estimate_adjacent_spacing_px};
use super::types::Trust;
use super::vote::resolve_id_conflicts;
use super::workspace::IdCorrectionWorkspace;

pub(super) fn cleanup_unverified_markers(ws: &mut IdCorrectionWorkspace<'_>) {
    if ws.config.remove_unverified {
        let mut i = 0usize;
        while i < ws.markers.len() {
            if !ws.trust[i].is_trusted() && ws.markers[i].id.is_some() {
                ws.markers.remove(i);
                ws.trust.remove(i);
                ws.outer_radii_px.remove(i);
                ws.stats.n_markers_removed += 1;
            } else {
                i += 1;
            }
        }
    } else {
        for i in 0..ws.markers.len() {
            if !ws.trust[i].is_trusted() && ws.markers[i].id.is_some() {
                ws.markers[i].id = None;
                ws.stats.n_ids_cleared += 1;
            }
        }
    }
}

pub(super) fn finalize_correction_stats(ws: &mut IdCorrectionWorkspace<'_>) {
    let n_conflicts_cleared = resolve_id_conflicts(ws.markers);
    ws.stats.n_ids_cleared += n_conflicts_cleared;

    for i in 0..ws.markers.len() {
        if ws.markers[i].id.is_none() {
            ws.trust[i] = Trust::Untrusted;
        }
    }

    ws.stats.n_verified = ws
        .markers
        .iter()
        .enumerate()
        .filter(|(i, m)| m.id.is_some() && ws.trust[*i].is_trusted())
        .count();

    ws.stats.n_inconsistent_remaining = count_inconsistent_remaining(ws);
    ws.stats.pitch_px_estimated = estimate_adjacent_spacing_px(ws);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::config::IdCorrectionConfig;
    use crate::detector::marker_build::MarkerRecord;
    use crate::marker::codec::Codebook;
    use crate::target::TargetLayout;

    fn marker(id: Option<usize>, center: [f64; 2], confidence: f32) -> MarkerRecord {
        MarkerRecord {
            id,
            center,
            confidence,
            ..MarkerRecord::default()
        }
    }

    fn min_cyclic() -> usize {
        Codebook::default().min_cyclic_dist()
    }

    #[test]
    fn cleanup_clears_ids_of_untrusted_markers_by_default() {
        let board = TargetLayout::default_hex();
        let cfg = IdCorrectionConfig {
            remove_unverified: false,
            ..IdCorrectionConfig::default()
        };
        let mut markers = vec![
            marker(Some(0), [100.0, 100.0], 0.9),
            marker(Some(1), [140.0, 100.0], 0.5),
            marker(None, [180.0, 100.0], 0.4),
        ];
        let mut ws = IdCorrectionWorkspace::new(&mut markers, &board, &cfg, min_cyclic());
        // Marker 0 is trusted; markers 1 and 2 remain untrusted.
        ws.trust[0] = Trust::AnchorStrong;

        cleanup_unverified_markers(&mut ws);

        assert_eq!(ws.markers.len(), 3, "clearing keeps the detections");
        assert_eq!(ws.markers[0].id, Some(0));
        assert_eq!(ws.markers[1].id, None, "untrusted decoded id is cleared");
        assert_eq!(ws.markers[2].id, None);
        assert_eq!(
            ws.stats.n_ids_cleared, 1,
            "only the untrusted decode counts"
        );
        assert_eq!(ws.stats.n_markers_removed, 0);
    }

    #[test]
    fn cleanup_removes_markers_when_configured() {
        let board = TargetLayout::default_hex();
        let cfg = IdCorrectionConfig {
            remove_unverified: true,
            ..IdCorrectionConfig::default()
        };
        let mut markers = vec![
            marker(Some(0), [100.0, 100.0], 0.9),
            marker(Some(1), [140.0, 100.0], 0.5),
            marker(None, [180.0, 100.0], 0.4),
        ];
        let mut ws = IdCorrectionWorkspace::new(&mut markers, &board, &cfg, min_cyclic());
        ws.trust[0] = Trust::AnchorStrong;

        cleanup_unverified_markers(&mut ws);

        // Only the untrusted marker that still carried an id is removed; the
        // parallel state vectors must stay length-consistent.
        assert_eq!(ws.markers.len(), 2);
        assert_eq!(ws.trust.len(), 2);
        assert_eq!(ws.outer_radii_px.len(), 2);
        assert_eq!(ws.stats.n_markers_removed, 1);
        assert!(ws.markers.iter().any(|m| m.id == Some(0)));
        assert!(ws.markers.iter().any(|m| m.id.is_none()));
    }

    #[test]
    fn finalize_resolves_duplicate_ids_and_counts_verified() {
        let board = TargetLayout::default_hex();
        let cfg = IdCorrectionConfig::default();
        let mut markers = vec![
            marker(Some(0), [100.0, 100.0], 0.9),
            marker(Some(0), [140.0, 100.0], 0.5), // duplicate id, lower confidence
            marker(Some(1), [180.0, 100.0], 0.8),
        ];
        let mut ws = IdCorrectionWorkspace::new(&mut markers, &board, &cfg, min_cyclic());
        ws.trust[0] = Trust::AnchorStrong;
        ws.trust[1] = Trust::AnchorWeak;
        ws.trust[2] = Trust::RecoveredLocal;

        finalize_correction_stats(&mut ws);

        // The lower-confidence duplicate loses its id and is demoted to untrusted.
        assert_eq!(ws.markers[0].id, Some(0));
        assert_eq!(ws.markers[1].id, None);
        assert_eq!(ws.trust[1], Trust::Untrusted);
        assert_eq!(ws.markers[2].id, Some(1));
        // Two distinct trusted ids survive.
        assert_eq!(ws.stats.n_verified, 2);
        assert!(ws.stats.n_ids_cleared >= 1);
    }
}
