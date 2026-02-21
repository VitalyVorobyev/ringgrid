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
