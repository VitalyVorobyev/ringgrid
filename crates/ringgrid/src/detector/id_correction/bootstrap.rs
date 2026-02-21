use super::types::Trust;
use super::workspace::{is_exact_decode, IdCorrectionWorkspace};

pub(super) fn bootstrap_trust_anchors(ws: &mut IdCorrectionWorkspace<'_>) -> usize {
    let mut n_seeds = 0usize;
    for i in 0..ws.markers.len() {
        let Some(id) = ws.markers[i].id else {
            continue;
        };
        if !ws.board_index.id_to_xy.contains_key(&id) {
            continue;
        }
        let exact = is_exact_decode(&ws.markers[i]);
        let decode_conf = ws.markers[i]
            .decode
            .as_ref()
            .map(|d| d.decode_confidence)
            .unwrap_or(ws.markers[i].confidence);
        if exact {
            ws.trust[i] = Trust::AnchorStrong;
            n_seeds += 1;
        } else if decode_conf >= ws.config.seed_min_decode_confidence {
            ws.trust[i] = Trust::AnchorWeak;
            n_seeds += 1;
        }
    }
    n_seeds
}
