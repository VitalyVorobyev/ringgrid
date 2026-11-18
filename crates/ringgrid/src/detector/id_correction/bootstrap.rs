use super::types::Trust;
use super::workspace::{IdCorrectionWorkspace, is_exact_decode};

pub(super) fn bootstrap_trust_anchors(ws: &mut IdCorrectionWorkspace<'_>) -> usize {
    let mut n_seeds = 0usize;
    for i in 0..ws.markers.len() {
        let Some(id) = ws.markers[i].id else {
            continue;
        };
        if !ws.board_index.id_to_xy.contains_key(&id) {
            continue;
        }
        // A marker with its own (exact) decode is a strong seed; one whose ID
        // came from elsewhere (e.g. inner-as-outer recovery) is a weak seed
        // when its marker confidence is high enough.
        if is_exact_decode(&ws.markers[i]) {
            ws.trust[i] = Trust::AnchorStrong;
            n_seeds += 1;
        } else if ws.markers[i].confidence >= ws.config.seed_min_confidence {
            ws.trust[i] = Trust::AnchorWeak;
            n_seeds += 1;
        }
    }
    n_seeds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::config::IdCorrectionConfig;
    use crate::detector::id_correction::workspace::IdCorrectionWorkspace;
    use crate::detector::marker_build::MarkerRecord;
    use crate::marker::decode::DecodeMetrics;
    use crate::target::TargetLayout;

    /// A marker with `id`; `decoded` attaches its own decode of that ID.
    fn marker(id: Option<usize>, decoded: bool, confidence: f32) -> MarkerRecord {
        MarkerRecord {
            id,
            confidence,
            decode: id.filter(|_| decoded).map(|best_id| DecodeMetrics {
                observed_word: 0,
                best_id,
                rotation: 0.0,
            }),
            ..MarkerRecord::default()
        }
    }

    #[test]
    fn bootstrap_classifies_exact_weak_and_untrusted_seeds() {
        let board = TargetLayout::default_hex();
        let cfg = IdCorrectionConfig::default();

        let mut markers = vec![
            // Own decode on a valid board id → strong anchor.
            marker(Some(0), true, 0.2),
            // ID without its own decode, confident marker → weak anchor.
            marker(Some(1), false, 0.85),
            // ID without its own decode, low confidence → stays untrusted.
            marker(Some(2), false, 0.30),
            // Undecoded marker → untrusted.
            marker(None, false, 0.9),
        ];

        let mut ws = IdCorrectionWorkspace::new(&mut markers, &board, &cfg);
        let n_seeds = bootstrap_trust_anchors(&mut ws);

        assert_eq!(n_seeds, 2);
        assert_eq!(ws.trust[0], Trust::AnchorStrong);
        assert_eq!(ws.trust[1], Trust::AnchorWeak);
        assert_eq!(ws.trust[2], Trust::Untrusted);
        assert_eq!(ws.trust[3], Trust::Untrusted);
    }

    #[test]
    fn bootstrap_ignores_ids_not_on_board() {
        let board = TargetLayout::default_hex();
        let cfg = IdCorrectionConfig::default();

        // An id far beyond the board's marker set must be skipped even with
        // its own decode.
        let off_board = board.n_cells() + 100_000;
        let mut markers = vec![marker(Some(off_board), true, 1.0)];

        let mut ws = IdCorrectionWorkspace::new(&mut markers, &board, &cfg);
        let n_seeds = bootstrap_trust_anchors(&mut ws);

        assert_eq!(n_seeds, 0);
        assert_eq!(ws.trust[0], Trust::Untrusted);
    }
}
