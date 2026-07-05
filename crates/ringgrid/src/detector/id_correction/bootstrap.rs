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
        let exact = is_exact_decode(&ws.markers[i], ws.codebook_min_cyclic_dist);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::config::IdCorrectionConfig;
    use crate::detector::id_correction::workspace::IdCorrectionWorkspace;
    use crate::detector::marker_build::MarkerRecord;
    use crate::marker::codec::Codebook;
    use crate::marker::decode::DecodeMetrics;
    use crate::target::TargetLayout;

    fn marker(
        id: Option<usize>,
        best_dist: u8,
        margin: u8,
        decode_confidence: f32,
    ) -> MarkerRecord {
        MarkerRecord {
            id,
            confidence: decode_confidence,
            decode: id.map(|best_id| DecodeMetrics {
                observed_word: 0,
                best_id,
                best_rotation: 0,
                best_dist,
                margin,
                decode_confidence,
            }),
            ..MarkerRecord::default()
        }
    }

    #[test]
    fn bootstrap_classifies_exact_weak_and_untrusted_seeds() {
        let board = TargetLayout::default_hex();
        let cfg = IdCorrectionConfig::default();
        let min_cyclic = Codebook::default().min_cyclic_dist();

        let mut markers = vec![
            // Exact decode on a valid board id → strong anchor.
            marker(Some(0), 0, min_cyclic as u8, 1.0),
            // Non-exact but confident decode → weak anchor.
            marker(Some(1), 1, 1, 0.85),
            // Non-exact, low-confidence decode → stays untrusted.
            marker(Some(2), 1, 1, 0.30),
            // Undecoded marker → untrusted.
            marker(None, 0, 0, 0.9),
        ];

        let mut ws = IdCorrectionWorkspace::new(&mut markers, &board, &cfg, min_cyclic);
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
        let min_cyclic = Codebook::default().min_cyclic_dist();

        // An id far beyond the board's marker set must be skipped even with an
        // otherwise-exact decode.
        let off_board = board.n_cells() + 100_000;
        let mut markers = vec![marker(Some(off_board), 0, min_cyclic as u8, 1.0)];

        let mut ws = IdCorrectionWorkspace::new(&mut markers, &board, &cfg, min_cyclic);
        let n_seeds = bootstrap_trust_anchors(&mut ws);

        assert_eq!(n_seeds, 0);
        assert_eq!(ws.trust[0], Trust::Untrusted);
    }
}
