/// Solve 3×3 linear system `A x = b` via Gaussian elimination with partial pivoting.
/// Returns `None` if the system is singular (|pivot| < 1e-12).
#[allow(clippy::needless_range_loop)]
pub(super) fn solve_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<[f64; 3]> {
    let mut aug = [[0.0f64; 4]; 3];
    for i in 0..3 {
        aug[i][..3].copy_from_slice(&a[i]);
        aug[i][3] = b[i];
    }
    for col in 0..3 {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in col + 1..3 {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        for row in col + 1..3 {
            let factor = aug[row][col] / pivot;
            for k in col..4 {
                let v = aug[col][k];
                aug[row][k] -= factor * v;
            }
        }
    }
    // Back substitution
    let mut x = [0.0f64; 3];
    for i in (0..3).rev() {
        x[i] = aug[i][3];
        for j in i + 1..3 {
            let v = aug[i][j];
            x[i] -= v * x[j];
        }
        x[i] /= aug[i][i];
    }
    Some(x)
}

/// Fit a 2D affine transform (board-mm → image-px) from N ≥ 3 point correspondences.
///
/// Returns a `[2 × 3]` matrix `A` such that `image ≈ A × [board_x, board_y, 1]ᵀ`.
/// Uses normal equations (least squares when N > 3, exact when N = 3).
pub(crate) fn fit_local_affine(
    board_pts: &[[f64; 2]],
    image_pts: &[[f64; 2]],
) -> Option<[[f64; 3]; 2]> {
    debug_assert_eq!(board_pts.len(), image_pts.len());
    if board_pts.len() < 3 {
        return None;
    }

    let mut xtx = [[0.0f64; 3]; 3];
    let mut xtu = [0.0f64; 3];
    let mut xtv = [0.0f64; 3];

    for (bp, ip) in board_pts.iter().zip(image_pts) {
        let row = [bp[0], bp[1], 1.0];
        for j in 0..3 {
            for k in 0..3 {
                xtx[j][k] += row[j] * row[k];
            }
            xtu[j] += row[j] * ip[0];
            xtv[j] += row[j] * ip[1];
        }
    }

    let row_u = solve_3x3(&xtx, &xtu)?;
    let row_v = solve_3x3(&xtx, &xtv)?;
    Some([row_u, row_v])
}

/// Apply a `[2 × 3]` affine transform that maps board-mm coordinates to image
/// or working-frame pixel coordinates.
pub(crate) fn affine_to_image(affine: &[[f64; 3]; 2], board_xy: [f64; 2]) -> [f64; 2] {
    [
        affine[0][0] * board_xy[0] + affine[0][1] * board_xy[1] + affine[0][2],
        affine[1][0] * board_xy[0] + affine[1][1] * board_xy[1] + affine[1][2],
    ]
}

/// Invert the 2×3 affine and apply it to an image-space point to recover the
/// board-mm position. Returns `None` if the 2×2 sub-matrix is singular.
pub(super) fn affine_to_board(affine: &[[f64; 3]; 2], image_xy: [f64; 2]) -> Option<[f64; 2]> {
    let a = affine[0][0];
    let b = affine[0][1];
    let c = affine[0][2];
    let d = affine[1][0];
    let e = affine[1][1];
    let f = affine[1][2];
    let det = a * e - b * d;
    if det.abs() < 1e-12 {
        return None;
    }
    let u = image_xy[0] - c;
    let v = image_xy[1] - f;
    Some([(e * u - b * v) / det, (a * v - d * u) / det])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_3x3_recovers_known_solution() {
        // A x = b with a hand-picked non-singular A and known x.
        let a = [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]];
        let x_true = [1.0, -2.0, 3.0];
        let b = [
            a[0][0] * x_true[0] + a[0][1] * x_true[1] + a[0][2] * x_true[2],
            a[1][0] * x_true[0] + a[1][1] * x_true[1] + a[1][2] * x_true[2],
            a[2][0] * x_true[0] + a[2][1] * x_true[1] + a[2][2] * x_true[2],
        ];
        let x = solve_3x3(&a, &b).expect("non-singular system must solve");
        for k in 0..3 {
            assert!(
                (x[k] - x_true[k]).abs() < 1e-12,
                "x[{k}] = {} != {}",
                x[k],
                x_true[k]
            );
        }
    }

    #[test]
    fn solve_3x3_requires_pivoting() {
        // Zero leading pivot: partial pivoting must swap rows to stay stable.
        let a = [[0.0, 2.0, 1.0], [1.0, 0.0, 1.0], [3.0, 1.0, 0.0]];
        let x_true = [0.5, -1.5, 2.25];
        let b = [
            a[0][1] * x_true[1] + a[0][2] * x_true[2],
            a[1][0] * x_true[0] + a[1][2] * x_true[2],
            a[2][0] * x_true[0] + a[2][1] * x_true[1],
        ];
        let x = solve_3x3(&a, &b).expect("pivoting must handle zero leading entry");
        for k in 0..3 {
            assert!((x[k] - x_true[k]).abs() < 1e-12, "x[{k}] = {}", x[k]);
        }
    }

    #[test]
    fn solve_3x3_returns_none_for_singular() {
        // Row 3 = row 1 + row 2 → rank-deficient → no unique solution.
        let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]];
        let b = [1.0, 2.0, 3.0];
        assert!(solve_3x3(&a, &b).is_none());
    }

    /// Reference affine applied by hand: `[a b c; d e f] · [x, y, 1]ᵀ`.
    fn apply(m: &[[f64; 3]; 2], p: [f64; 2]) -> [f64; 2] {
        [
            m[0][0] * p[0] + m[0][1] * p[1] + m[0][2],
            m[1][0] * p[0] + m[1][1] * p[1] + m[1][2],
        ]
    }

    #[test]
    fn fit_local_affine_recovers_exact_transform_from_minimal_set() {
        // Exactly 3 non-collinear correspondences determine the affine exactly.
        let truth = [[2.0, 0.5, 3.0], [-0.3, 1.7, -1.0]];
        let board = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let image: Vec<[f64; 2]> = board.iter().map(|&p| apply(&truth, p)).collect();

        let fitted = fit_local_affine(&board, &image).expect("3 non-collinear points fit");
        for r in 0..2 {
            for c in 0..3 {
                assert!(
                    (fitted[r][c] - truth[r][c]).abs() < 1e-9,
                    "A[{r}][{c}] = {} != {}",
                    fitted[r][c],
                    truth[r][c]
                );
            }
        }
    }

    #[test]
    fn fit_local_affine_overdetermined_exact_data_recovers_transform() {
        // Least squares over 5 exact (noise-free) points must reproduce the map.
        let truth = [[1.1, -0.4, 12.0], [0.35, 0.9, -7.0]];
        let board = [[0.0, 0.0], [3.0, 0.0], [0.0, 2.0], [4.0, 5.0], [-2.0, 1.5]];
        let image: Vec<[f64; 2]> = board.iter().map(|&p| apply(&truth, p)).collect();

        let fitted = fit_local_affine(&board, &image).expect("overdetermined fit");
        for r in 0..2 {
            for c in 0..3 {
                assert!(
                    (fitted[r][c] - truth[r][c]).abs() < 1e-9,
                    "A[{r}][{c}] = {}",
                    fitted[r][c]
                );
            }
        }
    }

    #[test]
    fn fit_local_affine_rejects_fewer_than_three_points() {
        let board = [[0.0, 0.0], [1.0, 0.0]];
        let image = [[0.0, 0.0], [2.0, 0.0]];
        assert!(fit_local_affine(&board, &image).is_none());
    }

    #[test]
    fn fit_local_affine_rejects_collinear_points() {
        // All board points lie on the x-axis → XᵀX is singular → None.
        let board = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let image = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        assert!(fit_local_affine(&board, &image).is_none());
    }

    #[test]
    fn affine_to_image_matches_matrix_product() {
        let m = [[2.0, 0.5, 3.0], [-0.3, 1.7, -1.0]];
        let p = [4.0, -2.0];
        let got = affine_to_image(&m, p);
        let expect = apply(&m, p);
        assert!((got[0] - expect[0]).abs() < 1e-12);
        assert!((got[1] - expect[1]).abs() < 1e-12);
    }

    #[test]
    fn affine_to_board_inverts_affine_to_image() {
        // Round-trip: board → image → board must return the original point.
        let m = [[2.0, 0.5, 3.0], [-0.3, 1.7, -1.0]];
        let board_pt = [1.25, -3.5];
        let image_pt = affine_to_image(&m, board_pt);
        let recovered = affine_to_board(&m, image_pt).expect("invertible 2x2 sub-matrix");
        assert!((recovered[0] - board_pt[0]).abs() < 1e-12);
        assert!((recovered[1] - board_pt[1]).abs() < 1e-12);
    }

    #[test]
    fn affine_to_board_returns_none_for_singular_linear_part() {
        // Linear part [[1,2],[2,4]] has zero determinant → not invertible.
        let m = [[1.0, 2.0, 5.0], [2.0, 4.0, 7.0]];
        assert!(affine_to_board(&m, [0.0, 0.0]).is_none());
    }
}
