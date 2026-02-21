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
pub(super) fn fit_local_affine(
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
