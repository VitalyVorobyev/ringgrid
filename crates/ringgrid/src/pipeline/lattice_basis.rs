//! Lattice-basis check of a grid labeling against the rings' shapes.
//!
//! The grid labeler (`projective_grid::detect_grid`) labels ring centres from
//! their positions alone. On a strongly foreshortened view the image-space
//! diagonal of the lattice can be as short as its foreshortened axis, and the
//! labeler then walks a valid but *sheared* basis of the same point lattice
//! (`c' = A c`, `A` unimodular). A homography fits such a labeling exactly as
//! tightly as the board's own — every centre is still a lattice point — so
//! nothing downstream notices; the board positions are wrong, and the origin
//! dots can never be matched.
//!
//! Each fitted outer ellipse is the image of a circle, so it measures the
//! local affine map of the board — independently of the labels. A labeling
//! is consistent when the homography fitted to it predicts those shapes. This
//! module compares the predicted and fitted shapes under the current labels
//! and under every orientation-preserving unimodular relabeling with small
//! entries, and reports whether the labels stand, must be re-indexed, or
//! cannot be trusted at all.

use nalgebra::{Matrix2, Matrix3};
use projective_grid::Coord;

use crate::conic::Ellipse;
use crate::target::TargetLayout;

use super::frame_xy_mm;

/// Largest entry magnitude of the candidate relabelings. A labeler's sheared
/// basis pairs a lattice axis with a diagonal (entries ≤ 1); 2 covers a
/// second-neighbour diagonal as well.
const MAX_BASIS_ENTRY: i32 = 2;
/// Median shape mismatch (see [`shape_error`]) below which a labeling
/// predicts the rings' shapes. Measured (ADR-026): correct labelings measure
/// a median of 0.03 and p99 of 0.18 over the 663 labelings of the precision
/// sweep, at most 0.56 (12 rings of a hex board tilted 65°), and up to 0.26
/// on blurred, strongly foreshortened real views; sheared labelings measure
/// 1.2–1.7.
const MAX_SHAPE_ERROR: f64 = 0.75;
/// A relabeling replaces the labels only when it predicts the shapes this
/// many times better than they do.
const MIN_IMPROVEMENT: f64 = 2.0;
/// Fewest labeled rings with an outer ellipse to judge a labeling by.
const MIN_RINGS: usize = 6;

/// An integer 2×2 lattice relabeling `c' = A c` with `det A = 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Relabel([[i32; 2]; 2]);

impl Relabel {
    pub(crate) fn apply(&self, c: Coord) -> Coord {
        let [[a, b], [cc, d]] = self.0;
        Coord::new(a * c.u + b * c.v, cc * c.u + d * c.v)
    }

    fn inverse_f64(&self) -> Matrix2<f64> {
        // det = 1: the inverse is the adjugate.
        let [[a, b], [c, d]] = self.0;
        Matrix2::new(f64::from(d), -f64::from(b), -f64::from(c), f64::from(a))
    }
}

/// Outcome of [`check_labeling`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum BasisCheck {
    /// The labels predict the rings' shapes (or too few rings to judge).
    Consistent,
    /// Re-indexing by this relabeling predicts the shapes; the labels do not.
    Relabel(Relabel),
    /// Neither the labels nor any relabeling predicts the shapes.
    Inconsistent,
}

/// One labeled ring: its lattice coordinate and fitted outer ellipse.
pub(crate) struct LabeledRing {
    pub coord: Coord,
    pub outer: Ellipse,
}

/// Judge a labeling by the rings' shapes.
///
/// `h` maps assignment-frame millimetres ([`frame_xy_mm`]) to the frame the
/// ellipses were fitted in.
pub(crate) fn check_labeling(
    rings: &[LabeledRing],
    h: &Matrix3<f64>,
    target: &TargetLayout,
) -> BasisCheck {
    if rings.len() < MIN_RINGS {
        return BasisCheck::Consistent;
    }
    // Local affine map of the homography at each ring, and the ring's
    // normalized fitted shape.
    let local: Vec<(Matrix2<f64>, Matrix2<f64>)> = rings
        .iter()
        .filter_map(|r| {
            let jac = homography_jacobian(h, frame_xy_mm(target, r.coord))?;
            Some((jac, normalized_ellipse_shape(&r.outer)?))
        })
        .collect();
    if local.len() < MIN_RINGS {
        return BasisCheck::Consistent;
    }
    // Relabeling c' = A c moves frame points x = B c to B A c, so the
    // homography consistent with the new labels is H ∘ (B A⁻¹ B⁻¹).
    let basis = frame_basis(target);
    let Some(basis_inv) = basis.try_inverse() else {
        return BasisCheck::Consistent;
    };
    let error_under = |k: &Matrix2<f64>| -> f64 {
        let mut errs: Vec<f64> = local
            .iter()
            .filter_map(|(jac, shape)| shape_error(&(jac * k), shape))
            .collect();
        median(&mut errs).unwrap_or(f64::INFINITY)
    };

    let current = error_under(&Matrix2::identity());
    tracing::debug!(current, "lattice basis: shape error of the labeling");
    if current <= MAX_SHAPE_ERROR {
        return BasisCheck::Consistent;
    }
    let best = unimodular_candidates()
        .map(|a| (a, error_under(&(basis * a.inverse_f64() * basis_inv))))
        .min_by(|x, y| x.1.total_cmp(&y.1));
    match best {
        Some((a, err)) if err <= MAX_SHAPE_ERROR && err * MIN_IMPROVEMENT <= current => {
            tracing::info!(
                ?a,
                current,
                relabeled = err,
                "lattice basis: labeling re-indexed to the basis the ring shapes predict"
            );
            BasisCheck::Relabel(a)
        }
        best => {
            tracing::warn!(
                current,
                best = best.map(|b| b.1),
                "lattice basis: no basis predicts the ring shapes; labeling rejected"
            );
            BasisCheck::Inconsistent
        }
    }
}

/// Orientation-preserving unimodular matrices with entries in
/// `[-MAX_BASIS_ENTRY, MAX_BASIS_ENTRY]`, identity excluded.
fn unimodular_candidates() -> impl Iterator<Item = Relabel> {
    let r = -MAX_BASIS_ENTRY..=MAX_BASIS_ENTRY;
    r.clone()
        .flat_map(move |a| r.clone().map(move |b| (a, b)))
        .flat_map(move |(a, b)| {
            (-MAX_BASIS_ENTRY..=MAX_BASIS_ENTRY)
                .flat_map(move |c| (-MAX_BASIS_ENTRY..=MAX_BASIS_ENTRY).map(move |d| (c, d)))
                .map(move |(c, d)| [[a, b], [c, d]])
        })
        .filter(|m| m[0][0] * m[1][1] - m[0][1] * m[1][0] == 1 && *m != [[1, 0], [0, 1]])
        .map(Relabel)
}

/// Linear part of `frame_xy_mm` in lattice coordinates (columns: the frame
/// steps of `+u` and `+v`).
fn frame_basis(target: &TargetLayout) -> Matrix2<f64> {
    let o = frame_xy_mm(target, Coord::new(0, 0));
    let u = frame_xy_mm(target, Coord::new(1, 0));
    let v = frame_xy_mm(target, Coord::new(0, 1));
    Matrix2::new(u[0] - o[0], v[0] - o[0], u[1] - o[1], v[1] - o[1])
}

/// Jacobian of the homography's point map at `p`.
fn homography_jacobian(h: &Matrix3<f64>, p: [f64; 2]) -> Option<Matrix2<f64>> {
    let w = h[(2, 0)] * p[0] + h[(2, 1)] * p[1] + h[(2, 2)];
    if w.abs() < 1e-12 {
        return None;
    }
    let x = (h[(0, 0)] * p[0] + h[(0, 1)] * p[1] + h[(0, 2)]) / w;
    let y = (h[(1, 0)] * p[0] + h[(1, 1)] * p[1] + h[(1, 2)]) / w;
    Some(
        Matrix2::new(
            h[(0, 0)] - x * h[(2, 0)],
            h[(0, 1)] - x * h[(2, 1)],
            h[(1, 0)] - y * h[(2, 0)],
            h[(1, 1)] - y * h[(2, 1)],
        ) / w,
    )
}

/// The ellipse's shape matrix `R diag(a², b²) Rᵀ`, scaled to unit determinant
/// (size drops out: only aspect and orientation are compared).
fn normalized_ellipse_shape(e: &Ellipse) -> Option<Matrix2<f64>> {
    if !(e.a > 0.0 && e.b > 0.0 && e.a.is_finite() && e.b.is_finite()) {
        return None;
    }
    let (s, c) = e.angle.sin_cos();
    let rot = Matrix2::new(c, -s, s, c);
    let diag = Matrix2::new(e.a / e.b, 0.0, 0.0, e.b / e.a);
    Some(rot * diag * rot.transpose())
}

/// Frobenius distance between the unit-determinant shape of the circle image
/// under the local map `jac` and a fitted ring's normalized shape.
fn shape_error(jac: &Matrix2<f64>, shape: &Matrix2<f64>) -> Option<f64> {
    let det = jac.determinant();
    if !(det.is_finite() && det.abs() > 1e-12) {
        return None;
    }
    let predicted = jac * jac.transpose() / det.abs();
    Some((predicted - shape).norm())
}

fn median(v: &mut [f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mid = v.len() / 2;
    let (_, m, _) = v.select_nth_unstable_by(mid, f64::total_cmp);
    Some(*m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::{LatticeGeometry, MarkerCoding, RectGeometry, RingGeometry};

    fn rect_target() -> TargetLayout {
        TargetLayout::new(
            "t",
            LatticeGeometry::Rect(RectGeometry {
                rows: 20,
                cols: 20,
                pitch_mm: 2.0,
            }),
            RingGeometry {
                outer_radius_mm: 0.8,
                inner_radius_mm: 0.4,
            },
            MarkerCoding::Plain,
            None,
        )
        .expect("target")
    }

    /// An oblique view: strong foreshortening along y, plus perspective.
    fn oblique_h() -> Matrix3<f64> {
        Matrix3::new(60.0, 8.0, 300.0, -4.0, 30.0, 200.0, 0.0005, 0.004, 1.0)
    }

    /// Rings of a 10×10 patch, with outer ellipses as projected circles.
    fn rings(h: &Matrix3<f64>, target: &TargetLayout) -> Vec<LabeledRing> {
        let r = 0.8;
        let mut out = Vec::new();
        for u in 0..10 {
            for v in 0..10 {
                let coord = Coord::new(u, v);
                let p = frame_xy_mm(target, coord);
                let jac = homography_jacobian(h, p).unwrap();
                let m = r * r * jac * jac.transpose();
                let eig = m.symmetric_eigen();
                let (i_max, i_min) = if eig.eigenvalues[0] >= eig.eigenvalues[1] {
                    (0, 1)
                } else {
                    (1, 0)
                };
                let axis = eig.eigenvectors.column(i_max);
                let w = h * nalgebra::Vector3::new(p[0], p[1], 1.0);
                out.push(LabeledRing {
                    coord,
                    outer: Ellipse {
                        cx: w[0] / w[2],
                        cy: w[1] / w[2],
                        a: eig.eigenvalues[i_max].sqrt(),
                        b: eig.eigenvalues[i_min].sqrt(),
                        angle: axis[1].atan2(axis[0]),
                    },
                });
            }
        }
        out
    }

    #[test]
    fn true_labeling_is_consistent() {
        let target = rect_target();
        let h = oblique_h();
        assert_eq!(
            check_labeling(&rings(&h, &target), &h, &target),
            BasisCheck::Consistent
        );
    }

    #[test]
    fn sheared_labeling_is_reindexed_to_the_board_basis() {
        let target = rect_target();
        let h = oblique_h();
        // Label the same rings in the sheared basis c' = S c and give the
        // homography that fits those labels exactly: H ∘ B S⁻¹ B⁻¹.
        let shear = Relabel([[1, 1], [0, 1]]);
        let basis = frame_basis(&target);
        let k = basis * shear.inverse_f64() * basis.try_inverse().unwrap();
        let mut k3 = Matrix3::identity();
        k3.fixed_view_mut::<2, 2>(0, 0).copy_from(&k);
        let h_sheared = h * k3;
        let sheared: Vec<LabeledRing> = rings(&h, &target)
            .into_iter()
            .map(|r| LabeledRing {
                coord: shear.apply(r.coord),
                outer: r.outer,
            })
            .collect();
        let BasisCheck::Relabel(a) = check_labeling(&sheared, &h_sheared, &target) else {
            panic!("sheared labeling not detected");
        };
        // The fix maps the sheared labels back to the board's up to a lattice
        // rotation: the composed map must be a rotation of the square lattice.
        let back = a.apply(shear.apply(Coord::new(1, 0)));
        let up = a.apply(shear.apply(Coord::new(0, 1)));
        assert_eq!(back.u.abs() + back.v.abs(), 1, "{back:?}");
        assert_eq!(up.u.abs() + up.v.abs(), 1, "{up:?}");
    }

    #[test]
    fn shapes_no_basis_explains_are_rejected() {
        let target = rect_target();
        let h = oblique_h();
        // Every ring fitted as a circle elongated across the view's actual
        // foreshortening: no lattice basis predicts that.
        let mut rs = rings(&h, &target);
        for r in &mut rs {
            r.outer.a = 10.0;
            r.outer.b = 2.0;
            r.outer.angle = 1.2;
        }
        assert_eq!(check_labeling(&rs, &h, &target), BasisCheck::Inconsistent);
    }

    #[test]
    fn too_few_rings_are_not_judged() {
        let target = rect_target();
        let h = oblique_h();
        let rs: Vec<LabeledRing> = rings(&h, &target).into_iter().take(MIN_RINGS - 1).collect();
        assert_eq!(check_labeling(&rs, &h, &target), BasisCheck::Consistent);
    }

    #[test]
    fn candidates_are_orientation_preserving_and_unimodular() {
        let n = unimodular_candidates()
            .inspect(|r| {
                let [[a, b], [c, d]] = r.0;
                assert_eq!(a * d - b * c, 1);
            })
            .count();
        assert!(n > 20);
    }
}
