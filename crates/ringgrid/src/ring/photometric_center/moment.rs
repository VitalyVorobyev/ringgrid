//! Study comparator based on the image-plane first moment of ink. A
//! normalized, zero-mean scatter PSF preserves this moment on an unbounded
//! patch even when its width varies with the source position.

use super::{
    CODE_SECTORS, Mat3, PhotometricFit, PhotometricProblem, PrintRadii, build_model, collect_patch,
    photometry,
};
use crate::homography::homography_project_rows;
use std::f64::consts::{PI, TAU};

/// Offset of the sharp ink's image centroid from the projected marker
/// origin. The caller accounts for the fit's affine correction separately.
fn sharp_offset(map: &Mat3, radii: PrintRadii, codeword: Option<u16>) -> Option<[f64; 2]> {
    let origin = homography_project_rows(map, [0.0; 2])?;
    let q = [map[2][0] / map[2][2], map[2][1] / map[2][2]];
    let spread = q[0].hypot(q[1]);
    if !spread.is_finite() || spread * radii[3] >= 1.0 {
        return None;
    }
    let [inner, code_inner, code_outer, outer] = radii;
    let mut mass = 0.0;
    let mut moment = [0.0; 2];
    let mut accumulate = |u: [f64; 2], area: f64| {
        let pos = homography_project_rows(map, u)?;
        mass += area;
        for ((m, p), o) in moment.iter_mut().zip(pos).zip(origin) {
            *m += area * (p - o);
        }
        Some(())
    };
    // A homography maps a disc to an ellipse. Its centre is the image
    // of -r²q, and its area scales as πr²/(1 - r²|q|²)^(3/2), up to
    // the common Jacobian at the marker origin.
    for (r, sign) in [(outer, 1.0), (inner, -1.0)] {
        let r2 = r * r;
        let area = PI * r2 / (1.0 - r2 * spread * spread).powf(1.5);
        accumulate([-r2 * q[0], -r2 * q[1]], sign * area)?;
    }
    // Subtract the paper sectors of the code band. Integrate the exact
    // projective area factor (1 + q·u)^-3 over these wedges by quadrature,
    // splitting at every sector boundary.
    if let Some(codeword) = codeword {
        const N: usize = 24;
        let da = TAU / (CODE_SECTORS * N) as f64;
        let dr = (code_outer - code_inner) / N as f64;
        for sector in 0..CODE_SECTORS {
            if codeword & (1 << sector) == 0 {
                continue;
            }
            for j in 0..N {
                let angle = -PI + (sector * N + j) as f64 * da + 0.5 * da;
                let (sin, cos) = angle.sin_cos();
                for k in 0..N {
                    let r = code_inner + (k as f64 + 0.5) * dr;
                    let u = [r * cos, r * sin];
                    let w = 1.0 + q[0] * u[0] + q[1] * u[1];
                    accumulate(u, -r * dr * da / w.powi(3))?;
                }
            }
        }
    }
    (mass > 0.0).then(|| moment.map(|m| m / mass))
}

fn subtract_sharp_offset(
    center: [f64; 2],
    correction: [[f64; 2]; 2],
    offset: [f64; 2],
) -> Option<[f64; 2]> {
    let [[a, b], [c, d]] = correction;
    let det = a * d - b * c;
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    let out = [
        center[0] - (d * offset[0] - b * offset[1]) / det,
        center[1] - (a * offset[1] - c * offset[0]) / det,
    ];
    out.into_iter().all(f64::is_finite).then_some(out)
}

/// The moment of observed ink, minus the shift predicted for the sharp
/// printed pattern by the local projective geometry. Image indices are pixel
/// centres. Weights are signed after removing the fitted paper and contrast
/// fields to avoid turning background noise into positive ink. A finite
/// patch can still lose ink or include neighbouring ink, and the local
/// homography only approximates distortion inside the patch.
pub(crate) fn ink_moment_center(
    problem: &PhotometricProblem<'_>,
    fit: &PhotometricFit,
) -> Option<[f64; 2]> {
    let model = build_model(problem)?;
    let (pixels, masked_fraction) = collect_patch(problem, &model);
    if masked_fraction > 0.0 {
        return None;
    }
    let mut mass = 0.0;
    let mut moment = [0.0; 2];
    let seed = problem.center_seed;
    let [w, h] = [problem.gray.width(), problem.gray.height()];
    for pixel in pixels {
        // A patch touching the image boundary may truncate the ink itself.
        if pixel.x[0] <= 0.0
            || pixel.x[1] <= 0.0
            || pixel.x[0] >= f64::from(w) - 1.0
            || pixel.x[1] >= f64::from(h) - 1.0
        {
            return None;
        }
        let (paper, contrast) = photometry(&fit.params.lin, pixel.xi, 0.0);
        if !contrast.is_finite() || contrast <= 0.0 {
            return None;
        }
        let ink = (paper - pixel.value) / contrast;
        mass += ink;
        for ((m, p), s) in moment.iter_mut().zip(pixel.x).zip(seed) {
            *m += ink * (p - s);
        }
    }
    if !mass.is_finite() || mass <= 0.0 {
        return None;
    }
    let center = std::array::from_fn(|axis| seed[axis] + moment[axis] / mass);
    let offset = sharp_offset(&problem.local_map, fit.radii, problem.codeword)?;
    subtract_sharp_offset(center, fit.params.c, offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::photometric_center::{
        InkDisc, fit_marker, mat3_inverse,
        tests::{CODEWORD, RADII, problem, quantize, radiance, true_map},
    };
    use image::Luma;

    const SIDE: usize = 81;
    const CENTER: [f64; 2] = [40.0; 2];

    /// Scatter ink along image y with source-dependent sigma, then apply
    /// the same paper and contrast gradients as the test renderer. Constant
    /// scene brightness and no noise apart from uint8 quantization.
    fn scatter(sharp: &[f64], gradient: f64) -> Vec<f64> {
        let illumination = |x: usize| 1.0 + 0.004 * (x as f64 - CENTER[0]);
        let mut ink = vec![0.0; SIDE * SIDE];
        let kernel = |sigma: f64| -> [f64; 13] {
            let weights =
                std::array::from_fn(|i| (-0.5 * ((i as f64 - 6.0) / sigma).powi(2)).exp());
            let sum: f64 = weights.iter().sum();
            weights.map(|w| w / sum)
        };
        let horizontal = kernel(1.2);
        for y in 6..SIDE - 6 {
            let sigma = 1.2 + gradient * (y as f64 - CENTER[1]);
            let vertical = kernel(sigma);
            for x in 6..SIDE - 6 {
                let p = (105.0 - sharp[y * SIDE + x] / illumination(x)) / 40.0;
                for (j, wy) in vertical.iter().enumerate() {
                    for (i, wx) in horizontal.iter().enumerate() {
                        ink[(y + j - 6) * SIDE + x + i - 6] += p * wx * wy;
                    }
                }
            }
        }
        ink.iter()
            .enumerate()
            .map(|(k, p)| illumination(k % SIDE) * (105.0 - 40.0 * p))
            .collect()
    }

    fn isolated_sharp(map: &Mat3) -> Vec<f64> {
        let mut sharp = radiance(map, SIDE as u32, 0.0, Some(CODEWORD));
        let inverse = mat3_inverse(map).unwrap();
        for (i, sample) in sharp.iter_mut().enumerate() {
            let x = (i % SIDE) as f64;
            let y = (i / SIDE) as f64;
            let u = homography_project_rows(&inverse, [x, y]).unwrap();
            if u[0].hypot(u[1]) > RADII[3] + 0.5 {
                *sample = 105.0 * (1.0 + 0.004 * (x - CENTER[0]));
            }
        }
        sharp
    }

    #[test]
    fn shaded_coded_centroid_is_projectively_corrected_for_both_gradient_directions() {
        let map = true_map(42.0, CENTER);
        let sharp = isolated_sharp(&map);
        let mut centers = Vec::new();
        for grad in [-0.018, 0.0, 0.018] {
            let image = quantize(&scatter(&sharp, grad), SIDE as u32, 0.0, 1);
            let mut problem = problem(&image, map, CENTER, &[]);
            // An isolated marker permits a wide aperture for the scatter tail.
            problem.patch_margin_mm = 4.0;
            let fit = fit_marker(&problem).expect("coded test patch fits");
            let center = ink_moment_center(&problem, &fit).expect("moment exists");
            for i in 0..2 {
                assert!((center[i] - CENTER[i]).abs() < 0.06, "{grad}: {center:?}");
            }
            centers.push(center);
        }
        for center in [centers[0], centers[2]] {
            assert!((center[1] - centers[1][1]).abs() < 0.04, "{centers:?}");
        }
    }

    #[test]
    fn moment_detects_neighbour_spill_and_rejects_saturated_aperture() {
        let map = true_map(0.0, CENTER);
        let sharp = isolated_sharp(&map);
        let mut image = quantize(&sharp, SIDE as u32, 0.0, 1);
        let known_neighbour = [InkDisc {
            center: [7.5, 0.0],
            radius: 2.0,
        }];
        let mut base = problem(&image, map, CENTER, &known_neighbour);
        base.patch_margin_mm = 2.0;
        let fit = fit_marker(&base).expect("coded test patch fits");
        // Keep the same paper model when adding a dark patch near the
        // right aperture boundary to isolate moment sensitivity.
        let before = ink_moment_center(&base, &fit).unwrap();
        for y in 36..=44 {
            for x in 61..=64 {
                image.put_pixel(x, y, Luma([78]));
            }
        }
        let mut contaminated = problem(&image, map, CENTER, &known_neighbour);
        contaminated.patch_margin_mm = 2.0;
        let after = ink_moment_center(&contaminated, &fit).unwrap();
        assert!(after[0] - before[0] > 0.2, "{before:?} -> {after:?}");

        image.put_pixel(40, 40, Luma([255]));
        let saturated = problem(&image, map, CENTER, &known_neighbour);
        assert!(ink_moment_center(&saturated, &fit).is_none());
    }
}
