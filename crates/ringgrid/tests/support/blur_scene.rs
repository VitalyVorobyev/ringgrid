//! Test-side renderer of a coded ring-grid board seen by a pinhole camera
//! under defocus blur, low contrast, an illumination gradient, a saturated
//! glare blob and additive noise, with exact ground-truth marker centres.
//!
//! The board is rasterized once by the crate's own printable-target renderer
//! at high density, then warped into the camera image by a homography with
//! 4×4 supersampling per pixel. Pixel centres sit at integer coordinates, the
//! detector's convention. The true projected centre of a marker is the
//! homography image of its board position (no lens distortion).

#![allow(dead_code)] // each integration-test binary uses a different subset

use image::{GrayImage, Luma};
use rand::{RngExt, SeedableRng};
use ringgrid::{TargetLayout, TargetRenderOptions};

/// Density of the board raster that the camera image is resampled from.
const BOARD_DPI: f32 = 600.0;
const SUPERSAMPLE: usize = 4;

/// A saturated glare blob: an additive Gaussian bump in grey levels.
#[derive(Debug, Clone, Copy)]
pub struct Glare {
    pub center_px: [f64; 2],
    pub sigma_px: f64,
    pub peak: f64,
}

/// Everything that defines one synthetic frame.
#[derive(Debug, Clone)]
pub struct SceneSpec {
    pub image_size: [u32; 2],
    pub focal_px: f64,
    /// Board scale at the image centre for a frontal view (sets the distance).
    pub px_per_mm: f64,
    /// Out-of-plane tilt of the board, degrees.
    pub tilt_deg: f64,
    /// Direction of the tilt axis in the board plane, degrees.
    pub tilt_axis_deg: f64,
    /// In-plane rotation of the board, degrees.
    pub roll_deg: f64,
    /// Offset (mm) from the board's centre to the board point that lands on
    /// the principal point.
    pub board_center_mm: [f64; 2],
    /// Isotropic Gaussian PSF sigma, pixels.
    pub blur_sigma_px: f64,
    /// Grey level of the white paper at the image centre.
    pub white: f64,
    /// White minus black, grey levels.
    pub contrast: f64,
    /// Fractional illumination change from the left to the right image edge.
    pub illum_gradient: f64,
    pub glare: Option<Glare>,
    /// Additive Gaussian noise sigma, grey levels.
    pub noise_sigma: f64,
    pub seed: u64,
}

impl SceneSpec {
    /// The acceptance scene: 720×540, ≈3.45 px/mm (outer ring ≈ 33 px across),
    /// 40 grey levels of contrast, σ = 2 noise, a ±15 % illumination gradient
    /// and a saturated blob over a few markers.
    pub fn acceptance(blur_sigma_px: f64, tilt_deg: f64, seed: u64) -> Self {
        Self::with_contrast(blur_sigma_px, tilt_deg, seed, 40.0)
    }

    /// [`acceptance`](Self::acceptance) with another paper-minus-ink contrast.
    pub fn with_contrast(blur_sigma_px: f64, tilt_deg: f64, seed: u64, contrast: f64) -> Self {
        Self {
            image_size: [720, 540],
            focal_px: 1400.0,
            px_per_mm: 3.45,
            tilt_deg,
            tilt_axis_deg: 30.0,
            roll_deg: 7.0,
            board_center_mm: [0.0, 0.0],
            blur_sigma_px,
            white: 70.0 + contrast,
            contrast,
            illum_gradient: 0.3,
            glare: Some(Glare {
                center_px: [250.0, 190.0],
                sigma_px: 22.0,
                peak: 600.0,
            }),
            noise_sigma: 2.0,
            seed,
        }
    }
}

/// High-density raster of a board plus the board-mm → raster-pixel map.
pub struct BoardRaster {
    pub target: TargetLayout,
    raster: GrayImage,
    px_per_mm: f64,
    offset_mm: [f64; 2],
}

impl BoardRaster {
    pub fn new(target: TargetLayout) -> Self {
        Self::with_code_band(target, true)
    }

    /// `code_band = false` paints every code sector black (diagnostic only:
    /// such markers cannot be decoded).
    pub fn with_code_band(target: TargetLayout, code_band: bool) -> Self {
        let options = TargetRenderOptions::default()
            .with_scale_bar(false)
            .with_png_dpi(BOARD_DPI);
        let mut raster = target
            .render_target_png(&options)
            .expect("render board raster");
        // Mirror of the renderer's fit-content canvas (no margin): the content
        // box is centred in a square page.
        let (min_xy, max_xy) = target.marker_bounds_mm().expect("non-empty board");
        let ext = f64::from(target.outer_draw_radius_mm());
        let min = [f64::from(min_xy[0]) - ext, f64::from(min_xy[1]) - ext];
        let span = [
            f64::from(max_xy[0]) - f64::from(min_xy[0]) + 2.0 * ext,
            f64::from(max_xy[1]) - f64::from(min_xy[1]) + 2.0 * ext,
        ];
        let side = span[0].max(span[1]);
        let offset_mm = [
            0.5 * (side - span[0]) - min[0],
            0.5 * (side - span[1]) - min[1],
        ];
        let px_per_mm = f64::from(BOARD_DPI) / 25.4;
        if !code_band && let Some((r0, r1)) = target.code_band_bounds_mm() {
            let (r0, r1) = (f64::from(r0) - 0.05, f64::from(r1) + 0.05);
            for c in target.cells() {
                let cx = (f64::from(c.xy_mm[0]) + offset_mm[0]) * px_per_mm;
                let cy = (f64::from(c.xy_mm[1]) + offset_mm[1]) * px_per_mm;
                let rr = (r1 * px_per_mm).ceil() as i64 + 1;
                for dy in -rr..=rr {
                    for dx in -rr..=rr {
                        let (x, y) = (cx.round() as i64 + dx, cy.round() as i64 + dy);
                        let d = ((x as f64 - cx).hypot(y as f64 - cy)) / px_per_mm;
                        if d >= r0
                            && d <= r1
                            && x >= 0
                            && y >= 0
                            && let Some(p) = raster.get_pixel_mut_checked(x as u32, y as u32)
                        {
                            p.0[0] = 0;
                        }
                    }
                }
            }
        }
        Self {
            target,
            raster,
            px_per_mm,
            offset_mm,
        }
    }

    /// Reflectance in [0, 1] (white paper = 1) at a board point, bilinear.
    fn reflectance(&self, xy_mm: [f64; 2]) -> f64 {
        let x = (xy_mm[0] + self.offset_mm[0]) * self.px_per_mm;
        let y = (xy_mm[1] + self.offset_mm[1]) * self.px_per_mm;
        let (w, h) = self.raster.dimensions();
        if !(x >= 0.0 && y >= 0.0 && x < f64::from(w - 1) && y < f64::from(h - 1)) {
            return 1.0;
        }
        let (x0, y0) = (x.floor() as u32, y.floor() as u32);
        let (fx, fy) = (x - f64::from(x0), y - f64::from(y0));
        let p = |xx: u32, yy: u32| f64::from(self.raster.get_pixel(xx, yy)[0]) / 255.0;
        let top = p(x0, y0) * (1.0 - fx) + p(x0 + 1, y0) * fx;
        let bot = p(x0, y0 + 1) * (1.0 - fx) + p(x0 + 1, y0 + 1) * fx;
        top * (1.0 - fy) + bot * fy
    }
}

/// A rendered frame with its board → image homography.
pub struct Scene {
    pub image: GrayImage,
    /// Board mm → image pixels (row-major 3×3).
    pub h: [[f64; 3]; 3],
}

impl Scene {
    pub fn project(&self, xy_mm: [f64; 2]) -> [f64; 2] {
        apply_h(&self.h, xy_mm)
    }
}

pub fn apply_h(h: &[[f64; 3]; 3], p: [f64; 2]) -> [f64; 2] {
    let x = h[0][0] * p[0] + h[0][1] * p[1] + h[0][2];
    let y = h[1][0] * p[0] + h[1][1] * p[1] + h[1][2];
    let w = h[2][0] * p[0] + h[2][1] * p[1] + h[2][2];
    [x / w, y / w]
}

fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn inverse3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let c =
        |r0: usize, r1: usize, c0: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    [
        [
            c(1, 2, 1, 2) / det,
            -c(0, 2, 1, 2) / det,
            c(0, 1, 1, 2) / det,
        ],
        [
            -c(1, 2, 0, 2) / det,
            c(0, 2, 0, 2) / det,
            -c(0, 1, 0, 2) / det,
        ],
        [
            c(1, 2, 0, 1) / det,
            -c(0, 2, 0, 1) / det,
            c(0, 1, 0, 1) / det,
        ],
    ]
}

/// Board mm → image homography of the scene's camera; `board_mid_mm` is the
/// board's centre.
pub fn scene_homography(spec: &SceneSpec, board_mid_mm: [f64; 2]) -> [[f64; 3]; 3] {
    let (st, ct) = spec.tilt_deg.to_radians().sin_cos();
    let (sa, ca) = spec.tilt_axis_deg.to_radians().sin_cos();
    let (sr, cr) = spec.roll_deg.to_radians().sin_cos();
    // Rotation about the in-plane axis a = (ca, sa, 0) by the tilt angle.
    let (ax, ay) = (ca, sa);
    let tilt = [
        [ct + ax * ax * (1.0 - ct), ax * ay * (1.0 - ct), ay * st],
        [ax * ay * (1.0 - ct), ct + ay * ay * (1.0 - ct), -ax * st],
        [-ay * st, ax * st, ct],
    ];
    let roll = [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]];
    let r = mat3_mul(&roll, &tilt);
    let z = spec.focal_px / spec.px_per_mm;
    let ox = board_mid_mm[0] + spec.board_center_mm[0];
    let oy = board_mid_mm[1] + spec.board_center_mm[1];
    // Camera point = R * (X - o) + (0, 0, z) for board points X = (x, y, 0).
    let t = [
        -(r[0][0] * ox + r[0][1] * oy),
        -(r[1][0] * ox + r[1][1] * oy),
        z - (r[2][0] * ox + r[2][1] * oy),
    ];
    let rt = [
        [r[0][0], r[0][1], t[0]],
        [r[1][0], r[1][1], t[1]],
        [r[2][0], r[2][1], t[2]],
    ];
    let cx = 0.5 * f64::from(spec.image_size[0] - 1);
    let cy = 0.5 * f64::from(spec.image_size[1] - 1);
    let k = [
        [spec.focal_px, 0.0, cx],
        [0.0, spec.focal_px, cy],
        [0.0, 0.0, 1.0],
    ];
    mat3_mul(&k, &rt)
}

fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    let r = (4.0 * sigma).ceil().max(1.0) as i64;
    let mut k: Vec<f64> = (-r..=r)
        .map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp())
        .collect();
    let s: f64 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k
}

/// Separable Gaussian blur with edge clamping.
fn blur(buf: &mut [f64], w: usize, h: usize, sigma: f64) {
    if sigma <= 0.0 {
        return;
    }
    let k = gaussian_kernel(sigma);
    let r = (k.len() / 2) as i64;
    let mut tmp = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (j, kv) in k.iter().enumerate() {
                let xx = (x as i64 + j as i64 - r).clamp(0, w as i64 - 1) as usize;
                acc += kv * buf[y * w + xx];
            }
            tmp[y * w + x] = acc;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (j, kv) in k.iter().enumerate() {
                let yy = (y as i64 + j as i64 - r).clamp(0, h as i64 - 1) as usize;
                acc += kv * tmp[yy * w + x];
            }
            buf[y * w + x] = acc;
        }
    }
}

/// Render one frame of `board` under `spec`.
pub fn render(board: &BoardRaster, spec: &SceneSpec) -> Scene {
    let (lo, hi) = board.target.marker_bounds_mm().expect("non-empty board");
    let mid = [
        0.5 * f64::from(lo[0] + hi[0]),
        0.5 * f64::from(lo[1] + hi[1]),
    ];
    let h = scene_homography(spec, mid);
    let h_inv = inverse3(&h);
    let (w, hh) = (spec.image_size[0] as usize, spec.image_size[1] as usize);

    // Reflectance, area-sampled.
    let mut buf = vec![0.0f64; w * hh];
    let n = SUPERSAMPLE as f64;
    for y in 0..hh {
        for x in 0..w {
            let mut acc = 0.0;
            for sy in 0..SUPERSAMPLE {
                for sx in 0..SUPERSAMPLE {
                    let px = x as f64 + (sx as f64 + 0.5) / n - 0.5;
                    let py = y as f64 + (sy as f64 + 0.5) / n - 0.5;
                    acc += board.reflectance(apply_h(&h_inv, [px, py]));
                }
            }
            buf[y * w + x] = acc / (n * n);
        }
    }

    // Radiance: illumination × (black + contrast × reflectance).
    let black = spec.white - spec.contrast;
    for y in 0..hh {
        for x in 0..w {
            let illum = 1.0 + spec.illum_gradient * (x as f64 / (w - 1) as f64 - 0.5);
            let v = &mut buf[y * w + x];
            *v = illum * (black + spec.contrast * *v);
        }
    }
    if let Some(g) = spec.glare {
        for y in 0..hh {
            for x in 0..w {
                let dx = x as f64 - g.center_px[0];
                let dy = y as f64 - g.center_px[1];
                buf[y * w + x] +=
                    g.peak * (-(dx * dx + dy * dy) / (2.0 * g.sigma_px.powi(2))).exp();
            }
        }
    }

    blur(&mut buf, w, hh, spec.blur_sigma_px);

    let mut rng = rand::rngs::StdRng::seed_from_u64(spec.seed);
    let mut image = GrayImage::new(spec.image_size[0], spec.image_size[1]);
    for y in 0..hh {
        for x in 0..w {
            // Box–Muller.
            let u1: f64 = rng.random_range(f64::EPSILON..1.0);
            let u2: f64 = rng.random_range(0.0..1.0);
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let v = buf[y * w + x] + spec.noise_sigma * noise;
            image.put_pixel(
                x as u32,
                y as u32,
                Luma([v.round().clamp(0.0, 255.0) as u8]),
            );
        }
    }
    Scene { image, h }
}

/// Centre-error statistics against ground truth for decoded markers.
#[derive(Debug, Clone, Default)]
pub struct CenterStats {
    /// Markers whose full outer ring lies inside the frame.
    pub n_visible: usize,
    /// Decoded markers with a board position.
    pub n_decoded: usize,
    /// Decoded markers whose id is wrong (centre > 0.5 pitch from the truth).
    pub n_wrong_id: usize,
    pub errors: Vec<f64>,
    /// `(centre source, error)` per scored marker.
    pub by_source: Vec<(ringgrid::CenterSource, f64)>,
    /// `(reported centre covariance, error vector)` for markers that carry one.
    pub with_cov: Vec<([[f64; 2]; 2], [f64; 2])>,
}

impl CenterStats {
    pub fn rms(&self) -> f64 {
        (self.errors.iter().map(|e| e * e).sum::<f64>() / self.errors.len().max(1) as f64).sqrt()
    }

    pub fn quantile(&self, q: f64) -> f64 {
        let mut v = self.errors.clone();
        v.sort_by(|a, b| a.total_cmp(b));
        if v.is_empty() {
            return f64::NAN;
        }
        v[((v.len() - 1) as f64 * q).round() as usize]
    }

    pub fn merge(&mut self, other: &CenterStats) {
        self.n_visible += other.n_visible;
        self.n_decoded += other.n_decoded;
        self.n_wrong_id += other.n_wrong_id;
        self.errors.extend_from_slice(&other.errors);
        self.by_source.extend_from_slice(&other.by_source);
        self.with_cov.extend_from_slice(&other.with_cov);
    }
}

/// Score detected centres against the scene's ground truth.
pub fn score(
    scene: &Scene,
    target: &TargetLayout,
    markers: &[ringgrid::DetectedMarker],
    match_undecoded: bool,
) -> CenterStats {
    let [w, h] = [scene.image.width() as f64, scene.image.height() as f64];
    let r_mm = f64::from(target.outer_draw_radius_mm());
    let n_visible = target
        .cells()
        .iter()
        .filter(|c| {
            let xy = [f64::from(c.xy_mm[0]), f64::from(c.xy_mm[1])];
            (0..16).all(|k| {
                let a = k as f64 * std::f64::consts::PI / 8.0;
                let p = scene.project([xy[0] + r_mm * a.cos(), xy[1] + r_mm * a.sin()]);
                p[0] >= 0.0 && p[1] >= 0.0 && p[0] <= w - 1.0 && p[1] <= h - 1.0
            })
        })
        .count();
    let mut stats = CenterStats {
        n_visible,
        ..CenterStats::default()
    };
    let gate_px = 0.5 * f64::from(target.min_center_spacing_mm()) * 3.0;
    for m in markers {
        let gt = match m.board_xy_mm {
            Some(xy) => {
                stats.n_decoded += 1;
                scene.project(xy)
            }
            None if match_undecoded => {
                let Some(gt) = target
                    .cells()
                    .iter()
                    .map(|c| scene.project([f64::from(c.xy_mm[0]), f64::from(c.xy_mm[1])]))
                    .min_by(|a, b| {
                        let da = (a[0] - m.center[0]).hypot(a[1] - m.center[1]);
                        let db = (b[0] - m.center[0]).hypot(b[1] - m.center[1]);
                        da.total_cmp(&db)
                    })
                else {
                    continue;
                };
                gt
            }
            None => continue,
        };
        let err = (m.center[0] - gt[0]).hypot(m.center[1] - gt[1]);
        if err > gate_px {
            stats.n_wrong_id += 1;
            continue;
        }
        stats.errors.push(err);
        stats.by_source.push((m.center_source, err));
        if let Some(cov) = m.center_cov {
            stats
                .with_cov
                .push((cov, [m.center[0] - gt[0], m.center[1] - gt[1]]));
        }
    }
    stats
}
