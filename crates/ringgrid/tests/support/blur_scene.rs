//! Test-side renderer of a ring-grid board (any target kind: coded or plain
//! markers, hex or rect lattice, with or without origin dots) seen by a
//! pinhole camera under defocus blur, low contrast, an illumination gradient,
//! a saturated glare blob and additive noise, with exact ground-truth marker
//! centres.
//!
//! The board is rasterized once by the crate's own printable-target renderer
//! at high density, then warped into the camera image by a homography with
//! 4×4 supersampling per pixel. Pixel centres sit at integer coordinates, the
//! detector's convention. The true projected centre of a marker is the
//! homography image of its board position, through the optional radial lens
//! distortion ([`RadialDivision`]).
//!
//! Besides the Gaussian PSF the detector models, a scene can carry effects the
//! photometric model cannot represent — an anisotropic Gaussian, a Gaussian
//! with a wide halo, a defocus disc, a gamma response, spatially correlated
//! sensor noise — to test how honest its covariance stays under misfit.
//! Rendering is split into the noise-free [`radiance`] and its noisy
//! [`expose`]ure so Monte-Carlo runs re-render only the noise.

#![allow(dead_code)] // each integration-test binary uses a different subset

use image::{GrayImage, Luma};
use rand::{RngExt, SeedableRng};
use ringgrid::{TargetLayout, TargetRenderOptions};

/// Density of the board raster that the camera image is resampled from.
const BOARD_DPI: f32 = 600.0;
const SUPERSAMPLE: usize = 4;

/// How a display rasterises the ideal pattern onto its pixel grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayRaster {
    /// A pixel is white when the ideal pattern at its centre is (reflectance
    /// at least one half), else black.
    Binary,
    /// Area coverage of the ideal pattern, quantised to 16 grey levels.
    Grey16,
}

/// A display of pixel pitch `pitch_mm`, axes aligned with the board.
#[derive(Debug, Clone, Copy)]
pub struct DisplaySpec {
    pub pitch_mm: f64,
    pub raster: DisplayRaster,
}

/// Fraction of a display pixel by which the display grid is shifted against
/// the board raster origin, so no marker sits at a special grid phase.
const DISPLAY_PHASE: f64 = 0.37;
/// Sub-samples per display-pixel side for the area coverage.
const COVERAGE_TAPS: usize = 8;

/// Display pixel values in 0..=15 (15 = white), row-major.
struct DisplayGrid {
    spec: DisplaySpec,
    w: usize,
    h: usize,
    levels: Vec<u8>,
}

/// A saturated glare blob: an additive Gaussian bump in grey levels.
#[derive(Debug, Clone, Copy)]
pub struct Glare {
    pub center_px: [f64; 2],
    pub sigma_px: f64,
    pub peak: f64,
}

/// Point spread function of the camera.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Psf {
    /// Isotropic Gaussian (the photometric model's PSF).
    Gaussian { sigma: f64 },
    /// Anisotropic Gaussian with principal sigmas along `angle_deg` and
    /// perpendicular to it.
    Anisotropic {
        sigma_major: f64,
        sigma_minor: f64,
        angle_deg: f64,
    },
    /// A Gaussian core plus a wide Gaussian halo carrying `halo_weight` of the
    /// energy (lens flare, scattering).
    Halo {
        sigma: f64,
        halo_sigma: f64,
        halo_weight: f64,
    },
    /// Uniform disc (geometric defocus) of `radius` px.
    Disc { radius: f64 },
    /// Isotropic Gaussian whose sigma varies along the image y axis:
    /// `sigma(y) = clamp(sigma0 + gradient * (y - y0), sigma_range)` (px; the
    /// clamp keeps a steep field positive far from the band of interest).
    /// `form` says whose position sets the kernel, see [`BlurForm`].
    VaryingGaussian {
        sigma0: f64,
        /// d sigma / d y, px per px.
        gradient: f64,
        y0: f64,
        sigma_range: [f64; 2],
        form: BlurForm,
    },
}

/// How a spatially varying PSF is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlurForm {
    /// Every output pixel averages the sharp scene with the kernel of its own
    /// position: `I(x) = sum_y S(y) G_{sigma(x)}(x - y)` (a constant scene
    /// stays constant; light is not conserved).
    Gather,
    /// Every scene point spreads its light with the kernel of its own
    /// position: `I(x) = sum_y S(y) G_{sigma(y)}(x - y)` (light is conserved;
    /// the physical form of depth-dependent defocus).
    Scatter,
}

impl Psf {
    /// Second-moment sigma (px) along one axis, for labelling.
    pub fn nominal_sigma(&self) -> f64 {
        match *self {
            Psf::Gaussian { sigma } => sigma,
            Psf::Anisotropic {
                sigma_major,
                sigma_minor,
                ..
            } => (sigma_major * sigma_minor).sqrt(),
            Psf::Halo { sigma, .. } => sigma,
            Psf::Disc { radius } => 0.5 * radius,
            Psf::VaryingGaussian { sigma0, .. } => sigma0,
        }
    }
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
    pub psf: Psf,
    /// Camera response exponent applied to the normalized radiance (1 =
    /// linear).
    pub gamma: f64,
    /// Grey level of the white paper at the image centre.
    pub white: f64,
    /// White minus black, grey levels.
    pub contrast: f64,
    /// Fractional illumination change from the left to the right image edge.
    pub illum_gradient: f64,
    pub glare: Option<Glare>,
    /// Additive Gaussian noise sigma, grey levels.
    pub noise_sigma: f64,
    /// Spatial correlation (Gaussian sigma, px) of the sensor noise; 0 is
    /// white.
    pub noise_corr_px: f64,
    /// Sub-pixel samples per side when integrating the board reflectance.
    pub supersample: usize,
    /// Radial lens distortion (division model, see [`RadialDivision`]); 0 is
    /// a distortion-free pinhole.
    pub radial_lambda: f64,
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
            psf: Psf::Gaussian {
                sigma: blur_sigma_px,
            },
            gamma: 1.0,
            white: 70.0 + contrast,
            contrast,
            illum_gradient: 0.3,
            glare: Some(Glare {
                center_px: [250.0, 190.0],
                sigma_px: 22.0,
                peak: 600.0,
            }),
            noise_sigma: 2.0,
            noise_corr_px: 0.0,
            supersample: SUPERSAMPLE,
            radial_lambda: 0.0,
            seed,
        }
    }

    /// A clean scene for any target: `outer_radius_px` sets the board
    /// distance (the marker's drawn outer radius at the image centre), no
    /// glare; otherwise as [`with_contrast`](Self::with_contrast).
    pub fn for_target(
        target: &TargetLayout,
        outer_radius_px: f64,
        blur_sigma_px: f64,
        tilt_deg: f64,
        contrast: f64,
        seed: u64,
    ) -> Self {
        Self {
            px_per_mm: outer_radius_px / f64::from(target.outer_draw_radius_mm()),
            glare: None,
            ..Self::with_contrast(blur_sigma_px, tilt_deg, seed, contrast)
        }
    }
}

/// High-density raster of a board plus the board-mm → raster-pixel map.
#[derive(Clone)]
pub struct BoardRaster {
    pub target: TargetLayout,
    raster: std::sync::Arc<GrayImage>,
    px_per_mm: f64,
    offset_mm: [f64; 2],
    display: Option<std::sync::Arc<DisplayGrid>>,
    /// Only the board within this disc (centre, radius, mm) is printed; the
    /// rest is paper.
    keep: Option<([f64; 2], f64)>,
}

impl BoardRaster {
    pub fn new(target: TargetLayout) -> Self {
        Self::with_code_band(target, true)
    }

    /// A raster at `dpi` (dense enough that its edge quantization stays well
    /// below a camera pixel: a few raster pixels per camera pixel).
    pub fn with_dpi(target: TargetLayout, dpi: f32) -> Self {
        Self::build(target, true, dpi)
    }

    /// `code_band = false` paints every code sector black (diagnostic only:
    /// such markers cannot be decoded).
    pub fn with_code_band(target: TargetLayout, code_band: bool) -> Self {
        Self::build(target, code_band, BOARD_DPI)
    }

    fn build(target: TargetLayout, code_band: bool, dpi: f32) -> Self {
        let options = TargetRenderOptions::default()
            .with_scale_bar(false)
            .with_png_dpi(dpi);
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
        let px_per_mm = f64::from(dpi) / 25.4;
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
            raster: std::sync::Arc::new(raster),
            px_per_mm,
            offset_mm,
            display: None,
            keep: None,
        }
    }

    /// The same board with everything outside the disc of `radius_mm` around
    /// `center_mm` replaced by paper (one isolated marker).
    pub fn isolated(&self, center_mm: [f64; 2], radius_mm: f64) -> Self {
        Self {
            keep: Some((center_mm, radius_mm)),
            ..self.clone()
        }
    }

    /// The same board shown on a display: the ideal pattern rasterised onto
    /// the display's pixel grid, which then replaces it as the imaged
    /// reflectance (piecewise constant per display pixel).
    pub fn displayed(&self, spec: DisplaySpec) -> Self {
        let (rw, rh) = self.raster.dimensions();
        let p = spec.pitch_mm;
        let side_x = f64::from(rw) / self.px_per_mm;
        let side_y = f64::from(rh) / self.px_per_mm;
        let (w, h) = (
            (side_x / p).ceil() as usize + 2,
            (side_y / p).ceil() as usize + 2,
        );
        let mut levels = vec![15u8; w * h];
        let taps = COVERAGE_TAPS;
        for j in 0..h {
            for i in 0..w {
                // Pixel square in board+offset mm.
                let x0 = (i as f64 + DISPLAY_PHASE - 1.0) * p;
                let y0 = (j as f64 + DISPLAY_PHASE - 1.0) * p;
                let sample =
                    |fx: f64, fy: f64| self.ideal_reflectance_offset([x0 + fx * p, y0 + fy * p]);
                levels[j * w + i] = match spec.raster {
                    DisplayRaster::Binary => {
                        if sample(0.5, 0.5) >= 0.5 {
                            15
                        } else {
                            0
                        }
                    }
                    DisplayRaster::Grey16 => {
                        let mut acc = 0.0;
                        for ty in 0..taps {
                            for tx in 0..taps {
                                acc += sample(
                                    (tx as f64 + 0.5) / taps as f64,
                                    (ty as f64 + 0.5) / taps as f64,
                                );
                            }
                        }
                        (acc / (taps * taps) as f64 * 15.0).round() as u8
                    }
                };
            }
        }
        Self {
            target: self.target.clone(),
            raster: self.raster.clone(),
            px_per_mm: self.px_per_mm,
            offset_mm: self.offset_mm,
            display: Some(std::sync::Arc::new(DisplayGrid { spec, w, h, levels })),
            keep: self.keep,
        }
    }

    /// Reflectance in [0, 1] at a board point as imaged: the display pixel's
    /// value when the board is displayed, else the ideal pattern.
    pub fn reflectance(&self, xy_mm: [f64; 2]) -> f64 {
        if let Some((c, r)) = self.keep
            && (xy_mm[0] - c[0]).hypot(xy_mm[1] - c[1]) > r
        {
            return 1.0;
        }
        let Some(d) = &self.display else {
            return self.ideal_reflectance(xy_mm);
        };
        let p = d.spec.pitch_mm;
        let i = ((xy_mm[0] + self.offset_mm[0]) / p - DISPLAY_PHASE + 1.0).floor();
        let j = ((xy_mm[1] + self.offset_mm[1]) / p - DISPLAY_PHASE + 1.0).floor();
        if i < 0.0 || j < 0.0 || i >= d.w as f64 || j >= d.h as f64 {
            return 1.0;
        }
        f64::from(d.levels[j as usize * d.w + i as usize]) / 15.0
    }

    /// Reflectance of the ideal (un-rasterised) pattern at a board point.
    pub fn ideal_reflectance(&self, xy_mm: [f64; 2]) -> f64 {
        self.ideal_reflectance_offset([xy_mm[0] + self.offset_mm[0], xy_mm[1] + self.offset_mm[1]])
    }

    /// Ideal reflectance at a point given in raster-origin mm.
    fn ideal_reflectance_offset(&self, mm: [f64; 2]) -> f64 {
        let x = mm[0] * self.px_per_mm;
        let y = mm[1] * self.px_per_mm;
        self.bilinear(x, y)
    }

    fn bilinear(&self, x: f64, y: f64) -> f64 {
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

/// One-parameter radial division model about the image centre, in
/// coordinates normalised by the half-diagonal `scale`: an image (distorted)
/// point `d` sees the pinhole point `c + (d − c) / (1 + λ |q|²)`, `q = (d −
/// c) / scale`. `λ < 0` is barrel distortion.
#[derive(Debug, Clone, Copy)]
pub struct RadialDivision {
    pub lambda: f64,
    pub center: [f64; 2],
    pub scale: f64,
}

impl RadialDivision {
    fn of(spec: &SceneSpec) -> Option<Self> {
        (spec.radial_lambda != 0.0).then(|| {
            let [w, h] = spec.image_size.map(f64::from);
            Self {
                lambda: spec.radial_lambda,
                center: [0.5 * (w - 1.0), 0.5 * (h - 1.0)],
                scale: 0.5 * w.hypot(h),
            }
        })
    }

    /// Image (distorted) point → pinhole point.
    pub fn undistort(&self, d: [f64; 2]) -> [f64; 2] {
        let q = [
            (d[0] - self.center[0]) / self.scale,
            (d[1] - self.center[1]) / self.scale,
        ];
        let f = 1.0 / (1.0 + self.lambda * (q[0] * q[0] + q[1] * q[1]));
        [
            self.center[0] + self.scale * q[0] * f,
            self.center[1] + self.scale * q[1] * f,
        ]
    }

    /// Pinhole point → image (distorted) point (fixed-point inverse of
    /// [`undistort`](Self::undistort)).
    pub fn distort(&self, u: [f64; 2]) -> [f64; 2] {
        let qu = [
            (u[0] - self.center[0]) / self.scale,
            (u[1] - self.center[1]) / self.scale,
        ];
        let ru = qu[0].hypot(qu[1]);
        let mut rd = ru;
        for _ in 0..100 {
            rd = ru * (1.0 + self.lambda * rd * rd);
        }
        let k = if ru > 0.0 { rd / ru } else { 1.0 };
        [
            self.center[0] + self.scale * qu[0] * k,
            self.center[1] + self.scale * qu[1] * k,
        ]
    }
}

/// A rendered frame with its board → image homography.
pub struct Scene {
    pub image: GrayImage,
    /// Board mm → pinhole image pixels (row-major 3×3).
    pub h: [[f64; 3]; 3],
    /// Lens distortion between the pinhole image and the rendered image.
    pub distortion: Option<RadialDivision>,
}

impl Scene {
    /// Board point (mm) → image pixel, through the lens distortion.
    pub fn project(&self, xy_mm: [f64; 2]) -> [f64; 2] {
        let p = apply_h(&self.h, xy_mm);
        self.distortion.map_or(p, |d| d.distort(p))
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

/// Convolution with a general kernel given on a `(2r+1)²` grid, edge
/// clamping. `kernel(dx, dy)` need not be normalized.
fn convolve(buf: &mut [f64], w: usize, h: usize, r: i64, kernel: impl Fn(f64, f64) -> f64) {
    // Supersampled kernel taps, so a disc edge is antialiased.
    let mut taps = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            let mut v = 0.0;
            for sy in 0..4 {
                for sx in 0..4 {
                    v += kernel(
                        dx as f64 + (sx as f64 + 0.5) / 4.0 - 0.5,
                        dy as f64 + (sy as f64 + 0.5) / 4.0 - 0.5,
                    );
                }
            }
            if v > 0.0 {
                taps.push((dx, dy, v));
            }
        }
    }
    let total: f64 = taps.iter().map(|t| t.2).sum();
    let src = buf.to_vec();
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let mut acc = 0.0;
            for &(dx, dy, v) in &taps {
                let xx = (x + dx).clamp(0, w as i64 - 1) as usize;
                let yy = (y + dy).clamp(0, h as i64 - 1) as usize;
                acc += v * src[yy * w + xx];
            }
            buf[y as usize * w + x as usize] = acc / total;
        }
    }
}

/// Sigma of a [`Psf::VaryingGaussian`] at image row `y`.
pub fn varying_sigma(psf: &Psf, y: f64) -> Option<f64> {
    match *psf {
        Psf::VaryingGaussian {
            sigma0,
            gradient,
            y0,
            sigma_range,
            ..
        } => Some((sigma0 + gradient * (y - y0)).clamp(sigma_range[0], sigma_range[1])),
        _ => None,
    }
}

/// Gaussian blur whose sigma depends on the image row (`sigma_of_row`), in
/// gather or scatter form. Separable: the kernel of a row is the product of
/// two 1-D Gaussians of that row's sigma. Rows outside the image repeat the
/// border row (as the uniform blur's edge clamping does) but keep their own
/// sigma.
fn varying_blur(
    buf: &mut [f64],
    w: usize,
    h: usize,
    form: BlurForm,
    sigma_of_row: &dyn Fn(f64) -> f64,
) {
    let max_sigma = (0..h)
        .map(|y| sigma_of_row(y as f64))
        .fold(0.0f64, f64::max);
    let pad = (4.0 * max_sigma).ceil() as i64 + 1;
    let horizontal = |row: &[f64], sigma: f64, out: &mut [f64]| {
        let k = gaussian_kernel(sigma);
        let r = (k.len() / 2) as i64;
        for (x, o) in out.iter_mut().enumerate().take(w) {
            let mut acc = 0.0;
            for (j, kv) in k.iter().enumerate() {
                let xx = (x as i64 + j as i64 - r).clamp(0, w as i64 - 1) as usize;
                acc += kv * row[xx];
            }
            *o = acc;
        }
    };
    let mut out = vec![0.0; w * h];
    match form {
        BlurForm::Gather => {
            let mut col = vec![0.0; w];
            for y in 0..h {
                let sigma = sigma_of_row(y as f64);
                let k = gaussian_kernel(sigma);
                let r = (k.len() / 2) as i64;
                col.iter_mut().for_each(|v| *v = 0.0);
                for (j, kv) in k.iter().enumerate() {
                    let yy = (y as i64 + j as i64 - r).clamp(0, h as i64 - 1) as usize;
                    for x in 0..w {
                        col[x] += kv * buf[yy * w + x];
                    }
                }
                horizontal(&col, sigma, &mut out[y * w..(y + 1) * w]);
            }
        }
        BlurForm::Scatter => {
            let mut row = vec![0.0; w];
            for ys in -pad..h as i64 + pad {
                let sigma = sigma_of_row(ys as f64);
                let src = ys.clamp(0, h as i64 - 1) as usize;
                horizontal(&buf[src * w..(src + 1) * w], sigma, &mut row);
                let k = gaussian_kernel(sigma);
                let r = (k.len() / 2) as i64;
                for (j, kv) in k.iter().enumerate() {
                    let yy = ys + j as i64 - r;
                    if yy < 0 || yy >= h as i64 {
                        continue;
                    }
                    let dst = &mut out[yy as usize * w..(yy as usize + 1) * w];
                    for x in 0..w {
                        dst[x] += kv * row[x];
                    }
                }
            }
        }
    }
    buf.copy_from_slice(&out);
}

/// Apply a PSF to a `w × h` row-major buffer.
pub fn apply_psf(buf: &mut [f64], w: usize, h: usize, psf: Psf) {
    match psf {
        Psf::Gaussian { sigma } => blur(buf, w, h, sigma),
        Psf::Anisotropic {
            sigma_major,
            sigma_minor,
            angle_deg,
        } => {
            let (s, c) = angle_deg.to_radians().sin_cos();
            let r = (4.0 * sigma_major).ceil() as i64;
            convolve(buf, w, h, r, |dx, dy| {
                let a = (c * dx + s * dy) / sigma_major;
                let b = (-s * dx + c * dy) / sigma_minor;
                (-0.5 * (a * a + b * b)).exp()
            });
        }
        Psf::Halo {
            sigma,
            halo_sigma,
            halo_weight,
        } => {
            let mut halo = buf.to_vec();
            blur(buf, w, h, sigma);
            blur(&mut halo, w, h, halo_sigma);
            for (v, hv) in buf.iter_mut().zip(halo) {
                *v = (1.0 - halo_weight) * *v + halo_weight * hv;
            }
        }
        Psf::Disc { radius } => {
            let r = radius.ceil() as i64 + 1;
            convolve(
                buf,
                w,
                h,
                r,
                |dx, dy| {
                    if dx.hypot(dy) <= radius { 1.0 } else { 0.0 }
                },
            );
        }
        Psf::VaryingGaussian { form, .. } => {
            varying_blur(buf, w, h, form, &|y| {
                varying_sigma(&psf, y).expect("varying PSF")
            });
        }
    }
}

/// Noise-free radiance of a frame (grey levels, before the camera response).
pub struct Radiance {
    pub buf: Vec<f64>,
    /// Board mm → image pixels (row-major 3×3).
    pub h: [[f64; 3]; 3],
}

/// Render the noise-free radiance of `board` under `spec` (geometry,
/// illumination, glare, PSF).
pub fn radiance(board: &BoardRaster, spec: &SceneSpec) -> Radiance {
    let (lo, hi) = board.target.marker_bounds_mm().expect("non-empty board");
    let mid = [
        0.5 * f64::from(lo[0] + hi[0]),
        0.5 * f64::from(lo[1] + hi[1]),
    ];
    let h = scene_homography(spec, mid);
    let h_inv = inverse3(&h);
    let lens = RadialDivision::of(spec);
    let (w, hh) = (spec.image_size[0] as usize, spec.image_size[1] as usize);

    // Reflectance, area-sampled.
    let mut buf = vec![0.0f64; w * hh];
    let n = spec.supersample as f64;
    for y in 0..hh {
        for x in 0..w {
            let mut acc = 0.0;
            for sy in 0..spec.supersample {
                for sx in 0..spec.supersample {
                    let px = x as f64 + (sx as f64 + 0.5) / n - 0.5;
                    let py = y as f64 + (sy as f64 + 0.5) / n - 0.5;
                    let p = lens.map_or([px, py], |d| d.undistort([px, py]));
                    acc += board.reflectance(apply_h(&h_inv, p));
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

    apply_psf(&mut buf, w, hh, spec.psf);
    Radiance { buf, h }
}

/// Unit-variance Gaussian noise field, white or Gaussian-correlated.
fn noise_field(w: usize, h: usize, corr_px: f64, seed: u64) -> Vec<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut field: Vec<f64> = (0..w * h)
        .map(|_| {
            // Box–Muller.
            let u1: f64 = rng.random_range(f64::EPSILON..1.0);
            let u2: f64 = rng.random_range(0.0..1.0);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
        .collect();
    if corr_px > 0.0 {
        blur(&mut field, w, h, corr_px);
        // A normalized Gaussian kernel shrinks white noise variance by
        // sum(k^2) per axis.
        let k = gaussian_kernel(corr_px);
        let gain = k.iter().map(|v| v * v).sum::<f64>();
        field.iter_mut().for_each(|v| *v /= gain);
    }
    field
}

/// One noisy 8-bit exposure of a radiance: camera response, noise,
/// quantization.
pub fn expose(radiance: &Radiance, spec: &SceneSpec, seed: u64) -> Scene {
    let (w, hh) = (spec.image_size[0] as usize, spec.image_size[1] as usize);
    let noise = noise_field(w, hh, spec.noise_corr_px, seed);
    let mut image = GrayImage::new(spec.image_size[0], spec.image_size[1]);
    for y in 0..hh {
        for x in 0..w {
            let mut v = radiance.buf[y * w + x];
            if spec.gamma != 1.0 {
                v = 255.0 * (v.max(0.0) / 255.0).powf(spec.gamma);
            }
            let v = v + spec.noise_sigma * noise[y * w + x];
            image.put_pixel(
                x as u32,
                y as u32,
                Luma([v.round().clamp(0.0, 255.0) as u8]),
            );
        }
    }
    Scene {
        image,
        h: radiance.h,
        distortion: RadialDivision::of(spec),
    }
}

/// Render one frame of `board` under `spec`.
pub fn render(board: &BoardRaster, spec: &SceneSpec) -> Scene {
    expose(&radiance(board, spec), spec, spec.seed)
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
