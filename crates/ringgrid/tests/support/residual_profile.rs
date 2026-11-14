//! Aggregate structure of the photometric fit residual.
//!
//! Input: the per-marker residual maps of
//! `Detector::detect_with_photometric_residuals` (accepted fits only, chosen by
//! the caller). Output: profile tables that show *where* the intensity model
//! fails, as plain data with a text and a JSON rendering. Shared by the
//! `photometric_residual_profile` example and the synthetic sanity test.
//!
//! Residual = observed − model grey level. Edge distances are image pixels,
//! positive on the ink side of the edge.

#![allow(dead_code)] // each user takes a subset

use ringgrid::photometric_residuals::PhotometricResidualSample as Sample;
use serde_json::{Value, json};

pub const N_TERCILES: usize = 3;
pub const RADIUS_BIN: f64 = 0.02;
pub const RADIUS_MAX: f64 = 1.5;
pub const N_RADIUS: usize = 75;
pub const EDGE_BIN: f64 = 0.25;
pub const EDGE_RANGE: f64 = 5.0;
pub const N_EDGE: usize = 40;
pub const N_ANGLE: usize = 16;
pub const N_GREY: usize = 16;
/// Pixels within this many px of an outer edge feed the angle table.
const ANGLE_BAND_PX: f64 = 2.0;
/// Flat-region pixels are farther than this many blur sigmas from any edge.
const FLAT_SIGMAS: f64 = 3.0;
const EDGE_CLASSES: [&str; 3] = ["inner_ring", "outer_ring", "code_sector"];
const BACKFIT_ROUNDS: usize = 8;

/// One accepted marker's fit residual map.
pub struct MarkerData {
    /// Fitted centre, image px.
    pub center: [f64; 2],
    /// Fitted PSF sigma, px.
    pub blur_sigma_px: f64,
    pub samples: Vec<Sample>,
}

/// Count, sum, sum of squares.
#[derive(Clone, Copy, Default, Debug)]
pub struct Acc {
    pub n: u64,
    pub sum: f64,
    pub sumsq: f64,
}

impl Acc {
    pub fn add(&mut self, v: f64) {
        self.n += 1;
        self.sum += v;
        self.sumsq += v * v;
    }
    pub fn mean(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.sum / self.n as f64
        }
    }
    pub fn rms(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            (self.sumsq / self.n as f64).sqrt()
        }
    }
    /// Standard deviation about the mean.
    pub fn std(&self) -> f64 {
        if self.n < 2 {
            f64::NAN
        } else {
            (self.sumsq / self.n as f64 - self.mean().powi(2))
                .max(0.0)
                .sqrt()
        }
    }
    fn json(&self) -> Value {
        json!({"n": self.n, "mean": jf(self.mean()), "rms": jf(self.rms())})
    }
}

fn jf(v: f64) -> Value {
    if v.is_finite() { json!(v) } else { Value::Null }
}

/// Variance split of one blur group.
#[derive(Clone, Debug, Default)]
pub struct Split {
    pub n_pixels: u64,
    /// Mean square of the raw residual.
    pub total_ms: f64,
    /// Mean square of the part the across-marker mean profile explains.
    pub explained_ms: f64,
    /// Mean square of what is left.
    pub remainder_ms: f64,
    /// Lag-1 autocorrelation (horizontal, vertical, pooled) of the raw residual.
    pub acf_raw: [f64; 3],
    /// The same for the remainder.
    pub acf_remainder: [f64; 3],
}

#[derive(Clone, Debug, Default)]
pub struct Flat {
    pub n: u64,
    pub sigma: f64,
    pub acf: [f64; 3],
}

pub struct Report {
    pub n_markers: usize,
    pub n_pixels: u64,
    /// Blur sigma boundaries between the terciles.
    pub blur_edges: [f64; 2],
    pub markers_per_tercile: [usize; N_TERCILES],
    /// `radius[t][bin]`.
    pub radius: Vec<Vec<Acc>>,
    /// `edge[class][t][bin]`.
    pub edge: Vec<Vec<Vec<Acc>>>,
    /// Near the outer edge by image-frame angle: ink side, paper side.
    pub angle_ink: Vec<Acc>,
    pub angle_paper: Vec<Acc>,
    /// The same per blur tercile: `angle_*_by_tercile[t][bin]`.
    pub angle_ink_by_tercile: Vec<Vec<Acc>>,
    pub angle_paper_by_tercile: Vec<Vec<Acc>>,
    /// Terciles then all.
    pub split: Vec<Split>,
    pub grey: Vec<Acc>,
    pub flat: Flat,
}

fn bin_index(v: f64, lo: f64, width: f64, n: usize) -> Option<usize> {
    if !v.is_finite() || v < lo {
        return None;
    }
    let i = ((v - lo) / width).floor() as usize;
    (i < n).then_some(i)
}

fn radius_bin(s: &Sample) -> Option<usize> {
    let r = f64::from(s.radius);
    if !r.is_finite() || r < 0.0 {
        return None;
    }
    Some(((r / RADIUS_BIN).floor() as usize).min(N_RADIUS - 1))
}

fn dist(s: &Sample, class: usize) -> f64 {
    f64::from(match class {
        0 => s.d_inner,
        1 => s.d_outer,
        _ => s.d_code,
    })
}

fn edge_bin(s: &Sample, class: usize) -> Option<usize> {
    bin_index(dist(s, class), -EDGE_RANGE, EDGE_BIN, N_EDGE)
}

/// Lag-1 autocorrelation of `value` over horizontally and vertically adjacent
/// pixels of every marker (`None` excludes a pixel): `[horizontal, vertical,
/// pooled]`, the pair-weighted normalised cross-product.
pub fn lag1(markers: &[MarkerData], value: &dyn Fn(usize, usize) -> Option<f64>) -> [f64; 3] {
    // [num, sum a^2, sum b^2] for horizontal and vertical pairs.
    let mut h = [0.0f64; 3];
    let mut v = [0.0f64; 3];
    for (m, marker) in markers.iter().enumerate() {
        if marker.samples.is_empty() {
            continue;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        for s in &marker.samples {
            x0 = x0.min(s.x);
            x1 = x1.max(s.x);
            y0 = y0.min(s.y);
            y1 = y1.max(s.y);
        }
        let (w, ht) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
        let mut grid = vec![None::<f64>; w * ht];
        for (i, s) in marker.samples.iter().enumerate() {
            grid[(s.y - y0) as usize * w + (s.x - x0) as usize] = value(m, i);
        }
        for y in 0..ht {
            for x in 0..w {
                let Some(a) = grid[y * w + x] else { continue };
                if x + 1 < w
                    && let Some(b) = grid[y * w + x + 1]
                {
                    h[0] += a * b;
                    h[1] += a * a;
                    h[2] += b * b;
                }
                if y + 1 < ht
                    && let Some(b) = grid[(y + 1) * w + x]
                {
                    v[0] += a * b;
                    v[1] += a * a;
                    v[2] += b * b;
                }
            }
        }
    }
    let acf = |t: [f64; 3]| t[0] / (t[1] * t[2]).sqrt();
    [acf(h), acf(v), acf([h[0] + v[0], h[1] + v[1], h[2] + v[2]])]
}

/// Fit `r ~ a[radius bin] + sum_c b_c[edge bin]` by backfitting over the
/// pixels of `members`; returns the fitted value of every pixel.
fn backfit(markers: &[MarkerData], members: &[usize]) -> Vec<Vec<f64>> {
    let mut a = vec![0.0f64; N_RADIUS];
    let mut b = vec![vec![0.0f64; N_EDGE]; 3];
    let resid = |s: &Sample| f64::from(s.residual());
    for _ in 0..BACKFIT_ROUNDS {
        // Radial term.
        let mut acc = vec![Acc::default(); N_RADIUS];
        for &m in members {
            for s in &markers[m].samples {
                let Some(rb) = radius_bin(s) else { continue };
                let mut other = 0.0;
                for (c, bc) in b.iter().enumerate() {
                    if let Some(eb) = edge_bin(s, c) {
                        other += bc[eb];
                    }
                }
                acc[rb].add(resid(s) - other);
            }
        }
        for (ai, ac) in a.iter_mut().zip(&acc) {
            *ai = if ac.n > 0 { ac.mean() } else { 0.0 };
        }
        // Edge terms.
        for c in 0..3 {
            let mut acc = vec![Acc::default(); N_EDGE];
            for &m in members {
                for s in &markers[m].samples {
                    let Some(eb) = edge_bin(s, c) else { continue };
                    let mut other = radius_bin(s).map_or(0.0, |rb| a[rb]);
                    for (c2, bc) in b.iter().enumerate() {
                        if c2 != c
                            && let Some(e2) = edge_bin(s, c2)
                        {
                            other += bc[e2];
                        }
                    }
                    acc[eb].add(resid(s) - other);
                }
            }
            for (bi, ac) in b[c].iter_mut().zip(&acc) {
                *bi = if ac.n > 0 { ac.mean() } else { 0.0 };
            }
        }
    }
    markers
        .iter()
        .map(|mk| {
            mk.samples
                .iter()
                .map(|s| {
                    let mut f = radius_bin(s).map_or(0.0, |rb| a[rb]);
                    for (c, bc) in b.iter().enumerate() {
                        if let Some(eb) = edge_bin(s, c) {
                            f += bc[eb];
                        }
                    }
                    f
                })
                .collect()
        })
        .collect()
}

/// Analyse the residual maps of accepted markers.
pub fn analyze(markers: &[MarkerData]) -> Report {
    let mut blurs: Vec<f64> = markers.iter().map(|m| m.blur_sigma_px).collect();
    blurs.sort_by(|a, b| a.total_cmp(b));
    let n = blurs.len();
    let blur_edges = if n == 0 {
        [f64::NAN; 2]
    } else {
        [blurs[n / 3], blurs[(2 * n / 3).min(n - 1)]]
    };
    let tercile =
        |sigma: f64| usize::from(sigma >= blur_edges[0]) + usize::from(sigma >= blur_edges[1]);
    let tercile_of: Vec<usize> = markers.iter().map(|m| tercile(m.blur_sigma_px)).collect();
    let mut markers_per_tercile = [0usize; N_TERCILES];
    for &t in &tercile_of {
        markers_per_tercile[t] += 1;
    }

    let mut radius = vec![vec![Acc::default(); N_RADIUS]; N_TERCILES];
    let mut edge = vec![vec![vec![Acc::default(); N_EDGE]; N_TERCILES]; 3];
    let mut angle_ink = vec![Acc::default(); N_ANGLE];
    let mut angle_paper = vec![Acc::default(); N_ANGLE];
    let mut angle_ink_by_tercile = vec![vec![Acc::default(); N_ANGLE]; N_TERCILES];
    let mut angle_paper_by_tercile = vec![vec![Acc::default(); N_ANGLE]; N_TERCILES];
    let mut grey = vec![Acc::default(); N_GREY];
    let mut n_pixels = 0u64;
    for (m, marker) in markers.iter().enumerate() {
        let t = tercile_of[m];
        for s in &marker.samples {
            n_pixels += 1;
            let r = f64::from(s.residual());
            if let Some(rb) = radius_bin(s) {
                radius[t][rb].add(r);
            }
            for (c, per_class) in edge.iter_mut().enumerate() {
                if let Some(eb) = edge_bin(s, c) {
                    per_class[t][eb].add(r);
                }
            }
            let d_out = f64::from(s.d_outer);
            if d_out.is_finite() && d_out.abs() <= ANGLE_BAND_PX {
                let a = f64::from(s.y) - marker.center[1];
                let b = f64::from(s.x) - marker.center[0];
                let ang = a.atan2(b);
                let i = (((ang + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)
                    * N_ANGLE as f64)
                    .floor() as usize)
                    .min(N_ANGLE - 1);
                if d_out > 0.0 {
                    angle_ink[i].add(r);
                    angle_ink_by_tercile[t][i].add(r);
                } else {
                    angle_paper[i].add(r);
                    angle_paper_by_tercile[t][i].add(r);
                }
            }
            if let Some(g) = bin_index(f64::from(s.observed), 0.0, 256.0 / N_GREY as f64, N_GREY) {
                grey[g].add(r);
            }
        }
    }

    // Variance split per tercile and for everything.
    let mut split = Vec::new();
    for g in 0..=N_TERCILES {
        let members: Vec<usize> = (0..markers.len())
            .filter(|&m| g == N_TERCILES || tercile_of[m] == g)
            .collect();
        let fitted = backfit(markers, &members);
        let mut tot = Acc::default();
        let mut exp = Acc::default();
        let mut rem = Acc::default();
        for &m in &members {
            for (s, &f) in markers[m].samples.iter().zip(&fitted[m]) {
                let r = f64::from(s.residual());
                tot.add(r);
                exp.add(f);
                rem.add(r - f);
            }
        }
        let in_group = |m: usize| members.binary_search(&m).is_ok();
        let acf_raw = lag1(markers, &|m, i| {
            in_group(m).then(|| f64::from(markers[m].samples[i].residual()))
        });
        let acf_remainder = lag1(markers, &|m, i| {
            in_group(m).then(|| f64::from(markers[m].samples[i].residual()) - fitted[m][i])
        });
        split.push(Split {
            n_pixels: tot.n,
            total_ms: tot.sumsq / tot.n.max(1) as f64,
            explained_ms: exp.sumsq / tot.n.max(1) as f64,
            remainder_ms: rem.sumsq / tot.n.max(1) as f64,
            acf_raw,
            acf_remainder,
        });
    }

    // Flat regions: far from every modelled edge.
    let is_flat = |m: usize, i: usize| {
        let s = &markers[m].samples[i];
        f64::from(s.d_any) > FLAT_SIGMAS * markers[m].blur_sigma_px
    };
    let mut flat_acc = Acc::default();
    for (m, marker) in markers.iter().enumerate() {
        for (i, sample) in marker.samples.iter().enumerate() {
            if is_flat(m, i) {
                flat_acc.add(f64::from(sample.residual()));
            }
        }
    }
    let flat_acf = lag1(markers, &|m, i| {
        is_flat(m, i).then(|| f64::from(markers[m].samples[i].residual()))
    });
    let flat = Flat {
        n: flat_acc.n,
        sigma: flat_acc.std(),
        acf: flat_acf,
    };

    Report {
        n_markers: markers.len(),
        n_pixels,
        blur_edges,
        markers_per_tercile,
        radius,
        edge,
        angle_ink,
        angle_paper,
        angle_ink_by_tercile,
        angle_paper_by_tercile,
        split,
        grey,
        flat,
    }
}

/// The part of an angle table (bins of `atan2(dy, dx)` from -180°) that
/// varies along the image y axis: the pixel-weighted means over the upper
/// half (above the centre, angles -180..0) and the lower half, and the first
/// harmonic `A` of `A sin(angle)` fitted to the bin means. A blur that grows
/// downwards across a marker makes the upper edge steeper than a single-PSF
/// model (ink side negative, paper side positive) and the lower edge
/// shallower.
#[derive(Clone, Copy, Debug)]
pub struct Dipole {
    pub upper: f64,
    pub lower: f64,
    pub harmonic: f64,
}

pub fn dipole(acc: &[Acc]) -> Dipole {
    let half = |range: std::ops::Range<usize>| {
        let (mut s, mut n) = (0.0, 0u64);
        for a in &acc[range] {
            s += a.sum;
            n += a.n;
        }
        s / n.max(1) as f64
    };
    let (mut num, mut den) = (0.0, 0.0);
    for (i, a) in acc.iter().enumerate() {
        if a.n == 0 {
            continue;
        }
        let ang = -std::f64::consts::PI
            + (i as f64 + 0.5) * 2.0 * std::f64::consts::PI / acc.len() as f64;
        num += a.mean() * ang.sin();
        den += ang.sin().powi(2);
    }
    Dipole {
        upper: half(0..acc.len() / 2),
        lower: half(acc.len() / 2..acc.len()),
        harmonic: if den > 0.0 { num / den } else { f64::NAN },
    }
}

fn cell(a: &Acc) -> String {
    if a.n == 0 {
        format!("{:>7} {:>7} {:>7}", "-", "-", 0)
    } else {
        format!("{:>7.2} {:>7.2} {:>7}", a.mean(), a.rms(), a.n)
    }
}

impl Report {
    pub fn print(&self) {
        let [e0, e1] = self.blur_edges;
        println!(
            "{} accepted markers, {} pixels; blur terciles (fitted sigma, px): \
             T0 < {e0:.2} <= T1 < {e1:.2} <= T2; markers {:?}",
            self.n_markers, self.n_pixels, self.markers_per_tercile
        );
        println!("residual = observed - model (grey levels); cells are mean / rms / n");

        println!("\n(a) residual vs normalised radius (outer print radius = 1)");
        println!(
            "{:>11} | {:^23} | {:^23} | {:^23}",
            "radius", "T0", "T1", "T2"
        );
        for b in 0..N_RADIUS {
            if (0..N_TERCILES).all(|t| self.radius[t][b].n == 0) {
                continue;
            }
            println!(
                "{:>4.2}-{:<4.2}   | {} | {} | {}",
                b as f64 * RADIUS_BIN,
                (b + 1) as f64 * RADIUS_BIN,
                cell(&self.radius[0][b]),
                cell(&self.radius[1][b]),
                cell(&self.radius[2][b]),
            );
        }

        println!("\n(b) residual vs signed distance to the nearest model edge (px, + = ink side)");
        for (c, name) in EDGE_CLASSES.iter().enumerate() {
            println!("  {name} edges");
            println!(
                "  {:>11} | {:^23} | {:^23} | {:^23}",
                "distance", "T0", "T1", "T2"
            );
            for b in 0..N_EDGE {
                if (0..N_TERCILES).all(|t| self.edge[c][t][b].n == 0) {
                    continue;
                }
                println!(
                    "  {:>5.2}..{:<5.2} | {} | {} | {}",
                    -EDGE_RANGE + b as f64 * EDGE_BIN,
                    -EDGE_RANGE + (b + 1) as f64 * EDGE_BIN,
                    cell(&self.edge[c][0][b]),
                    cell(&self.edge[c][1][b]),
                    cell(&self.edge[c][2][b]),
                );
            }
        }

        println!(
            "\n(c) mean residual vs image-frame angle of the pixel from the centre \
             (|distance to outer edge| <= {ANGLE_BAND_PX} px)"
        );
        println!(
            "{:>9} | {:^15} | {:^15}",
            "angle deg", "ink side", "paper side"
        );
        for i in 0..N_ANGLE {
            let lo = -180.0 + 360.0 * i as f64 / N_ANGLE as f64;
            println!(
                "{:>4.0}..{:<4.0} | {:>7.2} {:>7} | {:>7.2} {:>7}",
                lo,
                lo + 360.0 / N_ANGLE as f64,
                self.angle_ink[i].mean(),
                self.angle_ink[i].n,
                self.angle_paper[i].mean(),
                self.angle_paper[i].n,
            );
        }
        println!(
            "  dipole along image y (upper = above the centre; A = amplitude of A sin(angle)):"
        );
        println!(
            "  {:>5} | {:^23} | {:^23}",
            "group", "ink: upper lower A", "paper: upper lower A"
        );
        let groups = (0..N_TERCILES)
            .map(|t| {
                (
                    format!("T{t}"),
                    &self.angle_ink_by_tercile[t],
                    &self.angle_paper_by_tercile[t],
                )
            })
            .chain(std::iter::once((
                "all".to_string(),
                &self.angle_ink,
                &self.angle_paper,
            )));
        for (name, ink, paper) in groups {
            let (i, p) = (dipole(ink), dipole(paper));
            println!(
                "  {name:>5} | {:>+7.2} {:>+7.2} {:>+7.2} | {:>+7.2} {:>+7.2} {:>+7.2}",
                i.upper, i.lower, i.harmonic, p.upper, p.lower, p.harmonic
            );
        }

        println!(
            "\n(d) variance split: raw residual = part explained by the across-marker \
             mean profile of (a)+(b) + remainder"
        );
        println!(
            "{:>5} {:>9} {:>8} {:>10} {:>10} {:>8} {:>9} {:>9}",
            "group", "pixels", "rms", "explained", "remainder", "rem rms", "acf1 raw", "acf1 rem"
        );
        for (g, sp) in self.split.iter().enumerate() {
            let name = if g == N_TERCILES {
                "all".to_string()
            } else {
                format!("T{g}")
            };
            println!(
                "{:>5} {:>9} {:>8.2} {:>9.1}% {:>9.1}% {:>8.2} {:>9.2} {:>9.2}",
                name,
                sp.n_pixels,
                sp.total_ms.sqrt(),
                100.0 * sp.explained_ms / sp.total_ms,
                100.0 * sp.remainder_ms / sp.total_ms,
                sp.remainder_ms.sqrt(),
                sp.acf_raw[2],
                sp.acf_remainder[2],
            );
        }

        println!("\n(e) residual vs observed grey level");
        println!("{:>9} | {:^23}", "grey", "mean / rms / n");
        for (i, a) in self.grey.iter().enumerate() {
            if a.n == 0 {
                continue;
            }
            let w = 256 / N_GREY;
            println!("{:>3}..{:<3}  | {}", i * w, (i + 1) * w - 1, cell(a));
        }

        println!(
            "\n(f) flat regions (farther than {FLAT_SIGMAS} blur sigmas from any edge): \
             {} pixels, sigma {:.2}, lag-1 acf horizontal {:.2} vertical {:.2} pooled {:.2}",
            self.flat.n, self.flat.sigma, self.flat.acf[0], self.flat.acf[1], self.flat.acf[2]
        );
    }

    pub fn to_json(&self) -> Value {
        let accs = |v: &[Acc]| Value::Array(v.iter().map(Acc::json).collect());
        let acf =
            |a: [f64; 3]| json!({"horizontal": jf(a[0]), "vertical": jf(a[1]), "pooled": jf(a[2])});
        json!({
            "n_markers": self.n_markers,
            "n_pixels": self.n_pixels,
            "blur_tercile_edges_px": [self.blur_edges[0], self.blur_edges[1]],
            "markers_per_tercile": self.markers_per_tercile,
            "radius": {
                "bin_width": RADIUS_BIN,
                "units": "outer print radius",
                "by_tercile": self.radius.iter().map(|t| accs(t)).collect::<Vec<_>>(),
            },
            "edge": {
                "bin_width_px": EDGE_BIN,
                "range_px": [-EDGE_RANGE, EDGE_RANGE],
                "sign": "positive on the ink side",
                "classes": EDGE_CLASSES.iter().enumerate().map(|(c, name)| json!({
                    "class": name,
                    "by_tercile": self.edge[c].iter().map(|t| accs(t)).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
            },
            "angle_outer_edge": {
                "bins": N_ANGLE,
                "band_px": ANGLE_BAND_PX,
                "ink_side": accs(&self.angle_ink),
                "paper_side": accs(&self.angle_paper),
                "ink_side_by_tercile": self.angle_ink_by_tercile.iter().map(|t| accs(t)).collect::<Vec<_>>(),
                "paper_side_by_tercile": self.angle_paper_by_tercile.iter().map(|t| accs(t)).collect::<Vec<_>>(),
            },
            "variance_split": self.split.iter().enumerate().map(|(g, s)| json!({
                "group": if g == N_TERCILES { "all".to_string() } else { format!("T{g}") },
                "n_pixels": s.n_pixels,
                "total_ms": s.total_ms,
                "explained_ms": s.explained_ms,
                "remainder_ms": s.remainder_ms,
                "acf1_raw": acf(s.acf_raw),
                "acf1_remainder": acf(s.acf_remainder),
            })).collect::<Vec<_>>(),
            "grey": {"bins": N_GREY, "by_bin": accs(&self.grey)},
            "flat": {
                "n": self.flat.n,
                "sigma": jf(self.flat.sigma),
                "acf1": acf(self.flat.acf),
            },
        })
    }
}
