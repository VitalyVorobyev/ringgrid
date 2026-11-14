//! Localization-scatter metric on real captures of a coded hex board.
//!
//! Each input image is a horizontal strip of equally wide camera tiles (one
//! tile per camera). Every tile is detected with [`Detector::detect_adaptive`];
//! the decoded markers then give `(board_xy_mm, image_xy)` pairs. Per tile,
//! each image coordinate is fitted as a full quartic polynomial (15 terms) of
//! the normalized board position. A smooth warp of that order absorbs the
//! camera model, the pose and a smooth board bow, so what remains is
//! per-detection localization scatter.
//!
//! Fit: least squares, then 3 rounds of outlier rejection at
//! `max(4 * median|r|, 0.3 px)`; tiles with fewer than 27 markers are skipped;
//! residual magnitudes are scaled by `sqrt(n / (n - 15))`. Reported per camera
//! (tile index): RMS and median of the inlier residual magnitude.
//!
//! ```text
//! cargo run --release -p ringgrid-cli --example strip_warp_residual -- \
//!     --hex 8,15,14,4.8,3.2,1.152 --tile-width 720 \
//!     [--center-method projective-center|photometric] [--dump out.json] img0.png img1.png ...
//! ```

use std::path::PathBuf;
use std::time::Instant;

use nalgebra::{DMatrix, DVector};
use ringgrid::{CircleRefinementMethod, DetectConfig, Detector, TargetLayout};

const N_TERMS: usize = 15;
const MIN_MARKERS: usize = 27;
const OUTLIER_ROUNDS: usize = 3;

struct Args {
    hex: [f32; 6],
    tile_width: u32,
    method: Option<String>,
    dump: Option<PathBuf>,
    images: Vec<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut hex = None;
    let mut tile_width = None;
    let mut method = None;
    let mut dump = None;
    let mut images = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--hex" => {
                let v = it.next().ok_or("--hex needs a value")?;
                let parts: Vec<f32> = v
                    .split(',')
                    .map(|s| s.trim().parse::<f32>().map_err(|e| e.to_string()))
                    .collect::<Result<_, _>>()?;
                if parts.len() != 6 {
                    return Err(
                        "--hex takes pitch,rows,long_row_cols,outer,inner,ring_width".into(),
                    );
                }
                hex = Some([parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]]);
            }
            "--tile-width" => {
                let v = it.next().ok_or("--tile-width needs a value")?;
                tile_width = Some(v.parse::<u32>().map_err(|e| e.to_string())?);
            }
            "--center-method" => method = Some(it.next().ok_or("--center-method needs a value")?),
            "--dump" => dump = Some(PathBuf::from(it.next().ok_or("--dump needs a path")?)),
            _ if a.starts_with("--") => return Err(format!("unknown flag {a}")),
            _ => images.push(PathBuf::from(a)),
        }
    }
    Ok(Args {
        hex: hex.ok_or("--hex is required")?,
        tile_width: tile_width.ok_or("--tile-width is required")?,
        method,
        dump,
        images,
    })
}

/// The 15 monomials `x^i y^j`, `i + j <= 4`.
fn quartic_terms(x: f64, y: f64) -> [f64; N_TERMS] {
    let mut out = [0.0; N_TERMS];
    let mut k = 0;
    for deg in 0..=4 {
        for j in 0..=deg {
            let i = deg - j;
            out[k] = x.powi(i) * y.powi(j);
            k += 1;
        }
    }
    out
}

/// Least-squares quartic warp over `idx`; returns the residual vector of every
/// point (`image - fit`).
fn fit_warp(board: &[[f64; 2]], img: &[[f64; 2]], idx: &[usize]) -> Option<Vec<[f64; 2]>> {
    let n = idx.len();
    if n <= N_TERMS {
        return None;
    }
    let mut a = DMatrix::<f64>::zeros(n, N_TERMS);
    let mut bx = DVector::<f64>::zeros(n);
    let mut by = DVector::<f64>::zeros(n);
    for (r, &i) in idx.iter().enumerate() {
        let t = quartic_terms(board[i][0], board[i][1]);
        for c in 0..N_TERMS {
            a[(r, c)] = t[c];
        }
        bx[r] = img[i][0];
        by[r] = img[i][1];
    }
    let svd = a.svd(true, true);
    let cx = svd.solve(&bx, 1e-12).ok()?;
    let cy = svd.solve(&by, 1e-12).ok()?;
    Some(
        board
            .iter()
            .zip(img)
            .map(|(b, p)| {
                let t = quartic_terms(b[0], b[1]);
                let fx: f64 = t.iter().zip(cx.iter()).map(|(a, c)| a * c).sum();
                let fy: f64 = t.iter().zip(cy.iter()).map(|(a, c)| a * c).sum();
                [p[0] - fx, p[1] - fy]
            })
            .collect(),
    )
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Fit with outlier rejection; returns `(inlier indices, scaled residual
/// magnitudes, scaled residual vectors)`.
type Scatter = (Vec<usize>, Vec<f64>, Vec<[f64; 2]>);
fn tile_scatter(board_mm: &[[f64; 2]], img: &[[f64; 2]]) -> Option<Scatter> {
    // Normalize board coordinates to roughly [-1, 1].
    let n = board_mm.len();
    let (mx, my) = board_mm
        .iter()
        .fold((0.0, 0.0), |(a, b), p| (a + p[0], b + p[1]));
    let (mx, my) = (mx / n as f64, my / n as f64);
    let s = board_mm
        .iter()
        .map(|p| (p[0] - mx).abs().max((p[1] - my).abs()))
        .fold(1e-9, f64::max);
    let board: Vec<[f64; 2]> = board_mm
        .iter()
        .map(|p| [(p[0] - mx) / s, (p[1] - my) / s])
        .collect();

    let mut idx: Vec<usize> = (0..n).collect();
    let mut res = fit_warp(&board, img, &idx)?;
    for _ in 0..OUTLIER_ROUNDS {
        let mut mags: Vec<f64> = idx.iter().map(|&i| res[i][0].hypot(res[i][1])).collect();
        let thr = (4.0 * median(&mut mags)).max(0.3);
        idx = (0..n)
            .filter(|&i| res[i][0].hypot(res[i][1]) <= thr)
            .collect();
        if idx.len() < MIN_MARKERS {
            return None;
        }
        res = fit_warp(&board, img, &idx)?;
    }
    let m = idx.len() as f64;
    let scale = (m / (m - N_TERMS as f64)).sqrt();
    let mags = idx
        .iter()
        .map(|&i| res[i][0].hypot(res[i][1]) * scale)
        .collect();
    let vecs = idx
        .iter()
        .map(|&i| [res[i][0] * scale, res[i][1] * scale])
        .collect();
    Some((idx, mags, vecs))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let [pitch, rows, cols, outer, inner, width] = args.hex;
    let target = TargetLayout::coded_hex(pitch, rows as usize, cols as usize, outer, inner, width)?;
    let mut config = DetectConfig::from_target(target);
    match args.method.as_deref() {
        None | Some("projective-center") => {
            config.circle_refinement = CircleRefinementMethod::ProjectiveCenter
        }
        Some("none") => config.circle_refinement = CircleRefinementMethod::None,
        Some("photometric") => config.circle_refinement = CircleRefinementMethod::Photometric,
        Some(other) => return Err(format!("unknown --center-method {other}").into()),
    }
    let detector = Detector::with_config(config);

    let mut per_cam: Vec<Vec<f64>> = Vec::new();
    let mut radii_seen: Vec<[f64; 4]> = Vec::new();
    let mut dump = Vec::new();
    let mut total_s = 0.0;
    for path in &args.images {
        let gray = image::open(path)?.to_luma8();
        let (w, h) = gray.dimensions();
        let n_tiles = w / args.tile_width;
        if per_cam.len() < n_tiles as usize {
            per_cam.resize(n_tiles as usize, Vec::new());
        }
        for cam in 0..n_tiles {
            let tile =
                image::imageops::crop_imm(&gray, cam * args.tile_width, 0, args.tile_width, h)
                    .to_image();
            let t0 = Instant::now();
            let (result, diag) = detector.detect_adaptive_with_diagnostics(&tile)?;
            if let Some(r) = diag.photometric_print_radii_mm {
                radii_seen.push(r);
            }
            total_s += t0.elapsed().as_secs_f64();
            let pairs: Vec<_> = result
                .detected_markers
                .iter()
                .zip(&diag.markers)
                .filter_map(|(m, d)| m.board_xy_mm.map(|b| ((m, d), b)))
                .collect();
            if pairs.len() < MIN_MARKERS {
                continue;
            }
            let board: Vec<[f64; 2]> = pairs.iter().map(|(_, b)| *b).collect();
            let img: Vec<[f64; 2]> = pairs.iter().map(|((m, _), _)| m.center).collect();
            let Some((idx, mags, vecs)) = tile_scatter(&board, &img) else {
                continue;
            };
            if args.dump.is_some() {
                for (k, &i) in idx.iter().enumerate() {
                    let (m, d) = pairs[i].0;
                    dump.push(serde_json::json!({
                        "image": path.file_name().map(|s| s.to_string_lossy().to_string()),
                        "cam": cam,
                        "id": m.id,
                        "center": m.center,
                        "center_source": m.center_source,
                        "center_cov": m.center_cov,
                        "photometric": d.photometric,
                        "residual_px": mags[k],
                        "residual_vec_px": vecs[k],
                    }));
                }
            }
            per_cam[cam as usize].extend(mags);
        }
    }

    println!("cam  n_inliers  rms_px  median_px");
    for (cam, mags) in per_cam.iter_mut().enumerate() {
        if mags.is_empty() {
            println!("{cam:>3}  {:>9}  {:>6}  {:>9}", 0, "-", "-");
            continue;
        }
        let rms = (mags.iter().map(|r| r * r).sum::<f64>() / mags.len() as f64).sqrt();
        let n = mags.len();
        let med = median(mags);
        println!("{cam:>3}  {n:>9}  {rms:>6.3}  {med:>9.3}");
    }
    println!(
        "detection time: {:.1} ms per tile",
        1e3 * total_s / (args.images.len().max(1) as f64 * per_cam.len().max(1) as f64)
    );
    if !radii_seen.is_empty() {
        let n = radii_seen.len() as f64;
        let mean: Vec<f64> = (0..4)
            .map(|k| radii_seen.iter().map(|r| r[k]).sum::<f64>() / n)
            .collect();
        println!(
            "estimated print radii (mean over {} tiles, mm): {:.3} {:.3} {:.3} {:.3}",
            radii_seen.len(),
            mean[0],
            mean[1],
            mean[2],
            mean[3]
        );
    }
    if let Some(path) = args.dump {
        std::fs::write(path, serde_json::to_string(&dump)?)?;
    }
    Ok(())
}
