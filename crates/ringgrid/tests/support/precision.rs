//! Ground-truth precision harness: a catalogue of hard synthetic views of
//! every target kind, their exact ground truth, and one scorer for any
//! detection result (the library's, or a CLI run's JSON output — the same
//! scenes can be exported and run through another detector build).
//!
//! The scorer's job is to count *wrong* outputs — an ID, board position or
//! homography that would corrupt a camera calibration — next to recall, so a
//! precision/recall trade-off is visible:
//!
//! - **wrong ID**: a reported ID that is not the ID of the primary-board ring
//!   at that position — including an ID reported where no primary ring is
//!   (paper, clutter, a second board in view).
//! - **wrong position**: a reported absolute `board_xy_mm` / `grid_coord`
//!   that is not the cell at that position (plain targets: a wrong origin).
//! - **label error**: in an unanchored (relative) frame, a `grid_coord` that
//!   disagrees with the best lattice map of all reported labels onto truth.
//! - **wrong centre**: a reported centre matched to a ring but off its true
//!   centre by more than [`CENTRE_TIGHT_PX`] (tight) or
//!   `max(CENTRE_LOOSE_MIN_PX, CENTRE_LOOSE_FRACTION × diameter)` (loose).
//! - **wrong homography**: an absolute-frame homography that places the
//!   board wrongly — its median reprojection of the present true cells is
//!   off by more than `H_FRACTION` (half) of the lattice spacing, i.e. it
//!   predicts other rings. Plain approximation error (a homography cannot
//!   model lens distortion) is reported as the median, not counted.
//!
//! A reported marker is matched to the nearest present primary ring within
//! half the local lattice spacing; a ring is *present* when its centre is in
//! the frame and not covered by an occluder, mask or foreign region, and
//! *visible* (the recall denominator) when its whole outer rim is.

use std::collections::HashMap;

use image::{GrayImage, Luma};
use projective_grid::Coord;
use ringgrid::{
    CodebookProfile, CodedRingSpec, HexGeometry, LatticeGeometry, MarkerCoding, OriginDots,
    RingGeometry, TargetLayout,
};
use serde_json::{Value, json};

use crate::support::blur_scene::{BoardRaster, Scene, SceneSpec, render};

pub const CENTRE_TIGHT_PX: f64 = 1.0;
pub const CENTRE_LOOSE_MIN_PX: f64 = 2.0;
pub const CENTRE_LOOSE_FRACTION: f64 = 0.1;
pub const H_FRACTION: f64 = 0.5;

/// The boards of the harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Board {
    /// The default 203-cell coded hex board (thin code band).
    HexDefault,
    /// 20×20 coded rect, 14 mm pitch.
    RectCoded,
    /// 37×34 coded hex (1239 cells) on the Extended codebook.
    HexExtended,
    /// Plain hex with an origin-dot triad.
    PlainHexDots,
    /// The 24×24 plain rect preset with origin dots.
    PlainRectDots,
    /// A coded hex board of another geometry: only ever a foreign object in
    /// view (clutter whose rings carry valid codewords).
    OtherCodedHex,
}

impl Board {
    pub fn name(self) -> &'static str {
        match self {
            Board::HexDefault => "hex",
            Board::RectCoded => "rect",
            Board::HexExtended => "hexext",
            Board::PlainHexDots => "plainhex",
            Board::PlainRectDots => "plainrect",
            Board::OtherCodedHex => "otherhex",
        }
    }

    pub fn layout(self) -> TargetLayout {
        match self {
            Board::HexDefault => TargetLayout::default_hex(),
            Board::RectCoded => {
                TargetLayout::coded_rect(14.0, 20, 20, 4.8, 3.2, 1.152).expect("coded rect")
            }
            Board::HexExtended => TargetLayout::new(
                "precision_hex_extended",
                LatticeGeometry::Hex(HexGeometry {
                    rows: 37,
                    long_row_cols: 34,
                    pitch_mm: 8.0,
                }),
                RingGeometry {
                    outer_radius_mm: 4.8,
                    inner_radius_mm: 3.2,
                },
                MarkerCoding::Coded16(CodedRingSpec {
                    ring_width_mm: 1.152,
                    codebook_profile: CodebookProfile::Extended,
                    id_assignment: None,
                }),
                None,
            )
            .expect("extended hex"),
            Board::PlainHexDots => {
                TargetLayout::plain_hex(8.0, 15, 14, 4.8, 3.2, OriginDots::Auto).expect("plain hex")
            }
            Board::PlainRectDots => TargetLayout::rect_24x24(),
            Board::OtherCodedHex => {
                TargetLayout::coded_hex(10.0, 11, 10, 6.0, 4.0, 1.44).expect("other hex")
            }
        }
    }

    /// Board raster density: the 468 mm Extended board at 300 dpi keeps its
    /// raster near 30 Mpx (still ≥ 3 raster px per camera px at 56 px rings).
    fn dpi(self) -> f32 {
        match self {
            Board::HexExtended | Board::PlainRectDots => 300.0,
            _ => 600.0,
        }
    }

    /// Board offset (mm from the board centre) of the default view: the
    /// Extended board is viewed over its codebook-extension rows.
    fn view_center_mm(self) -> [f64; 2] {
        match self {
            Board::HexExtended => [0.0, 120.0],
            _ => [0.0, 0.0],
        }
    }
}

/// Lazily built board rasters, shared by every scene of a run.
#[derive(Default)]
pub struct Boards {
    rasters: HashMap<Board, BoardRaster>,
}

impl Boards {
    pub fn get(&mut self, board: Board) -> &BoardRaster {
        self.rasters
            .entry(board)
            .or_insert_with(|| BoardRaster::with_dpi(board.layout(), board.dpi()))
    }
}

/// What, besides the board, the frame shows.
#[derive(Debug, Clone)]
pub enum Overlay {
    None,
    /// A dark band (an occluder) through the image: points within
    /// `half_width_px` of the line through the image centre at `angle_deg`.
    Band {
        angle_deg: f64,
        half_width_px: f64,
    },
    /// Only the image rectangle `[x0, y0, x1, y1]` shows the board; the rest
    /// is plain paper.
    Window([f64; 4]),
    /// Only the markers of these board cells (indices) are printed.
    Cells(Vec<usize>),
    /// A second board (`board`, seen through `spec`) covers the image right
    /// of `x_split_px`.
    Foreign {
        board: Board,
        spec: Box<SceneSpec>,
        x_split_px: f64,
    },
}

/// One scene of the catalogue.
#[derive(Debug, Clone)]
pub struct Case {
    pub name: String,
    /// Condition group, the row of the report table.
    pub group: String,
    pub board: Board,
    pub spec: SceneSpec,
    pub overlay: Overlay,
    /// Nominal outer diameter (px) the detector's scale prior is built from.
    pub diameter_px: f64,
}

impl Case {
    /// Scale prior `[min, max]` diameter (px) given to every detector,
    /// rounded to 0.01 px so a CLI run sees exactly the library's prior.
    pub fn diameter_range(&self) -> [f64; 2] {
        [0.5, 1.6].map(|k| (k * self.diameter_px * 100.0).round() / 100.0)
    }
}

/// Ground truth of one primary-board cell.
#[derive(Debug, Clone)]
pub struct GtCell {
    pub id: Option<usize>,
    pub coord: [i32; 2],
    pub xy_mm: [f64; 2],
    pub px: [f64; 2],
    pub present: bool,
    pub visible: bool,
    pub diam_px: f64,
    pub spacing_px: f64,
}

/// Ground truth of a rendered case.
#[derive(Debug, Clone)]
pub struct Truth {
    pub name: String,
    pub group: String,
    pub board: String,
    pub coded: bool,
    pub hex: bool,
    pub cells: Vec<GtCell>,
    /// Centres (px) and lattice spacings of foreign rings in view.
    pub foreign: Vec<([f64; 2], f64)>,
}

fn base_spec(board: Board, outer_radius_px: f64, blur: f64, tilt: f64, seed: u64) -> SceneSpec {
    let target = board.layout();
    SceneSpec {
        image_size: [1024, 768],
        tilt_axis_deg: 25.0,
        roll_deg: 17.0,
        board_center_mm: board.view_center_mm(),
        ..SceneSpec::for_target(&target, outer_radius_px, blur, tilt, 60.0, seed)
    }
}

/// Drawn outer radius (px, what [`SceneSpec::for_target`] takes) of a ring
/// whose nominal outer diameter is `diameter_px`.
fn draw_radius(board: Board, diameter_px: f64) -> f64 {
    let t = board.layout();
    0.5 * diameter_px * f64::from(t.outer_draw_radius_mm()) / f64::from(t.ring().outer_radius_mm)
}

struct Builder {
    board: Board,
    cases: Vec<Case>,
    seed: u64,
}

impl Builder {
    fn push(&mut self, group: &str, tag: &str, d_px: f64, spec: SceneSpec, overlay: Overlay) {
        self.cases.push(Case {
            name: format!("{}-{}-{}", self.board.name(), group, tag),
            group: group.to_string(),
            board: self.board,
            spec,
            overlay,
            diameter_px: d_px,
        });
    }

    /// A scene of rings `d_px` across (nominal outer diameter).
    fn spec(&mut self, d_px: f64, blur: f64, tilt: f64) -> SceneSpec {
        self.seed += 1;
        base_spec(
            self.board,
            draw_radius(self.board, d_px),
            blur,
            tilt,
            self.seed,
        )
    }
}

/// Board cells nearest the view centre: `n` of them pairwise at least
/// `min_sep_mm` apart (0 for a compact cluster).
fn cells_near_center(board: Board, n: usize, min_sep_mm: f32) -> Vec<usize> {
    let t = board.layout();
    let (lo, hi) = t.marker_bounds_mm().expect("bounds");
    let v = board.view_center_mm();
    let mid = [
        0.5 * (lo[0] + hi[0]) + v[0] as f32,
        0.5 * (lo[1] + hi[1]) + v[1] as f32,
    ];
    let dist = |i: usize| {
        let p = t.cells()[i].xy_mm;
        (p[0] - mid[0]).hypot(p[1] - mid[1])
    };
    let mut order: Vec<usize> = (0..t.cells().len()).collect();
    order.sort_by(|&a, &b| dist(a).total_cmp(&dist(b)));
    let mut picked: Vec<usize> = Vec::new();
    for i in order {
        let p = t.cells()[i].xy_mm;
        if picked.iter().all(|&j| {
            let q = t.cells()[j].xy_mm;
            (p[0] - q[0]).hypot(p[1] - q[1]) >= min_sep_mm
        }) {
            picked.push(i);
        }
        if picked.len() == n {
            break;
        }
    }
    picked
}

/// The scene catalogue of one board.
pub fn board_cases(board: Board) -> Vec<Case> {
    let mut b = Builder {
        board,
        cases: Vec::new(),
        seed: 1000 * (board as u64 + 1),
    };
    let coded = board.layout().is_coded();
    let span = board.layout().marker_span_mm().expect("span");
    // The operating range: blur sigma <= 2 px, rings >= 20 px across.
    for d in [20.0, 32.0, 56.0] {
        for blur in [0.8, 1.5, 2.0] {
            let spec = b.spec(d, blur, 20.0);
            b.push("blur", &format!("d{d}-b{blur}"), d, spec, Overlay::None);
        }
    }
    for d in [20.0, 32.0] {
        for blur in [1.5, 2.0] {
            let spec = b.spec(d, blur, 45.0);
            b.push("blur", &format!("d{d}-b{blur}-t45"), d, spec, Overlay::None);
        }
    }
    // Tilt.
    for (tilt, d) in [
        (0.0, 32.0),
        (30.0, 32.0),
        (45.0, 32.0),
        (55.0, 32.0),
        (58.0, 32.0),
        (60.0, 32.0),
        (62.0, 32.0),
        (65.0, 32.0),
        (60.0, 24.0),
    ] {
        let spec = b.spec(d, 1.0, tilt);
        b.push("tilt", &format!("t{tilt}-d{d}"), d, spec, Overlay::None);
    }
    // Radial distortion, no camera model.
    for (tag, lambda) in [
        ("mild", -0.05),
        ("strong", -0.15),
        ("l0.10", -0.10),
        ("l0.20", -0.20),
        ("l0.25", -0.25),
    ] {
        let spec = SceneSpec {
            radial_lambda: lambda,
            ..b.spec(32.0, 1.0, 20.0)
        };
        b.push("distortion", tag, 32.0, spec, Overlay::None);
    }
    // Partial views.
    let corner = [-0.45 * f64::from(span[0]), -0.45 * f64::from(span[1])];
    let spec = SceneSpec {
        board_center_mm: corner,
        ..b.spec(32.0, 1.0, 20.0)
    };
    b.push("partial", "corner", 32.0, spec, Overlay::None);
    let spec = SceneSpec {
        board_center_mm: [0.0, -0.5 * f64::from(span[1])],
        ..b.spec(32.0, 1.0, 15.0)
    };
    b.push(
        "partial",
        "edge-strip",
        32.0,
        spec,
        Overlay::Window([0.0, 300.0, 1024.0, 470.0]),
    );
    for (tag, w) in [
        ("window-20", [380.0, 280.0, 644.0, 488.0]),
        ("window-8", [430.0, 320.0, 594.0, 448.0]),
    ] {
        let spec = b.spec(32.0, 1.0, 20.0);
        b.push("partial", tag, 32.0, spec, Overlay::Window(w));
    }
    // Occluder splitting the view.
    let spec = b.spec(32.0, 1.0, 20.0);
    b.push(
        "occlusion",
        "band",
        32.0,
        spec,
        Overlay::Band {
            angle_deg: 67.0,
            half_width_px: 60.0,
        },
    );
    // Very few markers in view.
    let sep = 2.5 * board.layout().pitch_mm();
    for (tag, n, s) in [
        ("spread-1", 1, sep),
        ("spread-3", 3, sep),
        ("spread-6", 6, sep),
        ("cluster-3", 3, 0.0),
        ("cluster-6", 6, 0.0),
    ] {
        let spec = b.spec(40.0, 1.0, 10.0);
        b.push(
            "few",
            tag,
            40.0,
            spec,
            Overlay::Cells(cells_near_center(board, n, s)),
        );
    }
    // Low contrast, noise.
    for (tag, contrast, noise, blur) in [("c25-n3", 25.0, 3.0, 1.0), ("c35-n8", 35.0, 8.0, 1.5)] {
        let spec = SceneSpec {
            contrast,
            white: 70.0 + contrast,
            noise_sigma: noise,
            ..b.spec(32.0, blur, 25.0)
        };
        b.push("contrast", tag, 32.0, spec, Overlay::None);
    }
    // Outside the operating range (no recall target; fail-closed checks
    // only): heavy blur, unusable contrast.
    for (tag, d, blur, tilt) in [
        ("d20-b3", 20.0, 3.0, 20.0),
        ("d32-b3", 32.0, 3.0, 20.0),
        ("d24-b3.5", 24.0, 3.5, 35.0),
        ("d32-b2.5-t50", 32.0, 2.5, 50.0),
    ] {
        let spec = b.spec(d, blur, tilt);
        b.push("oor", tag, d, spec, Overlay::None);
    }
    let spec = SceneSpec {
        contrast: 20.0,
        white: 90.0,
        noise_sigma: 6.0,
        ..b.spec(32.0, 1.5, 25.0)
    };
    b.push("oor", "c20-n6", 32.0, spec, Overlay::None);
    // A second board in view: the same board at another pose (duplicate IDs),
    // and — coded targets — another coded board (valid codewords off-board).
    let mut foreign = vec![("same", board)];
    if coded {
        foreign.push(("other", Board::OtherCodedHex));
    }
    for (tag, other) in foreign {
        let primary = SceneSpec {
            board_center_mm: [0.2 * f64::from(span[0]), 0.0],
            ..b.spec(32.0, 1.0, 20.0)
        };
        let fspec = SceneSpec {
            roll_deg: -33.0,
            tilt_deg: 30.0,
            tilt_axis_deg: 110.0,
            px_per_mm: primary.px_per_mm * if other == board { 1.0 } else { 0.8 },
            board_center_mm: [-0.3 * f64::from(span[0]), 0.1 * f64::from(span[1])],
            seed: primary.seed + 500,
            ..primary.clone()
        };
        b.push(
            "foreign",
            tag,
            32.0,
            primary,
            Overlay::Foreign {
                board: other,
                spec: Box::new(fspec),
                x_split_px: 640.0,
            },
        );
    }
    b.cases
}

/// The operating-range grid (blur sigma <= 2 px, rings >= 20 px across):
/// diameter × blur × tilt × two seeds/rolls per board, no overlay.
pub fn grid_cases(board: Board) -> Vec<Case> {
    let mut cases = Vec::new();
    for d in [20.0, 24.0, 32.0, 44.0, 56.0] {
        for blur in [0.8, 1.5, 2.0] {
            for tilt in [0.0, 30.0, 50.0] {
                for k in 0..2u64 {
                    let spec = SceneSpec {
                        roll_deg: 17.0 + 40.0 * k as f64,
                        ..base_spec(board, draw_radius(board, d), blur, tilt, 7000 + k)
                    };
                    cases.push(Case {
                        name: format!("{}-grid-d{d}-b{blur}-t{tilt}-s{k}", board.name()),
                        group: "grid".to_string(),
                        board,
                        spec,
                        overlay: Overlay::None,
                        diameter_px: d,
                    });
                }
            }
        }
    }
    cases
}

/// Every board's catalogue.
pub fn catalogue() -> Vec<Case> {
    BOARDS.into_iter().flat_map(board_cases).collect()
}

/// The boards scenes are built of (the foreign-only board excluded).
pub const BOARDS: [Board; 5] = [
    Board::HexDefault,
    Board::RectCoded,
    Board::HexExtended,
    Board::PlainHexDots,
    Board::PlainRectDots,
];

/// Does the overlay hide the primary board at `p`?
fn covered(overlay: &Overlay, kept: &[[f64; 2]], keep_r: f64, size: [f64; 2], p: [f64; 2]) -> bool {
    match overlay {
        Overlay::None => false,
        Overlay::Band {
            angle_deg,
            half_width_px,
        } => {
            let (s, c) = angle_deg.to_radians().sin_cos();
            let (dx, dy) = (p[0] - 0.5 * size[0], p[1] - 0.5 * size[1]);
            (-s * dx + c * dy).abs() < *half_width_px
        }
        Overlay::Window(w) => !(p[0] >= w[0] && p[0] <= w[2] && p[1] >= w[1] && p[1] <= w[3]),
        Overlay::Cells(_) => kept
            .iter()
            .all(|k| (p[0] - k[0]).hypot(p[1] - k[1]) > keep_r),
        Overlay::Foreign { x_split_px, .. } => p[0] >= *x_split_px,
    }
}

/// Local diameter and lattice spacing (px) of a ring at board point `xy`.
fn local_scale(scene: &Scene, target: &TargetLayout, xy: [f64; 2]) -> (f64, f64) {
    let e = 0.5;
    let o = scene.project(xy);
    let a = scene.project([xy[0] + e, xy[1]]);
    let b = scene.project([xy[0], xy[1] + e]);
    let det = ((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])).abs() / (e * e);
    let s = det.sqrt();
    (
        2.0 * f64::from(target.ring().outer_radius_mm) * s,
        f64::from(target.min_center_spacing_mm()) * s,
    )
}

fn in_frame(p: [f64; 2], size: [f64; 2]) -> bool {
    p[0] >= 0.0 && p[1] >= 0.0 && p[0] <= size[0] - 1.0 && p[1] <= size[1] - 1.0
}

/// Ground-truth cells of `target` seen in `scene`, under `covered`.
fn truth_cells(
    scene: &Scene,
    target: &TargetLayout,
    hidden: &dyn Fn([f64; 2]) -> bool,
) -> Vec<GtCell> {
    let size = [
        f64::from(scene.image.width()),
        f64::from(scene.image.height()),
    ];
    let r_mm = f64::from(target.outer_draw_radius_mm());
    target
        .cells()
        .iter()
        .map(|c| {
            let xy = [f64::from(c.xy_mm[0]), f64::from(c.xy_mm[1])];
            let px = scene.project(xy);
            let (diam_px, spacing_px) = local_scale(scene, target, xy);
            let ok = |p: [f64; 2]| in_frame(p, size) && !hidden(p);
            let present = ok(px);
            let visible = present
                && (0..16).all(|k| {
                    let a = f64::from(k) * std::f64::consts::PI / 8.0;
                    ok(scene.project([xy[0] + r_mm * a.cos(), xy[1] + r_mm * a.sin()]))
                });
            GtCell {
                id: target.id_of(c.coord),
                coord: [c.coord.u, c.coord.v],
                xy_mm: xy,
                px,
                present,
                visible,
                diam_px,
                spacing_px,
            }
        })
        .collect()
}

/// Render a case: the image and its ground truth.
pub fn render_case(case: &Case, boards: &mut Boards) -> (GrayImage, Truth) {
    let target = case.board.layout();
    let mut scene = render(boards.get(case.board), &case.spec);
    let size = [
        f64::from(scene.image.width()),
        f64::from(scene.image.height()),
    ];
    let keep_r = 0.7 * case.diameter_px;
    let kept: Vec<[f64; 2]> = match &case.overlay {
        Overlay::Cells(cells) => cells
            .iter()
            .map(|&i| {
                let xy = target.cells()[i].xy_mm;
                scene.project([f64::from(xy[0]), f64::from(xy[1])])
            })
            .collect(),
        _ => Vec::new(),
    };
    let hide = |p: [f64; 2]| covered(&case.overlay, &kept, keep_r, size, p);
    let mut foreign = Vec::new();
    let other = match &case.overlay {
        Overlay::Foreign { board, spec, .. } => {
            let fscene = render(boards.get(*board), spec);
            let flayout = board.layout();
            for c in truth_cells(&fscene, &flayout, &|p| !hide(p)) {
                if c.present {
                    foreign.push((c.px, c.spacing_px));
                }
            }
            Some(fscene.image)
        }
        _ => None,
    };
    let (w, h) = scene.image.dimensions();
    let paper = Luma([case.spec.white.round().clamp(0.0, 255.0) as u8]);
    for y in 0..h {
        for x in 0..w {
            let p = [f64::from(x), f64::from(y)];
            if !hide(p) {
                continue;
            }
            let v = match (&case.overlay, &other) {
                (Overlay::Band { .. }, _) => Luma([60]),
                (_, Some(img)) => *img.get_pixel(x, y),
                _ => paper,
            };
            scene.image.put_pixel(x, y, v);
        }
    }
    let cells = truth_cells(&scene, &target, &hide);
    let truth = Truth {
        name: case.name.clone(),
        group: case.group.clone(),
        board: case.board.name().to_string(),
        coded: target.is_coded(),
        hex: matches!(target.lattice(), LatticeGeometry::Hex(_)),
        cells,
        foreign,
    };
    (scene.image, truth)
}

impl Truth {
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "group": self.group,
            "board": self.board,
            "coded": self.coded,
            "hex": self.hex,
            "cells": self.cells.iter().map(|c| json!({
                "id": c.id, "coord": c.coord, "xy_mm": c.xy_mm, "px": c.px,
                "present": c.present, "visible": c.visible,
                "diam_px": c.diam_px, "spacing_px": c.spacing_px,
            })).collect::<Vec<_>>(),
            "foreign": self.foreign.iter().map(|(p, s)| json!([p[0], p[1], s])).collect::<Vec<_>>(),
        })
    }

    pub fn from_json(v: &Value) -> Self {
        let f2 = |v: &Value| [0, 1].map(|i| v[i].as_f64().unwrap_or(f64::NAN));
        Self {
            name: v["name"].as_str().unwrap().to_string(),
            group: v["group"].as_str().unwrap().to_string(),
            board: v["board"].as_str().unwrap().to_string(),
            coded: v["coded"].as_bool().unwrap(),
            hex: v["hex"].as_bool().unwrap(),
            cells: v["cells"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| GtCell {
                    id: c["id"].as_u64().map(|i| i as usize),
                    coord: [
                        c["coord"][0].as_i64().unwrap() as i32,
                        c["coord"][1].as_i64().unwrap() as i32,
                    ],
                    xy_mm: f2(&c["xy_mm"]),
                    px: f2(&c["px"]),
                    present: c["present"].as_bool().unwrap(),
                    visible: c["visible"].as_bool().unwrap(),
                    diam_px: c["diam_px"].as_f64().unwrap_or(f64::NAN),
                    spacing_px: c["spacing_px"].as_f64().unwrap_or(f64::NAN),
                })
                .collect(),
            foreign: v["foreign"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| {
                    (
                        [f[0].as_f64().unwrap(), f[1].as_f64().unwrap()],
                        f[2].as_f64().unwrap(),
                    )
                })
                .collect(),
        }
    }
}

/// Score of one detection against its truth.
#[derive(Debug, Clone, Default)]
pub struct Score {
    pub n_visible: usize,
    pub n_reported: usize,
    /// Distinct visible cells reported with the right identity (coded: ID;
    /// plain: absolute position).
    pub n_correct: usize,
    pub n_wrong_id: usize,
    /// Of the wrong IDs: reported on a foreign ring.
    pub n_foreign: usize,
    pub n_wrong_pos: usize,
    pub n_label_err: usize,
    pub n_centre_tight: usize,
    pub n_centre_loose: usize,
    pub h_reported: bool,
    pub h_wrong: bool,
    pub h_median_err: f64,
    pub frame: String,
}

impl Score {
    pub fn recall(&self) -> f64 {
        self.n_correct as f64 / self.n_visible.max(1) as f64
    }

    /// Any output that would corrupt a calibration.
    pub fn n_wrong(&self) -> usize {
        self.n_wrong_id + self.n_wrong_pos + self.n_label_err + usize::from(self.h_wrong)
    }

    pub fn add(&mut self, o: &Score) {
        self.n_visible += o.n_visible;
        self.n_reported += o.n_reported;
        self.n_correct += o.n_correct;
        self.n_wrong_id += o.n_wrong_id;
        self.n_foreign += o.n_foreign;
        self.n_wrong_pos += o.n_wrong_pos;
        self.n_label_err += o.n_label_err;
        self.n_centre_tight += o.n_centre_tight;
        self.n_centre_loose += o.n_centre_loose;
        self.h_reported |= o.h_reported;
        self.h_wrong |= o.h_wrong;
    }
}

fn apply_h(h: &Value, p: [f64; 2]) -> Option<[f64; 2]> {
    let m = |r: usize, c: usize| h[r][c].as_f64();
    let x = m(0, 0)? * p[0] + m(0, 1)? * p[1] + m(0, 2)?;
    let y = m(1, 0)? * p[0] + m(1, 1)? * p[1] + m(1, 2)?;
    let w = m(2, 0)? * p[0] + m(2, 1)? * p[1] + m(2, 2)?;
    Some([x / w, y / w])
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Label errors of relative-frame grid coordinates: reported labels minus
/// those agreeing with the best (lattice symmetry, translation) map onto the
/// true coordinates of the matched rings.
fn relative_label_errors(pairs: &[([i32; 2], [i32; 2])], hex: bool) -> usize {
    let kind = if hex {
        projective_grid::LatticeKind::Hex
    } else {
        projective_grid::LatticeKind::Square
    };
    let mut best = 0usize;
    for g in kind.symmetry_transforms() {
        let mut tally: HashMap<[i32; 2], usize> = HashMap::new();
        for (rel, truth) in pairs {
            let r = g.apply(Coord::new(rel[0], rel[1]));
            *tally.entry([truth[0] - r.u, truth[1] - r.v]).or_default() += 1;
        }
        best = best.max(tally.values().copied().max().unwrap_or(0));
    }
    pairs.len() - best
}

/// Score a detection result given as the library's / CLI's JSON.
pub fn score(truth: &Truth, det: &Value) -> Score {
    let present: Vec<&GtCell> = truth.cells.iter().filter(|c| c.present).collect();
    let frame = det["board_frame"].as_str().unwrap_or("none").to_string();
    let absolute = frame == "absolute";
    let mut s = Score {
        n_visible: truth.cells.iter().filter(|c| c.visible).count(),
        frame,
        ..Score::default()
    };
    let mut correct_cells: std::collections::HashSet<[i32; 2]> = Default::default();
    let mut rel_pairs = Vec::new();
    let markers = det["detected_markers"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    s.n_reported = markers.len();
    for m in &markers {
        let center = [
            m["center"][0].as_f64().unwrap(),
            m["center"][1].as_f64().unwrap(),
        ];
        let id = m["id"].as_u64().map(|i| i as usize);
        let board_xy = m["board_xy_mm"]
            .as_array()
            .map(|a| [a[0].as_f64().unwrap(), a[1].as_f64().unwrap()]);
        let grid = m["grid_coord"]
            .as_array()
            .map(|a| [a[0].as_i64().unwrap() as i32, a[1].as_i64().unwrap() as i32]);
        let dist = |p: [f64; 2]| (p[0] - center[0]).hypot(p[1] - center[1]);
        let matched = present
            .iter()
            .min_by(|a, b| dist(a.px).total_cmp(&dist(b.px)))
            .filter(|c| dist(c.px) <= 0.5 * c.spacing_px);
        let on_foreign = truth.foreign.iter().any(|(p, sp)| dist(*p) <= 0.5 * sp);
        let Some(cell) = matched else {
            if id.is_some() || board_xy.is_some() {
                s.n_wrong_id += 1;
                s.n_foreign += usize::from(on_foreign);
            }
            continue;
        };
        let err = dist(cell.px);
        s.n_centre_tight += usize::from(err > CENTRE_TIGHT_PX);
        s.n_centre_loose +=
            usize::from(err > CENTRE_LOOSE_MIN_PX.max(CENTRE_LOOSE_FRACTION * cell.diam_px));
        let id_ok = id.is_none() || id == cell.id;
        s.n_wrong_id += usize::from(!id_ok);
        let pos_ok = board_xy
            .is_none_or(|xy| (xy[0] - cell.xy_mm[0]).hypot(xy[1] - cell.xy_mm[1]) < 0.01)
            && (!absolute || grid.is_none_or(|g| g == cell.coord));
        s.n_wrong_pos += usize::from(id_ok && !pos_ok);
        if !absolute && let Some(g) = grid {
            rel_pairs.push((g, cell.coord));
        }
        let identified = if truth.coded {
            id.is_some()
        } else {
            absolute && board_xy.is_some()
        };
        if id_ok && pos_ok && identified && cell.visible {
            correct_cells.insert(cell.coord);
        }
    }
    s.n_correct = correct_cells.len();
    s.n_label_err = relative_label_errors(&rel_pairs, truth.hex);
    let h = &det["homography"];
    if !h.is_null() {
        s.h_reported = true;
        if absolute {
            let errs: Vec<f64> = present
                .iter()
                .filter_map(|c| {
                    let p = apply_h(h, c.xy_mm)?;
                    Some((p[0] - c.px[0]).hypot(p[1] - c.px[1]))
                })
                .collect();
            let pitch = median(present.iter().map(|c| c.spacing_px).collect());
            s.h_median_err = median(errs);
            // A NaN median (no present cell projects) counts as wrong.
            s.h_wrong = s.h_median_err.is_nan() || s.h_median_err > H_FRACTION * pitch;
        }
    }
    s
}

/// One-line summary of a score.
pub fn summary(s: &Score) -> String {
    format!(
        "vis {:4} rep {:4} ok {:4} rec {:.3} | wrongID {:3} (foreign {:2}) pos {:3} label {:3} | ctr>1px {:3} ctr-loose {:3} | H {} {} med {:.1}",
        s.n_visible,
        s.n_reported,
        s.n_correct,
        s.recall(),
        s.n_wrong_id,
        s.n_foreign,
        s.n_wrong_pos,
        s.n_label_err,
        s.n_centre_tight,
        s.n_centre_loose,
        if s.h_reported { "yes" } else { "no " },
        if s.h_wrong { "WRONG" } else { "ok" },
        s.h_median_err,
    )
}
