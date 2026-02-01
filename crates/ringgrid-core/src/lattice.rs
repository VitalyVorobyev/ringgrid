//! Neighbor graph, vanishing line estimation, and affine rectification.
//!
//! ## Why affine rectification?
//!
//! Under perspective projection, the image of a circle is an ellipse whose
//! geometric center does NOT correspond to the projection of the circle's
//! 3D center — this is called "center bias" or "eccentricity error".
//!
//! For a regular grid of circular markers, we can estimate the vanishing line
//! (line at infinity mapped into the image) from two vanishing points derived
//! from the grid's row and column directions. An affine rectification
//! homography that maps this vanishing line back to infinity removes the
//! projective component, leaving only an affine transformation. In the
//! affine-rectified image, ellipse centers coincide with the projections
//! of circle centers, eliminating the center bias.
//!
//! After affine rectification, per-marker refinement (dual-ring LM fit) can
//! proceed without systematic bias.
//!
//! TODO Milestone 2:
//! - Nearest-neighbor graph from detected ellipse centers.
//! - Dominant direction extraction (two grid directions).
//! - Vanishing point estimation per direction.
//! - Vanishing line from two vanishing points.
//! - Affine rectification homography from vanishing line.
//! - Center correction: transform refined centers back to original image coords.

use crate::conic::Ellipse;

/// A node in the marker neighbor graph.
#[derive(Debug, Clone)]
pub struct MarkerNode {
    pub index: usize,
    pub ellipse: Ellipse,
    /// Indices of neighbor markers in the graph.
    pub neighbors: Vec<usize>,
}

/// Vanishing line represented as [a, b, c] where ax + by + c = 0.
#[derive(Debug, Clone, Copy)]
pub struct VanishingLine(pub [f64; 3]);

/// Build a nearest-neighbor graph from detected ellipses.
///
/// Stub — returns empty graph for now.
pub fn build_neighbor_graph(_ellipses: &[Ellipse], _max_neighbors: usize) -> Vec<MarkerNode> {
    // TODO Milestone 2: implement k-NN graph based on center distances
    Vec::new()
}

/// Estimate vanishing points and vanishing line from grid directions.
///
/// Stub — returns None for now.
pub fn estimate_vanishing_line(_graph: &[MarkerNode]) -> Option<VanishingLine> {
    // TODO Milestone 2: implement vanishing point estimation
    None
}

/// Compute the affine rectification homography from a vanishing line.
///
/// The homography H maps the vanishing line l = [a, b, c] back to the
/// line at infinity [0, 0, 1], removing the projective component.
///
/// Stub — returns identity for now.
pub fn affine_rectification_homography(_vanishing_line: &VanishingLine) -> nalgebra::Matrix3<f64> {
    // TODO Milestone 2: implement H = [[1,0,0],[0,1,0],[l1,l2,l3]]
    nalgebra::Matrix3::identity()
}

/// Correct ellipse center positions using affine rectification.
///
/// Stub — returns centers unchanged for now.
pub fn correct_centers(
    ellipses: &[Ellipse],
    _homography: &nalgebra::Matrix3<f64>,
) -> Vec<[f64; 2]> {
    // TODO Milestone 2: transform centers through H, fit in rectified space,
    // transform back
    ellipses.iter().map(|e| [e.cx, e.cy]).collect()
}
