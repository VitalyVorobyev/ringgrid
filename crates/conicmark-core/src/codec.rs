//! Marker ID decoding from ring sector patterns.
//!
//! Each marker encodes a unique ID in a pattern of bright/dark sectors
//! between the inner and outer rings. The decoding process:
//!
//! 1. Sample pixel intensities along an elliptical arc between inner and
//!    outer ring boundaries.
//! 2. Threshold into binary sectors.
//! 3. Determine rotation-invariant canonical form (minimum cyclic rotation).
//! 4. Look up in codebook or decode directly.
//!
//! TODO Milestone 4:
//! - Define the encoding scheme (number of sectors, error correction).
//! - Implement intensity sampling along elliptical arcs.
//! - Binary thresholding with adaptive threshold.
//! - Cyclic rotation canonicalization.
//! - Codebook generation and lookup.

/// Decode a marker ID from the image around a detected marker.
///
/// Stub — returns None (no ID decoded).
pub fn decode_marker_id(
    _image: &image::GrayImage,
    _outer_ellipse: &crate::conic::Ellipse,
    _inner_ratio: f64,
) -> Option<u32> {
    // TODO Milestone 4: implement sector sampling and decoding
    None
}
