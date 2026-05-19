//! Codebook inspection diagnostics.
//!
//! This module exposes a small, stable surface for inspecting the embedded
//! marker codebooks and for decoding a single 16-bit observed word against a
//! chosen [`CodebookProfile`]. It is intended for diagnostic tooling (such as
//! the `ringgrid` CLI's `codebook-info` and `decode-test` commands) — regular
//! detection should go through the high-level [`Detector`](crate::Detector).
//!
//! Both types are thin, read-only views over the internal [`Codebook`] type;
//! they carry no logic of their own beyond gathering already-computed values.

use super::codec::{Codebook, CodebookProfile};

/// Static, read-only summary of one embedded codebook profile.
///
/// Produced by [`codebook_info`]. All values are inherent properties of the
/// shipped codebook tables and never change at runtime.
#[derive(Debug, Clone)]
pub struct CodebookInfo {
    /// Embedded profile this summary describes.
    pub profile: CodebookProfile,
    /// Number of bits in each codeword.
    pub bits: usize,
    /// Total number of codewords in the profile.
    pub len: usize,
    /// Minimum cyclic Hamming distance claimed for the profile.
    pub min_cyclic_dist: usize,
    /// Generator seed recorded for the profile.
    pub seed: u64,
    /// First codeword (marker ID 0), or `None` if the profile is empty.
    pub first_codeword: Option<u16>,
    /// Last codeword (highest marker ID), or `None` if the profile is empty.
    pub last_codeword: Option<u16>,
}

/// Result of decoding a single observed 16-bit word against the codebook.
///
/// Produced by [`decode_word`]. Fields mirror the best match found across all
/// cyclic rotations of every codeword in the chosen [`CodebookProfile`].
#[derive(Debug, Clone)]
pub struct CodewordMatch {
    /// Embedded profile the word was matched against.
    pub profile: CodebookProfile,
    /// Observed word that was decoded.
    pub word: u16,
    /// Marker ID of the best-matching codeword (index into the codebook).
    pub id: usize,
    /// Codeword stored at the matched marker ID, or `None` if unavailable.
    pub codeword: Option<u16>,
    /// Cyclic rotation (in sectors, 0..15) that produced the best match.
    pub rotation: u8,
    /// Hamming distance to the best-matching codeword after rotation.
    pub dist: u8,
    /// Margin: second-best distance minus best distance.
    pub margin: u8,
    /// Confidence heuristic in `[0, 1]` combining distance and margin.
    pub confidence: f32,
}

/// Summarize an embedded codebook profile.
///
/// Returns a [`CodebookInfo`] describing the profile's bit width, codeword
/// count, claimed minimum cyclic Hamming distance, generator seed, and its
/// first/last codewords.
pub fn codebook_info(profile: CodebookProfile) -> CodebookInfo {
    let cb = Codebook::from_profile(profile);
    let (first_codeword, last_codeword) = if cb.is_empty() {
        (None, None)
    } else {
        (cb.word(0), cb.word(cb.len() - 1))
    };
    CodebookInfo {
        profile: cb.profile(),
        bits: cb.bits(),
        len: cb.len(),
        min_cyclic_dist: cb.min_cyclic_dist(),
        seed: cb.seed(),
        first_codeword,
        last_codeword,
    }
}

/// Decode a single observed 16-bit word against an embedded codebook profile.
///
/// Tries all 16 cyclic rotations of every codeword in `profile` and returns a
/// [`CodewordMatch`] describing the best match (marker ID, matched codeword,
/// rotation, Hamming distance, margin, and confidence).
pub fn decode_word(word: u16, profile: CodebookProfile) -> CodewordMatch {
    let cb = Codebook::from_profile(profile);
    let m = cb.match_word(word);
    CodewordMatch {
        profile: cb.profile(),
        word,
        id: m.id,
        codeword: cb.word(m.id),
        rotation: m.rotation,
        dist: m.dist,
        margin: m.margin,
        confidence: m.confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_info_reports_base_profile() {
        let info = codebook_info(CodebookProfile::Base);
        assert_eq!(info.profile, CodebookProfile::Base);
        assert_eq!(info.bits, 16);
        assert!(info.len > 0);
        assert!(info.first_codeword.is_some());
        assert!(info.last_codeword.is_some());
    }

    #[test]
    fn codebook_info_matches_codebook_methods() {
        for profile in [CodebookProfile::Base, CodebookProfile::Extended] {
            let info = codebook_info(profile);
            let cb = Codebook::from_profile(profile);
            assert_eq!(info.bits, cb.bits());
            assert_eq!(info.len, cb.len());
            assert_eq!(info.min_cyclic_dist, cb.min_cyclic_dist());
            assert_eq!(info.seed, cb.seed());
            assert_eq!(info.first_codeword, cb.word(0));
            assert_eq!(info.last_codeword, cb.word(cb.len() - 1));
        }
    }

    #[test]
    fn decode_word_exact_match_is_zero_distance() {
        let info = codebook_info(CodebookProfile::Base);
        let first = info.first_codeword.expect("non-empty codebook");
        let m = decode_word(first, CodebookProfile::Base);
        assert_eq!(m.id, 0);
        assert_eq!(m.dist, 0);
        assert_eq!(m.rotation, 0);
        assert_eq!(m.codeword, Some(first));
        assert_eq!(m.word, first);
        assert_eq!(m.profile, CodebookProfile::Base);
    }
}
