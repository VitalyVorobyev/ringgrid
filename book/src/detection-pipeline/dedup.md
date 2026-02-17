# Deduplication

Multiple proposals can converge on the same physical marker, producing duplicate detections. The deduplication stage removes these duplicates while preserving the highest-quality detection for each marker.

## Two-Pass Deduplication

Deduplication operates in two passes:

### Pass 1: Spatial Dedup

Markers are sorted by confidence (descending). For each marker, any lower-confidence marker whose center is within `dedup_radius` pixels is suppressed. This is analogous to non-maximum suppression — only the strongest detection survives in each spatial neighborhood.

The `dedup_radius` is automatically derived from the `MarkerScalePrior` to approximately match the expected marker size.

### Pass 2: ID Dedup

After spatial dedup, markers that decoded to the same codebook ID are further deduplicated. When two or more markers share the same decoded ID, only the one with the highest confidence is retained.

This pass handles cases where spatially separated proposals happen to decode to the same codeword — which can occur with poor-quality fits at the image periphery.

## Ordering

The output of deduplication is a list of unique, high-confidence markers sorted by confidence. This ordering matters for downstream stages:

- The **global filter** uses decoded markers to build homography correspondences
- **Completion** attempts fits for missing IDs

Higher-confidence markers contribute more reliably to these stages.

**Source**: `detector/dedup.rs`
