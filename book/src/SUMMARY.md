# Summary

[Introduction](introduction.md)

# Marker Design

- [Ring Structure](marker-anatomy/ring-structure.md)
- [16-Sector Coding & Codebook](marker-anatomy/coding-scheme.md)
- [Hex Lattice Layout](marker-anatomy/hex-lattice.md)
- [Why Rings?](why-rings.md)

# Detection Pipeline

- [Pipeline Overview](detection-pipeline/overview.md)
- [Proposal Generation](detection-pipeline/proposal.md)
- [Outer Radius Estimation](detection-pipeline/outer-estimate.md)
- [Outer Ellipse Fit](detection-pipeline/outer-fit.md)
- [Code Decoding](detection-pipeline/decode.md)
- [Inner Ellipse Estimation](detection-pipeline/inner-estimate.md)
- [Deduplication](detection-pipeline/dedup.md)
- [Projective Center & Global Filter](detection-pipeline/projective-center.md)
- [ID Correction](detection-pipeline/id-correction.md)
- [Completion & Final Refit](detection-pipeline/completion.md)

# Mathematical Foundations

- [Fitzgibbon Ellipse Fitting](math/fitzgibbon-ellipse.md)
- [RANSAC Robust Estimation](math/ransac.md)
- [DLT Homography](math/dlt-homography.md)
- [Projective Center Recovery](math/projective-center-recovery.md)
- [Division Distortion Model](math/division-model.md)

# Using ringgrid

- [Configuration](configuration/detect-config.md)
  - [Marker Scale Prior](configuration/marker-scale-prior.md)
  - [Sub-Configurations](configuration/sub-configs.md)
- [Output Types](output-types/detection-result.md)
  - [DetectedMarker](output-types/detected-marker.md)
  - [Fit & Decode Metrics](output-types/fit-metrics.md)
  - [RansacStats](output-types/ransac-stats.md)
- [Detection Modes](detection-modes/simple.md)
  - [External PixelMapper](detection-modes/external-mapper.md)
  - [Self-Undistort Mode](detection-modes/self-undistort.md)
  - [Custom PixelMapper](detection-modes/custom-pixel-mapper.md)
- [Coordinate Frames](coordinate-frames.md)
- [Target Generation](target-generation.md)
- [CLI Guide](cli-guide.md)
