# Introduction

**ringgrid** is a pure-Rust library for detecting dense coded ring calibration targets arranged on a hexagonal lattice. It detects markers with subpixel accuracy, decodes unique IDs from a 893-codeword codebook, estimates a board-to-image homography, and returns structured results ready for downstream camera calibration.

> *No OpenCV bindings — all image processing is implemented in Rust.*

## The Problem

Camera calibration requires detecting fiducial markers — known patterns printed on a calibration target — with high geometric precision. Traditional approaches use checkerboard corners or square markers (ArUco). These patterns have limitations:

- **Checkerboards** provide subpixel corner accuracy but carry no per-corner identity, making automatic correspondence ambiguous when the full board is not visible.
- **Square markers** (ArUco, AprilTag) encode identity in a binary grid, but their corners are detected via contour intersection, which limits subpixel precision.

ringgrid introduces a different target design: **concentric ring markers** with binary-coded sectors, arranged on a hex lattice.

## The Solution

Each ringgrid marker consists of two concentric rings — an outer ring and an inner ring — separated by a 16-sector binary code band that encodes a unique ID. This design provides three key advantages:

1. **Subpixel edge detection.** Ring boundaries produce strong, omnidirectional intensity gradients. The detector samples edge points along radial rays and fits an ellipse using the Fitzgibbon direct least-squares method, achieving center localization well below one pixel.

2. **Projective center correction.** Under perspective projection, the center of a fitted ellipse is *not* the true projected center of the circle. ringgrid fits both the outer and inner ring ellipses and uses their conic pencil to recover the unbiased projected center — without requiring camera intrinsics.

3. **Large identification capacity.** The 16-sector binary code band provides 893 unique codewords with guaranteed minimum Hamming distance, enabling rotation-invariant decoding with error tolerance.

## What You Get

The detector returns a `DetectionResult` containing:

- A list of `DetectedMarker` structs, each with:
  - Decoded ID (from the 893-codeword codebook)
  - Subpixel center in image coordinates
  - Fitted outer and inner ellipses
  - Quality metrics (fit residuals, decode confidence)
- A board-to-image homography (when enough markers are decoded)
- Coordinate frame metadata describing the output conventions

## Detection Modes

ringgrid supports three detection modes:

1. **Simple detection** — single-pass detection in image coordinates. No distortion correction.
2. **External pixel mapper** — two-pass detection using a user-provided coordinate mapping (e.g., camera distortion model). Pass-1 finds seed positions, pass-2 refines in the undistorted working frame.
3. **Self-undistort** — automatic estimation of a single-parameter division distortion model from the detected ellipses, followed by a corrected second pass. No external calibration required.

## Target Audience

This book is for Rust developers working on:

- Camera calibration pipelines
- Photogrammetry and 3D reconstruction
- Computer vision applications requiring high-precision fiducial detection
- Metrology and measurement systems

## Book Structure

- **Marker Design** — anatomy of the ring marker, coding scheme, and hex lattice layout
- **Detection Pipeline** — detailed walkthrough of all 13 detection stages
- **Mathematical Foundations** — full derivations of the core algorithms (ellipse fitting, RANSAC, homography, projective center recovery, division model)
- **Using ringgrid** — configuration, output types, detection modes, and CLI usage

<!-- TODO: Add a figure showing a ringgrid target and detection overlay side by side -->
