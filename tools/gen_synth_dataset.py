#!/usr/bin/env python3
"""Generate synthetic ring-marker calibration target images with ground truth.

TODO Milestone 1:
- Render a grid of concentric ring markers on a plane.
- Apply a projective homography (simulating camera perspective).
- Apply spatially-varying Gaussian blur (simulating Scheimpflug defocus).
- Add Poisson/Gaussian noise.
- Apply vignetting / uneven illumination.
- Output images + ground truth JSON (marker centers, IDs, ellipse params).

Usage (planned):
    python gen_synth_dataset.py --out-dir data/synth --num-images 100

Ground truth JSON schema (per image):
{
    "image_file": "img_0001.png",
    "image_size": [width, height],
    "markers": [
        {
            "id": 0,
            "center_world": [X, Y, Z],
            "center_image": [x, y],
            "outer_ellipse": {"cx": ..., "cy": ..., "a": ..., "b": ..., "angle": ...},
            "inner_ellipse": {"cx": ..., "cy": ..., "a": ..., "b": ..., "angle": ...},
            "visible": true
        }
    ],
    "camera": {
        "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        "R": [...],
        "t": [...],
        "scheimpflug_angle_deg": 8.0,
        "blur_map": "blur_0001.png"
    }
}
"""

import sys


def main():
    print("TODO: Milestone 1 — synthetic dataset generation not yet implemented.")
    print("See docstring for planned schema and usage.")
    sys.exit(0)


if __name__ == "__main__":
    main()
