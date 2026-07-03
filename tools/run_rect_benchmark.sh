#!/usr/bin/env bash
./.venv/bin/python tools/run_rect_benchmark.py \
  --out_dir tools/out/rect_benchmark \
  --n_images 3 --blur_px 0.8 --noise_sigma 0.0 --marker_diameter 26.0
