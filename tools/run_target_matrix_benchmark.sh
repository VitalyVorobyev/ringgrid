#!/usr/bin/env bash
./.venv/bin/python tools/run_target_matrix_benchmark.py \
  --out_dir tools/out/target_matrix_benchmark \
  --n_images 3 --blur_px 0.8 --noise_sigma 0.0
