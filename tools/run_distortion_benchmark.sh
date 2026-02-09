./.venv/bin/python tools/run_reference_benchmark.py \
  --out_dir tools/out/r4_benchmark_distorted_threeway_v4_post_pipeline \
  --n_images 3 --blur_px 0.8 --noise_sigma 0.0 --marker_diameter 32.0 \
  --cam-fx 900 --cam-fy 900 --cam-cx 640 --cam-cy 480 \
  --cam-k1 -0.15 --cam-k2 0.05 --cam-p1 0.001 --cam-p2 -0.001 --cam-k3 0.0 \
  --corrections none external self_undistort \
  --modes projective_center