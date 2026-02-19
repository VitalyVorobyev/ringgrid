cargo run -p ringgrid-cli -- detect \
  --image data/target_3_split_01.png \
  --out data/target_3_split_01_det.json \
  --marker-diameter-min 12 --marker-diameter-max 44 \
  --ransac-thresh-px 20.0 --ransac-iters 4000 \
  --complete-min-conf 0.55 --complete-reproj-gate 4.0 \
  --circle-refine-method projective-center \
  --complete-require-perfect-decode