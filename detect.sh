cargo run -p ringgrid-cli -- detect \
  --image data/target_3_split_01.png \
  --out data/target_3_split_01_det.json \
  --marker-diameter-min 20 --marker-diameter-max 60 \
  --ransac-thresh-px 20.0 --ransac-iters 4000 \
  --complete-min-conf 0.55 --complete-reproj-gate 4.0 \
  --circle-refine-method projective-center \
  --complete-require-perfect-decode \
  --max-angular-gap-deg 60 \
  --no-global-filter --include-proposals