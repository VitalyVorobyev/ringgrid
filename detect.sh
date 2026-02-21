cargo run -p ringgrid-cli -- detect \
  --image data/target_3_split_01.png \
  --out data/target_3_split_01_det.json \
  --circle-refine-method projective-center \
  --complete-require-perfect-decode \
  --no-global-filter --include-proposals \
  --config data/config.json