cargo run -p ringgrid-cli -- detect \
  --image data/target_3_split_05.png \
  --out data/target_3_split_05_det.json \
  --circle-refine-method projective-center \
  --complete-require-perfect-decode \
  --no-global-filter --include-proposals \
  --config ./crates/ringgrid-cli/config_sample.json