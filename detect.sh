#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <split_index: 0..5> [viz]"
  echo "Examples:"
  echo "  $0 2"
  echo "  $0 2 viz"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

split_idx="$1"
if [[ ! "$split_idx" =~ ^[0-5]$ ]]; then
  echo "Error: split_index must be an integer from 0 to 5."
  usage
  exit 1
fi

run_viz=false
if [[ $# -eq 2 ]]; then
  viz_arg="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
  case "$viz_arg" in
    viz|--viz|1|true|yes)
      run_viz=true
      ;;
    noviz|--no-viz|0|false|no)
      run_viz=false
      ;;
    *)
      echo "Error: second argument must be one of: viz, --viz, 1, true, yes, noviz, --no-viz, 0, false, no"
      usage
      exit 1
      ;;
  esac
fi

suffix="$(printf "%02d" "$split_idx")"
image="data/target_3_split_${suffix}.png"
det_json="data/target_3_split_${suffix}_det.json"

cargo run -p ringgrid-cli -- detect \
  --image "$image" \
  --out "$det_json" \
  --circle-refine-method projective-center \
  --complete-require-perfect-decode \
  --no-global-filter --include-proposals \
  --config ./crates/ringgrid-cli/config_sample.json

if [[ "$run_viz" == "true" ]]; then
  .venv/bin/python tools/viz_detect.py --image "$image" --det_json "$det_json"
fi
