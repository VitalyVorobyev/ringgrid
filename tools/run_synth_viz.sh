#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run ringgrid detection + debug overlay on a synthetic dataset sample.

Usage:
  tools/run_synth_viz.sh <synth_dir> <index> [-- <extra viz_detect_debug.py args...>]

Examples:
  tools/run_synth_viz.sh tools/out/synth_001 0
  tools/run_synth_viz.sh tools/out/eval_run/synth 0 -- --id 42 --zoom 6 --show-edge-points
  tools/run_synth_viz.sh tools/out/synth_001 12 -- --stage stage1_fit_decode

Outputs (written next to the input image):
  det_XXXX.json          (normal DetectionResult)
  debug_XXXX.json        (ringgrid.debug.v8)
  det_overlay_XXXX.png   (overlay render)

Environment variables:
  RINGGRID_BIN   Override which ringgrid binary to run (e.g. target/release/ringgrid)
  SHOW=1         Show interactive matplotlib window (skip writing PNG)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

synth_dir="$1"
idx="$2"
shift 2

# Optional: pass through extra args to tools/viz_detect_debug.py (recommended with a `--` separator).
if [[ $# -gt 0 && "${1:-}" == "--" ]]; then
  shift
fi

printf -v idx4 "%04d" "$idx"

img="${synth_dir}/img_${idx4}.png"
det="${synth_dir}/det_${idx4}.json"
dbg="${synth_dir}/debug_${idx4}.json"
overlay="${synth_dir}/det_overlay_${idx4}.png"

if [[ ! -f "$img" ]]; then
  echo "ERROR: image not found: $img" >&2
  exit 1
fi

py=()
if [[ -n "${PYTHON:-}" ]]; then
  py=("$PYTHON")
elif [[ -x ".venv/bin/python" ]]; then
  py=(".venv/bin/python")
elif command -v python >/dev/null 2>&1; then
  py=("python")
else
  py=("python3")
fi

ringgrid_cmd=()
if [[ -n "${RINGGRID_BIN:-}" ]]; then
  ringgrid_cmd=("$RINGGRID_BIN")
elif [[ -x "target/debug/ringgrid" ]]; then
  ringgrid_cmd=("target/debug/ringgrid")
else
  ringgrid_cmd=("cargo" "run" "--quiet" "--")
fi

echo "[1/2] Detect: $img"
"${ringgrid_cmd[@]}" detect \
  --image "$img" \
  --out "$det" \
  --debug-json "$dbg"

echo "[2/2] Overlay: $overlay"
# Preflight: ensure matplotlib is available before we run the visualizer.
if ! "${py[@]}" -c 'import matplotlib; import numpy' >/dev/null 2>&1; then
  echo "ERROR: Python deps missing for visualization (need numpy + matplotlib)." >&2
  echo "Install in your current environment, e.g.:" >&2
  echo "  ${py[*]} -m pip install numpy matplotlib" >&2
  exit 1
fi

if [[ -n "${SHOW:-}" ]]; then
  echo "SHOW=1 set; opening interactive window (no output file)."
  "${py[@]}" tools/viz_detect_debug.py \
    --image "$img" \
    --debug_json "$dbg" \
    "$@"
else
  "${py[@]}" tools/viz_detect_debug.py \
    --image "$img" \
    --debug_json "$dbg" \
    --out "$overlay" \
    "$@"
fi

echo "Wrote:"
echo "  $det"
echo "  $dbg"
if [[ -z "${SHOW:-}" ]]; then
  echo "  $overlay"
fi
