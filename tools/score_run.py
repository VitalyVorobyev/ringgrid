#!/usr/bin/env python3
"""Score detection results against ground truth.

TODO Milestone 1:
- Load detection JSON and ground truth JSON.
- Match detected markers to ground truth by nearest center (Hungarian algorithm).
- Compute metrics:
  - Detection rate (recall).
  - False positive rate.
  - Center error: RMS and max (pixels).
  - Center error after removing center-bias (compare corrected vs uncorrected).
  - Semi-axis error.
  - ID decoding accuracy.
- Output summary table and per-image CSV.

Usage (planned):
    python score_run.py --gt data/synth/gt/ --det results/ --out scores.json
"""

import sys


def main():
    print("TODO: Milestone 1 — scoring harness not yet implemented.")
    sys.exit(0)


if __name__ == "__main__":
    main()
