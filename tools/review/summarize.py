#!/usr/bin/env python3
"""Print a compact summary of a ringgrid review feedback.json (stdlib only).

    python tools/review/summarize.py [feedback.json | review-output-dir]

Lists verdicts, flag counts per run and kind, every flagged marker (tile, run,
reported id, centre, correct id), missed-marker and note pins, and all remarks.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "ringgrid.review.v1"


def load(path: Path) -> dict:
    if path.is_dir():
        path = path / "feedback.json"
    fb = json.loads(path.read_text())
    if fb.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: expected schema {SCHEMA!r}, got {fb.get('schema')!r}")
    return fb


def render(fb: dict) -> str:
    out: list[str] = []
    tiles = fb.get("tiles", {})
    out.append(f"dataset {fb.get('dataset')}   updated {fb.get('updated_at')}   tiles with feedback: {len(tiles)}")
    verdicts = Counter((t.get("verdict") or "unset") for t in tiles.values())
    out.append("verdicts: " + (", ".join(f"{k}={v}" for k, v in sorted(verdicts.items())) or "none"))

    per_run: dict[str, Counter] = defaultdict(Counter)
    for t in tiles.values():
        for f in t.get("marker_flags", []):
            per_run[f["run"]][f["kind"]] += 1
    out.append("\nflags per run:")
    for run, c in sorted(per_run.items()):
        out.append(f"  {run}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    if not per_run:
        out.append("  (none)")

    out.append("\nflagged markers (tile  run  kind  reported id -> correct id  centre  remark):")
    n = 0
    for name, t in sorted(tiles.items()):
        for f in t.get("marker_flags", []):
            mk = f["marker"]
            cid = f.get("correct_id")
            idtxt = f"id {mk.get('id')}" + (f" -> {cid}" if cid is not None else "")
            x, y = mk["center"]
            rem = f"  \"{f['remark']}\"" if f.get("remark") else ""
            out.append(f"  {name}  {f['run']}  {f['kind']}  {idtxt}  ({x:.1f}, {y:.1f})  #{mk['index']}{rem}")
            n += 1
    if not n:
        out.append("  (none)")

    out.append("\nmissed markers and notes (image coordinates):")
    n = 0
    for name, t in sorted(tiles.items()):
        for p in t.get("pins", []):
            if p["type"] == "missed_marker":
                tid = f" true id {p['true_id']}" if p.get("true_id") is not None else ""
                rem = f"  \"{p['remark']}\"" if p.get("remark") else ""
                out.append(f"  {name}  MISSED ({p['x']:.1f}, {p['y']:.1f}){tid}{rem}")
            else:
                out.append(f"  {name}  NOTE   ({p['x']:.1f}, {p['y']:.1f})  \"{p.get('text', '')}\"")
            n += 1
    if not n:
        out.append("  (none)")

    out.append("\ntile verdicts and remarks:")
    n = 0
    for name, t in sorted(tiles.items()):
        if t.get("verdict") or t.get("remark"):
            seen = ",".join(t.get("runs_seen", [])) or "-"
            rem = f"  \"{t['remark']}\"" if t.get("remark") else ""
            out.append(f"  {name}  {t.get('verdict') or 'unset'}  [runs seen: {seen}]{rem}")
            n += 1
    if not n:
        out.append("  (none)")

    out.append("\ngeneral remarks:")
    for r in fb.get("general_remarks", []):
        out.append(f"  {r['at']}  {r['text']}")
    if not fb.get("general_remarks"):
        out.append("  (none)")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="feedback.json or the review output directory containing it")
    args = ap.parse_args(argv)
    print(render(load(args.path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
