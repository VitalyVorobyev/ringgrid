#!/usr/bin/env python3
"""Check benchmark outputs against tools/ci/regression_baseline.json.

The regression gate's Step 3 as a script. Each baseline entry names the
benchmark's ``output`` JSON and its thresholds, either at the top level
(``criteria``) or per ``modes`` / ``strategies`` entry, which are looked up
under the same key in the output. A threshold ``min_<metric>`` requires
``value >= bound`` and ``max_<metric>`` requires ``value <= bound``, where the
value is the output's ``<metric>`` or, failing that, ``avg_<metric>``. A
missing or null value fails. Entries with a ``requires`` path that does not
exist (private datasets) are skipped.

Run the benchmarks first; this only reads their outputs.

Usage:
    .venv/bin/python tools/ci/check_regression_baseline.py
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("modes", "strategies")


def metric_value(actual: dict, metric: str):
    if metric in actual:
        return actual[metric]
    return actual.get(f"avg_{metric}")


def check_thresholds(benchmark, group, actual, thresholds):
    rows = []
    for key, bound in thresholds.items():
        if key.startswith("min_"):
            metric, op = key[4:], ">="
        elif key.startswith("max_"):
            metric, op = key[4:], "<="
        else:
            continue
        value = metric_value(actual, metric)
        if value is None:
            ok = False
        elif op == ">=":
            ok = value >= bound
        else:
            ok = value <= bound
        rows.append((benchmark, group, metric, f"{op}{bound}", value, ok))
    return rows


def check_entry(name, entry):
    """Rows (benchmark, group, metric, bound, actual, ok) for one baseline entry."""
    requires = entry.get("requires")
    if requires and not (ROOT / requires).exists():
        return None
    output = ROOT / entry["output"]
    if not output.exists():
        return [(name, "-", "output", "exists", f"missing {entry['output']}", False)]
    with open(output) as f:
        data = json.load(f)

    rows = []
    if "criteria" in entry:
        rows += check_thresholds(name, "-", data, entry["criteria"])
    for group in GROUPS:
        for mode, thresholds in entry.get(group, {}).items():
            actual = data.get(group, {}).get(mode)
            if actual is None:
                rows.append((name, mode, "mode", "present", "missing", False))
                continue
            rows += check_thresholds(name, mode, actual, thresholds)
    return rows


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--baseline", type=Path, default=ROOT / "tools/ci/regression_baseline.json"
    )
    ap.add_argument(
        "--verbose", action="store_true", help="Print passing checks too."
    )
    args = ap.parse_args()

    with open(args.baseline) as f:
        baseline = json.load(f)

    n_checks, failures, skipped = 0, [], []
    for name, entry in baseline.items():
        if not isinstance(entry, dict) or "output" not in entry:
            continue
        rows = check_entry(name, entry)
        if rows is None:
            skipped.append(f"{name} (requires {entry['requires']})")
            continue
        n_checks += len(rows)
        for row in rows:
            if not row[-1]:
                failures.append(row)
            if args.verbose or not row[-1]:
                status = "OK" if row[-1] else "FAIL"
                print(
                    f"{status:4}  {row[0]:28} {row[1]:26} {row[2]:28} "
                    f"{row[3]:>10}  actual {fmt(row[4])}"
                )

    for s in skipped:
        print(f"SKIPPED  {s}")
    print(f"{n_checks - len(failures)}/{n_checks} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
