"""Read ringgrid target specs (`ringgrid.target.v6`) for the Python tools.

`load_target_spec` accepts any v6 target (the coverage scorer handles every
lattice and coding). `optimize_id_assignment.py` and `board_adjacency_report.py`
only handle the hex lattice with 16-sector coded rings; `load_hex_coded_spec`
rejects anything else with a clear message instead of a `KeyError` deep in the
lattice code.
"""

from __future__ import annotations

import json
from pathlib import Path

TARGET_SCHEMA = "ringgrid.target.v6"


def load_target_spec(path: str | Path) -> dict:
    """Load a v6 target JSON (any lattice, any coding)."""
    with open(path) as f:
        spec = json.load(f)
    schema = spec.get("schema")
    if schema != TARGET_SCHEMA:
        raise SystemExit(
            f"{path}: unsupported target schema {schema!r}; expected {TARGET_SCHEMA!r} "
            "(convert an older file by loading it with ringgrid 0.15 and re-saving)"
        )
    return spec


def load_hex_coded_spec(path: str | Path) -> dict:
    """Load a v6 target JSON and require a hex lattice with coded16 markers."""
    spec = load_target_spec(path)
    lattice = spec["lattice"]
    coding = spec["coding"]
    if lattice.get("kind") != "hex" or coding.get("kind") != "coded16":
        raise SystemExit(
            f"{path}: this tool needs a hex lattice with coded16 markers, "
            f"got lattice={lattice.get('kind')!r} coding={coding.get('kind')!r}"
        )
    return spec
