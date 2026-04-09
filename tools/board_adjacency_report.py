#!/usr/bin/env python3
"""Report cyclic Hamming distance statistics for hex-adjacent marker pairs.

Computes the adjacency graph of a hex-lattice board and reports the distribution
of cyclic Hamming distances between codewords assigned to neighboring markers.

Usage:
    python tools/board_adjacency_report.py
    python tools/board_adjacency_report.py --board tools/board/board_spec.json
    python tools/board_adjacency_report.py --board tools/board/board_spec_optimized.json --profile extended
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

BITS = 16
MASK = (1 << BITS) - 1


def popcount(x: int) -> int:
    return bin(x).count("1")


def rotate_left(word: int, k: int) -> int:
    k = k % BITS
    return ((word << k) | (word >> (BITS - k))) & MASK


def cyclic_hamming(a: int, b: int) -> int:
    return min(popcount(a ^ rotate_left(b, k)) for k in range(BITS))


def generate_hex_positions(rows: int, long_row_cols: int) -> list[tuple[int, int]]:
    """Generate (q, r) hex axial coordinates matching board_layout.rs logic."""
    short_row_cols = long_row_cols - 1
    mid = rows // 2
    positions = []
    for row_idx in range(rows):
        r = row_idx - mid
        if rows == 1 or ((r + long_row_cols - 1) & 1) == 0:
            n_cols = long_row_cols
        else:
            n_cols = short_row_cols
        q_start = -((r + n_cols - 1) // 2)
        for col_idx in range(n_cols):
            q = q_start + col_idx
            positions.append((q, r))
    return positions


def hex_neighbors(q: int, r: int) -> list[tuple[int, int]]:
    """Six hex neighbors in axial coordinates."""
    return [
        (q + 1, r), (q - 1, r),
        (q, r + 1), (q, r - 1),
        (q + 1, r - 1), (q - 1, r + 1),
    ]


def build_adjacency(positions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return list of (i, j) pairs where i < j and positions are hex-adjacent."""
    pos_set = {pos: idx for idx, pos in enumerate(positions)}
    edges = []
    for idx, (q, r) in enumerate(positions):
        for nq, nr in hex_neighbors(q, r):
            j = pos_set.get((nq, nr))
            if j is not None and idx < j:
                edges.append((idx, j))
    return edges


def load_codebook(path: Path, profile: str) -> list[int]:
    with open(path) as f:
        data = json.load(f)
    if profile == "extended":
        words_hex = data["profiles"]["extended"]["codewords"]
    else:
        words_hex = data["codewords"]
    return [int(s, 16) for s in words_hex]


def main() -> None:
    parser = argparse.ArgumentParser(description="Board adjacency distance report")
    parser.add_argument("--board", type=str, default="tools/board/board_spec.json")
    parser.add_argument("--codebook", type=str, default="tools/codebook.json")
    parser.add_argument("--profile", choices=["base", "extended"], default="base")
    args = parser.parse_args()

    with open(args.board) as f:
        spec = json.load(f)

    rows = spec["rows"]
    long_row_cols = spec["long_row_cols"]
    positions = generate_hex_positions(rows, long_row_cols)
    n_markers = len(positions)

    codewords = load_codebook(Path(args.codebook), args.profile)
    n_codewords = len(codewords)

    # Determine assignment
    id_assignment = spec.get("id_assignment")
    if id_assignment is not None:
        if len(id_assignment) != n_markers:
            print(f"ERROR: id_assignment length {len(id_assignment)} != {n_markers} markers", file=sys.stderr)
            sys.exit(1)
        assignment = id_assignment
        assignment_type = "optimized"
    else:
        assignment = list(range(n_markers))
        assignment_type = "sequential"

    if max(assignment) >= n_codewords:
        print(f"ERROR: assignment contains ID {max(assignment)} but codebook has only {n_codewords} entries", file=sys.stderr)
        sys.exit(1)

    edges = build_adjacency(positions)

    # Compute distances for all adjacent pairs
    distances = []
    for i, j in edges:
        d = cyclic_hamming(codewords[assignment[i]], codewords[assignment[j]])
        distances.append(d)

    if not distances:
        print("No adjacent pairs found.")
        return

    min_d = min(distances)
    max_d = max(distances)
    mean_d = sum(distances) / len(distances)
    sorted_d = sorted(distances)
    median_d = sorted_d[len(sorted_d) // 2]

    hist = Counter(distances)

    print(f"Board: {spec.get('name', '?')}, {rows} rows x {long_row_cols} cols, {n_markers} markers")
    print(f"Profile: {args.profile} ({n_codewords} codewords)")
    print(f"Assignment: {assignment_type}")
    print(f"Adjacent pairs: {len(edges)}")
    print()
    print(f"  Min distance:    {min_d}")
    print(f"  Max distance:    {max_d}")
    print(f"  Mean distance:   {mean_d:.2f}")
    print(f"  Median distance: {median_d}")
    print()
    print("  Distance histogram:")
    for d in range(min_d, max_d + 1):
        count = hist.get(d, 0)
        bar = "#" * min(count, 60)
        print(f"    {d:2d}: {count:4d}  {bar}")


if __name__ == "__main__":
    main()
