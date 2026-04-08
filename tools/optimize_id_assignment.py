#!/usr/bin/env python3
"""Optimize marker ID-to-position assignment for local non-similarity.

Assigns codeword IDs to hex-lattice board positions such that hex-adjacent
markers have maximally dissimilar codewords (highest minimum cyclic Hamming
distance). Uses greedy initialization followed by simulated annealing.

Works for any board size N and codebook size M:
  - N < M: select best N codewords from M (greedy + SA with replace moves)
  - N = M: pure permutation optimization (SA with swap moves only)

Usage:
    python tools/optimize_id_assignment.py
    python tools/optimize_id_assignment.py --profile extended --iters 500000
    python tools/optimize_id_assignment.py --board tools/board/board_spec.json --out optimized.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BITS = 16
MASK = (1 << BITS) - 1


def popcount(x: int) -> int:
    return bin(x).count("1")


def rotate_left(word: int, k: int) -> int:
    k = k % BITS
    return ((word << k) | (word >> (BITS - k))) & MASK


def cyclic_hamming(a: int, b: int) -> int:
    return min(popcount(a ^ rotate_left(b, k)) for k in range(BITS))


def compute_distance_matrix(codewords: list[int]) -> np.ndarray:
    """Compute M x M symmetric cyclic Hamming distance matrix."""
    m = len(codewords)
    words = np.array(codewords, dtype=np.uint32)
    dist = np.full((m, m), BITS, dtype=np.int8)

    for k in range(BITS):
        # Rotate all words by k
        rotated = ((words.astype(np.uint32) << k) | (words.astype(np.uint32) >> (BITS - k))) & MASK
        # XOR each pair and count bits
        xor = words[:, None] ^ rotated[None, :]
        # Vectorized popcount via bit manipulation
        hd = np.zeros_like(xor)
        x = xor.copy()
        while np.any(x > 0):
            hd += x & 1
            x >>= 1
        dist = np.minimum(dist, hd.astype(np.int8))

    return dist


def generate_hex_positions(rows: int, long_row_cols: int) -> list[tuple[int, int]]:
    """Generate (q, r) hex axial coordinates matching board_layout.rs."""
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


def build_adjacency(positions: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Return adjacency list: pos_idx -> list of neighbor pos_idxs."""
    pos_set = {pos: idx for idx, pos in enumerate(positions)}
    adj: dict[int, list[int]] = defaultdict(list)
    hex_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    for idx, (q, r) in enumerate(positions):
        for dq, dr in hex_dirs:
            j = pos_set.get((q + dq, r + dr))
            if j is not None:
                adj[idx].append(j)
    return adj


def min_neighbor_dist(pos: int, assignment: list[int], adj: dict[int, list[int]], dist_mat: np.ndarray) -> int:
    """Minimum cyclic Hamming distance from pos to its assigned neighbors."""
    cw = assignment[pos]
    min_d = BITS
    for nb in adj[pos]:
        d = int(dist_mat[cw, assignment[nb]])
        if d < min_d:
            min_d = d
    return min_d


def global_min_adj_dist(assignment: list[int], adj: dict[int, list[int]], dist_mat: np.ndarray) -> int:
    """Global minimum adjacent-pair distance."""
    min_d = BITS
    for i, neighbors in adj.items():
        ci = assignment[i]
        for j in neighbors:
            if j > i:
                d = int(dist_mat[ci, assignment[j]])
                if d < min_d:
                    min_d = d
    return min_d


def sum_adj_dist(assignment: list[int], adj: dict[int, list[int]], dist_mat: np.ndarray) -> int:
    """Sum of all adjacent-pair distances (for tie-breaking)."""
    total = 0
    for i, neighbors in adj.items():
        ci = assignment[i]
        for j in neighbors:
            if j > i:
                total += int(dist_mat[ci, assignment[j]])
    return total


def greedy_init(
    n_positions: int,
    n_codewords: int,
    adj: dict[int, list[int]],
    dist_mat: np.ndarray,
    rng: random.Random,
) -> list[int]:
    """Greedy BFS assignment: assign each position the codeword maximizing min distance to neighbors."""
    assignment = [-1] * n_positions
    used: set[int] = set()

    # Start from the most connected position (most neighbors)
    start = max(range(n_positions), key=lambda i: len(adj.get(i, [])))

    # BFS order
    queue = [start]
    visited = {start}
    order = []
    while queue:
        pos = queue.pop(0)
        order.append(pos)
        for nb in adj.get(pos, []):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    # Add any disconnected positions
    for i in range(n_positions):
        if i not in visited:
            order.append(i)

    # Assign first position randomly
    first_id = rng.randrange(n_codewords)
    assignment[order[0]] = first_id
    used.add(first_id)

    # Assign remaining greedily
    for pos in order[1:]:
        assigned_neighbors = [assignment[nb] for nb in adj.get(pos, []) if assignment[nb] >= 0]

        best_id = -1
        best_min_d = -1
        best_sum_d = -1

        # Sample candidates (all if feasible, random subset if huge)
        candidates = list(range(n_codewords))
        if n_codewords > n_positions * 3:
            # Subsample for speed, but always include unused
            unused = [c for c in range(n_codewords) if c not in used]
            if len(unused) > 2000:
                candidates = rng.sample(unused, 2000)
            else:
                candidates = unused

        for cid in candidates:
            if cid in used:
                continue
            if not assigned_neighbors:
                best_id = cid
                used.add(cid)
                break
            min_d = min(int(dist_mat[cid, nb_cid]) for nb_cid in assigned_neighbors)
            sum_d = sum(int(dist_mat[cid, nb_cid]) for nb_cid in assigned_neighbors)
            if min_d > best_min_d or (min_d == best_min_d and sum_d > best_sum_d):
                best_min_d = min_d
                best_sum_d = sum_d
                best_id = cid

        if best_id < 0:
            # All codewords used (N >= M), pick the best from all
            for cid in range(n_codewords):
                if assigned_neighbors:
                    min_d = min(int(dist_mat[cid, nb_cid]) for nb_cid in assigned_neighbors)
                else:
                    min_d = BITS
                if min_d > best_min_d:
                    best_min_d = min_d
                    best_id = cid

        assignment[pos] = best_id
        used.add(best_id)

    return assignment


def simulated_annealing(
    assignment: list[int],
    n_codewords: int,
    adj: dict[int, list[int]],
    dist_mat: np.ndarray,
    n_iters: int,
    rng: random.Random,
) -> list[int]:
    """Simulated annealing to improve assignment."""
    n = len(assignment)
    has_surplus = n_codewords > n
    used = set(assignment)
    unused = [c for c in range(n_codewords) if c not in used] if has_surplus else []

    cur_min = global_min_adj_dist(assignment, adj, dist_mat)
    cur_sum = sum_adj_dist(assignment, adj, dist_mat)
    best_assignment = assignment[:]
    best_min = cur_min
    best_sum = cur_sum

    t_start = 2.0
    t_end = 0.01

    for step in range(n_iters):
        t = t_start * (t_end / t_start) ** (step / max(n_iters - 1, 1))

        if has_surplus and unused and rng.random() < 0.5:
            # Replace move: swap one position's ID with an unused one
            pos = rng.randrange(n)
            old_id = assignment[pos]
            new_idx = rng.randrange(len(unused))
            new_id = unused[new_idx]

            assignment[pos] = new_id
            new_min = global_min_adj_dist(assignment, adj, dist_mat)
            new_sum = sum_adj_dist(assignment, adj, dist_mat)

            if _accept(cur_min, cur_sum, new_min, new_sum, t, rng):
                used.discard(old_id)
                used.add(new_id)
                unused[new_idx] = old_id
                cur_min = new_min
                cur_sum = new_sum
            else:
                assignment[pos] = old_id
        else:
            # Swap move: exchange IDs of two positions
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue

            assignment[i], assignment[j] = assignment[j], assignment[i]
            new_min = global_min_adj_dist(assignment, adj, dist_mat)
            new_sum = sum_adj_dist(assignment, adj, dist_mat)

            if _accept(cur_min, cur_sum, new_min, new_sum, t, rng):
                cur_min = new_min
                cur_sum = new_sum
            else:
                assignment[i], assignment[j] = assignment[j], assignment[i]

        if cur_min > best_min or (cur_min == best_min and cur_sum > best_sum):
            best_assignment = assignment[:]
            best_min = cur_min
            best_sum = cur_sum

        if step % 50000 == 0 and step > 0:
            print(f"  SA step {step}/{n_iters}: current min_dist={cur_min}, best min_dist={best_min}", file=sys.stderr)

    return best_assignment


def _accept(cur_min: int, cur_sum: int, new_min: int, new_sum: int, t: float, rng: random.Random) -> bool:
    """Metropolis acceptance with lexicographic energy (min_dist, sum_dist)."""
    if new_min > cur_min:
        return True
    if new_min < cur_min:
        delta = cur_min - new_min
        return rng.random() < math.exp(-delta / t)
    # Same min, compare sum
    if new_sum > cur_sum:
        return True
    if new_sum < cur_sum:
        delta = (cur_sum - new_sum) / max(cur_sum, 1)
        return rng.random() < math.exp(-delta / t)
    return False


def load_codebook(path: Path, profile: str) -> list[int]:
    with open(path) as f:
        data = json.load(f)
    if profile == "extended":
        words_hex = data["profiles"]["extended"]["codewords"]
    else:
        words_hex = data["codewords"]
    return [int(s, 16) for s in words_hex]


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize marker ID assignment for local non-similarity")
    parser.add_argument("--board", type=str, default="tools/board/board_spec.json")
    parser.add_argument("--codebook", type=str, default="tools/codebook.json")
    parser.add_argument("--profile", choices=["base", "extended"], default="base")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--iters", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.board) as f:
        spec = json.load(f)

    rows = spec["rows"]
    long_row_cols = spec["long_row_cols"]
    positions = generate_hex_positions(rows, long_row_cols)
    n_positions = len(positions)

    codewords = load_codebook(Path(args.codebook), args.profile)
    n_codewords = len(codewords)

    if n_positions > n_codewords:
        print(f"ERROR: board has {n_positions} positions but codebook ({args.profile}) has only {n_codewords} codewords", file=sys.stderr)
        sys.exit(1)

    adj = build_adjacency(positions)
    n_edges = sum(len(nbs) for nbs in adj.values()) // 2

    print(f"Board: {n_positions} positions, {n_edges} adjacent pairs", file=sys.stderr)
    print(f"Codebook: {args.profile} ({n_codewords} codewords)", file=sys.stderr)
    print(f"Computing {n_codewords}x{n_codewords} distance matrix...", file=sys.stderr)

    dist_mat = compute_distance_matrix(codewords)
    print(f"Distance matrix computed.", file=sys.stderr)

    rng = random.Random(args.seed)

    # Multi-restart greedy
    n_restarts = 10
    best_greedy = None
    best_greedy_min = -1
    for restart in range(n_restarts):
        r = random.Random(args.seed + restart * 7919)
        assignment = greedy_init(n_positions, n_codewords, adj, dist_mat, r)
        gmin = global_min_adj_dist(assignment, adj, dist_mat)
        if gmin > best_greedy_min:
            best_greedy_min = gmin
            best_greedy = assignment
    print(f"Greedy init ({n_restarts} restarts): min_adj_dist = {best_greedy_min}", file=sys.stderr)

    # Simulated annealing
    print(f"Running SA ({args.iters} iterations)...", file=sys.stderr)
    optimized = simulated_annealing(best_greedy, n_codewords, adj, dist_mat, args.iters, rng)
    opt_min = global_min_adj_dist(optimized, adj, dist_mat)
    opt_sum = sum_adj_dist(optimized, adj, dist_mat)
    print(f"SA result: min_adj_dist = {opt_min}, sum_adj_dist = {opt_sum}", file=sys.stderr)

    # Write output
    spec["id_assignment"] = optimized
    out_path = args.out or args.board.replace(".json", "_optimized.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")
    print(f"Optimized board spec written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
