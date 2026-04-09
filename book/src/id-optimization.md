# ID Assignment Optimization

By default, ringgrid boards assign codebook IDs to marker positions
sequentially: position 0 gets ID 0, position 1 gets ID 1, and so on. This
is simple but suboptimal: adjacent markers on the hex lattice may end up with
codewords that differ by only a few bits, making single-bit decode errors more
likely to produce a valid neighbor's ID.

The **ID assignment optimizer** reassigns codebook IDs to board positions so
that hex-adjacent markers have maximally dissimilar codewords, measured by
cyclic Hamming distance.

## Why it matters

The baseline codebook has a minimum cyclic Hamming distance of 2 across all
893 codewords. However, with sequential assignment, the minimum distance
between *adjacent markers on the board* is also 2 — meaning a single-bit
decode error can produce a codeword that matches a neighbor.

The ID correction stage (BFS hex-neighbor consensus) catches many of these
errors, but it works better when adjacent markers are far apart in code space.
Optimized assignment on the default 203-marker board raises the minimum
adjacent distance from 2 to 5:

| Metric | Sequential | Optimized (base) | Optimized (extended) |
|--------|-----------|-------------------|----------------------|
| Min adjacent distance | 2 | 5 | higher |
| Mean adjacent distance | 4.67 | 6.54 | higher |

## When to use it

**Recommended for all production boards.** There is no detection performance
cost — the detector reads IDs from the board spec JSON at runtime, and the
pipeline is unaffected by which IDs are assigned where. The optimization is a
one-time offline step that only changes the board spec file.

Especially valuable when:

- **High blur or low contrast** — decode errors are more frequent
- **Wide-angle or high-distortion setups** — markers near image edges suffer
  higher error rates
- **Dense boards** — more neighbors means more chances for ID confusion
- **No camera calibration available** — the ID correction stage is the primary
  defense against decode errors

## Tradeoffs

| Consideration | Impact |
|---------------|--------|
| Detection speed | None — IDs are loaded at startup, no runtime cost |
| Board spec size | Adds ~200 lines of `id_assignment` array to JSON |
| Backward compatibility | Full — omitting `id_assignment` gives sequential (existing boards work unchanged) |
| Codebook profile choice | Extended codebook gives better adjacency distances but lower codebook Hamming distance (1 vs 2) |
| Reproducibility | Optimizer uses a fixed seed (default 42) for deterministic results |

## How to use

### Option 1: Use a pre-optimized board

Two reference boards are included in the repository:

```bash
# Base codebook (893 codewords), min adjacent distance = 5
tools/board/board_spec_optimized.json

# Extended codebook (2180 codewords), even better adjacency
tools/board/board_spec_extended_opt.json
```

Use these directly with any detection interface:

```bash
# CLI
ringgrid detect --board tools/board/board_spec_optimized.json --image photo.png --out result.json

# Python
import ringgrid
det = ringgrid.Detector(board_spec_json=open("tools/board/board_spec_optimized.json").read())

# Rust
let board = BoardLayout::from_json_file(Path::new("tools/board/board_spec_optimized.json"))?;
let detector = Detector::new(board);
```

### Option 2: Optimize your own board

Generate a board spec first (see [Target Generation](target-generation.md)), then
optimize it:

```bash
# Optimize with base codebook (default)
.venv/bin/python tools/optimize_id_assignment.py \
    --board tools/board/your_board.json \
    --out tools/board/your_board_optimized.json

# Optimize with extended codebook for better adjacency
.venv/bin/python tools/optimize_id_assignment.py \
    --board tools/board/your_board.json \
    --profile extended \
    --iters 500000 \
    --out tools/board/your_board_ext_opt.json
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--board` | `tools/board/board_spec.json` | Input board spec |
| `--codebook` | `tools/codebook.json` | Codebook JSON |
| `--profile` | `base` | `base` (893) or `extended` (2180) |
| `--iters` | 200000 | Simulated annealing iterations |
| `--seed` | 42 | RNG seed for reproducibility |
| `--out` | auto-derived | Output path |

### Step 3: Verify the result

```bash
.venv/bin/python tools/board_adjacency_report.py --board tools/board/your_board_optimized.json
```

This prints min/max/mean/median adjacency distances and a histogram.

## Algorithm overview

The optimizer runs in two stages:

1. **Greedy initialization** (10 random restarts): BFS from the most-connected
   position, selecting at each step the unused codeword that maximizes the
   minimum cyclic Hamming distance to all already-assigned neighbors. The best
   of 10 restarts is kept.

2. **Simulated annealing** (200,000 iterations by default): Metropolis-acceptance
   random moves with exponential cooling. Move types depend on whether the
   codebook has spare capacity:
   - **Replace moves** (N < M): swap a position's ID with an unused codeword
   - **Swap moves** (N >= M): exchange IDs between two random positions

   The energy function is lexicographic: first minimize the negative of the
   minimum adjacent-pair distance, then minimize the negative sum of all
   adjacent-pair distances. This ensures the optimizer never trades a higher
   minimum distance for a better mean.

## Board spec format

The `id_assignment` field is an array of codebook IDs, one per marker in
generation order. It appears at the top level of the `ringgrid.target.v4`
JSON:

```json
{
  "schema": "ringgrid.target.v4",
  "name": "ringgrid_200mm_hex",
  "pitch_mm": 8.0,
  "rows": 15,
  "long_row_cols": 14,
  "marker_outer_radius_mm": 2.8,
  "marker_inner_radius_mm": 1.68,
  "marker_ring_width_mm": 0.56,
  "id_assignment": [447, 612, 201, 55, ...],
  "markers": [...]
}
```

When `id_assignment` is absent, IDs are assigned sequentially. When present,
`id_assignment[i]` is the codebook ID for the i-th marker in the `markers`
array. The array length must equal the number of markers, and all IDs must be
unique.
