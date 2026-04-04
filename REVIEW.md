# Pre-Release Review — ringgrid 0.5.3
*Reviewed: 2026-04-04*
*Scope: full workspace*

## Executive Summary

The workspace is in good shape for release. The major change — replacing the
internal proposal stage with radsym RSD — is functionally correct and performance
is at parity with the old code on realistic images (5.5% faster on the fixture
benchmark). All 4 regression benchmarks pass.

Key issues to fix before release: MSRV mismatch in ringgrid-py, incomplete
CHANGELOG (missing radsym 0.1.3 + Scharr switch), and a missing performance
note in the changelog. No P0 blockers found.

## Findings

### R01 ringgrid-py MSRV mismatch
- **Severity**: P1
- **Category**: workspace
- **Location**: `crates/ringgrid-py/Cargo.toml:8`
- **Status**: todo
- **Problem**: ringgrid-py declares `rust-version = "1.87"` while workspace declares `"1.88"`
- **Fix**: Update to `rust-version = "1.88"`

### R02 CHANGELOG incomplete for 0.5.3
- **Severity**: P1
- **Category**: docs
- **Location**: `CHANGELOG.md:23`
- **Status**: todo
- **Problem**: Missing: radsym 0.1.3 upgrade, Scharr gradient switch, fused voting mode, performance parity result. Version still says 0.5.2.
- **Fix**: Bump version to 0.5.3, add performance notes to changelog

### R03 CLAUDE.md proposal module description outdated
- **Severity**: P2
- **Category**: docs
- **Location**: `.claude/CLAUDE.md:23-28`
- **Status**: todo
- **Problem**: Module layout shows old proposal submodules (gradient.rs, voting.rs, nms.rs) that were deleted
- **Fix**: Update to show current mod.rs + config.rs + tests.rs structure

### R04 radsym version bound is loose
- **Severity**: P2
- **Category**: workspace
- **Location**: `Cargo.toml:31`
- **Status**: todo
- **Problem**: `radsym = "0.1"` allows any 0.1.x but we require 0.1.3 features (scharr_gradient, rsd_response_fused)
- **Fix**: Change to `radsym = "0.1.3"`

### R05 cargo doc zero warnings
- **Severity**: P2
- **Category**: contracts
- **Location**: workspace
- **Status**: verified (0 warnings)
- **Problem**: N/A — passes clean

## Out-of-Scope Pointers
- Long functions in finalize.rs (481 lines), decode.rs (426 lines), completion.rs (256 lines) → existing tech debt, not introduced in this release. Track via `algo-review` skill.
- Python binding config serialization duplication → `rust-python-bindings` skill.

## Strong Points
- Zero `unsafe` blocks across the entire workspace
- Zero `TODO`/`FIXME` comments — code is complete
- Comprehensive documentation on all public items
- Clean `cargo doc`, `cargo fmt`, `cargo clippy`
- Well-structured public API with clear re-exports
- Generated codebook is current and documented
- 4-benchmark regression gate catches real issues
