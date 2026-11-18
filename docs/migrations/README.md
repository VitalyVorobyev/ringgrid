# Migration notes

Pre-1.0 upgrade notes, kept out of the main user guide to keep it lean. `ringgrid`
is pre-1.0 and favors the cleaner design over source stability, so breaking
changes are documented here per release.

- [0.7 → 0.8](0.7-to-0.8.md) — compositional target model, public-API tiering, field renames.
- [0.8 → 0.9](0.8-to-0.9.md) — `BoardLayout` removal; v4 `board_spec.json` still auto-migrated (until 0.16).
- [0.10 → 0.11](0.10-to-0.11.md) — centered rect coordinates, derived origin-dot
  positions, target schema v6.
- [0.15 → 0.16](0.15-to-0.16.md) — v4/v5 target JSON is no longer loaded; dead and
  research-only items removed; confidence, `min_confidence` and fail-closed output.
- [0.16 → 0.17](0.16-to-0.17.md) — exact decoding against the target's codewords:
  decode gates, `codebook_profile` and the distance fields of `DecodeMetrics` removed.

Legacy v4 and v5 `board_spec.json` / `target_spec.json` files auto-migrated to
the v6 schema through 0.15. From 0.16 only `ringgrid.target.v6` loads; convert
older files first (see the 0.15 → 0.16 notes).
