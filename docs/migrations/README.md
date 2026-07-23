# Migration notes

Pre-1.0 upgrade notes, kept out of the main user guide to keep it lean. `ringgrid`
is pre-1.0 and favors the cleaner design over source stability, so breaking
changes are documented here per release.

- [0.7 → 0.8](0.7-to-0.8.md) — compositional target model, public-API tiering, field renames.
- [0.8 → 0.9](0.8-to-0.9.md) — `BoardLayout` removal; v4 `board_spec.json` still auto-migrates.
- [0.10 → 0.11](0.10-to-0.11.md) — centered rect coordinates, derived origin-dot
  positions, target schema v6.

Legacy `board_spec.json` files continue to load in all versions —
`TargetLayout::from_json_*` auto-migrates the v4 and v5 schemas to the canonical
v6 schema.
