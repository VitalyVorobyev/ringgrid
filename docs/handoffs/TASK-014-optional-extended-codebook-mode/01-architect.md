# Architect Report - TASK-014-optional-extended-codebook-mode

- Task ID: `TASK-014-optional-extended-codebook-mode`
- Backlog ID: `ALGO-014`
- Role: `architect`
- Date: `2026-03-10`
- Status: `ready_for_implementer`

## Inputs Consulted
- `docs/backlog.md`
- `docs/templates/task-handoff-report.md`
- `.agents/skills/architect/SKILL.md`
- `/Users/vitalyvorobyev/.codex/skills/api-shaping/SKILL.md`
- `docs/handoffs/TASK-013-create-dedicated-tools-gen-target-py/01-architect.md`
- `docs/decisions/006-codebook-invariants.md`
- `tools/gen_codebook.py`
- `tools/gen_synth.py`
- `crates/ringgrid/src/marker/codebook.rs`
- `crates/ringgrid/src/marker/codec.rs`
- `crates/ringgrid/src/marker/decode.rs`
- `crates/ringgrid/src/detector/config.rs`
- `crates/ringgrid/src/pipeline/result.rs`
- `crates/ringgrid-cli/src/main.rs`
- `README.md`
- `book/src/marker-anatomy/coding-scheme.md`

## Summary
- `ALGO-014` is the next unstarted `P1` algorithm backlog item and has no existing workflow directory, so this work is mapped to `TASK-014-optional-extended-codebook-mode`.
- The current codebook implementation is single-profile and hard-wired around one embedded 893-word, 16-bit codebook across generation, matching, decode config, CLI helpers, and docs.
- The backlog requirement is additive rather than replacement-based: IDs `0..892` and the default decode path must remain unchanged, while a larger profile becomes available only through explicit opt-in.
- The safest implementation is a profile-aware codebook abstraction with baseline-compatible generated artifacts, a small `DecodeConfig` surface addition, and regression tests that lock the base profile before any extension IDs are exposed.

## Decisions Made
- Use `TASK-014-optional-extended-codebook-mode` because no previous architect handoff exists for backlog item `ALGO-014` and `TASK-014` is the next unused workflow id.
- Keep the shipped 893-word profile as the canonical default. The extension must be additive, not a regeneration that reassigns or reorders baseline IDs.
- Add the opt-in at the existing decode configuration seam:
  - introduce an explicit codebook profile selector
  - default it to the current baseline profile
  - route matching through profile metadata instead of one set of global `CODEBOOK_*` constants
- Keep `tools/codebook.json` backward-compatible for current synth tooling. If extra profile metadata is added, existing top-level baseline `codewords` access must keep working or all first-party consumers must be updated in the same change.
- Treat “max-feasible extension size” as the largest deterministic extension the committed generator can produce while holding the shipped 893-word prefix fixed and preserving the 16-bit rotational invariants. This task should not attempt a formal proof of the global coding-theory optimum.
- Keep scope tight:
  - in scope: generator/artifacts, decode plumbing, compatibility tests, and minimal docs/CLI surfacing
  - out of scope: board-layout redesign, larger target schemas, or a new detector output schema

## Files/Modules Affected (Or Expected)
- `tools/gen_codebook.py` - must generate a fixed baseline profile plus an additive extension profile and record per-profile metadata deterministically.
- `tools/codebook.json` - generated artifact; needs a baseline-compatible JSON contract while carrying any extension metadata required by tooling/docs.
- `crates/ringgrid/src/marker/codebook.rs` - generated Rust artifact; should expose baseline and extended slices/metadata without changing baseline IDs.
- `crates/ringgrid/src/marker/codec.rs` - matcher currently assumes one global codebook and one global minimum-distance constant.
- `crates/ringgrid/src/marker/decode.rs` - `DecodeConfig`, decode metrics docs, and matcher calls need to become profile-aware while preserving default behavior.
- `crates/ringgrid/src/detector/config.rs` - transitively affected because `DetectConfig` embeds `DecodeConfig` and must stay serde-compatible.
- `crates/ringgrid/src/lib.rs` - likely re-export point for any public `CodebookProfile` type.
- `crates/ringgrid-cli/src/main.rs` - `codebook-info` and `decode-test` are the minimal CLI touchpoints worth updating for profile visibility/manual smoke checks.
- `tools/gen_synth.py` - schema-sensitive consumer of `tools/codebook.json`; expected to remain unchanged only if the JSON stays backward-compatible.
- `docs/decisions/006-codebook-invariants.md` - needs to distinguish baseline invariants from extended-profile invariants.
- `README.md` - current public description hardcodes a single 893-codeword story.
- `book/src/marker-anatomy/coding-scheme.md` - currently describes only one embedded profile and one constant set.
- `book/src/detection-pipeline/decode.md` - currently documents decode against a single 893-word codebook.
- `book/src/cli-guide.md` - should reflect any `codebook-info` / `decode-test` profile surfacing.

## Validation / Tests
- Commands run:
  - none; architect planning only
- Results:
  - not run

## Risks / Open Questions
- “Max-feasible” is underspecified if interpreted as a mathematically proven maximum. The implementer should document the deterministic search procedure and the achieved extension size rather than overclaiming optimality.
- `tools/codebook.json` is consumed by first-party Python tools via `data["codewords"]`. A schema break here will spill beyond decode work unless compatibility is preserved deliberately.
- Many docs and comments currently hardcode `893` and `0..892`. Missing one of those surfaces would leave the repo in a confusing mixed state after the code change.

## Next Handoff
- To: `Implementer`
- Requested action: add a profile-aware codebook design that preserves the current 893-word baseline exactly, compute and commit an additive extended profile, thread explicit opt-in through `DecodeConfig`, and lock the behavior with compatibility/isolation tests plus the minimum docs/CLI updates needed to explain the new mode.

---

## Architect Required Sections

### Problem Statement
- The codebook subsystem is currently built around one generated artifact set:
  - `tools/gen_codebook.py` emits one flat JSON file and one flat Rust constant array
  - `codec.rs` and `decode.rs` assume one embedded codebook and one global `CODEBOOK_MIN_CYCLIC_DIST`
  - docs and CLI helpers describe a single 893-codeword profile
- `ALGO-014` requires a new capability without breaking the current detector contract:
  - baseline IDs `0..892` must remain unchanged
  - the default decode path must remain the current baseline mode
  - an additive extended profile must be available explicitly for future larger-ID use cases
- Without a profile abstraction, any attempt to “just regenerate a bigger codebook” would either silently renumber the shipped IDs or leak the new mode into existing users through defaults, serde, and docs.

### Scope
- In scope:
  - Add an explicit codebook profile concept with baseline and extended modes.
  - Regenerate codebook artifacts so the baseline 893-word prefix is preserved and the extension is additive.
  - Thread profile selection through the core matcher/decode configuration path.
  - Add compatibility and isolation tests for baseline identity and opt-in extended decoding.
  - Update the minimal docs/CLI surfaces that currently hardcode a single-profile story.
- Out of scope:
  - Changing marker bit length away from 16 bits.
  - Changing the default detector behavior or default config semantics.
  - Redesigning board layout ID assignment, target-generation schemas, or detection-result JSON.
  - Adding synth/print tooling flags for extended IDs unless required to preserve codebook JSON compatibility.
  - Proving a globally optimal maximum codebook size under all possible 16-bit constructions.

### Constraints
- Baseline compatibility is the top priority:
  - codewords `0..892` must remain byte-for-byte identical
  - `DecodeConfig::default()` must keep today’s behavior
  - older JSON config files that omit the new profile field must deserialize to the baseline profile
- Keep the public API diff small:
  - prefer one explicit enum on `DecodeConfig`
  - avoid global mutable state, alternate detector constructors, or duplicated config structs
- The confidence heuristic and decode docs must use profile-local metadata rather than a single global minimum-distance constant.
- `tools/codebook.json` must either remain backward-compatible for current baseline consumers or be accompanied by first-party consumer updates in the same task.
- The change must remain deterministic and fully committed:
  - generator output must be reproducible
  - no hand-edited generated Rust or JSON artifacts

### Assumptions
- `ALGO-014` is intentionally limited to decode/codebook infrastructure and does not require immediate board designs that exceed 893 IDs.
- The existing 893-word codebook remains the canonical shipped baseline profile for docs, examples, and default operation.
- A backward-compatible JSON contract is sufficient for current Python tooling because existing synth/eval flows only need the baseline codeword list today.
- The repo is willing to accept an achieved deterministic extension size documented by the generator, even if it is not presented as a formally proven maximum.

### Affected Areas
- `tools/gen_codebook.py` - generator logic and output schema contract.
- `tools/codebook.json` - generated baseline/extension metadata surface for tools.
- `crates/ringgrid/src/marker/codebook.rs` - generated constants and profile metadata.
- `crates/ringgrid/src/marker/codec.rs` - profile-aware matching and profile-local confidence denominator.
- `crates/ringgrid/src/marker/decode.rs` - config surface, docs, and decode-path wiring.
- `crates/ringgrid/src/detector/config.rs` - serde/default compatibility through embedded decode config.
- `crates/ringgrid/src/lib.rs` - public export of the new profile type.
- `crates/ringgrid-cli/src/main.rs` - profile visibility for helper commands.
- `docs/decisions/006-codebook-invariants.md` - baseline vs extended invariant documentation.
- `README.md` and book pages under `book/src/` that currently present the codebook as a single fixed 893-word mode.

### Plan
1. Regenerate the codebook artifacts around explicit profiles.
   - Update `tools/gen_codebook.py` so it treats the committed 893-word codebook as a fixed baseline prefix.
   - Compute an additive extension set under the same 16-bit rotational constraints, record its achieved size deterministically, and emit baseline plus extended metadata in both JSON and Rust artifacts.
   - Preserve backward compatibility for baseline JSON consumers, either by keeping top-level `codewords` as the baseline list or by updating all first-party loaders in the same patch.
2. Thread profile selection through core decode surfaces.
   - Introduce a small public `CodebookProfile` enum with `base` as the serde/default profile and `extended` as the explicit opt-in profile.
   - Make `Codebook` and decode logic resolve words, length, and minimum-distance metadata from the selected profile instead of one set of globals.
   - Keep the default detector path unchanged and avoid adding new top-level detector constructors or broad CLI flags.
   - Update `codebook-info` and, if useful for manual inspection, `decode-test` to expose profile information without changing default semantics.
3. Lock compatibility and document the new contract.
   - Add tests that prove baseline profile identity for IDs `0..892`, verify extended-only IDs decode only when the extended profile is selected, and confirm `DecodeConfig` serde defaults remain backward-compatible.
   - Refresh `DEC-006`, `README.md`, and the relevant book pages so the repo clearly distinguishes the baseline default from the explicit extension mode.
   - Keep the doc language precise about what is guaranteed:
     - stable baseline IDs
     - explicit opt-in extension
     - deterministic achieved extension size, not a proof of global optimality

### Acceptance Criteria
- The generated Rust and JSON codebook artifacts expose a baseline profile and an additive extended profile, with the baseline profile preserving the current 893 codewords exactly in the same order.
- `DecodeConfig` gains an explicit codebook-profile selector whose default is the baseline profile, and old configs without that field continue to deserialize to current behavior.
- Baseline decode behavior is unchanged:
  - exact baseline codewords still map to the same IDs `0..892`
  - default-mode decode never requires the user to opt out of new behavior
  - default profile metadata remains the shipped baseline metadata
- Extended-mode behavior is explicit:
  - extension IDs are appended after `892`
  - extension words are available only through the extended profile
  - profile-local metadata drives matching/confidence calculations
- Compatibility tests prove:
  - the baseline profile is identical to the pre-change codebook for IDs `0..892`
  - `DecodeConfig` serde remains backward-compatible when the profile field is absent
  - extended-profile-only words resolve to appended IDs only when the extended profile is selected
- Any change to `tools/codebook.json` either preserves existing `data["codewords"]` baseline access or updates all first-party consumers in the same patch with corresponding validation.
- Public docs and helper CLI output no longer imply that the repository supports only one embedded 893-codeword mode, while still stating clearly that baseline remains the default shipped profile.

### Test Plan
- Regenerate and validate the codebook artifacts as part of the implementation change.
- Run Rust quality gates:
  - `cargo fmt --all --check`
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  - `cargo test --workspace --all-features`
- Add targeted regression coverage for:
  - baseline profile identity
  - extended-profile decode of appended IDs
  - `DecodeConfig` serde default/backward compatibility
  - generated profile invariants in `codec.rs` / `decode.rs`
- Smoke-check the helper CLI surfaces after the change:
  - `cargo run -- codebook-info`
  - `cargo run -- decode-test 0x035D`
  - if `decode-test` becomes profile-aware, include one extended-profile sample word as a manual smoke run
- Rebuild docs if any public docs change:
  - `mdbook build book`
- If `tools/codebook.json` consumers are modified, add a Python smoke run of `tools/gen_synth.py` against the committed JSON artifact to confirm the loader contract still works.

### Out Of Scope
- New marker families with more than 16 sectors.
- A detector-result schema field that records the active codebook profile.
- Board-generation or board-layout changes that actually consume IDs above `892`.
- New Python/Rust user-facing commands beyond the minimum profile visibility needed for codebook inspection and manual decode smoke tests.
- Any attempt to re-optimize or renumber the baseline 893-word codebook.

### Handoff To Implementer
- Preserve the existing baseline codebook exactly. If any baseline word, ID, or default decode result changes, treat that as a regression and stop to fix it before broadening the patch.
- Add the new mode through the existing configuration surface:
  - `DecodeConfig` gets the profile selector
  - default stays baseline
  - serde for omitted profile field must remain backward-compatible
- Keep the generated-artifact story explicit and reviewable:
  - no hand edits to `tools/codebook.json`
  - no hand edits to `crates/ringgrid/src/marker/codebook.rs`
  - document the deterministic extension-count search instead of claiming a formal optimum
- Do not break `tools/gen_synth.py` accidentally. Either preserve baseline `data["codewords"]` access in `tools/codebook.json` or update and validate all first-party consumers in the same change.
- Prefer targeted compatibility tests over broad rewrites. The core of the task is proving that baseline behavior is unchanged while the extension remains explicitly opt-in.
