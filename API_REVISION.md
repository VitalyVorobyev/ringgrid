# ringgrid API Revision Report

Date: 2026-05-18
Status: report only, no implementation
Audience: Claude Code implementation handoff

## Executive Summary

`ringgrid` already has the right public entry point shape: a `Detector`, a
validated `BoardLayout`, a config-driven `detect()` flow, serializable detection
results, and explicit proposal APIs. The API problem is not the absence of a
facade. The problem is that the facade currently exposes too much of the
implementation: stage-level tuning structs, low-level diagnostics, hidden raw
codec/codebook modules, experimental pipelines, and mutable data that can violate
internal invariants.

This should be treated as a breaking public API cleanup for `ringgrid 0.6.0`.
The crate is published as `ringgrid = "0.5.6"` on crates.io, so removals and
field privatization should be batched intentionally. The in-repo Python and WASM
bindings mirror much of the Rust surface, so they must be migrated in the same
series rather than after the fact.

Primary recommendation:

1. Keep `Detector`, `BoardLayout`, `DetectionResult`, `DetectedMarker`,
   proposal APIs, camera mapping, and target generation as the supported public
   workflow.
2. Split primary outputs from diagnostics. Keep normal detection results compact;
   move fit/decode/proposal/ransac/debug details into an explicit diagnostics
   channel.
3. Split stable detection options from advanced stage tuning. Stop making every
   internal threshold part of the durable API contract.
4. Remove or feature-gate `experimental` before it becomes accidental API.
5. Replace hidden raw `codec`/`codebook` module re-exports with small explicit
   APIs needed by the CLI and diagnostics.
6. Fix private/internal types that currently appear in public fields and method
   signatures.

## Audit Scope And Tooling

Reviewed:

- Rust public facade in `crates/ringgrid/src/lib.rs`
- Core entry points in `crates/ringgrid/src/api.rs`
- Detection/result/config types in `crates/ringgrid/src/detector/*` and
  `crates/ringgrid/src/pipeline/*`
- Proposal, marker, ring, homography, pixel mapping, target generation, and board
  layout modules
- `ringgrid-cli`, `ringgrid-py`, and `ringgrid-wasm` consumers
- Workspace decision records under `docs/decisions/`

Attempted:

```bash
cargo public-api -p ringgrid --color never
cargo public-api -p ringgrid --no-default-features --color never
cargo public-api -p ringgrid --all-features --color never
```

All three failed while rustdoc was processing `fixed 1.31.0`, due unstable const
function calls around `unchecked_shifts` and `unchecked_neg` on the current local
toolchain (`rustc 1.95.0`). Exact rustdoc public API inventory should be retried
after a tool/dependency workaround.

Fallback inventory:

- `crates/ringgrid/src` has 306 non-`pub(crate)`/non-`pub(super)` `pub`
  declarations by text scan.
- `crates/ringgrid/src` has 414 `pub` fields by text scan.
- These counts are not exact external API counts because some items live in
  private modules, but they are a useful signal: many internal structs are public
  inside the crate tree and several leak through root exports or public field
  types.

Package status observed:

- `ringgrid` is published on crates.io as `0.5.6`.
- `ringgrid-cli` was not found by `cargo search ringgrid-cli --limit 5`, but its
  `Cargo.toml` does not set `publish = false`.
- `ringgrid-py` and `ringgrid-wasm` both set `publish = false` in Cargo metadata,
  but they are important binding/package consumers.

## Intended Stable Contract

The supported Rust user story should stay small:

```rust
use ringgrid::{BoardLayout, Detector};

let board = BoardLayout::default();
let detector = Detector::new(board);
let result = detector.detect(&image);

for marker in result.detected_markers {
    println!("{:?} {:?}", marker.id, marker.center);
}
```

The intended public contract appears to be:

- `Detector` as the main algorithm facade.
- `DetectConfig` or successor as the top-level configuration object.
- `BoardLayout` and `BoardMarker` as target geometry.
- `DetectionResult` and `DetectedMarker` as serializable detection output.
- `Proposal`, `ProposalConfig`, `ProposalResult`, and proposal helpers as an
  intentionally supported proposal-only diagnostic workflow.
- `Ellipse`, marker scale types, and camera/pixel mapping types where needed for
  integration.
- Target generation options and errors for SVG/PNG target output.

Everything else should either be clearly advanced, diagnostics-only, hidden
behind an unstable feature, or private.

## Current Surface Classification

| API group | Current status | Classification | Recommended target |
| --- | --- | --- | --- |
| `Detector` and `Detector::{new, with_config, detect, detect_with_mapper}` | Root exported | Stable facade | Keep. Add only additive convenience APIs. |
| `Detector::config_mut()` | Root exported | Stable but broad mutation point | Keep only if config remains cohesive. Prefer replacing broad mutation with narrower builders in the cleanup. |
| `detect_adaptive`, `detect_adaptive_with_hint`, `detect_multiscale`, `adaptive_tiers` | Root exported methods | Advanced workflow | Keep if documented as supported, or consolidate behind mode/options in `0.6.0`. |
| Top-level `propose_*` helpers and `Detector::propose*` | Root exported | Supported proposal workflow | Keep, but document as proposal-only and not the full detector contract. |
| `Proposal`, `ProposalConfig`, `ProposalResult` | Root exported and `pub mod proposal` | Supported diagnostic/proposal API | Keep, but review config fields for `#[non_exhaustive]` and advanced stability. |
| `DetectionResult` | Root exported | Primary output plus diagnostics | Keep name, but split low-level diagnostics out of the primary result. |
| `DetectionResult::seed_proposals()` | Public method on primary output | Pipeline helper leak | Move to internal pipeline code or diagnostics helper. |
| `DetectedMarker` | Root exported | Primary output plus diagnostics | Keep core fields; move edge samples, detailed fit/decode internals, and stage source details to diagnostics. |
| `FitMetrics`, `DecodeMetrics`, `RansacStats` | Root exported | Diagnostics | Move behind explicit diagnostics output, or keep only compact stable quality summaries in primary output. |
| `DetectConfig` | Root exported | Stable config overloaded with internal tuning | Split into stable user config and advanced/stage config. |
| `InnerFitConfig`, `OuterFitConfig`, `SeedProposalParams`, `CompletionParams`, `IdCorrectionConfig`, `InnerAsOuterRecoveryConfig`, `ProjectiveCenterParams` | Root exported | Stage tuning | Move under advanced config or feature-gated diagnostics. |
| `EdgeSampleConfig`, `OuterEstimationConfig`, `DecodeConfig`, `RansacHomographyConfig` | Root exported | Stage tuning, partly reusable | Keep only if intentionally user-tunable. Otherwise move under advanced config. |
| `BoardLayout` | Root exported with public mutable fields | Stable target model with invariant risk | Make validated state private or introduce a builder/spec type. |
| `BoardMarker` | Root exported with public fields | Stable data record | Keep. Add `#[non_exhaustive]` only if future fields are likely. |
| Camera/pixel mapping types | Root exported | Stable integration API | Keep. Fix hidden type leak from `UndistortConfig`. |
| `PixelMapper` trait | Root exported | Stable extension point | Keep unsealed if custom mappings are intended; otherwise seal before `1.0`. |
| Target generation types | Root exported | Stable utility API | Keep, but review `std` feature behavior and `#[non_exhaustive]`. |
| `#[doc(hidden)] pub use marker::{codec, codebook}` | Hidden root exports | Raw implementation leak | Replace with explicit codebook info/decode API. |
| `#[doc(hidden)] pub mod experimental` | Hidden root module | Accidental unstable API | Remove from default public surface or gate under `unstable-experimental`. |

## API Smells To Fix

### 1. `#[doc(hidden)]` Is Being Used As A Stability Boundary

The root facade currently exposes:

```rust
#[doc(hidden)]
pub use marker::codebook;
#[doc(hidden)]
pub use marker::codec;

#[doc(hidden)]
pub mod experimental;
```

`#[doc(hidden)]` hides docs. It does not prevent downstream crates from compiling
against these items. Once published, these paths become real compatibility
liability.

Current known in-repo consumers:

- `ringgrid-cli` uses `ringgrid::codec::{Codebook, CodebookProfile, Match}` for
  codebook and decode diagnostic commands.
- Experimental examples and benches use `ringgrid::experimental::*`.

Recommended fix:

- Add a small intentional codebook diagnostics API, for example:

```rust
pub struct CodebookInfo {
    pub profile: CodebookProfile,
    pub len: usize,
    pub min_distance: u32,
}

pub struct CodewordMatch {
    pub id: Option<u16>,
    pub rotation: u8,
    pub distance: u32,
    pub confidence: f32,
}

pub fn codebook_info(profile: CodebookProfile) -> CodebookInfo;
pub fn decode_word(word: u16, profile: CodebookProfile) -> CodewordMatch;
```

- Update CLI diagnostic commands to use that API.
- Remove hidden raw `codec`/`codebook` exports in the breaking release.
- Move `experimental` behind a feature such as `unstable-experimental`, or move
  those examples/benches into in-crate tests/tools that do not require a public
  crate path.

### 2. Primary Results Contain Too Much Diagnostic State

`DetectedMarker` currently exposes core detection data and detailed internals:

- Core output: `id`, `confidence`, `center`, `center_mapped`, `board_xy_mm`
- Geometry details: `ellipse_outer`, `ellipse_inner`
- Debug samples: `edge_points_outer`, `edge_points_inner`
- Stage metrics: `fit`, `decode`, `source`

`DetectionResult` similarly exposes normal output and pipeline state:

- `detected_markers`, frames, image size
- optional homography and ransac diagnostics
- optional self-undistort result
- `seed_proposals()` helper for second-pass pipeline behavior

This makes every intermediate scoring detail part of the serialized API and
binding contract. It also makes it hard to improve the detector without either
retaining stale fields or breaking users.

Recommended target:

```rust
pub struct DetectionResult {
    pub markers: Vec<MarkerDetection>,
    pub center_frame: DetectionFrame,
    pub homography_frame: Option<DetectionFrame>,
    pub image_size: [u32; 2],
    pub homography: Option<HomographyEstimate>,
    pub self_undistort: Option<SelfUndistortSummary>,
}

pub struct MarkerDetection {
    pub id: Option<u16>,
    pub confidence: f32,
    pub center: [f32; 2],
    pub center_mapped: Option<[f32; 2]>,
    pub board_xy_mm: Option<[f32; 2]>,
    pub quality: MarkerQuality,
}

pub struct DetectionDiagnostics {
    pub proposal: Option<ProposalResult>,
    pub markers: Vec<MarkerDiagnostics>,
    pub ransac: Option<RansacStats>,
    pub self_undistort: Option<SelfUndistortResult>,
}

impl Detector {
    pub fn detect(&self, image: &GrayImage) -> DetectionResult;
    pub fn detect_with_diagnostics(&self, image: &GrayImage) -> (DetectionResult, DetectionDiagnostics);
}
```

The exact names can differ, but the contract should be explicit:

- Normal detection returns stable data needed by calibration consumers.
- Diagnostics return detailed algorithm internals for debugging, tuning, and
  scoring.

### 3. `DetectConfig` Is A God Config

`DetectConfig` currently mixes durable user choices with internal stage knobs:

- target geometry and marker scale
- proposal generation
- seed proposal behavior
- edge sampling
- decode options
- marker spec
- inner and outer fitting options
- circle refinement
- projective center options
- completion
- global filter toggles
- ransac homography
- self-undistort
- id correction
- inner-as-outer recovery
- confidence scaling
- proposal downscale

This violates the project note that defaults should have one source of truth and
that near-identical config layers should not be duplicated. It also makes every
algorithm threshold a public commitment.

Recommended target:

```rust
pub struct DetectConfig {
    pub board: BoardLayout,
    pub marker_scale: MarkerScalePrior,
    pub mode: DetectionMode,
    pub self_undistort: SelfUndistortConfig,
    pub advanced: AdvancedDetectConfig,
}

pub struct AdvancedDetectConfig {
    pub proposal: ProposalConfig,
    pub decode: DecodeConfig,
    pub fitting: FittingConfig,
    pub homography: RansacHomographyConfig,
    pub completion: CompletionConfig,
}
```

Implementation guidance:

- Preserve one source of truth for defaults.
- Do not add adapters between near-identical `Params` and `Config` structs.
- Move stable user choices to the top level.
- Put stage tuning under an explicitly advanced namespace.
- Use `#[non_exhaustive]` on config structs intended to grow without breaking
  downstream struct literals.
- Consider builders or constructor methods for configs with invariants.

### 4. Public Mutable `BoardLayout` Fields Can Break Invariants

`BoardLayout` exposes fields such as:

- `name`
- `pitch_mm`
- `rows`
- `long_row_cols`
- marker/ring dimensions

But it also has private derived state:

- `markers`
- `id_to_idx`

Direct field mutation can make the public scalar fields disagree with the private
marker cache. The Python binding already contains refresh logic for mutations,
which is a sign that the Rust type is leaking a construction invariant.

Recommended target:

- Make validated `BoardLayout` immutable from the outside.
- Provide accessors for scalar properties.
- Provide a `BoardLayoutBuilder` or `BoardSpec` for mutation and construction.
- Keep `BoardMarker` as a plain data record.
- If breaking field privacy is too large for `0.6.0`, add a clear migration path:
  constructors first, field deprecation second, field privacy in the breaking
  release.

### 5. Private Types Appear In Public Signatures Or Fields

Several root-exported public types contain fields whose types are not themselves
part of the root facade:

- `OuterFitConfig` and `InnerFitConfig` expose `crate::conic::RansacConfig`.
- `OuterEstimationConfig` exposes marker-level types such as `AngularAggregator`
  and `GradPolarity`.
- `CameraModel` exposes methods involving `UndistortConfig`, while `pixelmap`
  is private and `UndistortConfig` is not root re-exported.

This creates confusing API ergonomics even if the compiler accepts the current
shape: consumers see fields or methods involving types they cannot easily name
from the documented facade.

Recommended fix:

- If the type is stable and useful, re-export it intentionally at the root.
- If the type is implementation detail, hide the containing config from stable
  public API or wrap the field in a stable public abstraction.
- Add a public API lint/check after `cargo public-api` is working again.

### 6. Bindings Mirror Internal Rust Details

The Python and WASM bindings mirror `DetectConfig` heavily. That means Rust API
cleanup has to include binding cleanup.

Observed Python drift:

- Rust `FitMetrics` includes `radii_std_outer_px` and `h_reproj_err_px`; the
  Python dataclass mirror does not.
- Rust `DetectedMarker` includes `source`; the Python dataclass mirror does not.
- Python docs claim geometry and quality fields match Rust `DetectionResult`.

Recommended fix:

- Before the main breaking cleanup, add missing mirror fields or intentionally
  document the binding as a reduced schema.
- After the cleanup, generate or test binding schema parity more directly.
- Keep the Python/WASM config schema aligned to the new stable/advanced split.

## Feature And Package Boundaries

Current feature behavior:

- `ringgrid` default feature enables `std`, which includes PNG support and file
  convenience helpers.
- `ringgrid-wasm` depends on `ringgrid` with `default-features = false`.
- `all-features` currently does not provide a separate experimental boundary.

Recommended feature policy:

- `std`: file I/O conveniences and PNG target generation.
- `unstable-experimental`: contour/annulus/hybrid experiment APIs, examples,
  and benches if they must remain externally callable.
- No hidden public modules for experimental work in default builds.
- Keep WASM compiling against the stable no-default surface.

## Proposed 0.6.0 Migration Plan

### Phase 0: Report Approval

No code changes. Decide which public workflows must remain source-compatible
through the cleanup.

Questions for maintainers:

- Should proposal-only APIs be supported long term?
- Should `BoardLayout` remain directly mutable?
- Which `DetectedMarker` fields are needed by normal calibration consumers?
- Should target generation file helpers stay in the core crate under `std`?
- Should experimental contour/annulus APIs be publishable at all?
- How much Python backwards compatibility is required for config dumps?

### Phase 1: Add Explicit Replacement APIs

Additive changes only where possible:

- Add explicit codebook diagnostics APIs used by the CLI.
- Add explicit `DetectionDiagnostics` output path.
- Add a compact marker quality/result shape while keeping old fields temporarily.
- Add stable/advanced config struct shape or builders.
- Add missing Python mirror fields or document intentional differences.
- Add feature flag scaffolding for experimental code.

Verification:

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
.venv/bin/python -m pytest crates/ringgrid-py/tests -q
python3 tools/ci/maintainability_guardrails.py --check static
```

### Phase 2: Migrate In-Repo Consumers

Update internal consumers to prove the replacement API works:

- CLI codebook commands stop using `ringgrid::codec`.
- CLI detection commands use stable config APIs.
- Python and WASM bindings use the new config/result schema.
- Experimental examples/benches move behind `unstable-experimental` or out of
  the public crate path.
- README and user guide examples use only the intended facade.

### Phase 3: Remove Or Hide Leaked Surface

Breaking changes for `0.6.0`:

- Remove hidden raw `marker::codec` and `marker::codebook` root exports.
- Remove or feature-gate default `experimental` module exposure.
- Remove `DetectionResult::seed_proposals()` from the public primary result.
- Move detailed `FitMetrics`, `DecodeMetrics`, edge samples, and stage source
  state out of primary marker output if the diagnostics replacement is ready.
- Make `BoardLayout` invariant-bearing fields private if approved.
- Remove or relocate stage tuning structs from the root facade unless they are
  explicitly part of advanced public config.

### Phase 4: Harden The Facade

After the breaking cleanup:

- Add `#[non_exhaustive]` to public structs/enums that should grow without
  downstream breakage.
- Re-export or wrap every type used in public fields and method signatures.
- Add a public API snapshot check once `cargo public-api` works in this repo.
- Add binding schema parity tests for Rust/Python/WASM result/config shapes.
- Document the stable API tiers: primary, proposal, diagnostics, advanced,
  unstable.

## Suggested Handoff Tasks For Claude Code

Use small implementation slices. Do not combine all phases into one change.

1. Create explicit codebook diagnostics API and migrate `ringgrid-cli` off
   `ringgrid::codec`.
2. Introduce `DetectionDiagnostics` and a diagnostics-returning detector method,
   preserving old result fields temporarily.
3. Add `unstable-experimental` feature and gate `experimental` examples/benches.
4. Split stable and advanced detection config without duplicating defaults.
5. Migrate Python and WASM bindings to the new config/result schema.
6. Perform the `0.6.0` breaking removal pass after replacement APIs and in-repo
   consumers are green.
7. Add public API and binding schema checks.

## Residual Risks

- `cargo public-api` failed locally, so the exact rustdoc-visible item list is
  not included in this report.
- Some text-scan `pub` counts include items inside private modules; use them as
  scale signals, not as exact semver inventory.
- The current working tree already contains unrelated/uncommitted changes,
  including experimental source and examples. This report intentionally does not
  modify or revert them.
- Moving diagnostics out of serialized result structs will affect JSON consumers,
  Python consumers, and downstream evaluation tools. That needs explicit release
  notes and migration examples.

## Recommended Acceptance Criteria

For the eventual implementation:

- Normal Rust detection examples compile using only root facade items.
- CLI no longer imports hidden raw codec/codebook modules.
- Default builds do not expose experimental APIs unless explicitly feature-gated.
- Python and WASM config/result schemas match the intended Rust schema.
- `cargo public-api` or an equivalent snapshot check runs in CI.
- `cargo fmt --all`, clippy, Rust tests, Python binding tests, and static
  maintainability guardrails pass.
