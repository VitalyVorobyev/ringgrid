# Print & Verify

Generating a target produces geometry on disk. Detection assumes that geometry
is what the camera actually sees. **Printing is the step that can silently break
that assumption** — a printer scaled the page to fit, the sheet is curled, or the
`target_spec.json` handed to the detector is not the one that was printed. None
of those raise an error; they show up as quietly wrong millimeters.

This page covers choosing a printable geometry, printing it at true scale,
verifying the print, and mounting it.

## 1. Size the target for your camera

Detection works on the marker's **outer ring diameter in pixels**. The default
[`MarkerScalePrior`](../configuration/marker-scale-prior.md) accepts markers
from **14 px to 66 px** across; outside that range you must widen the prior or
use [adaptive scale detection](../detection-modes/adaptive-scale.md).

For a marker at distance `Z` from a camera of focal length `f` (both in the same
unit — pixels for `f`, millimeters for `Z`):

```text
marker_diameter_px  ≈  2 · outer_radius_mm · f_px / Z_mm
```

Worked example — a 2000 px-focal-length camera at 800 mm, with the
`rect_plain_dots` recipe's `outer_radius_mm = 5.6`:

```text
2 · 5.6 · 2000 / 800  =  28 px       comfortably inside [14, 66]
```

Move to 2000 mm and the same marker images at 11 px — below the prior's floor.
Either increase `outer_radius_mm` (and `pitch_mm` with it, since markers must
not touch) or plan on a wider scale prior. Decide this **before** printing: the
whole point of a printed target is that its geometry is fixed.

## 2. Check the board fits your paper

`ringgrid gen` reports the physical size, because it is rarely what you expect:

```console
$ ringgrid gen rect_plain_dots.toml --out ./out/rect
wrote ./out/rect/target_spec.json
wrote ./out/rect/target_print.svg
wrote ./out/rect/target_print.png
wrote ./out/rect/target_print.dxf
target: rect_plain_dots — rect 24x24, pitch 14.0 mm, plain, 3 origin dots
markers: 576
print size: 343.2 x 343.2 mm (exceeds A3 297x420 mm — use a plotter or tile the print)
png: 4054 x 4054 px @ 300 dpi
print at 100% scale, then verify the printed scale bar with a ruler
```

The default 24×24 board is **343 mm square** — larger than A3. That is a normal
size for a calibration target, but it means a desktop printer is not an option.
Your choices are a plotter or print shop, a smaller board (fewer rows/cols, or a
smaller `pitch_mm`), or tiling — and tiling is the worst of the three, because
every seam is a place the board stops being flat and planar.

The page is always **square**: markers and dots are fitted into a square content
box, so one number describes both dimensions. From code, ask before you render:

```rust
# use ringgrid::TargetLayout;
let target = TargetLayout::rect_24x24();
let side_mm = target.print_side_mm(5.0);   // margin_mm, as passed to the writers
assert!((side_mm - 343.2).abs() < 1e-3);
```

To shrink a board, reduce `rows`/`cols` first. Reducing `pitch_mm` alone forces
`outer_radius_mm` down with it (markers may not overlap, and origin dots need a
gap to sit in), which pushes you back into the pixel-size problem above.

## 3. Print at 100 % scale

- **Prefer the SVG.** It is resolution-independent; the PNG is a convenience
  raster and is only as good as its `dpi`.
- **Disable every scaling option** — "fit to page", "shrink oversized pages",
  "scale to paper size", auto-rotate. These are on by default in many print
  dialogs and are the single most common cause of a wrong-scale target.
- **Print margin.** Set `margin_mm` in the recipe if your printer clips near the
  page edge; the markers must not be cropped, and neither must the dots.
- **Paper.** Matte, not glossy — specular highlights destroy ring contrast.
  Heavier stock curls less.

## 4. Verify the print with the scale bar

Every SVG and PNG carries a **scale bar** in the lower-left corner (unless you
set `scale_bar = false`): a black bar with white tick marks and its true length
printed on it, e.g. `100 mm`. The bar is sized from the board — up to 100 mm
long, with ticks every 10 mm on larger boards.

**Measure it with a ruler.** If the bar reads `100 mm` and measures 100 mm, the
print is true scale. If it measures 97 mm, the page was scaled to ~97 % and
every detected millimeter will be wrong by that factor.

If the scale is off, **re-print** — do not compensate by editing the spec.
Rescaling `pitch_mm` to match a bad print bakes a printer artifact into the
target definition, and the next print from the same spec will be wrong twice
over.

## 5. Mount it flat

The detector estimates a homography, which assumes the target is **planar**. A
sheet of paper is not planar unless you make it so:

- Mount on rigid, dimensionally stable backing — aluminium composite, glass, or
  thick acrylic. Foam board warps with humidity.
- Use spray adhesive over the whole surface, not tape at the corners; taped
  sheets bow in the middle.
- Do not fold, roll, or crease the sheet before mounting.
- Keep it clean and unmarked. Do not write on the target — including in the
  margins, which is where the scale bar lives.

## 6. Archive the spec that was printed

The `target_spec.json` is the detector's input, and nothing in the image ties it
back to the sheet on your desk. If you regenerate with a different `pitch_mm`,
radius, or dot placement, detection will run happily against the wrong geometry
and return plausible, wrong numbers.

- Keep `target_spec.json` next to the printed board and under version control.
- If you print more than one target, give each a distinct `name` in the recipe —
  it is carried in the spec and is your only provenance link.
- Re-printing from the same spec is safe. Re-generating is only safe if nothing
  in the recipe changed.

## 7. Detect against it

```bash
ringgrid detect \
  --target ./out/rect/target_spec.json \
  --image path/to/photo.png \
  --out ./out/rect/detect.json
```

For a plain target with origin dots, check `board_frame` in the result: it reads
`absolute` once the dots have been located, and `relative_canonical` if they
have not — see [Origin Fiducials](origin-fiducials.md). If it stays
`relative_canonical` on a board that *does* have dots, the usual causes are dots
outside the field of view, too few labeled markers (at least four are needed to
fit the anchoring homography), or insufficient contrast at the dots.

## Related Chapters

- [Target Generation](../target-generation.md) — recipes and every render option
- [Origin Fiducials](origin-fiducials.md) — how dots anchor the board
- [Marker Scale Prior](../configuration/marker-scale-prior.md) — the pixel-size window
- [Tutorial: Both Targets End to End](../tutorial-both-targets.md)
