# Displayed target: display rasterisation, code-band width and blur

Simulation study on top of the grid-first pipeline (0.15.0 + phases 1–4).
Every number below comes from an extended version of
`crates/ringgrid/tests/displayed_target.rs` (decode/ID tallies, grid snapping,
display grey depth and dither, band-width × blur tables, scene export). That
extended harness was not merged; only this write-up was kept. The commands
under *Reproduce* describe that harness, and most of the tests they name are
not in the tree.

## Question

The project's real images show the coded target on an e-reader display, not
on paper. In those images, 22–30 % of raw decoded IDs contradict the marker's
lattice cell. This study asks two things:

- **A.** How much of the decode failure and of the centre error comes from the
  display (pixel grid, grey quantisation)? How much from the thin code band?
  How much from blur and noise?
- **B.** What display resolution and what target geometry make a displayed
  target as good as a printed one? What is the best configuration on the
  existing device?

## Setup

**Board.** `TargetLayout::default_hex()`: hex lattice, pitch 8 mm
(nearest-neighbour distance √3·8 = 13.86 mm, row spacing 12 mm), ring
centreline radii 4.8 / 3.2 mm, ring stroke 1.152 mm. The renderer
(`target_generation.rs`, `render_target_png`) draws these edges:

| | radius / width, mm | at 166 dpi (0.1530 mm px) |
|---|---|---|
| outer edge (drawn outer radius) | 5.376 | 35.1 px |
| outer stroke | 4.224–5.376 (1.152) | 7.5 px |
| code band | 3.776–4.224 (0.448) | 2.9 px |
| inner stroke | 2.624–3.776 (1.152) | 7.5 px |
| drawn outer diameter | 10.752 | 70.3 px |

The detector's marker diameter is the drawn outer diameter, 10.752 mm, not
the 9.6 mm centreline diameter. The decoder samples the band at
0.744 × the outer edge × {0.9, 1.0, 1.1}, i.e. at 3.60 / 4.00 / 4.40 mm. Two of
those three radii fall on the black strokes.

**Device operating point.** 166 dpi, 0.1530 mm per display pixel, with the
target shown at its physical size. The real images show markers 20–56 camera
px across. At that scale a display pixel is 0.29–0.80 camera px and the code
band is 0.83–2.33 camera px wide:

| camera px per marker Ø | camera px per display px | band, camera px | stroke, camera px |
|---|---|---|---|
| 20 | 0.285 | 0.83 | 2.14 |
| 32 | 0.455 | 1.33 | 3.43 |
| 56 | 0.797 | 2.33 | 6.00 |

**Imaging.** The ideal board is rendered at 1200 dpi.

- **Display rasterisation.** The board is rasterised onto a display grid
  (`tests/support/blur_scene.rs`, `DisplaySpec`) in one of these modes:
  - `binary`: white when the ideal pattern at the pixel centre is white;
  - `greyN`: area coverage (8×8 taps per pixel) rounded to N levels;
  - `ditherN`: area coverage reduced to N levels with a 4×4 Bayer ordered
    dither. This is an assumption about what a reader may do; it is not
    known.

  The grid has a phase (default 0.37 px), a rotation, and optional per-marker
  snapping of marker centres to pixel centres.
- **Camera.** The display pixels are imaged as constant squares through a
  pinhole homography with supersampling (at least 4 samples per display-pixel
  side, never below 10×10 per camera pixel). After that come a Gaussian PSF,
  radiance with contrast 80 grey levels and a ±15 % illumination gradient,
  Gaussian noise σ = 2, and 8-bit quantisation.

**Frames and pooling.** Each cell pools 3 board poses (roll 7°, 41°, −23°, with
small offsets) × tilts 0° and 20° × 4 noise seeds. That is 24 frames and about
1630 visible markers per cell, so rates have roughly ±1 percentage point (pp)
of sampling error.

**Detector.** `DetectConfig::from_target_and_scale_prior` with a diameter prior
of nominal ±30 %. A fixed-size prior loses markers on tilted views, see
[Finding A3](#a3-where-the-undecoded-markers-went-the-scale-prior).

**Metrics per cell.**
- *found*: a result marker lies within 0.4 spacing of the true centre.
- *completion-only*: found only by the completion stage.
- *decoded*: an accepted decode from the marker's own code band, on a
  fit-decoded ring. Completion keeps only decodes that agree with the cell,
  so completed rings count as undecoded.
- *wrong among decodes*: the decoded ID names another marker.
- *correct ID* / *wrong final IDs*: the ID in the final result.
- *photometric RMS* / *projective RMS*: centre error against the true centre,
  camera px.
- *marker-to-marker*: the part of the photometric error that is repeatable per
  marker across seeds and poses (board frame, noise contribution removed),
  minus its board-wide common part. A common part is a board translation and
  does not hurt calibration. Shown in camera px and in µm on the board.

## Results

### Existing study: photometric centre error on a displayed target

`displayed_target_centre_error` uses a wider-band coded hex variant (band
1.15 mm) at 34 camera px per marker diameter. Its display pitches of 0.4 and
0.6 camera px bracket the device's 0.48 (blur 0.8, tilt 0°):

| raster | projective RMS | photometric RMS | per-marker repeatable |
|---|---|---|---|
| print | 0.117 px | 0.007 px | 0.003 px |
| grey16, 0.4 / 0.6 cam px per display px | 0.121 / 0.124 px | 0.007 / 0.007 px | 0.002 / 0.002 px |
| binary, 0.4 / 0.6 cam px per display px | 0.120 / 0.131 px | 0.010 / 0.019 px | 0.008 / 0.018 px (2.5 / 5.7 µm) |

At this operating point the rasterisation floor is 0 for 16 grey levels and
0.008–0.018 camera px (2.5–5.7 µm) for a binary display. That is 1–3 times
the photometric noise. It is invisible next to the projective centre's own
0.12 px error.

### Display resolution (default board, blur 0.8)

Display px per drawn marker diameter; the device is 70.3.

| cell | found % | decoded % | wrong among decodes % | correct ID % | wrong final IDs | photometric RMS px | marker-to-marker px | marker-to-marker µm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cam 20 print | 99.4 | 90.8 | 12.3 | 99.4 | 0 | 0.012 | 0.002 | 1.2 |
| cam 20 disp 20 binary | 99.0 | 71.3 | 45.9 | 99.0 | 0 | 0.048 | 0.047 | 25.5 |
| cam 20 disp 20 grey16 | 99.3 | 88.8 | 14.0 | 99.3 | 0 | 0.013 | 0.005 | 2.7 |
| cam 20 disp 30 binary | 99.1 | 80.3 | 27.3 | 99.1 | 0 | 0.027 | 0.025 | 13.6 |
| cam 20 disp 30 grey16 | 99.6 | 90.2 | 11.6 | 99.6 | 0 | 0.012 | 0.003 | 1.8 |
| cam 20 disp 45 binary | 99.6 | 87.5 | 15.5 | 99.6 | 0 | 0.017 | 0.012 | 6.6 |
| cam 20 disp 45 grey16 | 99.4 | 90.9 | 12.5 | 99.4 | 0 | 0.012 | 0.003 | 1.5 |
| cam 20 disp 70.3 binary | 99.6 | 89.3 | 13.0 | 99.6 | 0 | 0.017 | 0.012 | 6.7 |
| cam 20 disp 70.3 grey16 | 99.6 | 90.4 | 12.4 | 99.6 | 0 | 0.012 | 0.002 | 1.3 |
| cam 20 disp 105 binary | 99.5 | 91.0 | 12.9 | 99.5 | 0 | 0.013 | 0.006 | 3.4 |
| cam 20 disp 140 binary | 99.3 | 90.3 | 11.3 | 99.3 | 0 | 0.012 | 0.005 | 2.5 |
| cam 20 disp 200 binary | 99.5 | 90.9 | 10.6 | 99.5 | 0 | 0.012 | 0.003 | 1.4 |
| cam 32 print | 99.8 | 92.8 | 6.9 | 99.8 | 0 | 0.007 | 0.002 | 0.5 |
| cam 32 disp 20 binary | 98.9 | 77.7 | 42.8 | 98.9 | 0 | 0.084 | 0.086 | 28.8 |
| cam 32 disp 20 grey16 | 99.6 | 92.3 | 9.6 | 99.6 | 0 | 0.013 | 0.009 | 3.2 |
| cam 32 disp 30 binary | 99.5 | 84.7 | 21.5 | 99.5 | 0 | 0.034 | 0.034 | 11.3 |
| cam 32 disp 30 grey16 | 99.6 | 92.2 | 7.6 | 99.6 | 0 | 0.009 | 0.004 | 1.5 |
| cam 32 disp 45 binary | 99.7 | 90.6 | 11.3 | 99.7 | 0 | 0.018 | 0.014 | 4.6 |
| cam 32 disp 45 grey16 | 99.7 | 92.3 | 7.9 | 99.7 | 0 | 0.008 | 0.003 | 0.9 |
| cam 32 disp 70.3 binary | 99.8 | 91.5 | 8.2 | 99.8 | 0 | 0.017 | 0.015 | 5.0 |
| cam 32 disp 70.3 grey16 | 99.8 | 93.4 | 7.1 | 99.8 | 0 | 0.008 | 0.002 | 0.6 |
| cam 32 disp 105 binary | 99.6 | 92.8 | 8.2 | 99.6 | 0 | 0.010 | 0.007 | 2.5 |
| cam 32 disp 140 binary | 99.8 | 92.5 | 9.2 | 99.8 | 0 | 0.010 | 0.007 | 2.2 |
| cam 32 disp 200 binary | 99.6 | 92.0 | 9.0 | 99.6 | 0 | 0.008 | 0.003 | 1.1 |
| cam 56 print | 96.8 | 88.9 | 10.7 | 96.8 | 0 | 0.005 | 0.002 | 0.4 |
| cam 56 disp 20 binary | 92.6 | 70.1 | 40.8 | 92.6 | 0 | 0.185 | 0.189 | 36.3 |
| cam 56 disp 20 grey16 | 97.7 | 89.7 | 10.9 | 97.7 | 0 | 0.028 | 0.027 | 5.1 |
| cam 56 disp 30 grey16 | 97.9 | 89.6 | 10.2 | 97.9 | 0 | 0.010 | 0.008 | 1.5 |
| cam 56 disp 45 binary | 97.0 | 87.6 | 14.4 | 97.0 | 0 | 0.028 | 0.023 | 4.5 |
| cam 56 disp 70.3 binary | 97.5 | 89.4 | 12.0 | 97.5 | 0 | 0.022 | 0.021 | 4.0 |
| cam 56 disp 70.3 grey16 | 97.4 | 89.5 | 9.6 | 97.4 | 0 | 0.006 | 0.003 | 0.5 |
| cam 56 disp 200 binary | 97.3 | 89.1 | 10.1 | 97.3 | 0 | 0.007 | 0.005 | 1.0 |

The grey16 rows at 105–200 display px are all within 0.1 µm of print; the full
printout has them. Supersampling check: going from 4 to 8 samples per
display-pixel side changes decoded % by 1.7 pp at most and the marker-to-marker
error by 0.1 µm at most (`display_supersampling`).

### Grey depth and dither (device pitch)

| cell | decoded % | wrong among decodes % | photometric RMS px | marker-to-marker px | marker-to-marker µm |
|---|---:|---:|---:|---:|---:|
| cam 32 print | 92.8 | 6.9 | 0.007 | 0.002 | 0.5 |
| cam 32 binary | 91.5 | 8.2 | 0.017 | 0.015 | 5.0 |
| cam 32 grey2 | 92.0 | 9.4 | 0.017 | 0.016 | 5.3 |
| cam 32 grey4 | 92.8 | 8.7 | 0.010 | 0.006 | 2.0 |
| cam 32 grey16 | 93.4 | 7.1 | 0.008 | 0.002 | 0.6 |
| cam 32 grey256 | 92.7 | 6.5 | 0.008 | 0.002 | 0.5 |
| cam 32 dither2 | 92.3 | 10.2 | 0.019 | 0.017 | 5.7 |
| cam 32 dither4 | 93.0 | 8.3 | 0.009 | 0.005 | 1.6 |
| cam 32 dither16 | 92.4 | 7.7 | 0.008 | 0.002 | 0.7 |
| cam 20 print | 90.8 | 12.3 | 0.012 | 0.002 | 1.2 |
| cam 20 binary | 89.3 | 13.0 | 0.017 | 0.012 | 6.7 |
| cam 20 grey4 | 90.9 | 11.5 | 0.012 | 0.005 | 2.6 |
| cam 20 grey16 | 90.4 | 12.4 | 0.012 | 0.002 | 1.3 |
| cam 20 dither16 | 90.6 | 12.2 | 0.012 | 0.002 | 1.1 |

### Alignment and lattice (device pitch, camera 32)

`rect91` is a rect lattice whose pitch is a whole 91 display px (13.92 mm, the
closest to the hex nearest-neighbour distance), with the default ring
geometry.

| cell | decoded % | wrong among decodes % | photometric RMS px | common px | marker-to-marker px | marker-to-marker µm |
|---|---:|---:|---:|---:|---:|---:|
| hex print | 92.8 | 6.9 | 0.007 | 0.001 | 0.002 | 0.5 |
| hex binary, phase .37 | 91.5 | 8.2 | 0.017 | 0.001 | 0.015 | 5.0 |
| hex binary, rotated 1° | 92.1 | 9.7 | 0.015 | 0.001 | 0.013 | 4.5 |
| hex binary, rotated 3° | 91.7 | 10.6 | 0.014 | 0.001 | 0.012 | 3.9 |
| rect91 print | 92.4 | 8.3 | 0.007 | 0.000 | 0.001 | 0.4 |
| rect91 binary, phase .37 | 90.2 | 8.5 | 0.017 | 0.014 | 0.006 | 1.9 |
| rect91 binary, centres on pixel centres | 90.7 | 10.2 | 0.012 | 0.001 | 0.009 | 3.0 |
| rect91 binary, centres on pixel corners | 90.4 | 10.3 | 0.010 | 0.001 | 0.007 | 2.2 |
| rect91 binary, rotated 1° | 90.9 | 8.9 | 0.016 | 0.001 | 0.014 | 4.6 |
| hex grey16 (any phase / rotation) | 93.2–93.4 | 7.1–7.8 | 0.008 | 0.001 | 0.001–0.002 | 0.5–0.6 |
| rect91 grey16 (any phase / rotation) | 90.6–93.1 | 6.5–8.9 | 0.008 | 0.000–0.001 | 0.001–0.002 | 0.5 |

With an integer pitch and an axis-aligned grid, every marker is rasterised
identically. A binary display's bias then becomes a common board translation
(0.014 px), and only 1.9 µm varies from marker to marker. One degree of
rotation removes that benefit. Even with snapped centres, a binary marker is
not error-free, because the 16-sector code pattern is not symmetric.

### Snapping marker centres to pixels (camera 32)

`TargetLayout` cannot represent per-cell positions: cells come from the
lattice geometry. Snapping was therefore scored two ways. Against the moved
(true) centres, binary goes from 5.0 to 3.0 µm and grey16 stays at 0.6 µm.
Against the layout, which is what a calibration would use, the error is
62.6 µm marker to marker: a uniform ±½ pixel displacement. Snapping is worth
nothing unless the layout can carry the snapped positions, and even then it
gains only 2 µm and only for a binary display.

### Blur (default board, device pitch)

| cell | found % | completion-only % | decoded % | wrong among decodes % | correct ID % | wrong final IDs | photometric RMS px | marker-to-marker µm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cam 20 blur 0.8 print | 99.4 | 0.6 | 90.8 | 12.3 | 99.4 | 0 | 0.012 | 1.2 |
| cam 20 blur 0.8 grey16 | 99.6 | 0.4 | 90.4 | 12.4 | 99.6 | 0 | 0.012 | 1.3 |
| cam 20 blur 1.5 print | 94.0 | 7.5 | 57.6 | 66.5 | 91.9 | 34 | 0.022 | 3.3 |
| cam 20 blur 1.5 binary | 93.8 | 8.8 | 58.4 | 66.4 | 90.6 | 52 | 0.027 | 8.9 |
| cam 20 blur 1.5 grey16 | 95.2 | 6.4 | 60.0 | 65.2 | 95.2 | 0 | 0.022 | 3.5 |
| cam 20 blur 3.0 print | 60.5 | 0.0 | 17.0 | 97.8 | 0.3 | 55 | – | – |
| cam 20 blur 3.0 grey16 | 60.9 | 0.0 | 15.7 | 98.0 | 0.2 | 50 | – | – |
| cam 32 blur 0.8 print | 99.8 | 0.6 | 92.8 | 6.9 | 99.8 | 0 | 0.007 | 0.5 |
| cam 32 blur 0.8 grey16 | 99.8 | 0.5 | 93.4 | 7.1 | 99.8 | 0 | 0.008 | 0.6 |
| cam 32 blur 1.5 print | 99.8 | 0.9 | 90.5 | 12.7 | 99.8 | 0 | 0.013 | 1.2 |
| cam 32 blur 1.5 grey16 | 99.8 | 0.9 | 91.0 | 13.1 | 99.8 | 0 | 0.013 | 1.3 |
| cam 32 blur 2.0 print (band-blur run) | 98.8 | 2.6 | 82.9 | 27.7 | 98.8 | 0 | 0.020 | 2.6 |
| cam 32 blur 3.0 print | 79.9 | 24.1 | 35.6 | 82.6 | 55.1 | 42 | 0.040 | 7.2 |
| cam 32 blur 3.0 binary | 80.3 | 16.6 | 39.5 | 86.5 | 42.5 | 170 | 0.043 | 7.2 |
| cam 32 blur 3.0 grey16 | 80.2 | 21.6 | 36.5 | 83.8 | 51.2 | 73 | 0.042 | 7.4 |

"–": no marker received a photometric centre in that cell.

### Target geometry (device pitch, blur 0.8)

Recipes are written as outer edge / stroke / band in device pixels.
"wide band" is the drawing of the older generator `tools/gen_synth.py`: the
same outer and inner edges, 0.736 mm strokes and a 1.28 mm band.

| cell | found % | decoded % | wrong among decodes % | projective RMS px | photometric RMS px | marker-to-marker µm |
|---|---:|---:|---:|---:|---:|---:|
| cam 20 default print | 99.4 | 90.8 | 12.3 | 0.111 | 0.012 | 1.2 |
| cam 20 default grey16 | 99.6 | 90.4 | 12.4 | 0.109 | 0.012 | 1.3 |
| cam 20 default, decoder on band centreline only, grey16 | 99.7 | 89.0 | 13.6 | 0.111 | 0.012 | 1.3 |
| cam 20 35/7/3 px grey16 | 99.8 | 91.7 | 11.0 | 0.110 | 0.012 | 1.3 |
| cam 20 wide band grey16 | 99.4 | 90.4 | 10.4 | 0.167 | 0.011 | 1.3 |
| cam 20 35/5/8 px grey16 | 99.0 | 90.8 | 10.4 | 0.153 | 0.011 | 1.1 |
| cam 20 rect91 grey16 | 99.7 | 89.9 | 13.5 | 0.089 | 0.012 | 1.5 |
| cam 32 default print | 99.8 | 92.8 | 6.9 | 0.192 | 0.007 | 0.5 |
| cam 32 default grey16 | 99.8 | 93.4 | 7.1 | 0.196 | 0.008 | 0.6 |
| cam 32 default, decoder on band centreline only, grey16 | 99.8 | 92.0 | 7.9 | 0.193 | 0.008 | 0.6 |
| cam 32 35/7/3 px grey16 | 99.8 | 93.6 | 8.5 | 0.202 | 0.008 | 0.7 |
| cam 32 wide band grey16 | 99.8 | 94.7 | 7.6 | 0.300 | 0.007 | 0.8 |
| cam 32 35/5/8 px grey16 | 98.2 | 92.2 | 7.5 | 0.421 | 0.007 | 0.8 |
| cam 32 rect91 grey16 | 99.9 | 91.2 | 8.9 | 0.146 | 0.008 | 0.5 |

At blur 0.8, geometry changes decoding by 1.5 pp at most. A larger marker on
the same device, seen from the same camera distance (1.5×: 105 display px,
camera 20→30 and 32→48), changes decoding by −4 to +3 pp. Photometric RMS
in camera px drops from 0.012 to 0.008 and from 0.007 to 0.006
(`display_scaled_marker`).

### Code band against blur (device pitch)

Recipes are outer edge / stroke / band in device pixels:

| recipe | outer edge | stroke | band | drawn Ø |
|---|---|---|---|---|
| default | 5.376 mm | 1.152 mm | 0.448 mm | 70.3 px |
| 35/5/8 | 5.355 mm | 0.765 mm | 1.224 mm | 70 px |
| 39/5/10 | 5.967 mm | 0.765 mm | 1.530 mm | 78 px |

39/5/10 is seen from the same camera distance, so its diameter in camera px
is 1.11× larger.

| cell | found % | completion-only % | decoded % | wrong among decodes % | correct ID % | wrong final IDs | projective RMS px | photometric RMS px |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cam 20 b0.8 default grey16 | 99.6 | 0.4 | 90.4 | 12.4 | 99.6 | 0 | 0.109 | 0.012 |
| cam 20 b0.8 35/5/8 grey16 | 99.4 | 0.6 | 91.9 | 10.7 | 99.4 | 0 | 0.167 | 0.011 |
| cam 20 b0.8 39/5/10 grey16 | 99.2 | 1.0 | 91.1 | 9.7 | 99.2 | 0 | 0.162 | 0.010 |
| cam 20 b1.5 default print | 94.0 | 7.5 | 57.6 | 66.5 | 91.9 | 34 | 0.199 | 0.022 |
| cam 20 b1.5 default grey16 | 95.2 | 6.4 | 60.0 | 65.2 | 95.2 | 0 | 0.201 | 0.022 |
| cam 20 b1.5 35/5/8 grey16 | 96.5 | 4.1 | 69.0 | 47.2 | 96.5 | 0 | 0.310 | 0.026 |
| cam 20 b1.5 39/5/10 grey16 | 95.9 | 5.4 | 77.2 | 29.5 | 95.9 | 0 | 0.342 | 0.023 |
| cam 20 b2.0 default grey16 | 71.4 | 15.4 | 34.0 | 90.6 | 19.1 | 177 | 0.239 | – |
| cam 20 b2.0 35/5/8 grey16 | 73.7 | 16.7 | 35.2 | 86.0 | 36.6 | 76 | 0.322 | 0.075 |
| cam 20 b2.0 39/5/10 grey16 | 60.9 | 13.8 | 30.6 | 74.3 | 52.6 | 16 | 0.403 | 0.033 |
| cam 20 b3.0 default grey16 | 60.9 | 0.0 | 15.7 | 98.0 | 0.2 | 50 | 0.341 | – |
| cam 20 b3.0 35/5/8 grey16 | 9.6 | 0.0 | 6.0 | 100.0 | 0.0 | 27 | – | – |
| cam 20 b3.0 39/5/10 grey16 | 0.1 | 0.0 | 0.1 | 100.0 | 0.0 | 0 | – | – |
| cam 32 b0.8 default grey16 | 99.8 | 0.5 | 93.4 | 7.1 | 99.8 | 0 | 0.196 | 0.008 |
| cam 32 b0.8 39/5/10 grey16 | 99.8 | 0.4 | 93.4 | 6.9 | 99.8 | 0 | 0.347 | 0.007 |
| cam 32 b1.5 default grey16 | 99.8 | 0.9 | 91.0 | 13.1 | 99.8 | 0 | 0.289 | 0.013 |
| cam 32 b1.5 39/5/10 grey16 | 99.3 | 1.0 | 91.9 | 9.8 | 99.3 | 0 | 0.447 | 0.013 |
| cam 32 b2.0 default grey16 | 98.9 | 2.6 | 83.2 | 27.4 | 98.9 | 0 | 0.385 | 0.019 |
| cam 32 b2.0 35/5/8 grey16 | 98.8 | 2.5 | 86.9 | 19.6 | 98.8 | 0 | 0.462 | 0.022 |
| cam 32 b2.0 39/5/10 grey16 | 98.7 | 2.9 | 89.3 | 15.3 | 98.7 | 0 | 0.495 | 0.019 |
| cam 32 b3.0 default grey16 | 80.2 | 21.6 | 36.5 | 83.8 | 51.2 | 73 | 0.485 | 0.042 |
| cam 32 b3.0 35/5/8 grey16 | 66.1 | 10.7 | 32.5 | 81.4 | 38.1 | 32 | 0.559 | 0.057 |
| cam 32 b3.0 39/5/10 grey16 | 40.5 | 9.1 | 23.4 | 50.1 | 37.7 | 7 | 0.651 | 0.037 |

The full printout (print rows, 35/4/10) is produced by `display_band_blur`.

## Findings

### A1. The display is not the cause of decode failures

At the device pitch, display rasterisation changes the decode rate and the
wrong-decode rate by 2 pp or less with grey output (4 or more levels), and by
4 pp or less with binary or 2-level output, at every camera scale and blur
tested. It matters for decoding only below about 45 display px
per marker (a code band under about 1.9 display px), and only for a binary
display: at 20 display px, binary decodes 70–78 % with 41–46 % of decodes
wrong. Area coverage at 16 levels still matches print at 20 display px, where
the band is 0.83 display px wide.

### A2. Blur against band width is the cause

On print, wrong decodes rise from 7–12 % at blur 0.8 to 13 % (camera 32) and
66 % (camera 20) at blur 1.5, then to 28 % at camera 32 / blur 2.0, and to
84–98 % at blur 3.0. At blur 3, the wrong final IDs (42–170 per cell) come
from a few whole-board failures of 19–50 IDs each. The differences between
print, binary and grey16 there are therefore not significant. The project's real images contradict their lattice cell in
22–30 % of raw decodes. In this simulation that corresponds to a blur σ of
about 1.5–2 camera px at the small end of the 20–56 px range. A 0.448 mm band
is 0.8–1.3 camera px wide there, so blur alone removes most of its contrast.

Wrong decodes are confident. The default gates (distance ≤ 3, margin ≥ 1,
confidence ≥ 0.3, both polarities) accept 52.6 % of all 65 536 16-bit words,
and 37.1 % at distance 0 (`codebook_random_word_acceptance`). An unreadable
band therefore yields a "perfect" wrong ID about a third of the time. In the
heavy-blur frames below, 108 of the 130 wrong ID reports in frames without a
homography had distance 0 and margin 2.

Decoding only on the band centreline (one radial sampling ring instead of
three at ±10 %, two of which land on the strokes of the default board) changes
nothing: within ±1.4 pp.

### A3. Where the "undecoded" markers went: the scale prior

A first pass of these tables used a fixed-size scale prior (`[d, d]`). On
print at blur 0.8, it left 25–30 % of visible markers undecoded even at
camera 56. The tilt split shows the cause, which is not decoding.

| default print, blur 0.8 | prior | tilt | found % | completion-only % | decoded % |
|---|---|---|---:|---:|---:|
| cam 20 | fixed `[d, d]` | 0° | 99.5 | 0.6 | 91.8 |
| cam 20 | fixed `[d, d]` | 20° | 86.1 | 30.2 | 50.6 |
| cam 20 | ±30 % | 20° | 99.5 | 0.6 | 89.7 |
| cam 32 | fixed `[d, d]` | 0° | 99.9 | 0.4 | 94.1 |
| cam 32 | fixed `[d, d]` | 20° | 92.5 | 30.6 | 59.0 |
| cam 32 | ±30 % | 20° | 99.5 | 0.7 | 92.2 |

At 20° tilt the far-side markers are foreshortened to about 0.8–0.95 of the
nominal size. With a fixed prior, the proposal stage misses about a third of
the markers. On one frame at camera 32, 51 proposals were made for 72 visible
markers and 24 visible markers had none; a 22–40 px prior gave 78 proposals
and no visible marker without one. Completion then fits those markers at the
correct cell but keeps no decode, so they counted as "undecoded". With a ±30 %
prior, fit-decode finds them and they decode at 90–93 %. All design tables
above use the ±30 % prior.

With the fixed prior, the wide band suffered less (6–12 % completion-only
against 30 %); why the proposal response depends on the band was not
investigated. The scenario has two parts. With a fixed prior, the remaining
gap is the tilt; with a realistic prior, it is 7–10 % undecoded plus 7–12 %
wrong among decodes, the band-against-noise limit of A2.

### A4. Wrong final IDs come from the decode-anchored fallback, in 0.15.0 too

Same scenes run through the released 0.15.0 dev CLI (`704a4dfa`) and the
grid-first dev CLI. The images, target JSON and ground truth come from
`display_export_scenes`; 24 frames per scene; diameter prior ±30 %:

| scene (default board) | 0.15.0 correct / wrong / H | grid-first correct / wrong / H |
|---|---|---|
| cam 20, blur 1.5, print | 1486 / 34 / 24 of 24 | 1500 / 34 / 24 of 24 |
| cam 20, blur 1.5, grey16 | 1543 / 0 / 24 | 1554 / 0 / 24 |
| cam 20, blur 3.0, print | 5 / 55 / 0 | 5 / 55 / 0 |
| cam 32, blur 2.0, print | 1608 / 0 / 24 | 1609 / 0 / 24 |
| cam 32, blur 3.0, print | 767 / 61 / 15 | 897 / 42 / 16 |
| cam 32, blur 3.0, grey16 | 740 / 75 / 14 | 833 / 73 / 15 |
| cam 32, blur 0.8 / cam 56, blur 0.8, print | 1624 / 0, 1590 / 0 | 1624 / 0, 1591 / 0 |

With a fixed `[d, d]` prior (the configuration that produced the earlier 180
wrong IDs), 0.15.0 had 257 wrong at camera 32 / blur 3 grey16 against 180 for
grid-first, and 108 against 75 on print. The grid-first CLI reproduces the
library study's counts exactly.

- **The pattern is not new with grid-first.** Grid-first is equal or better in
  every scene.
- **Anchored frames have no wrong IDs.** In every frame where the grid-first
  decoded-ID anchor resolved, there are 0 wrong final IDs (116 frames with the
  fixed prior, 120 with ±30 %).
- **All wrong final IDs come from the fallback.** They occur in frames where
  the anchor vote had no majority. For example, `n_decoded=7 votes=3
  runner_up=2` fails the strict-majority rule; so do `4/2/1` and `1/1/1`.
  Those frames go to the decode-anchored fallback, which is the 0.15.0
  pipeline: ID correction, global RANSAC homography, completion. 0.15.0 makes
  the same errors on the same frames.
- **How a whole board goes wrong.** ID correction "recovers" 10–44 IDs from
  their neighbours around wrong seeds. That makes a self-consistent wrong
  labelling, and RANSAC then fits it with tight residuals: 7/15 inliers at
  0.09 px mean, or 21/30 at 0.25 px. The result is a wrong board homography
  and 19–50 wrong IDs in one frame.
- **No homography means the raw decodes pass through.** Without a homography,
  frames report the raw decodes, and at blur 3 about 78 % of those are wrong.

**A candidate acceptance rule (report only, not implemented).** Accept a
fallback board homography only if at least 3 markers' own raw decodes agree
with the IDs it assigns.

| prior | wrong homographies rejected | wrong IDs removed | correct IDs lost |
|---|---|---|---|
| fixed `[d, d]` | 6 of 6 | 192 | 33 (one correct frame with support 2/16) |
| ±30 % | 2 of 3 | 47 of 81 | 0 |

Under the ±30 % prior it misses one frame, with support 3/8. Raising the
threshold to 4 removes all 81 but loses 215 correct IDs. Raw-decode quality
cannot separate good from bad, because heavy-blur misreads sit at distance 0.
This is an in-sample rule from 72 fallback frames, not a validated design.

### A5. Centre error

The photometric centre carries a rasterisation term only on a coarse
display:

- **Binary display:** 4–7 µm marker-to-marker at the device pitch (0.012–0.022
  camera px), about 2–3 times print. It falls below 2 µm only at 140–200
  display px per marker.
- **4 or more grey levels, or ordered dither at 4 or more levels:** at most
  2.6 µm at the device pitch.
- **16 or more levels:** equal to print, about 0.5–1.3 µm, from 30 display px
  per marker up.

Projective centres (0.11–0.29 px at blur 0.8–1.5) are dominated by the
method's own bias and do not see the display. They do see the geometry: a
thin-stroke wide-band marker raises projective RMS from 0.19 to 0.30–0.43 px
at camera 32. Photometric RMS is unchanged.

### B. Display and geometry rule

**Display px per marker.**
- *Area coverage at 4 or more grey levels:* about 30 display px per drawn
  marker diameter, i.e. a code band of at least about 1.2 display px, is
  enough for decoding and centres to match print. With 16 levels, 20 display
  px already matches print on decoding.
- *Binary:* at least 45 display px per diameter (band ≥ 1.9 display px) for
  decoding, and at least 140–200 for the centre to match print within
  about 1–2 µm.

**Grey depth.** 16 levels is indistinguishable from 256. 4 levels leaves
about 2 µm. Binary and 2-level output (rounded or dithered) cost 5–7 µm.

**Alignment and lattice.** Irrelevant with grey output. With binary output, a
rect lattice whose pitch is a whole number of display pixels, axis-aligned,
moves most of the bias (0.014 of 0.015 px) into a harmless common
translation. Rotating it
1° undoes that.

**Geometry.**
- *Blur at or below about 1 camera px at the markers:* the default geometry is
  fine. Neither a wider band nor whole-pixel edges change decoding by more
  than 1.5 pp.
- *Blur of 1.5–2 camera px:* a wider band pays. 39/5/10 against default at the
  same camera distance:

  | | decoded | wrong among decodes | correct final IDs | wrong final IDs |
  |---|---|---|---|---|
  | camera 20, blur 1.5 | 77 % vs 60 % | 30 % vs 65 % | 96 % vs 95 % | 0 vs 0 (grey16) |
  | camera 32, blur 2.0 | 89 % vs 83 % | 15 % vs 27 % | — | — |
  | camera 20, blur 2.0 | — | — | 53 % vs 19 % | 16 vs 177 |

- *Blur of 3 camera px:* the thin 5 px strokes stop being found at all
  (camera 20: 0.1 % found against 61 %; camera 32: 41 % against 80 %). Thinner
  strokes also raise projective-centre RMS. Thicker strokes with a wider band
  inside a larger outer edge were not tested.

### Recommended displayed-target recipe for the 166 dpi device

1. **Show the target with area-coverage grey levels, not as a binary image.**
   The library's PNG renderer is binary (see below). Render at 8 × 166 =
   1328 dpi, box-average each 8×8 block to a 166 dpi greyscale PNG, and show
   it at 100 %. At the device's 16 grey levels this is the `grey16` case:
   centres equal to print (0.6 µm against 5 µm marker-to-marker at camera 32;
   1.3 against 6.7 µm at camera 20). Decoding is unchanged. Keep the lattice
   axis-aligned; its phase does not matter.
2. **Geometry, only if the real blur is 1.5–2 camera px.** On the existing
   8 mm hex pitch, draw the outer edge at 39 px (5.967 mm), strokes of 5 px
   (0.765 mm), a band of 10 px (1.530 mm) and the inner edge at 19 px
   (2.907 mm). In library terms: `outer_radius_mm` 5.5849,
   `inner_radius_mm` 3.2898, `ring_width_mm` 0.7651. The drawn diameter is
   11.93 mm against a 13.86 mm neighbour distance, which validates.

   Simulated gain over the default:

   | | decoded | wrong among decodes |
   |---|---|---|
   | camera 20, blur 1.5 | 60 → 77 % | 65 → 30 % |
   | camera 32, blur 2.0 | 83 → 89 % | 27 → 15 % |
   | camera 20, blur 2.0 | — | — |

   At camera 20 / blur 2.0, wrong final IDs drop from 177 to 16. The costs:
   ring detection collapses at blur 3, and projective centres are worse.
   Photometric centres are unchanged. If the real blur is about 1 px or less,
   keep the default geometry.
3. **Do not snap marker centres to pixels.** It costs 63 µm against the
   layout, which cannot represent the snapped positions.

### The PNG renderer (`render_target_png`)

`crates/ringgrid/src/target_generation.rs`, `render_target_png`, tests every
output pixel at its integer index:

```rust
let dx = f64::from(x) - cx;
let dy = f64::from(y) - cy;
let dist_sq = dx * dx + dy * dy;
```

It writes 0 or 255 by a range test on `dist_sq`, with
`cx = (cell.x + offset_x) * px/mm`. So:

- **The output is binary.** A rendered plain 1×2 rect target (outer 3.95 mm,
  inner 2.05 mm, 254 dpi, margin 0, 179×179 px) contains exactly the values
  {0, 255}.
- **It is half a pixel off.** The first marker's layout centre is 39.5 px
  (3.95 mm × 10 px/mm). The black run through it spans pixel indices 1..78
  (rows) and 51..128 (columns), centred on 39.5 / 89.5 in index units. Under
  the usual area convention, where pixel *i* covers [*i*, *i* + 1) and which
  is how a display or printer shows the PNG, the drawn disc is centred at
  40.0 / 90.0 px, i.e. **+0.5 px in x and y**. At 166 dpi that is 0.077 mm.
  Every marker shifts the same way, so it is a board translation and
  harmless to calibration; but the pixel content is a point sample at the
  pixel corner.

A PNG rendered at the display's own dpi is therefore exactly this study's
`binary` mode: 5–7 µm marker-to-marker photometric error, 1–2 pp lower
decoding. Area-coverage rendering would make it the `grey16` / `grey256`
mode: equal to print on both. It would also centre the pattern correctly,
because coverage is a property of the pixel area. Whether the reader resamples
a higher-dpi PNG with area averaging, nearest neighbour or dithering is
unknown; it decides which row of the grey-depth table applies.

## Limits of the simulation

- **Not modelled:** e-ink pixel gaps and fill factor, reflectance
  non-uniformity, the front-light and front glass (reflections, parallax at
  oblique views), ghosting, lower contrast than paper (contrast here is 80 grey
  levels for both), the panel's actual grey response, and whether the reader
  resamples, dithers or thresholds the PNG.
- **The camera is idealised.** It is a pinhole with a Gaussian PSF: no lens
  distortion, no anisotropic or spatially varying blur, no compression.
- **The views are limited.** Tilts are 0° and 20°, each cell has 3 poses × 4
  seeds, and the noise is white. With about 1630 markers per cell, rates have
  about ±1 pp of sampling error, and 0.1–0.2 µm differences in
  marker-to-marker error are noise.
- **"Decoded" counts only fit-decoded rings that survive into the result.** A
  ring the pipeline drops does not count, and completion keeps only agreeing
  decodes.
- **Edge markers are clipped.** The ideal raster's bilinear lookup clips the
  last raster pixel at the board edge. This affects only markers touching the
  board edge, which are outside the simulated views (test code only).

## Validation measurement

Measure the camera's blur σ, in camera px, at the code band on the real images
of the displayed target. Use the edge-spread width of the outer ring edge on
markers at the small end of the 20–56 px range. The blur and code-band tables
then predict the rate of raw decodes that contradict their lattice cell. If σ
is about 1.5–2 px, the prediction is about 15–30 %, against the observed
22–30 %, and the conclusion stands: blur against band width, not display
aliasing.

As a secondary check, take a macro photograph of the display showing the
target. It tells whether the reader shows the PNG 1:1, thresholds it,
anti-aliases it, or dithers it.

## Reproduce

All in release, from the repository root; `DISPLAY_THREADS` sets the worker
count.

```text
# Existing centre-error study
cargo test --release -p ringgrid --features study --test displayed_target \
    displayed_target_centre_error -- --ignored --nocapture --exact

# Design tables (DISPLAY_SEEDS=4, ±30 % prior by default)
for t in display_resolution display_raster display_alignment display_blur \
         display_geometry display_scaled_marker display_band_blur \
         display_supersampling codebook_random_word_acceptance; do
  cargo test --release -p ringgrid --features study --test displayed_target \
      $t -- --ignored --nocapture --exact
done
DISPLAY_SS=8 cargo test ... display_supersampling ...   # supersampling check

# Tilt split and the fixed-size prior of finding A3
DISPLAY_TILTS=0  cargo test ... display_geometry ...
DISPLAY_TILTS=20 cargo test ... display_geometry ...
DISPLAY_PRIOR_SPAN=0 DISPLAY_TILTS=20 cargo test ... display_geometry ...

# Scenes for the 0.15.0 comparison (PNG + target.json + *.truth.json)
DISPLAY_EXPORT_DIR=<dir> cargo test --release -p ringgrid --features study \
    --test displayed_target display_export_scenes -- --ignored --exact
# then, per image, for both builds (0.15.0 = git 704a4dfa):
ringgrid-dev detect --image <png> --target <dir>/target.json --out <json> \
    --marker-diameter-min <0.7 d> --marker-diameter-max <1.3 d>
```

`DISPLAY_CAMS`, `DISPLAY_BLURS`, `DISPLAY_DISPS` and `DISPLAY_TILTS`
(comma-separated) narrow a sweep. The default-run test
`display_raster_preserves_marker_geometry` guards the display model's
phase / rotation / snapping maps.
