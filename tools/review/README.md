# Detection review tool

A local web page for reviewing detection results on real images: flag wrong
IDs / centres / false positives, pin missed markers, leave remarks, compare
detector runs. The feedback is plain JSON meant to be read back as ground
truth. Nothing leaves the machine.

The page is a small React app in `app/`, built on the `@vitavision/*` packages
(`stage2d` image stage and overlay layers, `workbench` shell, `ui` controls,
`charts` colour map). Build it once (and after pulling changes to `app/`):

```bash
cd tools/review/app && bun install && bun run build   # -> app/dist, served by server.py
```

`app/src/shims/` holds two temporary stand-ins for package code contributed to
lab-ui but not released yet (`ImageLayer` gain/gamma, `ShortcutList`); each
says which import replaces it once released.

Until it is built the server answers the page with the build command and says
so at startup. `bun run dev` serves the app with hot reload on 127.0.0.1:5174,
proxying `/api` to a server on port 8766 (`REVIEW_API` to change it).

```bash
# rtv3d strips (4320x540 -> 6 tiles of 720x540): one adaptive run (the
# published CLI without --marker-diameter = detect_adaptive, the configuration
# run_rtv3d_eval.py gates as primary); the page opens it as A
.venv/bin/python tools/review/server.py --data data/rtv3d_0 --split 6 --open \
    --detect adaptive=target/release/ringgrid -- --target tools/board/board_spec.json

# reopen later: runs are cached, --primary picks the run shown as A
.venv/bin/python tools/review/server.py --data data/rtv3d_0 --split 6 --open --primary adaptive

# a plain image directory, plus an already-produced directory of detection JSONs
python3 tools/review/server.py --data some/images --run release-0.15=path/to/jsons

# read the feedback back
python3 tools/review/summarize.py tools/out/review/rtv3d_0
```

Another run can be added later from the page ("new run...", only binaries the
server knows are offered) or by restarting with a second
`--detect other=/path/to/other-binary -- <args>` (all `--detect` runs of one
start share the args after `--`). The page opens `--primary` as run A
(default: the first `--detect`, else the first `--run`, else the first run by
name). Runs are cached under
`<out>/runs/<name>/` (detections + `manifest.json` with binary, version, git
commit, args, timestamp) and reused on restart. `--run NAME=DIR` imports
`<tile>.json` files in place (also `<tile>_*.json` when unambiguous).

Output (`--out`, default `tools/out/review/<dataset>/`): `feedback.json`
(`ringgrid.review.v1`, written atomically), `runs/`, `tiles/` (cropped PNG cache).

## Confidence threshold

**Min confidence** in the header (a slider and a field, 0–1 in steps of 0.01, `[`/`]`
step by 0.05) hides markers whose confidence is below it, for both runs. It shows
what cutting the detections there would remove. At 0 (the default) every marker is
shown. A marker without a confidence is never cut. The tile list counts the kept
markers ("56 of 63"), and the status bar shows the current tile's counts. In compare
mode the server matches the two runs after cutting both at the threshold, so a
marker whose partner was cut shows as "only A/B". Flagged and selected markers stay
visible, labelled "cut", so a threshold that removes a confirmed marker or a false
positive shows it. The stage's layers menu has **Markers below the threshold**, which
draws the cut markers greyed out instead of hiding them. The threshold is remembered
per browser (`localStorage`) and is never written to the feedback. The "differs
between runs" filter and badge still use all markers.

## Generate a screen target

Open **Generate screen target** in the React app and choose a coded hex or
plain rectangular template. The InkPad Lite preset fills in the panel
resolution and nominal physical pixel pitch. Enter the drawn marker diameter,
generate, and download the native PNG and both JSON files. Display the PNG
at 1:1 and pass the target spec to the detector.

This dialog uses the RingGrid Python extension in the local server process.
Build the bindings with
`VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
and run the server with `.venv/bin/python`.

## Privacy

- Binds to loopback only (non-loopback `--host` is refused); `Host`/`Origin`
  are checked; the page loads no external resources (scripts, styles and the
  IBM Plex fonts are all in the bundle) and is served with a CSP that allows
  `'self'` only. Only files under `app/dist` with a known type are served.
- At startup `--out` must be git-ignored (or outside any repo), otherwise the
  server refuses to start (`--allow-unignored-out` overrides, with a warning).
- Only detector binaries given with `--detect` or the repo's own
  `target/{release,debug}/{ringgrid-dev,ringgrid}` can be executed; request-supplied commands are never run, and `--image/--out`
  cannot be overridden.

## Requirements

Server: Python standard library only. Pillow is needed only for `--split`, non-PNG
inputs or tile caching (the repo `.venv` has it). Strips are split by
`tools/strip_tiles.py`, shared with `run_rtv3d_eval.py`; tiles of the strips
that evaluation skips (`target_0`, `target_7`) are shown with a `skip` badge.
App: [bun](https://bun.sh) to install and build; the packages come from npm.

## Feedback file

Per tile: `verdict` (`ok|has_issues|unusable_image`), `remark`, `reviewed_at`,
`runs_seen`, `marker_flags[]` (`run`, `kind` = `wrong_id|wrong_center|false_positive|ok_confirmed`,
`marker` = `{index, id, center}` at flag time, optional `correct_id`, `remark`),
`pins[]` (`missed_marker` with `x, y, true_id, remark`; `note` with `x, y, text`;
image coordinates of the tile). Top level: `general_remarks[]`.

## Keys

Press `?` in the page (the list is generated from the same table that binds the
keys, `app/src/model/keymap.ts`). Main ones: `j`/`k` tiles, `1/2/3` verdict,
`w c f o` flag selected marker, `u` clear, `m` missed pin (or shift-click), `n`
note pin, `v` compare, `d` differences only, `x` swap A/B, `[`/`]` confidence threshold, `/` remark, `0` fit,
`.` 1:1, `+`/`-` zoom. Gain and gamma are in the stage toolbar; gamma above 1
lifts the shadows.

Tests: `python -m unittest tools.tests.test_review_server` (HTTP tests need
Pillow); in `app/`: `bun run typecheck`, `bun run lint`, `bun run test`.
