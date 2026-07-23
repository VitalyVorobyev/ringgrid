// ringgrid interactive demo — WASM driver + pixel-accurate overlay engine.
//
// Everything runs client-side: the page loads the wasm-pack bundle, builds a
// RinggridDetector for the selected target, and renders the DetectionResult on
// a device-pixel-ratio-aware overlay canvas.
//
// Coordinate convention: ringgrid reports centers/ellipses where an integer
// coordinate is a PIXEL CENTER. On a canvas, drawImage(img, 0, 0) places that
// pixel's center at (x + 0.5, y + 0.5). Every positional overlay therefore maps
// image coord `c` to display coord `(c + 0.5) * displayScale` — the `+0.5`
// removes the classic half-pixel overlay offset.

import init, {
  RinggridDetector,
  target_fiducial_dots_mm,
  version,
} from './pkg/ringgrid_wasm.js';

const DPR = Math.max(1, window.devicePixelRatio || 1);
const $ = (id) => document.getElementById(id);

const els = {};
[
  'runBtn', 'autoRun', 'status', 'timing', 'target', 'gallery', 'fileInput',
  'galleryCaption', 'showEllipses', 'showCenters', 'showLabels', 'showOrigin',
  'resultSub', 'stats', 'viewportWrap', 'viewport', 'cvImage', 'cvOverlay',
  'tooltip', 'version',
].forEach((id) => (els[id] = $(id)));

// Targets are data-driven: every sample carries its own `target` spec, so the
// demo covers all six {hex,rect}×{coded,plain}×{dots,no dots} combinations with
// no per-target WASM helpers. Specs are indexed by their `name`; one detector is
// built and cached per distinct target.
const targetJson = {};   // name -> target spec JSON string

// ── State ───────────────────────────────────────────────────────────
let detectors = {};   // target name -> RinggridDetector
let specs = {};       // target name -> parsed target spec
let fiducialDots = {}; // target name -> derived origin dots [[x_mm, y_mm], ...]
let manifest = [];
let activeSample = null;
let img = null;       // { rgba: Uint8Array, w, h, imageData }
let result = null;    // last parsed DetectionResult
let displayScale = 1;
let runSeq = 0;

// ── Boot ────────────────────────────────────────────────────────────
async function boot() {
  try {
    await init();
    els.version.textContent = 'v' + version();
    manifest = await fetch('./samples.json').then((r) => r.json()).then((d) => d.samples);
    registerTargets();
    buildGallery();
    wireControls();
    await selectSample(manifest[0]);
    els.runBtn.disabled = false;
    setStatus('Ready.');
  } catch (e) {
    setStatus('Failed to initialise: ' + e, true);
    console.error(e);
  }
}

function setStatus(msg, isError = false) {
  els.status.textContent = msg;
  els.status.classList.toggle('error', isError);
}

// Collect the distinct target specs carried by the samples, index them by name,
// and populate the target <select> (used when detecting an uploaded image).
function registerTargets() {
  const options = new Map(); // name -> label (first sample wins)
  for (const s of manifest) {
    const spec = s.target;
    if (!spec || !spec.name) continue;
    if (!(spec.name in targetJson)) {
      targetJson[spec.name] = JSON.stringify(spec);
      specs[spec.name] = spec;
      // Dot positions are derived from the lattice, not stored in the spec —
      // ask the library rather than duplicating the placement rule here.
      try {
        fiducialDots[spec.name] = JSON.parse(target_fiducial_dots_mm(targetJson[spec.name]));
      } catch {
        fiducialDots[spec.name] = [];
      }
    }
    if (!options.has(spec.name)) options.set(spec.name, s.label || spec.name);
  }
  els.target.innerHTML = '';
  for (const [name, label] of options) {
    const o = document.createElement('option');
    o.value = name;
    o.textContent = label;
    els.target.appendChild(o);
  }
}

function getDetector(name) {
  if (!detectors[name]) {
    detectors[name] = new RinggridDetector(targetJson[name]);
  }
  return detectors[name];
}

// ── Gallery & image loading ─────────────────────────────────────────
function buildGallery() {
  els.gallery.innerHTML = '';
  for (const s of manifest) {
    const b = document.createElement('button');
    b.className = 'thumb';
    b.dataset.id = s.id;
    b.innerHTML = `<img src="./${s.file}" alt="${s.label}" /><span class="cap">${s.label}</span>`;
    b.addEventListener('click', () => selectSample(s));
    els.gallery.appendChild(b);
  }
  const up = document.createElement('button');
  up.className = 'thumb upload';
  up.innerHTML = `<span class="plus">+</span><span>Upload</span>`;
  up.addEventListener('click', () => els.fileInput.click());
  els.gallery.appendChild(up);

  els.fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    activeSample = null;
    markActiveThumb(null);
    const url = URL.createObjectURL(file);
    await loadFromURL(url);
    URL.revokeObjectURL(url);
    els.galleryCaption.textContent =
      `Uploaded: ${file.name} — detecting with the selected target (${els.target.value}).`;
    maybeRun(true);
  });
}

function markActiveThumb(id) {
  els.gallery.querySelectorAll('.thumb').forEach((t) => t.classList.toggle('active', t.dataset.id === id));
}

async function selectSample(sample) {
  activeSample = sample;
  markActiveThumb(sample.id);
  els.galleryCaption.textContent = sample.caption || '';
  if (sample.target && sample.target.name) els.target.value = sample.target.name;
  await loadFromURL('./' + sample.file);
  maybeRun(true);
}

function loadFromURL(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const w = image.naturalWidth, h = image.naturalHeight;
      const off = document.createElement('canvas');
      off.width = w; off.height = h;
      const octx = off.getContext('2d', { willReadFrequently: true });
      octx.drawImage(image, 0, 0);
      const imageData = octx.getImageData(0, 0, w, h);
      img = { rgba: new Uint8Array(imageData.data), w, h, imageData };
      onImageLoaded();
      resolve();
    };
    image.onerror = reject;
    image.src = url;
  });
}

function onImageLoaded() {
  result = null;
  els.cvImage.width = img.w; els.cvImage.height = img.h;
  els.cvImage.getContext('2d').putImageData(img.imageData, 0, 0);
  sizeViewport();
  clearOverlay();
  renderStats();
}

// ── Viewport sizing (DPR-aware overlay) ─────────────────────────────
function sizeViewport() {
  if (!img) return;
  const avail = els.viewportWrap.clientWidth - 2;
  displayScale = Math.min(1.5, avail / img.w); // never upscale small samples past 1.5×
  const dispW = img.w * displayScale;
  const dispH = img.h * displayScale;
  els.viewport.style.width = dispW + 'px';
  els.viewport.style.height = dispH + 'px';
  els.cvImage.style.width = dispW + 'px';
  els.cvImage.style.height = dispH + 'px';
}

function prepOverlay() {
  const cssW = img.w * displayScale, cssH = img.h * displayScale;
  const c = els.cvOverlay;
  c.width = Math.round(cssW * DPR);
  c.height = Math.round(cssH * DPR);
  c.style.width = cssW + 'px';
  c.style.height = cssH + 'px';
  const ctx = c.getContext('2d');
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  return ctx;
}
function clearOverlay() {
  const ctx = els.cvOverlay.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, els.cvOverlay.width, els.cvOverlay.height);
}

const TX = (ix) => (ix + 0.5) * displayScale;
const TY = (iy) => (iy + 0.5) * displayScale;
const TR = (r) => r * displayScale;

// ── Detection run ───────────────────────────────────────────────────
let runScheduled = null;
function maybeRun(force = false) {
  if (!img) return;
  if (force || els.autoRun.checked) {
    clearTimeout(runScheduled);
    runScheduled = setTimeout(run, force ? 0 : 60);
  }
}

function run() {
  if (!img) return;
  const targetKey = els.target.value;
  try {
    const det = getDetector(targetKey);
    const t0 = performance.now();
    const json = det.detect_adaptive_rgba(img.rgba, img.w, img.h);
    const ms = performance.now() - t0;
    result = JSON.parse(json);
    els.timing.textContent = `adaptive · ${ms.toFixed(0)} ms`;
    renderOverlay();
    renderStats();
    runSeq++;
    const n = result.detected_markers.length;
    setStatus(`Detected ${n} marker${n === 1 ? '' : 's'}.`);
  } catch (e) {
    setStatus('' + e, true);
    console.error(e);
  }
}

// ── Overlay rendering ───────────────────────────────────────────────
function confColor(c) {
  const css = getComputedStyle(document.documentElement);
  if (c >= 0.7) return css.getPropertyValue('--good').trim() || '#1a9a54';
  if (c >= 0.4) return css.getPropertyValue('--warn').trim() || '#c67b12';
  return css.getPropertyValue('--bad').trim() || '#d1433a';
}
const ORIGIN_COLOR = '#8a5cf6';

function renderOverlay() {
  if (!img || !result) { clearOverlay(); return; }
  const ctx = prepOverlay();
  const showEllipses = els.showEllipses.checked;
  const showCenters = els.showCenters.checked;
  const showLabels = els.showLabels.checked;

  for (const m of result.detected_markers) {
    const color = confColor(m.confidence);
    const [cx, cy] = m.center;

    if (showEllipses && m.ellipse_outer) {
      const e = m.ellipse_outer;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.ellipse(TX(e.cx), TY(e.cy), TR(e.a), TR(e.b), e.angle, 0, Math.PI * 2);
      ctx.stroke();
    }

    if (showCenters) {
      const x = TX(cx), y = TY(cy);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(x - 4, y); ctx.lineTo(x + 4, y);
      ctx.moveTo(x, y - 4); ctx.lineTo(x, y + 4);
      ctx.stroke();
    }

    if (showLabels) {
      const label = m.id != null ? String(m.id)
        : m.grid_coord ? `${m.grid_coord[0]},${m.grid_coord[1]}`
        : '';
      if (label) drawLabel(ctx, TX(cx), TY(cy), label, color);
    }
  }

  if (els.showOrigin.checked) renderOrigin(ctx);
}

function drawLabel(ctx, x, y, text, color) {
  ctx.font = '600 10px ui-monospace, monospace';
  ctx.textAlign = 'center';
  const ty = y - 7;
  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(4,7,13,0.85)';
  ctx.strokeText(text, x, ty);
  ctx.fillStyle = color;
  ctx.fillText(text, x, ty);
  ctx.textAlign = 'start';
}

// Origin overlay for plain targets: project the derived fiducial dots (board mm)
// with the resolved homography and mark cell (0, 0), the central cell the dots
// surround. Only meaningful when the origin actually resolved
// (board_frame === "absolute").
function renderOrigin(ctx) {
  const spec = specs[els.target.value];
  const fid = spec && spec.fiducials;
  const dots = fiducialDots[els.target.value] || [];
  const resolved = result.board_frame === 'absolute';

  if (fid && dots.length && result.homography && resolved) {
    const H = result.homography;
    const r = fid.dot_radius_mm || 1.4;
    ctx.fillStyle = ORIGIN_COLOR;
    for (const [mx, my] of dots) {
      const c = applyH(H, mx, my);
      const edge = applyH(H, mx + r, my);
      const pr = Math.max(3, Math.hypot(edge[0] - c[0], edge[1] - c[1]) * displayScale);
      ctx.beginPath();
      ctx.arc(TX(c[0]), TY(c[1]), pr, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Cell (0, 0) is central on both lattices — it is the cell the origin-dot
  // triad surrounds, not a board corner.
  if (resolved) {
    const origin = result.detected_markers.find(
      (m) => m.grid_coord && m.grid_coord[0] === 0 && m.grid_coord[1] === 0,
    );
    if (origin) {
      const [ox, oy] = origin.center;
      ctx.strokeStyle = ORIGIN_COLOR;
      ctx.lineWidth = 2.2;
      const rr = origin.ellipse_outer ? TR(origin.ellipse_outer.a) + 5 : 12;
      ctx.beginPath();
      ctx.arc(TX(ox), TY(oy), rr, 0, Math.PI * 2);
      ctx.stroke();
      drawLabel(ctx, TX(ox), TY(oy) - rr + 4, 'cell (0,0)', ORIGIN_COLOR);
    }
  }
}

function applyH(H, x, y) {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  return [
    (H[0][0] * x + H[0][1] * y + H[0][2]) / w,
    (H[1][0] * x + H[1][1] * y + H[1][2]) / w,
  ];
}

// ── Stats chips ─────────────────────────────────────────────────────
function chip(text, kind) {
  return `<span class="chip${kind ? ' ' + kind : ''}">${text}</span>`;
}

function renderStats() {
  if (!result) { els.stats.innerHTML = chip('no detection yet'); return; }
  const markers = result.detected_markers;
  const decoded = markers.filter((m) => m.id != null).length;
  const labeled = markers.filter((m) => m.grid_coord).length;
  const hasH = !!result.homography;

  const frame = result.board_frame; // "absolute" | "relative_canonical" | undefined
  let originChip;
  if (frame === 'absolute') originChip = chip('origin: <b>resolved</b>', 'good');
  else if (frame === 'relative_canonical') originChip = chip('origin: <b>relative</b>', 'warn');
  else originChip = chip('origin: <b>—</b>');

  const idChip = decoded > 0
    ? chip(`decoded IDs: <b>${decoded}</b>`, 'good')
    : chip(`labeled cells: <b>${labeled}</b>`, labeled > 0 ? 'good' : '');

  // board_complete is Some(bool) only when a board frame was resolved; it is the
  // success criterion for plain, no-dots targets (whole board must be found).
  const bc = result.board_complete;
  const completeChip = bc == null ? ''
    : bc ? chip('board: <b>complete</b>', 'good')
         : chip('board: <b>partial</b>', 'warn');

  els.stats.innerHTML =
    chip(`markers: <b>${markers.length}</b>`, markers.length ? 'good' : 'bad') +
    idChip +
    chip(`homography: <b>${hasH ? 'yes' : 'no'}</b>`, hasH ? 'good' : 'warn') +
    originChip +
    completeChip;
}

// ── Hover inspector ─────────────────────────────────────────────────
function pick(clientX, clientY) {
  if (!result) return null;
  const rect = els.cvOverlay.getBoundingClientRect();
  const ix = (clientX - rect.left) / displayScale - 0.5;
  const iy = (clientY - rect.top) / displayScale - 0.5;
  let best = null, bestD = Infinity;
  for (const m of result.detected_markers) {
    const d = Math.hypot(m.center[0] - ix, m.center[1] - iy);
    const rr = m.ellipse_outer ? m.ellipse_outer.a + 4 : 8;
    if (d < rr && d < bestD) { bestD = d; best = m; }
  }
  return best;
}

function showTooltip(m, clientX, clientY) {
  const t = els.tooltip;
  if (!m) { t.style.display = 'none'; return; }
  const key = m.id != null ? `id ${m.id}` : m.grid_coord ? `cell (${m.grid_coord[0]}, ${m.grid_coord[1]})` : 'marker';
  const lines = [
    key,
    `center  ${m.center[0].toFixed(2)}, ${m.center[1].toFixed(2)}`,
    `conf    ${m.confidence.toFixed(3)}`,
  ];
  if (m.ellipse_outer) lines.push(`axes    ${m.ellipse_outer.a.toFixed(1)} × ${m.ellipse_outer.b.toFixed(1)}`);
  t.textContent = lines.join('\n');
  t.style.display = 'block';
  t.style.left = (clientX + 14) + 'px';
  t.style.top = (clientY + 14) + 'px';
}

// ── Controls ────────────────────────────────────────────────────────
function wireControls() {
  els.runBtn.addEventListener('click', () => run());
  els.target.addEventListener('change', () => maybeRun(true));
  for (const id of ['showEllipses', 'showCenters', 'showLabels', 'showOrigin']) {
    els[id].addEventListener('change', () => renderOverlay());
  }

  els.cvOverlay.addEventListener('mousemove', (e) => showTooltip(pick(e.clientX, e.clientY), e.clientX, e.clientY));
  els.cvOverlay.addEventListener('mouseleave', () => showTooltip(null));

  let rz;
  window.addEventListener('resize', () => {
    clearTimeout(rz);
    rz = setTimeout(() => { sizeViewport(); renderOverlay(); }, 120);
  });
}

// Debug hook for automated verification (harmless in normal use).
window.__demo = {
  get ready() { return !!img; },
  get result() { return result; },
  get runSeq() { return runSeq; },
  get displayScale() { return displayScale; },
  tx: TX, ty: TY, tr: TR,
  run,
  selectTarget(target) { els.target.value = target; },
  async selectSampleById(id) { await selectSample(manifest.find((s) => s.id === id)); },
};

boot();
