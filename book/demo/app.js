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
  default_board_json,
  isra_rect_board_json,
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

// Board JSON + parsed spec, built lazily per target and cached (the spec is
// re-parsed from the wasm-provided JSON so origin fiducials stay a single
// source of truth with the Rust preset).
const BOARD_JSON = { hex: default_board_json, rect: isra_rect_board_json };

// ── State ───────────────────────────────────────────────────────────
let detectors = {};   // targetKey -> RinggridDetector
let specs = {};       // targetKey -> parsed target spec (for fiducials)
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

function getDetector(targetKey) {
  if (!detectors[targetKey]) {
    const json = BOARD_JSON[targetKey]();
    specs[targetKey] = JSON.parse(json);
    detectors[targetKey] = new RinggridDetector(json);
  }
  return detectors[targetKey];
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
  if (sample.target) els.target.value = sample.target;
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
  const mode = els.mode.value;
  try {
    const det = getDetector(targetKey);
    const t0 = performance.now();
    const json = mode === 'adaptive'
      ? det.detect_adaptive_rgba(img.rgba, img.w, img.h)
      : det.detect_rgba(img.rgba, img.w, img.h);
    const ms = performance.now() - t0;
    result = JSON.parse(json);
    els.timing.textContent = `${mode === 'adaptive' ? 'adaptive' : 'single-pass'} · ${ms.toFixed(0)} ms`;
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

// Origin overlay for plain targets: project the fiducial dots (board mm) with
// the resolved homography and mark the origin cell (grid coord [0,0]). Only
// meaningful when the origin actually resolved (board_frame === "absolute").
function renderOrigin(ctx) {
  const spec = specs[els.target.value];
  const fid = spec && spec.fiducials;
  const resolved = result.board_frame === 'absolute';

  if (fid && Array.isArray(fid.dots_mm) && result.homography && resolved) {
    const H = result.homography;
    const r = fid.dot_radius_mm || 1.4;
    ctx.fillStyle = ORIGIN_COLOR;
    for (const [mx, my] of fid.dots_mm) {
      const c = applyH(H, mx, my);
      const edge = applyH(H, mx + r, my);
      const pr = Math.max(3, Math.hypot(edge[0] - c[0], edge[1] - c[1]) * displayScale);
      ctx.beginPath();
      ctx.arc(TX(c[0]), TY(c[1]), pr, 0, Math.PI * 2);
      ctx.fill();
    }
  }

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
      drawLabel(ctx, TX(ox), TY(oy) - rr + 4, 'origin', ORIGIN_COLOR);
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

  els.stats.innerHTML =
    chip(`markers: <b>${markers.length}</b>`, markers.length ? 'good' : 'bad') +
    idChip +
    chip(`homography: <b>${hasH ? 'yes' : 'no'}</b>`, hasH ? 'good' : 'warn') +
    originChip;
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
  els.mode.addEventListener('change', () => maybeRun(true));
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
  selectTargetMode(target, mode) { els.target.value = target; els.mode.value = mode; },
  async selectSampleById(id) { await selectSample(manifest.find((s) => s.id === id)); },
};

boot();
