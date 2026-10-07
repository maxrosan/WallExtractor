/* WallExtractor editor. Vanilla JS + Konva. All geometry in pixels of the base render;
   scale.px_per_m converts to metres. State is a WallPlan JSON (see wallextractor/schema.py). */
(() => {
"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

// ------------------------------------------------------------------ state
const S = {
  token: localStorage.getItem("we_token") || "",
  filter: "pending", plans: [], pid: null, row: null, plan: null, prims: [],
  tool: "select", sel: null, undo: [], redo: [], dirty: false, saveTimer: null,
  draw: null, calib: null, grid: null, showPrims: false, snap: true, clip: null, picked: new Set(),
};
const COLORS = { wall: "rgba(192,57,43,0.55)", wallSel: "rgba(192,57,43,0.85)", door: "rgba(46,139,87,0.8)", window: "rgba(59,111,217,0.8)",
  handle: "#2457A6", snap: "#B7791F", prim: "rgba(36,87,166,0.35)", prim0: "rgba(36,87,166,0.18)" };

// ------------------------------------------------------------------ api
async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (S.token) headers["X-Token"] = S.token;
  if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(opts.json); }
  const r = await fetch("/api" + path, Object.assign({}, opts, { headers }));
  if (r.status === 401) { askToken(); throw new Error("token"); }
  if (!r.ok) { let d = ""; try { d = (await r.json()).detail; } catch (e) {} throw new Error(typeof d === "string" ? d : JSON.stringify(d) || r.statusText); }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r;
}
function askToken() { $("#token-modal").hidden = false; $("#token-input").focus(); }
$("#token-ok").addEventListener("click", () => { S.token = $("#token-input").value.trim(); localStorage.setItem("we_token", S.token); $("#token-modal").hidden = true; loadQueue(); });
$("#token-input").addEventListener("keydown", e => { if (e.key === "Enter") $("#token-ok").click(); });

// ------------------------------------------------------------------ geometry helpers
const dist = (a, b) => Math.hypot(b[0] - a[0], b[1] - a[1]);
const ppm = () => (S.plan && S.plan.scale && S.plan.scale.px_per_m) || null;
const fmtM = px => ppm() ? (px / ppm()).toFixed(2) + " m" : Math.round(px) + " px";
function axis(w) { const L = dist(w.start, w.end) || 1e-9; return { ux: (w.end[0] - w.start[0]) / L, uy: (w.end[1] - w.start[1]) / L, L }; }
function proj(w, p) { const { ux, uy } = axis(w); return (p[0] - w.start[0]) * ux + (p[1] - w.start[1]) * uy; }
function atT(w, t) { const { ux, uy } = axis(w); return [w.start[0] + ux * t, w.start[1] + uy * t]; }
function perpDist(w, p) { const { ux, uy } = axis(w); return Math.abs(-(p[0] - w.start[0]) * uy + (p[1] - w.start[1]) * ux); }
function wallById(id) { return S.plan.walls.find(w => w.id === id); }
function nextId(prefix, list) { let n = 1; const ids = new Set(list.map(x => x.id)); while (ids.has(prefix + n)) n++; return prefix + n; }
function nearestWall(p, tol) {
  let best = null;
  for (const w of S.plan.walls) {
    const t = proj(w, p); const { L } = axis(w);
    if (t < -tol || t > L + tol) continue;
    const d = perpDist(w, p);
    if (d <= Math.max(tol, w.thickness) && (!best || d < best.d)) best = { w, d, t };
  }
  return best;
}

// spatial grid over PDF primitives + wall endpoints for snapping
function buildGrid() {
  const cell = 60; const g = new Map();
  const put = (x, y, item) => { const k = Math.floor(x / cell) + "," + Math.floor(y / cell); if (!g.has(k)) g.set(k, []); g.get(k).push(item); };
  for (const p of S.prims) { put(p[0], p[1], { pt: [p[0], p[1]], kind: "prim" }); put(p[2], p[3], { pt: [p[2], p[3]], kind: "prim" });
    // register the segment in the cells its bbox covers (short segments only, long ones checked directly)
    const x0 = Math.min(p[0], p[2]), x1 = Math.max(p[0], p[2]), y0 = Math.min(p[1], p[3]), y1 = Math.max(p[1], p[3]);
    if ((x1 - x0) + (y1 - y0) < 600) for (let cx = Math.floor(x0 / cell); cx <= Math.floor(x1 / cell); cx++) for (let cy = Math.floor(y0 / cell); cy <= Math.floor(y1 / cell); cy++) { const k = cx + "," + cy; if (!g.has(k)) g.set(k, []); g.get(k).push({ seg: p, kind: "primseg" }); }
  }
  S.grid = { cell, g };
}
function snapPoint(p, opts = {}) {
  if (!S.snap) return { pt: p, kind: null };
  const tol = (opts.tol || 9) / stage.scaleX();
  let best = null;
  const consider = (pt, kind, w = 1) => { const d = dist(pt, p) * w; if (d <= tol && (!best || d < best.d)) best = { pt, kind, d }; };
  for (const w of S.plan.walls) { if (w.id === opts.skipWall) continue; consider(w.start, "wall", 0.8); consider(w.end, "wall", 0.8); }
  if (S.grid) {
    const { cell, g } = S.grid; const cx = Math.floor(p[0] / cell), cy = Math.floor(p[1] / cell);
    for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (const it of (g.get((cx + i) + "," + (cy + j)) || [])) {
      if (it.kind === "prim") consider(it.pt, "prim", 1.0);
      else if (it.kind === "primseg") { // projection onto the primitive line
        const s = it.seg; const w = { start: [s[0], s[1]], end: [s[2], s[3]] }; const t = proj(w, p); const { L } = axis(w);
        if (t >= 0 && t <= L) consider(atT(w, t), "primline", 1.3);
      }
    }
  }
  // orthogonal snap relative to an anchor (drawing a wall)
  if (opts.anchor) {
    const a = opts.anchor; const q = best ? best.pt : p; const dx = q[0] - a[0], dy = q[1] - a[1];
    const ang = Math.abs(Math.atan2(dy, dx)) * 180 / Math.PI;
    const near = v => Math.abs(ang - v) < 4;
    if (!best && (near(0) || near(180))) return { pt: [q[0], a[1]], kind: "ortho" };
    if (!best && near(90)) return { pt: [a[0], q[1]], kind: "ortho" };
  }
  return best ? { pt: best.pt, kind: best.kind } : { pt: p, kind: null };
}

// ------------------------------------------------------------------ stage
const stage = new Konva.Stage({ container: "stage", width: 10, height: 10, draggable: true });
const world = new Konva.Group();
const bgLayer = new Konva.Layer({ listening: false }); const primLayer = new Konva.Layer({ listening: false });
const wallLayer = new Konva.Layer(); const openLayer = new Konva.Layer(); const uiLayer = new Konva.Layer();
// one world transform shared by all layers: we scale/position the stage itself
stage.add(bgLayer, primLayer, wallLayer, openLayer, uiLayer);
let baseImg = null, tileImg = null, tileTimer = null;

function fit() {
  const wrap = $("#stage-wrap"); stage.width(wrap.clientWidth); stage.height(wrap.clientHeight);
  if (S.plan) { const k = Math.min(stage.width() / S.plan.image.width, stage.height() / S.plan.image.height) * 0.96;
    stage.scale({ x: k, y: k }); stage.position({ x: (stage.width() - S.plan.image.width * k) / 2, y: (stage.height() - S.plan.image.height * k) / 2 }); }
  onView();
}
window.addEventListener("resize", fit);
stage.on("wheel", e => {
  e.evt.preventDefault(); const old = stage.scaleX(); const p = stage.getPointerPosition();
  const k = Math.min(12, Math.max(0.05, old * (e.evt.deltaY > 0 ? 1 / 1.15 : 1.15)));
  stage.scale({ x: k, y: k }); stage.position({ x: p.x - (p.x - stage.x()) * (k / old), y: p.y - (p.y - stage.y()) * (k / old) });
  onView();
});
stage.on("dragend", onView);
function onView() {
  $("#zoom").textContent = Math.round(stage.scaleX() * 100) + "%";
  renderHandles();
  clearTimeout(tileTimer); tileTimer = setTimeout(refineTile, 350);
}
function worldPointer() { const p = stage.getPointerPosition(); if (!p) return null; return [(p.x - stage.x()) / stage.scaleX(), (p.y - stage.y()) / stage.scaleY()]; }

async function refineTile() {
  if (!S.pid || !S.plan) return; const k = stage.scaleX(); if (k <= 1.25) { if (tileImg) { tileImg.destroy(); tileImg = null; bgLayer.draw(); } return; }
  const x0 = Math.max(0, -stage.x() / k), y0 = Math.max(0, -stage.y() / k);
  const x1 = Math.min(S.plan.image.width, (stage.width() - stage.x()) / k), y1 = Math.min(S.plan.image.height, (stage.height() - stage.y()) / k);
  const w = x1 - x0, h = y1 - y0; if (w <= 0 || h <= 0) return;
  const s = Math.min(6, Math.max(1.5, k)); if (w * s > 4000 || h * s > 4000) return;
  const url = `/api/plans/${S.pid}/tile?x=${x0.toFixed(1)}&y=${y0.toFixed(1)}&w=${w.toFixed(1)}&h=${h.toFixed(1)}&s=${s.toFixed(2)}&token=${encodeURIComponent(S.token)}`;
  const img = new Image(); img.onload = () => { if (tileImg) tileImg.destroy(); tileImg = new Konva.Image({ image: img, x: x0, y: y0, width: w, height: h }); bgLayer.add(tileImg); bgLayer.draw(); };
  img.src = url;
}

// ------------------------------------------------------------------ rendering
function render() {
  wallLayer.destroyChildren(); openLayer.destroyChildren(); uiLayer.destroyChildren(); primLayer.destroyChildren();
  if (!S.plan) { [wallLayer, openLayer, uiLayer, primLayer].forEach(l => l.draw()); return; }
  const k = stage.scaleX();
  if (S.showPrims) { for (const p of S.prims) primLayer.add(new Konva.Line({ points: p, stroke: p[4] ? COLORS.prim : COLORS.prim0, strokeWidth: 1 / k })); }
  for (const w of S.plan.walls) {
    const sel = S.sel && S.sel.kind === "wall" && S.sel.id === w.id;
    const ln = new Konva.Line({ points: [...w.start, ...w.end], stroke: sel ? COLORS.wallSel : COLORS.wall, strokeWidth: w.thickness, lineCap: "butt",
      hitStrokeWidth: Math.max(w.thickness, 14 / k), draggable: S.tool === "select", id: "w:" + w.id });
    ln.on("mousedown touchstart", e => { if (S.tool === "select") { select("wall", w.id); e.cancelBubble = true; } });
    ln.on("dragstart", () => { hideHandles(); ln._orig = { s: [...w.start], e: [...w.end], ops: S.plan.openings.filter(o => o.wall_id === w.id).map(o => ({ o, s: [...o.start], e: [...o.end] })) }; });
    ln.on("dragmove", () => { const dx = ln.x(), dy = ln.y(); showMeasure(`Δ ${fmtM(Math.hypot(dx, dy))}`); });
    ln.on("dragend", () => { const dx = ln.x(), dy = ln.y(); ln.position({ x: 0, y: 0 }); pushUndo();
      w.start = [ln._orig.s[0] + dx, ln._orig.s[1] + dy]; w.end = [ln._orig.e[0] + dx, ln._orig.e[1] + dy];
      for (const { o, s, e } of ln._orig.ops) { o.start = [s[0] + dx, s[1] + dy]; o.end = [e[0] + dx, e[1] + dy]; }
      w.polygon = null; changed(); });
    wallLayer.add(ln);
  }
  for (const o of S.plan.openings) {
    const w = wallById(o.wall_id); const th = (w ? w.thickness : 8) + 6 / k;
    const sel = S.sel && S.sel.kind === "opening" && S.sel.id === o.id;
    const ln = new Konva.Line({ points: [...o.start, ...o.end], stroke: COLORS[o.type] || COLORS.door, strokeWidth: th, opacity: sel ? 1 : 0.85,
      hitStrokeWidth: Math.max(th, 14 / k), draggable: S.tool === "select", dash: o.confidence < 1 ? [6 / k, 4 / k] : undefined, id: "o:" + o.id });
    ln.on("mousedown touchstart", e => { if (S.tool === "select") { select("opening", o.id); e.cancelBubble = true; } });
    ln.on("dragstart", () => { hideHandles(); ln._orig = { s: [...o.start], e: [...o.end] }; });
    ln.on("dragmove", () => { // slide along the wall only
      if (!w) return; const dx = ln.x(), dy = ln.y(); const { ux, uy } = axis(w); const t = dx * ux + dy * uy; ln.position({ x: ux * t, y: uy * t }); showMeasure(`deslocado ${fmtM(Math.abs(t))}`); });
    ln.on("dragend", () => { const dx = ln.x(), dy = ln.y(); ln.position({ x: 0, y: 0 }); pushUndo();
      o.start = [ln._orig.s[0] + dx, ln._orig.s[1] + dy]; o.end = [ln._orig.e[0] + dx, ln._orig.e[1] + dy]; o.polygon = null; changed(); });
    const lbl = new Konva.Text({ x: (o.start[0] + o.end[0]) / 2, y: (o.start[1] + o.end[1]) / 2 - th / 2 - 14 / k, text: o.code || (o.type === "door" ? "P?" : "J?"),
      fontSize: 12 / k, fontStyle: "bold", fill: o.type === "door" ? "#1f6b40" : "#2a54b0", listening: false });
    lbl.offsetX(lbl.width() / 2);
    openLayer.add(ln, lbl);
  }
  [primLayer, wallLayer, openLayer].forEach(l => l.draw());
  renderHandles(); renderPanels();
}

function renderHandles() {
  uiLayer.destroyChildren(); if (!S.plan || !S.sel) { uiLayer.draw(); return; }
  const k = stage.scaleX(); const r = 6 / k;
  const item = S.sel.kind === "wall" ? wallById(S.sel.id) : S.plan.openings.find(o => o.id === S.sel.id);
  if (!item) { uiLayer.draw(); return; }
  for (const end of ["start", "end"]) {
    const h = new Konva.Circle({ x: item[end][0], y: item[end][1], radius: r, fill: "#fff", stroke: COLORS.handle, strokeWidth: 2 / k, draggable: true });
    h.on("dragstart", () => { h._snapshot = JSON.stringify(S.plan); });
    h.on("dragmove", () => {
      let p = [h.x(), h.y()];
      if (S.sel.kind === "wall") { const other = end === "start" ? item.end : item.start; const sn = snapPoint(p, { skipWall: item.id, anchor: other }); p = sn.pt; showSnap(sn); item[end] = p; item.polygon = null;
        showMeasure(`${fmtM(dist(item.start, item.end))} · espessura ${fmtM(item.thickness)}`); }
      else { const w = wallById(item.wall_id); if (w) p = atT(w, proj(w, p)); // along the wall line, also across a gap past its end
        const sn = snapPoint(p, { tol: 7 }); if (sn.kind === "prim" || sn.kind === "primline") { const w2 = wallById(item.wall_id); p = w2 ? atT(w2, proj(w2, sn.pt)) : sn.pt; showSnap(sn); } else showSnap(null);
        item[end] = p; item.width = dist(item.start, item.end); item.polygon = null; item.width_source = "editor"; showMeasure(`largura ${fmtM(item.width)}`); }
      h.position({ x: p[0], y: p[1] }); redrawItem(item);
    });
    h.on("dragend", () => { S.undo.push(h._snapshot); S.redo = []; changed(); });
    uiLayer.add(h);
  }
  uiLayer.draw();
}
function hideHandles() { uiLayer.destroyChildren(); uiLayer.batchDraw(); }
function redrawItem(item) {
  const node = stage.findOne("#" + (S.sel.kind === "wall" ? "w:" : "o:") + item.id); if (node) node.points([...item.start, ...item.end]);
  if (S.sel.kind === "wall") for (const o of S.plan.openings.filter(o => o.wall_id === item.id)) { // keep openings on the moved axis
    const t0 = proj(item, o.start), t1 = proj(item, o.end); o.start = atT(item, t0); o.end = atT(item, t1); const n = stage.findOne("#o:" + o.id); if (n) n.points([...o.start, ...o.end]); }
  wallLayer.batchDraw(); openLayer.batchDraw();
}
let snapMark = null;
function showSnap(sn) { if (snapMark) { snapMark.destroy(); snapMark = null; } if (!sn || !sn.kind) { uiLayer.batchDraw(); return; }
  const k = stage.scaleX(); snapMark = new Konva.Rect({ x: sn.pt[0] - 5 / k, y: sn.pt[1] - 5 / k, width: 10 / k, height: 10 / k, stroke: COLORS.snap, strokeWidth: 2 / k, listening: false }); uiLayer.add(snapMark); uiLayer.batchDraw(); }
function showMeasure(t) { $("#measure").textContent = t || ""; }

// ------------------------------------------------------------------ selection + editing
// selecting only restyles the existing nodes: rebuilding them on mousedown would destroy the node the
// user is pressing and Konva would never start dragging it
function select(kind, id) {
  const next = kind ? { kind, id } : null;
  if ((next && S.sel && next.kind === S.sel.kind && next.id === S.sel.id) || (!next && !S.sel)) return;
  S.sel = next; if (!S.plan) return;
  for (const w of S.plan.walls) { const n = stage.findOne("#w:" + w.id); if (n) n.stroke(S.sel && S.sel.kind === "wall" && S.sel.id === w.id ? COLORS.wallSel : COLORS.wall); }
  for (const o of S.plan.openings) { const n = stage.findOne("#o:" + o.id); if (n) n.opacity(S.sel && S.sel.kind === "opening" && S.sel.id === o.id ? 1 : 0.85); }
  wallLayer.batchDraw(); openLayer.batchDraw(); renderHandles(); renderPanels();
}
function pushUndo() { S.undo.push(JSON.stringify(S.plan)); if (S.undo.length > 80) S.undo.shift(); S.redo = []; }
function undo() { if (!S.undo.length) return; S.redo.push(JSON.stringify(S.plan)); S.plan = JSON.parse(S.undo.pop()); S.sel = null; changed(); }
function redo() { if (!S.redo.length) return; S.undo.push(JSON.stringify(S.plan)); S.plan = JSON.parse(S.redo.pop()); S.sel = null; changed(); }
function changed() { S.dirty = true; setSave("Alterações não salvas…", "dirty"); clearTimeout(S.saveTimer); S.saveTimer = setTimeout(save, 1500); render(); }
function setSave(t, cls) { const el = $("#save-status"); el.textContent = t; el.className = "small " + (cls || "muted"); }
async function save() {
  if (!S.pid || !S.plan || !S.dirty) return;
  try { await api(`/plans/${S.pid}/annotation`, { method: "PUT", json: S.plan }); S.dirty = false; setSave("Salvo " + new Date().toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" })); }
  catch (e) { setSave("Erro ao salvar: " + e.message, "err"); }
}
function deleteSel() {
  if (!S.sel) return; pushUndo();
  if (S.sel.kind === "wall") { S.plan.walls = S.plan.walls.filter(w => w.id !== S.sel.id); S.plan.openings = S.plan.openings.filter(o => o.wall_id !== S.sel.id); }
  else S.plan.openings = S.plan.openings.filter(o => o.id !== S.sel.id);
  S.sel = null; changed();
}
function splitSel() {
  if (!S.sel || S.sel.kind !== "wall") return; const w = wallById(S.sel.id); if (!w) return; pushUndo();
  const mid = atT(w, axis(w).L / 2); const w2 = { ...w, id: nextId("w", S.plan.walls), start: mid, end: [...w.end], polygon: null }; w.end = mid; w.polygon = null;
  for (const o of S.plan.openings.filter(o => o.wall_id === w.id)) if (proj(w, [(o.start[0] + o.end[0]) / 2, (o.start[1] + o.end[1]) / 2]) > axis(w).L) o.wall_id = w2.id;
  S.plan.walls.push(w2); changed();
}

// rotation: deg > 0 is clockwise on screen (y points down)
function rotPt(p, c, deg) { const a = deg * Math.PI / 180, cs = Math.cos(a), sn = Math.sin(a), dx = p[0] - c[0], dy = p[1] - c[1];
  return [c[0] + dx * cs - dy * sn, c[1] + dx * sn + dy * cs]; }
function pivotOf(item, pivot) { return pivot === "start" || pivot === "end" ? [...item[pivot]] : [(item.start[0] + item.end[0]) / 2, (item.start[1] + item.end[1]) / 2]; }
// the wall an opening belongs to: parallel within 5°, on the wall's line, touching or overlapping its extent
// (the extractor splits walls at openings, so a door usually sits in the gap between two collinear walls).
// overlap = share of the opening covered by solid wall; a real opening is mostly in a gap.
function wallUnder(o) {
  const mid = [(o.start[0] + o.end[0]) / 2, (o.start[1] + o.end[1]) / 2]; const L = dist(o.start, o.end) || 1e-9;
  const ox = (o.end[0] - o.start[0]) / L, oy = (o.end[1] - o.start[1]) / L; const tol = 12 / stage.scaleX();
  let best = null, covered = 0;
  for (const w of S.plan.walls) { const { ux, uy, L: wl } = axis(w); const band = Math.max(tol, w.thickness);
    if (Math.abs(ox * ux + oy * uy) < Math.cos(5 * Math.PI / 180)) continue;
    const d = perpDist(w, mid); if (d > band) continue;
    const t0 = proj(w, o.start), t1 = proj(w, o.end), lo = Math.min(t0, t1), hi = Math.max(t0, t1);
    const apart = Math.max(0, lo - wl, -hi); if (apart > band) continue;
    covered += Math.max(0, Math.min(hi, wl) - Math.max(lo, 0));
    if (!best || d + apart < best.d + best.apart) best = { w, d, apart };
  }
  return best && { w: best.w, d: best.d, overlap: Math.min(1, covered / L) };
}
function attachOpening(o) {
  const hit = wallUnder(o); o.polygon = null;
  if (!hit) { o.wall_id = null; return false; }
  const w = hit.w; o.wall_id = w.id; o.start = atT(w, proj(w, o.start)); o.end = atT(w, proj(w, o.end)); o.width = dist(o.start, o.end); return true;
}
function rotateSel(deg, pivot = "center") {
  if (!S.sel || !deg) return; const item = selItem(); if (!item) return; pushUndo();
  const c = pivotOf(item, pivot);
  if (S.sel.kind === "wall") {
    item.start = rotPt(item.start, c, deg); item.end = rotPt(item.end, c, deg); item.polygon = null;
    for (const o of S.plan.openings.filter(o => o.wall_id === item.id)) { o.start = rotPt(o.start, c, deg); o.end = rotPt(o.end, c, deg); o.polygon = null; }
  } else {
    item.start = rotPt(item.start, c, deg); item.end = rotPt(item.end, c, deg);
    hint(attachOpening(item) ? "Abertura girada e encaixada na parede." : "Abertura girada; não está sobre nenhuma parede.");
  }
  changed();
}
// the door leaf case: the machine drew the door along the open leaf. Try ±90° around each end (the hinge)
// and the centre; keep the candidate that lands in a wall gap (least solid wall under it), then the closest.
function rotateToWall(prefer) {
  if (!S.sel || S.sel.kind !== "opening") return; const o = selItem(); if (!o) return;
  const other = prefer === "start" ? "end" : "start"; let best = null;
  for (const pivot of [prefer || "start", other, "center"]) for (const deg of [90, -90]) {
    const c = pivotOf(o, pivot); const cand = { start: rotPt(o.start, c, deg), end: rotPt(o.end, c, deg) }; const hit = wallUnder(cand);
    if (!hit) continue;
    if (!best || hit.overlap < best.hit.overlap - 0.05 || (Math.abs(hit.overlap - best.hit.overlap) <= 0.05 && hit.d < best.hit.d - 1e-6)) best = { pivot, deg, hit };
  }
  if (!best) { hint("Nenhuma parede encontrada girando 90° nas pontas ou no centro."); return; }
  rotateSel(best.deg, best.pivot);
}
function selItem() { return S.sel && (S.sel.kind === "wall" ? wallById(S.sel.id) : S.plan.openings.find(o => o.id === S.sel.id)); }

// copy / paste. The clipboard keeps the item and the scale it was copied at, so a door copied from one
// plan pastes with the same size in metres into another.
function copySel() {
  const item = selItem(); if (!item) return;
  S.clip = { kind: S.sel.kind, item: JSON.parse(JSON.stringify(item)), ppm: ppm() };
  hint(`${S.sel.kind === "wall" ? "Parede" : item.type === "door" ? "Porta" : "Janela"} copiada. Ctrl+V cola onde está o mouse.`);
}
// the wall to paste an opening on: the pointer on its line, inside it or in a gap past its end
function wallAt(p, half) {
  const tol = 12 / stage.scaleX(); let best = null;
  for (const w of S.plan.walls) { const band = Math.max(tol, w.thickness); const d = perpDist(w, p); if (d > band) continue;
    const t = proj(w, p), apart = Math.max(0, t - axis(w).L, -t); if (apart > half + band) continue;
    if (!best || d + apart < best.d + best.apart) best = { w, d, apart, t }; }
  return best;
}
function paste(p) {
  if (!S.clip || !S.plan) return; p = p || viewCenter();
  const src = S.clip.item; const f = S.clip.ppm && ppm() ? ppm() / S.clip.ppm : 1;
  const L = dist(src.start, src.end) * f, ux = (src.end[0] - src.start[0]) / (dist(src.start, src.end) || 1), uy = (src.end[1] - src.start[1]) / (dist(src.start, src.end) || 1);
  pushUndo();
  if (S.clip.kind === "wall") {
    const w = { ...src, id: nextId("w", S.plan.walls), start: [p[0] - ux * L / 2, p[1] - uy * L / 2], end: [p[0] + ux * L / 2, p[1] + uy * L / 2], thickness: src.thickness * f, polygon: null };
    S.plan.walls.push(w); S.sel = { kind: "wall", id: w.id };
  } else {
    const o = { ...src, id: nextId("o", S.plan.openings), width: L, polygon: null, width_source: src.width_source || "editor" };
    const hit = wallAt(p, L / 2);
    if (hit) { o.wall_id = hit.w.id; o.start = atT(hit.w, hit.t - L / 2); o.end = atT(hit.w, hit.t + L / 2); hint("Colada na parede sob o mouse."); }
    else { o.wall_id = null; o.start = [p[0] - ux * L / 2, p[1] - uy * L / 2]; o.end = [p[0] + ux * L / 2, p[1] + uy * L / 2]; hint("Colada solta: nenhuma parede sob o mouse. Arraste ou cole sobre uma parede."); }
    S.plan.openings.push(o); S.sel = { kind: "opening", id: o.id };
  }
  changed();
}
// a wall is duplicated beside itself; an opening next to itself on the same wall
function duplicateSel() {
  const it = selItem(); if (!it) return; const keep = S.clip; copySel();
  const L = dist(it.start, it.end), g = 20 / stage.scaleX(); const mid = [(it.start[0] + it.end[0]) / 2, (it.start[1] + it.end[1]) / 2];
  const ux = (it.end[0] - it.start[0]) / (L || 1), uy = (it.end[1] - it.start[1]) / (L || 1);
  paste(S.sel.kind === "wall" ? [mid[0] - uy * (it.thickness + g), mid[1] + ux * (it.thickness + g)] : [mid[0] + ux * (L + g), mid[1] + uy * (L + g)]);
  S.clip = keep; hint("Duplicado.");
}
function viewCenter() { const k = stage.scaleX(); return [(stage.width() / 2 - stage.x()) / k, (stage.height() / 2 - stage.y()) / k]; }
function pointerInView() { const p = stage.getPointerPosition(); return p && p.x >= 0 && p.y >= 0 && p.x <= stage.width() && p.y <= stage.height() ? worldPointer() : null; }

// context menu (right click on a wall or opening, or on an empty spot to paste)
const ctx = $("#ctx"); let ctxEnd = "start", ctxMark = null, ctxAt = null;
function closeCtx() { ctx.hidden = true; if (ctxMark) { ctxMark.destroy(); ctxMark = null; uiLayer.batchDraw(); } }
stage.on("contextmenu", e => {
  e.evt.preventDefault(); closeCtx(); if (!S.plan) return;
  const id = e.target && e.target.id && e.target.id(); const onItem = !!id && /^[wo]:/.test(id); ctxAt = worldPointer();
  let mode = "empty";
  if (onItem) {
    select(id[0] === "w" ? "wall" : "opening", id.slice(2)); const item = selItem(); const p = ctxAt; if (!item || !p) return; mode = S.sel.kind;
    ctxEnd = dist(p, item.start) <= dist(p, item.end) ? "start" : "end";
    const k = stage.scaleX(); ctxMark = new Konva.Circle({ x: item[ctxEnd][0], y: item[ctxEnd][1], radius: 8 / k, stroke: COLORS.snap, strokeWidth: 3 / k, listening: false });
    uiLayer.add(ctxMark); uiLayer.batchDraw();
  } else if (!S.clip) return;
  // data-only lists the modes an entry shows in; entries without it show on walls and openings
  $$("#ctx > *").forEach(el => { const only = el.dataset.only ? el.dataset.only.split(" ") : ["wall", "opening"]; el.hidden = !only.includes(mode); });
  const pb = $("#ctx [data-act=paste]"); pb.hidden = !S.clip;
  if (S.clip) pb.firstChild.textContent = "Colar aqui (" + (S.clip.kind === "wall" ? "parede" : (S.clip.item.code || (S.clip.item.type === "door" ? "porta" : "janela"))) + ") ";
  const wrap = $("#stage-wrap").getBoundingClientRect(); ctx.hidden = false;
  ctx.style.left = Math.min(e.evt.clientX - wrap.left, wrap.width - ctx.offsetWidth - 4) + "px";
  ctx.style.top = Math.min(e.evt.clientY - wrap.top, wrap.height - ctx.offsetHeight - 4) + "px";
});
ctx.addEventListener("contextmenu", e => e.preventDefault());
ctx.addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return; const end = ctxEnd; closeCtx();
  if (b.dataset.act === "copy") return copySel();
  if (b.dataset.act === "dup") return duplicateSel();
  if (b.dataset.act === "paste") return paste(ctxAt);
  if (b.dataset.act === "del") return deleteSel();
  if (b.dataset.act === "split") return splitSel();
  if (b.dataset.rot === "fit") return rotateToWall(end);
  const pivot = b.dataset.pivot === "end" ? end : "center";
  if (b.dataset.rot === "ask") { const v = parseFloat(String(prompt("Ângulo em graus (positivo gira no sentido horário):", "90") || "").replace(",", ".")); if (Number.isFinite(v)) rotateSel(v, pivot); return; }
  rotateSel(parseFloat(b.dataset.rot), pivot);
});
document.addEventListener("mousedown", e => { if (!ctx.hidden && !ctx.contains(e.target)) closeCtx(); });
stage.on("wheel dragstart", closeCtx);

// ------------------------------------------------------------------ drawing tools
stage.on("mousedown touchstart", e => {
  const p = worldPointer(); if (!p || !S.plan || (e.evt && e.evt.button === 2)) return;
  if (S.tool === "select") { if (e.target === stage) select(null); return; }
  if (S.tool === "wall") {
    const sn = snapPoint(p, S.draw ? { anchor: S.draw.p1 } : {});
    if (!S.draw) { S.draw = { p1: sn.pt }; }
    else { const p2 = sn.pt; if (dist(S.draw.p1, p2) > 3) { pushUndo();
        const th = medianThickness(); S.plan.walls.push({ id: nextId("w", S.plan.walls), start: S.draw.p1, end: p2, thickness: th, kind: null, polygon: null }); }
      S.draw = null; showMeasure(""); changed(); }
    return;
  }
  if (S.tool === "door" || S.tool === "window") {
    stage.draggable(false);
    const hit = nearestWall(p, 12 / stage.scaleX());
    // on a wall: slide along its axis (past its ends too, into the gap the extractor left for the opening);
    // off any wall: free segment between two points, snapped to wall ends and PDF lines, attached on release
    if (hit) { const sn = snapPoint(p, { tol: 7 }); const t0 = sn.kind ? proj(hit.w, sn.pt) : hit.t; showSnap(sn); S.draw = { wall: hit.w, t0, t1: t0, type: S.tool }; }
    else { const sn = snapPoint(p); showSnap(sn); S.draw = { p1: sn.pt, p2: sn.pt, type: S.tool }; }
    return;
  }
  if (S.tool === "calib") {
    if (!S.calib) S.calib = { p1: p }; else { S.calib.p2 = p; $("#calib-box").hidden = false; $("#calib-dist").focus(); showMeasure(`${dist(S.calib.p1, p).toFixed(0)} px marcados`); }
  }
});
stage.on("mousemove touchmove", () => {
  const p = worldPointer(); if (!p || !S.plan || !S.draw) return;
  const k = stage.scaleX();
  if (S.tool === "wall" && S.draw.p1) { const sn = snapPoint(p, { anchor: S.draw.p1 }); showSnap(sn); preview([...S.draw.p1, ...sn.pt], medianThickness(), COLORS.wall); showMeasure(fmtM(dist(S.draw.p1, sn.pt))); }
  else if ((S.tool === "door" || S.tool === "window") && S.draw.wall) {
    const w = S.draw.wall; let t = proj(w, p);
    // snap the opening length to the schedule widths and to primitive endpoints
    const m = ppm(); if (m) { for (const wd of codeWidths(S.draw.type)) { const tt = S.draw.t0 + Math.sign(t - S.draw.t0) * wd * m; if (Math.abs(tt - t) < 6 / k) t = tt; } }
    const sn = snapPoint(atT(w, t), { tol: 7 }); if (sn.kind) { t = proj(w, sn.pt); showSnap(sn); } else showSnap(null);
    S.draw.t1 = t; preview([...atT(w, S.draw.t0), ...atT(w, t)], w.thickness + 6 / k, COLORS[S.draw.type]); showMeasure(`${S.draw.type === "door" ? "porta" : "janela"} ${fmtM(Math.abs(t - S.draw.t0))}`);
  }
  else if ((S.tool === "door" || S.tool === "window") && S.draw.p1) {
    const sn = snapPoint(p, { anchor: S.draw.p1 }); showSnap(sn); S.draw.p2 = sn.pt;
    preview([...S.draw.p1, ...sn.pt], medianThickness() + 6 / k, COLORS[S.draw.type]); showMeasure(`${S.draw.type === "door" ? "porta" : "janela"} ${fmtM(dist(S.draw.p1, sn.pt))}`);
  }
});
stage.on("mouseup touchend", () => {
  if (!S.draw || !(S.tool === "door" || S.tool === "window")) return;
  const d = S.draw; S.draw = null; stage.draggable(true); clearPreview(); showSnap(null);
  const len = d.wall ? Math.abs(d.t1 - d.t0) : dist(d.p1, d.p2); const m = ppm(); if (len < (m ? 0.15 * m : 8)) { showMeasure(""); return; }
  pushUndo();
  const o = { id: nextId("o", S.plan.openings), type: d.type, width: len, code: defaultCode(d.type, len), wall_id: null,
    polygon: null, confidence: 1, width_source: "editor", height_m: null, sill_m: null, kind: null };
  if (d.wall) { const a = Math.min(d.t0, d.t1), b = Math.max(d.t0, d.t1); o.start = atT(d.wall, a); o.end = atT(d.wall, b); o.wall_id = d.wall.id; }
  else { o.start = d.p1; o.end = d.p2; if (!attachOpening(o)) hint("Abertura criada solta: não está na linha de nenhuma parede."); }
  S.plan.openings.push(o); S.sel = { kind: "opening", id: o.id }; changed();
});
let previewNode = null;
function preview(points, width, color) { if (!previewNode) { previewNode = new Konva.Line({ stroke: color, strokeWidth: width, opacity: 0.6, listening: false }); uiLayer.add(previewNode); } previewNode.points(points); previewNode.stroke(color); previewNode.strokeWidth(width); uiLayer.batchDraw(); }
function clearPreview() { if (previewNode) { previewNode.destroy(); previewNode = null; uiLayer.batchDraw(); } }
function medianThickness() { const t = S.plan.walls.map(w => w.thickness).sort((a, b) => a - b); if (t.length) return t[t.length >> 1]; const m = ppm(); return m ? 0.13 * m : 8; }
function codeWidths(type) { const out = []; for (const [c, v] of Object.entries(S.plan.codes || {})) if (v.type === type && v.width_m) out.push(v.width_m); return out; }
function defaultCode(type, lenPx) { const m = ppm(); let best = null; for (const [c, v] of Object.entries(S.plan.codes || {})) { if (v.type !== type || !v.width_m || !m) continue; const d = Math.abs(v.width_m * m - lenPx); if (!best || d < best.d) best = { c, d }; } return best && best.d < 0.08 * m ? best.c : null; }
function hint(t) { $("#hint").textContent = t; }

// calibration
$("#calib-ok").addEventListener("click", () => {
  const d = parseFloat($("#calib-dist").value); if (!S.calib || !S.calib.p2 || !(d > 0)) return;
  pushUndo(); S.plan.scale = S.plan.scale || {}; S.plan.scale.px_per_m = dist(S.calib.p1, S.calib.p2) / d; S.plan.scale.method = "manual";
  S.calib = null; $("#calib-box").hidden = true; setTool("select"); changed();
});

// ------------------------------------------------------------------ tools + keys
function setTool(t) { S.tool = t; S.draw = null; S.calib = null; clearPreview(); showSnap(null); stage.draggable(true);
  $$("#toolbar .tool").forEach(b => b.classList.toggle("on", b.dataset.tool === t));
  $("#stage").style.cursor = t === "select" ? "default" : "crosshair";
  hint({ select: "Clique para selecionar, arraste para mover. Pontas: redimensionar. Botão direito: girar, copiar. Ctrl+C / Ctrl+V: copiar e colar no mouse. Roda do mouse: zoom.", wall: "Clique no início e no fim da parede. Encaixa nas linhas do PDF e em 0/90°.",
    door: "Arraste sobre uma parede, ou de um ponto a outro no vão entre paredes. Encaixa nas pontas e nas linhas do PDF.", window: "Arraste sobre uma parede, ou de um ponto a outro no vão entre paredes. Encaixa nas pontas e nas linhas do PDF.",
    calib: "Clique em dois pontos com distância conhecida e informe a medida." }[t]); render(); }
$$("#toolbar .tool").forEach(b => b.addEventListener("click", () => setTool(b.dataset.tool)));
$("#undo").addEventListener("click", undo); $("#redo").addEventListener("click", redo);
$("#show-prims").addEventListener("change", e => { S.showPrims = e.target.checked; render(); });
$("#snap").addEventListener("change", e => { S.snap = e.target.checked; });
document.addEventListener("keydown", e => {
  if (e.target.matches("input, select, textarea")) return;
  const k = e.key.toLowerCase();
  if (e.ctrlKey || e.metaKey) { // Ctrl combos first, so Ctrl+C / Ctrl+V / Ctrl+D do not switch tools
    if (k === "c") copySel();
    else if (k === "v") paste(pointerInView());
    else if (k === "d") duplicateSel();
    else if (k === "z") e.shiftKey ? redo() : undo();
    else if (k === "y") redo();
    else if (k === "s") save();
    else return;
    e.preventDefault(); return;
  }
  if (k === "v") setTool("select"); else if (k === "w") setTool("wall"); else if (k === "d") setTool("door"); else if (k === "j") setTool("window"); else if (k === "c") setTool("calib");
  else if (k === "r" && S.sel) { rotateSel(e.shiftKey ? -90 : 90); closeCtx(); }
  else if (k === "escape") { closeCtx(); $("#calib-box").hidden = true; select(null); setTool("select"); } // setTool also cancels any drawing in progress
  else if (k === "delete" || k === "backspace") { deleteSel(); e.preventDefault(); }
  else if ((k === "[" || k === "]") && S.sel && S.sel.kind === "wall") { const w = wallById(S.sel.id); const m = ppm() || 100; pushUndo(); w.thickness = Math.max(2, w.thickness + (k === "]" ? 1 : -1) * 0.01 * m); changed(); }
});

// ------------------------------------------------------------------ panels
function renderPanels() {
  const p = S.plan; if (!p) return;
  const m = ppm();
  $("#sel-none").hidden = !!S.sel; $("#sel-wall").hidden = !(S.sel && S.sel.kind === "wall"); $("#sel-open").hidden = !(S.sel && S.sel.kind === "opening");
  if (S.sel && S.sel.kind === "wall") { const w = wallById(S.sel.id); if (w) { $("#w-thick").value = m ? Math.round(w.thickness / m * 100) : Math.round(w.thickness); $("#w-len").textContent = fmtM(dist(w.start, w.end)); $("#w-kind").value = w.kind || ""; } }
  if (S.sel && S.sel.kind === "opening") { const o = p.openings.find(x => x.id === S.sel.id); if (o) { $("#o-type").value = o.type; $("#o-code").value = o.code || ""; $("#o-width").value = m ? (o.width / m).toFixed(2) : Math.round(o.width); $("#o-conf").textContent = (o.confidence < 1 ? "só etiqueta" : "geometria") + (o.width_source ? " · largura: " + o.width_source : ""); } }
  // codes
  const codes = p.codes || (p.codes = {}); for (const o of p.openings) if (o.code && !codes[o.code]) codes[o.code] = { type: o.type, width_m: m ? +(o.width / m).toFixed(2) : null };
  $("#codes-list").innerHTML = Object.keys(codes).sort().map(c => `<option value="${c}">`).join("");
  $("#codes").innerHTML = `<tr><th>Código</th><th>Tipo</th><th>Largura (m)</th><th></th></tr>` + Object.entries(codes).sort().map(([c, v]) =>
    `<tr><td class="mono">${c}</td><td><span class="pill ${v.type}">${v.type === "door" ? "porta" : "janela"}</span></td><td><input type="number" step="0.05" data-code="${c}" value="${v.width_m ?? ""}"></td><td><button data-apply="${c}">aplicar</button></td></tr>`).join("");
  $$("#codes input").forEach(i => i.addEventListener("change", () => { codes[i.dataset.code].width_m = parseFloat(i.value) || null; changed(); }));
  $$("#codes button").forEach(b => b.addEventListener("click", () => applyCode(b.dataset.apply)));
  // openings table
  $("#open-count").textContent = p.openings.length ? `(${p.openings.length})` : "";
  $("#openings").innerHTML = `<tr><th>#</th><th>Código</th><th>Tipo</th><th>Largura</th></tr>` + p.openings.map((o, i) =>
    `<tr data-oid="${o.id}" class="${S.sel && S.sel.kind === "opening" && S.sel.id === o.id ? "on" : ""}"><td class="mono">${i + 1}</td><td class="mono">${o.code || "—"}</td><td><span class="pill ${o.type}">${o.type === "door" ? "porta" : "janela"}</span></td><td class="mono">${m ? (o.width / m).toFixed(2) : Math.round(o.width)}${o.confidence < 1 ? " ?" : ""}</td></tr>`).join("");
  $$("#openings tr[data-oid]").forEach(tr => tr.addEventListener("click", () => { select("opening", tr.dataset.oid); focusOn(p.openings.find(o => o.id === tr.dataset.oid)); }));
  $("#scale-info").textContent = m ? `${m.toFixed(2)} px/m · ${p.scale.method || ""}` : "sem escala: use a ferramenta Escala";
  $("#questions").innerHTML = [...(p.questions || []), ...((S.row && S.row.meta && S.row.meta.notes) || [])].map(q => `<li>${esc(q)}</li>`).join("") || "<li class='muted'>nenhuma</li>";
}
function applyCode(c) { const v = S.plan.codes[c]; const m = ppm(); if (!v || !v.width_m || !m) return; pushUndo();
  for (const o of S.plan.openings) if (o.code === c) { const cx = (o.start[0] + o.end[0]) / 2, cy = (o.start[1] + o.end[1]) / 2; const w = wallById(o.wall_id); const { ux, uy } = w ? axis(w) : { ux: (o.end[0] - o.start[0]) / (o.width || 1), uy: (o.end[1] - o.start[1]) / (o.width || 1) };
    const half = v.width_m * m / 2; o.start = [cx - ux * half, cy - uy * half]; o.end = [cx + ux * half, cy + uy * half]; o.width = v.width_m * m; o.width_source = "answer"; o.polygon = null; }
  changed(); }
function focusOn(o) { if (!o) return; const k = Math.max(stage.scaleX(), 2.5); const cx = (o.start[0] + o.end[0]) / 2, cy = (o.start[1] + o.end[1]) / 2; stage.scale({ x: k, y: k }); stage.position({ x: stage.width() / 2 - cx * k, y: stage.height() / 2 - cy * k }); onView(); }
function esc(s) { return String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

$("#w-thick").addEventListener("change", e => { const w = wallById(S.sel && S.sel.id); const m = ppm(); if (!w) return; pushUndo(); w.thickness = m ? parseFloat(e.target.value) / 100 * m : parseFloat(e.target.value); changed(); });
$("#w-kind").addEventListener("change", e => { const w = wallById(S.sel && S.sel.id); if (!w) return; pushUndo(); w.kind = e.target.value || null; changed(); });
$("#w-split").addEventListener("click", splitSel); $("#w-del").addEventListener("click", deleteSel); $("#o-del").addEventListener("click", deleteSel);
$("#o-type").addEventListener("change", e => { const o = S.plan.openings.find(x => x.id === S.sel.id); if (!o) return; pushUndo(); o.type = e.target.value; changed(); });
$("#o-code").addEventListener("change", e => { const o = S.plan.openings.find(x => x.id === S.sel.id); if (!o) return; pushUndo(); o.code = e.target.value.trim().toUpperCase() || null; changed(); });
$("#o-width").addEventListener("change", e => { const o = S.plan.openings.find(x => x.id === S.sel.id); const m = ppm(); const v = parseFloat(e.target.value); if (!o || !(v > 0)) return; pushUndo();
  const px = m ? v * m : v; const cx = (o.start[0] + o.end[0]) / 2, cy = (o.start[1] + o.end[1]) / 2; const w = wallById(o.wall_id); const { ux, uy } = w ? axis(w) : axis(o);
  o.start = [cx - ux * px / 2, cy - uy * px / 2]; o.end = [cx + ux * px / 2, cy + uy * px / 2]; o.width = px; o.width_source = "editor"; o.polygon = null; changed(); });
$("#title").addEventListener("change", e => { if (S.pid) api(`/plans/${S.pid}/title`, { method: "POST", json: { title: e.target.value } }).then(loadQueue).catch(() => {}); });

// ------------------------------------------------------------------ queue + loading
async function loadQueue() {
  try { S.plans = await api("/plans" + (S.filter ? "?status=" + S.filter : "")); } catch (e) { return; }
  $("#queue").innerHTML = S.plans.map(p => `<li data-id="${p.id}" class="${p.id === S.pid ? "on" : ""}"><div class="t" title="${esc(p.file)}"><input type="checkbox" class="pick" data-id="${p.id}" ${S.picked.has(p.id) ? "checked" : ""} title="Selecionar para o lote">${esc(p.title)}</div>
    <div class="m">${p.walls} paredes · ${p.openings} aberturas${p.questions ? " · " + p.questions + " dúvida(s)" : ""} <span class="st ${p.status}">${{ pending: "pendente", corrected: "corrigida", skipped: "pulada" }[p.status]}</span>${jobBadge(p.id)}</div></li>`).join("") || "<li class='muted small'>Fila vazia.</li>";
  $$("#queue li[data-id]").forEach(li => li.addEventListener("click", () => openPlan(li.dataset.id)));
  $$("#queue input.pick").forEach(b => { b.addEventListener("click", e => e.stopPropagation());
    b.addEventListener("change", () => { if (b.checked) S.picked.add(b.dataset.id); else S.picked.delete(b.dataset.id); pickedUpdate(); }); });
  pickedUpdate();
}
$$(".filters button").forEach(b => b.addEventListener("click", () => { S.filter = b.dataset.f; $$(".filters button").forEach(x => x.classList.toggle("on", x === b)); loadQueue(); }));

async function openPlan(pid) {
  if (S.dirty) await save();
  const row = await api(`/plans/${pid}`); S.row = row; S.pid = pid; S.plan = row.corrected || JSON.parse(JSON.stringify(row.machine));
  S.plan.codes = S.plan.codes || {}; S.prims = (row.meta && row.meta.primitives) || []; S.sel = null; S.undo = []; S.redo = []; S.dirty = false; buildGrid();
  $("#title").value = row.title; $("#plan-meta").textContent = `${row.file} · página ${row.page} · ${row.status}` + (row.corrected ? " · com correção salva" : "");
  setSave(row.corrected ? "Correção carregada" : "Saída da máquina (ainda sem correção)");
  const img = new Image(); img.onload = () => { if (baseImg) baseImg.destroy(); if (tileImg) { tileImg.destroy(); tileImg = null; } baseImg = new Konva.Image({ image: img }); bgLayer.add(baseImg); bgLayer.draw(); fit(); };
  img.src = `/api/plans/${pid}/image?token=${encodeURIComponent(S.token)}`;
  setTool("select"); loadQueue();
  if (row.ai_job) S.jobs[pid] = row.ai_job;
  claudePanel(); claudeAutoApply(row.ai_job);
}
async function markStatus(status) {
  if (!S.pid) return; S.dirty = true; await save();
  await api(`/plans/${S.pid}/status`, { method: "POST", json: { status } });
  const cur = S.pid; await loadQueue();
  const next = S.plans.find(p => p.status === "pending" && p.id !== cur); if (next) openPlan(next.id); else { hint("Fila de pendentes vazia."); }
}
// ------------------------------------------------------------------ AI help (Claude, ChatGPT...): package out, correction in
function aiStatus(t, cls) { const el = $("#ai-status"); el.textContent = t; el.className = "small " + (cls || "muted"); }
// A package takes a few seconds (a batch, longer): a modal shows each step (save, build on the server, receive with
// progress) and ends with a real "Baixar" link, so the download never depends on a script click the browser may drop.
const AI = { url: null };
function aiStep(name, state, extra) {
  const li = $(`#ai-steps li[data-step="${name}"]`); li.className = state || "";
  if (extra !== undefined) { const x = $("[data-extra]", li); if (x) x.textContent = extra; }
}
function aiFail(step, msg) { aiStep(step, "fail"); const e = $("#ai-err"); e.textContent = "Falhou: " + msg; e.hidden = false; }
$("#ai-close").addEventListener("click", () => { $("#ai-modal").hidden = true; });
$("#ai-copy").addEventListener("click", async () => {
  const t = $("#ai-prompt"); let ok = false;
  try { await navigator.clipboard.writeText(t.value); ok = true; } catch (e) { t.focus(); t.select(); ok = document.execCommand("copy"); }
  $("#ai-copy").textContent = ok ? "Copiado ✓" : "Selecione e copie (Ctrl+C)";
});
async function aiDownload(title, url, opts, fallbackName, nextSteps) {
  if (AI.url) { URL.revokeObjectURL(AI.url); AI.url = null; }
  $("#ai-title").textContent = title; $("#ai-next").innerHTML = nextSteps;
  $$("#ai-steps li").forEach(li => { li.className = ""; const x = $("[data-extra]", li); if (x) x.textContent = ""; });
  $("#ai-ready").hidden = true; $("#ai-err").hidden = true; $("#ai-modal").hidden = false;
  // 1. save the open plan, so the package carries the last edits
  if (S.dirty) { aiStep("save", "active"); await save(); if (S.dirty) { aiFail("save", "não consegui salvar a planta aberta"); return null; } }
  aiStep("save", "done", "");
  // 2. build: the server answers only when the zip is ready
  aiStep("build", "active", "0 s"); const t0 = Date.now();
  const tick = setInterval(() => aiStep("build", "active", `${Math.round((Date.now() - t0) / 1000)} s`), 500);
  let r;
  try { r = await fetch(url, Object.assign({}, opts, { headers: Object.assign({ "X-Token": S.token }, (opts && opts.headers) || {}) })); }
  catch (err) { clearInterval(tick); aiFail("build", "sem conexão com o servidor"); return null; }
  clearInterval(tick);
  if (r.status === 401) { askToken(); aiFail("build", "token inválido"); return null; }
  if (!r.ok) { let d = ""; try { d = (await r.json()).detail; } catch (e) {} aiFail("build", d || `o servidor respondeu ${r.status}`); return null; }
  aiStep("build", "done", `${((Date.now() - t0) / 1000).toFixed(1)} s`);
  // 3. receive, with progress when the size is known
  aiStep("recv", "active"); const total = Number(r.headers.get("content-length")) || 0; const parts = []; let got = 0;
  try {
    const reader = r.body.getReader();
    for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); got += value.length;
      aiStep("recv", "active", total ? `${Math.round(100 * got / total)}%` : `${Math.round(got / 1024)} KB`); }
  } catch (err) { aiFail("recv", "a conexão caiu durante o download"); return null; }
  aiStep("recv", "done", got > 2 * 1024 * 1024 ? `${(got / 1048576).toFixed(1)} MB` : `${Math.round(got / 1024)} KB`);
  // 4. ready
  const name = (/filename="([^"]+)"/.exec(r.headers.get("content-disposition") || "") || [])[1] || fallbackName;
  AI.url = URL.createObjectURL(new Blob(parts, { type: "application/zip" }));
  const a = $("#ai-dl"); a.href = AI.url; a.download = name; a.textContent = `Baixar ${name}`;
  let prompt = ""; try { prompt = decodeURIComponent(r.headers.get("x-ai-prompt") || ""); } catch (e) {}
  $("#ai-prompt").value = prompt; $(".prompt-box").hidden = !prompt; $("#ai-copy").textContent = "Copiar texto";
  aiStep("done", "done"); $("#ai-ready").hidden = false; a.focus();
  return name;
}
$("#ai-get").addEventListener("click", async () => {
  if (!S.pid) { aiStatus("Abra uma planta da fila primeiro.", "err"); return; }
  const name = await aiDownload("Pacote para a IA", `/api/plans/${S.pid}/ai`, {}, `ia_${S.pid}.zip`,
    `Próximos passos: envie o zip ao Claude ou ao ChatGPT com o texto do <span class="mono">PROMPT.txt</span>
     (se o chat não abrir zip, descompacte e anexe os arquivos). Depois use <b>Importar correção</b> (arquivo) ou
     <b>Colar resposta</b> (texto do chat).`);
  if (name) aiStatus(`Pacote ${name} pronto. Envie à IA com o PROMPT.txt e importe o correcao.json (ou cole a resposta).`);
});

// Models answer in slightly different shapes; accept the reasonable ones. Text: a ```json fence or the outermost {...}/[...].
function parseAnswer(text) {
  const fence = /```(?:json)?\s*([\s\S]*?)```/i.exec(text);
  let body = (fence ? fence[1] : text).trim();
  const open = body.search(/[{[]/); if (open < 0) throw new Error("não achei JSON no texto");
  const close = body[open] === "[" ? body.lastIndexOf("]") : body.lastIndexOf("}");
  if (close < open) throw new Error("não achei JSON no texto");
  try { return JSON.parse(body.slice(open, close + 1)); } catch (e) { throw new Error("JSON inválido: " + e.message); }
}
// a segment as {start, end} | {x1, y1, x2, y2} | [x1, y1, x2, y2(, t)] | {p1/a, p2/b}
function segOf(x) {
  if (Array.isArray(x) && x.length >= 4) return { start: [x[0], x[1]], end: [x[2], x[3]], t: x[4] };
  if (x && x.x1 !== undefined) return { start: [x.x1, x.y1], end: [x.x2, x.y2], t: x.thickness ?? x.espessura };
  const s = x && (x.start || x.inicio || x.p1 || x.a), e = x && (x.end || x.fim || x.p2 || x.b);
  return { start: s, end: e, t: x && (x.thickness ?? x.espessura) };
}
function typeOf(o, bucket) {
  const t = String((o && (o.type || o.tipo)) || bucket || "").toLowerCase();
  if (/^(window|janela|j)/.test(t) || /^j\d/i.test(String(o && o.code || ""))) return "window";
  return "door";
}
function planMedianT(plan) { const t = plan.walls.map(w => w.thickness).sort((a, b) => a - b); if (t.length) return t[t.length >> 1];
  const m = plan.scale && plan.scale.px_per_m; return m ? 0.13 * m : 8; }
function looksLikePlan(d) { return d && typeof d === "object" && !Array.isArray(d) && (d.walls || d.paredes || d.correcao || d.plan); }
// One answer -> walls and openings in the pixels of `plan` (a different image size is rescaled; ids made unique).
function normalizeAnswer(d, plan) {
  if (d && !d.walls && (d.paredes || d.correcao || d.plan)) d = d.correcao || d.plan || { ...d, walls: d.paredes, openings: d.aberturas };
  const rawOpen = d && (d.openings || d.aberturas ||
    (d.doors || d.windows ? [...(d.doors || []).map(x => ({ x, b: "door" })), ...(d.windows || []).map(x => ({ x, b: "window" }))] : null));
  if (!d || !Array.isArray(d.walls) || !Array.isArray(rawOpen)) throw new Error("a resposta precisa ter walls e openings");
  const k = d.image && Number(d.image.width) > 0 ? plan.image.width / Number(d.image.width) : 1;
  const pt = p => { if (!Array.isArray(p) || p.length < 2 || !isFinite(p[0]) || !isFinite(p[1])) throw new Error("ponto inválido: " + JSON.stringify(p)); return [Number(p[0]) * k, Number(p[1]) * k]; };
  const walls = [], ids = new Map(), tDefault = planMedianT(plan);
  for (const w of d.walls) {
    const sg = segOf(w); const id = nextId("w", walls); if (w && w.id !== undefined) ids.set(String(w.id), id);
    walls.push({ id, start: pt(sg.start), end: pt(sg.end), thickness: Number(sg.t) > 0 ? Number(sg.t) * k : tDefault, kind: (w && w.kind) || null, polygon: null });
  }
  const openings = [];
  for (const item of rawOpen) {
    const o = item && item.x !== undefined && item.b ? item.x : item, bucket = item && item.b;
    const sg = segOf(o); const start = pt(sg.start), end = pt(sg.end);
    openings.push({ id: nextId("o", openings), type: typeOf(o, bucket), start, end, width: dist(start, end),
      wall_id: (o && o.wall_id !== undefined && ids.get(String(o.wall_id))) || null, polygon: null, confidence: 1,
      code: (o && (o.code || o.codigo)) || null, width_source: "ai", height_m: (o && o.height_m) ?? null, sill_m: (o && o.sill_m) ?? null,
      kind: (o && o.kind) || null });
  }
  return { walls, openings, notes: d.notas || d.notes || null };
}
// Opening with no (known) wall: link it to the wall on whose line it lies, as on the canvas, and put it on that line.
function linkOpening(plan, o) {
  const mid = [(o.start[0] + o.end[0]) / 2, (o.start[1] + o.end[1]) / 2]; const L = dist(o.start, o.end) || 1e-9;
  const ox = (o.end[0] - o.start[0]) / L, oy = (o.end[1] - o.start[1]) / L; let best = null;
  for (const w of plan.walls) { const { ux, uy, L: wl } = axis(w); const band = Math.max(6, w.thickness);
    if (Math.abs(ox * ux + oy * uy) < Math.cos(5 * Math.PI / 180)) continue;
    const d = perpDist(w, mid); if (d > band) continue;
    const t0 = proj(w, o.start), t1 = proj(w, o.end), apart = Math.max(0, Math.min(t0, t1) - wl, -Math.max(t0, t1)); if (apart > band) continue;
    if (!best || d + apart < best.d + best.apart) best = { w, d, apart }; }
  if (!best) { o.wall_id = null; return; }
  const w = best.w; o.wall_id = w.id; o.start = atT(w, proj(w, o.start)); o.end = atT(w, proj(w, o.end)); o.width = dist(o.start, o.end);
}
// Checks on an imported answer, as warnings only (the reviewer decides).
// Signed deviation (degrees) of a segment from the nearest of horizontal/vertical, in (-45, 45].
function skewOf(x) { const a = Math.atan2(x.end[1] - x.start[1], x.end[0] - x.start[0]) * 180 / Math.PI; return ((a % 90) + 135) % 90 - 45; }
function planWarnings(plan) {
  const out = [], W = plan.image.width, H = plan.image.height, t = planMedianT(plan);
  const off = p => p[0] < 0 || p[1] < 0 || p[0] > W || p[1] > H;
  // Tilts under 1° are scanning drift. When many walls are tilted, the image itself is askew or in
  // perspective (photo of a sheet): one notice with the range replaces a warning per wall. Only a few
  // tilted walls among straight ones are suspicious and listed one by one.
  const tilted = [...plan.walls, ...plan.openings].filter(x => Math.abs(skewOf(x)) >= 1 && Math.abs(skewOf(x)) < 10);
  const nWallsTilted = plan.walls.filter(x => Math.abs(skewOf(x)) >= 1).length;
  const askew = plan.walls.length >= 4 && nWallsTilted >= 0.4 * plan.walls.length;
  if (askew) { const ang = tilted.map(x => Math.abs(skewOf(x))); out.push(`imagem inclinada ou em perspectiva: ${tilted.length} paredes/aberturas entre ${Math.min(...ang).toFixed(1)}° e ${Math.max(...ang).toFixed(1)}° (normal em foto ou escaneamento torto; confira só se algo destoar)`); }
  const tilt = x => { const a = Math.abs(skewOf(x)); return !askew && a >= 1 && a < 10 ? a : 0; };
  for (const w of plan.walls) {
    if (off(w.start) || off(w.end)) out.push(`parede ${w.id} fora da imagem`);
    const a = tilt(w); if (a) out.push(`parede ${w.id} torta (${a.toFixed(1)}°)`);
    if (dist(w.start, w.end) < 0.5 * t) out.push(`parede ${w.id} muito curta (${Math.round(dist(w.start, w.end))} px)`);
  }
  for (const o of plan.openings) {
    if (off(o.start) || off(o.end)) out.push(`abertura ${o.id} fora da imagem`);
    const a = tilt(o); if (a) out.push(`abertura ${o.id} torta (${a.toFixed(1)}°)`);
    if (!o.wall_id) out.push(`abertura ${o.id} sem parede na sua linha`);
    const L = dist(o.start, o.end) || 1e-9, ux = (o.end[0] - o.start[0]) / L, uy = (o.end[1] - o.start[1]) / L;
    for (const w of plan.walls) {
      const ax = axis(w); if (Math.abs(ux * ax.ux + uy * ax.uy) < Math.cos(5 * Math.PI / 180)) continue;
      if (perpDist(w, [(o.start[0] + o.end[0]) / 2, (o.start[1] + o.end[1]) / 2]) > Math.max(3, w.thickness / 2)) continue;
      const t0 = proj(w, o.start), t1 = proj(w, o.end), lo = Math.min(t0, t1), hi = Math.max(t0, t1);
      const over = Math.min(hi, ax.L) - Math.max(lo, 0);
      if (over > Math.max(3, 0.25 * w.thickness)) out.push(`parede ${w.id} passa por cima do vão ${o.id} (${Math.round(over)} px)`);
    }
  }
  return out;
}
function warnHtml(warn) {
  return warn.length ? `<div class="warn-list"><b>${warn.length} aviso(s) para conferir:</b><ul>${warn.slice(0, 12).map(w => `<li>${esc(w)}</li>`).join("")}` +
    `${warn.length > 12 ? `<li>… e mais ${warn.length - 12}</li>` : ""}</ul></div>` : "";
}
function importOpen(d, from, statusSel = "#ai-status") {
  if (looksLikePlan(d) && d.plan_id && d.plan_id !== S.pid &&
      !confirm(`Esta resposta é da planta ${d.plan_id}, não da aberta (${S.pid}). Importar mesmo assim?`)) return;
  if (!looksLikePlan(d)) { const mine = batchEntries(d, from).find(e => e.pid === S.pid); if (!mine) throw new Error("a resposta não tem walls e openings (para várias plantas use Importar lote)"); d = mine.d; }
  const got = normalizeAnswer(d, S.plan);
  pushUndo(); S.plan.walls = got.walls; S.plan.openings = got.openings;
  for (const o of S.plan.openings) if (!o.wall_id) linkOpening(S.plan, o);
  S.sel = null; changed();
  const warn = planWarnings(S.plan); const el = $(statusSel); el.className = "small";
  el.innerHTML = `Importado ${esc(from)}: ${got.walls.length} paredes, ${got.openings.length} aberturas. Ctrl+Z desfaz.` +
    (got.notes ? `<br>Notas: ${esc(got.notes)}` : "") + (warn.length ? warnHtml(warn) : `<br>Nenhum aviso na verificação automática.`);
}
$("#faces-get").addEventListener("click", async () => {
  if (!S.pid) { $("#faces-status").textContent = "Abra uma planta da fila primeiro."; return; }
  const btn = $("#faces-get"), el = $("#faces-status"); btn.disabled = true; el.className = "small muted";
  const t0 = Date.now(); const tick = setInterval(() => { el.textContent = `Lendo a imagem… ${Math.round((Date.now() - t0) / 1000)} s`; }, 300);
  try { const d = await api(`/plans/${S.pid}/faces`, { method: "POST" }); clearInterval(tick); importOpen(d, "detector de faces", "#faces-status"); }
  catch (err) { clearInterval(tick); el.textContent = "Não refeito: " + err.message; el.className = "small err"; }
  finally { btn.disabled = false; }
});
$("#ai-put").addEventListener("change", async e => {
  const f = e.target.files[0]; e.target.value = ""; if (!f || !S.pid) return;
  try { importOpen(parseAnswer(await f.text()), f.name); } catch (err) { aiStatus("Não importado: " + err.message, "err"); }
});
$("#ai-paste").addEventListener("click", () => { if (!S.pid) return; $("#paste-text").value = ""; $("#paste-modal").hidden = false; $("#paste-text").focus(); });
$("#paste-cancel").addEventListener("click", () => { $("#paste-modal").hidden = true; });
$("#paste-ok").addEventListener("click", () => {
  $("#paste-modal").hidden = true;
  try { importOpen(parseAnswer($("#paste-text").value), "texto colado"); } catch (err) { aiStatus("Não importado: " + err.message, "err"); }
});

// ------------------------------------------------------------------ batches: several plans out, several corrections in
function pickedUpdate() {
  const n = S.picked.size; $("#batch-get").textContent = `Baixar lote para IA (${n})`; $("#batch-get").disabled = !n;
  $("#claude-batch").textContent = `Corrigir com Claude (${n})`; $("#claude-batch").disabled = !n;
  $("#codex-batch").textContent = `Corrigir com ChatGPT (${n})`; $("#codex-batch").disabled = !n;
  const boxes = $$("#queue input.pick"); $("#pick-all").checked = boxes.length > 0 && boxes.every(b => b.checked);
}
$("#pick-all").addEventListener("change", e => { for (const b of $$("#queue input.pick")) { b.checked = e.target.checked;
  if (b.checked) S.picked.add(b.dataset.id); else S.picked.delete(b.dataset.id); } pickedUpdate(); });
$("#batch-get").addEventListener("click", async () => {
  const ids = [...S.picked]; if (!ids.length) return;
  const name = await aiDownload(`Lote de ${ids.length} planta(s) para a IA`, "/api/ai/batch",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids }) }, "ia_lote.zip",
    `Próximos passos: envie o zip ao ChatGPT ou ao Claude com o texto do <span class="mono">PROMPT_LOTE.txt</span> (na raiz do zip).
     A IA corrige uma planta depois da outra e devolve <span class="mono">correcoes.json</span>; se parar no meio, diga "continue".
     Depois use <b>Importar lote</b> aqui na fila. No fim ela entrega também <span class="mono">ferramentas.py</span> e
     <span class="mono">relatorio_lote.md</span> (o que se repetiu no lote): guarde para analisar. Se o relatório
     vier curto, peça na mesma conversa: "agora preencha o modelo do passo 6 do LEIA-ME_LOTE.md com detalhe".`);
  if (name) $("#batch-status").textContent = `Lote ${name} pronto.`;
});
// The answers of a batch: one combined object {plan_id: correction}, a list of corrections with plan_id, or one
// correction per file (plan_id inside, or in the file name as correcoes/<plan_id>.json).
function batchEntries(d, fname) {
  const fromName = (/([0-9a-f]{10})/.exec(fname || "") || [])[1];
  if (Array.isArray(d)) return d.map(x => ({ pid: x && x.plan_id, d: x }));
  if (looksLikePlan(d)) return [{ pid: d.plan_id || fromName, d }];
  if (d && typeof d === "object") {
    const inner = d.correcoes || d.plans || d.plantas; if (inner) return batchEntries(inner, fname);
    return Object.entries(d).filter(([, v]) => looksLikePlan(v)).map(([k, v]) => v.plan_id && v.plan_id !== k
      ? { pid: k, d: v, err: `a chave ${k} e o plan_id ${v.plan_id} de dentro não batem` } : { pid: k, d: v });
  }
  return [];
}
$("#batch-put").addEventListener("change", async e => {
  const files = [...e.target.files]; e.target.value = ""; if (!files.length) return;
  if (S.dirty) await save();
  const res = $("#batch-result"); $("#batch-modal").hidden = false; res.innerHTML = "<p class='small muted'>Lendo os arquivos…</p>";
  const entries = [];
  for (const f of files) { try { for (const x of batchEntries(parseAnswer(await f.text()), f.name)) entries.push({ ...x, file: f.name }); }
    catch (err) { entries.push({ pid: null, file: f.name, err: err.message }); } }
  const rows = []; let ok = 0; const seen = new Set();
  for (const x of entries) if (!x.err && x.pid) { if (seen.has(x.pid)) x.err = "planta repetida no lote (ficou a primeira)"; seen.add(x.pid); }
  for (const [i, x] of entries.entries()) {
    res.innerHTML = `<p class='small muted'>Importando ${i + 1}/${entries.length}…</p>`;
    const head = `<b>${esc(x.pid || "?")}</b> <span class="muted">(${esc(x.file)})</span>`;
    if (x.err) { rows.push(`<li class="err">${head}: ${esc(x.err)}</li>`); continue; }
    if (!x.pid) { rows.push(`<li class="err">${head}: sem plan_id</li>`); continue; }
    try {
      const row = await api(`/plans/${x.pid}`); const plan = row.corrected || JSON.parse(JSON.stringify(row.machine));
      const got = normalizeAnswer(x.d, plan); plan.walls = got.walls; plan.openings = got.openings;
      for (const o of plan.openings) if (!o.wall_id) linkOpening(plan, o);
      await api(`/plans/${x.pid}/annotation`, { method: "PUT", json: plan }); ok++;
      const warn = planWarnings(plan);
      rows.push(`<li>${head} ${esc(row.title)}: ${got.walls.length} paredes, ${got.openings.length} aberturas` +
        (got.notes ? `<div class="small muted">Notas: ${esc(got.notes)}</div>` : "") + warnHtml(warn) + `</li>`);
    } catch (err) { rows.push(`<li class="err">${head}: ${esc(err.message === "Not Found" ? "planta não está no editor" : err.message)}</li>`); }
  }
  res.innerHTML = `<p><b>${ok} de ${entries.length}</b> correção(ões) importada(s) como rascunho. As plantas continuam pendentes:
    abra cada uma, confira e marque como corrigida.</p><ul class="batch-list">${rows.join("")}</ul>`;
  $("#batch-status").textContent = `${ok} correção(ões) importada(s).`;
  await loadQueue(); if (S.pid && entries.some(x => x.pid === S.pid)) openPlan(S.pid);
});
$("#batch-close").addEventListener("click", () => { $("#batch-modal").hidden = true; });

// ------------------------------------------------------------------ AI agents on the user's computer
// The plans go to a queue on the server; scripts/claude_worker.py (on the user's computer) takes them one by one,
// runs `claude -p` (Claude) or `codex exec` (ChatGPT) on the AI package and posts the correction back. A finished correction is loaded into the plan
// as soon as it is open here (as an import: warnings shown, Ctrl+Z undoes), and stays a draft until reviewed.
S.jobs = {}; S.workers = []; S.jobTimer = null; S.queuePos = [];
const ENGINE = { claude: "Claude", codex: "ChatGPT" };
const eng = j => ENGINE[j.engine] || "Claude";
function jobBadge(pid) {
  const j = S.jobs[pid]; if (!j || j.status === "cancelled") return "";
  const t = { queued: `na fila do ${eng(j)}`, running: `${eng(j)} corrigindo`, error: `${eng(j)} falhou`,
    done: j.applied ? `${eng(j)} ✓` : `${eng(j)}: pronta` }[j.status];
  return ` <span class="st job ${j.status}" title="${esc(j.message || "")}">${esc(t)}</span>`;
}
function ago(t) { const s = Math.max(0, Math.round(Date.now() / 1000 - t)); return s < 90 ? `${s} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${Math.round(s / 3600)} h`; }
async function loadJobs() {
  clearTimeout(S.jobTimer);
  try {
    const d = await api("/ai/jobs"); S.jobs = {}; S.workers = d.workers || [];
    for (const j of d.jobs) if (!S.jobs[j.pid]) S.jobs[j.pid] = j;  // newest first: the first one is the latest
    S.queuePos = d.jobs.filter(j => j.status === "queued").sort((a, b) => a.id - b.id).map(j => j.pid);
  } catch (e) { S.jobTimer = setTimeout(loadJobs, 30000); return; }
  for (const li of $$("#queue li[data-id]")) {
    const m = $(".m", li); if (!m) continue; const old = $(".st.job", m); if (old) old.remove();
    m.insertAdjacentHTML("beforeend", jobBadge(li.dataset.id));
  }
  claudePanel();
  const cur = S.pid && S.jobs[S.pid];
  if (cur && cur.status === "done" && !cur.applied) { try { await claudeAutoApply((await api(`/plans/${S.pid}`)).ai_job); } catch (e) { /* next round */ } }
  const busy = Object.values(S.jobs).some(j => j.status === "queued" || j.status === "running");
  S.jobTimer = setTimeout(loadJobs, busy ? 8000 : 60000);
}
function claudePanel() {
  const j = S.pid && S.jobs[S.pid]; const el = $("#claude-status");
  const active = !!(j && (j.status === "queued" || j.status === "running"));
  $("#claude-cancel").hidden = !active; $("#claude-reapply").hidden = !(j && j.status === "done");
  $("#claude-one").disabled = $("#codex-one").disabled = !S.pid || active;
  const live = S.workers.filter(w => w.seen_s < 90);
  $("#claude-worker").innerHTML = live.length ? `Computador conectado: ${live.map(w => `${esc(w.name)} (${(w.engines || ["claude"]).map(e => ENGINE[e] || e).join(" e ")})`).join(", ")}.`
    : `<span class="err">Nenhum computador buscando trabalho agora.</span> No Lenovo, rode <span class="mono">python scripts/claude_worker.py --engines claude,codex</span> (ver docs/editor.md).`;
  if (j && j.status === "done" && el.dataset.job === String(j.id)) return;  // keeps the import warnings on screen
  el.className = "small muted"; delete el.dataset.job;
  if (!S.pid) el.textContent = "Abra uma planta da fila.";
  else if (!j) el.textContent = "O Claude Code ou o ChatGPT (Codex) do seu computador corrige a planta sozinho e devolve um rascunho para você revisar.";
  else if (j.status === "queued") { const k = S.queuePos.indexOf(S.pid) + 1; el.textContent = `Na fila${k ? ` (posição ${k})` : ""}, há ${ago(j.created)}.`; }
  else if (j.status === "running") el.textContent = `${eng(j)} corrigindo (${j.worker || "?"}): ${j.message || "…"} · última notícia há ${ago(j.updated)}.`;
  else if (j.status === "done") el.textContent = `${eng(j)} terminou há ${ago(j.updated)}${j.applied ? "; a correção já foi carregada nesta planta" : ""}. ${j.message || ""}`;
  else if (j.status === "error") { el.textContent = `${eng(j)} falhou: ` + (j.message || "sem detalhe") + ". Pode pedir de novo."; el.className = "small err"; }
  else el.textContent = "Cancelado. Pode pedir de novo.";
}
async function claudeAutoApply(job) {
  if (!job || job.status !== "done" || job.applied || !job.result || job.pid !== S.pid) return;
  const el = $("#claude-status"); importOpen(job.result, `do ${eng(job)}`, "#claude-status"); el.dataset.job = String(job.id);
  if (job.message) el.insertAdjacentHTML("beforeend", `<br><span class="muted">Do computador: ${esc(job.message)}</span>`);
  try { S.jobs[job.pid] = await api(`/ai/jobs/${job.id}`, { method: "POST", json: { applied: true } }); } catch (e) { /* loaded again next time */ }
}
async function claudeQueue(ids, engine) { const d = await api("/ai/jobs", { method: "POST", json: { ids, engine } }); await loadJobs(); return d; }
for (const engine of ["claude", "codex"]) {
  $(`#${engine}-one`).addEventListener("click", async () => {
    if (!S.pid) return; if (S.dirty) await save();
    try { const d = await claudeQueue([S.pid], engine); if (!d.added.length) $("#claude-status").textContent = d.skipped.map(x => x.why).join("; "); }
    catch (err) { $("#claude-status").textContent = "Não enfileirado: " + err.message; $("#claude-status").className = "small err"; }
  });
  $(`#${engine}-batch`).addEventListener("click", async () => {
    const ids = [...S.picked]; if (!ids.length) return; if (S.dirty) await save();
    try {
      const d = await claudeQueue(ids, engine);
      $("#batch-status").textContent = `${d.added.length} planta(s) na fila do ${ENGINE[engine]}` + (d.skipped.length ? `; ${d.skipped.length} já estava(m) na fila ou não existe(m)` : "") + ".";
    } catch (err) { $("#batch-status").textContent = "Não enfileirado: " + err.message; }
  });
}
$("#claude-cancel").addEventListener("click", async () => {
  const j = S.pid && S.jobs[S.pid]; if (!j) return;
  try { await api(`/ai/jobs/${j.id}`, { method: "POST", json: { status: "cancelled", message: "cancelado no editor" } }); } catch (e) { /* refreshed below */ }
  loadJobs();
});
$("#claude-reapply").addEventListener("click", async () => {
  const job = (await api(`/plans/${S.pid}`)).ai_job;
  if (job && job.result) { importOpen(job.result, `do ${eng(job)}`, "#claude-status"); $("#claude-status").dataset.job = String(job.id); }
});

$("#mark-corrected").addEventListener("click", () => markStatus("corrected"));
$("#mark-skip").addEventListener("click", () => markStatus("skipped"));
$("#upload").addEventListener("change", async e => {
  const files = [...e.target.files]; let n = 0;
  for (const f of files) { $("#upload-status").textContent = `Processando ${f.name} (${++n}/${files.length})…`;
    const fd = new FormData(); fd.append("file", f); fd.append("title", f.name.replace(/\.pdf$/i, ""));
    try { await api("/plans", { method: "POST", body: fd }); } catch (err) { $("#upload-status").textContent = `Falhou ${f.name}: ${err.message}`; await new Promise(r => setTimeout(r, 1500)); } }
  $("#upload-status").textContent = `${files.length} arquivo(s) na fila.`; e.target.value = ""; loadQueue();
});
window.addEventListener("beforeunload", e => { if (S.dirty) { save(); e.preventDefault(); e.returnValue = ""; } });

fit(); loadQueue().then(loadJobs);
})();
