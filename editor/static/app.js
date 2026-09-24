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
  draw: null, calib: null, grid: null, showPrims: false, snap: true,
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
    ln.on("dragstart", () => { ln._orig = { s: [...w.start], e: [...w.end], ops: S.plan.openings.filter(o => o.wall_id === w.id).map(o => ({ o, s: [...o.start], e: [...o.end] })) }; });
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
    ln.on("dragstart", () => { ln._orig = { s: [...o.start], e: [...o.end] }; });
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
function select(kind, id) { S.sel = kind ? { kind, id } : null; render(); }
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

// context menu (right click on a wall or opening)
const ctx = $("#ctx"); let ctxEnd = "start", ctxMark = null;
function closeCtx() { ctx.hidden = true; if (ctxMark) { ctxMark.destroy(); ctxMark = null; uiLayer.batchDraw(); } }
stage.on("contextmenu", e => {
  e.evt.preventDefault(); closeCtx();
  const id = e.target && e.target.id && e.target.id(); if (!S.plan || !id || !/^[wo]:/.test(id)) return;
  select(id[0] === "w" ? "wall" : "opening", id.slice(2)); const item = selItem(); const p = worldPointer(); if (!item || !p) return;
  ctxEnd = dist(p, item.start) <= dist(p, item.end) ? "start" : "end";
  const k = stage.scaleX(); ctxMark = new Konva.Circle({ x: item[ctxEnd][0], y: item[ctxEnd][1], radius: 8 / k, stroke: COLORS.snap, strokeWidth: 3 / k, listening: false });
  uiLayer.add(ctxMark); uiLayer.batchDraw();
  $$("[data-only]", ctx).forEach(el => { el.hidden = el.dataset.only !== S.sel.kind; });
  const wrap = $("#stage-wrap").getBoundingClientRect(); ctx.hidden = false;
  ctx.style.left = Math.min(e.evt.clientX - wrap.left, wrap.width - ctx.offsetWidth - 4) + "px";
  ctx.style.top = Math.min(e.evt.clientY - wrap.top, wrap.height - ctx.offsetHeight - 4) + "px";
});
ctx.addEventListener("contextmenu", e => e.preventDefault());
ctx.addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return; const end = ctxEnd; closeCtx();
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
    const hit = nearestWall(p, 12 / stage.scaleX()); if (!hit) { hint("Arraste sobre uma parede para criar a abertura."); return; }
    stage.draggable(false); S.draw = { wall: hit.w, t0: hit.t, t1: hit.t, type: S.tool }; return;
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
    const w = S.draw.wall; let t = Math.max(0, Math.min(axis(w).L, proj(w, p)));
    // snap the opening length to the schedule widths and to primitive endpoints
    const m = ppm(); if (m) { for (const wd of codeWidths(S.draw.type)) { const tt = S.draw.t0 + Math.sign(t - S.draw.t0) * wd * m; if (Math.abs(tt - t) < 6 / k) t = tt; } }
    const sn = snapPoint(atT(w, t), { tol: 7 }); if (sn.kind) { t = Math.max(0, Math.min(axis(w).L, proj(w, sn.pt))); showSnap(sn); } else showSnap(null);
    S.draw.t1 = t; preview([...atT(w, S.draw.t0), ...atT(w, t)], w.thickness + 6 / k, COLORS[S.draw.type]); showMeasure(`${S.draw.type === "door" ? "porta" : "janela"} ${fmtM(Math.abs(t - S.draw.t0))}`);
  }
});
stage.on("mouseup touchend", () => {
  if (!S.draw || !(S.tool === "door" || S.tool === "window")) return;
  const d = S.draw; S.draw = null; stage.draggable(true); clearPreview(); showSnap(null);
  const len = Math.abs(d.t1 - d.t0); const m = ppm(); if (len < (m ? 0.15 * m : 8)) { showMeasure(""); return; }
  pushUndo(); const a = Math.min(d.t0, d.t1), b = Math.max(d.t0, d.t1);
  const code = defaultCode(d.type, len);
  S.plan.openings.push({ id: nextId("o", S.plan.openings), type: d.type, start: atT(d.wall, a), end: atT(d.wall, b), width: len, wall_id: d.wall.id,
    polygon: null, confidence: 1, code, width_source: "editor", height_m: null, sill_m: null, kind: null });
  S.sel = { kind: "opening", id: S.plan.openings[S.plan.openings.length - 1].id }; changed();
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
  hint({ select: "Clique para selecionar, arraste para mover. Pontas: redimensionar. Botão direito: girar. Roda do mouse: zoom.", wall: "Clique no início e no fim da parede. Encaixa nas linhas do PDF e em 0/90°.",
    door: "Pressione sobre uma parede e arraste ao longo dela. Solte na largura certa.", window: "Pressione sobre uma parede e arraste ao longo dela. Solte na largura certa.",
    calib: "Clique em dois pontos com distância conhecida e informe a medida." }[t]); render(); }
$$("#toolbar .tool").forEach(b => b.addEventListener("click", () => setTool(b.dataset.tool)));
$("#undo").addEventListener("click", undo); $("#redo").addEventListener("click", redo);
$("#show-prims").addEventListener("change", e => { S.showPrims = e.target.checked; render(); });
$("#snap").addEventListener("change", e => { S.snap = e.target.checked; });
document.addEventListener("keydown", e => {
  if (e.target.matches("input, select, textarea")) return;
  const k = e.key.toLowerCase();
  if (k === "v") setTool("select"); else if (k === "w") setTool("wall"); else if (k === "d") setTool("door"); else if (k === "j") setTool("window"); else if (k === "c") setTool("calib");
  else if (k === "r" && !e.ctrlKey && !e.metaKey && S.sel) { rotateSel(e.shiftKey ? -90 : 90); closeCtx(); }
  else if (k === "escape") { closeCtx(); S.draw = null; S.calib = null; clearPreview(); showSnap(null); $("#calib-box").hidden = true; select(null); }
  else if (k === "delete" || k === "backspace") { deleteSel(); e.preventDefault(); }
  else if ((e.ctrlKey || e.metaKey) && k === "z") { e.shiftKey ? redo() : undo(); e.preventDefault(); }
  else if ((e.ctrlKey || e.metaKey) && k === "y") { redo(); e.preventDefault(); }
  else if ((e.ctrlKey || e.metaKey) && k === "s") { save(); e.preventDefault(); }
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
  $("#queue").innerHTML = S.plans.map(p => `<li data-id="${p.id}" class="${p.id === S.pid ? "on" : ""}"><div class="t" title="${esc(p.file)}">${esc(p.title)}</div>
    <div class="m">${p.walls} paredes · ${p.openings} aberturas${p.questions ? " · " + p.questions + " dúvida(s)" : ""} <span class="st ${p.status}">${{ pending: "pendente", corrected: "corrigida", skipped: "pulada" }[p.status]}</span></div></li>`).join("") || "<li class='muted small'>Fila vazia.</li>";
  $$("#queue li[data-id]").forEach(li => li.addEventListener("click", () => openPlan(li.dataset.id)));
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
}
async function markStatus(status) {
  if (!S.pid) return; S.dirty = true; await save();
  await api(`/plans/${S.pid}/status`, { method: "POST", json: { status } });
  const cur = S.pid; await loadQueue();
  const next = S.plans.find(p => p.status === "pending" && p.id !== cur); if (next) openPlan(next.id); else { hint("Fila de pendentes vazia."); }
}
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

fit(); loadQueue();
})();
