"""Walls from vector PDFs of architectural drawings (Brazilian CAD exports).

Observed convention (NBR 8403 pen weights, confirmed on real project PDFs):
walls in section are drawn with the thickest pen as double-line outlines;
door/window frames use a thinner pen; text, dimensions and furniture are
thinner still. The wall mass is therefore recovered by

1. locating the floor-plan drawing on the sheet (label "PLANTA BAIXA" or
   the densest cluster of thick-pen segments),
2. reading the scale from the "ESC. 1:75" style label near the drawing,
3. keeping the segments of the thickest pen inside that region,
4. pairing parallel segments whose distance is a plausible wall thickness
   (8 to 40 cm) and emitting the mid-line of their overlap as a wall.

Coordinates are PDF points on the page; the caller converts to pixels.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from .schema import Opening, Wall

Point = Tuple[float, float]
PT_PER_M = 2834.6457  # 1 m of drawing at 1:1 in PDF points (72 / 25.4 * 1000)


@dataclass
class Segment:
    a: Point
    b: Point
    width: float

    @property
    def length(self) -> float:
        return math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])


@dataclass
class PlanRegion:
    rect: Tuple[float, float, float, float]  # x0, y0, x1, y1 in page points
    label: Optional[str] = None
    scale_denominator: Optional[float] = None  # effective scale, 69.2 for a 1:75 sheet plotted at 108%
    pt_per_m: Optional[float] = None
    scale_method: Optional[str] = None  # "dimension_text" | "scale_label" | None
    nominal_scale: Optional[float] = None  # the printed "ESC. 1:N", if any
    wall_pen_width: Optional[float] = None
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------- primitives
def _display_xform(page):
    """Drawings and words come in unrotated coordinates; page.rect and pixmaps follow /Rotate."""
    return page.rotation_matrix if page.rotation else None


def _xy(m, x: float, y: float) -> Point:
    if m is None:
        return (float(x), float(y))
    p = fitz.Point(x, y) * m
    return (float(p.x), float(p.y))


def page_segments(page, min_width: float = 0.0) -> List[Segment]:
    """Straight stroked segments of the page (rect edges included), in display page points."""
    segs: List[Segment] = []
    page_area = page.rect.width * page.rect.height
    m = _display_xform(page)
    for d in page.get_drawings():
        if d.get("color") is None and d.get("fill") is None:
            continue
        w = float(d.get("width") or 0.0)
        if w < min_width:
            continue
        r = fitz.Rect(d["rect"])
        if r.width * r.height > 0.5 * page_area:
            continue  # sheet frame / title block border
        for it in d["items"]:
            if it[0] == "l":
                segs.append(Segment(_xy(m, it[1].x, it[1].y), _xy(m, it[2].x, it[2].y), w))
            elif it[0] == "re":
                q = it[1]
                pts = [(q.x0, q.y0), (q.x1, q.y0), (q.x1, q.y1), (q.x0, q.y1)]
                for i in range(4):
                    segs.append(Segment(_xy(m, *pts[i]), _xy(m, *pts[(i + 1) % 4]), w))
            elif it[0] == "qu":
                q = it[1]
                pts = [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)]
                for i in range(4):
                    segs.append(Segment(_xy(m, *pts[i]), _xy(m, *pts[(i + 1) % 4]), w))
    return segs


def _words(page):
    """Words as (x0, y0, x1, y1, text) in display page points."""
    m = _display_xform(page)
    out = []
    for w in page.get_text("words"):
        if m is None:
            out.append((w[0], w[1], w[2], w[3], w[4]))
        else:
            r = fitz.Rect(w[:4]) * m
            out.append((r.x0, r.y0, r.x1, r.y1, w[4]))
    return out


# ---------------------------------------------------------------- scale
_SCALE_RE = re.compile(r"1\s*[:/]\s*(\d{1,4})")


def scale_from_text(page, near: Optional[Point] = None, max_dist: float = 400.0) -> Optional[float]:
    """Return the scale denominator N of the nearest 'ESC 1:N' label (or any '1:N' if no label)."""
    words = _words(page)
    cands = []
    for i, w in enumerate(words):
        m = _SCALE_RE.search(w[4])
        if not m:
            # 'ESCALA' then '1:50' as the next word
            if re.match(r"(?i)^ESC", w[4]) and i + 1 < len(words):
                m = _SCALE_RE.search(words[i + 1][4])
                if not m:
                    continue
            else:
                continue
        n = float(m.group(1))
        if not (10 <= n <= 2000):
            continue
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        dist = math.hypot(cx - near[0], cy - near[1]) if near else 0.0
        cands.append((dist, n))
    if not cands:
        return None
    cands.sort()
    if near and cands[0][0] > max_dist:
        return None
    return cands[0][1]


# ---------------------------------------------------------------- region
def _clusters(segs: Sequence[Segment], cell: float, page_w: float, page_h: float, dilate: int = 2):
    """Connected clusters of segments on a coarse grid. Returns list of (bbox, [segment indices])."""
    nx, ny = int(page_w / cell) + 1, int(page_h / cell) + 1
    grid = np.zeros((ny, nx), dtype=bool)
    cell_of: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, s in enumerate(segs):
        n = max(2, int(s.length / cell) + 2)
        for t in np.linspace(0, 1, n):
            x = s.a[0] + t * (s.b[0] - s.a[0])
            y = s.a[1] + t * (s.b[1] - s.a[1])
            cx, cy = min(nx - 1, max(0, int(x / cell))), min(ny - 1, max(0, int(y / cell)))
            grid[cy, cx] = True
            cell_of[(cy, cx)].append(i)
    import cv2

    k = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
    dil = cv2.dilate(grid.astype(np.uint8), k)
    n, labels = cv2.connectedComponents(dil, connectivity=8)
    members: Dict[int, set] = defaultdict(set)
    for (cy, cx), idxs in cell_of.items():
        members[int(labels[cy, cx])].update(idxs)
    out = []
    for lab, idxs in members.items():
        if lab == 0 or not idxs:
            continue
        pts = np.array([p for i in idxs for p in (segs[i].a, segs[i].b)])
        bbox = (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))
        out.append((bbox, sorted(idxs)))
    return out


def wall_pen_width(segs: Sequence[Segment], min_count: int = 10) -> Optional[float]:
    """Thickest stroke width used by at least ``min_count`` segments."""
    counts = Counter(round(s.width, 2) for s in segs)
    for w, c in sorted(counts.items(), reverse=True):
        if w > 0 and c >= min_count:
            return w
    return None


def locate_plan_region(page, segs: Optional[List[Segment]] = None, pad_pt: float = 12.0) -> PlanRegion:
    """Find the floor-plan drawing on the sheet.

    Strategy: take the thickest-pen segments, cluster them spatially and
    pick the cluster closest to a "PLANTA BAIXA" label (label is normally
    just below the drawing); without a label, the cluster with the most
    thick-pen ink wins.
    """
    segs = segs if segs is not None else page_segments(page)
    notes: List[str] = []
    pen = wall_pen_width(segs)
    if pen is None:
        return PlanRegion(rect=tuple(page.rect), notes=["no stroked segments"])
    thick = [s for s in segs if abs(s.width - pen) < 1e-3 and s.length > 0.5]
    clusters = _clusters(thick, cell=max(8.0, page.rect.width / 150), page_w=page.rect.width,
                         page_h=page.rect.height)
    if not clusters:
        return PlanRegion(rect=tuple(page.rect), wall_pen_width=pen, notes=["no clusters"])

    words = _words(page)
    # Every "PLANTA BAIXA" occurrence; the title block repeats it as "PLANTA BAIXA, PLANTA DE COBERTURA, ..."
    labels: List[Tuple[Point, str, bool]] = []
    for i, w in enumerate(words):
        if re.match(r"(?i)^PLANTA$", w[4]) and i + 1 < len(words) and re.match(r"(?i)^BAIXA", words[i + 1][4]):
            nxt = words[i + 1]
            pt = ((min(w[0], nxt[0]) + max(w[2], nxt[2])) / 2, (min(w[1], nxt[1]) + max(w[3], nxt[3])) / 2)
            exact = re.match(r"(?i)^BAIXA[\s\-:]*$", nxt[4]) is not None
            labels.append((pt, f"{w[4]} {nxt[4]}", exact))

    def ink(c):
        return sum(thick[i].length for i in c[1])

    def bbox_dist(p, bbox):
        x0, y0, x1, y1 = bbox
        return math.hypot(max(x0 - p[0], 0, p[0] - x1), max(y0 - p[1], 0, p[1] - y1))

    best = None
    label_pt: Optional[Point] = None
    label_text = None
    if labels:
        scored = []
        for pt, text, exact in labels:
            for c in clusters:
                d = bbox_dist(pt, c[0])
                if d > 300:
                    continue
                scored.append((d - 0.05 * ink(c) - (100 if exact else 0), d, pt, text, c))
        if scored:
            _, d, label_pt, label_text, best = min(scored, key=lambda z: z[0])
            notes.append(f"label '{label_text}' at ({label_pt[0]:.0f},{label_pt[1]:.0f}), {d:.0f} pt from drawing")
    if best is None:
        best = max(clusters, key=ink)
        notes.append("no usable PLANTA BAIXA label; densest thick-pen cluster used")
    x0, y0, x1, y1 = best[0]
    rect = (max(0.0, x0 - pad_pt), max(0.0, y0 - pad_pt), min(page.rect.width, x1 + pad_pt),
            min(page.rect.height, y1 + pad_pt))
    region = PlanRegion(rect=rect, label=label_text, wall_pen_width=pen, notes=notes)
    anchor = label_pt or ((x0 + x1) / 2, y1)
    n = scale_from_text(page, near=anchor)
    consensus = scale_from_dimensions(page, segs, rect, pen, words=words)
    region.nominal_scale = n
    if consensus:
        # Geometry wins: sheets are often plotted "fit to page", so the label is only nominal.
        region.pt_per_m = consensus[0]
        region.scale_denominator = PT_PER_M / consensus[0]
        region.scale_method = "dimension_text"
        msg = f"scale from {consensus[1]} dimension strings: 1:{region.scale_denominator:.1f}"
        if n:
            ratio = (PT_PER_M / n) / consensus[0]
            msg += f" (label says 1:{n:.0f}; plotted at {100 / ratio:.0f}% of nominal)"
        region.notes.append(msg)
    elif n:
        region.scale_denominator = n
        region.pt_per_m = PT_PER_M / n
        region.scale_method = "scale_label"
        region.notes.append(f"scale from label 1:{n:.0f} (no dimension consensus to confirm)")
    else:
        region.notes.append("no scale label and no dimension consensus")
    return region


_DIM_RE = re.compile(r"^\.?\d{1,2}[.,]\d{2}$")


def scale_from_dimensions(page, segs: Sequence[Segment], rect, pen: Optional[float], words=None,
                          max_gap_pt: float = 14.0, min_agree: int = 4, tol: float = 0.04) -> Optional[Tuple[float, int]]:
    """Estimate pt-per-metre from the dimension strings and their dimension lines.

    A CAD dimension prints the value on a thin line whose length is the
    measured distance. For every numeric word inside the region, the nearest
    thin segment with the same orientation whose span covers the text gives
    one candidate ``length / value`` (collinear pieces split around the text
    are merged first). The median of the largest group agreeing within
    ``tol`` is returned with the group size.

    This is the primary scale source: sheets are often plotted "fit to page",
    so the printed "ESC. 1:75" is nominal while the geometry is at 1:69.
    """
    words = words if words is not None else _words(page)
    x0, y0, x1, y1 = rect
    m = 40.0
    thin = [s for s in segs if (pen is None or s.width < pen - 1e-3) and s.length > 3.0
            and x0 - m <= s.a[0] <= x1 + m and y0 - m <= s.a[1] <= y1 + m]
    if not thin:
        return None
    A = np.array([s.a for s in thin])
    B = np.array([s.b for s in thin])
    D = B - A
    L = np.linalg.norm(D, axis=1)
    horiz = np.abs(D[:, 0]) >= np.abs(D[:, 1])
    cands: List[float] = []
    for w in words:
        if not _DIM_RE.match(w[4]) or not (x0 - m <= w[0] <= x1 + m and y0 - m <= w[1] <= y1 + m):
            continue
        try:
            val = float(w[4].replace(",", "."))
        except ValueError:
            continue
        if not (0.05 <= val <= 80):
            continue
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        text_h = (w[2] - w[0]) >= (w[3] - w[1])
        text_len = (w[2] - w[0]) if text_h else (w[3] - w[1])
        sel = np.nonzero(horiz == text_h)[0]
        if len(sel) == 0:
            continue
        if text_h:
            perp = np.abs(A[sel, 1] - cy)
            lo, hi = np.minimum(A[sel, 0], B[sel, 0]), np.maximum(A[sel, 0], B[sel, 0])
            c = cx
        else:
            perp = np.abs(A[sel, 0] - cx)
            lo, hi = np.minimum(A[sel, 1], B[sel, 1]), np.maximum(A[sel, 1], B[sel, 1])
            c = cy
        # the line may pass under the text, or be split in two pieces around it
        near = (perp <= max_gap_pt) & (lo - text_len <= c) & (c <= hi + text_len)
        if not near.any():
            continue
        k = np.argmin(np.where(near, perp, np.inf))
        # merge collinear pieces (same offset within 0.6 pt, gaps up to 1.5 x text length)
        coll = near & (np.abs(perp - perp[k]) <= 0.6)
        span_lo, span_hi = float(lo[k]), float(hi[k])
        changed = True
        while changed:
            changed = False
            for j in np.nonzero(coll)[0]:
                if lo[j] < span_lo - 1e-6 and hi[j] >= span_lo - 1.5 * text_len:
                    span_lo, changed = float(lo[j]), True
                if hi[j] > span_hi + 1e-6 and lo[j] <= span_hi + 1.5 * text_len:
                    span_hi, changed = float(hi[j]), True
        ppm = (span_hi - span_lo) / val
        if PT_PER_M / 500 <= ppm <= PT_PER_M / 10:
            cands.append(ppm)
    if len(cands) < min_agree:
        return None
    cands.sort()
    best: List[float] = []
    for c in cands:
        group = [v for v in cands if abs(v - c) <= tol * c]
        if len(group) > len(best):
            best = group
    if len(best) < min_agree:
        return None
    return float(np.median(best)), len(best)


# ---------------------------------------------------------------- pairing
def _inside(s: Segment, rect) -> bool:
    x0, y0, x1, y1 = rect
    return all(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in (s.a, s.b))


def pair_parallel_segments(segs: Sequence[Segment], d_min: float, d_max: float, angle_tol_deg: float = 2.0,
                           min_overlap: float = 1.0) -> List[Tuple[Point, Point, float]]:
    """Return (start, end, thickness) for every pair of parallel segments at wall distance."""
    n = len(segs)
    if n == 0:
        return []
    A = np.array([s.a for s in segs], dtype=np.float64)
    B = np.array([s.b for s in segs], dtype=np.float64)
    D = B - A
    L = np.linalg.norm(D, axis=1)
    keep = L > 1e-6
    U = np.zeros_like(D)
    U[keep] = D[keep] / L[keep, None]
    ang = np.degrees(np.arctan2(U[:, 1], U[:, 0])) % 180.0
    out: List[Tuple[Point, Point, float]] = []
    cos_tol = math.cos(math.radians(angle_tol_deg))
    for i in range(n):
        if not keep[i]:
            continue
        # candidate j: parallel and roughly nearby (bbox prefilter)
        par = np.abs(U[i] @ U.T) >= cos_tol
        par[: i + 1] = False
        par &= keep
        if not par.any():
            continue
        ui = U[i]
        ni = np.array([-ui[1], ui[0]])
        for j in np.nonzero(par)[0]:
            # perpendicular distance of j's midpoint to line i
            mid = (A[j] + B[j]) / 2
            d = float(np.dot(mid - A[i], ni))
            if not (d_min <= abs(d) <= d_max):
                continue
            # both endpoints of j should sit at about the same offset (true parallel)
            if abs(float(np.dot(A[j] - A[i], ni)) - d) > 0.3 * abs(d) + 0.5:
                continue
            ti = sorted([0.0, float(np.dot(B[i] - A[i], ui))])
            tj = sorted([float(np.dot(A[j] - A[i], ui)), float(np.dot(B[j] - A[i], ui))])
            lo, hi = max(ti[0], tj[0]), min(ti[1], tj[1])
            if hi - lo < max(min_overlap, 0.5 * abs(d)):
                continue
            off = ni * (d / 2.0)
            s = A[i] + ui * lo + off
            e = A[i] + ui * hi + off
            out.append(((float(s[0]), float(s[1])), (float(e[0]), float(e[1])), abs(d)))
    return out


def _merge_walls(cands: List[Tuple[Point, Point, float]], angle_tol_deg: float = 2.0, offset_tol: float = 1.0,
                 gap: float = 2.0) -> List[Tuple[Point, Point, float]]:
    """Merge collinear, overlapping candidates with similar thickness (keeps the longest span)."""
    from .geometry import point_segment_distance, segment_length

    cands = sorted(cands, key=lambda c: -segment_length(c[0], c[1]))
    out: List[Tuple[Point, Point, float]] = []
    for s in cands:
        merged = False
        for k, t in enumerate(out):
            a1 = math.degrees(math.atan2(s[1][1] - s[0][1], s[1][0] - s[0][0])) % 180
            a2 = math.degrees(math.atan2(t[1][1] - t[0][1], t[1][0] - t[0][0])) % 180
            if min(abs(a1 - a2), 180 - abs(a1 - a2)) > angle_tol_deg:
                continue
            if abs(s[2] - t[2]) > 0.4 * max(s[2], t[2]):
                continue
            if max(point_segment_distance(s[0], t[0], t[1]), point_segment_distance(s[1], t[0], t[1])) > offset_tol:
                continue
            dx, dy = t[1][0] - t[0][0], t[1][1] - t[0][1]
            nrm = math.hypot(dx, dy) or 1.0
            ux, uy = dx / nrm, dy / nrm
            proj = [((p[0] - t[0][0]) * ux + (p[1] - t[0][1]) * uy, p) for p in (t[0], t[1], s[0], s[1])]
            pt_ = sorted(proj[:2], key=lambda z: z[0])
            ps = sorted(proj[2:], key=lambda z: z[0])
            if ps[0][0] > pt_[1][0] + gap or ps[1][0] < pt_[0][0] - gap:
                continue
            lo = min(proj, key=lambda z: z[0])[1]
            hi = max(proj, key=lambda z: z[0])[1]
            out[k] = (lo, hi, (s[2] + t[2]) / 2)
            merged = True
            break
        if not merged:
            out.append(s)
    return out


def extract_walls(page, region: Optional[PlanRegion] = None, thickness_m: Tuple[float, float] = (0.08, 0.40),
                  min_length_m: float = 0.20, with_openings: bool = True):
    """Vector wall (and opening) extraction for one page.

    Returns ``(walls, openings, region)`` with coordinates in page points.
    """
    segs = page_segments(page)
    region = region or locate_plan_region(page, segs)
    pen = region.wall_pen_width or wall_pen_width(segs)
    thick = [s for s in segs if pen is not None and abs(s.width - pen) < 1e-3 and _inside(s, region.rect)]
    if region.pt_per_m:
        d_min, d_max = thickness_m[0] * region.pt_per_m, thickness_m[1] * region.pt_per_m
        min_len = min_length_m * region.pt_per_m
    else:
        d_min, d_max, min_len = 2.0, 20.0, 4.0
        region.notes.append("scale unknown: thickness window 2-20 pt")
    cands = pair_parallel_segments(thick, d_min, d_max)
    merged = _merge_walls(cands)
    walls = [Wall(id=f"w{i + 1}", start=s, end=e, thickness=t) for i, (s, e, t) in
             enumerate(m for m in merged if math.hypot(m[1][0] - m[0][0], m[1][1] - m[0][1]) >= min_len)]
    walls = dedupe_walls(walls)
    region.notes.append(f"pen={pen} thick_segments={len(thick)} pairs={len(cands)} walls={len(walls)}")
    openings: List[Opening] = []
    if with_openings:
        openings = find_openings(page, walls, segs, region)
    return walls, openings, region


# ---------------------------------------------------------------- openings
_TAG_RE = re.compile(r"^\(?([PJ])\s?\d{1,2}\)?[.,]?$", re.IGNORECASE)


def page_curves(page) -> List[Tuple[Point, Point, float]]:
    """Bezier items as (start, end, stroke width) in display coordinates (door swings are arcs)."""
    m = _display_xform(page)
    out = []
    for d in page.get_drawings():
        w = float(d.get("width") or 0.0)
        for it in d["items"]:
            if it[0] == "c":
                out.append((_xy(m, it[1].x, it[1].y), _xy(m, it[4].x, it[4].y), w))
    return out


def _wall_gaps(walls: Sequence[Wall], gap_range: Tuple[float, float], offset_tol: float = 2.0,
               angle_tol_deg: float = 2.0):
    """Gaps between collinear walls: (start, end, width, wall_id_left, wall_id_right, thickness)."""
    n = len(walls)
    if n < 2:
        return []
    A = np.array([w.start for w in walls]);
    B = np.array([w.end for w in walls])
    D = B - A
    L = np.linalg.norm(D, axis=1)
    U = D / np.maximum(L, 1e-9)[:, None]
    cos_tol = math.cos(math.radians(angle_tol_deg))
    gaps = []
    for i in range(n):
        ui = U[i]
        ni = np.array([-ui[1], ui[0]])
        proj_i = sorted([0.0, float(L[i])])
        # every wall on (nearly) the same line, as intervals along i's axis
        on_line = []
        for j in range(n):
            if j == i or abs(float(ui @ U[j])) < cos_tol:
                continue
            off = float(np.dot((A[j] + B[j]) / 2 - A[i], ni))
            if abs(off) > max(offset_tol, 0.35 * walls[i].thickness):
                continue
            t0, t1 = sorted([float(np.dot(A[j] - A[i], ui)), float(np.dot(B[j] - A[i], ui))])
            on_line.append((t0, t1, j))
        for t0, t1, j in on_line:
            if t0 <= proj_i[1]:  # only walls further along the axis, so every pair is seen once
                continue
            gap = t0 - proj_i[1]
            if not (gap_range[0] <= gap <= gap_range[1]):
                continue
            # nothing else on the line inside the gap
            if any(k != j and not (s1 <= proj_i[1] or s0 >= t0) for s0, s1, k in on_line):
                continue
            s = A[i] + ui * proj_i[1]
            e = A[i] + ui * t0
            gaps.append(((float(s[0]), float(s[1])), (float(e[0]), float(e[1])), gap, walls[i].id, walls[j].id,
                         (walls[i].thickness + walls[j].thickness) / 2))
    return gaps


def _wall_frame(w: Wall):
    ax, ay = w.start
    dx, dy = w.end[0] - ax, w.end[1] - ay
    L = math.hypot(dx, dy) or 1e-9
    ux, uy = dx / L, dy / L
    return (ax, ay), (ux, uy), (-uy, ux), L


def _merge_intervals(items, gap: float):
    """items: list of (t0, t1, payload); merges overlapping/near intervals, concatenating payloads."""
    items = sorted(items, key=lambda z: z[0])
    out = []
    for t0, t1, pay in items:
        if out and t0 <= out[-1][1] + gap:
            o0, o1, opay = out[-1]
            out[-1] = (o0, max(o1, t1), opay + list(pay))
        else:
            out.append((t0, t1, list(pay)))
    return out


def dedupe_walls(walls: List[Wall], offset_tol: float = 1.0, angle_tol_deg: float = 2.0) -> List[Wall]:
    """Drop walls that lie on another, longer wall (same line, overlapping span)."""
    keep: List[Wall] = []
    cos_tol = math.cos(math.radians(angle_tol_deg))
    for w in sorted(walls, key=lambda z: -z.length):
        dup = False
        for k in keep:
            (ax, ay), (ux, uy), (nx, ny), L = _wall_frame(k)
            if abs(ux * (w.end[0] - w.start[0]) + uy * (w.end[1] - w.start[1])) < cos_tol * w.length:
                continue
            offs = [(p[0] - ax) * nx + (p[1] - ay) * ny for p in (w.start, w.end)]
            if max(abs(o) for o in offs) > max(offset_tol, 0.35 * k.thickness):
                continue
            t0, t1 = sorted((p[0] - ax) * ux + (p[1] - ay) * uy for p in (w.start, w.end))
            overlap = min(t1, L) - max(t0, 0.0)
            if overlap >= 0.5 * w.length:
                dup = True
                break
        if not dup:
            keep.append(w)
    keep.sort(key=lambda z: int(z.id[1:]))
    for i, w in enumerate(keep):
        w.id = f"w{i + 1}"
    return keep


def chain_curves(curves: Sequence[Tuple[Point, Point, float]], tol: float = 0.6):
    """Join bezier pieces end-to-start into arcs. Returns (start, end, n_pieces, path_len) per chain."""
    n = len(curves)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ends: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, (a, b, _w) in enumerate(curves):
        for p in (a, b):
            ends[(round(p[0] / tol), round(p[1] / tol))].append(i)
    for idxs in ends.values():
        for j in idxs[1:]:
            ri, rj = find(idxs[0]), find(j)
            if ri != rj:
                parent[rj] = ri
    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    out = []
    for idxs in groups.values():
        cnt: Counter = Counter()
        path = 0.0
        for i in idxs:
            a, b, _w = curves[i]
            path += math.hypot(b[0] - a[0], b[1] - a[1])
            for p in (a, b):
                cnt[(round(p[0] / tol), round(p[1] / tol))] += 1
        free = [k for k, c in cnt.items() if c == 1]
        if len(free) != 2:
            continue  # closed loop (circle) or branching
        pts = []
        for k in free:
            for i in idxs:
                for p in curves[i][:2]:
                    if (round(p[0] / tol), round(p[1] / tol)) == k:
                        pts.append(p)
                        break
                else:
                    continue
                break
        if len(pts) == 2:
            out.append((pts[0], pts[1], len(idxs), path))
    return out


def find_openings(page, walls: Sequence[Wall], segs: Sequence[Segment], region: PlanRegion,
                  width_m: Tuple[float, float] = (0.30, 4.00)) -> List[Opening]:
    """Doors and windows of a vector plan, cue first.

    Per wall, candidates come from independent cues and are merged by overlap:
      * door leaf: a thin segment perpendicular to the wall, one end on the
        wall band, 0.55-1.3 m long; the swing arc (bezier pieces chained
        end-to-end) confirms the side the door opens to;
      * window frame: two or more mid-pen segments with matching ends running
        along the wall inside its band;
      * wall gap: an interruption between collinear walls.
    A frame tag next to the opening ("P2" door, "J1" window) sets the type;
    otherwise leaf => door, frame => window, and a bare gap is dropped.
    """
    ppm = region.pt_per_m or (PT_PER_M / 75.0)
    w_min, w_max = width_m[0] * ppm, width_m[1] * ppm
    x0, y0, x1, y1 = region.rect
    pen = region.wall_pen_width or 1.0
    band_tol = 0.5
    words = _words(page)
    tags = []
    for w in words:
        m = _TAG_RE.match(w[4].strip())
        if m and x0 - 20 <= w[0] <= x1 + 20 and y0 - 20 <= w[1] <= y1 + 20:
            tags.append((((w[0] + w[2]) / 2, (w[1] + w[3]) / 2), "door" if m.group(1).upper() == "P" else "window"))
    curves = [c for c in page_curves(page) if x0 <= c[0][0] <= x1 and y0 <= c[0][1] <= y1]
    arcs = [a for a in chain_curves(curves) if 0.5 * w_min <= math.hypot(a[1][0] - a[0][0], a[1][1] - a[0][1])]
    thin_all = [s for s in segs if 1e-3 < s.width < pen - 1e-3 and _inside(s, region.rect) and s.length > 2.0]
    mid_pen = [s for s in thin_all if s.width >= 0.2 * pen]
    # pieces that may belong to a swing arc: bezier bits, or short thin lines of a polyline arc
    arc_pieces = [(c[0], c[1]) for c in curves] + [
        (s.a, s.b) for s in segs if 1e-3 < s.width < pen - 1e-3 and s.length <= 0.2 * ppm and _inside(s, region.rect)]

    def on_circle(hx, hy, radius):
        return sum(1 for a, b in arc_pieces
                   if abs(math.hypot((a[0] + b[0]) / 2 - hx, (a[1] + b[1]) / 2 - hy) - radius) <= 0.15 * radius)

    per_wall: Dict[int, List[Tuple[float, float, List[str]]]] = defaultdict(list)
    for wi, w in enumerate(walls):
        (ax, ay), (ux, uy), (nx, ny), L = _wall_frame(w)
        half = 0.75 * w.thickness + band_tol

        def proj(p):
            return (p[0] - ax) * ux + (p[1] - ay) * uy, (p[0] - ax) * nx + (p[1] - ay) * ny

        # door leaves drawn closed: a thin rectangle 0.6-1.3 m long lying parallel to the wall,
        # just outside its band (the leaf sits in the opening, the wall lines run behind it)
        leaf_lines = []
        for s in thin_all:
            if not (0.6 * ppm <= s.length <= 1.3 * ppm):
                continue
            dx, dy = s.b[0] - s.a[0], s.b[1] - s.a[1]
            if abs(dx * ux + dy * uy) < 0.985 * s.length:
                continue
            ta, oa = proj(s.a)
            tb, ob = proj(s.b)
            off = (oa + ob) / 2
            if not (half < abs(off) <= 0.45 * ppm):
                continue
            lo, hi = sorted([ta, tb])
            if hi < -0.1 * ppm or lo > L + 0.1 * ppm:
                continue
            leaf_lines.append((lo, hi, off))
        used = set()
        for i in range(len(leaf_lines)):
            if i in used:
                continue
            lo, hi, off = leaf_lines[i]
            for j in range(i + 1, len(leaf_lines)):
                lo2, hi2, off2 = leaf_lines[j]
                if j in used or abs(off - off2) > 3.0 or abs(off - off2) < 0.3:
                    continue
                if abs(lo - lo2) <= 0.1 * (hi - lo) and abs(hi - hi2) <= 0.1 * (hi - lo):
                    used.update((i, j))
                    # confirmation: dashed swing pieces lie on a circle of radius = leaf around a hinge
                    leaf = hi - lo
                    n_arc = max(on_circle(ax + ux * t_h, ay + uy * t_h, leaf) for t_h in (lo, hi))
                    per_wall[wi].append((lo, hi, ["leaf"] + (["swing"] if n_arc >= 3 else [])))
                    break
        # door leaves drawn open: perpendicular thin segment hinged on the wall band, with a swing arc
        for s in thin_all:
            if not (0.55 * ppm <= s.length <= 1.3 * ppm):
                continue
            dx, dy = s.b[0] - s.a[0], s.b[1] - s.a[1]
            if abs(dx * ux + dy * uy) > 0.12 * s.length:
                continue
            for hinge, tip in ((s.a, s.b), (s.b, s.a)):
                th, oh = proj(hinge)
                if abs(oh) > half or not (-0.1 * ppm <= th <= L + 0.1 * ppm):
                    continue
                leaf = s.length
                side = None
                for a0, a1, _n, _path in arcs:
                    for p_tip, p_on in ((a0, a1), (a1, a0)):
                        if math.hypot(p_tip[0] - tip[0], p_tip[1] - tip[1]) > 0.2 * leaf:
                            continue
                        t_on, o_on = proj(p_on)
                        if abs(o_on) <= half + 0.15 * leaf and abs(abs(t_on - th) - leaf) <= 0.25 * leaf:
                            side = 1.0 if t_on > th else -1.0
                            break
                    if side is not None:
                        break
                if side is None and on_circle(hinge[0], hinge[1], leaf) >= 5:
                    # dashed or polyline swing: pieces on the circle; the arc ends on the wall line
                    plus = sum(1 for a, b in arc_pieces
                               if abs(math.hypot((a[0] + b[0]) / 2 - hinge[0], (a[1] + b[1]) / 2 - hinge[1]) - leaf) <= 0.15 * leaf
                               and proj(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))[0] > th)
                    side = 1.0 if plus >= on_circle(hinge[0], hinge[1], leaf) / 2 else -1.0
                if side is None:
                    continue
                lo, hi = sorted([th, th + side * leaf])
                per_wall[wi].append((lo, hi, ["leaf", "swing"]))
                break
        # window frames
        frames = []
        for s in mid_pen:
            dx, dy = s.b[0] - s.a[0], s.b[1] - s.a[1]
            if abs(dx * ux + dy * uy) < 0.985 * s.length:
                continue
            ta, oa = proj(s.a)
            tb, ob = proj(s.b)
            if max(abs(oa), abs(ob)) > half:
                continue
            lo, hi = sorted([ta, tb])
            if hi < -0.2 * ppm or lo > L + 0.2 * ppm:
                continue
            frames.append((lo, hi, [(lo, hi)]))
        for lo, hi, spans in _merge_intervals(frames, gap=0.15 * ppm):
            width = hi - lo
            if not (w_min <= width <= w_max):
                continue
            consistent = sum(1 for a, b in spans if abs(a - lo) <= 0.12 * width and abs(b - hi) <= 0.12 * width)
            if consistent >= 2:
                per_wall[wi].append((lo, hi, ["frame"] * consistent))

    wall_index = {w.id: i for i, w in enumerate(walls)}
    seen_gaps = set()
    for gs, ge, width, wid_l, wid_r, thick in _wall_gaps(walls, (w_min, w_max)):
        key = (round((gs[0] + ge[0]) / 2), round((gs[1] + ge[1]) / 2))
        if key in seen_gaps:
            continue
        seen_gaps.add(key)
        wi = wall_index[wid_l]
        (ax, ay), (ux, uy), _n, L = _wall_frame(walls[wi])
        t0 = (gs[0] - ax) * ux + (gs[1] - ay) * uy
        t1 = (ge[0] - ax) * ux + (ge[1] - ay) * uy
        per_wall[wi].append((min(t0, t1), max(t0, t1), ["gap"]))

    cands = []
    for wi, items in per_wall.items():
        primary = [(lo, hi, cues) for lo, hi, cues in items if "leaf" in cues or "frame" in cues]
        gaps_w = [(lo, hi) for lo, hi, cues in items if cues == ["gap"]]
        taken = [False] * len(gaps_w)
        # overlapping primaries (e.g. a leaf and its frame lines) become one opening
        merged = []
        for lo, hi, cues in sorted(primary, key=lambda z: z[0]):
            if merged and lo < merged[-1][1] - 0.05 * ppm:
                m0, m1, mc = merged[-1]
                # keep the leaf's own extent when a leaf is involved (door width = leaf length)
                if "leaf" in mc and "leaf" not in cues:
                    merged[-1] = (m0, m1, mc + cues)
                elif "leaf" in cues and "leaf" not in mc:
                    merged[-1] = (lo, hi, mc + cues)
                else:
                    merged[-1] = (m0, max(m1, hi), mc + cues)
            else:
                merged.append((lo, hi, list(cues)))
        for lo, hi, cues in merged:
            for gi, (g0, g1) in enumerate(gaps_w):
                ov = min(hi, g1) - max(lo, g0)
                if ov >= 0.5 * min(hi - lo, g1 - g0):
                    cues = cues + ["gap"]
                    taken[gi] = True
            if w_min <= hi - lo <= w_max:
                cands.append((wi, lo, hi, cues))
        for gi, (g0, g1) in enumerate(gaps_w):
            if not taken[gi]:
                cands.append((wi, g0, g1, ["gap"]))
    cands = [{"wall": wi, "start": (walls[wi].start[0] + _wall_frame(walls[wi])[1][0] * lo,
                                    walls[wi].start[1] + _wall_frame(walls[wi])[1][1] * lo),
              "end": (walls[wi].start[0] + _wall_frame(walls[wi])[1][0] * hi,
                      walls[wi].start[1] + _wall_frame(walls[wi])[1][1] * hi),
              "width": hi - lo, "cues": Counter(cues), "tag": None} for wi, lo, hi, cues in cands]

    # Tag -> candidate assignment. Cost = distance plus penalties when the candidate's cues do not
    # fit the tag type (a "P" tag wants a door leaf, a "J" tag wants frame lines or a wall gap).
    pairs = []
    for k, (tp, ttype) in enumerate(tags):
        for ci, c in enumerate(cands):
            cx, cy = (c["start"][0] + c["end"][0]) / 2, (c["start"][1] + c["end"][1]) / 2
            d = math.hypot(tp[0] - cx, tp[1] - cy)
            if d > max(1.2 * ppm, 1.0 * c["width"]):
                continue
            cues = c["cues"]
            cost = d
            if ttype == "door":
                cost += 0 if cues["leaf"] else 0.6 * ppm
                cost += 0.4 * ppm if cues["leaf"] >= 3 else 0  # merged clutter (furniture) is unlikely a door
                cost += 0.3 * ppm if not (0.55 * ppm <= c["width"] <= 1.3 * ppm) else 0
            else:
                cost += 0 if (cues["frame"] or cues["gap"]) else 0.6 * ppm
                cost += 0.4 * ppm if cues["leaf"] else 0
            pairs.append((cost, k, ci, ttype))
    used_t, used_c = set(), set()
    for cost, k, ci, ttype in sorted(pairs):
        if k in used_t or ci in used_c:
            continue
        used_t.add(k)
        used_c.add(ci)
        cands[ci]["tag"] = ttype
    tagged_drawing = len(tags) >= 3

    region.candidates = cands  # type: ignore[attr-defined]  (inspection / annotation aid)
    # Tags with no candidate: the opening is on the nearest wall. A short wall piece next to the tag is
    # a window/door drawn with the wall pen (its lines were paired as a "wall"); otherwise use a default width.
    for k, (tp, ttype) in enumerate(tags):
        if k in used_t:
            continue
        best = None
        for wi, w in enumerate(walls):
            (ax, ay), (ux, uy), (nx, ny), L = _wall_frame(w)
            t = (tp[0] - ax) * ux + (tp[1] - ay) * uy
            o = abs((tp[0] - ax) * nx + (tp[1] - ay) * ny)
            if -0.3 * ppm <= t <= L + 0.3 * ppm and o <= 1.2 * ppm and (best is None or o < best[0]):
                best = (o, wi, t)
        if best is None:
            continue
        o, wi, t = best
        (ax, ay), (ux, uy), _n, L = _wall_frame(walls[wi])
        if L <= 2.0 * ppm:
            lo, hi = 0.0, L
        else:
            half_w = (0.8 if ttype == "door" else 0.6) * ppm / 2
            lo, hi = max(0.0, t - half_w), min(L, t + half_w)
        cands.append({"wall": wi, "start": (ax + ux * lo, ay + uy * lo), "end": (ax + ux * hi, ay + uy * hi),
                      "width": hi - lo, "cues": Counter(["tag_only"]), "tag": ttype})
        used_t.add(k)

    typed = []
    for c in cands:
        cues = c["cues"]
        if c["tag"]:
            otype = c["tag"]
        elif tagged_drawing:
            continue  # every opening carries a frame tag in this drawing; untagged cues are clutter
        elif cues["leaf"] and (cues["swing"] or cues["gap"]):
            otype = "door"
        elif cues["frame"] >= 3 or (cues["frame"] and cues["gap"]):
            otype = "window"
        else:
            continue
        score = (1 if c["tag"] else 0, sum(cues.values()))
        typed.append((score, otype, c))
    # the same physical opening can be seen from two overlapping walls: keep the best-supported one
    typed.sort(key=lambda z: z[0], reverse=True)
    openings: List[Opening] = []
    for score, otype, c in typed:
        cx, cy = (c["start"][0] + c["end"][0]) / 2, (c["start"][1] + c["end"][1]) / 2
        if any(math.hypot((o.start[0] + o.end[0]) / 2 - cx, (o.start[1] + o.end[1]) / 2 - cy) <= 0.3 * ppm
               for o in openings):
            continue
        openings.append(Opening(id=f"o{len(openings) + 1}", type=otype, start=c["start"], end=c["end"],
                                width=c["width"], wall_id=walls[c["wall"]].id,
                                confidence=0.5 if c["cues"]["tag_only"] else 1.0))
    n_d = sum(o.type == "door" for o in openings)
    region.notes.append(f"opening cues: candidates={len(cands)} tags={len(tags)} matched_tags={len(used_t)} "
                        f"tags_required={tagged_drawing} "
                        f"arcs={len(arcs)} -> doors={n_d} windows={len(openings) - n_d}")
    return openings
