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

from .schema import Wall

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
                  min_length_m: float = 0.20) -> Tuple[List[Wall], PlanRegion]:
    """Vector wall extraction for one page. Returns walls in page points and the region used."""
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
    region.notes.append(f"pen={pen} thick_segments={len(thick)} pairs={len(cands)} walls={len(walls)}")
    return walls, region
