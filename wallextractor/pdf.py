"""PDF ingestion: the entry point of the system is a PDF of the floor plan.

Two branches:

* ``render_page`` turns a page into an RGB image for the segmentation model
  (raster branch, works for any PDF, scanned or not).
* ``extract_vector_primitives`` reads the drawing commands of a vector PDF
  with PyMuPDF. Walls in CAD exports are usually filled polygons or thick
  stroked lines, so the primitives are returned with their fill/stroke and
  width so a classifier (rules now, a learned model later) can label them
  with exact coordinates and no neural network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    try:
        import fitz  # older PyMuPDF
    except ImportError as exc:
        raise ImportError("PyMuPDF is required: pip install pymupdf") from exc

Point = Tuple[float, float]


@dataclass
class Primitive:
    kind: str  # "line" | "rect" | "polygon" | "curve"
    points: List[Point]  # in pixels of the rendered image
    stroke_width: float
    filled: bool
    fill_gray: Optional[float]  # 0 = black, 1 = white, None = no fill
    stroke_gray: Optional[float]
    closed: bool


def page_count(pdf_path: str) -> int:
    with fitz.open(pdf_path) as doc:
        return doc.page_count


def render_page(pdf_path: str, page: int = 1, max_side: int = 1024) -> Tuple[np.ndarray, float]:
    """Render page ``page`` (1-based) so its longer side is ``max_side`` px.

    Returns ``(rgb_uint8_HxWx3, px_per_pt)``; the second value converts PDF
    points to pixels of the returned image.
    """
    with fitz.open(pdf_path) as doc:
        pg = doc[page - 1]
        rect = pg.rect
        zoom = max_side / max(rect.width, rect.height)
        pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False, colorspace=fitz.csRGB)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
    return arr, zoom


def is_vector(pdf_path: str, page: int = 1, min_paths: int = 50) -> bool:
    """True when the page carries enough drawing commands to be a CAD export."""
    with fitz.open(pdf_path) as doc:
        pg = doc[page - 1]
        return len(pg.get_drawings()) >= min_paths


def _gray(color) -> Optional[float]:
    if color is None:
        return None
    if len(color) == 1:
        return float(color[0])
    r, g, b = color[:3]
    return float(0.299 * r + 0.587 * g + 0.114 * b)


def extract_vector_primitives(pdf_path: str, page: int = 1, px_per_pt: float = 1.0) -> List[Primitive]:
    """Return every drawing path of the page as a Primitive (coordinates scaled by ``px_per_pt``)."""
    prims: List[Primitive] = []
    s = px_per_pt
    with fitz.open(pdf_path) as doc:
        pg = doc[page - 1]
        for d in pg.get_drawings():
            width = float(d.get("width") or 0.0) * s
            fill = d.get("fill")
            stroke = d.get("color")
            filled = fill is not None
            items = d.get("items", [])
            if filled and items and all(it[0] in ("l", "c") for it in items):
                # A filled path made of straight/curved pieces is one polygon (CAD walls come like this).
                pts: List[Point] = []
                for it in items:
                    p1 = it[1]
                    p_last = it[2] if it[0] == "l" else it[4]
                    if not pts or (abs(pts[-1][0] - p1.x * s) > 1e-6 or abs(pts[-1][1] - p1.y * s) > 1e-6):
                        pts.append((p1.x * s, p1.y * s))
                    pts.append((p_last.x * s, p_last.y * s))
                if len(pts) >= 2 and abs(pts[0][0] - pts[-1][0]) < 1e-6 and abs(pts[0][1] - pts[-1][1]) < 1e-6:
                    pts.pop()
                if len(pts) >= 3:
                    prims.append(Primitive("polygon", pts, width, True, _gray(fill), _gray(stroke), True))
                    continue
            for item in items:
                op = item[0]
                if op == "l":
                    p1, p2 = item[1], item[2]
                    pts = [(p1.x * s, p1.y * s), (p2.x * s, p2.y * s)]
                    prims.append(Primitive("line", pts, width, filled, _gray(fill), _gray(stroke), False))
                elif op == "re":
                    r = item[1]
                    pts = [(r.x0 * s, r.y0 * s), (r.x1 * s, r.y0 * s), (r.x1 * s, r.y1 * s), (r.x0 * s, r.y1 * s)]
                    prims.append(Primitive("rect", pts, width, filled, _gray(fill), _gray(stroke), True))
                elif op == "qu":
                    q = item[1]
                    pts = [(q.ul.x * s, q.ul.y * s), (q.ur.x * s, q.ur.y * s), (q.lr.x * s, q.lr.y * s),
                           (q.ll.x * s, q.ll.y * s)]
                    prims.append(Primitive("polygon", pts, width, filled, _gray(fill), _gray(stroke), True))
                elif op == "c":
                    p1, p4 = item[1], item[4]
                    pts = [(p1.x * s, p1.y * s), (p4.x * s, p4.y * s)]
                    prims.append(Primitive("curve", pts, width, filled, _gray(fill), _gray(stroke), False))
    return prims


def wall_candidates_from_primitives(prims: List[Primitive], min_thickness_px: float = 3.0,
                                    max_gray: float = 0.35) -> List[List[Point]]:
    """Rule-based v0: dark filled rectangles/polygons, or thick dark strokes, are wall candidates.

    Returns polygons in image pixels. This is a placeholder for a learned
    primitive classifier (FloorPlanCAD style); it exists so the vector branch
    produces something measurable from day one.
    """
    polys: List[List[Point]] = []
    for p in prims:
        if p.kind in ("rect", "polygon") and p.filled and p.fill_gray is not None and p.fill_gray <= max_gray:
            polys.append(p.points)
        elif p.kind == "line" and p.stroke_width >= min_thickness_px and (p.stroke_gray or 0.0) <= max_gray:
            (x1, y1), (x2, y2) = p.points
            dx, dy = x2 - x1, y2 - y1
            n = (dx * dx + dy * dy) ** 0.5 or 1.0
            ox, oy = -dy / n * p.stroke_width / 2.0, dx / n * p.stroke_width / 2.0
            polys.append([(x1 + ox, y1 + oy), (x2 + ox, y2 + oy), (x2 - ox, y2 - oy), (x1 - ox, y1 - oy)])
    return polys
