"""Mask -> wall segments.

Pipeline: wall mask -> distance transform (thickness) -> skeleton -> line
segments (probabilistic Hough on the skeleton) -> merge of collinear pieces.
It is deliberately simple; it is the v0 that makes the segmentation output
measurable as geometry. Opening polygons come from connected components of
the door/window classes.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .geometry import point_segment_distance, segment_length
from .schema import ImageInfo, Opening, Scale, Source, Wall, WallPlan

Point = Tuple[float, float]


def _merge_collinear(segs: List[Tuple[Point, Point]], angle_tol_deg: float = 6.0, gap: float = 8.0,
                     offset: float = 4.0) -> List[Tuple[Point, Point]]:
    segs = sorted(segs, key=lambda s: -segment_length(*s))
    out: List[Tuple[Point, Point]] = []
    for s in segs:
        merged = False
        for i, t in enumerate(out):
            a1 = math.degrees(math.atan2(s[1][1] - s[0][1], s[1][0] - s[0][0])) % 180
            a2 = math.degrees(math.atan2(t[1][1] - t[0][1], t[1][0] - t[0][0])) % 180
            if min(abs(a1 - a2), 180 - abs(a1 - a2)) > angle_tol_deg:
                continue
            if max(point_segment_distance(s[0], *t), point_segment_distance(s[1], *t)) > offset:
                continue
            # project all four endpoints on t's direction; require overlap or a small gap
            dx, dy = t[1][0] - t[0][0], t[1][1] - t[0][1]
            n = math.hypot(dx, dy) or 1.0
            ux, uy = dx / n, dy / n
            proj = [((p[0] - t[0][0]) * ux + (p[1] - t[0][1]) * uy, p) for p in (t[0], t[1], s[0], s[1])]
            pt = sorted(proj[:2], key=lambda z: z[0])
            ps = sorted(proj[2:], key=lambda z: z[0])
            if ps[0][0] > pt[1][0] + gap or ps[1][0] < pt[0][0] - gap:
                continue
            lo = min(proj, key=lambda z: z[0])[1]
            hi = max(proj, key=lambda z: z[0])[1]
            out[i] = (lo, hi)
            merged = True
            break
        if not merged:
            out.append(s)
    return out


def mask_to_walls(mask: np.ndarray, wall_class: int = 1, min_length: float = 12.0) -> List[Wall]:
    """Extract wall segments with thickness from a class mask."""
    wall = (mask == wall_class).astype(np.uint8)
    if wall.sum() == 0:
        return []
    wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dist = cv2.distanceTransform(wall, cv2.DIST_L2, 5)
    skel = skeletonize(wall.astype(bool)).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(skel, rho=1, theta=np.pi / 180, threshold=15, minLineLength=int(min_length),
                            maxLineGap=6)
    if lines is None:
        return []
    segs = [((float(x1), float(y1)), (float(x2), float(y2))) for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4)]
    segs = _merge_collinear(segs)
    walls: List[Wall] = []
    for i, (a, b) in enumerate(segs):
        if segment_length(a, b) < min_length:
            continue
        # thickness = 2 x median distance-to-edge sampled along the centreline
        n = max(2, int(segment_length(a, b) / 3))
        xs = np.linspace(a[0], b[0], n).round().astype(int).clip(0, mask.shape[1] - 1)
        ys = np.linspace(a[1], b[1], n).round().astype(int).clip(0, mask.shape[0] - 1)
        vals = dist[ys, xs]
        vals = vals[vals > 0]
        thickness = float(2.0 * np.median(vals)) if len(vals) else 2.0
        walls.append(Wall(id=f"w{i + 1}", start=a, end=b, thickness=thickness))
    return walls


def mask_to_openings(mask: np.ndarray, door_class: int = 2, window_class: int = 3, min_area: int = 20) -> List[Opening]:
    ops: List[Opening] = []
    for cls, otype in ((door_class, "door"), (window_class, "window")):
        m = (mask == cls).astype(np.uint8)
        if m.sum() == 0:
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < min_area:
                continue
            ys, xs = np.where(labels == k)
            pts = np.stack([xs, ys], axis=1).astype(np.float32)
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)
            from .geometry import polygon_to_segment
            s, e, w = polygon_to_segment(box)
            ops.append(Opening(id=f"o{len(ops) + 1}", type=otype, start=s, end=e, width=w,
                               polygon=[(float(x), float(y)) for x, y in box]))
    return ops


def mask_to_plan(mask: np.ndarray, source_file: str = "", page: int = 1, kind: str = "raster") -> WallPlan:
    h, w = mask.shape[:2]
    return WallPlan(
        source=Source(file=source_file, page=page, kind=kind),
        image=ImageInfo(width=int(w), height=int(h)),
        walls=mask_to_walls(mask),
        openings=mask_to_openings(mask),
        scale=Scale(),
    )


def draw_plan(image: np.ndarray, plan: WallPlan) -> np.ndarray:
    """Overlay walls (red) and openings (door green, window blue) on an RGB image."""
    out = image.copy()
    for wl in plan.walls:
        cv2.line(out, tuple(map(int, wl.start)), tuple(map(int, wl.end)), (220, 30, 30), max(1, int(round(wl.thickness))))
    for op in plan.openings:
        color = (30, 180, 30) if op.type == "door" else (30, 60, 220)
        cv2.line(out, tuple(map(int, op.start)), tuple(map(int, op.end)), color, max(1, int(round(op.width))))
    return out
