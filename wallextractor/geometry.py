"""Small geometry helpers shared by the converters and the vectorizer."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


def polygon_to_segment(points: Sequence[Point]) -> Tuple[Point, Point, float]:
    """Fit a wall polygon with its minimum-area rectangle.

    Returns ``(start, end, thickness)``: the centreline of the long side and the
    length of the short side. Works for quads with mitred ends, which is what
    CubiCasa5K and most CAD exports produce.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    if w >= h:
        length, thickness, theta = w, h, math.radians(angle)
    else:
        length, thickness, theta = h, w, math.radians(angle + 90.0)
    dx, dy = math.cos(theta) * length / 2.0, math.sin(theta) * length / 2.0
    start = (float(cx - dx), float(cy - dy))
    end = (float(cx + dx), float(cy + dy))
    # keep a stable orientation: left-to-right, then top-to-bottom
    if (end[0], end[1]) < (start[0], start[1]):
        start, end = end, start
    return start, end, float(max(thickness, 1e-3))


def segment_length(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def rdp(points: Sequence[Point], epsilon: float) -> List[Point]:
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(points) < 3:
        return list(points)
    pts = np.asarray(points, dtype=np.float64)
    start, end = pts[0], pts[-1]
    line = end - start
    norm = np.linalg.norm(line)
    if norm == 0:
        dists = np.linalg.norm(pts - start, axis=1)
    else:
        dists = np.abs(np.cross(line, pts - start)) / norm
    idx = int(np.argmax(dists))
    if dists[idx] > epsilon:
        left = rdp(points[: idx + 1], epsilon)
        right = rdp(points[idx:], epsilon)
        return left[:-1] + right
    return [tuple(start), tuple(end)]


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab2))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def rasterize_polygons(shape: Tuple[int, int], polygons: Sequence[Sequence[Point]], value: int,
                       canvas: np.ndarray | None = None) -> np.ndarray:
    """Fill polygons with ``value`` into a uint8 canvas of ``shape`` (H, W)."""
    if canvas is None:
        canvas = np.zeros(shape, dtype=np.uint8)
    for poly in polygons:
        pts = np.round(np.asarray(poly, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(canvas, [pts], int(value))
    return canvas
