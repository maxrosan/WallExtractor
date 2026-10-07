"""Matching and F1 of walls, doors and windows between a predicted and a gold plan, shared by
``wallextractor.eval_vlm`` (Qwen) and ``scripts/eval_pipeline.py`` (raster drafts). No torch here.

Walls match when both ends are within the tolerance (either direction); doors and windows when the type agrees,
the midpoints are within the tolerance, the directions within 20 degrees and the lengths within the tolerance.
``length_counts`` measures how much wall length of each side lies within the tolerance of the other.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


def _match(pairs) -> int:
    used_p, used_g, n = set(), set(), 0
    for _d, i, j in sorted(pairs):
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            n += 1
    return n


def match_walls(pred: List[Dict], gold: List[Dict], tol: float) -> int:
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            d = min(max(math.dist(p["start"], g["start"]), math.dist(p["end"], g["end"])),
                    max(math.dist(p["start"], g["end"]), math.dist(p["end"], g["start"])))
            if d <= tol:
                pairs.append((d, i, j))
    return _match(pairs)


def _sample(walls: List[Dict], step: float = 2.0):
    import numpy as np

    out = []
    for w in walls:
        a, b = w["start"], w["end"]
        n = max(2, int(math.dist(a, b) / step))
        for k in range(n + 1):
            out.append((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n))
    return np.array(out).reshape(-1, 2)


def _within(points, walls: List[Dict], tol: float) -> int:
    import numpy as np

    if not len(points) or not walls:
        return 0
    best = np.full(len(points), np.inf)
    for w in walls:
        a, b = np.array(w["start"], float), np.array(w["end"], float)
        ab = b - a
        t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-9), 0, 1)
        best = np.minimum(best, np.linalg.norm(points - (a + t[:, None] * ab), axis=1))
    return int((best <= tol).sum())


def length_counts(pred: List[Dict], gold: List[Dict], tol: float) -> Tuple[int, int, int, int]:
    """(pred samples near gold, pred samples, gold samples near pred, gold samples), 2 px apart."""
    P, G = _sample(pred), _sample(gold)
    return _within(P, gold, tol), len(P), _within(G, pred, tol), len(G)


def _dir(o):
    return math.degrees(math.atan2(o["end"][1] - o["start"][1], o["end"][0] - o["start"][0])) % 180


def match_openings(pred: List[Dict], gold: List[Dict], tol: float) -> int:
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if p["type"] != g["type"]:
                continue
            mp = ((p["start"][0] + p["end"][0]) / 2, (p["start"][1] + p["end"][1]) / 2)
            mg = ((g["start"][0] + g["end"][0]) / 2, (g["start"][1] + g["end"][1]) / 2)
            da = abs(_dir(p) - _dir(g))
            da = min(da, 180 - da)
            d = math.dist(mp, mg)
            if d <= tol and da <= 20 and abs(math.dist(p["start"], p["end"]) - math.dist(g["start"], g["end"])) <= tol:
                pairs.append((d, i, j))
    return _match(pairs)


def f1(tp: int, n_pred: int, n_gold: int) -> Tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)
