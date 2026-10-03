"""CubiCasa5K -> WallPlan JSON + segmentation masks.

CubiCasa5K (Zenodo 2613548, CC BY-NC 4.0) ships one folder per plan with
``F1_scaled.png`` and ``model.svg``. The SVG coordinates are already in pixels
of ``F1_scaled.png`` (verified on samples: wall polygons land on the dark
pixels without any rescaling), so masks are rasterized directly.

Elements used:
  <g id="Wall" class="Wall External|Internal ..."><polygon points=.../>
  <g id="Door" class="Door ..."><polygon .../>
  <g id="Window" class="Window ..."><polygon .../>

Doors and windows are nested inside the wall group they cut; their polygon
is the opening's footprint inside the wall.

Usage:
  python -m wallextractor.cubicasa --root data/cubicasa5k --out data/prepared [--split train --limit 400]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Iterable, List, Optional, Sequence, Tuple
from xml.dom import minidom

import numpy as np
from PIL import Image

from .geometry import polygon_to_segment, rasterize_polygons
from .schema import ImageInfo, Opening, Scale, Source, Wall, WallPlan

# Mask classes shared with the training code.
CLASSES = ["background", "wall", "door", "window"]
BG, WALL, DOOR, WINDOW = 0, 1, 2, 3

Point = Tuple[float, float]
Matrix = Tuple[float, float, float, float, float, float]  # a b c d e f (SVG order)

IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _mat_mul(m1: Matrix, m2: Matrix) -> Matrix:
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _parse_transform(text: str) -> Matrix:
    m = IDENTITY
    for name, args in re.findall(r"(matrix|translate|scale)\s*\(([^)]*)\)", text or ""):
        nums = [float(x) for x in re.split(r"[\s,]+", args.strip()) if x]
        if name == "matrix" and len(nums) == 6:
            t = tuple(nums)  # type: ignore[assignment]
        elif name == "translate":
            t = (1.0, 0.0, 0.0, 1.0, nums[0], nums[1] if len(nums) > 1 else 0.0)
        elif name == "scale":
            sx = nums[0]
            sy = nums[1] if len(nums) > 1 else sx
            t = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        else:
            continue
        m = _mat_mul(m, t)  # type: ignore[arg-type]
    return m


def _ancestor_transform(node) -> Matrix:
    chain = []
    n = node
    while n is not None and getattr(n, "getAttribute", None) is not None:
        t = n.getAttribute("transform")
        if t:
            chain.append(_parse_transform(t))
        n = n.parentNode
    m = IDENTITY
    for t in reversed(chain):
        m = _mat_mul(m, t)
    return m


def _apply(m: Matrix, p: Point) -> Point:
    a, b, c, d, e, f = m
    x, y = p
    return (a * x + c * y + e, b * x + d * y + f)


def _polygon_points(g) -> Optional[List[Point]]:
    polys = g.getElementsByTagName("polygon")
    if not polys:
        return None
    raw = polys[0].getAttribute("points").strip()
    pts: List[Point] = []
    for chunk in raw.split():
        xy = chunk.split(",")
        if len(xy) == 2:
            pts.append((float(xy[0]), float(xy[1])))
    if len(pts) < 3:
        return None
    m = _ancestor_transform(polys[0])
    if m != IDENTITY:
        pts = [_apply(m, p) for p in pts]
    return pts


def _direct_children_with_id(g, wanted: str):
    return [c for c in g.childNodes if getattr(c, "getAttribute", None) and c.getAttribute("id") == wanted]


def parse_svg(svg_path: str, image_size: Tuple[int, int], source_file: str = "") -> WallPlan:
    """Parse a CubiCasa5K ``model.svg`` into a WallPlan (pixels of F1_scaled.png)."""
    doc = minidom.parse(svg_path)
    width, height = image_size
    plan = WallPlan(
        source=Source(file=source_file or svg_path, page=1, kind="vector"),
        image=ImageInfo(width=width, height=height),
        scale=Scale(px_per_m=None, method=None),
    )
    wall_groups = [g for g in doc.getElementsByTagName("g") if g.getAttribute("id") == "Wall"]
    for wi, g in enumerate(wall_groups):
        pts = _polygon_points(g)
        if pts is None:
            continue
        start, end, thickness = polygon_to_segment(pts)
        cls = g.getAttribute("class").split()
        kind = None
        if len(cls) > 1:
            kind = cls[1].lower()
        wall_id = f"w{wi + 1}"
        plan.walls.append(Wall(id=wall_id, start=start, end=end, thickness=thickness, kind=kind, polygon=pts))
        for kind_id, otype in (("Door", "door"), ("Window", "window")):
            for og in g.getElementsByTagName("g"):
                if og.getAttribute("id") != kind_id:
                    continue
                opts = _polygon_points(og)
                if opts is None:
                    continue
                s, e, w = polygon_to_segment(opts)
                plan.openings.append(
                    Opening(id=f"o{len(plan.openings) + 1}", type=otype, start=s, end=e, width=w,
                            wall_id=wall_id, polygon=opts)
                )
    # Openings that are not nested in a wall group (rare) are still collected.
    seen = {tuple(map(tuple, o.polygon)) for o in plan.openings if o.polygon}
    for kind_id, otype in (("Door", "door"), ("Window", "window")):
        for og in doc.getElementsByTagName("g"):
            if og.getAttribute("id") != kind_id:
                continue
            opts = _polygon_points(og)
            if opts is None or tuple(map(tuple, opts)) in seen:
                continue
            s, e, w = polygon_to_segment(opts)
            plan.openings.append(Opening(id=f"o{len(plan.openings) + 1}", type=otype, start=s, end=e,
                                         width=w, wall_id=None, polygon=opts))
    return plan


def plan_to_mask(plan: WallPlan) -> np.ndarray:
    """Rasterize a WallPlan into a uint8 mask with classes BG/WALL/DOOR/WINDOW."""
    shape = (plan.image.height, plan.image.width)
    mask = rasterize_polygons(shape, [w.polygon for w in plan.walls if w.polygon], WALL)
    rasterize_polygons(shape, [o.polygon for o in plan.openings if o.polygon and o.type == "window"], WINDOW, mask)
    rasterize_polygons(shape, [o.polygon for o in plan.openings if o.polygon and o.type == "door"], DOOR, mask)
    return mask


def read_split(root: str, split: str) -> List[str]:
    path = os.path.join(root, f"{split}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().strip("/") for line in f if line.strip()]


def prepare(root: str, out: str, splits: Sequence[str] = ("train", "val"), limit: Optional[int] = None,
            log=print) -> dict:
    """Convert every available plan folder of the given splits.

    Writes ``out/<split>/<name>.png`` (image), ``<name>_mask.png`` and
    ``<name>.json`` and returns per-split counts.
    """
    stats = {}
    for split in splits:
        folders = read_split(root, split)
        if limit:
            folders = folders[:limit]
        os.makedirs(os.path.join(out, split), exist_ok=True)
        done = skipped = 0
        for rel in folders:
            src_dir = os.path.join(root, rel)
            img_path = os.path.join(src_dir, "F1_scaled.png")
            svg_path = os.path.join(src_dir, "model.svg")
            if not (os.path.isfile(img_path) and os.path.isfile(svg_path)):
                skipped += 1
                continue
            name = rel.replace("/", "_")
            try:
                img = Image.open(img_path).convert("RGB")
                plan = parse_svg(svg_path, img.size, source_file=rel)
                mask = plan_to_mask(plan)
            except Exception as exc:  # noqa: BLE001 - keep converting the rest
                log(f"[cubicasa] {rel}: {exc}")
                skipped += 1
                continue
            img.save(os.path.join(out, split, f"{name}.png"))
            Image.fromarray(mask).save(os.path.join(out, split, f"{name}_mask.png"))
            plan.save(os.path.join(out, split, f"{name}.json"))
            done += 1
        stats[split] = {"converted": done, "skipped": skipped}
        log(f"[cubicasa] {split}: converted {done}, skipped {skipped}")
    with open(os.path.join(out, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(CLASSES, f)
    return stats


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder that contains train.txt/val.txt and the plan folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", action="append", help="train/val/test (repeatable); default train and val")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)
    prepare(args.root, args.out, args.split or ("train", "val"), args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
