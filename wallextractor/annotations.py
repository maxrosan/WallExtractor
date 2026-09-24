"""Corrected plans (from the editor) as training data.

The editor exports pairs ``<id>.png`` + ``<id>.json`` (WallPlan JSON with
walls as segments + thickness and openings as segments on a wall). This
module rasterizes those into the 4-class masks used by ``train_seg`` and
writes them in the prepared-dataset layout, so corrected plans train
together with CubiCasa5K.

Usage:
  python -m wallextractor.annotations --src data/corrections --out data/prepared_corr [--val-fraction 0.2]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from .cubicasa import BG, CLASSES, DOOR, WALL, WINDOW


def plan_to_mask(plan: Dict, shape: Optional[tuple] = None) -> np.ndarray:
    """Rasterize a segment-based WallPlan into a uint8 mask (0 bg, 1 wall, 2 door, 3 window)."""
    h = shape[0] if shape else int(plan["image"]["height"])
    w = shape[1] if shape else int(plan["image"]["width"])
    mask = np.full((h, w), BG, np.uint8)
    thick_by_wall = {}
    for wl in plan.get("walls", []):
        th = max(1, int(round(wl["thickness"])))
        thick_by_wall[wl["id"]] = th
        if wl.get("polygon"):
            pts = np.round(np.asarray(wl["polygon"], dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], WALL)
        else:
            a = tuple(int(round(v)) for v in wl["start"])
            b = tuple(int(round(v)) for v in wl["end"])
            cv2.line(mask, a, b, WALL, th)
    for op in plan.get("openings", []):
        cls = DOOR if op.get("type") == "door" else WINDOW
        th = thick_by_wall.get(op.get("wall_id"), 6)
        if op.get("polygon"):
            pts = np.round(np.asarray(op["polygon"], dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], cls)
        else:
            a = tuple(int(round(v)) for v in op["start"])
            b = tuple(int(round(v)) for v in op["end"])
            cv2.line(mask, a, b, cls, max(1, th))
    return mask


def prepare(src: str, out: str, val_fraction: float = 0.2, seed: int = 0, log=print) -> dict:
    pairs: List[tuple] = []
    for jp in sorted(glob.glob(os.path.join(src, "*.json"))):
        if os.path.basename(jp) == "manifest.json":
            continue
        ip = jp[:-5] + ".png"
        if os.path.isfile(ip):
            pairs.append((ip, jp))
    if not pairs:
        raise SystemExit(f"[annotations] no png/json pairs in {src}")
    rnd = random.Random(seed)
    rnd.shuffle(pairs)
    n_val = int(round(len(pairs) * val_fraction)) if len(pairs) >= 5 else 0
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}
    stats = {}
    for split, items in splits.items():
        d = os.path.join(out, split)
        os.makedirs(d, exist_ok=True)
        for ip, jp in items:
            with open(jp, encoding="utf-8") as f:
                plan = json.load(f)
            img = Image.open(ip).convert("RGB")
            mask = plan_to_mask(plan, shape=(img.size[1], img.size[0]))
            name = os.path.splitext(os.path.basename(ip))[0]
            img.save(os.path.join(d, f"{name}.png"))
            Image.fromarray(mask).save(os.path.join(d, f"{name}_mask.png"))
            with open(os.path.join(d, f"{name}.json"), "w", encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False)
        stats[split] = len(items)
        log(f"[annotations] {split}: {len(items)} plans")
    with open(os.path.join(out, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(CLASSES, f)
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with <id>.png + <id>.json (editor export, unzipped)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    prepare(args.src, args.out, args.val_fraction, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
