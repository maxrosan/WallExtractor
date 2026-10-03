"""WallPlan <-> compact text target for fine-tuning a vision-language model (Qwen3-VL).

The model sees the plan image and writes one JSON object:

    {"walls":[[x1,y1,x2,y2,t],...],"doors":[[x1,y1,x2,y2],...],"windows":[[x1,y1,x2,y2],...]}

Coordinates are integers in 0..1000 relative to the image width/height (the
convention Qwen3-VL was trained on for grounding); ``t`` is the wall thickness
in the same units of the long side. Walls are listed top to bottom, left to
right, each with its start at the top/left end, so the order is learnable.
Openings carry no wall id: it is recovered afterwards from the geometry.

Building a dataset (images resized to a long side of ``--side`` px, multiple of 32):

  python -m wallextractor.vlm_data --cubicasa data/prepared --corrections data/prepared_corr \
      --out data/vlm --side 1024 --repeat-corr 20 --restyle-prob 0.5

writes ``out/{train,val}.jsonl`` (``{"image": rel_path, "target": text, "source": ...}``)
and the resized images under ``out/images``.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

SCALE = 1000
PROMPT = ("Floor plan. List every wall as [x1,y1,x2,y2,thickness] and every door and window as [x1,y1,x2,y2] "
          "along its wall, coordinates 0-1000 relative to the image. Answer with JSON only.")


def _q(v: float) -> int:
    return int(min(SCALE, max(0, round(v))))


def _canon(x1, y1, x2, y2):
    """Start at the top/left end so the same segment always reads the same way."""
    return (x1, y1, x2, y2) if (y1, x1) <= (y2, x2) else (x2, y2, x1, y1)


def encode_plan(plan: Dict) -> str:
    """WallPlan dict (pixels) -> target text (0..1000)."""
    w, h = float(plan["image"]["width"]), float(plan["image"]["height"])
    sx, sy, st = SCALE / w, SCALE / h, SCALE / max(w, h)
    walls = []
    for wl in plan.get("walls", []):
        x1, y1, x2, y2 = _canon(_q(wl["start"][0] * sx), _q(wl["start"][1] * sy),
                                _q(wl["end"][0] * sx), _q(wl["end"][1] * sy))
        if (x1, y1) == (x2, y2):
            continue
        walls.append([x1, y1, x2, y2, max(1, int(round(wl["thickness"] * st)))])
    walls.sort(key=lambda v: (v[1], v[0], v[3], v[2]))
    out = {"walls": walls, "doors": [], "windows": []}
    for op in plan.get("openings", []):
        seg = list(_canon(_q(op["start"][0] * sx), _q(op["start"][1] * sy), _q(op["end"][0] * sx), _q(op["end"][1] * sy)))
        if seg[:2] == seg[2:]:
            continue
        out["doors" if op.get("type") == "door" else "windows"].append(seg)
    for k in ("doors", "windows"):
        out[k].sort(key=lambda v: (v[1], v[0]))
    return json.dumps(out, separators=(",", ":"))


_ROW = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
                  r"(?:\s*,\s*(-?\d+(?:\.\d+)?))?\s*\]")


def _section(text: str, key: str) -> str:
    i = text.find(f'"{key}"')
    if i < 0:
        return ""
    j = len(text)
    for other in ("walls", "doors", "windows"):
        k = text.find(f'"{other}"', i + 1)
        if other != key and 0 <= k < j:
            j = k
    return text[i:j]


def decode_text(text: str, width: int, height: int) -> Tuple[Dict, bool]:
    """Model output -> WallPlan-like dict in pixels. Tolerates truncated output (keeps complete rows).

    Returns ``(plan, valid_json)``.
    """
    try:
        json.loads(text)
        valid = True
    except ValueError:
        valid = False
    sx, sy, st = width / SCALE, height / SCALE, max(width, height) / SCALE
    walls, openings = [], []
    for m in _ROW.finditer(_section(text, "walls")):
        x1, y1, x2, y2 = (float(m.group(i)) for i in range(1, 5))
        t = float(m.group(5)) if m.group(5) else 1.0
        walls.append({"id": f"w{len(walls) + 1}", "start": [x1 * sx, y1 * sy], "end": [x2 * sx, y2 * sy],
                      "thickness": max(1.0, t * st)})
    for key, typ in (("doors", "door"), ("windows", "window")):
        for m in _ROW.finditer(_section(text, key)):
            x1, y1, x2, y2 = (float(m.group(i)) for i in range(1, 5))
            s, e = [x1 * sx, y1 * sy], [x2 * sx, y2 * sy]
            openings.append({"id": f"o{len(openings) + 1}", "type": typ, "start": s, "end": e,
                             "width": math.dist(s, e), "wall_id": _nearest_wall(walls, s, e)})
    plan = {"version": "0.1", "source": {"file": "", "page": 1, "kind": "vlm"},
            "image": {"width": int(width), "height": int(height)}, "scale": {"px_per_m": None, "method": None},
            "walls": walls, "openings": openings, "questions": []}
    return plan, valid


def _nearest_wall(walls: List[Dict], s, e) -> Optional[str]:
    """The wall whose line carries the opening: parallel, small perpendicular offset, and the opening on the
    wall or in the gap just past one of its ends (walls are usually interrupted at doors and windows)."""
    mx, my = (s[0] + e[0]) / 2, (s[1] + e[1]) / 2
    ow = math.dist(s, e) or 1e-9
    ox, oy = (e[0] - s[0]) / ow, (e[1] - s[1]) / ow
    best, best_d = None, None
    for w in walls:
        (ax, ay), (bx, by) = w["start"], w["end"]
        L = math.hypot(bx - ax, by - ay) or 1e-9
        ux, uy = (bx - ax) / L, (by - ay) / L
        if abs(ux * ox + uy * oy) < math.cos(math.radians(10)):
            continue
        t = (mx - ax) * ux + (my - ay) * uy
        d = abs(-(mx - ax) * uy + (my - ay) * ux)
        if -ow <= t <= L + ow and d <= max(1.5 * w["thickness"], 4.0) and (best_d is None or d < best_d):
            best, best_d = w["id"], d
    return best


def scale_plan(plan: Dict, f: float, size: Tuple[int, int]) -> Dict:
    """Copy of a WallPlan dict with every coordinate multiplied by ``f`` and the image set to ``size``."""
    p = json.loads(json.dumps(plan))
    p["image"]["width"], p["image"]["height"] = int(size[0]), int(size[1])
    for wl in p.get("walls", []):
        wl["start"] = [v * f for v in wl["start"]]
        wl["end"] = [v * f for v in wl["end"]]
        wl["thickness"] = wl["thickness"] * f
        wl.pop("polygon", None)
    for op in p.get("openings", []):
        op["start"] = [v * f for v in op["start"]]
        op["end"] = [v * f for v in op["end"]]
        op.pop("polygon", None)
    if p.get("scale", {}).get("px_per_m"):
        p["scale"]["px_per_m"] *= f
    return p


def fit_side(w: int, h: int, side: int, multiple: int = 32) -> Tuple[int, int, float]:
    """Size with the long side ``side`` (rounded to ``multiple``) and the uniform factor used."""
    f = side / max(w, h)
    nw = max(multiple, int(round(w * f / multiple)) * multiple)
    nh = max(multiple, int(round(h * f / multiple)) * multiple)
    return nw, nh, f


def _items(folder: str) -> List[Tuple[str, str, Optional[str]]]:
    out = []
    for jp in sorted(glob.glob(os.path.join(folder, "*.json"))):
        ip = jp[:-5] + ".png"
        if os.path.isfile(ip):
            mp = jp[:-5] + "_mask.png"
            out.append((ip, jp, mp if os.path.isfile(mp) else None))
    return out


def build(cubicasa: Optional[str], corrections: Optional[str], out: str, side: int = 1024, repeat_corr: int = 20,
          restyle_prob: float = 0.0, limit_cubicasa: int = 0, seed: int = 0, log=print) -> Dict[str, int]:
    from .styles import random_style, restyle_walls

    rng = random.Random(seed)
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)
    stats: Dict[str, int] = {}
    for split in ("train", "val"):
        rows = []
        sources = []
        if cubicasa:
            items = _items(os.path.join(cubicasa, split))
            if limit_cubicasa:
                items = items[:limit_cubicasa if split == "train" else max(1, limit_cubicasa // 4)]
            sources.append(("cubicasa", items, 1))
        if corrections and os.path.isdir(os.path.join(corrections, split)):
            sources.append(("editor", _items(os.path.join(corrections, split)), repeat_corr if split == "train" else 1))
        for name, items, rep in sources:
            for ip, jp, mp in items:
                with open(jp, encoding="utf-8") as f:
                    plan = json.load(f)
                img = Image.open(ip).convert("RGB")
                nw, nh, _ = fit_side(img.width, img.height, side)
                fx = nw / img.width  # x and y factors differ by < 1/32 after rounding; use x for lengths
                base = os.path.splitext(os.path.basename(ip))[0]
                arr = np.asarray(img)
                # restyle only CubiCasa (solid walls) in training; the editor plans are already real drawings
                variants = [("", arr)]
                if name == "cubicasa" and split == "train" and mp and rng.random() < restyle_prob:
                    mask = np.asarray(Image.open(mp))
                    style = random_style(rng, exclude_solid=True)
                    variants = [("_" + style, restyle_walls(arr, mask, style, rng=rng))]
                for suffix, a in variants:
                    rel = os.path.join("images", f"{name}_{split}_{base}{suffix}.png")
                    Image.fromarray(a).resize((nw, nh), Image.BILINEAR).save(os.path.join(out, rel))
                    p = scale_plan(plan, fx, (nw, nh))
                    target = encode_plan(p)
                    for _ in range(rep):
                        rows.append({"image": rel.replace(os.sep, "/"), "target": target, "source": name,
                                     "plan": base})
        if split == "train":
            rng.shuffle(rows)
        with open(os.path.join(out, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[split] = len(rows)
        log(f"[vlm_data] {split}: {len(rows)} rows")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cubicasa", default=None, help="prepared CubiCasa folder (train/, val/)")
    ap.add_argument("--corrections", default=None, help="prepared editor corrections (wallextractor.annotations)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--side", type=int, default=1024)
    ap.add_argument("--repeat-corr", type=int, default=20)
    ap.add_argument("--restyle-prob", type=float, default=0.0)
    ap.add_argument("--limit-cubicasa", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    build(a.cubicasa, a.corrections, a.out, a.side, a.repeat_corr, a.restyle_prob, a.limit_cubicasa, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
