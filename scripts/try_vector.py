"""Run the vector wall extractor on PDFs and write overlays for inspection.

Usage: python scripts/try_vector.py out_dir file1.pdf [file2.pdf ...] [--page N] [--px-per-pt 3]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import pymupdf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wallextractor.vector_walls import extract_walls  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--px-per-pt", type=float, default=3.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for path in args.pdfs:
        name = os.path.basename(path)[:8]
        doc = pymupdf.open(path)
        page = doc[args.page - 1]
        walls, region = extract_walls(page)
        x0, y0, x1, y1 = region.rect
        s = args.px_per_pt
        pix = page.get_pixmap(matrix=pymupdf.Matrix(s, s), clip=pymupdf.Rect(*region.rect), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
        for w in walls:
            a = (int(round((w.start[0] - x0) * s)), int(round((w.start[1] - y0) * s)))
            b = (int(round((w.end[0] - x0) * s)), int(round((w.end[1] - y0) * s)))
            cv2.line(img, a, b, (30, 30, 220), max(1, int(round(w.thickness * s * 0.6))))
        cv2.imwrite(os.path.join(args.out, f"{name}_vecwalls.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        total_m = sum(w.length for w in walls) / region.pt_per_m if region.pt_per_m else None
        th = [w.thickness / region.pt_per_m for w in walls] if region.pt_per_m else []
        info = {"file": name, "region_pt": [round(v, 1) for v in region.rect], "label": region.label,
                "scale": region.scale_denominator, "pen": region.wall_pen_width, "walls": len(walls),
                "total_length_m": round(total_m, 1) if total_m else None,
                "thickness_m_median": round(float(np.median(th)), 3) if th else None, "notes": region.notes}
        print(json.dumps(info, ensure_ascii=False))
        with open(os.path.join(args.out, f"{name}_walls.json"), "w", encoding="utf-8") as f:
            json.dump([{"start": w.start, "end": w.end, "thickness": w.thickness} for w in walls], f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
