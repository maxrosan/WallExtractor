"""PDF -> WallPlan JSON with the trained segmenter (ONNX or PyTorch).

Usage:
  python -m wallextractor.infer plan.pdf --model runs/b1/best.onnx --out plan.json [--overlay plan.png]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
from PIL import Image

from .imageops import normalize, resize_pad
from .pdf import render_page
from .vectorize import draw_plan, mask_to_plan


class OnnxSegmenter:
    def __init__(self, path: str, threads: int = 2):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def __call__(self, img: np.ndarray) -> np.ndarray:
        logits = self.sess.run(None, {self.input: normalize(img)})[0][0]  # C x h/4 x w/4
        return logits


class TorchSegmenter:
    def __init__(self, model_dir: str):
        import torch
        from transformers import SegformerForSemanticSegmentation

        self.torch = torch
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_dir).eval()

    def __call__(self, img: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(normalize(img))
        with torch.no_grad():
            return self.model(pixel_values=x).logits[0].numpy()


def load_segmenter(path: str):
    return OnnxSegmenter(path) if path.endswith(".onnx") else TorchSegmenter(path)


def segment_image(segmenter, rgb: np.ndarray, size: int = 512) -> np.ndarray:
    """Return a class mask at the resolution of ``rgb``."""
    import cv2

    h, w = rgb.shape[:2]
    padded, _, scale = resize_pad(Image.fromarray(rgb), None, size)
    logits = segmenter(padded)
    up = cv2.resize(logits.transpose(1, 2, 0), (size, size), interpolation=cv2.INTER_LINEAR)
    pred = up.argmax(-1).astype(np.uint8)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    pred = pred[:nh, :nw]
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)


def extract_vector(pdf_path: str, page: int = 1, max_side: int = 2048, min_walls: int = 4):
    """Vector branch: locate the floor plan on the sheet, pair thick-pen lines into walls.

    Returns ``(plan, rgb)`` with coordinates in pixels of the rendered plan
    region, or ``None`` when the page does not look like a CAD export.
    """
    import pymupdf

    from .schema import ImageInfo, Opening, Scale, Source, Wall, WallPlan
    from .vector_walls import extract_walls

    with pymupdf.open(pdf_path) as doc:
        pg = doc[page - 1]
        if len(pg.get_drawings()) < 50:
            return None
        walls_pt, openings_pt, region = extract_walls(pg)
        if len(walls_pt) < min_walls:
            return None
        x0, y0, x1, y1 = region.rect
        s = max_side / max(x1 - x0, y1 - y0)
        pix = pg.get_pixmap(matrix=pymupdf.Matrix(s, s), clip=pymupdf.Rect(*region.rect), alpha=False,
                            colorspace=pymupdf.csRGB)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
    walls = [Wall(id=w.id, start=((w.start[0] - x0) * s, (w.start[1] - y0) * s),
                  end=((w.end[0] - x0) * s, (w.end[1] - y0) * s), thickness=w.thickness * s) for w in walls_pt]
    openings = [Opening(id=o.id, type=o.type, start=((o.start[0] - x0) * s, (o.start[1] - y0) * s),
                        end=((o.end[0] - x0) * s, (o.end[1] - y0) * s), width=o.width * s, wall_id=o.wall_id,
                        confidence=o.confidence)
                for o in openings_pt]
    plan = WallPlan(
        source=Source(file=os.path.basename(pdf_path), page=page, kind="vector",
                      region_pt=[round(v, 2) for v in region.rect], px_per_pt=round(s, 4)),
        image=ImageInfo(width=int(rgb.shape[1]), height=int(rgb.shape[0]), dpi=round(72.0 * s, 2)),
        walls=walls,
        openings=openings,
        scale=Scale(px_per_m=round(region.pt_per_m * s, 3) if region.pt_per_m else None,
                    method=region.scale_method),
    )
    plan.notes = region.notes  # type: ignore[attr-defined]
    return plan, rgb


def extract(pdf_path: str, model_path: Optional[str], page: int = 1, max_side: int = 1024, size: int = 512,
            prefer_vector: bool = True):
    plan = None
    rgb = None
    if prefer_vector:
        res = extract_vector(pdf_path, page=page, max_side=max(max_side, 2048))
        if res is not None:
            plan, rgb = res
    if plan is None:
        rgb, px_per_pt = render_page(pdf_path, page=page, max_side=max_side)
        if not model_path:
            raise SystemExit("raster branch needs --model")
        mask = segment_image(load_segmenter(model_path), rgb, size=size)
        plan = mask_to_plan(mask, source_file=os.path.basename(pdf_path), page=page, kind="raster")
        plan.image.dpi = round(72.0 * px_per_pt, 2)
    return plan, rgb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--model", default=None, help="best.onnx or a HF model dir")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--no-vector", action="store_true", help="always use the raster branch")
    ap.add_argument("--out", default=None)
    ap.add_argument("--overlay", default=None)
    args = ap.parse_args(argv)
    plan, rgb = extract(args.pdf, args.model, args.page, args.max_side, args.size, prefer_vector=not args.no_vector)
    text = plan.to_json(indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)
    if args.overlay:
        Image.fromarray(draw_plan(rgb, plan)).save(args.overlay)
    print(json.dumps({"kind": plan.source.kind, "walls": len(plan.walls), "openings": len(plan.openings),
                      "px_per_m": plan.scale.px_per_m, "notes": getattr(plan, "notes", None)}, ensure_ascii=False),
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
