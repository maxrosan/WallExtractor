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
from .pdf import extract_vector_primitives, is_vector, render_page, wall_candidates_from_primitives
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


def extract(pdf_path: str, model_path: Optional[str], page: int = 1, max_side: int = 1024, size: int = 512,
            prefer_vector: bool = True):
    rgb, px_per_pt = render_page(pdf_path, page=page, max_side=max_side)
    plan = None
    if prefer_vector and is_vector(pdf_path, page):
        prims = extract_vector_primitives(pdf_path, page, px_per_pt)
        polys = wall_candidates_from_primitives(prims)
        if polys:
            from .geometry import rasterize_polygons

            mask = rasterize_polygons(rgb.shape[:2], polys, 1)
            plan = mask_to_plan(mask, source_file=os.path.basename(pdf_path), page=page, kind="vector")
    if plan is None:
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
    print(json.dumps({"kind": plan.source.kind, "walls": len(plan.walls), "openings": len(plan.openings)}),
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
