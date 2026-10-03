"""Evaluate a trained checkpoint (HF dir) on a prepared split, optionally with restyled walls.

Usage:
  python -m wallextractor.eval_seg --model runs/b1/best --data data/prepared --split val --size 768 \
      --styles solid,hatch45,outline,gray,cross
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

from .cubicasa import CLASSES
from .dataset import PlanSegDataset
from .train_seg import evaluate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--styles", default="solid,hatch45,outline,gray,cross")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from transformers import SegformerForSemanticSegmentation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(device).eval()
    results = {}
    for style in args.styles.split(","):
        ds = PlanSegDataset(os.path.join(args.data, args.split), size=args.size, augment=False,
                            fixed_style=None if style == "solid" else style)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)
        m = evaluate(model, dl, device, len(CLASSES))
        results[style] = {k: round(v, 4) for k, v in m.items()}
        print(f"[eval_seg] {style:<9} " + " ".join(f"{c}={m['iou_' + c]:.3f}" for c in CLASSES[1:]), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
