"""Train a SegFormer wall/door/window segmenter.

The model is meant to run on a CPU-only server with a few GB of RAM, so the
default encoder is MiT-B1 (13.7M parameters). Metrics per epoch go to
``<out>/metrics.jsonl``; the best checkpoint (by wall IoU) is saved to
``<out>/best`` and optionally exported to ONNX.

Usage:
  python -m wallextractor.train_seg --data data/prepared --out runs/b1 --epochs 15 --size 512 --batch 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .cubicasa import CLASSES
from .dataset import IGNORE_INDEX, PlanSegDataset


def build_model(name: str, num_labels: int):
    from transformers import SegformerForSemanticSegmentation

    return SegformerForSemanticSegmentation.from_pretrained(
        name, num_labels=num_labels, ignore_mismatched_sizes=True,
        id2label={i: c for i, c in enumerate(CLASSES)}, label2id={c: i for i, c in enumerate(CLASSES)},
    )


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int) -> Dict[str, float]:
    model.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(pixel_values=x).logits
        logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1)
        valid = y != IGNORE_INDEX
        idx = y[valid] * num_classes + pred[valid]
        conf += torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    conf = conf.double()
    tp = conf.diag()
    iou = tp / (conf.sum(0) + conf.sum(1) - tp).clamp(min=1)
    out = {f"iou_{CLASSES[i]}": float(iou[i]) for i in range(num_classes)}
    out["miou"] = float(iou[1:].mean())  # foreground classes only
    out["pixel_acc"] = float(tp.sum() / conf.sum().clamp(min=1))
    return out


def train(args) -> Dict[str, float]:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "metrics.jsonl")

    train_ds = PlanSegDataset(os.path.join(args.data, "train"), size=args.size, augment=True,
                              restyle_prob=args.restyle_prob)
    val_ds = PlanSegDataset(os.path.join(args.data, "val"), size=args.size, augment=False)
    if args.limit_train:
        train_ds.samples = train_ds.samples[: args.limit_train]
    if args.limit_val:
        val_ds.samples = val_ds.samples[: args.limit_val]
    workers = min(args.workers, os.cpu_count() or 1)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=workers, pin_memory=True,
                          drop_last=len(train_ds) >= args.batch)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=workers, pin_memory=True)
    # Second validation view: same plans with walls redrawn in a fixed style (domain-shift probe).
    restyled_dls = {}
    for style in [s for s in args.eval_styles.split(",") if s]:
        ds = PlanSegDataset(os.path.join(args.data, "val"), size=args.size, augment=False, fixed_style=style)
        ds.samples = val_ds.samples
        restyled_dls[style] = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=workers,
                                         pin_memory=True)

    num_classes = len(CLASSES)
    model = build_model(args.model, num_classes).to(device)
    class_weights = torch.tensor([float(w) for w in args.class_weights.split(",")], device=device)
    assert len(class_weights) == num_classes, "class-weights must have one value per class"

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, args.epochs * len(train_dl))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (1 - min(s, total_steps) / total_steps) ** 0.9)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    print(f"[train] device={device} model={args.model} train={len(train_ds)} val={len(val_ds)} "
          f"size={args.size} batch={args.batch} epochs={args.epochs} steps/epoch={len(train_dl)}", flush=True)
    t0 = time.time()
    best = {"iou_wall": -1.0}
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(pixel_values=x).logits
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss = F.cross_entropy(logits.float(), y, weight=class_weights, ignore_index=IGNORE_INDEX)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += float(loss) * x.shape[0]
            n += x.shape[0]
            step += 1
            if step % args.log_every == 0:
                print(f"[train] epoch {epoch} step {step} loss {float(loss):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
            if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
                break
        metrics = evaluate(model, val_dl, device, num_classes)
        for style, dl in restyled_dls.items():
            m = evaluate(model, dl, device, num_classes)
            metrics.update({f"{k}@{style}": v for k, v in m.items() if k in ("iou_wall", "iou_door", "iou_window")})
        metrics.update({"epoch": epoch, "train_loss": running / max(n, 1), "elapsed_s": round(time.time() - t0)})
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        print("[eval] " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}),
              flush=True)
        if metrics["iou_wall"] > best["iou_wall"]:
            best = metrics
            model.save_pretrained(os.path.join(args.out, "best"))
        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            print("[train] time budget reached, stopping", flush=True)
            break

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"best": best, "args": vars(args), "train_samples": len(train_ds), "val_samples": len(val_ds),
                   "elapsed_s": round(time.time() - t0)}, f, indent=1)
    if args.export_onnx and best["iou_wall"] >= 0:
        export_onnx(os.path.join(args.out, "best"), os.path.join(args.out, "best.onnx"), args.size)
    return best


def export_onnx(model_dir: str, onnx_path: str, size: int) -> None:
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(model_dir).eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(pixel_values=x).logits

    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(Wrapper(model).eval(), dummy, onnx_path, opset_version=17, input_names=["pixel_values"],
                      output_names=["logits"], dynamic_axes={"pixel_values": {0: "b", 2: "h", 3: "w"},
                                                             "logits": {0: "b", 2: "h4", 3: "w4"}})
    # Newer exporters write weights to a side file; fold them into one self-contained .onnx for deployment.
    import onnx

    m = onnx.load(onnx_path)
    onnx.save(m, onnx_path, save_as_external_data=False)
    side = onnx_path + ".data"
    if os.path.isfile(side):
        os.remove(side)
    print(f"[export] wrote {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)", flush=True)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="nvidia/mit-b1")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--class-weights", default="0.5,1.0,2.0,2.0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-minutes", type=float, default=0, help="stop after this wall-clock budget")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--restyle-prob", type=float, default=0.0,
                    help="fraction of training samples whose walls are redrawn in a random style")
    ap.add_argument("--eval-styles", default="", help="comma list of styles for extra validation views, e.g. hatch45,outline")
    ap.add_argument("--export-onnx", action="store_true")
    args = ap.parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
