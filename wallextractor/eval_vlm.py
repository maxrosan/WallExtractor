"""Evaluate a Qwen3-VL (base or + LoRA adapter) on the validation plans of ``wallextractor.vlm_data``.

Walls match when both ends are within the tolerance (either direction); doors and windows when the type
agrees, the midpoints are within the tolerance, the directions within 20 degrees and the lengths within the
tolerance. Tolerances are fractions of the image's long side (F1@0.015 and F1@0.05, as in fpvec-lab).
Also reports how often the answer is valid JSON. Predictions go to ``--out`` for inspection.

Usage:
  python -m wallextractor.eval_vlm --data data/vlm --adapter runs/vlm4b/best --out runs/vlm4b/eval.json
  python -m wallextractor.eval_vlm --data data/vlm --model Qwen/Qwen3-VL-4B-Instruct --out runs/base_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
from PIL import Image

from .train_vlm import load_rows, messages
from .vlm_data import decode_text


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


def load(model_name: str, adapter: str = None):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if adapter:
        cfg = json.load(open(os.path.join(adapter, "adapter_config.json"), encoding="utf-8"))
        model_name = cfg.get("base_model_name_or_path") or model_name
    processor = AutoProcessor.from_pretrained(adapter if adapter and os.path.isfile(
        os.path.join(adapter, "preprocessor_config.json")) else model_name)
    model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.bfloat16, attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    return model.to("cuda").eval(), processor


@torch.no_grad()
def generate(model, processor, img: Image.Image, max_new_tokens: int) -> str:
    text = processor.apply_chat_template(messages(), tokenize=False, add_generation_prompt=True)
    enc = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA folder from train_vlm (omit to test the base model)")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--split", default="val")
    ap.add_argument("--source", default=None, help="only rows of this source (editor / cubicasa)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=3000)
    ap.add_argument("--tols", default="0.015,0.05")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = [r for r in load_rows(os.path.join(a.data, f"{a.split}.jsonl"), unique=True)
            if not a.source or r["source"] == a.source]
    if a.limit:
        rows = rows[:a.limit]
    model, processor = load(a.model, a.adapter)
    tols = [float(t) for t in a.tols.split(",")]
    tot = {t: {k: [0, 0, 0] for k in ("wall", "door", "window")} for t in tols}
    per_source: Dict[str, Dict] = {}
    preds = []
    valid = 0
    t0 = time.time()
    for n, r in enumerate(rows, 1):
        img = Image.open(os.path.join(a.data, r["image"])).convert("RGB")
        text = generate(model, processor, img, a.max_new_tokens)
        pred, ok = decode_text(text, img.width, img.height)
        gold, _ = decode_text(r["target"], img.width, img.height)
        valid += ok
        side = max(img.width, img.height)
        row = {"image": r["image"], "source": r["source"], "valid_json": ok, "text": text}
        for t in tols:
            tol = t * side
            src = per_source.setdefault(r["source"], {t2: {k: [0, 0, 0] for k in ("wall", "door", "window")}
                                                       for t2 in tols})
            tp = match_walls(pred["walls"], gold["walls"], tol)
            for acc in (tot[t], src[t]):
                acc["wall"][0] += tp
                acc["wall"][1] += len(pred["walls"])
                acc["wall"][2] += len(gold["walls"])
            for typ in ("door", "window"):
                P = [o for o in pred["openings"] if o["type"] == typ]
                G = [o for o in gold["openings"] if o["type"] == typ]
                tp = match_openings(P, G, tol)
                for acc in (tot[t], src[t]):
                    acc[typ][0] += tp
                    acc[typ][1] += len(P)
                    acc[typ][2] += len(G)
        preds.append(row)
        print(f"[eval] {n}/{len(rows)} {r['source']} valid={ok} walls {len(pred['walls'])}/{len(gold['walls'])} "
              f"openings {len(pred['openings'])}/{len(gold['openings'])} {time.time() - t0:.0f}s", flush=True)

    def table(acc):
        return {str(t): {k: dict(zip(("P", "R", "F1"), (round(v, 3) for v in f1(*acc[t][k])))) for k in acc[t]}
                for t in acc}

    report = {"plans": len(rows), "valid_json": valid / max(1, len(rows)), "all": table(tot),
              "by_source": {s: table(v) for s, v in per_source.items()}, "adapter": a.adapter, "model": a.model,
              "elapsed_s": round(time.time() - t0)}
    print(json.dumps({k: report[k] for k in ("plans", "valid_json", "all", "by_source")}, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"report": report, "predictions": preds}, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
