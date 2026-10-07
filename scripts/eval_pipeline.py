"""Score a non-VLM drafting pipeline with the VLM metric, on the same images and gold as ``wallextractor.eval_vlm``.

The raster branch (SegFormer ONNX + ``vectorize.mask_to_plan``, as the editor runs it) reads the 1024 px images of
a ``vlm_data`` folder and is compared with the same gold (the compact target decoded back) and the same matching,
so its numbers sit next to Qwen's: wall F1 by endpoints and by length, door and window F1, at 1.5% and 5% of the
long side.

  python scripts/eval_pipeline.py --data data/vlm_q3 --onnx results_e6/best.onnx --plans data/val_q3_g2.txt \
      --out results/eval_e6_g2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wallextractor.annotations import read_ids  # noqa: E402
from wallextractor.vlm_metrics import f1, length_counts, match_openings, match_walls  # noqa: E402
from wallextractor.vlm_data import decode_text  # noqa: E402


def raster(onnx_path: str, size: int):
    from wallextractor.infer import load_segmenter, segment_image
    from wallextractor.vectorize import mask_to_plan

    seg = load_segmenter(onnx_path)

    def run(img: Image.Image) -> dict:
        plan = mask_to_plan(segment_image(seg, np.asarray(img), size=size)).to_dict()
        return {"walls": [{"id": w["id"], "start": w["start"], "end": w["end"], "thickness": w["thickness"]}
                          for w in plan["walls"]],
                "openings": [{"type": o["type"], "start": o["start"], "end": o["end"]} for o in plan["openings"]]}
    return run


def cleanup(plan: dict) -> dict:
    """Generic geometry on a raster draft (no drawing-style rules): near-axis segments snapped to the axis, then
    the face detector's merging of collinear pieces, corner joining, cutting at openings and splitting at every
    junction, as the editor's convention asks (wallextractor.faces)."""
    import math
    import statistics

    from wallextractor import faces as F

    ws = [dict(w, start=list(w["start"]), end=list(w["end"])) for w in plan["walls"]]
    ops = [dict(o, start=list(o["start"]), end=list(o["end"])) for o in plan["openings"]]
    if not ws:
        return plan
    t = statistics.median(w["thickness"] for w in ws) or 8.0
    for x in ws + ops:  # snap: Hough segments come a degree or two off the axis
        a = math.degrees(math.atan2(x["end"][1] - x["start"][1], x["end"][0] - x["start"][0])) % 180
        if min(a, 180 - a) <= 6:
            m = (x["start"][1] + x["end"][1]) / 2
            x["start"][1] = x["end"][1] = m
        elif abs(a - 90) <= 6:
            m = (x["start"][0] + x["end"][0]) / 2
            x["start"][0] = x["end"][0] = m
    ws = F.mesclar_colineares(ws, t)
    ws, _ = F.remover_isoladas(ws, t)
    ws = F.ligar_cantos(ws, t)
    ws = F.cortar_nos_vaos(ws, ops, t)
    ws = F.remover_paredes_em_vaos(ws, ops, t)
    ws = F.ligar_cantos(ws, t)
    ws = F.dividir_nos_encontros(ws, t, aberturas=ops)
    ws = F.remover_degeneradas(ws, t)
    return {"walls": ws, "openings": ops}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder written by wallextractor.vlm_data")
    ap.add_argument("--split", default="val")
    ap.add_argument("--onnx", required=True, help="SegFormer: .onnx exported by train_seg or its checkpoint folder")
    ap.add_argument("--size", type=int, default=768, help="segmentation input side (the editor uses 768)")
    ap.add_argument("--plans", default=None, help="list file of plan ids (default: every editor plan of the split)")
    ap.add_argument("--tols", default="0.015,0.05")
    ap.add_argument("--cleanup", action="store_true", help="merge/join/split the raster walls (see cleanup)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    rows = [json.loads(ln) for ln in open(os.path.join(a.data, f"{a.split}.jsonl"), encoding="utf-8") if ln.strip()]
    rows = [r for r in rows if r["source"] == "editor"]
    if a.plans:
        keep = set(read_ids(a.plans))
        rows = [r for r in rows if r.get("plan") in keep]
    seen, uniq = set(), []
    for r in rows:
        if r["image"] not in seen:
            seen.add(r["image"])
            uniq.append(r)
    run = raster(a.onnx, a.size)
    tols = [float(t) for t in a.tols.split(",")]
    tot = {t: {k: [0, 0, 0] for k in ("wall", "door", "window")} for t in tols}
    lens = {t: [0, 0, 0, 0] for t in tols}
    per_plan = []
    t0 = time.time()
    for n, r in enumerate(uniq, 1):
        img = Image.open(os.path.join(a.data, r["image"])).convert("RGB")
        pred = run(img)
        if a.cleanup:
            pred = cleanup(pred)
        gold, _ = decode_text(r["target"], img.width, img.height)
        side = max(img.width, img.height)
        rec = {"plan": r.get("plan"), "walls": [len(pred["walls"]), len(gold["walls"])],
               "openings": [len(pred["openings"]), len(gold["openings"])]}
        for t in tols:
            tol = t * side
            tp = match_walls(pred["walls"], gold["walls"], tol)
            tot[t]["wall"] = [x + y for x, y in zip(tot[t]["wall"], (tp, len(pred["walls"]), len(gold["walls"])))]
            lens[t] = [x + y for x, y in zip(lens[t], length_counts(pred["walls"], gold["walls"], tol))]
            for typ in ("door", "window"):
                P = [o for o in pred["openings"] if o["type"] == typ]
                G = [o for o in gold["openings"] if o["type"] == typ]
                tot[t][typ] = [x + y for x, y in zip(tot[t][typ], (match_openings(P, G, tol), len(P), len(G)))]
        per_plan.append(rec)
        print(f"[pipe] {n}/{len(uniq)} {r.get('plan')} walls {rec['walls'][0]}/{rec['walls'][1]} "
              f"openings {rec['openings'][0]}/{rec['openings'][1]} {time.time() - t0:.0f}s", flush=True)

    table = {str(t): {k: dict(zip(("P", "R", "F1"), (round(v, 3) for v in f1(*tot[t][k])))) for k in tot[t]}
             for t in tols}
    wl = {}
    for t, (cp, n_p, cg, n_g) in lens.items():
        p, r = cp / max(1, n_p), cg / max(1, n_g)
        wl[str(t)] = {"P": round(p, 3), "R": round(r, 3), "F1": round(2 * p * r / (p + r), 3) if p + r else 0.0}
    report = {"plans": len(uniq), "all": table, "wall_len": wl, "onnx": a.onnx, "size": a.size, "cleanup": a.cleanup,
              "elapsed_s": round(time.time() - t0)}
    print(json.dumps({k: report[k] for k in ("plans", "all", "wall_len")}, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"report": report, "per_plan": per_plan}, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
