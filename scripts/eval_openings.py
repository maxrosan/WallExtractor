"""Score the vector extractor's doors and windows against corrected plans from the editor.

Input: a directory with <id>.pdf (GET /api/plans/{id}/pdf) and <id>.json (GET /api/plans/{id}),
as saved by --download. Only plans with a correction are scored.

A predicted opening matches a corrected one when the type is the same, the midpoints are within
--tol metres, the directions agree within 20 degrees and the widths within --tol metres (one to one,
greedy by distance). "rotated" counts predictions that sit on a corrected opening's spot but
cross it (the door drawn along its open leaf).

    python scripts/eval_openings.py --download data/gt --editor https://... --token ...
    python scripts/eval_openings.py data/gt [-v]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE_SIDE = 2000  # editor/app.py renders and stores coordinates at this size


def download(out: str, editor: str, token: str) -> None:
    os.makedirs(out, exist_ok=True)

    def get(path):
        return urllib.request.urlopen(urllib.request.Request(editor.rstrip("/") + path, headers={"X-Token": token})).read()

    for p in json.loads(get("/api/plans?status=corrected")):
        with open(os.path.join(out, p["id"] + ".json"), "wb") as f:
            f.write(get(f"/api/plans/{p['id']}"))
        with open(os.path.join(out, p["id"] + ".pdf"), "wb") as f:
            f.write(get(f"/api/plans/{p['id']}/pdf"))
        print("saved", p["id"], p["title"])


def _mid(o):
    return ((o["start"][0] + o["end"][0]) / 2, (o["start"][1] + o["end"][1]) / 2)


def _dir(o):
    return math.degrees(math.atan2(o["end"][1] - o["start"][1], o["end"][0] - o["start"][0])) % 180


def _len(o):
    return math.dist(o["start"], o["end"])


def score(pred, gold, ppm, tol_m):
    """Greedy one-to-one matching. Returns (matched pairs, unmatched pred, unmatched gold, rotated pairs)."""
    tol = tol_m * ppm
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if p["type"] != g["type"]:
                continue
            d = math.dist(_mid(p), _mid(g))
            da = abs(_dir(p) - _dir(g))
            da = min(da, 180 - da)
            if d <= tol and da <= 20 and abs(_len(p) - _len(g)) <= tol:
                pairs.append((d, i, j))
    used_p, used_g, match = set(), set(), []
    for d, i, j in sorted(pairs):
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            match.append((i, j))
    fp = [i for i in range(len(pred)) if i not in used_p]
    fn = [j for j in range(len(gold)) if j not in used_g]
    rotated = []
    for i in fp:
        p = pred[i]
        for j in fn:
            g = gold[j]
            da = abs(_dir(p) - _dir(g))
            da = min(da, 180 - da)
            # a leaf hinged at one end of the opening: its midpoint is ~half a width from the opening's
            if p["type"] == g["type"] and da > 60 and math.dist(_mid(p), _mid(g)) <= 0.9 * _len(g) + tol:
                rotated.append((i, j))
                break
    return match, fp, fn, rotated


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--download", metavar="DIR")
    ap.add_argument("--editor", default=os.environ.get("WE_EDITOR_URL", ""))
    ap.add_argument("--token", default=os.environ.get("EDITOR_TOKEN", ""))
    ap.add_argument("--tol", type=float, default=0.15, help="position and width tolerance in metres")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if a.download:
        download(a.download, a.editor, a.token)
        return

    from wallextractor.infer import extract_vector

    tot = {t: [0, 0, 0, 0] for t in ("door", "window")}  # tp, fp, fn, rotated
    for name in sorted(os.listdir(a.dir)):
        if not name.endswith(".json"):
            continue
        row = json.load(open(os.path.join(a.dir, name)))
        gold_plan = row.get("corrected")
        if not gold_plan:
            continue
        res = extract_vector(os.path.join(a.dir, name[:-5] + ".pdf"), page=row.get("page", 1), max_side=BASE_SIDE)
        if res is None:
            print(f"{row['title'][:50]}: extractor returned nothing")
            continue
        pred = res[0].to_dict()
        ppm = gold_plan["scale"]["px_per_m"]
        line = [f"{row['title'][:48]:48s}"]
        for t in ("door", "window"):
            P = [o for o in pred["openings"] if o["type"] == t]
            G = [o for o in gold_plan["openings"] if o["type"] == t]
            m, fp, fn, rot = score(P, G, ppm, a.tol)
            tot[t][0] += len(m)
            tot[t][1] += len(fp)
            tot[t][2] += len(fn)
            tot[t][3] += len(rot)
            line.append(f"{t}s {len(m)}/{len(G)} (+{len(fp)} wrong, {len(rot)} rotated)")
            if a.verbose:
                for i in fp:
                    o = P[i]
                    print(f"   FP {t} {o.get('code')} wall={o.get('wall_id')} mid=({_mid(o)[0]:.0f},{_mid(o)[1]:.0f}) "
                          f"w={_len(o) / ppm:.2f} m dir={_dir(o):.0f}")
                for j in fn:
                    o = G[j]
                    print(f"   FN {t} {o.get('code')} wall={o.get('wall_id')} mid=({_mid(o)[0]:.0f},{_mid(o)[1]:.0f}) "
                          f"w={_len(o) / ppm:.2f} m dir={_dir(o):.0f}")
        print(" | ".join(line))
    for t, (tp, fp, fn, rot) in tot.items():
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"{t:6s}: TP {tp:3d}  FP {fp:3d}  FN {fn:3d}  rotated {rot:3d}  P {p:.2f}  R {r:.2f}  F1 {f1:.2f}")


if __name__ == "__main__":
    main()
