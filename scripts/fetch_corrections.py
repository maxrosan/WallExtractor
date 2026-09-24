"""Pull corrected plans from the editor and prepare them as training data.

Usage:
  python scripts/fetch_corrections.py --editor https://editor.exemplo.com --token XXX --out data/corrections \
      [--prepared data/prepared_corr] [--status corrected]

Downloads /api/export (zip of <id>.png + <id>.json + manifest.json), unzips
into --out and, when --prepared is given, rasterizes masks there for
``train_seg --extra-train/--extra-val``.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--editor", required=True, help="base URL of the editor")
    ap.add_argument("--token", default=os.environ.get("EDITOR_TOKEN", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prepared", default=None)
    ap.add_argument("--status", default="corrected", choices=["corrected", "all", "pending", "skipped"])
    ap.add_argument("--val-fraction", type=float, default=0.2)
    args = ap.parse_args()
    url = args.editor.rstrip("/") + f"/api/export?status={args.status}"
    req = urllib.request.Request(url, headers={"X-Token": args.token} if args.token else {})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    os.makedirs(args.out, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(args.out)
        names = [n for n in z.namelist() if n.endswith(".json") and n != "manifest.json"]
    print(f"[fetch] {len(names)} plans -> {args.out}")
    if args.prepared:
        from wallextractor.annotations import prepare

        prepare(args.out, args.prepared, val_fraction=args.val_fraction)
    return 0


if __name__ == "__main__":
    sys.exit(main())
