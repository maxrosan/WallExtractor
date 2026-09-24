"""Fetch a subset of CubiCasa5K straight from the Zenodo zip with HTTP range requests.

The full archive is 5.5 GB; a 400-plan subset is ~0.5 GB and takes a few
minutes. Only ``F1_scaled.png`` and ``model.svg`` are fetched per plan.

Usage:
  python scripts/fetch_cubicasa_subset.py --out data/cubicasa5k --train 400 --val 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time

URL = "https://zenodo.org/api/records/2613548/files/cubicasa5k.zip/content"
NEEDED = ("F1_scaled.png", "model.svg")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--delay", type=float, default=0.7, help="seconds between range requests (rate limit)")
    args = ap.parse_args()

    import requests
    from remotezip import RemoteZip
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    # Zenodo rate-limits range requests (HTTP 429); back off and respect Retry-After.
    session = requests.Session()
    retry = Retry(total=10, backoff_factor=3, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET", "HEAD"], respect_retry_after_header=True)
    session.mount("https://", HTTPAdapter(max_retries=retry))

    t0 = time.time()
    z = RemoteZip(args.url, session=session)
    names = set(z.namelist())
    root = "cubicasa5k"
    os.makedirs(args.out, exist_ok=True)
    for split in ("train", "val", "test"):
        member = f"{root}/{split}.txt"
        with z.open(member) as f, open(os.path.join(args.out, f"{split}.txt"), "wb") as g:
            g.write(f.read())
    print(f"[fetch] index read in {time.time() - t0:.1f}s", flush=True)

    total = 0
    for split, n in (("train", args.train), ("val", args.val), ("test", args.test)):
        if n <= 0:
            continue
        with open(os.path.join(args.out, f"{split}.txt"), encoding="utf-8") as f:
            folders = [line.strip().strip("/") for line in f if line.strip()][:n]
        for i, rel in enumerate(folders, 1):
            dest = os.path.join(args.out, rel)
            if all(os.path.isfile(os.path.join(dest, fn)) for fn in NEEDED):
                continue
            os.makedirs(dest, exist_ok=True)
            for fn in NEEDED:
                member = f"{root}/{rel}/{fn}"
                if member not in names:
                    continue
                for attempt in range(6):
                    try:
                        with z.open(member) as src, open(os.path.join(dest, fn), "wb") as dst:
                            dst.write(src.read())
                        break
                    except Exception as exc:  # noqa: BLE001 - RemoteIOError wraps the HTTP error
                        wait = 20 * (attempt + 1)
                        print(f"[fetch] {member}: {exc}; retry in {wait}s", flush=True)
                        time.sleep(wait)
                else:
                    raise SystemExit(f"[fetch] giving up on {member}")
                time.sleep(args.delay)
            total += 1
            if i % 25 == 0:
                print(f"[fetch] {split} {i}/{len(folders)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[fetch] done: {total} plans in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
