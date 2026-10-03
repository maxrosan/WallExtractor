"""Send PDFs to the editor's correction queue.

Usage:
  python scripts/enqueue.py --editor https://editor.exemplo.com --token XXX planta1.pdf planta2.pdf ... [--page 1]

Each PDF is uploaded, extracted on the server and queued as "pending".
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys
import urllib.request
import uuid


def _multipart(fields: dict, file_field: str, path: str) -> tuple[bytes, str]:
    boundary = "----we" + uuid.uuid4().hex
    body = []
    for k, v in fields.items():
        body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    ctype = mimetypes.guess_type(path)[0] or "application/pdf"
    with open(path, "rb") as f:
        data = f.read()
    body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{os.path.basename(path)}\"\r\n"
                f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n")
    body.append(f"--{boundary}--\r\n".encode())
    return b"".join(body), f"multipart/form-data; boundary={boundary}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--editor", required=True)
    ap.add_argument("--token", default=os.environ.get("EDITOR_TOKEN", ""))
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("pdfs", nargs="+")
    args = ap.parse_args()
    for path in args.pdfs:
        body, ctype = _multipart({"title": os.path.splitext(os.path.basename(path))[0], "page": str(args.page)}, "file", path)
        headers = {"Content-Type": ctype}
        if args.token:
            headers["X-Token"] = args.token
        req = urllib.request.Request(args.editor.rstrip("/") + "/api/plans", data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                print(f"[enqueue] {os.path.basename(path)}: {r.status} {r.read()[:120].decode(errors='replace')}")
        except Exception as exc:  # noqa: BLE001
            print(f"[enqueue] {os.path.basename(path)}: FAILED {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
