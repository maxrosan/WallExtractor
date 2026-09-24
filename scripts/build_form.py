"""Build the review form (HTML artifact) for a set of floor plan PDFs.

Usage: python scripts/build_form.py out.html "Titulo 1=plan1.pdf" "Titulo 2=plan2.pdf" ...

Each plan is rendered with the extracted walls/openings drawn on it, and the
form (scripts/form_template.html) lets a reviewer confirm widths per frame
code, flag mispositioned openings and click on the drawing to mark what is
missing. Answers are stored in the artifact database (collection
"respostas", one document per plan id); scripts/respostas_do_form.py turns
them into answers files for `wallextractor.infer --answers`.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys

import pymupdf
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wallextractor.infer import extract_vector  # noqa: E402
from wallextractor.schedule import read_schedule  # noqa: E402
from wallextractor.vectorize import draw_plan  # noqa: E402


def plan_entry(title: str, path: str, max_side: int = 2000) -> dict:
    fid = hashlib.md5(os.path.basename(path).encode()).hexdigest()[:8]
    plan, rgb = extract_vector(path, max_side=max_side)
    img = Image.fromarray(draw_plan(rgb, plan))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82, optimize=True)
    W, H = img.size
    ppm = plan.scale.px_per_m or 1.0
    with pymupdf.open(path) as doc:
        sched = read_schedule(doc, exclude={0: tuple(plan.source.region_pt)})
    ops = [{"id": o.id, "code": o.code, "type": o.type, "width_m": round(o.width / ppm, 2), "source": o.width_source,
            "confidence": o.confidence, "x": round(100 * (o.start[0] + o.end[0]) / 2 / W, 2),
            "y": round(100 * (o.start[1] + o.end[1]) / 2 / H, 2), "wall": o.wall_id} for o in plan.openings]
    codes = {}
    for o in plan.openings:
        if o.code and o.code not in codes:
            row = sched.get(o.code) if sched.rows else None
            codes[o.code] = {"type": o.type, "geom": round(o.width / ppm, 2), "schedule": row.width_m if row else None}
    return {"id": fid, "title": title, "file": os.path.basename(path),
            "image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), "w": W, "h": H,
            "px_per_m": ppm, "walls": len(plan.walls), "openings": ops, "codes": dict(sorted(codes.items())),
            "questions": plan.questions, "schedule_read": bool(sched.rows)}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    out = sys.argv[1]
    data = []
    for spec in sys.argv[2:]:
        title, path = spec.split("=", 1) if "=" in spec else (os.path.basename(spec), spec)
        data.append(plan_entry(title, path))
        print(f"[form] {title}: {len(data[-1]['openings'])} openings, {len(data[-1]['questions'])} questions")
    with open(os.path.join(os.path.dirname(__file__), "form_template.html"), encoding="utf-8") as f:
        html = f.read().replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[form] wrote {out} ({os.path.getsize(out) // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
