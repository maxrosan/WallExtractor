"""Convert answers saved by the review form into the answers files the extractor consumes.

The form stores one document per plan (``respostas/<id>``) with:
  codes:    {"P1": {"width": 0.70, "type": "door"}, ...}   confirmed width per frame code
  openings: {"o3": {"status": "ok"|"pos_errada"|"nao_existe", "type": ..., "note": ...}, ...}
  marks:    [{"x": %, "y": %, "kind": "door"|"window"|"wall", "note": ...}]  things drawn on the plan
  notes:    free text

Usage: python scripts/respostas_do_form.py form_docs_dir out_dir
  form_docs_dir: folder with <id>.json documents (as exported from the artifact database)
  out_dir:       writes <id>.json answers ({"P1": 0.70, ...}) plus <id>.review.json with the rest
"""

from __future__ import annotations

import glob
import json
import os
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(src, "*.json"))):
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        fid = os.path.splitext(os.path.basename(path))[0]
        answers = {code: float(v["width"]) for code, v in (doc.get("codes") or {}).items()
                   if isinstance(v, dict) and v.get("width") is not None}
        with open(os.path.join(out, f"{fid}.json"), "w", encoding="utf-8") as f:
            json.dump(answers, f, indent=1, ensure_ascii=False)
        review = {
            "file": doc.get("file"),
            "updated_at": doc.get("updated_at"),
            "wrong_position": [k for k, v in (doc.get("openings") or {}).items() if v.get("status") == "pos_errada"],
            "not_existing": [k for k, v in (doc.get("openings") or {}).items() if v.get("status") == "nao_existe"],
            "type_changes": {k: v["type"] for k, v in (doc.get("openings") or {}).items() if v.get("type")},
            "opening_notes": {k: v["note"] for k, v in (doc.get("openings") or {}).items() if v.get("note")},
            "marks": doc.get("marks") or [],
            "notes": doc.get("notes") or "",
        }
        with open(os.path.join(out, f"{fid}.review.json"), "w", encoding="utf-8") as f:
            json.dump(review, f, indent=1, ensure_ascii=False)
        print(f"{fid}: {len(answers)} widths, {len(review['wrong_position'])} wrong positions, "
              f"{len(review['not_existing'])} not existing, {len(review['marks'])} marks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
