"""Run the extractor on PDFs and write DUVIDAS.md plus one answer template per file.

Usage: python scripts/duvidas.py out_dir file1.pdf [file2.pdf ...] [--answers-dir out_dir/respostas]

The answer template maps each frame code to the width the geometry found
(metres). Correct the numbers, then run:
  python -m wallextractor.infer planta.pdf --answers out_dir/respostas/<name>.json --out planta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wallextractor.infer import extract  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--answers-dir", default=None)
    args = ap.parse_args()
    ans_dir = args.answers_dir or os.path.join(args.out, "respostas")
    os.makedirs(ans_dir, exist_ok=True)
    md = ["# Dúvidas para confirmar\n\n",
          "Responda editando o JSON de respostas de cada planta (largura em metros por código) ",
          "e rode o infer com `--answers`.\n"]
    for path in args.pdfs:
        name = os.path.splitext(os.path.basename(path))[0]
        plan, _ = extract(path, None)
        plan.save(os.path.join(args.out, name[:8] + ".json"))
        ppm = plan.scale.px_per_m or 1.0
        md.append(f"\n## {name}\n\n")
        if not plan.questions:
            md.append("Sem dúvidas: todas as larguras vieram do quadro de esquadrias.\n")
        for q in plan.questions:
            md.append(f"- {q}\n")
        template = {}
        for o in plan.openings:
            if o.code and o.width_source != "schedule":
                template.setdefault(o.code, round(o.width / ppm, 2))
        if template:
            ans_path = os.path.join(ans_dir, name[:8] + ".json")
            if not os.path.isfile(ans_path):  # never overwrite answers already given
                with open(ans_path, "w", encoding="utf-8") as f:
                    json.dump(template, f, indent=1, ensure_ascii=False)
            md.append(f"\nModelo de respostas: `{os.path.relpath(ans_path, args.out)}` -> {json.dumps(template)}\n")
    with open(os.path.join(args.out, "DUVIDAS.md"), "w", encoding="utf-8") as f:
        f.write("".join(md))
    print("".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
