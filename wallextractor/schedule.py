"""Read the door/window schedule ("QUADRO DE ESQUADRIAS") printed on the sheet.

Brazilian project sheets list every frame type with its code and sizes:

    NOME  LARG.  ALT.  [PEIT.]  TIPO       MATERIAL                 QUANT.
    P1    0.70   2.10           GIRO       MADEIRA NA COR NATURAL   02
    J1    0.40   0.40  1.70     BASCULANTE ESTRUTURA DE ALUMÍNIO    02

When the table is real text (not outlined glyphs) it is the most reliable
source of opening widths, so the geometry is overridden by it. Rows are
found as a code word ("P1", "J3") outside the drawing region followed, on
the same text line, by at least two sizes in the ``0.70`` format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_CODE_RE = re.compile(r"^([PJ])(\d{1,2})$", re.IGNORECASE)
_NUM_RE = re.compile(r"^\d{1,2}[.,]\d{2}$")
_QTY_RE = re.compile(r"^\d{1,3}$")
_KINDS = ("GIRO", "CORRER", "BASCULANTE", "PIVOTANTE", "MAXIM", "MAXIM-AR", "FIXA", "FIXO", "SANFONADA",
          "VENEZIANA", "GUILHOTINA", "ENROLAR", "CAMARÃO", "VAI-VEM", "VAI-E-VEM")


@dataclass
class ScheduleRow:
    code: str  # "P1"
    type: str  # "door" | "window"
    width_m: float
    height_m: Optional[float] = None
    sill_m: Optional[float] = None
    kind: Optional[str] = None  # GIRO, CORRER, BASCULANTE, ...
    material: Optional[str] = None
    quantity: Optional[int] = None
    page: int = 1
    raw: str = ""


@dataclass
class Schedule:
    rows: Dict[str, ScheduleRow] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def get(self, code: str) -> Optional[ScheduleRow]:
        return self.rows.get(code.upper())

    def __len__(self) -> int:
        return len(self.rows)


def _display_words(page):
    m = page.rotation_matrix if page.rotation else None
    out = []
    for w in page.get_text("words"):
        if m is None:
            out.append((w[0], w[1], w[2], w[3], w[4]))
        else:
            import pymupdf

            r = pymupdf.Rect(w[:4]) * m
            out.append((r.x0, r.y0, r.x1, r.y1, w[4]))
    return out


def parse_row(code: str, cells: Sequence[str], page: int = 1) -> Optional[ScheduleRow]:
    """Parse the words to the right of a code word. Numbers come first (LARG, ALT, [PEIT]), then text, QUANT last."""
    nums: List[float] = []
    i = 0
    while i < len(cells) and _NUM_RE.match(cells[i]):
        nums.append(float(cells[i].replace(",", ".")))
        i += 1
    if len(nums) < 2:
        return None
    rest = list(cells[i:])
    qty = None
    if rest and _QTY_RE.match(rest[-1]):
        qty = int(rest[-1])
        rest = rest[:-1]
    kind = next((c.upper() for c in rest if c.upper().rstrip(".") in _KINDS), None)
    material = " ".join(c for c in rest if c.upper().rstrip(".") not in _KINDS) or None
    m = _CODE_RE.match(code)
    otype = "door" if m and m.group(1).upper() == "P" else "window"
    width, height = nums[0], nums[1]
    sill = nums[2] if len(nums) >= 3 else None
    if not (0.2 <= width <= 6.0 and 0.2 <= height <= 4.0):
        return None
    return ScheduleRow(code=code.upper(), type=otype, width_m=width, height_m=height, sill_m=sill, kind=kind,
                       material=material, quantity=qty, page=page, raw=" ".join([code] + list(cells)))


def read_schedule(doc, exclude: Optional[Dict[int, Tuple[float, float, float, float]]] = None) -> Schedule:
    """Scan every page for schedule rows. ``exclude`` maps page index (0-based) to a drawing region to skip."""
    sched = Schedule()
    for pi, page in enumerate(doc):
        words = _display_words(page)
        region = (exclude or {}).get(pi)
        for w in words:
            if not _CODE_RE.match(w[4].strip()):
                continue
            if region and region[0] - 20 <= w[0] <= region[2] + 20 and region[1] - 20 <= w[1] <= region[3] + 20:
                continue  # a frame tag next to a door in the drawing, not a table row
            cy, h = (w[1] + w[3]) / 2, max(w[3] - w[1], 1.0)
            line = sorted((v for v in words if v is not w and abs((v[1] + v[3]) / 2 - cy) <= 0.6 * h
                           and v[0] >= w[2] - 1 and v[0] - w[2] <= 60 * h), key=lambda v: v[0])
            row = parse_row(w[4].strip(), [v[4] for v in line], page=pi + 1)
            if row is None:
                continue
            if row.code in sched.rows:
                continue  # first occurrence wins (a sheet may repeat the table)
            sched.rows[row.code] = row
    if sched.rows:
        sched.notes.append(f"schedule: {len(sched.rows)} rows " + ", ".join(
            f"{r.code}={r.width_m:.2f}" for r in sorted(sched.rows.values(), key=lambda r: r.code)))
    else:
        sched.notes.append("schedule: no text rows found (table absent or drawn as outlines)")
    return sched
