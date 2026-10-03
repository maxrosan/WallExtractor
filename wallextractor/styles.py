"""Wall restyling: redraw the walls of a plan image in other drafting conventions.

CubiCasa5K draws walls as solid black blocks. Brazilian project drawings
(NBR 6492 practice) usually show walls as a double line with a 45-degree
hatch, a grey fill, or the outline only. Because the wall mask is pixel
aligned with the image, the wall pixels can be replaced by any of these
patterns while furniture, text, dimensions and symbols stay untouched.

Styles:
  solid    original pixels (no change)
  hatch45  white interior with diagonal hatch lines and a dark outline
  hatch135 same, other diagonal
  outline  white interior, dark outline only (double line)
  gray     grey fill with a dark outline
  cross    cross hatch (concrete / masonry variants)
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

import cv2
import numpy as np

STYLES: Sequence[str] = ("solid", "hatch45", "hatch135", "outline", "gray", "cross")


def restyle_walls(img: np.ndarray, mask: np.ndarray, style: str, wall_class: int = 1,
                  rng: Optional[random.Random] = None, spacing: Optional[int] = None,
                  line_gray: Optional[int] = None, outline_px: int = 1) -> np.ndarray:
    """Return a copy of ``img`` (uint8 HxWx3) with wall pixels redrawn in ``style``."""
    if style == "solid":
        return img
    rng = rng or random
    h, w = mask.shape[:2]
    wall = (mask == wall_class).astype(np.uint8)
    if wall.sum() == 0:
        return img
    out = img.copy()
    k = np.ones((2 * outline_px + 1, 2 * outline_px + 1), np.uint8)
    inner = cv2.erode(wall, k)
    outline = (wall == 1) & (inner == 0)
    interior = inner == 1
    dark = int(line_gray if line_gray is not None else rng.randint(0, 70))
    paper = int(rng.randint(235, 255))

    out[interior] = paper
    if style in ("hatch45", "hatch135", "cross"):
        sp = int(spacing or rng.randint(5, 10))
        yy, xx = np.mgrid[0:h, 0:w]
        if style == "hatch45":
            pat = ((xx + yy) % sp) == 0
        elif style == "hatch135":
            pat = ((xx - yy) % sp) == 0
        else:
            pat = (((xx + yy) % sp) == 0) | (((xx - yy) % sp) == 0)
        out[interior & pat] = dark
    elif style == "gray":
        out[interior] = int(rng.randint(140, 215))
    elif style == "outline":
        pass
    else:
        raise ValueError(f"unknown style {style}")
    out[outline] = dark
    return out


def random_style(rng: Optional[random.Random] = None, exclude_solid: bool = False) -> str:
    rng = rng or random
    choices = [s for s in STYLES if not (exclude_solid and s == "solid")]
    return rng.choice(choices)
