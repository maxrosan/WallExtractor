"""Image preprocessing shared by training and inference (no torch dependency)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

IGNORE_INDEX = 255
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_pad(img: Image.Image, mask: Optional[Image.Image], size: int) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """Resize so the long side is ``size`` and pad to a square (white image, ignore-index mask).

    Returns ``(img_uint8_HxWx3, mask_uint8_HxW_or_None, scale)``.
    """
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img_r = np.asarray(img.resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    canvas[:nh, :nw] = img_r
    mask_c = None
    if mask is not None:
        mask_r = np.asarray(mask.resize((nw, nh), Image.NEAREST), dtype=np.uint8)
        mask_c = np.full((size, size), IGNORE_INDEX, dtype=np.uint8)
        mask_c[:nh, :nw] = mask_r
    return canvas, mask_c, scale


def normalize(img: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float32 1x3xHxW normalized with ImageNet statistics."""
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))[None]
