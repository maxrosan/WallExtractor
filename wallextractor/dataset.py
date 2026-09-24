"""Torch dataset over the folders written by ``wallextractor.cubicasa.prepare``."""

from __future__ import annotations

import glob
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .imageops import IGNORE_INDEX, MEAN, STD, normalize, resize_pad  # noqa: F401 (re-exported)


def list_samples(folder: str) -> List[Tuple[str, str]]:
    imgs = sorted(p for p in glob.glob(os.path.join(folder, "*.png")) if not p.endswith("_mask.png"))
    pairs = []
    for p in imgs:
        m = p[:-4] + "_mask.png"
        if os.path.isfile(m):
            pairs.append((p, m))
    return pairs


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(normalize(img)[0])


class PlanSegDataset(Dataset):
    """Image/mask pairs with optional geometric augmentation and wall restyling.

    ``restyle_prob`` redraws the walls in a random drafting style (see
    ``wallextractor.styles``) for that fraction of samples. ``fixed_style``
    applies one style to every sample deterministically (for evaluation).
    """

    def __init__(self, folder: str, size: int = 512, augment: bool = False, crop_prob: float = 0.5,
                 restyle_prob: float = 0.0, fixed_style: str | None = None):
        self.samples = list_samples(folder)
        if not self.samples:
            raise FileNotFoundError(f"no image/mask pairs in {folder}")
        self.size = size
        self.augment = augment
        self.crop_prob = crop_prob
        self.restyle_prob = restyle_prob
        self.fixed_style = fixed_style

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.fixed_style or (self.restyle_prob > 0 and random.random() < self.restyle_prob):
            from .styles import random_style, restyle_walls

            rng = random.Random(idx) if self.fixed_style else random
            style = self.fixed_style or random_style(rng, exclude_solid=True)
            arr = restyle_walls(np.asarray(img, dtype=np.uint8), np.asarray(mask, dtype=np.uint8), style, rng=rng)
            img = Image.fromarray(arr)
        if self.augment and random.random() < self.crop_prob:
            # zoomed-in crop: keeps walls thick enough to learn thickness at 512 px
            w, h = img.size
            cw, ch = int(w * random.uniform(0.5, 0.9)), int(h * random.uniform(0.5, 0.9))
            x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
            img = img.crop((x0, y0, x0 + cw, y0 + ch))
            mask = mask.crop((x0, y0, x0 + cw, y0 + ch))
        arr, m, _ = resize_pad(img, mask, self.size)
        if self.augment:
            if random.random() < 0.5:
                arr, m = arr[:, ::-1], m[:, ::-1]
            if random.random() < 0.5:
                arr, m = arr[::-1], m[::-1]
            k = random.randint(0, 3)
            if k:
                arr, m = np.rot90(arr, k), np.rot90(m, k)
        x = to_tensor(np.ascontiguousarray(arr))
        y = torch.from_numpy(np.ascontiguousarray(m).astype(np.int64))
        return x, y
