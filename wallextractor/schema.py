"""Output schema of WallExtractor.

Every coordinate is in pixels of the working image (``image.width`` x
``image.height``), origin at the top-left corner, y pointing down. When a
scale is known, ``scale.px_per_m`` converts pixels to metres.

A wall is a straight segment (``start`` -> ``end``) with a ``thickness``. The
source polygon is kept in ``polygon`` when it is known (ground truth or vector
PDF), so nothing is lost by the segment simplification.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

SCHEMA_VERSION = "0.1"

Point = Tuple[float, float]


@dataclass
class Wall:
    id: str
    start: Point
    end: Point
    thickness: float
    kind: Optional[str] = None  # "external" | "internal" | None
    polygon: Optional[List[Point]] = None

    @property
    def length(self) -> float:
        return ((self.end[0] - self.start[0]) ** 2 + (self.end[1] - self.start[1]) ** 2) ** 0.5


@dataclass
class Opening:
    id: str
    type: str  # "door" | "window"
    start: Point
    end: Point
    width: float
    wall_id: Optional[str] = None
    polygon: Optional[List[Point]] = None


@dataclass
class Source:
    file: str
    page: int = 1
    kind: str = "raster"  # "vector" | "raster"


@dataclass
class ImageInfo:
    width: int
    height: int
    dpi: Optional[float] = None


@dataclass
class Scale:
    px_per_m: Optional[float] = None
    method: Optional[str] = None  # "dimension_text" | "manual" | "dataset" | None


@dataclass
class WallPlan:
    source: Source
    image: ImageInfo
    walls: List[Wall] = field(default_factory=list)
    openings: List[Opening] = field(default_factory=list)
    scale: Scale = field(default_factory=Scale)
    version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        d = asdict(self)
        # asdict turns tuples into lists already; round floats for compact files
        return _round_floats(d)

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "WallPlan":
        return cls(
            source=Source(**d["source"]),
            image=ImageInfo(**d["image"]),
            walls=[Wall(**w) for w in d.get("walls", [])],
            openings=[Opening(**o) for o in d.get("openings", [])],
            scale=Scale(**d.get("scale", {})),
            version=d.get("version", SCHEMA_VERSION),
        )

    @classmethod
    def load(cls, path: str) -> "WallPlan":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=1))


def _round_floats(obj, ndigits: int = 2):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def validate(d: dict) -> List[str]:
    """Return a list of problems; empty means the document is valid."""
    problems: List[str] = []
    for key in ("source", "image", "walls", "openings"):
        if key not in d:
            problems.append(f"missing key: {key}")
    if problems:
        return problems
    w, h = d["image"].get("width"), d["image"].get("height")
    if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
        problems.append("image.width/height must be positive integers")
    ids = set()
    for i, wall in enumerate(d["walls"]):
        for key in ("id", "start", "end", "thickness"):
            if key not in wall:
                problems.append(f"walls[{i}] missing {key}")
        if wall.get("id") in ids:
            problems.append(f"duplicate wall id {wall.get('id')}")
        ids.add(wall.get("id"))
        if wall.get("thickness", 1) <= 0:
            problems.append(f"walls[{i}] thickness must be > 0")
    for i, op in enumerate(d["openings"]):
        if op.get("type") not in ("door", "window"):
            problems.append(f"openings[{i}] type must be door or window")
        if op.get("wall_id") is not None and op["wall_id"] not in ids:
            problems.append(f"openings[{i}] references unknown wall {op['wall_id']}")
    return problems
