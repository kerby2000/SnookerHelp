from __future__ import annotations

import math
from typing import Sequence


def distance_2d(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def midpoint_2d(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [(float(a[0]) + float(b[0])) / 2.0, (float(a[1]) + float(b[1])) / 2.0]


__all__ = ["distance_2d", "midpoint_2d"]
