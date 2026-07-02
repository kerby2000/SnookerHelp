from __future__ import annotations


SNOOKER_BALL_DIAMETER_MM = 52.5
SNOOKER_BALL_RADIUS_MM = SNOOKER_BALL_DIAMETER_MM / 2.0


def px_to_mm(value_px: float, px_per_mm: float) -> float:
    return float(value_px) / float(px_per_mm)


def mm_to_px(value_mm: float, px_per_mm: float) -> float:
    return float(value_mm) * float(px_per_mm)


__all__ = [
    "SNOOKER_BALL_DIAMETER_MM",
    "SNOOKER_BALL_RADIUS_MM",
    "mm_to_px",
    "px_to_mm",
]
