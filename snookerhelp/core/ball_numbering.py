from __future__ import annotations

from typing import Any


CANONICAL_BALL_NUMBERING_SCHEME = "snookerhelp.v1.color_slots_red_table_order"

FIXED_COLOR_IDS: dict[str, int] = {
    "white": 1,
    "yellow": 2,
    "green": 3,
    "brown": 4,
    "blue": 5,
    "pink": 6,
    "black": 7,
}

RED_FIRST_ID = 8
RED_COUNT = 15


def canonical_ball_id_map(balls: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Return deterministic v1 ball-numbering metadata keyed by raw detector id.

    The detector's numeric ids are processing-order ids. They are useful for
    low-level debugging, but they are not stable enough for human review across
    images. The v1 review layer therefore uses a canonical display scheme:

    - fixed colored balls: white=1, yellow=2, green=3, brown=4, blue=5,
      pink=6, black=7;
    - red balls: 8-22, sorted by table position.

    Red balls are physically interchangeable in snooker, so "red #8" means the
    first red in deterministic table order, not a tracked physical identity.
    """

    entries = [_entry(ball) for ball in balls if _raw_id(ball) is not None]
    by_raw: dict[int, dict[str, Any]] = {}
    used_raw: set[int] = set()
    used_canonical: set[int] = set()

    for label, canonical_id in FIXED_COLOR_IDS.items():
        candidates = [
            entry
            for entry in entries
            if entry["label"] == label and entry["raw_id"] not in used_raw
        ]
        if not candidates:
            continue
        chosen = min(candidates, key=_entry_sort_key)
        _assign(
            by_raw,
            chosen,
            canonical_id=canonical_id,
            slot=label,
            status="canonical",
        )
        used_raw.add(chosen["raw_id"])
        used_canonical.add(canonical_id)

    red_candidates = sorted(
        [
            entry
            for entry in entries
            if entry["label"] == "red" and entry["raw_id"] not in used_raw
        ],
        key=_entry_sort_key,
    )
    for red_index, entry in enumerate(red_candidates, start=1):
        if red_index <= RED_COUNT:
            canonical_id = RED_FIRST_ID + red_index - 1
            slot = f"red_{red_index:02d}"
            status = "canonical"
        else:
            canonical_id = _next_overflow_id(used_canonical)
            slot = f"red_overflow_{red_index:02d}"
            status = "overflow_extra_red"
        _assign(
            by_raw,
            entry,
            canonical_id=canonical_id,
            slot=slot,
            status=status,
        )
        used_raw.add(entry["raw_id"])
        used_canonical.add(canonical_id)

    for entry in sorted(
        [entry for entry in entries if entry["raw_id"] not in used_raw],
        key=lambda item: (item["label"], _entry_sort_key(item)),
    ):
        canonical_id = _next_overflow_id(used_canonical)
        _assign(
            by_raw,
            entry,
            canonical_id=canonical_id,
            slot=f"{entry['label']}_overflow",
            status="overflow_duplicate_or_unknown",
        )
        used_raw.add(entry["raw_id"])
        used_canonical.add(canonical_id)

    return by_raw


def _assign(
    mapping: dict[int, dict[str, Any]],
    entry: dict[str, Any],
    *,
    canonical_id: int,
    slot: str,
    status: str,
) -> None:
    mapping[entry["raw_id"]] = {
        "scheme": CANONICAL_BALL_NUMBERING_SCHEME,
        "raw_detector_id": entry["raw_id"],
        "canonical_ball_id": int(canonical_id),
        "label": entry["label"],
        "slot": slot,
        "status": status,
        "sort_xy": [round(float(entry["xy"][0]), 4), round(float(entry["xy"][1]), 4)],
        "note": (
            "Canonical v1 review numbering. Detector raw id is preserved for "
            "debug traceability; red ids are deterministic table-order ids."
        ),
    }


def _entry(ball: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_id": int(_raw_id(ball)),
        "label": _label(ball),
        "xy": _sort_xy(ball),
    }


def _raw_id(ball: dict[str, Any]) -> int | None:
    try:
        return int(ball.get("id", ball.get("ball_id")))
    except (TypeError, ValueError):
        return None


def _label(ball: dict[str, Any]) -> str:
    return str(
        ball.get("color_label")
        or ball.get("class")
        or ball.get("label")
        or "unknown"
    ).strip().lower()


def _sort_xy(ball: dict[str, Any]) -> tuple[float, float]:
    for value in (
        ball.get("source_refined_table_xy_mm"),
        ball.get("table_xy_mm"),
        _xy_by_z(ball),
        ball.get("source_refined_center_px"),
        ball.get("source_rough_center_px"),
        ball.get("warped_center_px"),
    ):
        xy = _point(value)
        if xy is not None:
            return xy
    raw_id = _raw_id(ball) or 0
    return (float(raw_id), float(raw_id))


def _xy_by_z(ball: dict[str, Any]) -> Any:
    by_z = ball.get("source_refined_table_xy_by_z_mm") or {}
    if not isinstance(by_z, dict):
        return None
    preferred = by_z.get("z_26_25")
    if isinstance(preferred, dict):
        return preferred.get("xy_mm")
    for value in by_z.values():
        if isinstance(value, dict) and value.get("xy_mm") is not None:
            return value.get("xy_mm")
    return None


def _point(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError, IndexError):
        return None


def _entry_sort_key(entry: dict[str, Any]) -> tuple[float, float, int]:
    x, y = entry["xy"]
    return (float(y), float(x), int(entry["raw_id"]))


def _next_overflow_id(used: set[int]) -> int:
    candidate = 23
    while candidate in used:
        candidate += 1
    return candidate


__all__ = [
    "CANONICAL_BALL_NUMBERING_SCHEME",
    "FIXED_COLOR_IDS",
    "RED_COUNT",
    "RED_FIRST_ID",
    "canonical_ball_id_map",
]
