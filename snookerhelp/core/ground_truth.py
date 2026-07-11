from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from snookerhelp.core.schema import (
    GroundTruthBall,
    GroundTruthEllipse,
    GroundTruthImage,
)


GROUND_TRUTH_SCHEMA = "snookerhelp.ground_truth.v1"


def empty_ground_truth(
    *,
    image_name: str,
    image_path: str | None = None,
) -> GroundTruthImage:
    return GroundTruthImage(
        schema_version=GROUND_TRUTH_SCHEMA,
        image_name=image_name,
        image_path=image_path,
        coordinate_system="source_px",
    )


def ground_truth_from_dict(
    payload: dict[str, Any],
    *,
    image_name: str,
    image_path: str | None = None,
) -> GroundTruthImage:
    schema = str(payload.get("schema_version") or GROUND_TRUTH_SCHEMA)
    if schema != GROUND_TRUTH_SCHEMA:
        raise ValueError(f"Unsupported ground-truth schema: {schema}")
    coordinate_system = str(payload.get("coordinate_system") or "source_px")
    if coordinate_system not in {"source_px", "warped_px", "table_mm"}:
        raise ValueError(f"Unsupported coordinate system: {coordinate_system}")

    balls = [
        _ball_from_dict(item, coordinate_system=coordinate_system)
        for item in payload.get("balls", [])
    ]
    return GroundTruthImage(
        schema_version=GROUND_TRUTH_SCHEMA,
        image_name=str(payload.get("image_name") or image_name),
        image_path=payload.get("image_path") or image_path,
        coordinate_system=coordinate_system,  # type: ignore[arg-type]
        balls=balls,
        reviewer=payload.get("reviewer"),
        notes=payload.get("notes"),
    )


def load_ground_truth(
    path: str | Path,
    *,
    image_name: str | None = None,
    image_path: str | None = None,
) -> GroundTruthImage:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return ground_truth_from_dict(
        payload,
        image_name=image_name or source.stem,
        image_path=image_path,
    )


def save_ground_truth(value: GroundTruthImage, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def upsert_ball_ground_truth(
    value: GroundTruthImage,
    annotation: GroundTruthBall,
) -> GroundTruthImage:
    balls = list(value.balls)
    if annotation.ball_id is None:
        balls.append(annotation)
    else:
        for index, current in enumerate(balls):
            if current.ball_id == annotation.ball_id:
                balls[index] = annotation
                break
        else:
            balls.append(annotation)
    balls.sort(key=lambda item: (item.ball_id is None, item.ball_id or 0, item.label))
    return GroundTruthImage(
        schema_version=value.schema_version,
        image_name=value.image_name,
        coordinate_system=value.coordinate_system,
        balls=balls,
        image_path=value.image_path,
        reviewer=value.reviewer,
        notes=value.notes,
    )


def _ball_from_dict(
    payload: dict[str, Any],
    *,
    coordinate_system: str,
) -> GroundTruthBall:
    ellipse_payload = payload.get("ellipse_px")
    ellipse = _ellipse_from_dict(ellipse_payload) if ellipse_payload else None
    point = payload.get("point") or (ellipse.center_px if ellipse else None)
    if point is None or len(point) < 2:
        raise ValueError("Ground-truth ball requires point or ellipse center")
    ball_coordinate_system = str(payload.get("coordinate_system") or coordinate_system)
    if ball_coordinate_system not in {"source_px", "warped_px", "table_mm"}:
        raise ValueError(f"Unsupported ball coordinate system: {ball_coordinate_system}")
    return GroundTruthBall(
        label=str(payload.get("label") or "unknown"),
        coordinate_system=ball_coordinate_system,  # type: ignore[arg-type]
        point=[float(point[0]), float(point[1])],
        image_name=payload.get("image_name"),
        ball_id=int(payload["ball_id"]) if payload.get("ball_id") is not None else None,
        ellipse_px=ellipse,
        uncertainty=dict(payload.get("uncertainty") or {}),
        notes=payload.get("notes"),
    )


def _ellipse_from_dict(payload: dict[str, Any]) -> GroundTruthEllipse:
    center = payload.get("center_px")
    if center is None or len(center) < 2:
        raise ValueError("Manual ellipse requires center_px")
    major = float(payload["major_axis_px"])
    minor = float(payload["minor_axis_px"])
    angle = float(payload.get("angle_deg", 0.0)) % 180.0
    if major <= 0.0 or minor <= 0.0:
        raise ValueError("Manual ellipse axes must be positive")
    if minor > major:
        major, minor = minor, major
        angle = (angle + 90.0) % 180.0
    return GroundTruthEllipse(
        center_px=[float(center[0]), float(center[1])],
        major_axis_px=major,
        minor_axis_px=minor,
        angle_deg=angle,
        visible_arcs_deg=_arc_ranges(payload.get("visible_arcs_deg")),
        occluded_arcs_deg=_arc_ranges(payload.get("occluded_arcs_deg")),
        source=str(payload.get("source") or "manual"),
        uncertainty=dict(payload.get("uncertainty") or {}),
    )


def _arc_ranges(value: Any) -> list[list[float]]:
    ranges: list[list[float]] = []
    for item in value or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        ranges.append([float(item[0]) % 360.0, float(item[1]) % 360.0])
    return ranges


__all__ = [
    "GROUND_TRUTH_SCHEMA",
    "empty_ground_truth",
    "ground_truth_from_dict",
    "load_ground_truth",
    "save_ground_truth",
    "upsert_ball_ground_truth",
]
