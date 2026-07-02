from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

from snookerhelp.recognition.color import BallColorClassifier
from snookerhelp.calibration.camera import parse_z_center_method, z_center_method
from snookerhelp.core.config import PROJECT_ROOT, resolve_path


REGION_NAMES = (
    "center",
    "left_edge",
    "right_edge",
    "top_edge",
    "bottom_edge",
    "pockets/corners",
)


def load_json(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml_or_json(path: str | Path) -> Any:
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        if resolved.suffix.lower() == ".json":
            return json.load(handle)
        return yaml.safe_load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def table_dimensions_from_state(state: dict[str, Any]) -> tuple[float, float]:
    table = state.get("table", {})
    return float(table["length_mm"]), float(table["width_mm"])


def table_px_per_mm_from_state(state: dict[str, Any]) -> float:
    return float(state.get("table", {}).get("warp_px_per_mm", 1.0))


def table_margin_mm_from_state(state: dict[str, Any]) -> float:
    return float(state.get("table", {}).get("processing_margin_mm", 0.0))


def default_region_margin_mm(ball_diameter_mm: float = 52.5) -> float:
    return 2.0 * ball_diameter_mm


def classify_table_region(
    x_mm: float,
    y_mm: float,
    table_length_mm: float,
    table_width_mm: float,
    edge_margin_mm: float,
) -> str:
    near_left = x_mm <= edge_margin_mm
    near_right = x_mm >= table_length_mm - edge_margin_mm
    near_bottom = y_mm <= edge_margin_mm
    near_top = y_mm >= table_width_mm - edge_margin_mm

    if (near_left or near_right) and (near_bottom or near_top):
        return "pockets/corners"
    if near_left:
        return "left_edge"
    if near_right:
        return "right_edge"
    if near_top:
        return "top_edge"
    if near_bottom:
        return "bottom_edge"
    return "center"


def add_region_to_row(
    row: dict[str, Any],
    x_mm: float,
    y_mm: float,
    table_length_mm: float,
    table_width_mm: float,
    edge_margin_mm: float,
) -> dict[str, Any]:
    row["region"] = classify_table_region(
        x_mm=x_mm,
        y_mm=y_mm,
        table_length_mm=table_length_mm,
        table_width_mm=table_width_mm,
        edge_margin_mm=edge_margin_mm,
    )
    return row


def summarize_values(values: Iterable[float | int | None]) -> dict[str, Any]:
    finite = np.array(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ],
        dtype=float,
    )
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "p95": float(np.percentile(finite, 95)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def summarize_by_region(
    rows: Iterable[dict[str, Any]],
    value_key: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = {region: [] for region in REGION_NAMES}
    for row in rows:
        region = str(row.get("region", "center"))
        if region not in grouped:
            grouped[region] = []
        value = row.get(value_key)
        if value is not None:
            grouped[region].append(float(value))
    return {
        region: summarize_values(values)
        for region, values in grouped.items()
    }


def state_display_name(state: dict[str, Any], state_path: str | Path | None = None) -> str:
    source_image = state.get("source_image")
    if source_image:
        return Path(str(source_image)).stem
    if state_path is not None:
        return resolve_path(state_path).stem
    return "state"


def source_image_path_from_state(state: dict[str, Any]) -> Path | None:
    source_image = state.get("source_image")
    if not source_image:
        return None
    return resolve_path(str(source_image))


def state_matches_selector(
    state: dict[str, Any],
    state_path: str | Path,
    selector: str | None,
) -> bool:
    if not selector:
        return True

    selector_path = Path(str(selector))
    wanted = {str(selector), selector_path.name, selector_path.stem}
    state_resolved = resolve_path(state_path)
    candidates = {
        str(state_resolved),
        state_resolved.name,
        state_resolved.stem,
    }
    source_image = state.get("source_image")
    if source_image:
        source_path = Path(str(source_image))
        candidates.update(
            {
                str(source_image),
                source_path.name,
                source_path.stem,
            }
        )
    return bool(wanted & candidates)


def ball_points_from_state(
    state: dict[str, Any],
    center_mode: str = "warped",
) -> list[dict[str, Any]]:
    requested_z = parse_z_center_method(center_mode)
    if center_mode not in {"warped", "source_refined"} and requested_z is None:
        raise ValueError(
            "center_mode must be 'warped', 'source_refined', or 'source_z_*'"
        )
    points: list[dict[str, Any]] = []
    for index, ball in enumerate(state.get("balls", []), start=1):
        center = ball.get("warped_center_px", ball.get("refined_center_px"))
        if center is None:
            center = ball.get("debug", {}).get("warped_center_px")
        if center is None:
            raise ValueError(f"Detector ball {index} has no warped center")
        table_xy = ball.get("table_xy_mm", [ball["x_mm"], ball["y_mm"]])
        radius_px = ball.get("radius_px")
        radius_mm = ball.get("radius_mm")
        source_refinement_success = bool(
            ball.get("source_refinement_success", False)
        )
        if center_mode == "source_refined":
            source_table_xy = ball.get("source_refined_table_xy_mm")
            if source_table_xy is not None:
                table_xy = source_table_xy
                source_warped = ball.get("source_refined_warped_center_px")
                if source_warped is not None:
                    center = source_warped
                else:
                    center = table_mm_to_warped_px_from_state(
                        state,
                        float(source_table_xy[0]),
                        float(source_table_xy[1]),
                    )
                radius_px = ball.get("source_radius_px", radius_px)
        elif requested_z is not None:
            projection = _projection_for_center_mode(ball, center_mode)
            if projection is not None:
                table_xy = projection["xy_mm"]
                center = table_mm_to_warped_px_from_state(
                    state,
                    float(table_xy[0]),
                    float(table_xy[1]),
                )
                radius_px = ball.get("source_radius_px", radius_px)
        points.append(
            {
                "id": int(ball.get("id", index)),
                "label": str(ball.get("color_label", ball.get("class", "unknown"))),
                "x_px": float(center[0]),
                "y_px": float(center[1]),
                "x_mm": float(table_xy[0]),
                "y_mm": float(table_xy[1]),
                "radius_px": (
                    float(radius_px)
                    if radius_px is not None
                    else None
                ),
                "radius_mm": (
                    float(radius_mm)
                    if radius_mm is not None
                    else None
                ),
                "center_mode": center_mode,
                "z_mm": requested_z,
                "source_refinement_success": source_refinement_success,
                "source_center_px": ball.get("source_refined_center_px"),
                "warped_center_px": ball.get("warped_center_px"),
                "confidence": float(
                    ball.get("detection_confidence", ball.get("confidence", 0.0))
                ),
            }
        )
    return points


def available_source_z_center_modes(state: dict[str, Any]) -> list[str]:
    z_values = state.get("camera_model", {}).get("projection_z_planes_mm")
    if z_values:
        return [z_center_method(float(value)) for value in z_values]
    for ball in state.get("balls", []):
        projections = ball.get("source_refined_table_xy_by_z_mm")
        if projections:
            values = [
                float(projection["z_mm"])
                for projection in projections.values()
            ]
            return [z_center_method(value) for value in sorted(values)]
    return []


def _projection_for_center_mode(
    ball: dict[str, Any],
    center_mode: str,
) -> dict[str, Any] | None:
    projections = ball.get("source_refined_table_xy_by_z_mm") or {}
    key = center_mode[len("source_") :] if center_mode.startswith("source_") else center_mode
    if key in projections:
        return projections[key]
    requested_z = parse_z_center_method(center_mode)
    if requested_z is None:
        return None
    best_projection = None
    best_delta = float("inf")
    for projection in projections.values():
        delta = abs(float(projection["z_mm"]) - requested_z)
        if delta < best_delta:
            best_delta = delta
            best_projection = projection
    return best_projection if best_delta < 1e-6 else None


def ball_by_id(
    state: dict[str, Any],
    ball_id: int,
    center_mode: str = "warped",
) -> dict[str, Any]:
    for point in ball_points_from_state(state, center_mode=center_mode):
        if int(point["id"]) == int(ball_id):
            return point
    raise KeyError(f"No detected ball with id {ball_id}")


def distance_mm(a: dict[str, Any], b: dict[str, Any]) -> float:
    return float(np.hypot(float(a["x_mm"]) - float(b["x_mm"]), float(a["y_mm"]) - float(b["y_mm"])))


def max_pairwise_range_mm(samples: list[dict[str, Any]]) -> float:
    if len(samples) < 2:
        return 0.0
    maximum = 0.0
    for index, a in enumerate(samples[:-1]):
        for b in samples[index + 1 :]:
            maximum = max(maximum, distance_mm(a, b))
    return float(maximum)


def table_mm_to_warped_px_from_state(
    state: dict[str, Any],
    x_mm: float,
    y_mm: float,
) -> tuple[float, float]:
    table = state.get("table", {})
    width_mm = float(table["width_mm"])
    px_per_mm = float(table.get("warp_px_per_mm", 1.0))
    margin_px = float(table.get("processing_margin_mm", 0.0)) * px_per_mm
    x_px = x_mm * px_per_mm + margin_px
    origin = str(table.get("coordinate_origin", "bottom_left"))
    if origin.startswith("bottom_left"):
        y_px = (width_mm - y_mm) * px_per_mm + margin_px
    else:
        y_px = y_mm * px_per_mm + margin_px
    return x_px, y_px


def warped_overlay_base(
    state: dict[str, Any],
    config_path: str | Path = "configs/sony_dev.yaml",
) -> np.ndarray:
    source_path = source_image_path_from_state(state)
    if source_path is not None and source_path.exists():
        try:
            from .state_estimator import StateEstimator

            estimator = StateEstimator.from_config(config_path)
            source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if source is not None:
                return estimator.table_warp.warp_image(source)
        except Exception:
            pass

    length_mm, width_mm = table_dimensions_from_state(state)
    px_per_mm = table_px_per_mm_from_state(state)
    margin_mm = table_margin_mm_from_state(state)
    width_px = int(round((length_mm + 2.0 * margin_mm) * px_per_mm))
    height_px = int(round((width_mm + 2.0 * margin_mm) * px_per_mm))
    image = np.full((height_px, width_px, 3), (50, 110, 55), dtype=np.uint8)
    margin_px = int(round(margin_mm * px_per_mm))
    cv2.rectangle(
        image,
        (margin_px, margin_px),
        (width_px - margin_px - 1, height_px - margin_px - 1),
        (30, 80, 35),
        2,
        cv2.LINE_AA,
    )
    return image


def draw_text_with_outline(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.5,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_ball_marker(
    image: np.ndarray,
    point: dict[str, Any],
    radius: int = 7,
    selected: bool = False,
) -> None:
    center = (int(round(point["x_px"])), int(round(point["y_px"])))
    color = BallColorClassifier.display_bgr(str(point.get("label", "unknown")))
    cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
    cv2.circle(
        image,
        center,
        radius + 3 if selected else radius + 1,
        (255, 255, 255) if selected else (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def format_float(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"

