from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def classify_boundary_points(
    points_px: list[Any] | tuple[Any, ...] | np.ndarray,
    *,
    target_ellipse: dict[str, Any] | None,
    neighbor_ellipses: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    target_residual_px: float = 4.5,
    neighbor_margin_px: float = 1.5,
) -> dict[str, Any]:
    """Classify boundary samples by likely ownership.

    The important distinction for dense clusters is not just accepted/rejected.
    A point can be a good image edge and still belong to a neighbouring ball or
    to a contact seam. This helper keeps the visible UI simple, but records the
    ownership evidence needed by the optimizer.
    """

    points = _points_array(points_px)
    if len(points) == 0:
        return {
            "status": "no_points",
            "point_count": 0,
            "points": [],
            "summary": {},
        }
    target = _ellipse_model(target_ellipse)
    neighbors = [_ellipse_model(ellipse) for ellipse in (neighbor_ellipses or [])]
    neighbors = [ellipse for ellipse in neighbors if ellipse is not None]
    if target is None:
        return {
            "status": "no_target_ellipse",
            "point_count": int(len(points)),
            "points": [
                {
                    "point_px": _round_point(point),
                    "category": "unknown",
                    "target_residual_px": None,
                    "nearest_neighbor_id": None,
                    "nearest_neighbor_residual_px": None,
                }
                for point in points
            ],
            "summary": {"unknown": int(len(points))},
        }

    payload: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for point in points:
        target_residual = _ellipse_residual_px(point, target)
        nearest_neighbor: dict[str, Any] | None = None
        nearest_residual = float("inf")
        inside_neighbor = False
        for neighbor in neighbors:
            residual = _ellipse_residual_px(point, neighbor)
            if residual < nearest_residual:
                nearest_residual = residual
                nearest_neighbor = neighbor
            inside_neighbor = inside_neighbor or _ellipse_normalized_radius(point, neighbor) < 0.92

        category = _ownership_category(
            target_residual_px_value=target_residual,
            nearest_neighbor_residual_px=nearest_residual,
            inside_neighbor=inside_neighbor,
            target_residual_px=target_residual_px,
            neighbor_margin_px=neighbor_margin_px,
        )
        counts[category] += 1
        payload.append(
            {
                "point_px": _round_point(point),
                "category": category,
                "target_residual_px": round(float(target_residual), 4),
                "nearest_neighbor_id": (
                    None if nearest_neighbor is None else nearest_neighbor.get("id")
                ),
                "nearest_neighbor_residual_px": (
                    None
                    if not np.isfinite(nearest_residual)
                    else round(float(nearest_residual), 4)
                ),
                "inside_neighbor_ellipse": bool(inside_neighbor),
            }
        )

    summary = {key: int(value) for key, value in sorted(counts.items())}
    target_owned = summary.get("target_boundary", 0) + summary.get("contact_seam", 0)
    neighbor_owned = summary.get("neighbor_owned", 0)
    return {
        "status": "computed",
        "point_count": int(len(points)),
        "summary": summary,
        "target_owned_fraction": round(float(target_owned) / float(len(points)), 4),
        "neighbor_owned_fraction": round(float(neighbor_owned) / float(len(points)), 4),
        "target_residual_threshold_px": round(float(target_residual_px), 4),
        "neighbor_margin_px": round(float(neighbor_margin_px), 4),
        "neighbor_count": len(neighbors),
        "points": payload,
    }


def ownership_summary_for_points(
    points_px: list[Any] | tuple[Any, ...] | np.ndarray,
    *,
    target_ellipse: dict[str, Any] | None,
    neighbor_ellipses: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    """Compact score payload used during candidate ranking."""

    ownership = classify_boundary_points(
        points_px,
        target_ellipse=target_ellipse,
        neighbor_ellipses=neighbor_ellipses,
    )
    summary = ownership.get("summary") or {}
    count = int(ownership.get("point_count") or 0)
    if count <= 0:
        return {
            "status": ownership.get("status", "no_points"),
            "point_count": 0,
            "neighbor_owned_fraction": 0.0,
            "target_owned_fraction": 0.0,
            "summary": summary,
        }
    return {
        "status": ownership.get("status"),
        "point_count": count,
        "neighbor_owned_fraction": ownership.get("neighbor_owned_fraction", 0.0),
        "target_owned_fraction": ownership.get("target_owned_fraction", 0.0),
        "summary": summary,
    }


def analyze_ball_boundary_ownership(
    ball: dict[str, Any],
    *,
    neighbor_ellipses: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    """Return per-ball ownership diagnostics for the current final boundary."""

    policy = ball.get("source_final_center_policy") or {}
    target_ellipse = (
        policy.get("ellipse_fit")
        or ball.get("source_ellipse_fit")
        or _fallback_circle_ellipse(ball)
    )
    accepted_points = (
        policy.get("boundary_points_px")
        or ball.get("source_boundary_points_px")
        or []
    )
    rejected_points = (
        policy.get("boundary_rejected_points_px")
        or ball.get("source_boundary_rejected_points_px")
        or []
    )
    accepted = classify_boundary_points(
        accepted_points,
        target_ellipse=target_ellipse,
        neighbor_ellipses=neighbor_ellipses,
    )
    rejected = classify_boundary_points(
        rejected_points,
        target_ellipse=target_ellipse,
        neighbor_ellipses=neighbor_ellipses,
    )
    return {
        "status": "computed",
        "model": "ellipse_residual_neighbor_ownership",
        "target_ellipse": _review_ellipse(target_ellipse),
        "neighbor_ellipses_px": [_review_ellipse(ellipse) for ellipse in (neighbor_ellipses or [])],
        "accepted_points": _compact_ownership(accepted),
        "rejected_points": _compact_ownership(rejected),
        "note": (
            "Ownership diagnostics are not separate UI colors. They explain "
            "which red/white samples may belong to this ball, a contact seam, "
            "or a neighboring ball."
        ),
    }


def _ownership_category(
    *,
    target_residual_px_value: float,
    nearest_neighbor_residual_px: float,
    inside_neighbor: bool,
    target_residual_px: float,
    neighbor_margin_px: float,
) -> str:
    close_to_target = target_residual_px_value <= target_residual_px
    close_to_neighbor = (
        np.isfinite(nearest_neighbor_residual_px)
        and nearest_neighbor_residual_px + neighbor_margin_px < target_residual_px_value
    )
    if close_to_target and not (inside_neighbor and close_to_neighbor):
        return "target_boundary"
    if inside_neighbor and close_to_target:
        return "contact_seam"
    if inside_neighbor or close_to_neighbor:
        return "neighbor_owned"
    if target_residual_px_value <= target_residual_px * 1.8:
        return "weak_target_boundary"
    return "unowned_outlier"


def _ellipse_model(ellipse: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(ellipse, dict):
        return None
    center = ellipse.get("center_px")
    if center is None and ellipse.get("center_x_px") is not None:
        center = [ellipse.get("center_x_px"), ellipse.get("center_y_px")]
    if center is None:
        return None
    try:
        cx = float(center[0])
        cy = float(center[1])
        major = float(ellipse.get("major_axis_px") or 0.0)
        minor = float(ellipse.get("minor_axis_px") or 0.0)
    except (TypeError, ValueError, IndexError):
        return None
    if major <= 0.0 or minor <= 0.0:
        radius = ellipse.get("radius_px")
        if radius is None:
            return None
        try:
            major = minor = float(radius) * 2.0
        except (TypeError, ValueError):
            return None
    if major <= 0.0 or minor <= 0.0:
        return None
    return {
        "id": ellipse.get("id"),
        "label": ellipse.get("label"),
        "center": np.asarray([cx, cy], dtype=np.float64),
        "major_axis_px": float(major),
        "minor_axis_px": float(minor),
        "angle_deg": float(ellipse.get("angle_deg") or 0.0) % 180.0,
        "source": ellipse.get("source"),
    }


def _ellipse_normalized_radius(point: np.ndarray, ellipse: dict[str, Any]) -> float:
    shifted = np.asarray(point, dtype=np.float64).reshape(2) - ellipse["center"]
    theta = np.deg2rad(float(ellipse.get("angle_deg") or 0.0))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    x_axis = shifted[0] * cos_t + shifted[1] * sin_t
    y_axis = -shifted[0] * sin_t + shifted[1] * cos_t
    rx = max(float(ellipse["major_axis_px"]) / 2.0, 1.0)
    ry = max(float(ellipse["minor_axis_px"]) / 2.0, 1.0)
    return float(np.sqrt((x_axis / rx) ** 2 + (y_axis / ry) ** 2))


def _ellipse_residual_px(point: np.ndarray, ellipse: dict[str, Any]) -> float:
    normalized = _ellipse_normalized_radius(point, ellipse)
    mean_radius = (float(ellipse["major_axis_px"]) + float(ellipse["minor_axis_px"])) / 4.0
    return abs(normalized - 1.0) * mean_radius


def _fallback_circle_ellipse(ball: dict[str, Any]) -> dict[str, Any] | None:
    center = (
        ball.get("source_final_center_px")
        or ball.get("source_refined_center_px")
        or ball.get("source_rough_center_px")
    )
    radius = ball.get("source_radius_px") or ball.get("radius_px")
    if center is None or radius is None:
        return None
    try:
        x = float(center[0])
        y = float(center[1])
        r = float(radius)
    except (TypeError, ValueError, IndexError):
        return None
    return {
        "center_px": [x, y],
        "major_axis_px": 2.0 * r,
        "minor_axis_px": 2.0 * r,
        "angle_deg": 0.0,
        "source": "fallback_circle",
    }


def _compact_ownership(ownership: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": ownership.get("status"),
        "point_count": ownership.get("point_count", 0),
        "summary": ownership.get("summary", {}),
        "target_owned_fraction": ownership.get("target_owned_fraction"),
        "neighbor_owned_fraction": ownership.get("neighbor_owned_fraction"),
    }


def _review_ellipse(ellipse: dict[str, Any] | None) -> dict[str, Any] | None:
    model = _ellipse_model(ellipse)
    if model is None:
        return None
    return {
        "id": model.get("id"),
        "label": model.get("label"),
        "center_px": _round_point(model["center"]),
        "major_axis_px": round(float(model["major_axis_px"]), 4),
        "minor_axis_px": round(float(model["minor_axis_px"]), 4),
        "angle_deg": round(float(model["angle_deg"]), 4),
        "source": model.get("source"),
    }


def _points_array(points_px: list[Any] | tuple[Any, ...] | np.ndarray | None) -> np.ndarray:
    if points_px is None:
        return np.empty((0, 2), dtype=np.float64)
    points = np.asarray(points_px, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return points.reshape(-1, 2)


def _round_point(point: np.ndarray) -> list[float]:
    return [round(float(point[0]), 4), round(float(point[1]), 4)]
