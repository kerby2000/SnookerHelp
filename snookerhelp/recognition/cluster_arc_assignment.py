from __future__ import annotations

from typing import Any

import numpy as np


def assign_boundary_points_globally(
    points_px: list[Any] | tuple[Any, ...] | np.ndarray,
    *,
    ellipses_by_id: dict[int, dict[str, Any]],
    residual_gate_px: float = 5.0,
    ambiguity_margin_px: float = 0.6,
    dedupe_quantum_px: float = 0.75,
) -> dict[str, Any]:
    """Give each observed boundary sample at most one cluster owner.

    Per-ball radial searches see the same contact seam, neighbour silhouette,
    or lamp reflection more than once.  Treating those samples independently
    lets several balls claim the same pixels.  This function deduplicates the
    union of samples and compares every point with every proposed ellipse.

    A point is accepted only when one ellipse is both close enough and clearly
    better than the runner-up.  Ambiguous contact samples remain visible in the
    diagnostic payload but do not steer a center fit.
    """

    points = _dedupe_points(points_px, quantum_px=dedupe_quantum_px)
    model_ids = sorted(
        ball_id
        for ball_id, ellipse in ellipses_by_id.items()
        if _valid_ellipse(ellipse)
    )
    if len(points) == 0 or not model_ids:
        return {
            "status": "no_points_or_models",
            "point_count": int(len(points)),
            "model_count": len(model_ids),
            "by_ball_id": {},
            "ambiguous_points_px": [],
            "unowned_points_px": _round_points(points),
            "points": [],
        }

    residuals = np.column_stack(
        [_ellipse_residuals_px(points, ellipses_by_id[ball_id]) for ball_id in model_ids]
    )
    order = np.argsort(residuals, axis=1)
    best_index = order[:, 0]
    best_residual = residuals[np.arange(len(points)), best_index]
    if len(model_ids) > 1:
        second_residual = residuals[np.arange(len(points)), order[:, 1]]
    else:
        second_residual = np.full(len(points), np.inf, dtype=np.float64)

    accepted = best_residual <= float(residual_gate_px)
    unambiguous = (
        second_residual - best_residual
    ) >= float(ambiguity_margin_px)
    owned = accepted & unambiguous
    ambiguous = accepted & ~unambiguous

    by_ball_id: dict[str, dict[str, Any]] = {}
    for model_index, ball_id in enumerate(model_ids):
        mask = owned & (best_index == model_index)
        owned_points = points[mask]
        owned_residuals = best_residual[mask]
        by_ball_id[str(ball_id)] = {
            "ball_id": int(ball_id),
            "owned_point_count": int(mask.sum()),
            "owned_points_px": _round_points(owned_points),
            "mean_residual_px": _round_or_none(
                float(np.mean(owned_residuals)) if len(owned_residuals) else None
            ),
            "rms_residual_px": _round_or_none(
                float(np.sqrt(np.mean(owned_residuals**2)))
                if len(owned_residuals)
                else None
            ),
        }

    point_payload: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        if owned[index]:
            category = "owned"
            owner_id: int | None = int(model_ids[int(best_index[index])])
        elif ambiguous[index]:
            category = "ambiguous_contact"
            owner_id = None
        else:
            category = "unowned_noise"
            owner_id = None
        point_payload.append(
            {
                "point_px": _round_point(point),
                "category": category,
                "owner_ball_id": owner_id,
                "best_model_ball_id": int(model_ids[int(best_index[index])]),
                "best_residual_px": round(float(best_residual[index]), 4),
                "second_residual_px": (
                    None
                    if not np.isfinite(second_residual[index])
                    else round(float(second_residual[index]), 4)
                ),
            }
        )

    return {
        "status": "computed",
        "method": "global_nearest_ellipse_with_ambiguity_gate",
        "point_count": int(len(points)),
        "model_count": len(model_ids),
        "owned_point_count": int(owned.sum()),
        "ambiguous_point_count": int(ambiguous.sum()),
        "unowned_point_count": int((~accepted).sum()),
        "owned_fraction": round(float(owned.mean()), 4),
        "residual_gate_px": round(float(residual_gate_px), 4),
        "ambiguity_margin_px": round(float(ambiguity_margin_px), 4),
        "dedupe_quantum_px": round(float(dedupe_quantum_px), 4),
        "by_ball_id": by_ball_id,
        "ambiguous_points_px": _round_points(points[ambiguous]),
        "unowned_points_px": _round_points(points[~accepted]),
        "points": point_payload,
        "note": (
            "White/accepted points have one global owner. Ambiguous contact "
            "samples and interior highlights remain diagnostic and do not fit "
            "more than one ball."
        ),
    }


def ellipse_residuals_px(
    points_px: list[Any] | tuple[Any, ...] | np.ndarray,
    ellipse: dict[str, Any],
) -> np.ndarray:
    points = _points_array(points_px)
    if len(points) == 0 or not _valid_ellipse(ellipse):
        return np.full(len(points), np.inf, dtype=np.float64)
    return _ellipse_residuals_px(points, ellipse)


def _ellipse_residuals_px(points: np.ndarray, ellipse: dict[str, Any]) -> np.ndarray:
    center = np.asarray(ellipse["center_px"], dtype=np.float64).reshape(2)
    shifted = points - center.reshape(1, 2)
    angle = np.deg2rad(float(ellipse.get("angle_deg") or 0.0))
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    local_x = shifted[:, 0] * cos_a + shifted[:, 1] * sin_a
    local_y = -shifted[:, 0] * sin_a + shifted[:, 1] * cos_a
    rx = max(float(ellipse["major_axis_px"]) / 2.0, 1.0)
    ry = max(float(ellipse["minor_axis_px"]) / 2.0, 1.0)
    normalized = np.sqrt((local_x / rx) ** 2 + (local_y / ry) ** 2)
    return np.abs(normalized - 1.0) * ((rx + ry) * 0.5)


def _dedupe_points(
    points_px: list[Any] | tuple[Any, ...] | np.ndarray,
    *,
    quantum_px: float,
) -> np.ndarray:
    points = _points_array(points_px)
    if len(points) <= 1:
        return points
    quantum = max(float(quantum_px), 1e-3)
    keys = np.rint(points / quantum).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def _points_array(
    points_px: list[Any] | tuple[Any, ...] | np.ndarray | None,
) -> np.ndarray:
    try:
        source = [] if points_px is None else points_px
        points = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return np.empty((0, 2), dtype=np.float64)
    if len(points) == 0:
        return points
    return points[np.all(np.isfinite(points), axis=1)]


def _valid_ellipse(ellipse: dict[str, Any] | None) -> bool:
    if not isinstance(ellipse, dict):
        return False
    try:
        center = np.asarray(ellipse.get("center_px"), dtype=np.float64).reshape(2)
        major = float(ellipse.get("major_axis_px"))
        minor = float(ellipse.get("minor_axis_px"))
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(center)) and major > 4.0 and minor > 4.0)


def _round_points(points: np.ndarray) -> list[list[float]]:
    return [_round_point(point) for point in np.asarray(points).reshape(-1, 2)]


def _round_point(point: np.ndarray) -> list[float]:
    return [round(float(point[0]), 4), round(float(point[1]), 4)]


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


__all__ = ["assign_boundary_points_globally", "ellipse_residuals_px"]
