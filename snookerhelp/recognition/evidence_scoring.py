from __future__ import annotations

from typing import Any

import numpy as np

from snookerhelp.recognition.sphere_projection import (
    score_observed_points_against_silhouette,
)


def boundary_view_score(
    *,
    points_px: list[Any],
    rejected_points_px: list[Any],
    ellipse_fit: dict[str, Any] | None,
    sphere_projection: dict[str, Any] | None,
    radius_px: float | None,
) -> dict[str, Any]:
    """Return the shared diagnostic score used by reports and experiments.

    This score is deliberately named diagnostic: it is not ground-truth
    accuracy. The decomposed components are returned so callers do not treat one
    opaque percentage as evidence.
    """

    accepted_count = len(points_px or [])
    rejected_count = len(rejected_points_px or [])
    total_count = accepted_count + rejected_count
    formula = (
        "diagnostic_score = 45% physical residual + 30% accepted point count "
        "+ 20% inlier ratio + 5% ellipse availability"
    )
    if accepted_count < 3 or not ellipse_fit:
        return {
            "status": "unavailable",
            "score": None,
            "level": "unknown",
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "reason": "not enough accepted points or ellipse fit is missing",
            "formula": formula,
            "is_ground_truth_accuracy": False,
        }

    silhouette_score = score_observed_points_against_silhouette(
        points_px,
        sphere_projection,
    )
    rms_error = None
    residual_component = 0.5
    residual_reason = "physical outline unavailable; neutral residual component"
    if silhouette_score and silhouette_score.get("status") == "scored":
        rms_error = float(silhouette_score.get("rms_error_px") or 999.0)
        normalizer = max(8.0, float(radius_px or 40.0) * 0.30)
        residual_component = float(
            np.clip(1.0 - rms_error / normalizer, 0.0, 1.0)
        )
        residual_reason = "accepted points scored against physical sphere outline"

    point_component = float(np.clip(accepted_count / 90.0, 0.0, 1.0))
    inlier_ratio = accepted_count / max(1, total_count)
    inlier_component = float(np.clip(inlier_ratio, 0.0, 1.0))
    ellipse_component = 1.0
    score = (
        0.45 * residual_component
        + 0.30 * point_component
        + 0.20 * inlier_component
        + 0.05 * ellipse_component
    )
    level = "high" if score >= 0.78 else "medium" if score >= 0.45 else "low"
    return {
        "status": "computed",
        "score": round(float(score), 4),
        "level": level,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "inlier_ratio": round(float(inlier_ratio), 4),
        "physical_rms_error_px": (
            round(float(rms_error), 4) if rms_error is not None else None
        ),
        "components": {
            "physical_residual": round(float(residual_component), 4),
            "accepted_point_count": round(float(point_component), 4),
            "inlier_ratio": round(float(inlier_component), 4),
            "ellipse_available": round(float(ellipse_component), 4),
        },
        "reason": residual_reason,
        "formula": formula,
        "is_ground_truth_accuracy": False,
    }


__all__ = ["boundary_view_score"]
