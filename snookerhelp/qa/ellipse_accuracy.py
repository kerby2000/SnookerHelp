from __future__ import annotations

from typing import Any

import numpy as np


def compare_ellipses(
    observed: dict[str, Any] | None,
    expected: dict[str, Any] | None,
    *,
    sample_count: int = 180,
) -> dict[str, Any]:
    """Compare an observed ellipse with a human image-space annotation."""

    if not observed or not expected:
        return {"status": "unavailable", "reason": "both ellipses are required"}
    try:
        observed_values = _ellipse_values(observed)
        expected_values = _ellipse_values(expected)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return {"status": "unavailable", "reason": str(exc)}

    observed_points = ellipse_points(observed, sample_count=sample_count)
    expected_points = ellipse_points(expected, sample_count=sample_count)
    distances = np.linalg.norm(
        observed_points[:, None, :] - expected_points[None, :, :],
        axis=2,
    )
    contour_rms = float(
        np.sqrt(
            0.5
            * (
                np.mean(np.square(np.min(distances, axis=1)))
                + np.mean(np.square(np.min(distances, axis=0)))
            )
        )
    )
    center_error = float(
        np.linalg.norm(observed_values["center"] - expected_values["center"])
    )
    major_error = abs(observed_values["major"] - expected_values["major"])
    minor_error = abs(observed_values["minor"] - expected_values["minor"])
    angle_error = _angle_delta(observed_values["angle"], expected_values["angle"])
    size = max(8.0, 0.25 * (expected_values["major"] + expected_values["minor"]))
    normalized_error = (
        0.40 * min(1.0, center_error / size)
        + 0.45 * min(1.0, contour_rms / size)
        + 0.10 * min(1.0, major_error / max(expected_values["major"], 1.0))
        + 0.05 * min(1.0, angle_error / 45.0)
    )
    return {
        "status": "computed",
        "center_error_px": round(center_error, 4),
        "major_axis_error_px": round(float(major_error), 4),
        "minor_axis_error_px": round(float(minor_error), 4),
        "angle_error_deg": round(float(angle_error), 4),
        "contour_rms_error_px": round(contour_rms, 4),
        "annotation_score": round(100.0 * (1.0 - normalized_error), 2),
        "score_is_ground_truth_based": True,
    }


def ellipse_points(
    ellipse: dict[str, Any],
    *,
    sample_count: int = 180,
) -> np.ndarray:
    values = _ellipse_values(ellipse)
    angles = np.linspace(0.0, 2.0 * np.pi, max(24, int(sample_count)), endpoint=False)
    local = np.column_stack(
        [
            0.5 * values["major"] * np.cos(angles),
            0.5 * values["minor"] * np.sin(angles),
        ]
    )
    radians = np.deg2rad(values["angle"])
    rotation = np.array(
        [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]],
        dtype=np.float64,
    )
    return local @ rotation.T + values["center"][None, :]


def _ellipse_values(ellipse: dict[str, Any]) -> dict[str, Any]:
    center = np.asarray(ellipse["center_px"], dtype=np.float64).reshape(2)
    major = float(ellipse["major_axis_px"])
    minor = float(ellipse["minor_axis_px"])
    angle = float(ellipse.get("angle_deg", 0.0)) % 180.0
    if major <= 0.0 or minor <= 0.0:
        raise ValueError("ellipse axes must be positive")
    if minor > major:
        major, minor = minor, major
        angle = (angle + 90.0) % 180.0
    return {"center": center, "major": major, "minor": minor, "angle": angle}


def _angle_delta(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 90.0) % 180.0 - 90.0)


__all__ = ["compare_ellipses", "ellipse_points"]
