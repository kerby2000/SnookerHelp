from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def fit_ellipse_payload(
    points_px: np.ndarray,
    *,
    source: str | None = None,
) -> dict[str, Any] | None:
    """Fit an ellipse and normalize angle to the major-axis direction.

    OpenCV's `fitEllipse` returns an angle associated with the first returned
    axis dimension, not necessarily with the larger axis. The review UI draws
    `major_axis_px` as `rx` and `minor_axis_px` as `ry`, so if we swap axes we
    must also rotate the angle by 90 degrees. Without this normalization,
    elongated balls can display an ellipse that is visually perpendicular to
    the actual silhouette.
    """
    points = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    if len(points) < 5:
        return None
    try:
        (center_x, center_y), (axis_a, axis_b), angle_deg = cv2.fitEllipse(points)
    except cv2.error:
        return None

    axis_a = float(axis_a)
    axis_b = float(axis_b)
    if axis_a <= 0.0 or axis_b <= 0.0:
        return None

    if axis_a >= axis_b:
        major = axis_a
        minor = axis_b
        major_angle = float(angle_deg)
    else:
        major = axis_b
        minor = axis_a
        major_angle = float(angle_deg) + 90.0
    major_angle = major_angle % 180.0
    payload: dict[str, Any] = {
        "status": "candidate",
        "center_px": [float(center_x), float(center_y)],
        "center_x_px": float(center_x),
        "center_y_px": float(center_y),
        "major_axis_px": major,
        "minor_axis_px": minor,
        "angle_deg": major_angle,
        "axis_ratio": major / minor,
    }
    if source:
        payload["source"] = source
    return payload
