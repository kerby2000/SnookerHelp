import math

import numpy as np
import pytest

from snookerhelp.recognition.image_model import fit_ellipse_payload
from snookerhelp.recognition.source_refinement import _filter_radial_boundary_points


def test_fit_ellipse_payload_reports_major_axis_angle() -> None:
    center = np.array([120.0, 80.0], dtype=np.float32)
    major_radius = 45.0
    minor_radius = 18.0
    angle_deg = 27.0
    theta = math.radians(angle_deg)
    rotation = np.array(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ],
        dtype=np.float32,
    )
    t = np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False)
    local = np.column_stack((major_radius * np.cos(t), minor_radius * np.sin(t))).astype(np.float32)
    points = center + local @ rotation.T

    ellipse = fit_ellipse_payload(points)

    assert ellipse is not None
    assert ellipse["major_axis_px"] == pytest.approx(major_radius * 2.0, abs=0.15)
    assert ellipse["minor_axis_px"] == pytest.approx(minor_radius * 2.0, abs=0.15)
    assert _angle_delta_deg(float(ellipse["angle_deg"]), angle_deg) < 0.3


def test_radial_boundary_filter_removes_local_radius_spikes() -> None:
    center = np.array([100.0, 80.0], dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * math.pi, 120, endpoint=False)
    radii = np.full_like(angles, 40.0, dtype=np.float64)
    radii[[12, 14]] += 15.0
    points = np.column_stack(
        (
            center[0] + np.cos(angles) * radii,
            center[1] + np.sin(angles) * radii,
        )
    )

    result = _filter_radial_boundary_points(
        angles=angles,
        radii=radii,
        points=points,
        rough_radius=40.0,
        minimum_points=42,
        settings={},
    )

    stats = result["stats"]
    assert stats["status"] == "filtered"
    assert stats["rejected_count"] >= 2
    assert len(result["accepted_points"]) >= 42

    unfiltered = fit_ellipse_payload(points)
    filtered = fit_ellipse_payload(result["accepted_points"])
    assert unfiltered is not None
    assert filtered is not None
    assert abs(float(filtered["center_x_px"]) - center[0]) < abs(
        float(unfiltered["center_x_px"]) - center[0]
    )


def _angle_delta_deg(a: float, b: float) -> float:
    return abs((a - b + 90.0) % 180.0 - 90.0)
