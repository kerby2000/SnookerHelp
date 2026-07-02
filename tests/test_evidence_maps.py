import cv2
import numpy as np

from snookerhelp.recognition.evidence_maps import (
    compute_ball_evidence_maps,
)
from snookerhelp.recognition.source_refinement import (
    fit_radial_boundary_variant_from_feature,
)


def test_green_ball_evidence_maps_weight_chroma_probability() -> None:
    image = np.zeros((180, 220, 3), dtype=np.uint8)
    image[:, :] = (35, 145, 55)
    cv2.circle(image, (110, 90), 34, (70, 190, 110), -1)
    cv2.circle(image, (100, 82), 8, (245, 245, 245), -1)
    projection = {
        "status": "predicted",
        "projected_center_px": [110.0, 90.0],
        "contour_points_px": _ellipse_points(110.0, 90.0, 37.0, 31.0),
    }

    maps = compute_ball_evidence_maps(
        source_image=image,
        center_px=[110.0, 90.0],
        radius_px=34.0,
        label="green",
        sphere_projection=projection,
    )

    assert maps is not None
    summary = maps.summary
    assert summary["local_color_model"]["ball_sample_count"] > 20
    assert summary["local_color_model"]["cloth_sample_count"] > 20
    assert summary["weights"]["chroma"] > summary["weights"]["edge"]
    assert summary["maps"]["ball_vs_cloth_probability"]["p95"] > 0.5


def test_evidence_maps_include_physical_band_without_recovered_points() -> None:
    image = np.zeros((180, 220, 3), dtype=np.uint8)
    image[:, :] = (35, 145, 55)
    cv2.circle(image, (110, 90), 34, (70, 190, 110), -1)
    projection = {
        "status": "predicted",
        "projected_center_px": [110.0, 90.0],
        "contour_points_px": _ellipse_points(110.0, 90.0, 37.0, 31.0),
    }
    maps = compute_ball_evidence_maps(
        source_image=image,
        center_px=[110.0, 90.0],
        radius_px=34.0,
        label="green",
        sphere_projection=projection,
    )

    assert maps is not None
    assert maps.physical_band_score.shape == maps.gray_edge.shape
    assert maps.summary["maps"]["physical_projection_band"]["p95"] > 0.0
    assert maps.summary["maps"]["combined_boundary_score"]["p95"] > 0.0


def test_evidence_map_boundary_variant_fits_observed_ellipse() -> None:
    height, width = 180, 220
    center = np.array([112.0, 88.0], dtype=np.float32)
    rx, ry = 41.0, 33.0
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    normalized_radius = np.sqrt(
        ((xx - center[0]) / rx) ** 2 + ((yy - center[1]) / ry) ** 2
    )
    feature = np.exp(-((normalized_radius - 1.0) ** 2) / (2.0 * 0.022**2)).astype(np.float32)

    variant = fit_radial_boundary_variant_from_feature(
        feature=feature,
        roi=(0, 0, width, height),
        center_px=[float(center[0]), float(center[1])],
        radius_px=37.0,
        evidence_source="evidence_map_synthetic_ring",
        settings={
            "map_boundary_angle_count": 180,
            "map_boundary_minimum_points": 80,
            "map_boundary_minimum_strength": 0.02,
            "boundary_outlier_filter_enabled": True,
        },
    )

    assert variant is not None
    assert variant["status"] == "computed"
    assert len(variant["points_px"]) >= 80
    assert variant["ellipse_fit"] is not None
    assert abs(variant["ellipse_fit"]["center_px"][0] - float(center[0])) < 1.0
    assert abs(variant["ellipse_fit"]["center_px"][1] - float(center[1])) < 1.0
    assert 1.15 < variant["ellipse_fit"]["axis_ratio"] < 1.35


def _ellipse_points(cx: float, cy: float, rx: float, ry: float) -> list[list[float]]:
    angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    return [
        [float(cx + rx * np.cos(angle)), float(cy + ry * np.sin(angle))]
        for angle in angles
    ]
