from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from snookerhelp.recognition.image_model import fit_ellipse_payload


def project_sphere_silhouette(
    camera_model: Any,
    center_xyz_mm: list[float] | tuple[float, float, float] | np.ndarray,
    radius_mm: float,
    *,
    sample_count: int = 180,
) -> dict[str, Any]:
    """Project the apparent silhouette of a physical sphere into source pixels.

    This is the v1 physical-model outline. It is available for real calibrated
    pinhole models and for the temporary approximate pinhole-from-corners mode.
    A homography cannot model the tangent cone from the camera center to a 3D
    sphere.
    """
    required = (
        "camera_matrix",
        "distortion_coefficients",
        "rotation_world_to_camera",
        "translation_world_to_camera",
    )
    if any(not hasattr(camera_model, name) for name in required):
        return _unavailable(
            camera_model,
            "camera model lacks pinhole parameters; manual homography cannot predict sphere silhouette",
        )
    approximate = not bool(getattr(camera_model, "is_calibrated", False))

    center_world = np.asarray(center_xyz_mm, dtype=np.float64).reshape(3)
    radius = float(radius_mm)
    if radius <= 0:
        return _unavailable(camera_model, "sphere radius must be positive")

    rotation = np.asarray(camera_model.rotation_world_to_camera, dtype=np.float64)
    translation = np.asarray(camera_model.translation_world_to_camera, dtype=np.float64)
    center_camera = rotation @ center_world + translation.reshape(3)
    distance = float(np.linalg.norm(center_camera))
    if distance <= radius:
        return _unavailable(camera_model, "camera center is inside or on the sphere")
    if center_camera[2] <= 0:
        return _unavailable(camera_model, "sphere center is behind the camera")

    axis = center_camera / distance
    tangent_plane_center = ((distance * distance - radius * radius) / (distance * distance)) * center_camera
    tangent_radius = radius * np.sqrt(distance * distance - radius * radius) / distance
    basis_u, basis_v = _orthonormal_basis(axis)
    angles = np.linspace(0.0, 2.0 * np.pi, int(sample_count), endpoint=False)
    circle_camera = (
        tangent_plane_center[None, :]
        + tangent_radius
        * (
            np.cos(angles)[:, None] * basis_u[None, :]
            + np.sin(angles)[:, None] * basis_v[None, :]
        )
    )

    image_points, _ = cv2.projectPoints(
        circle_camera.reshape(-1, 1, 3),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        np.asarray(camera_model.camera_matrix, dtype=np.float64),
        np.asarray(camera_model.distortion_coefficients, dtype=np.float64),
    )
    contour = image_points.reshape(-1, 2)
    center_px = np.asarray(camera_model.world_point_to_image(center_world), dtype=np.float64)
    ellipse = fit_projected_ellipse(contour)
    return {
        "status": "predicted",
        "method": "pinhole_sphere_tangent_cone",
        "camera_model": getattr(camera_model, "model_name", "calibrated_pinhole"),
        "approximate": bool(approximate),
        "center_xyz_mm": _round_array(center_world),
        "radius_mm": round(radius, 6),
        "projected_center_px": _round_array(center_px),
        "contour_points_px": _round_points(contour),
        "ellipse_fit": ellipse,
        "sample_count": int(len(contour)),
        "note": (
            "Approximate projection of the physical sphere silhouette from "
            "lens/sensor metadata and table corners. This is the blue physics "
            "candidate, not an observed image fit."
            if approximate
            else "Projection of the physical sphere silhouette. This is the "
            "blue physics candidate, not an observed image fit."
        ),
    }


def score_observed_points_against_silhouette(
    observed_points_px: list[list[float]] | tuple[Any, ...] | np.ndarray,
    sphere_projection: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not sphere_projection or sphere_projection.get("status") not in {"predicted", "optimized"}:
        return None
    predicted = np.asarray(
        sphere_projection.get("contour_points_px", []),
        dtype=np.float64,
    ).reshape(-1, 2)
    if observed_points_px is None:
        observed = np.empty((0, 2), dtype=np.float64)
    else:
        observed = np.asarray(observed_points_px, dtype=np.float64).reshape(-1, 2)
    if len(predicted) < 3 or len(observed) < 3:
        return {
            "status": "insufficient_points",
            "observed_point_count": int(len(observed)),
            "predicted_point_count": int(len(predicted)),
        }
    distances = _point_polyline_distances(observed, predicted)
    return {
        "status": "scored",
        "observed_point_count": int(len(observed)),
        "predicted_point_count": int(len(predicted)),
        "mean_abs_error_px": round(float(np.mean(distances)), 4),
        "rms_error_px": round(float(np.sqrt(np.mean(distances * distances))), 4),
        "median_abs_error_px": round(float(np.median(distances)), 4),
        "p95_abs_error_px": round(float(np.percentile(distances, 95)), 4),
        "max_abs_error_px": round(float(np.max(distances)), 4),
    }


def fit_projected_ellipse(points_px: np.ndarray) -> dict[str, Any] | None:
    payload = fit_ellipse_payload(points_px, source="sphere_projection")
    if payload is None:
        return None
    return {
        "status": "predicted",
        "center_px": [
            round(float(payload["center_x_px"]), 4),
            round(float(payload["center_y_px"]), 4),
        ],
        "major_axis_px": round(float(payload["major_axis_px"]), 4),
        "minor_axis_px": round(float(payload["minor_axis_px"]), 4),
        "angle_deg": round(float(payload["angle_deg"]), 4),
        "axis_ratio": round(float(payload["axis_ratio"]), 6),
        "source": "sphere_projection",
    }


def _point_polyline_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    starts = polyline
    ends = np.roll(polyline, -1, axis=0)
    distances = []
    for point in points:
        segment = ends - starts
        length_sq = np.sum(segment * segment, axis=1)
        valid = length_sq > 1e-12
        t = np.zeros(len(starts), dtype=np.float64)
        t[valid] = np.sum((point[None, :] - starts[valid]) * segment[valid], axis=1) / length_sq[valid]
        t = np.clip(t, 0.0, 1.0)
        closest = starts + t[:, None] * segment
        distances.append(float(np.min(np.linalg.norm(closest - point[None, :], axis=1))))
    return np.asarray(distances, dtype=np.float64)


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(helper, axis))) > 0.92:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    basis_u = helper - axis * float(np.dot(helper, axis))
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(axis, basis_u)
    basis_v /= np.linalg.norm(basis_v)
    return basis_u, basis_v


def _unavailable(camera_model: Any, reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "method": "pinhole_sphere_tangent_cone",
        "camera_model": getattr(camera_model, "model_name", "unknown"),
        "reason": reason,
    }


def _round_points(points: np.ndarray) -> list[list[float]]:
    return [
        [round(float(point[0]), 4), round(float(point[1]), 4)]
        for point in np.asarray(points).reshape(-1, 2)
    ]


def _round_array(values: np.ndarray) -> list[float]:
    return [round(float(value), 4) for value in np.asarray(values).reshape(-1)]
