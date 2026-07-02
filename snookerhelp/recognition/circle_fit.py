from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CircleFitResult:
    x: float
    y: float
    radius: float
    residual_px: float | None
    point_count: int
    success: bool


def fit_circle_least_squares(points: np.ndarray) -> CircleFitResult:
    """Fit a circle to Nx2 points using linear least squares and robust pruning."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 3:
        return CircleFitResult(0.0, 0.0, 0.0, None, len(values), False)

    active = values
    for _ in range(4):
        fit = _algebraic_circle_fit(active)
        if fit is None:
            return CircleFitResult(0.0, 0.0, 0.0, None, len(active), False)
        x, y, radius = fit
        residuals = np.abs(
            np.hypot(active[:, 0] - x, active[:, 1] - y) - radius
        )
        if len(active) < 12:
            break
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        threshold = max(0.35, median + 3.5 * max(1.4826 * mad, 0.05))
        keep = residuals <= threshold
        if int(np.count_nonzero(keep)) < max(8, int(len(active) * 0.55)):
            break
        if bool(np.all(keep)):
            break
        active = active[keep]

    final_fit = _algebraic_circle_fit(active)
    if final_fit is None:
        return CircleFitResult(0.0, 0.0, 0.0, None, len(active), False)
    x, y, radius = final_fit
    radial_errors = np.hypot(active[:, 0] - x, active[:, 1] - y) - radius
    residual = float(np.sqrt(np.mean(radial_errors * radial_errors)))
    return CircleFitResult(
        x=float(x),
        y=float(y),
        radius=float(radius),
        residual_px=residual,
        point_count=len(active),
        success=True,
    )


def refine_circle(
    warped_image: np.ndarray,
    difference: np.ndarray,
    approximate_center: tuple[float, float],
    approximate_radius: float,
    config: dict[str, Any] | None = None,
) -> CircleFitResult:
    """Refine a Hough circle from radial foreground-boundary samples.

    The difference image is expected to be high on the ball and low on the
    unchanged background. One strongest outward drop is sampled per angle,
    followed by a robust least-squares circle fit.
    """
    settings = config or {}
    x0, y0 = (float(approximate_center[0]), float(approximate_center[1]))
    radius0 = float(approximate_radius)
    fallback = CircleFitResult(x0, y0, radius0, None, 0, False)
    if radius0 <= 1 or difference.ndim != 2 or warped_image.ndim < 2:
        return fallback

    angle_count = int(settings.get("angle_count", 180))
    radial_step = float(settings.get("radial_step_px", 0.25))
    inner_factor = float(settings.get("inner_radius_factor", 0.62))
    outer_factor = float(settings.get("outer_radius_factor", 1.42))
    minimum_edge_strength = float(settings.get("minimum_edge_strength", 1.5))
    minimum_points = int(settings.get("minimum_points", 36))

    radii = np.arange(
        radius0 * inner_factor,
        radius0 * outer_factor + radial_step * 0.5,
        radial_step,
        dtype=np.float32,
    )
    if len(radii) < 5:
        return fallback

    angles = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)
    cosines = np.cos(angles).astype(np.float32)
    sines = np.sin(angles).astype(np.float32)
    map_x = x0 + cosines[:, None] * radii[None, :]
    map_y = y0 + sines[:, None] * radii[None, :]

    height, width = difference.shape
    valid_rows = (
        (map_x[:, 0] >= 0)
        & (map_x[:, -1] < width - 1)
        & (map_y.min(axis=1) >= 0)
        & (map_y.max(axis=1) < height - 1)
    )
    if int(np.count_nonzero(valid_rows)) < minimum_points:
        return fallback

    profiles = cv2.remap(
        difference.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    outward_drop = -np.gradient(profiles, radial_step, axis=1)
    proximity = np.exp(
        -0.5 * ((radii - radius0) / max(radius0 * 0.22, 1.0)) ** 2
    )
    weighted_strength = outward_drop * (0.65 + 0.35 * proximity[None, :])
    best_indices = np.argmax(weighted_strength, axis=1)
    row_indices = np.arange(angle_count)
    strengths = outward_drop[row_indices, best_indices]

    finite_strengths = strengths[np.isfinite(strengths) & valid_rows]
    if len(finite_strengths) < minimum_points:
        return fallback
    adaptive_strength = max(
        minimum_edge_strength,
        float(np.percentile(finite_strengths, 30)),
    )
    accepted = valid_rows & np.isfinite(strengths) & (strengths >= adaptive_strength)
    if int(np.count_nonzero(accepted)) < minimum_points:
        return fallback

    refined_radii = radii[best_indices].astype(np.float64)
    for row in np.flatnonzero(accepted):
        index = int(best_indices[row])
        if index <= 0 or index >= len(radii) - 1:
            continue
        left = float(weighted_strength[row, index - 1])
        center = float(weighted_strength[row, index])
        right = float(weighted_strength[row, index + 1])
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-9:
            offset = float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))
            refined_radii[row] += offset * radial_step

    accepted_angles = angles[accepted]
    accepted_radii = refined_radii[accepted]
    points = np.column_stack(
        (
            x0 + np.cos(accepted_angles) * accepted_radii,
            y0 + np.sin(accepted_angles) * accepted_radii,
        )
    )
    fit = fit_circle_least_squares(points)
    if not fit.success or fit.residual_px is None:
        return fallback

    maximum_center_shift = radius0 * float(
        settings.get("maximum_center_shift_radius_factor", 0.35)
    )
    minimum_radius = radius0 * float(settings.get("minimum_radius_factor", 0.72))
    maximum_radius = radius0 * float(settings.get("maximum_radius_factor", 1.30))
    maximum_residual = float(settings.get("maximum_residual_px", 2.5))
    if (
        np.hypot(fit.x - x0, fit.y - y0) > maximum_center_shift
        or not minimum_radius <= fit.radius <= maximum_radius
        or fit.residual_px > maximum_residual
        or fit.point_count < minimum_points
    ):
        return fallback
    return fit


def _algebraic_circle_fit(
    points: np.ndarray,
) -> tuple[float, float, float] | None:
    x_values = points[:, 0]
    y_values = points[:, 1]
    design = np.column_stack((2.0 * x_values, 2.0 * y_values, np.ones(len(points))))
    target = x_values * x_values + y_values * y_values
    try:
        solution, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3:
        return None
    center_x, center_y, constant = solution
    radius_squared = constant + center_x * center_x + center_y * center_y
    if not np.isfinite(radius_squared) or radius_squared <= 0:
        return None
    return float(center_x), float(center_y), float(np.sqrt(radius_squared))
