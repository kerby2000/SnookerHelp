from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np

from snookerhelp.recognition.circle_fit import fit_circle_least_squares
from snookerhelp.recognition.image_model import fit_ellipse_payload
from snookerhelp.calibration.homography_bootstrap import TableWarp


@dataclass(frozen=True)
class SourceBallRefinement:
    rough_x: float
    rough_y: float
    rough_radius: float
    x: float
    y: float
    radius: float
    residual_px: float | None
    point_count: int
    success: bool
    roi: tuple[int, int, int, int]
    boundary_points: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    boundary_rejected_points: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    boundary_filter_stats: dict[str, Any] = field(default_factory=dict)
    ellipse_fit: dict[str, float | str] | None = None
    mask_centroid: tuple[float, float] | None = None
    mask_area_px: float | None = None
    mask_contour_points: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    silhouette_ellipse_fit: dict[str, float | str] | None = None
    boundary_evidence_source: str | None = None


def estimate_source_radius_px(
    table_warp: TableWarp,
    warped_center: tuple[float, float],
    warped_radius_px: float,
) -> float:
    """Estimate local source-image ball radius from inverse homography samples."""
    center_x, center_y = float(warped_center[0]), float(warped_center[1])
    radius = float(warped_radius_px)
    if radius <= 0:
        return 0.0
    warped_points = np.float32(
        [
            [center_x, center_y],
            [center_x + radius, center_y],
            [center_x - radius, center_y],
            [center_x, center_y + radius],
            [center_x, center_y - radius],
        ]
    )
    source_points = table_warp.warped_to_source(warped_points)
    source_center = source_points[0]
    distances = np.linalg.norm(source_points[1:] - source_center, axis=1)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return 0.0
    return float(np.median(distances))


def source_roi_bounds(
    image_shape: tuple[int, int] | tuple[int, int, int],
    center: tuple[float, float],
    radius_px: float,
    margin_factor: float = 1.85,
    minimum_half_size_px: int = 24,
) -> tuple[int, int, int, int]:
    """Return clipped [x0, y0, x1, y1] ROI bounds around a source candidate."""
    height, width = int(image_shape[0]), int(image_shape[1])
    half_size = max(
        int(round(float(radius_px) * float(margin_factor))),
        int(minimum_half_size_px),
    )
    center_x, center_y = float(center[0]), float(center[1])
    x0 = max(0, int(np.floor(center_x - half_size)))
    y0 = max(0, int(np.floor(center_y - half_size)))
    x1 = min(width, int(np.ceil(center_x + half_size + 1)))
    y1 = min(height, int(np.ceil(center_y + half_size + 1)))
    return x0, y0, x1, y1


def refine_source_ball(
    source_image: np.ndarray,
    source_background: np.ndarray | None,
    table_warp: TableWarp,
    warped_center: tuple[float, float],
    warped_radius_px: float,
    config: dict[str, Any] | None = None,
) -> SourceBallRefinement:
    """Refine a warped candidate by fitting the ball contour in source pixels.

    The rough candidate still comes from the cloth-plane warped detector. This
    function maps it back to the original camera image, builds a local
    source-image ROI, samples the ball/background boundary radially in that
    ROI, and fits a circle in source coordinates.
    """
    settings = config or {}
    rough_point = table_warp.warped_to_source(
        np.float32([[float(warped_center[0]), float(warped_center[1])]])
    )[0]
    rough_x = float(rough_point[0])
    rough_y = float(rough_point[1])
    rough_radius = estimate_source_radius_px(
        table_warp,
        warped_center,
        warped_radius_px,
    )
    roi = source_roi_bounds(
        source_image.shape,
        (rough_x, rough_y),
        rough_radius,
        margin_factor=float(settings.get("roi_margin_radius_factor", 1.85)),
        minimum_half_size_px=int(settings.get("minimum_roi_half_size_px", 24)),
    )
    fallback = SourceBallRefinement(
        rough_x=rough_x,
        rough_y=rough_y,
        rough_radius=rough_radius,
        x=rough_x,
        y=rough_y,
        radius=rough_radius,
        residual_px=None,
        point_count=0,
        success=False,
        roi=roi,
        boundary_points=(),
        ellipse_fit=None,
    )
    if (
        not bool(settings.get("enabled", True))
        or source_image.ndim < 2
        or rough_radius <= 2.0
    ):
        return fallback

    x0, y0, x1, y1 = roi
    if x1 - x0 < 12 or y1 - y0 < 12:
        return fallback
    if source_background is None or source_background.shape != source_image.shape:
        feature = _edge_feature(source_image[y0:y1, x0:x1])
        use_outward_drop = False
    else:
        feature = _difference_feature(
            source_image[y0:y1, x0:x1],
            source_background[y0:y1, x0:x1],
            int(settings.get("blur_kernel", 5)),
        )
        use_outward_drop = True

    mask_evidence = _mask_evidence_from_feature(
        feature=feature,
        roi_origin=(x0, y0),
        rough_center=(rough_x, rough_y),
        rough_radius=rough_radius,
        settings=settings,
    )
    fallback = _with_mask_evidence(fallback, mask_evidence)

    fit = _fit_from_radial_boundary(
        feature=feature,
        roi_origin=(x0, y0),
        rough_center=(rough_x, rough_y),
        rough_radius=rough_radius,
        settings=settings,
        use_outward_drop=use_outward_drop,
        evidence_source="radial_boundary",
    )
    if fit is None and source_background is not None:
        fit = _fit_from_radial_boundary(
            feature=_edge_feature(source_image[y0:y1, x0:x1]),
            roi_origin=(x0, y0),
            rough_center=(rough_x, rough_y),
            rough_radius=rough_radius,
            settings=settings,
            use_outward_drop=False,
            evidence_source="radial_edge",
        )
    if fit is None:
        return fallback
    return _with_mask_evidence(fit, mask_evidence)


def _difference_feature(
    source_crop: np.ndarray,
    background_crop: np.ndarray,
    blur_kernel: int,
) -> np.ndarray:
    kernel = max(1, int(blur_kernel))
    if kernel % 2 == 0:
        kernel += 1
    source_blur = cv2.GaussianBlur(source_crop, (kernel, kernel), 0)
    background_blur = cv2.GaussianBlur(background_crop, (kernel, kernel), 0)
    source_lab = cv2.cvtColor(source_blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    background_lab = cv2.cvtColor(background_blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    difference = np.linalg.norm(source_lab - background_lab, axis=2)
    return cv2.GaussianBlur(difference.astype(np.float32), (3, 3), 0)


def _edge_feature(source_crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(source_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120).astype(np.float32)
    return cv2.GaussianBlur(edges, (5, 5), 0)


def _with_mask_evidence(
    refinement: SourceBallRefinement,
    evidence: dict[str, Any] | None,
) -> SourceBallRefinement:
    if not evidence:
        return refinement
    return replace(
        refinement,
        mask_centroid=evidence.get("centroid"),
        mask_area_px=evidence.get("area_px"),
        mask_contour_points=evidence.get("contour_points", ()),
        silhouette_ellipse_fit=evidence.get("ellipse_fit"),
    )


def _mask_evidence_from_feature(
    feature: np.ndarray,
    roi_origin: tuple[int, int],
    rough_center: tuple[float, float],
    rough_radius: float,
    settings: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract coarse blob evidence from the source ROI feature image.

    This is intentionally an evidence channel, not a detector decision. It gives
    the review UI a mask centroid and a silhouette ellipse even when the radial
    circle fit is rejected.
    """
    if feature.ndim != 2 or feature.size == 0 or rough_radius <= 2.0:
        return None

    local_x = float(rough_center[0]) - float(roi_origin[0])
    local_y = float(rough_center[1]) - float(roi_origin[1])
    height, width = feature.shape[:2]
    if not (0 <= local_x < width and 0 <= local_y < height):
        return None

    yy, xx = np.mgrid[0:height, 0:width]
    distance = np.hypot(xx.astype(np.float32) - local_x, yy.astype(np.float32) - local_y)
    outer_radius = rough_radius * float(settings.get("mask_outer_radius_factor", 1.75))
    search_mask = distance <= outer_radius
    values = feature[search_mask & np.isfinite(feature)]
    if values.size < 12:
        return None

    threshold = max(
        float(settings.get("minimum_mask_feature_strength", 1.0)),
        float(np.percentile(values, float(settings.get("mask_threshold_percentile", 62.0)))),
    )
    binary = np.zeros_like(feature, dtype=np.uint8)
    binary[(feature >= threshold) & search_mask] = 255
    kernel_size = int(settings.get("mask_morph_kernel_px", 5))
    if kernel_size > 1:
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    component = _component_near_point(binary, (local_x, local_y))
    if component is None:
        return None
    component_mask, area = component
    if area < max(20.0, rough_radius * rough_radius * 0.06):
        return None

    moments = cv2.moments(component_mask, binaryImage=True)
    if abs(moments["m00"]) < 1e-9:
        return None
    centroid = (
        float(roi_origin[0]) + float(moments["m10"] / moments["m00"]),
        float(roi_origin[1]) + float(moments["m01"] / moments["m00"]),
    )

    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {
            "centroid": centroid,
            "area_px": float(area),
            "contour_points": (),
            "ellipse_fit": None,
        }
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    contour_global = contour + np.array([[roi_origin[0], roi_origin[1]]], dtype=np.float32)
    contour_points = _sample_points(contour_global, int(settings.get("mask_contour_max_points", 240)))
    ellipse_fit = (
        fit_ellipse_payload(contour_global, source="mask_contour")
        if len(contour_global) >= 5
        else None
    )
    return {
        "centroid": centroid,
        "area_px": float(area),
        "contour_points": tuple((float(x), float(y)) for x, y in contour_points),
        "ellipse_fit": ellipse_fit,
    }


def _component_near_point(
    binary: np.ndarray,
    point: tuple[float, float],
) -> tuple[np.ndarray, float] | None:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return None
    px = int(np.clip(round(point[0]), 0, binary.shape[1] - 1))
    py = int(np.clip(round(point[1]), 0, binary.shape[0] - 1))
    center_label = int(labels[py, px])
    if center_label > 0:
        mask = np.where(labels == center_label, 255, 0).astype(np.uint8)
        return mask, float(stats[center_label, cv2.CC_STAT_AREA])

    candidates = []
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        cx, cy = centroids[label]
        distance = float(np.hypot(cx - point[0], cy - point[1]))
        candidates.append((distance / max(np.sqrt(area), 1.0), label, area))
    if not candidates:
        return None
    _, label, area = min(candidates, key=lambda item: item[0])
    mask = np.where(labels == label, 255, 0).astype(np.uint8)
    return mask, area


def _sample_points(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int32)
    return points[indices]


def _fit_from_radial_boundary(
    feature: np.ndarray,
    roi_origin: tuple[int, int],
    rough_center: tuple[float, float],
    rough_radius: float,
    settings: dict[str, Any],
    use_outward_drop: bool,
    evidence_source: str,
    neighbor_ellipses: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> SourceBallRefinement | None:
    angle_count = int(settings.get("angle_count", 180))
    radial_step = float(settings.get("radial_step_px", 0.35))
    inner_factor = float(settings.get("inner_radius_factor", 0.55))
    outer_factor = float(settings.get("outer_radius_factor", 1.45))
    minimum_points = int(settings.get("minimum_points", 42))
    minimum_edge_strength = float(settings.get("minimum_edge_strength", 1.25))

    if feature.ndim != 2 or rough_radius <= 2.0:
        return None
    radii = np.arange(
        rough_radius * inner_factor,
        rough_radius * outer_factor + radial_step * 0.5,
        radial_step,
        dtype=np.float32,
    )
    if len(radii) < 6:
        return None

    rough_x, rough_y = float(rough_center[0]), float(rough_center[1])
    local_x = rough_x - float(roi_origin[0])
    local_y = rough_y - float(roi_origin[1])
    angles = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)
    cosines = np.cos(angles).astype(np.float32)
    sines = np.sin(angles).astype(np.float32)
    map_x = local_x + cosines[:, None] * radii[None, :]
    map_y = local_y + sines[:, None] * radii[None, :]

    height, width = feature.shape[:2]
    valid_rows = (
        (map_x.min(axis=1) >= 0)
        & (map_x.max(axis=1) < width - 1)
        & (map_y.min(axis=1) >= 0)
        & (map_y.max(axis=1) < height - 1)
    )
    if int(np.count_nonzero(valid_rows)) < minimum_points:
        return None

    profiles = cv2.remap(
        feature.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    proximity = np.exp(
        -0.5 * ((radii - rough_radius) / max(rough_radius * 0.24, 1.0)) ** 2
    )
    if use_outward_drop:
        strength_map = -np.gradient(profiles, radial_step, axis=1)
        weighted_strength = strength_map * (0.65 + 0.35 * proximity[None, :])
    else:
        strength_map = profiles
        weighted_strength = profiles * (0.65 + 0.35 * proximity[None, :])

    best_indices = np.argmax(weighted_strength, axis=1)
    row_indices = np.arange(angle_count)
    strengths = strength_map[row_indices, best_indices]
    finite_strengths = strengths[np.isfinite(strengths) & valid_rows]
    if len(finite_strengths) < minimum_points:
        return None
    adaptive_strength = max(
        minimum_edge_strength,
        float(np.percentile(finite_strengths, 30)),
    )
    accepted = valid_rows & np.isfinite(strengths) & (strengths >= adaptive_strength)
    if int(np.count_nonzero(accepted)) < minimum_points:
        return None

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
    raw_points = np.column_stack(
        (
            rough_x + np.cos(accepted_angles) * accepted_radii,
            rough_y + np.sin(accepted_angles) * accepted_radii,
        )
    )
    filter_result = _filter_radial_boundary_points(
        angles=accepted_angles,
        radii=accepted_radii,
        points=raw_points,
        rough_radius=rough_radius,
        minimum_points=minimum_points,
        settings=settings,
        neighbor_ellipses=neighbor_ellipses,
    )
    points = filter_result["accepted_points"]
    rejected_points = filter_result["rejected_points"]
    filter_stats = filter_result["stats"]
    ellipse_source = (
        f"{evidence_source}_filtered"
        if len(rejected_points)
        else evidence_source
    )
    ellipse_fit = fit_ellipse_payload(points, source=ellipse_source)
    fit = fit_circle_least_squares(points)
    if not fit.success or fit.residual_px is None:
        return _radial_evidence_only_refinement(
            rough_x=rough_x,
            rough_y=rough_y,
            rough_radius=rough_radius,
            roi_origin=roi_origin,
            local_center=(local_x, local_y),
            feature_shape=feature.shape,
            points=points,
            rejected_points=rejected_points,
            filter_stats=filter_stats,
            ellipse_fit=ellipse_fit,
            evidence_source=ellipse_source,
        )

    maximum_center_shift = rough_radius * float(
        settings.get("maximum_center_shift_radius_factor", 0.45)
    )
    minimum_radius = rough_radius * float(settings.get("minimum_radius_factor", 0.65))
    maximum_radius = rough_radius * float(settings.get("maximum_radius_factor", 1.45))
    maximum_residual = float(settings.get("maximum_residual_px", 3.5))
    if (
        np.hypot(fit.x - rough_x, fit.y - rough_y) > maximum_center_shift
        or not minimum_radius <= fit.radius <= maximum_radius
        or fit.residual_px > maximum_residual
        or fit.point_count < minimum_points
    ):
        return _radial_evidence_only_refinement(
            rough_x=rough_x,
            rough_y=rough_y,
            rough_radius=rough_radius,
            roi_origin=roi_origin,
            local_center=(local_x, local_y),
            feature_shape=feature.shape,
            points=points,
            rejected_points=rejected_points,
            filter_stats=filter_stats,
            ellipse_fit=ellipse_fit,
            evidence_source=ellipse_source,
        )

    roi = source_roi_bounds(
        (feature.shape[0], feature.shape[1]),
        (local_x, local_y),
        rough_radius,
    )
    return SourceBallRefinement(
        rough_x=rough_x,
        rough_y=rough_y,
        rough_radius=rough_radius,
        x=fit.x,
        y=fit.y,
        radius=fit.radius,
        residual_px=fit.residual_px,
        point_count=fit.point_count,
        success=True,
        roi=(
            roi_origin[0] + roi[0],
            roi_origin[1] + roi[1],
            roi_origin[0] + roi[2],
            roi_origin[1] + roi[3],
        ),
        boundary_points=tuple((float(x), float(y)) for x, y in points),
        boundary_rejected_points=tuple((float(x), float(y)) for x, y in rejected_points),
        boundary_filter_stats=filter_stats,
        ellipse_fit=ellipse_fit,
        boundary_evidence_source=ellipse_source,
    )


def fit_radial_boundary_variant_from_feature(
    *,
    feature: np.ndarray,
    roi: tuple[int, int, int, int],
    center_px: tuple[float, float] | list[float] | None,
    radius_px: float | None,
    evidence_source: str,
    settings: dict[str, Any] | None = None,
    use_outward_drop: bool = False,
    neighbor_ellipses: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any] | None:
    """Sample a source boundary from an arbitrary crop-local scalar feature map.

    This is used for review/diagnostic evidence-map variants. It deliberately
    returns data only; it does not replace the detector final center or
    confidence by itself.
    """

    if center_px is None or radius_px is None:
        return None
    try:
        center = (float(center_px[0]), float(center_px[1]))
        radius = float(radius_px)
    except (TypeError, ValueError, IndexError):
        return None
    if feature.ndim != 2 or feature.size == 0 or radius <= 2.0:
        return None

    cfg = dict(settings or {})
    cfg.setdefault("angle_count", int(cfg.get("map_boundary_angle_count", 180)))
    cfg.setdefault("radial_step_px", float(cfg.get("map_boundary_radial_step_px", 0.35)))
    cfg.setdefault("inner_radius_factor", float(cfg.get("map_boundary_inner_radius_factor", 0.55)))
    cfg.setdefault("outer_radius_factor", float(cfg.get("map_boundary_outer_radius_factor", 1.45)))
    cfg.setdefault("minimum_points", int(cfg.get("map_boundary_minimum_points", 32)))
    cfg.setdefault("minimum_edge_strength", float(cfg.get("map_boundary_minimum_strength", 0.035)))
    cfg.setdefault("maximum_center_shift_radius_factor", 0.65)
    cfg.setdefault("minimum_radius_factor", 0.45)
    cfg.setdefault("maximum_radius_factor", 1.65)
    cfg.setdefault("maximum_residual_px", 9999.0)

    normalized = np.asarray(feature, dtype=np.float32)
    finite = normalized[np.isfinite(normalized)]
    if finite.size and float(np.max(finite)) > 1.5:
        lo = float(np.percentile(finite, 1))
        hi = float(np.percentile(finite, 99))
        if hi > lo + 1e-6:
            normalized = np.clip((normalized - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    fit = _fit_from_radial_boundary(
        feature=normalized,
        roi_origin=(int(roi[0]), int(roi[1])),
        rough_center=center,
        rough_radius=radius,
        settings=cfg,
        use_outward_drop=bool(use_outward_drop),
        evidence_source=evidence_source,
        neighbor_ellipses=neighbor_ellipses,
    )
    if fit is None or not fit.boundary_points:
        return {
            "status": "unavailable",
            "source": evidence_source,
            "points_px": [],
            "rejected_points_px": [],
            "ellipse_fit": None,
            "filter": {
                "status": "unavailable",
                "accepted_count": 0,
                "rejected_count": 0,
            },
        }

    return {
        "status": "computed",
        "source": fit.boundary_evidence_source or evidence_source,
        "points_px": [
            [round(float(x), 4), round(float(y), 4)]
            for x, y in fit.boundary_points
        ],
        "rejected_points_px": [
            [round(float(x), 4), round(float(y), 4)]
            for x, y in fit.boundary_rejected_points
        ],
        "ellipse_fit": _review_ellipse_payload(fit.ellipse_fit),
        "filter": fit.boundary_filter_stats,
        "circle_baseline": {
            "success": bool(fit.success),
            "center_px": [round(float(fit.x), 4), round(float(fit.y), 4)],
            "radius_px": round(float(fit.radius), 4),
            "residual_px": (
                None if fit.residual_px is None else round(float(fit.residual_px), 4)
            ),
            "point_count": int(fit.point_count),
        },
    }


def _review_ellipse_payload(ellipse: dict[str, Any] | None) -> dict[str, Any] | None:
    if not ellipse:
        return None
    return {
        "status": ellipse.get("status", "candidate"),
        "center_px": [
            round(float(ellipse["center_x_px"]), 4),
            round(float(ellipse["center_y_px"]), 4),
        ],
        "major_axis_px": round(float(ellipse["major_axis_px"]), 4),
        "minor_axis_px": round(float(ellipse["minor_axis_px"]), 4),
        "angle_deg": round(float(ellipse["angle_deg"]), 4),
        "axis_ratio": (
            None
            if ellipse.get("axis_ratio") is None
            else round(float(ellipse["axis_ratio"]), 4)
        ),
        "source": ellipse.get("source"),
    }


def _radial_evidence_only_refinement(
    *,
    rough_x: float,
    rough_y: float,
    rough_radius: float,
    roi_origin: tuple[int, int],
    local_center: tuple[float, float],
    feature_shape: tuple[int, int],
    points: np.ndarray,
    rejected_points: np.ndarray,
    filter_stats: dict[str, Any],
    ellipse_fit: dict[str, Any] | None,
    evidence_source: str,
) -> SourceBallRefinement | None:
    """Return C-only radial/edge evidence even when circle fitting is rejected."""
    if ellipse_fit is None or len(points) < 5:
        return None
    roi = source_roi_bounds(
        (feature_shape[0], feature_shape[1]),
        local_center,
        rough_radius,
    )
    return SourceBallRefinement(
        rough_x=rough_x,
        rough_y=rough_y,
        rough_radius=rough_radius,
        x=rough_x,
        y=rough_y,
        radius=rough_radius,
        residual_px=None,
        point_count=int(len(points)),
        success=False,
        roi=(
            roi_origin[0] + roi[0],
            roi_origin[1] + roi[1],
            roi_origin[0] + roi[2],
            roi_origin[1] + roi[3],
        ),
        boundary_points=tuple((float(x), float(y)) for x, y in points),
        boundary_rejected_points=tuple((float(x), float(y)) for x, y in rejected_points),
        boundary_filter_stats=filter_stats,
        ellipse_fit=ellipse_fit,
        boundary_evidence_source=evidence_source,
    )


def _filter_radial_boundary_points(
    *,
    angles: np.ndarray,
    radii: np.ndarray,
    points: np.ndarray,
    rough_radius: float,
    minimum_points: int,
    settings: dict[str, Any],
    neighbor_ellipses: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    """Split radial boundary evidence into accepted and rejected points.

    The radial sampler can occasionally lock onto a highlight, cushion edge,
    shadow, or unrelated texture for one or two angles. Those local spikes
    should remain visible in review, but they should not pull the observed
    cream ellipse away from the real ball boundary.
    """
    raw_count = int(len(points))
    if (
        not bool(settings.get("boundary_outlier_filter_enabled", True))
        or raw_count < max(8, minimum_points)
    ):
        return _boundary_filter_payload(
            points=points,
            rejected=np.empty((0, 2), dtype=np.float64),
            status="disabled",
            raw_count=raw_count,
            reasons=["disabled_or_insufficient_points"],
        )

    inlier_mask = np.ones(raw_count, dtype=bool)
    reasons: list[str] = []

    endpoint_outliers = _segment_endpoint_outliers(
        angles=np.asarray(angles, dtype=np.float64).reshape(-1),
        trim_points=int(settings.get("boundary_outlier_segment_endpoint_trim_points", 1)),
        gap_factor=float(settings.get("boundary_outlier_segment_gap_factor", 2.25)),
    )
    if np.count_nonzero(endpoint_outliers):
        reasons.append("angular_segment_endpoint")
    inlier_mask &= ~endpoint_outliers

    window = int(settings.get("boundary_outlier_window_points", 9))
    if window < 5:
        window = 5
    if window % 2 == 0:
        window += 1
    half_window = window // 2
    radius_threshold = max(
        float(settings.get("boundary_outlier_min_radius_px", 3.0)),
        float(rough_radius) * float(settings.get("boundary_outlier_radius_factor", 0.085)),
    )
    smooth_outliers = np.zeros(raw_count, dtype=bool)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    for index in range(raw_count):
        neighbor_indices = (np.arange(index - half_window, index + half_window + 1) % raw_count)
        local = radii[neighbor_indices]
        median = float(np.median(local))
        mad = float(np.median(np.abs(local - median)))
        threshold = max(radius_threshold, 4.0 * 1.4826 * mad)
        if abs(float(radii[index]) - median) > threshold:
            smooth_outliers[index] = True
    if np.count_nonzero(smooth_outliers):
        reasons.append("local_radius_spike")
    inlier_mask &= ~smooth_outliers

    neighbor_outliers = _neighbor_ellipse_outliers(
        points=points,
        neighbor_ellipses=neighbor_ellipses,
        settings=settings,
    )
    candidate_without_neighbors = inlier_mask & ~neighbor_outliers
    neighbor_outliers_applied = np.zeros(raw_count, dtype=bool)
    if int(np.count_nonzero(candidate_without_neighbors)) >= minimum_points:
        neighbor_outliers_applied = inlier_mask & neighbor_outliers
        if np.count_nonzero(neighbor_outliers_applied):
            reasons.append("neighbor_ellipse_overlap")
        inlier_mask = candidate_without_neighbors

    ellipse_outliers = np.zeros(raw_count, dtype=bool)
    ellipse_passes = int(settings.get("boundary_outlier_ellipse_passes", 2))
    for _ in range(max(0, ellipse_passes)):
        if int(np.count_nonzero(inlier_mask)) < max(5, minimum_points):
            break
        ellipse = fit_ellipse_payload(points[inlier_mask], source="boundary_filter")
        residuals = _ellipse_boundary_residuals_px(points, ellipse)
        if residuals is None:
            break
        current = residuals[inlier_mask]
        median = float(np.median(current))
        mad = float(np.median(np.abs(current - median)))
        threshold = max(
            float(settings.get("boundary_outlier_min_ellipse_residual_px", 3.25)),
            float(rough_radius) * float(settings.get("boundary_outlier_ellipse_radius_factor", 0.08)),
            median + 4.5 * 1.4826 * mad,
        )
        new_outliers = residuals > threshold
        candidate_mask = inlier_mask & ~new_outliers
        if int(np.count_nonzero(candidate_mask)) < minimum_points:
            break
        ellipse_outliers |= inlier_mask & new_outliers
        inlier_mask = candidate_mask
    if np.count_nonzero(ellipse_outliers):
        reasons.append("ellipse_residual_outlier")

    accepted_count = int(np.count_nonzero(inlier_mask))
    if accepted_count < minimum_points:
        return _boundary_filter_payload(
            points=points,
            rejected=np.empty((0, 2), dtype=np.float64),
            status="fallback_unfiltered",
            raw_count=raw_count,
            reasons=["filter_would_drop_too_many_points"],
        )

    rejected = points[~inlier_mask].astype(np.float64)
    accepted = points[inlier_mask].astype(np.float64)
    return _boundary_filter_payload(
        points=accepted,
        rejected=rejected,
        status="filtered" if len(rejected) else "no_outliers",
        raw_count=raw_count,
        reasons=reasons or ["no_outliers"],
        extra={
            "local_radius_rejected_count": int(np.count_nonzero(smooth_outliers)),
            "ellipse_residual_rejected_count": int(np.count_nonzero(ellipse_outliers)),
            "segment_endpoint_rejected_count": int(np.count_nonzero(endpoint_outliers)),
            "neighbor_ellipse_rejected_count": int(np.count_nonzero(neighbor_outliers_applied)),
            "neighbor_ellipse_candidate_count": int(np.count_nonzero(neighbor_outliers)),
            "minimum_points": int(minimum_points),
        },
    )


def _boundary_filter_payload(
    *,
    points: np.ndarray,
    rejected: np.ndarray,
    status: str,
    raw_count: int,
    reasons: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "accepted_points": np.asarray(points, dtype=np.float64).reshape(-1, 2),
        "rejected_points": np.asarray(rejected, dtype=np.float64).reshape(-1, 2),
        "stats": {
            "status": status,
            "method": "radial_radius_hampel_plus_ellipse_residual",
            "raw_count": int(raw_count),
            "accepted_count": int(len(points)),
            "rejected_count": int(len(rejected)),
            "reasons": list(reasons),
            **(extra or {}),
        },
    }


def _segment_endpoint_outliers(
    *,
    angles: np.ndarray,
    trim_points: int,
    gap_factor: float,
) -> np.ndarray:
    count = int(len(angles))
    outliers = np.zeros(count, dtype=bool)
    if count < 12 or trim_points <= 0:
        return outliers
    unwrapped = np.asarray(angles, dtype=np.float64).reshape(-1)
    diffs = np.diff(np.r_[unwrapped, unwrapped[0] + 2.0 * np.pi])
    positive = diffs[diffs > 1e-9]
    if len(positive) == 0:
        return outliers
    nominal_step = float(np.median(positive))
    if nominal_step <= 0.0:
        return outliers
    gap_indices = np.flatnonzero(diffs > nominal_step * max(1.5, float(gap_factor)))
    if len(gap_indices) == 0:
        return outliers
    for gap_index in gap_indices:
        for offset in range(trim_points):
            outliers[(int(gap_index) - offset) % count] = True
            outliers[(int(gap_index) + 1 + offset) % count] = True
    return outliers


def _ellipse_boundary_residuals_px(
    points: np.ndarray,
    ellipse: dict[str, Any] | None,
) -> np.ndarray | None:
    if not ellipse:
        return None
    major = float(ellipse.get("major_axis_px") or 0.0)
    minor = float(ellipse.get("minor_axis_px") or 0.0)
    if major <= 0.0 or minor <= 0.0:
        return None
    center = np.array(
        [float(ellipse["center_x_px"]), float(ellipse["center_y_px"])],
        dtype=np.float64,
    )
    theta = np.deg2rad(float(ellipse.get("angle_deg") or 0.0))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    delta = np.asarray(points, dtype=np.float64).reshape(-1, 2) - center[None, :]
    x_axis = cos_t * delta[:, 0] + sin_t * delta[:, 1]
    y_axis = -sin_t * delta[:, 0] + cos_t * delta[:, 1]
    radius = np.sqrt((x_axis / (major / 2.0)) ** 2 + (y_axis / (minor / 2.0)) ** 2)
    return np.abs(radius - 1.0) * ((major + minor) / 4.0)


def _neighbor_ellipse_outliers(
    *,
    points: np.ndarray,
    neighbor_ellipses: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    settings: dict[str, Any],
) -> np.ndarray:
    """Return points likely owned by neighboring ball silhouettes.

    In dense clusters, radial sampling can lock onto specular highlight edges
    or internal texture on adjacent balls. Those points are often inside the
    neighboring ball's observed ellipse, so they should stay visible as rejected
    evidence but should not pull the selected ball's cream ellipse.
    """

    count = int(len(points))
    outliers = np.zeros(count, dtype=bool)
    if count == 0 or not neighbor_ellipses:
        return outliers
    if not bool(settings.get("neighbor_ellipse_rejection_enabled", True)):
        return outliers

    axis_scale = float(settings.get("neighbor_ellipse_rejection_axis_scale", 0.92))
    if axis_scale <= 0:
        return outliers
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    for ellipse in neighbor_ellipses:
        if not isinstance(ellipse, dict):
            continue
        center = _ellipse_center(ellipse)
        major = _ellipse_axis(ellipse, "major_axis_px")
        minor = _ellipse_axis(ellipse, "minor_axis_px")
        if center is None or major is None or minor is None:
            continue
        if major <= 2.0 or minor <= 2.0:
            continue
        theta = np.deg2rad(float(ellipse.get("angle_deg") or 0.0))
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        delta = pts - center[None, :]
        x_axis = cos_t * delta[:, 0] + sin_t * delta[:, 1]
        y_axis = -sin_t * delta[:, 0] + cos_t * delta[:, 1]
        a = max((major / 2.0) * axis_scale, 1.0)
        b = max((minor / 2.0) * axis_scale, 1.0)
        normalized = (x_axis / a) ** 2 + (y_axis / b) ** 2
        outliers |= normalized <= 1.0
    return outliers


def _ellipse_center(ellipse: dict[str, Any]) -> np.ndarray | None:
    center = ellipse.get("center_px")
    if center is None and "center_x_px" in ellipse and "center_y_px" in ellipse:
        center = [ellipse["center_x_px"], ellipse["center_y_px"]]
    if center is None:
        return None
    try:
        return np.array([float(center[0]), float(center[1])], dtype=np.float64)
    except (TypeError, ValueError, IndexError):
        return None


def _ellipse_axis(ellipse: dict[str, Any], key: str) -> float | None:
    value = ellipse.get(key)
    if value is None and key == "major_axis_px" and ellipse.get("radius_px") is not None:
        value = float(ellipse["radius_px"]) * 2.0
    if value is None and key == "minor_axis_px" and ellipse.get("radius_px") is not None:
        value = float(ellipse["radius_px"]) * 2.0
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
