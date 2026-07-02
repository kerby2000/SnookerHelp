from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from snookerhelp.recognition.source_refinement import source_roi_bounds


@dataclass(frozen=True)
class BallEvidenceMaps:
    """Local per-ball evidence maps used to explain and score weak boundaries.

    Arrays are ROI-local float32 maps normalized to 0..1. The JSON report should
    store `summary`, not these arrays.
    """

    roi: tuple[int, int, int, int]
    label: str
    gray_edge: np.ndarray
    lab_delta_e: np.ndarray
    chroma_difference: np.ndarray
    ball_probability: np.ndarray
    physical_band_score: np.ndarray
    combined_boundary_score: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    summary: dict[str, Any]

    @property
    def origin(self) -> tuple[int, int]:
        return int(self.roi[0]), int(self.roi[1])


def compute_ball_evidence_maps(
    *,
    source_image: np.ndarray,
    center_px: tuple[float, float] | list[float] | None,
    radius_px: float | None,
    label: str,
    sphere_projection: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
) -> BallEvidenceMaps | None:
    """Build local contrast/edge maps for one ball crop.

    The maps are diagnostic evidence and a weak search prior for physical
    optimization. They deliberately do not replace the main detector by
    themselves.
    """

    if source_image.ndim != 3 or center_px is None or radius_px is None:
        return None
    radius = float(radius_px)
    if radius <= 3.0:
        return None
    center = (float(center_px[0]), float(center_px[1]))
    cfg = settings or {}
    roi = _map_roi(source_image.shape, center, radius, sphere_projection, cfg)
    x0, y0, x1, y1 = roi
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None

    crop = source_image[y0:y1, x0:x1]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    local_center = np.array([center[0] - x0, center[1] - y0], dtype=np.float32)
    yy, xx = np.mgrid[0 : crop.shape[0], 0 : crop.shape[1]]
    distance = np.hypot(xx.astype(np.float32) - local_center[0], yy.astype(np.float32) - local_center[1])

    valid_color = (hsv[:, :, 2] > float(cfg.get("minimum_value_for_color_model", 28.0))) & (
        hsv[:, :, 2] < float(cfg.get("highlight_value_limit", 245.0))
    )
    inner_mask = (distance <= radius * float(cfg.get("ball_inner_radius_factor", 0.55))) & valid_color
    cloth_mask = (
        (distance >= radius * float(cfg.get("cloth_inner_radius_factor", 1.25)))
        & (distance <= radius * float(cfg.get("cloth_outer_radius_factor", 1.95)))
        & valid_color
    )
    if int(np.count_nonzero(cloth_mask)) < 20:
        cloth_mask = (distance >= radius * 1.15) & (distance <= radius * 2.1)
    if int(np.count_nonzero(inner_mask)) < 20:
        inner_mask = distance <= radius * 0.45

    local_cloth_lab = _median_lab(lab, cloth_mask)
    ball_lab = _median_lab(lab, inner_mask)
    global_cloth_model = _global_cloth_model_from_settings(cfg)
    cloth_reference_mode = str(cfg.get("cloth_reference_mode", "global")).lower()
    use_global_cloth = (
        cloth_reference_mode in {"global", "global_table", "global_table_cloth"}
        and global_cloth_model is not None
    )
    active_cloth_lab = (
        np.asarray(global_cloth_model["cloth_lab"], dtype=np.float32)
        if use_global_cloth
        else local_cloth_lab
    )
    active_cloth_sample_count = (
        int(global_cloth_model.get("sample_count", 0))
        if use_global_cloth and global_cloth_model is not None
        else int(np.count_nonzero(cloth_mask))
    )

    lab_delta = np.linalg.norm(lab - active_cloth_lab[None, None, :], axis=2)
    chroma_delta = np.linalg.norm(lab[:, :, 1:3] - active_cloth_lab[None, None, 1:3], axis=2)

    gray_edge = _normalized_sobel(gray)
    normalization_mask = valid_color if use_global_cloth else (cloth_mask | inner_mask)
    lab_delta_norm = _robust_normalize(lab_delta, normalization_mask)
    chroma_norm = _robust_normalize(chroma_delta, normalization_mask)
    probability = _ball_probability_map(lab, active_cloth_lab, ball_lab, lab_delta_norm, chroma_norm, label)
    physical_band = _physical_band_score(crop.shape[:2], roi, sphere_projection, cfg)
    weights = _class_weights(label)
    combined = np.clip(
        weights["edge"] * gray_edge
        + weights["lab"] * lab_delta_norm
        + weights["chroma"] * chroma_norm
        + weights["probability"] * probability
        + weights["physical_band"] * physical_band,
        0.0,
        1.0,
    ).astype(np.float32)
    gradient_source = cv2.GaussianBlur(probability + 0.45 * gray_edge, (3, 3), 0)
    gradient_x = cv2.Sobel(gradient_source, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gradient_source, cv2.CV_32F, 0, 1, ksize=3)

    local_separation_lab = float(np.linalg.norm(ball_lab - local_cloth_lab))
    local_separation_chroma = float(np.linalg.norm(ball_lab[1:3] - local_cloth_lab[1:3]))
    active_separation_lab = float(np.linalg.norm(ball_lab - active_cloth_lab))
    active_separation_chroma = float(np.linalg.norm(ball_lab[1:3] - active_cloth_lab[1:3]))
    active_color_model = {
        "cloth_reference_mode": "global_table_cloth" if use_global_cloth else "local_annulus",
        "cloth_lab": _round_array(active_cloth_lab),
        "ball_lab": _round_array(ball_lab),
        "separation_lab": round(active_separation_lab, 4),
        "separation_chroma": round(active_separation_chroma, 4),
        "cloth_sample_count": active_cloth_sample_count,
        "ball_sample_count": int(np.count_nonzero(inner_mask)),
        "low_contrast": bool(active_separation_lab < 18.0 or active_separation_chroma < 9.0),
    }
    local_color_model = {
        "cloth_lab": _round_array(local_cloth_lab),
        "ball_lab": _round_array(ball_lab),
        "separation_lab": round(local_separation_lab, 4),
        "separation_chroma": round(local_separation_chroma, 4),
        "cloth_sample_count": int(np.count_nonzero(cloth_mask)),
        "ball_sample_count": int(np.count_nonzero(inner_mask)),
        "low_contrast": bool(local_separation_lab < 18.0 or local_separation_chroma < 9.0),
    }
    summary = {
        "status": "computed",
        "roi_px": [int(value) for value in roi],
        "label": str(label),
        "active_color_model": active_color_model,
        "local_color_model": local_color_model,
        "global_cloth_model": global_cloth_model,
        "color_model_parameters": {
            "cloth_reference_mode": str(cfg.get("cloth_reference_mode", "global")),
            "minimum_value_for_color_model": float(cfg.get("minimum_value_for_color_model", 28.0)),
            "highlight_value_limit": float(cfg.get("highlight_value_limit", 245.0)),
            "ball_inner_radius_factor": float(cfg.get("ball_inner_radius_factor", 0.55)),
            "cloth_inner_radius_factor": float(cfg.get("cloth_inner_radius_factor", 1.25)),
            "cloth_outer_radius_factor": float(cfg.get("cloth_outer_radius_factor", 1.95)),
            "global_cloth_exclusion_radius_factor": float(
                cfg.get("global_cloth_exclusion_radius_factor", 2.2),
            ),
            "global_cloth_erode_px": int(cfg.get("global_cloth_erode_px", 24)),
        },
        "maps": {
            "gray_edge": _map_stats(gray_edge),
            "lab_delta_e": _map_stats(lab_delta_norm),
            "chroma_difference": _map_stats(chroma_norm),
            "ball_vs_cloth_probability": _map_stats(probability),
            "physical_projection_band": _map_stats(physical_band),
            "combined_boundary_score": _map_stats(combined),
        },
        "weights": {key: round(float(value), 4) for key, value in weights.items()},
        "explanation": (
            "Likely ball color is sampled from the inner disk with highlights "
            "and deep shadows ignored. Cloth is taken from the configured "
            "reference mode: global table cloth when available, otherwise the "
            "local annulus. Green/blue balls weight Lab/chroma contrast more "
            "heavily than raw grayscale edges."
        ),
    }
    return BallEvidenceMaps(
        roi=roi,
        label=str(label),
        gray_edge=gray_edge,
        lab_delta_e=lab_delta_norm,
        chroma_difference=chroma_norm,
        ball_probability=probability.astype(np.float32),
        physical_band_score=physical_band,
        combined_boundary_score=combined,
        gradient_x=gradient_x.astype(np.float32),
        gradient_y=gradient_y.astype(np.float32),
        summary=summary,
    )


def estimate_global_cloth_reference(
    *,
    source_image: np.ndarray,
    table_corners_px: list[Any] | tuple[Any, ...] | np.ndarray | None = None,
    balls: list[Any] | tuple[Any, ...] | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate one robust cloth Lab reference for the whole source image.

    This is deliberately simple and auditable: start from the table polygon,
    erode away cushion edges, exclude detected ball neighborhoods, reject very
    dark/highlight pixels, then take the median Lab color.
    """

    if source_image.ndim != 3 or source_image.size == 0:
        return {"status": "unavailable", "reason": "source image is not BGR"}
    cfg = settings or {}
    height, width = int(source_image.shape[0]), int(source_image.shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    corners_input = [] if table_corners_px is None else table_corners_px
    corners = np.asarray(corners_input, dtype=np.float32).reshape(-1, 2)
    if corners.shape[0] >= 3:
        cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 255)
        method = "table_polygon"
    else:
        mask[:, :] = 255
        method = "full_image_fallback"

    erode_px = int(cfg.get("global_cloth_erode_px", 24))
    if erode_px > 0:
        kernel_size = max(3, erode_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.erode(mask, kernel, iterations=1)

    exclusion_factor = float(cfg.get("global_cloth_exclusion_radius_factor", 2.2))
    excluded_balls = 0
    for ball in balls or []:
        center, radius = _ball_center_radius(ball)
        if center is None or radius is None or radius <= 1.0:
            continue
        cv2.circle(
            mask,
            (int(round(center[0])), int(round(center[1]))),
            max(3, int(round(radius * exclusion_factor))),
            0,
            thickness=-1,
        )
        excluded_balls += 1

    hsv = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV)
    valid_color = (
        (hsv[:, :, 2].astype(np.float32) > float(cfg.get("minimum_value_for_color_model", 28.0)))
        & (hsv[:, :, 2].astype(np.float32) < float(cfg.get("highlight_value_limit", 245.0)))
    )
    sample_mask = (mask > 0) & valid_color
    sample_count = int(np.count_nonzero(sample_mask))
    minimum_samples = int(cfg.get("global_cloth_minimum_samples", 500))
    if sample_count < minimum_samples:
        fallback_mask = valid_color
        fallback_count = int(np.count_nonzero(fallback_mask))
        if fallback_count < minimum_samples:
            return {
                "status": "unavailable",
                "reason": "not enough valid cloth-color samples",
                "sample_count": sample_count,
                "fallback_sample_count": fallback_count,
                "method": method,
                "excluded_ball_count": excluded_balls,
            }
        sample_mask = fallback_mask
        sample_count = fallback_count
        method = f"{method}_valid_color_fallback"

    lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    sample = lab[sample_mask]
    cloth_lab = np.median(sample, axis=0).astype(np.float32)
    p10 = np.percentile(sample, 10, axis=0).astype(np.float32)
    p90 = np.percentile(sample, 90, axis=0).astype(np.float32)
    return {
        "status": "computed",
        "method": method,
        "cloth_lab": _round_array(cloth_lab),
        "lab_p10": _round_array(p10),
        "lab_p90": _round_array(p90),
        "sample_count": sample_count,
        "excluded_ball_count": excluded_balls,
        "table_mask_erode_px": erode_px,
        "ball_exclusion_radius_factor": round(exclusion_factor, 4),
    }


def sample_map_at_points(
    evidence_maps: BallEvidenceMaps | None,
    points_px: list[list[float]] | tuple[Any, ...] | np.ndarray,
    map_name: str = "combined_boundary_score",
) -> np.ndarray:
    if evidence_maps is None:
        return np.empty(0, dtype=np.float32)
    values = getattr(evidence_maps, map_name)
    origin_x, origin_y = evidence_maps.origin
    points = np.asarray(points_px or [], dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return np.empty(0, dtype=np.float32)
    xs = np.round(points[:, 0] - float(origin_x)).astype(np.int32)
    ys = np.round(points[:, 1] - float(origin_y)).astype(np.int32)
    valid = (xs >= 0) & (ys >= 0) & (xs < values.shape[1]) & (ys < values.shape[0])
    if not np.any(valid):
        return np.empty(0, dtype=np.float32)
    return values[ys[valid], xs[valid]].astype(np.float32)


def _global_cloth_model_from_settings(cfg: dict[str, Any]) -> dict[str, Any] | None:
    model = cfg.get("global_cloth_model")
    if not isinstance(model, dict) or model.get("status") != "computed":
        return None
    values = model.get("cloth_lab")
    if values is None:
        return None
    try:
        lab = np.asarray(values, dtype=np.float32).reshape(3)
    except (TypeError, ValueError):
        return None
    return {**model, "cloth_lab": _round_array(lab)}


def _ball_center_radius(ball: Any) -> tuple[tuple[float, float] | None, float | None]:
    def get(name: str) -> Any:
        if isinstance(ball, dict):
            return ball.get(name)
        return getattr(ball, name, None)

    center = (
        get("source_final_center_px")
        or get("source_refined_center_px")
        or get("source_initial_refined_center_px")
        or get("source_rough_center_px")
    )
    if center is None:
        x = get("source_x_px")
        y = get("source_y_px")
        if x is not None and y is not None:
            center = [x, y]
    if center is None:
        x = get("source_rough_x_px")
        y = get("source_rough_y_px")
        if x is not None and y is not None:
            center = [x, y]
    radius = get("source_radius_px") or get("radius_px")
    try:
        if center is None or len(center) < 2:
            return None, None
        return (float(center[0]), float(center[1])), float(radius)
    except (TypeError, ValueError, IndexError):
        return None, None


def _map_roi(
    image_shape: tuple[int, int] | tuple[int, int, int],
    center: tuple[float, float],
    radius: float,
    sphere_projection: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> tuple[int, int, int, int]:
    base = source_roi_bounds(
        image_shape,
        center,
        radius,
        margin_factor=float(cfg.get("evidence_map_roi_radius_factor", 2.55)),
        minimum_half_size_px=int(cfg.get("evidence_map_minimum_half_size_px", 96)),
    )
    contour = np.asarray((sphere_projection or {}).get("contour_points_px") or [], dtype=np.float32).reshape(-1, 2)
    if len(contour) == 0:
        return base
    height, width = int(image_shape[0]), int(image_shape[1])
    pad = max(12, int(round(radius * 0.45)))
    x0 = max(0, min(base[0], int(np.floor(float(np.min(contour[:, 0])))) - pad))
    y0 = max(0, min(base[1], int(np.floor(float(np.min(contour[:, 1])))) - pad))
    x1 = min(width, max(base[2], int(np.ceil(float(np.max(contour[:, 0])))) + pad))
    y1 = min(height, max(base[3], int(np.ceil(float(np.max(contour[:, 1])))) + pad))
    return x0, y0, x1, y1


def _median_lab(lab: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = lab[mask]
    if pixels.size == 0:
        pixels = lab.reshape(-1, 3)
    return np.median(pixels, axis=0).astype(np.float32)


def _normalized_sobel(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    return _robust_normalize(magnitude, np.ones_like(gray, dtype=bool))


def _robust_normalize(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    finite = np.isfinite(values)
    if mask is not None:
        finite &= mask
    sample = values[finite]
    if sample.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo = float(np.percentile(sample, 5))
    hi = float(np.percentile(sample, 98))
    if hi <= lo + 1e-6:
        hi = float(np.max(sample))
    if hi <= lo + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _ball_probability_map(
    lab: np.ndarray,
    cloth_lab: np.ndarray,
    ball_lab: np.ndarray,
    lab_delta_norm: np.ndarray,
    chroma_norm: np.ndarray,
    label: str,
) -> np.ndarray:
    direction = (ball_lab - cloth_lab).astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 2.0:
        return np.maximum(lab_delta_norm, chroma_norm).astype(np.float32)
    projection = np.tensordot(lab - cloth_lab[None, None, :], direction / norm, axes=([2], [0]))
    ball_projection = float(np.dot(ball_lab - cloth_lab, direction / norm))
    scale = max(2.0, abs(ball_projection) * 0.20)
    probability = 1.0 / (1.0 + np.exp(-(projection - 0.20 * ball_projection) / scale))
    if label.lower() in {"green", "blue"}:
        probability = np.maximum(probability, 0.70 * chroma_norm + 0.30 * lab_delta_norm)
    return np.clip(probability, 0.0, 1.0).astype(np.float32)


def _physical_band_score(
    crop_shape: tuple[int, int],
    roi: tuple[int, int, int, int],
    sphere_projection: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> np.ndarray:
    contour = np.asarray((sphere_projection or {}).get("contour_points_px") or [], dtype=np.float32).reshape(-1, 2)
    height, width = int(crop_shape[0]), int(crop_shape[1])
    if len(contour) < 3:
        return np.zeros((height, width), dtype=np.float32)
    origin = np.array([roi[0], roi[1]], dtype=np.float32)
    local = np.round(contour - origin[None, :]).astype(np.int32).reshape(-1, 1, 2)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.polylines(mask, [local], isClosed=True, color=255, thickness=max(1, int(cfg.get("physical_band_line_thickness_px", 1))))
    distance = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    sigma = float(cfg.get("physical_band_sigma_px", 5.5))
    return np.exp(-(distance * distance) / (2.0 * sigma * sigma)).astype(np.float32)


def _class_weights(label: str) -> dict[str, float]:
    if label.lower() in {"green", "blue"}:
        return {
            "edge": 0.16,
            "lab": 0.22,
            "chroma": 0.24,
            "probability": 0.28,
            "physical_band": 0.10,
        }
    return {
        "edge": 0.34,
        "lab": 0.18,
        "chroma": 0.14,
        "probability": 0.24,
        "physical_band": 0.10,
    }


def _map_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": round(float(np.mean(finite)), 4),
        "p95": round(float(np.percentile(finite, 95)), 4),
        "max": round(float(np.max(finite)), 4),
    }


def _round_array(values: np.ndarray) -> list[float]:
    return [round(float(value), 4) for value in np.asarray(values).reshape(-1)]
