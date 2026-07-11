from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

from snookerhelp.qa.ellipse_accuracy import compare_ellipses
from snookerhelp.recognition.evidence_maps import (
    FullTableEvidenceMaps,
    compute_ball_evidence_maps,
    compute_full_table_evidence_maps,
    estimate_global_cloth_reference,
    evidence_map_array,
    evidence_map_uses_outward_drop,
)
from snookerhelp.recognition.evidence_scoring import boundary_view_score
from snookerhelp.recognition.source_refinement import (
    fit_radial_boundary_variant_from_feature,
)


EVIDENCE_MAP_KEYS = (
    "gray_edge",
    "lab_delta_e",
    "chroma_difference",
    "ball_vs_cloth_probability",
    "physical_projection_band",
    "combined_boundary_score",
)

_FLOAT_PARAMETERS = {
    "ball_inner_radius_factor": (0.20, 0.95),
    "probability_offset_factor": (-0.25, 1.25),
    "probability_scale_factor": (0.02, 1.00),
    "green_blue_chroma_weight": (0.0, 1.0),
    "map_boundary_radial_step_px": (0.10, 2.00),
    "map_boundary_inner_radius_factor": (0.10, 1.20),
    "map_boundary_outer_radius_factor": (0.70, 2.50),
    "map_boundary_minimum_strength": (0.0, 0.50),
    "boundary_outlier_segment_gap_factor": (1.10, 8.00),
    "boundary_outlier_radius_factor": (0.01, 0.50),
    "boundary_outlier_ellipse_radius_factor": (0.01, 0.50),
    "neighbor_ellipse_rejection_axis_scale": (0.50, 1.30),
}

_INTEGER_PARAMETERS = {
    "map_boundary_angle_count": (24, 720),
    "map_boundary_minimum_points": (5, 360),
    "boundary_outlier_window_points": (3, 61),
    "boundary_outlier_ellipse_passes": (0, 6),
}

_BOOLEAN_PARAMETERS = {
    "boundary_outlier_filter_enabled",
    "neighbor_ellipse_rejection_enabled",
}


def run_evidence_experiment(
    *,
    source_image: np.ndarray,
    table_state: dict[str, Any],
    ball_id: int,
    evidence_settings: dict[str, Any],
    parameters: dict[str, Any] | None = None,
    ground_truth_ball: dict[str, Any] | None = None,
    global_cloth_model: dict[str, Any] | None = None,
    full_table_evidence_maps: FullTableEvidenceMaps | None = None,
) -> dict[str, Any]:
    """Recompute one ball's map, boundary, ellipse, and diagnostic score."""

    request = dict(parameters or {})
    ball = _ball_by_id(table_state, ball_id)
    effective = _effective_parameters(evidence_settings, request)
    map_key = str(request.get("map_key") or _production_map_key(ball, evidence_settings))
    if map_key not in EVIDENCE_MAP_KEYS:
        raise ValueError(f"Unsupported evidence map: {map_key}")

    settings = dict(evidence_settings)
    settings.update(effective["algorithm"])
    ball_reference = _ball_reference(
        source_image=source_image,
        table_state=table_state,
        selected_ball=ball,
        mode=effective["ball_reference_mode"],
        reference_ball_id=effective.get("reference_ball_id"),
        inner_radius_factor=float(settings.get("ball_inner_radius_factor", 0.55)),
        minimum_value=float(settings.get("minimum_value_for_color_model", 28.0)),
        highlight_limit=float(settings.get("highlight_value_limit", 245.0)),
    )
    if ball_reference is not None:
        settings["ball_reference_lab"] = ball_reference["lab"]
        settings["ball_reference_source"] = ball_reference["source"]

    cloth_balls = [
        {
            "source_refined_center_px": item.get("source_px"),
            "source_radius_px": item.get("radius_px"),
        }
        for item in table_state.get("balls", [])
    ]
    global_cloth = global_cloth_model or estimate_global_cloth_reference(
        source_image=source_image,
        table_corners_px=table_state.get("table_corners_px"),
        balls=cloth_balls,
        settings=settings,
    )
    settings["global_cloth_model"] = global_cloth
    full_maps = full_table_evidence_maps or compute_full_table_evidence_maps(
        source_image=source_image,
        table_corners_px=table_state.get("table_corners_px"),
        settings=settings,
    )
    if full_maps is not None:
        settings["_full_table_evidence_maps"] = full_maps

    physical = ((ball.get("evidence") or {}).get("physical_model") or {})
    sphere_projection = {
        "contour_points_px": physical.get("projected_outline_px") or [],
        "projected_center_px": physical.get("projected_center_px"),
        "status": physical.get("status"),
    }
    maps = compute_ball_evidence_maps(
        source_image=source_image,
        center_px=ball.get("source_px"),
        radius_px=ball.get("radius_px"),
        label=str(ball.get("label") or "unknown"),
        sphere_projection=sphere_projection,
        settings=settings,
    )
    if maps is None:
        raise ValueError("Evidence maps could not be computed for this ball")

    feature = evidence_map_array(maps, map_key)
    diagnostics = ((ball.get("evidence") or {}).get("diagnostics") or {})
    neighbors = diagnostics.get("neighbor_ellipses") or []
    variant = fit_radial_boundary_variant_from_feature(
        feature=feature,
        roi=maps.roi,
        center_px=ball.get("source_px"),
        radius_px=ball.get("radius_px"),
        evidence_source=f"experiment_{map_key}",
        settings=settings,
        use_outward_drop=evidence_map_uses_outward_drop(map_key),
        neighbor_ellipses=neighbors,
    ) or {
        "status": "unavailable",
        "points_px": [],
        "rejected_points_px": [],
        "ellipse_fit": None,
        "filter": {},
    }
    score = boundary_view_score(
        points_px=variant.get("points_px") or [],
        rejected_points_px=variant.get("rejected_points_px") or [],
        ellipse_fit=variant.get("ellipse_fit"),
        sphere_projection=sphere_projection,
        radius_px=ball.get("radius_px"),
    )
    baseline = _baseline_variant(ball, map_key)
    baseline_score = (baseline or {}).get("view_score") or boundary_view_score(
        points_px=(baseline or {}).get("points_px") or [],
        rejected_points_px=(baseline or {}).get("rejected_points_px") or [],
        ellipse_fit=(baseline or {}).get("ellipse_fit"),
        sphere_projection=sphere_projection,
        radius_px=ball.get("radius_px"),
    )
    manual_ellipse = (ground_truth_ball or {}).get("ellipse_px")
    annotation_comparison = compare_ellipses(
        variant.get("ellipse_fit"),
        manual_ellipse,
    )
    baseline_annotation_comparison = compare_ellipses(
        (baseline or {}).get("ellipse_fit"),
        manual_ellipse,
    )

    return {
        "status": "computed",
        "ball_id": int(ball_id),
        "label": ball.get("label"),
        "map_key": map_key,
        "roi_px": [int(value) for value in maps.roi],
        "map_png_data_uri": _map_data_uri(feature),
        "map_summary": maps.summary,
        "effective_parameters": {
            **effective,
            "global_cloth_model": global_cloth,
            "ball_reference": ball_reference,
        },
        "experiment": {
            **variant,
            "view_score": score,
            "annotation_comparison": annotation_comparison,
        },
        "baseline": {
            "variant": baseline,
            "view_score": baseline_score,
            "annotation_comparison": baseline_annotation_comparison,
        },
        "comparison": _variant_comparison(variant, baseline),
        "ground_truth_available": bool(manual_ellipse),
        "note": (
            "Experiment output is transient and does not overwrite report.json "
            "or the production final estimate."
        ),
    }


def _effective_parameters(
    baseline: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    algorithm: dict[str, Any] = {}
    for key, (minimum, maximum) in _FLOAT_PARAMETERS.items():
        if key not in request:
            continue
        algorithm[key] = float(np.clip(float(request[key]), minimum, maximum))
    for key, (minimum, maximum) in _INTEGER_PARAMETERS.items():
        if key not in request:
            continue
        algorithm[key] = int(np.clip(int(request[key]), minimum, maximum))
    for key in _BOOLEAN_PARAMETERS:
        if key in request:
            algorithm[key] = bool(request[key])

    mode = str(request.get("ball_reference_mode") or "selected_ball")
    if mode not in {"selected_ball", "median_red_balls", "reference_ball"}:
        raise ValueError(f"Unsupported ball reference mode: {mode}")
    reference_ball_id = request.get("reference_ball_id")
    return {
        "ball_reference_mode": mode,
        "reference_ball_id": (
            int(reference_ball_id) if reference_ball_id is not None else None
        ),
        "algorithm": {
            key: value
            for key, value in {
                **{
                    name: baseline.get(name)
                    for name in (
                        *_FLOAT_PARAMETERS.keys(),
                        *_INTEGER_PARAMETERS.keys(),
                        *_BOOLEAN_PARAMETERS,
                    )
                    if name in baseline
                },
                **algorithm,
            }.items()
        },
    }


def _ball_reference(
    *,
    source_image: np.ndarray,
    table_state: dict[str, Any],
    selected_ball: dict[str, Any],
    mode: str,
    reference_ball_id: int | None,
    inner_radius_factor: float,
    minimum_value: float,
    highlight_limit: float,
) -> dict[str, Any] | None:
    if mode == "selected_ball":
        return None
    if mode == "median_red_balls":
        candidates = [
            item for item in table_state.get("balls", [])
            if str(item.get("label") or "").lower() == "red"
        ]
        source = "median_inner_pixels_of_all_detected_red_balls"
    else:
        candidates = [
            item for item in table_state.get("balls", [])
            if int(item.get("ball_id", -1)) == int(reference_ball_id or -1)
        ]
        if not candidates:
            raise ValueError(f"Unknown reference ball: {reference_ball_id}")
        source = f"inner_pixels_of_reference_ball_{int(reference_ball_id)}"
    lab, sample_count = _sample_ball_lab(
        source_image,
        candidates,
        inner_radius_factor=inner_radius_factor,
        minimum_value=minimum_value,
        highlight_limit=highlight_limit,
    )
    if lab is None:
        raise ValueError(f"Could not sample color reference for mode {mode}")
    return {
        "mode": mode,
        "source": source,
        "lab": [round(float(value), 4) for value in lab],
        "sample_count": int(sample_count),
        "selected_ball_id": int(selected_ball.get("ball_id", 0)),
    }


def _sample_ball_lab(
    source_image: np.ndarray,
    balls: list[dict[str, Any]],
    *,
    inner_radius_factor: float,
    minimum_value: float,
    highlight_limit: float,
) -> tuple[np.ndarray | None, int]:
    lab_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    samples: list[np.ndarray] = []
    for ball in balls:
        center = ball.get("source_px")
        radius = ball.get("radius_px")
        if not center or radius is None:
            continue
        x, y = float(center[0]), float(center[1])
        r = max(2.0, float(radius) * inner_radius_factor)
        x0 = max(0, int(np.floor(x - r)))
        y0 = max(0, int(np.floor(y - r)))
        x1 = min(source_image.shape[1], int(np.ceil(x + r + 1)))
        y1 = min(source_image.shape[0], int(np.ceil(y + r + 1)))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        values = hsv[y0:y1, x0:x1, 2]
        mask &= (values > minimum_value) & (values < highlight_limit)
        if np.any(mask):
            samples.append(lab_image[y0:y1, x0:x1][mask])
    if not samples:
        return None, 0
    combined = np.concatenate(samples, axis=0)
    return np.median(combined, axis=0).astype(np.float32), int(len(combined))


def _ball_by_id(table_state: dict[str, Any], ball_id: int) -> dict[str, Any]:
    for ball in table_state.get("balls", []):
        if int(ball.get("ball_id", -1)) == int(ball_id):
            return ball
    raise ValueError(f"Unknown ball id: {ball_id}")


def _production_map_key(
    ball: dict[str, Any],
    evidence_settings: dict[str, Any],
) -> str:
    policy = evidence_settings.get("final_position_policy") or {}
    label = str(ball.get("label") or "").lower()
    return str(
        (policy.get("label_overrides") or {}).get(label)
        or policy.get("default_map")
        or "ball_vs_cloth_probability"
    )


def _baseline_variant(ball: dict[str, Any], map_key: str) -> dict[str, Any] | None:
    diagnostics = ((ball.get("evidence") or {}).get("diagnostics") or {})
    variants = ((diagnostics.get("evidence_maps") or {}).get("boundary_variants") or {})
    value = variants.get(map_key)
    return dict(value) if isinstance(value, dict) else None


def _variant_comparison(
    experiment: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    experiment_ellipse = experiment.get("ellipse_fit") or {}
    baseline_ellipse = (baseline or {}).get("ellipse_fit") or {}
    if not experiment_ellipse.get("center_px") or not baseline_ellipse.get("center_px"):
        return {"status": "unavailable", "reason": "both fitted ellipses are required"}
    exp_center = np.asarray(experiment_ellipse["center_px"], dtype=np.float64)
    base_center = np.asarray(baseline_ellipse["center_px"], dtype=np.float64)
    return {
        "status": "computed",
        "center_shift_px": round(float(np.linalg.norm(exp_center - base_center)), 4),
        "major_axis_delta_px": round(
            float(experiment_ellipse.get("major_axis_px", 0.0))
            - float(baseline_ellipse.get("major_axis_px", 0.0)),
            4,
        ),
        "minor_axis_delta_px": round(
            float(experiment_ellipse.get("minor_axis_px", 0.0))
            - float(baseline_ellipse.get("minor_axis_px", 0.0)),
            4,
        ),
    }


def _map_data_uri(values: np.ndarray) -> str:
    image = np.round(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode experiment map")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


__all__ = ["EVIDENCE_MAP_KEYS", "run_evidence_experiment"]
