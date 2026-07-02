from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from snookerhelp.recognition.evidence_maps import (
    compute_ball_evidence_maps,
    estimate_global_cloth_reference,
)
from snookerhelp.recognition.image_model import fit_ellipse_payload
from snookerhelp.recognition.source_refinement import (
    fit_radial_boundary_variant_from_feature,
)
from snookerhelp.recognition.sphere_projection import (
    score_observed_points_against_silhouette,
)
from snookerhelp.recognition.confidence import (
    combined_confidence,
    physics_c_only_score,
    physics_first_score,
)
from snookerhelp.qa.report_metrics import nearest_cushion_info, rough_to_refined_shift_px
from snookerhelp.qa.validation import table_dimensions_from_state


REVIEW_ISSUE_TAGS = [
    "bad_center",
    "bad_radius",
    "wrong_label",
    "elongated",
    "model_disagreement",
    "circle_ellipse_disagreement",
    "mask_centroid_disagreement",
    "sphere_projection_mismatch",
    "low_trust_position",
    "no_points",
    "fallback_suspicious",
    "weak_radial_fit",
    "near_cushion",
    "near_pocket",
    "touching_pair_problem",
    "missed_ball_nearby",
    "duplicate_detection",
    "cushion_line_wrong",
    "other",
]


def build_review_evidence(
    state: dict[str, Any],
    source_image: np.ndarray,
    warped_image: np.ndarray,
    output_directory: str | Path,
    evidence_map_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write clean review assets and return JSON overlay/editing evidence.

    This deliberately stores geometry as data. The browser review UI draws
    circles, centers, boundary points, and cushion lines dynamically.
    """
    output = Path(output_directory)
    crops_dir = output / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output / "source.jpg"), source_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(output / "warped.jpg"), warped_image, [cv2.IMWRITE_JPEG_QUALITY, 90])

    map_settings = dict(evidence_map_settings or {})
    global_cloth_model = (state.get("detection") or {}).get("global_cloth_model")
    if not isinstance(global_cloth_model, dict) or global_cloth_model.get("status") != "computed":
        global_cloth_model = estimate_global_cloth_reference(
            source_image=source_image,
            table_corners_px=(state.get("table") or {}).get("corner_points_px"),
            balls=state.get("balls", []),
            settings=map_settings,
        )
    map_settings["global_cloth_model"] = global_cloth_model

    ball_evidence = []
    for ball in sorted(state.get("balls", []), key=lambda item: int(item["id"])):
        bounds = _review_crop_bounds(source_image.shape, ball)
        x0, y0, x1, y1 = bounds
        crop = source_image[y0:y1, x0:x1]
        crop_path = f"crops/ball_{int(ball['id']):03d}.jpg"
        if crop.size:
            cv2.imwrite(
                str(output / crop_path),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, 94],
            )
        evidence_map_assets, evidence_map_boundary_variants = _write_evidence_map_assets(
            source_image=source_image,
            ball=ball,
            bounds=bounds,
            output_directory=output,
            settings=map_settings,
            neighbor_ellipses=_neighbor_source_ellipses_for_ball(
                state.get("balls", []),
                ball,
                map_settings,
            ),
        )

        cushion = nearest_cushion_info(state, ball)
        cushion["line_source_px"] = _cushion_line_source_px(
            state,
            str(cushion["name"]),
        )
        evidence_agreement = _boundary_mask_agreement(ball)
        disagreement = _model_disagreement(ball)
        warnings = _warnings_for_ball(state, ball, cushion, disagreement)
        uncertainty = _position_uncertainty(
            ball,
            warnings,
            disagreement,
            evidence_agreement,
        )
        if uncertainty.get("confidence") == "low" and "low_trust_position" not in warnings:
            warnings.append("low_trust_position")
        consensus_ellipse = _consensus_ellipse_fit_payload(
            ball,
            evidence_agreement,
        )
        model_decision = _model_decision(ball, warnings, uncertainty, disagreement)
        legacy_review_confidence = _review_confidence(
            ball,
            warnings,
            evidence_agreement,
        )
        physics_first_decision = physics_first_score(
            ball={**ball, "review_confidence": legacy_review_confidence},
            evidence_agreement=evidence_agreement,
            consensus_ellipse=consensus_ellipse,
            current_decision=model_decision,
            warnings=warnings,
        )
        physics_c_only_decision = physics_c_only_score(
            ball={**ball, "review_confidence": legacy_review_confidence},
            candidate_c_ellipse=_candidate_c_ellipse_fit_payload(ball),
            current_decision=model_decision,
            warnings=warnings,
        )
        review_confidence = combined_confidence(
            legacy_review_confidence,
            physics_first_decision,
            physics_c_only_decision,
        )
        ball_evidence.append(
            {
                "id": int(ball["id"]),
                "label": ball.get("color_label", ball.get("class")),
                "source_crop_path": crop_path,
                "source_crop_bounds_px": [int(value) for value in bounds],
                "source_center_px": ball.get("source_final_center_px")
                or ball.get("source_refined_center_px"),
                "source_initial_refined_center_px": ball.get(
                    "source_initial_refined_center_px",
                ),
                "final_image_evidence": ball.get("source_final_center_policy", {}),
                "rough_center_px": ball.get("source_rough_center_px"),
                "warped_center_px": ball.get("warped_center_px"),
                "table_xy_mm": ball.get("table_xy_mm"),
                "source_refined_table_xy_mm": ball.get("source_refined_table_xy_mm"),
                "source_refined_table_xy_by_z_mm": ball.get(
                    "source_refined_table_xy_by_z_mm",
                    {},
                ),
                "circle_fit": _circle_fit_payload(ball),
                "ellipse_fit": _best_ellipse_fit_payload(
                    ball,
                    evidence_agreement,
                    consensus_ellipse,
                ),
                "consensus_ellipse_fit": consensus_ellipse,
                "radial_ellipse_fit": _ellipse_fit_payload(
                    ball,
                    "source_radial_ellipse_fit",
                )
                or _ellipse_fit_payload(ball, "source_ellipse_fit"),
                "silhouette_ellipse_fit": _ellipse_fit_payload(
                    ball,
                    "source_silhouette_ellipse_fit",
                ),
                "sphere_projection": ball.get("source_sphere_projection"),
                "physical_optimization": ball.get("source_sphere_optimization"),
                "neighbor_ellipses_px": _neighbor_source_ellipses_for_ball(
                    state.get("balls", []),
                    ball,
                    map_settings,
                ),
                "joint_cluster_optimization": ball.get(
                    "source_joint_cluster_optimization",
                    {},
                ),
                "evidence_maps": _evidence_map_summary_with_assets(
                    ball.get("source_evidence_maps"),
                    evidence_map_assets,
                    evidence_map_boundary_variants,
                ),
                "mask_centroid": _mask_centroid_payload(ball),
                "mask_contour_points_px": ball.get("source_mask_contour_points_px", []),
                "boundary_points_px": ball.get(
                    "source_radial_boundary_points_px",
                    ball.get("source_boundary_points_px", []),
                ),
                "boundary_rejected_points_px": ball.get(
                    "source_radial_boundary_rejected_points_px",
                    ball.get("source_boundary_rejected_points_px", []),
                ),
                "boundary_filter": ball.get(
                    "source_radial_boundary_filter",
                    ball.get("source_boundary_filter", {}),
                ),
                "boundary_view_score": _boundary_view_score(
                    points_px=ball.get(
                        "source_radial_boundary_points_px",
                        ball.get("source_boundary_points_px", []),
                    ),
                    rejected_points_px=ball.get(
                        "source_radial_boundary_rejected_points_px",
                        ball.get("source_boundary_rejected_points_px", []),
                    ),
                    ellipse_fit=ball.get("source_radial_ellipse_fit")
                    or ball.get("source_ellipse_fit"),
                    sphere_projection=ball.get("source_sphere_projection"),
                    radius_px=ball.get("source_radius_px"),
                ),
                "boundary_evidence_source": ball.get(
                    "source_radial_boundary_evidence_source",
                    ball.get("source_boundary_evidence_source"),
                ),
                "final_boundary_points_px": ball.get("source_boundary_points_px", []),
                "final_boundary_rejected_points_px": ball.get(
                    "source_boundary_rejected_points_px",
                    [],
                ),
                "final_boundary_filter": ball.get("source_boundary_filter", {}),
                "final_boundary_evidence_source": ball.get(
                    "source_boundary_evidence_source",
                ),
                "nearest_cushion": cushion,
                "warnings": warnings,
                "model_disagreement": disagreement,
                "evidence_agreement": evidence_agreement,
                "position_uncertainty_mm": uncertainty,
                "model_candidates": _model_candidates(
                    ball,
                    evidence_agreement,
                    consensus_ellipse,
                ),
                "model_decision": model_decision,
                "physics_first_model_decision": physics_first_decision,
                "physics_c_only_model_decision": physics_c_only_decision,
                "rough_to_refined_shift_px": rough_to_refined_shift_px(ball),
                "detection_confidence": ball.get("detection_confidence"),
                "legacy_review_confidence": legacy_review_confidence,
                "physics_first_review_confidence": physics_first_decision.get(
                    "confidence",
                ),
                "physics_c_only_review_confidence": physics_c_only_decision.get(
                    "confidence",
                ),
                "review_confidence": review_confidence,
                "color_confidence": ball.get("color_confidence"),
                "model_used": model_decision["selected_model"],
            }
        )

    return {
        "schema_version": "1.0",
        "source_image_path": "source.jpg",
        "warped_image_path": "warped.jpg",
        "source_size_px": state.get("source_size_px"),
        "table_corner_points_px": (state.get("table") or {}).get("corner_points_px", []),
        "issue_tags": REVIEW_ISSUE_TAGS,
        "balls": ball_evidence,
        "notes": [
            "report.json is immutable algorithm output",
            "review.json stores human decisions and manual corrections",
            "all overlays are drawn by HTML/JavaScript from this geometry",
        ],
    }


def _write_evidence_map_assets(
    *,
    source_image: np.ndarray,
    ball: dict[str, Any],
    bounds: tuple[int, int, int, int],
    output_directory: Path,
    settings: dict[str, Any],
    neighbor_ellipses: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Write diagnostic scalar maps aligned to the review crop coordinate frame."""

    if not bool(settings.get("enabled", True)):
        return [], {}
    center = ball.get("source_refined_center_px") or ball.get("source_rough_center_px")
    maps = compute_ball_evidence_maps(
        source_image=source_image,
        center_px=center,
        radius_px=ball.get("source_radius_px"),
        label=str(ball.get("color_label") or ball.get("class") or "unknown"),
        sphere_projection=ball.get("source_sphere_projection"),
        settings=settings,
    )
    if maps is None:
        return [], {}

    assets_dir = output_directory / "evidence_maps"
    assets_dir.mkdir(parents=True, exist_ok=True)
    ball_id = int(ball["id"])
    map_specs: list[tuple[str, str, str, np.ndarray, bool]] = [
        (
            "gray_edge",
            "Grayscale edge",
            "Raw local edge strength. Brighter means stronger luminance edge.",
            maps.gray_edge,
            False,
        ),
        (
            "lab_delta_e",
            "Lab Delta-E",
            "Color distance from active cloth reference. Brighter means less cloth-like.",
            maps.lab_delta_e,
            True,
        ),
        (
            "chroma_difference",
            "Chroma difference",
            "a*/b* color difference from active cloth reference. Brighter means stronger chroma contrast.",
            maps.chroma_difference,
            True,
        ),
        (
            "ball_vs_cloth_probability",
            "Ball-vs-cloth probability",
            "Learned ball-vs-cloth score using the active cloth reference. Brighter means more ball-like.",
            maps.ball_probability,
            True,
        ),
        (
            "physical_projection_band",
            "Physical projection band",
            "Weak prior around the projected sphere outline. Diagnostic only.",
            maps.physical_band_score,
            False,
        ),
        (
            "combined_boundary_score",
            "Combined boundary score",
            "Weighted combination used by physical scoring. Brighter means stronger boundary evidence.",
            maps.combined_boundary_score,
            True,
        ),
    ]
    assets: list[dict[str, str]] = []
    variants: dict[str, Any] = {}
    center = ball.get("source_refined_center_px") or ball.get("source_rough_center_px")
    radius = ball.get("source_radius_px")
    for key, label, description, values, use_outward_drop in map_specs:
        canvas = _map_canvas_for_review_crop(values, maps.roi, bounds)
        relative_path = f"evidence_maps/ball_{ball_id:03d}_{key}.png"
        cv2.imwrite(str(output_directory / relative_path), canvas)
        assets.append(
            {
                "key": key,
                "label": label,
                "description": description,
                "uri": relative_path,
            }
        )
        variant = fit_radial_boundary_variant_from_feature(
            feature=values,
            roi=maps.roi,
            center_px=center,
            radius_px=radius,
            evidence_source=f"evidence_map_{key}",
            settings=settings,
            use_outward_drop=use_outward_drop,
            neighbor_ellipses=neighbor_ellipses,
        )
        if variant is not None:
            variants[key] = {
                "key": key,
                "label": label,
                "description": description,
                "sampling": "outward_drop" if use_outward_drop else "peak_response",
                "view_score": _boundary_view_score(
                    points_px=variant.get("points_px") or [],
                    rejected_points_px=variant.get("rejected_points_px") or [],
                    ellipse_fit=variant.get("ellipse_fit"),
                    sphere_projection=ball.get("source_sphere_projection"),
                    radius_px=ball.get("source_radius_px"),
                ),
                **variant,
            }
    return assets, variants


def _map_canvas_for_review_crop(
    values: np.ndarray,
    map_roi: tuple[int, int, int, int],
    review_bounds: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = review_bounds
    width = max(1, int(x1 - x0))
    height = max(1, int(y1 - y0))
    canvas = np.zeros((height, width), dtype=np.uint8)
    mx0, my0, mx1, my1 = map_roi
    overlap_x0 = max(int(x0), int(mx0))
    overlap_y0 = max(int(y0), int(my0))
    overlap_x1 = min(int(x1), int(mx1))
    overlap_y1 = min(int(y1), int(my1))
    if overlap_x1 <= overlap_x0 or overlap_y1 <= overlap_y0:
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    src_x0 = overlap_x0 - int(mx0)
    src_y0 = overlap_y0 - int(my0)
    src_x1 = src_x0 + (overlap_x1 - overlap_x0)
    src_y1 = src_y0 + (overlap_y1 - overlap_y0)
    dst_x0 = overlap_x0 - int(x0)
    dst_y0 = overlap_y0 - int(y0)
    dst_x1 = dst_x0 + (overlap_x1 - overlap_x0)
    dst_y1 = dst_y0 + (overlap_y1 - overlap_y0)
    patch = np.clip(values[src_y0:src_y1, src_x0:src_x1], 0.0, 1.0)
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = np.round(patch * 255.0).astype(np.uint8)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def _boundary_view_score(
    *,
    points_px: list[Any],
    rejected_points_px: list[Any],
    ellipse_fit: dict[str, Any] | None,
    sphere_projection: dict[str, Any] | None,
    radius_px: float | None,
) -> dict[str, Any]:
    """Diagnostic score for comparing evidence views.

    This is not ground-truth accuracy. It estimates whether one evidence view
    produced enough clean boundary points and whether those points agree with
    the current physical sphere projection.
    """

    accepted_count = len(points_px or [])
    rejected_count = len(rejected_points_px or [])
    total_count = accepted_count + rejected_count
    if accepted_count < 3 or not ellipse_fit:
        return {
            "status": "unavailable",
            "score": None,
            "level": "unknown",
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "reason": "not enough accepted points or ellipse fit is missing",
            "formula": (
                "diagnostic_score = 45% physical residual + 30% accepted "
                "point count + 20% inlier ratio + 5% ellipse availability"
            ),
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
        residual_component = float(np.clip(1.0 - rms_error / normalizer, 0.0, 1.0))
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
        "formula": (
            "diagnostic_score = 45% physical residual + 30% accepted point "
            "count + 20% inlier ratio + 5% ellipse availability"
        ),
    }


def _evidence_map_summary_with_assets(
    summary: dict[str, Any] | None,
    assets: list[dict[str, str]],
    boundary_variants: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if summary is None and not assets and not boundary_variants:
        return None
    payload = dict(summary or {})
    payload["assets"] = assets
    payload["boundary_variants"] = boundary_variants or {}
    return payload


def _review_crop_bounds(
    image_shape: tuple[int, int] | tuple[int, int, int],
    ball: dict[str, Any],
) -> tuple[int, int, int, int]:
    height, width = int(image_shape[0]), int(image_shape[1])
    center = ball.get("source_refined_center_px") or ball.get("source_rough_center_px")
    if center is None:
        return 0, 0, min(width, 256), min(height, 256)
    radius = float(ball.get("source_radius_px") or 42.0)
    half = int(max(105.0, radius * 3.6))
    x0 = max(0, int(round(float(center[0]) - half)))
    y0 = max(0, int(round(float(center[1]) - half)))
    x1 = min(width, int(round(float(center[0]) + half)))
    y1 = min(height, int(round(float(center[1]) + half)))
    return x0, y0, x1, y1


def _neighbor_source_ellipses_for_ball(
    balls: list[dict[str, Any]],
    ball: dict[str, Any],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    if not bool(settings.get("neighbor_ellipse_rejection_enabled", True)):
        return []
    current_id = int(ball.get("id", -1))
    current_center = _source_center_for_ball(ball)
    current_radius = _source_radius_for_ball(ball)
    if current_center is None or current_radius is None:
        return []
    distance_factor = float(settings.get("neighbor_ellipse_rejection_distance_factor", 3.2))
    neighbors: list[dict[str, Any]] = []
    for other in balls:
        try:
            other_id = int(other.get("id", -1))
        except (TypeError, ValueError):
            continue
        if other_id == current_id:
            continue
        other_center = _source_center_for_ball(other)
        other_radius = _source_radius_for_ball(other)
        if other_center is None or other_radius is None:
            continue
        distance = float(
            np.hypot(
                float(other_center[0]) - float(current_center[0]),
                float(other_center[1]) - float(current_center[1]),
            ),
        )
        maximum_distance = max(
            current_radius + other_radius,
            (current_radius + other_radius) * max(1.0, distance_factor),
        )
        if distance > maximum_distance:
            continue
        ellipse = _source_ellipse_for_ball(
            other,
            fallback_center=other_center,
            fallback_radius=other_radius,
        )
        if ellipse:
            ellipse["id"] = other_id
            ellipse["label"] = other.get("color_label", other.get("class", "unknown"))
            ellipse["distance_px"] = round(distance, 4)
            neighbors.append(ellipse)
    neighbors.sort(key=lambda item: float(item.get("distance_px") or 0.0))
    return neighbors


def _source_center_for_ball(ball: dict[str, Any]) -> list[float] | None:
    center = (
        ball.get("source_final_center_px")
        or ball.get("source_refined_center_px")
        or ball.get("source_rough_center_px")
    )
    if center is None:
        return None
    try:
        return [float(center[0]), float(center[1])]
    except (TypeError, ValueError, IndexError):
        return None


def _source_radius_for_ball(ball: dict[str, Any]) -> float | None:
    value = ball.get("source_radius_px") or ball.get("radius_px")
    if value is None:
        return None
    try:
        radius = float(value)
    except (TypeError, ValueError):
        return None
    return radius if radius > 2.0 else None


def _source_ellipse_for_ball(
    ball: dict[str, Any],
    *,
    fallback_center: list[float],
    fallback_radius: float,
) -> dict[str, Any] | None:
    ellipse = (
        ball.get("source_ellipse_fit")
        or ball.get("source_radial_ellipse_fit")
        or ball.get("source_silhouette_ellipse_fit")
    )
    if isinstance(ellipse, dict):
        center = ellipse.get("center_px")
        if center is None and "center_x_px" in ellipse and "center_y_px" in ellipse:
            center = [ellipse["center_x_px"], ellipse["center_y_px"]]
        major = ellipse.get("major_axis_px")
        minor = ellipse.get("minor_axis_px")
        if center is not None and major is not None and minor is not None:
            try:
                return {
                    "center_px": [round(float(center[0]), 4), round(float(center[1]), 4)],
                    "major_axis_px": round(float(major), 4),
                    "minor_axis_px": round(float(minor), 4),
                    "angle_deg": round(float(ellipse.get("angle_deg") or 0.0), 4),
                    "axis_ratio": (
                        None
                        if ellipse.get("axis_ratio") is None
                        else round(float(ellipse.get("axis_ratio")), 4)
                    ),
                    "source": ellipse.get("source") or "neighbor_source_ellipse",
                }
            except (TypeError, ValueError, IndexError):
                pass
    return {
        "center_px": [round(float(fallback_center[0]), 4), round(float(fallback_center[1]), 4)],
        "major_axis_px": round(float(fallback_radius) * 2.0, 4),
        "minor_axis_px": round(float(fallback_radius) * 2.0, 4),
        "angle_deg": 0.0,
        "axis_ratio": 1.0,
        "source": "neighbor_source_radius_fallback",
    }


def _circle_fit_payload(ball: dict[str, Any]) -> dict[str, Any]:
    success = bool(ball.get("source_refinement_success"))
    center = ball.get("source_refined_center_px") or ball.get("source_rough_center_px")
    return {
        "status": "accepted" if success else "failed",
        "center_px": center,
        "radius_px": ball.get("source_radius_px"),
        "residual_px": ball.get("source_fit_residual_px"),
        "points_used": (ball.get("debug") or {}).get("source_circle_fit_point_count", 0),
        "fallback_used": not success,
    }


def _best_ellipse_fit_payload(
    ball: dict[str, Any],
    agreement: dict[str, Any] | None = None,
    consensus_ellipse: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    return _candidate_c_ellipse_fit_payload(ball)


def _candidate_c_ellipse_fit_payload(ball: dict[str, Any]) -> dict[str, Any] | None:
    """Candidate C experiment: observed ellipse from radial/edge evidence only."""
    return _ellipse_fit_payload(ball, "source_ellipse_fit")


def _ellipse_fit_payload(ball: dict[str, Any], key: str) -> dict[str, Any] | None:
    ellipse = ball.get(key)
    if not ellipse:
        return None
    center = ellipse.get("center_px")
    if center is None and "center_x_px" in ellipse and "center_y_px" in ellipse:
        center = [ellipse["center_x_px"], ellipse["center_y_px"]]
    if center is None:
        return None
    return {
        "status": ellipse.get("status", "candidate"),
        "center_px": [float(center[0]), float(center[1])],
        "major_axis_px": ellipse.get("major_axis_px"),
        "minor_axis_px": ellipse.get("minor_axis_px"),
        "angle_deg": ellipse.get("angle_deg"),
        "axis_ratio": ellipse.get("axis_ratio"),
        "source": ellipse.get("source", key),
    }


def _mask_centroid_payload(ball: dict[str, Any]) -> dict[str, Any] | None:
    centroid = ball.get("source_mask_centroid_px")
    if centroid is None:
        return None
    center = ball.get("source_refined_center_px") or ball.get("source_rough_center_px")
    delta = (
        float(np.hypot(centroid[0] - center[0], centroid[1] - center[1]))
        if center is not None
        else None
    )
    return {
        "center_px": centroid,
        "area_px": ball.get("source_mask_area_px"),
        "delta_from_source_center_px": delta,
    }


def _model_candidates(
    ball: dict[str, Any],
    agreement: dict[str, Any] | None = None,
    consensus_ellipse: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_center = ball.get("source_refined_center_px") or ball.get("source_rough_center_px")
    sphere_projection = ball.get("source_sphere_projection") or _sphere_unavailable()
    if consensus_ellipse is None:
        consensus_ellipse = _consensus_ellipse_fit_payload(ball, agreement)
    return {
        "candidate_a_radial_circle": {
            **_circle_fit_payload(ball),
            "meaning": "Circle fitted to radial boundary points in the source image.",
        },
        "candidate_b_mask_centroid": {
            **(_mask_centroid_payload(ball) or {}),
            "status": "candidate" if _mask_centroid_payload(ball) else "missing",
            "meaning": "Centroid of the segmented ball mask. This is evidence, not the physical sphere center.",
        },
        "candidate_c_radial_ellipse": _with_meaning(
            _ellipse_fit_payload(ball, "source_ellipse_fit"),
            "Ellipse fitted to the same source-image boundary evidence.",
        ),
        "candidate_c_silhouette_ellipse": _with_meaning(
            _ellipse_fit_payload(ball, "source_silhouette_ellipse_fit"),
            "Ellipse fitted to the segmented source-image silhouette.",
        ),
        "candidate_c_consensus_ellipse": _with_meaning(
            consensus_ellipse,
            "Ellipse fitted to combined radial-boundary and mask-contour evidence when both agree.",
        ),
        "candidate_d_sphere_projection": {
            **sphere_projection,
            "meaning": (
                "Physics prediction of the projected sphere silhouette from "
                "pinhole camera geometry. In approximate mode this comes from "
                "lens/sensor metadata plus manual table corners."
            ),
        },
        "sphere_center_estimate": {
            "status": "approximate",
            "method": "source_center_ray_z26_25",
            "source_px": source_center,
            "table_xy_by_z_mm": ball.get("source_refined_table_xy_by_z_mm", {}),
            "note": (
                "This is a ray/height-plane placeholder until calibrated "
                "sphere-silhouette geometry is implemented."
            ),
        },
    }


def _with_meaning(payload: dict[str, Any] | None, meaning: str) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {**payload, "meaning": meaning}


def _consensus_ellipse_fit_payload(
    ball: dict[str, Any],
    agreement: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    agreement = agreement or _boundary_mask_agreement(ball)
    if agreement.get("status") != "agreement_high":
        return None
    radial = np.asarray(ball.get("source_boundary_points_px") or [], dtype=np.float32).reshape(-1, 2)
    mask = np.asarray(ball.get("source_mask_contour_points_px") or [], dtype=np.float32).reshape(-1, 2)
    if len(radial) < 5 or len(mask) < 5:
        return None
    maximum_mask_points = min(180, len(mask))
    if len(mask) > maximum_mask_points:
        indices = np.linspace(0, len(mask) - 1, maximum_mask_points, dtype=np.int32)
        mask = mask[indices]
    payload = fit_ellipse_payload(
        np.vstack([radial, mask]),
        source="radial_mask_consensus",
    )
    if payload is None:
        return None
    return {
        "status": payload.get("status", "candidate"),
        "center_px": [payload["center_x_px"], payload["center_y_px"]],
        "major_axis_px": payload["major_axis_px"],
        "minor_axis_px": payload["minor_axis_px"],
        "angle_deg": payload["angle_deg"],
        "axis_ratio": payload.get("axis_ratio"),
        "source": payload.get("source", "radial_mask_consensus"),
    }


def _boundary_mask_agreement(ball: dict[str, Any]) -> dict[str, Any]:
    radial = np.asarray(ball.get("source_boundary_points_px") or [], dtype=np.float64).reshape(-1, 2)
    mask = np.asarray(ball.get("source_mask_contour_points_px") or [], dtype=np.float64).reshape(-1, 2)
    radius = float(ball.get("source_radius_px") or 40.0)
    threshold = max(4.0, radius * 0.12)
    if len(radial) < 5 or len(mask) < 5:
        return {
            "status": "insufficient_points",
            "threshold_px": round(threshold, 3),
            "radial_point_count": int(len(radial)),
            "mask_point_count": int(len(mask)),
        }
    radial_distances = _nearest_distances(radial, mask)
    mask_distances = _nearest_distances(mask, radial)
    radial_overlap = float(np.mean(radial_distances <= threshold))
    mask_overlap = float(np.mean(mask_distances <= threshold))

    radial_ellipse = _ellipse_fit_payload(ball, "source_ellipse_fit")
    mask_ellipse = _ellipse_fit_payload(ball, "source_silhouette_ellipse_fit")
    ellipse_center_delta = None
    ellipse_angle_delta = None
    ellipse_ratio_delta = None
    if radial_ellipse and mask_ellipse:
        ellipse_center_delta = float(
            np.hypot(
                float(radial_ellipse["center_px"][0]) - float(mask_ellipse["center_px"][0]),
                float(radial_ellipse["center_px"][1]) - float(mask_ellipse["center_px"][1]),
            )
        )
        ellipse_angle_delta = _angle_delta_deg(
            float(radial_ellipse.get("angle_deg") or 0.0),
            float(mask_ellipse.get("angle_deg") or 0.0),
        )
        ellipse_ratio_delta = abs(
            float(radial_ellipse.get("axis_ratio") or 1.0)
            - float(mask_ellipse.get("axis_ratio") or 1.0)
        )

    high = (
        radial_overlap >= 0.72
        and mask_overlap >= 0.55
        and (
            ellipse_center_delta is None
            or ellipse_center_delta <= max(9.0, radius * 0.22)
        )
        and (
            ellipse_angle_delta is None
            or ellipse_angle_delta <= 25.0
        )
    )
    medium = radial_overlap >= 0.55 and mask_overlap >= 0.35
    status = "agreement_high" if high else ("agreement_medium" if medium else "agreement_low")
    return {
        "status": status,
        "threshold_px": round(threshold, 3),
        "radial_point_count": int(len(radial)),
        "mask_point_count": int(len(mask)),
        "radial_to_mask_overlap_fraction": round(radial_overlap, 4),
        "mask_to_radial_overlap_fraction": round(mask_overlap, 4),
        "radial_to_mask_median_px": round(float(np.median(radial_distances)), 4),
        "mask_to_radial_median_px": round(float(np.median(mask_distances)), 4),
        "ellipse_center_delta_px": (
            round(ellipse_center_delta, 4)
            if ellipse_center_delta is not None
            else None
        ),
        "ellipse_angle_delta_deg": (
            round(ellipse_angle_delta, 4)
            if ellipse_angle_delta is not None
            else None
        ),
        "ellipse_axis_ratio_delta": (
            round(ellipse_ratio_delta, 4)
            if ellipse_ratio_delta is not None
            else None
        ),
        "meaning": (
            "Agreement between white radial boundary evidence and purple mask "
            "contour evidence. High agreement increases object-detection trust, "
            "but does not by itself prove the physical sphere center."
        ),
    }


def _nearest_distances(points: np.ndarray, references: np.ndarray) -> np.ndarray:
    distances = []
    chunk_size = 128
    for start in range(0, len(points), chunk_size):
        chunk = points[start : start + chunk_size]
        diff = chunk[:, None, :] - references[None, :, :]
        distances.append(np.min(np.linalg.norm(diff, axis=2), axis=1))
    return np.concatenate(distances) if distances else np.empty((0,), dtype=np.float64)


def _angle_delta_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 90.0) % 180.0 - 90.0)


def _cushion_line_source_px(
    state: dict[str, Any],
    cushion_name: str,
) -> list[list[float]] | None:
    corners = (state.get("table") or {}).get("corner_points_px")
    if not corners or len(corners) < 4:
        return None
    mapping = {
        "top": (corners[0], corners[1]),
        "right": (corners[1], corners[2]),
        "bottom": (corners[2], corners[3]),
        "left": (corners[3], corners[0]),
    }
    line = mapping.get(cushion_name)
    if line is None:
        return None
    return [[float(line[0][0]), float(line[0][1])], [float(line[1][0]), float(line[1][1])]]


def _warnings_for_ball(
    state: dict[str, Any],
    ball: dict[str, Any],
    cushion: dict[str, Any],
    disagreement: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if not ball.get("source_refinement_success"):
        warnings.append("fallback_suspicious")
    point_count = (ball.get("debug") or {}).get("source_circle_fit_point_count", 0)
    if int(point_count or 0) <= 0:
        warnings.append("no_points")
    residual = ball.get("source_fit_residual_px")
    if residual is not None and float(residual) > 2.5:
        warnings.append("weak_radial_fit")
    ellipse = ball.get("source_ellipse_fit") or {}
    if ellipse.get("axis_ratio") is not None and float(ellipse["axis_ratio"]) > 1.25:
        warnings.append("elongated")
    if disagreement.get("mask_centroid_disagrees"):
        warnings.append("mask_centroid_disagreement")
    if disagreement.get("circle_ellipse_disagrees"):
        warnings.append("circle_ellipse_disagreement")
    if disagreement.get("model_disagrees"):
        warnings.append("model_disagreement")
    sphere_projection = ball.get("source_sphere_projection") or {}
    score = sphere_projection.get("observed_fit_score") or {}
    if (
        sphere_projection.get("status") == "predicted"
        and score.get("status") == "scored"
        and float(score.get("rms_error_px", 0.0)) > max(
            3.0,
            float(ball.get("source_radius_px") or 40.0) * 0.08,
        )
    ):
        warnings.append("sphere_projection_mismatch")
    if cushion.get("is_near"):
        warnings.append("near_cushion")
    if _near_pocket(state, ball):
        warnings.append("near_pocket")
    return list(dict.fromkeys(warnings))


def _model_disagreement(ball: dict[str, Any]) -> dict[str, Any]:
    center = ball.get("source_refined_center_px") or ball.get("source_rough_center_px")
    radius = float(ball.get("source_radius_px") or 40.0)
    threshold = max(6.0, radius * 0.16)

    centroid_payload = _mask_centroid_payload(ball)
    centroid_delta = (
        centroid_payload.get("delta_from_source_center_px")
        if centroid_payload
        else None
    )

    ellipse = _best_ellipse_fit_payload(ball)
    ellipse_delta = None
    axis_ratio = None
    if center is not None and ellipse and ellipse.get("center_px"):
        ellipse_center = ellipse["center_px"]
        ellipse_delta = float(
            np.hypot(
                float(ellipse_center[0]) - float(center[0]),
                float(ellipse_center[1]) - float(center[1]),
            )
        )
        axis_ratio = ellipse.get("axis_ratio")

    mask_disagrees = centroid_delta is not None and float(centroid_delta) > threshold
    ellipse_disagrees = ellipse_delta is not None and float(ellipse_delta) > threshold
    strong_ellipse = axis_ratio is not None and float(axis_ratio) > 1.35
    return {
        "threshold_px": round(threshold, 3),
        "mask_centroid_delta_px": (
            round(float(centroid_delta), 3) if centroid_delta is not None else None
        ),
        "circle_ellipse_center_delta_px": (
            round(float(ellipse_delta), 3) if ellipse_delta is not None else None
        ),
        "ellipse_axis_ratio": (
            round(float(axis_ratio), 3) if axis_ratio is not None else None
        ),
        "mask_centroid_disagrees": bool(mask_disagrees),
        "circle_ellipse_disagrees": bool(ellipse_disagrees),
        "strong_ellipse_aspect": bool(strong_ellipse),
        "model_disagrees": bool(mask_disagrees or ellipse_disagrees or strong_ellipse),
    }


def _near_pocket(state: dict[str, Any], ball: dict[str, Any]) -> bool:
    length_mm, width_mm = table_dimensions_from_state(state)
    xy = ball.get("source_refined_table_xy_mm") or ball.get("table_xy_mm")
    if not xy:
        return False
    x_mm, y_mm = float(xy[0]), float(xy[1])
    edge_x = min(x_mm, length_mm - x_mm)
    edge_y = min(y_mm, width_mm - y_mm)
    return bool(edge_x <= 160.0 and edge_y <= 160.0)


def _review_confidence(
    ball: dict[str, Any],
    warnings: list[str],
    evidence_agreement: dict[str, Any] | None = None,
) -> float:
    if not ball.get("source_refinement_success"):
        return 0.18 if "no_points" in warnings else 0.32
    residual = float(ball.get("source_fit_residual_px") or 0.0)
    confidence = 0.92 - min(0.45, residual / 8.0)
    if "near_cushion" in warnings:
        confidence -= 0.08
    if "elongated" in warnings:
        confidence -= 0.12
    if "model_disagreement" in warnings:
        confidence -= 0.18
    if "mask_centroid_disagreement" in warnings:
        confidence -= 0.05
    if "circle_ellipse_disagreement" in warnings:
        confidence -= 0.05
    if "sphere_projection_mismatch" in warnings:
        confidence -= 0.18
    if "low_trust_position" in warnings:
        confidence -= 0.12
    if (evidence_agreement or {}).get("status") == "agreement_high":
        confidence += 0.08
    return float(np.clip(confidence, 0.05, 0.98))


def _position_uncertainty(
    ball: dict[str, Any],
    warnings: list[str],
    disagreement: dict[str, Any],
    evidence_agreement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rule-based uncertainty until calibrated camera/sphere geometry exists."""
    reasons: list[str] = []
    sigma = 1.5
    residual = ball.get("source_fit_residual_px")
    if residual is not None:
        sigma += min(6.0, float(residual) * 0.9)
        if float(residual) > 2.5:
            reasons.append("high_residual")
    else:
        sigma += 5.0

    if not ball.get("source_refinement_success"):
        sigma += 7.0
        reasons.append("fallback_used")
    for warning, increment in (
        ("no_points", 4.0),
        ("weak_radial_fit", 2.5),
        ("elongated", 4.0),
        ("model_disagreement", 2.5),
        ("mask_centroid_disagreement", 1.0),
        ("circle_ellipse_disagreement", 1.0),
        ("sphere_projection_mismatch", 3.0),
        ("near_cushion", 4.0),
        ("near_pocket", 5.0),
    ):
        if warning in warnings:
            sigma += increment
            reasons.append(warning)

    centroid_delta = disagreement.get("mask_centroid_delta_px")
    if centroid_delta is not None and disagreement.get("mask_centroid_disagrees"):
        sigma += min(8.0, float(centroid_delta) * 0.08)
        reasons.append("centroid_disagreement")

    ellipse = ball.get("source_ellipse_fit") or {}
    axis_ratio = ellipse.get("axis_ratio")
    if axis_ratio is not None and float(axis_ratio) > 1.35:
        sigma += 3.0
        reasons.append("strong_ellipse_aspect")

    if (evidence_agreement or {}).get("status") == "agreement_high":
        sigma = max(1.0, sigma - 1.5)
        reasons.append("radial_mask_agreement")

    sigma = float(np.clip(sigma, 1.0, 30.0))
    if sigma <= 3.5:
        confidence = "high"
    elif sigma <= 9.0:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "sigma_x": round(sigma, 3),
        "sigma_y": round(sigma, 3),
        "radial_sigma": round(sigma, 3),
        "confidence": confidence,
        "reason": sorted(set(reasons)),
        "method": "heuristic_v1",
        "note": "Heuristic until calibrated camera and sphere geometry are available.",
    }


def _model_decision(
    ball: dict[str, Any],
    warnings: list[str],
    uncertainty: dict[str, Any],
    disagreement: dict[str, Any],
) -> dict[str, Any]:
    circle = _circle_fit_payload(ball)
    source_success = bool(ball.get("source_refinement_success"))
    final_center_source = (
        "source_refined_center_px" if source_success else "source_rough_center_px"
    )
    final_center = (
        ball.get("source_refined_center_px")
        if source_success
        else ball.get("source_rough_center_px")
    )
    selected_model = "circle_radial" if source_success else "fallback_radial"
    reasons: list[str] = []

    if not source_success:
        status = "fallback"
        table_position_trust = "low"
        reasons.append("no accepted circle baseline")
        if _best_ellipse_fit_payload(ball) or _mask_centroid_payload(ball):
            reasons.append("mask/ellipse evidence exists but is not yet converted into a sphere-center estimate")
        summary = (
            "No reliable circle baseline fit. Current table position uses the rough "
            "source point fallback and should be manually reviewed."
        )
    elif any(
        tag in warnings
        for tag in (
            "model_disagreement",
            "elongated",
            "near_cushion",
            "near_pocket",
            "weak_radial_fit",
        )
    ):
        status = "review"
        table_position_trust = str(uncertainty.get("confidence") or "medium")
        for tag in warnings:
            if tag in (
                "model_disagreement",
                "mask_centroid_disagreement",
                "circle_ellipse_disagreement",
                "sphere_projection_mismatch",
                "elongated",
                "near_cushion",
                "near_pocket",
                "weak_radial_fit",
            ):
                reasons.append(tag)
        summary = (
            "Circle fit produced a center, but the model candidates or table "
            "region make the physical sphere center uncertain. Treat this as "
            "a review item, not a clean acceptance."
        )
    else:
        status = "accepted"
        table_position_trust = str(uncertainty.get("confidence") or "high")
        reasons.append("radial boundary support and candidate agreement passed")
        summary = "Circle baseline is currently accepted by the legacy decision model."

    if table_position_trust == "low" and "low_trust_position" not in reasons:
        reasons.append("low_trust_position")

    return {
        "status": status,
        "selected_model": selected_model,
        "final_center_source": final_center_source,
        "final_center_px": final_center,
        "table_position_trust": table_position_trust,
        "reasons": sorted(set(reasons)),
        "summary": summary,
        "circle_candidate_status": circle.get("status"),
        "candidate_count": int(
            bool(circle.get("center_px"))
            + bool(_mask_centroid_payload(ball))
            + bool(_ellipse_fit_payload(ball, "source_ellipse_fit"))
            + bool(_ellipse_fit_payload(ball, "source_silhouette_ellipse_fit"))
            + bool((ball.get("source_sphere_projection") or {}).get("status") == "predicted")
        ),
        "disagreement": disagreement,
        "note": (
            "This is the legacy 2D evidence decision. Candidate D-first scoring "
            "is reported separately as physics_first_model_decision."
        ),
    }


def _sphere_unavailable() -> dict[str, Any]:
    return {
        "status": "unavailable",
        "method": "pinhole_sphere_tangent_cone",
        "reason": "camera calibration is required",
    }
