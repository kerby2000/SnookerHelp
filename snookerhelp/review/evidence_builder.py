from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from snookerhelp.recognition.evidence_maps import (
    compute_ball_evidence_maps,
    compute_full_table_evidence_maps,
    estimate_global_cloth_reference,
)
from snookerhelp.recognition.arc_combo_fit import (
    arc_combination_refit as recognition_arc_combination_refit,
)
from snookerhelp.recognition.image_model import fit_ellipse_payload
from snookerhelp.recognition.evidence_scoring import boundary_view_score
from snookerhelp.recognition.source_refinement import (
    fit_radial_boundary_variant_from_feature,
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
    "cluster_shape_outlier",
    "cluster_ellipse_size_outlier",
    "cluster_ellipse_angle_outlier",
    "neighbor_ellipse_ownership_conflict",
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
    full_table_evidence_maps = compute_full_table_evidence_maps(
        source_image=source_image,
        table_corners_px=(state.get("table") or {}).get("corner_points_px"),
        settings=map_settings,
    )

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
            full_table_evidence_maps=full_table_evidence_maps,
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
                    "boundary_view_score": boundary_view_score(
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
                "rejection_addback_scenarios": _rejection_addback_scenarios(
                    points_px=ball.get(
                        "source_radial_boundary_points_px",
                        ball.get("source_boundary_points_px", []),
                    ),
                    filter_stats=ball.get(
                        "source_radial_boundary_filter",
                        ball.get("source_boundary_filter", {}),
                    ),
                    cluster_shape_prior=_cluster_shape_prior(ball),
                ),
                "consensus_reject_refit": _consensus_reject_refit(
                    points_px=ball.get(
                        "source_radial_boundary_points_px",
                        ball.get("source_boundary_points_px", []),
                    ),
                    filter_stats=ball.get(
                        "source_radial_boundary_filter",
                        ball.get("source_boundary_filter", {}),
                    ),
                    cluster_shape_prior=_cluster_shape_prior(ball),
                ),
                "arc_combination_refit": _arc_combination_refit(
                    points_px=ball.get(
                        "source_radial_boundary_points_px",
                        ball.get("source_boundary_points_px", []),
                    ),
                    rejected_points_px=ball.get(
                        "source_radial_boundary_rejected_points_px",
                        ball.get("source_boundary_rejected_points_px", []),
                    ),
                    filter_stats=ball.get(
                        "source_radial_boundary_filter",
                        ball.get("source_boundary_filter", {}),
                    ),
                    cluster_shape_prior=_cluster_shape_prior(ball),
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
    full_table_evidence_maps: Any | None = None,
    neighbor_ellipses: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Write diagnostic scalar maps aligned to the review crop coordinate frame."""

    if not bool(settings.get("enabled", True)):
        return [], {}
    center = ball.get("source_refined_center_px") or ball.get("source_rough_center_px")
    call_settings = dict(settings)
    if full_table_evidence_maps is not None:
        call_settings["_full_table_evidence_maps"] = full_table_evidence_maps
    maps = compute_ball_evidence_maps(
        source_image=source_image,
        center_px=center,
        radius_px=ball.get("source_radius_px"),
        label=str(ball.get("color_label") or ball.get("class") or "unknown"),
        sphere_projection=ball.get("source_sphere_projection"),
        settings=call_settings,
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
            settings=call_settings,
            use_outward_drop=use_outward_drop,
            neighbor_ellipses=neighbor_ellipses,
        )
        if variant is not None:
            variants[key] = {
                "key": key,
                "label": label,
                "description": description,
                "sampling": "outward_drop" if use_outward_drop else "peak_response",
                "view_score": boundary_view_score(
                    points_px=variant.get("points_px") or [],
                    rejected_points_px=variant.get("rejected_points_px") or [],
                    ellipse_fit=variant.get("ellipse_fit"),
                    sphere_projection=ball.get("source_sphere_projection"),
                    radius_px=ball.get("source_radius_px"),
                ),
                "addback_scenarios": _rejection_addback_scenarios(
                    points_px=variant.get("points_px") or [],
                    filter_stats=variant.get("filter") or {},
                    cluster_shape_prior=_cluster_shape_prior(ball),
                ),
                "consensus_reject_refit": _consensus_reject_refit(
                    points_px=variant.get("points_px") or [],
                    filter_stats=variant.get("filter") or {},
                    cluster_shape_prior=_cluster_shape_prior(ball),
                ),
                "arc_combination_refit": _arc_combination_refit(
                    points_px=variant.get("points_px") or [],
                    rejected_points_px=variant.get("rejected_points_px") or [],
                    filter_stats=variant.get("filter") or {},
                    cluster_shape_prior=_cluster_shape_prior(ball),
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


def _rejection_addback_scenarios(
    *,
    points_px: list[Any],
    filter_stats: dict[str, Any] | None,
    cluster_shape_prior: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Fit diagnostic ellipses after adding rejected-point categories back.

    These scenarios are read-only diagnostics. They answer: if a rejection
    category was too aggressive, which ellipse would we get? The final estimate
    is not changed here.
    """

    accepted = _points_array(points_px)
    records = _rejected_point_records(filter_stats)
    if len(accepted) < 5 or not records:
        return []

    scenarios = [
        ("baseline", "accepted only", set()),
        ("add_endpoint", "add angular endpoints", {"angular_segment_endpoint"}),
        ("add_local_radius", "add local radius spikes", {"local_radius_spike"}),
        (
            "add_neighbor_overlap",
            "add neighbor-overlap rejects",
            {"neighbor_ellipse_overlap"},
        ),
        ("add_residual", "add ellipse residual rejects", {"ellipse_residual_outlier"}),
        (
            "add_endpoint_local",
            "add endpoints + local radius",
            {"angular_segment_endpoint", "local_radius_spike"},
        ),
        (
            "add_all_rejected",
            "add all rejected",
            {
                "angular_segment_endpoint",
                "local_radius_spike",
                "neighbor_ellipse_overlap",
                "ellipse_residual_outlier",
                "other_rejected",
                "unknown_rejected",
            },
        ),
    ]

    payloads: list[dict[str, Any]] = []
    for key, label, reasons in scenarios:
        add_points = _points_for_reasons(records, reasons)
        scenario_points = accepted if not reasons else _stack_points(accepted, add_points)
        ellipse = fit_ellipse_payload(
            scenario_points,
            source=f"rejection_addback_{key}",
        )
        comparison = _compare_ellipse_to_cluster_shape(
            ellipse,
            cluster_shape_prior or {},
        )
        payloads.append(
            {
                "key": key,
                "label": label,
                "added_count": int(len(add_points)),
                "total_count": int(len(scenario_points)),
                "ellipse_fit": ellipse,
                "cluster_shape_comparison": comparison,
                "diagnostic_only": True,
            }
        )

    ranked = [
        item
        for item in payloads
        if item["ellipse_fit"]
        and item["cluster_shape_comparison"].get("score") is not None
    ]
    if ranked:
        best_key = max(
            ranked,
            key=lambda item: float(item["cluster_shape_comparison"]["score"]),
        )["key"]
        for item in payloads:
            item["best_shape_match"] = item["key"] == best_key
    return payloads


def _consensus_reject_refit(
    *,
    points_px: list[Any],
    filter_stats: dict[str, Any] | None,
    cluster_shape_prior: dict[str, Any] | None,
) -> dict[str, Any]:
    """Search rejected-point subsets for a cluster-consistent ellipse.

    This is deliberately diagnostic. It answers the user's current question:
    "if some purple neighbor-overlap rejects were actually useful for this
    ball, which subset best matches the same-color cluster consensus?"
    """

    accepted = _points_array(points_px)
    records = _rejected_point_records(filter_stats)
    if len(accepted) < 5:
        return {"status": "not_computed", "reason": "not enough accepted points"}
    if not records:
        return {"status": "not_computed", "reason": "no rejected points"}

    prior = cluster_shape_prior or {}
    baseline_ellipse = fit_ellipse_payload(accepted, source="consensus_refit_baseline")
    baseline_comparison = _compare_ellipse_to_cluster_shape(baseline_ellipse, prior)
    if baseline_comparison.get("score") is None:
        return {
            "status": "not_computed",
            "reason": "no cluster consensus shape prior",
            "baseline": {
                "ellipse_fit": baseline_ellipse,
                "cluster_shape_comparison": baseline_comparison,
            },
        }

    candidate_records = _records_for_reasons(records, {"neighbor_ellipse_overlap"})
    preferred_reason = "neighbor_ellipse_overlap"
    if len(candidate_records) < 3:
        candidate_records = _records_for_reasons(
            records,
            {
                "neighbor_ellipse_overlap",
                "ellipse_residual_outlier",
                "local_radius_spike",
                "angular_segment_endpoint",
                "other_rejected",
                "unknown_rejected",
            },
        )
        preferred_reason = "all_rejected"
    if len(candidate_records) < 3:
        return {
            "status": "not_computed",
            "reason": "too few candidate rejected points",
            "baseline": {
                "ellipse_fit": baseline_ellipse,
                "cluster_shape_comparison": baseline_comparison,
            },
        }

    center = _ellipse_center_or_mean(baseline_ellipse, accepted)
    groups = _angular_rejected_groups(
        candidate_records,
        center_px=center,
        rough_radius_px=_rough_radius_from_points(accepted, center),
    )
    if not groups:
        return {
            "status": "not_computed",
            "reason": "no rejected point groups",
            "baseline": {
                "ellipse_fit": baseline_ellipse,
                "cluster_shape_comparison": baseline_comparison,
            },
        }

    evaluated: list[dict[str, Any]] = []
    groups_for_search = sorted(
        groups,
        key=lambda group: (-int(group["count"]), float(group["angle_start_deg"])),
    )[:12]
    max_combo_size = min(3, len(groups_for_search))
    for combo_size in range(1, max_combo_size + 1):
        for group_combo in combinations(groups_for_search, combo_size):
            add_points = _stack_points(
                np.empty((0, 2), dtype=np.float64),
                _points_array(
                    [
                        point
                        for group in group_combo
                        for point in group.get("points_px", [])
                    ]
                ),
            )
            if len(add_points) < 1:
                continue
            scenario_points = _stack_points(accepted, add_points)
            ellipse = fit_ellipse_payload(
                scenario_points,
                source="consensus_selected_reject_refit",
            )
            comparison = _compare_ellipse_to_cluster_shape(ellipse, prior)
            if not ellipse or comparison.get("score") is None:
                continue
            residual_px = _ellipse_rms_residual_px(ellipse, scenario_points)
            reason_counts: dict[str, int] = {}
            for group in group_combo:
                for reason, count in (group.get("reason_counts") or {}).items():
                    reason_counts[reason] = reason_counts.get(reason, 0) + int(count)
            shape_score = float(comparison["score"])
            residual_penalty = min(22.0, max(0.0, residual_px - 2.0) * 2.5)
            count_bonus = min(4.0, float(len(add_points)) * 0.18)
            ranking_score = shape_score - residual_penalty + count_bonus
            evaluated.append(
                {
                    "group_ids": [int(group["group_id"]) for group in group_combo],
                    "added_count": int(len(add_points)),
                    "selected_rejected_points_px": _round_points(add_points),
                    "added_reason_counts": reason_counts,
                    "ellipse_fit": ellipse,
                    "cluster_shape_comparison": comparison,
                    "ellipse_rms_residual_px": round(float(residual_px), 4),
                    "ranking_score": round(float(ranking_score), 4),
                }
            )

    if not evaluated:
        return {
            "status": "no_fit",
            "reason": "candidate subsets did not produce a fitted ellipse",
            "candidate_reason_mode": preferred_reason,
            "groups": groups,
            "baseline": {
                "ellipse_fit": baseline_ellipse,
                "cluster_shape_comparison": baseline_comparison,
            },
        }

    best = max(evaluated, key=lambda item: float(item["ranking_score"]))
    baseline_score = float(baseline_comparison["score"])
    best_score = float(best["cluster_shape_comparison"]["score"])
    improvement = best_score - baseline_score
    return {
        "status": "improved" if improvement > 4.0 else "diagnostic_only",
        "diagnostic_only": True,
        "candidate_reason_mode": preferred_reason,
        "groups_considered": len(groups),
        "candidate_count": len(evaluated),
        "baseline": {
            "ellipse_fit": baseline_ellipse,
            "cluster_shape_comparison": baseline_comparison,
        },
        "best": {
            **best,
            "shape_score_improvement": round(float(improvement), 4),
            "note": (
                "best rejected-point subset by cluster consensus shape; "
                "not used for final center yet"
            ),
        },
    }


def _arc_combination_refit(
    *,
    points_px: list[Any],
    rejected_points_px: list[Any] | None,
    filter_stats: dict[str, Any] | None,
    cluster_shape_prior: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fit ellipses to all combinations of raw boundary arc clusters.

    This diagnostic answers a different question than the normal outlier
    filter. It starts from the pre-filter radial samples, splits them into
    angular/spatial clusters, tries every non-empty cluster combination, and
    ranks the resulting ellipses against the cluster-wide expected shape.
    """
    return recognition_arc_combination_refit(
        points_px=points_px,
        rejected_points_px=rejected_points_px,
        filter_stats=filter_stats,
        cluster_shape_prior=cluster_shape_prior,
        include_fixed_shape_candidates=True,
        max_fixed_shape_combo_size=5,
    )

    accepted = _points_array(points_px)
    stats = filter_stats or {}
    raw = _points_array(stats.get("raw_points_px") or [])
    if len(raw) < 5:
        raw = _stack_points(
            accepted,
            _points_array(rejected_points_px or []),
        )
    if len(raw) < 8:
        return {
            "status": "not_computed",
            "reason": "not enough raw boundary points",
            "raw_count": int(len(raw)),
            "diagnostic_only": True,
        }

    prior = cluster_shape_prior or {}
    baseline_points = accepted if len(accepted) >= 5 else raw
    baseline_ellipse = fit_ellipse_payload(
        baseline_points,
        source="arc_combo_current_filtered_baseline",
    )
    center = _ellipse_center_or_mean(baseline_ellipse, raw)
    rough_radius = _rough_radius_from_points(raw, center)
    groups = _raw_arc_groups(
        raw,
        center_px=center,
        rough_radius_px=rough_radius,
    )
    eligible_groups = [group for group in groups if int(group.get("count") or 0) >= 2]
    if not eligible_groups:
        return {
            "status": "not_computed",
            "reason": "raw points did not form usable arc clusters",
            "raw_count": int(len(raw)),
            "groups": groups,
            "diagnostic_only": True,
        }

    max_search_groups = 10
    groups_for_search = eligible_groups
    capped = False
    if len(groups_for_search) > max_search_groups:
        capped = True
        groups_for_search = sorted(
            groups_for_search,
            key=lambda group: (-int(group["count"]), float(group["angle_start_deg"])),
        )[:max_search_groups]
        groups_for_search = sorted(
            groups_for_search,
            key=lambda group: float(group["angle_start_deg"]),
        )

    evaluated: list[dict[str, Any]] = []
    theoretical_non_empty = (1 << len(groups_for_search)) - 1
    minimum_combo_points = max(5, min(18, int(round(len(raw) * 0.12))))
    for combo_size in range(1, len(groups_for_search) + 1):
        for group_combo in combinations(groups_for_search, combo_size):
            combo_points = _points_array(
                [
                    point
                    for group in group_combo
                    for point in group.get("points_px", [])
                ]
            )
            if len(combo_points) < minimum_combo_points:
                continue
            ellipse = fit_ellipse_payload(
                combo_points,
                source="raw_arc_cluster_combination",
            )
            if not ellipse:
                continue
            comparison = _compare_ellipse_to_cluster_shape(ellipse, prior)
            residual_px = _ellipse_rms_residual_px(ellipse, combo_points)
            shape_score = (
                50.0
                if comparison.get("score") is None
                else float(comparison["score"])
            )
            residual_score = max(0.0, 100.0 - min(100.0, residual_px * 12.0))
            point_fraction = min(1.0, float(len(combo_points)) / max(1.0, float(len(raw))))
            point_score = min(100.0, point_fraction * 155.0)
            multi_arc_score = min(100.0, float(combo_size) * 28.0)
            ranking_score = (
                0.55 * shape_score
                + 0.25 * residual_score
                + 0.15 * point_score
                + 0.05 * multi_arc_score
            )
            evaluated.append(
                {
                    "group_ids": [int(group["group_id"]) for group in group_combo],
                    "group_count": int(combo_size),
                    "point_count": int(len(combo_points)),
                    "selected_points_px": _round_points(combo_points),
                    "ellipse_fit": _review_ellipse_payload_from_fit(ellipse),
                    "cluster_shape_comparison": comparison,
                    "ellipse_rms_residual_px": round(float(residual_px), 4),
                    "point_fraction": round(float(point_fraction), 4),
                    "ranking_score": round(float(ranking_score), 4),
                    "score_components": {
                        "shape_score": round(float(shape_score), 4),
                        "residual_score": round(float(residual_score), 4),
                        "point_score": round(float(point_score), 4),
                        "multi_arc_score": round(float(multi_arc_score), 4),
                    },
                }
            )

    baseline_comparison = _compare_ellipse_to_cluster_shape(baseline_ellipse, prior)
    baseline_residual = _ellipse_rms_residual_px(baseline_ellipse, baseline_points)
    baseline_score = (
        None
        if baseline_comparison.get("score") is None
        else float(baseline_comparison["score"])
    )
    if not evaluated:
        return {
            "status": "no_fit",
            "reason": "cluster combinations did not produce a fitted ellipse",
            "raw_count": int(len(raw)),
            "accepted_count": int(len(accepted)),
            "cluster_count": int(len(eligible_groups)),
            "combination_formula": "2^n - 1",
            "theoretical_combination_count": int(theoretical_non_empty),
            "combination_count": 0,
            "minimum_combo_points": int(minimum_combo_points),
            "capped": bool(capped),
            "groups": groups,
            "baseline": {
                "ellipse_fit": _review_ellipse_payload_from_fit(baseline_ellipse),
                "cluster_shape_comparison": baseline_comparison,
                "ellipse_rms_residual_px": (
                    None
                    if not np.isfinite(baseline_residual)
                    else round(float(baseline_residual), 4)
                ),
            },
            "diagnostic_only": True,
        }

    ranked = sorted(
        evaluated,
        key=lambda item: float(item["ranking_score"]),
        reverse=True,
    )
    best = ranked[0]
    best_shape_score = best["cluster_shape_comparison"].get("score")
    shape_improvement = (
        None
        if baseline_score is None or best_shape_score is None
        else float(best_shape_score) - float(baseline_score)
    )
    return {
        "status": (
            "improved"
            if shape_improvement is not None and shape_improvement > 4.0
            else "diagnostic_only"
        ),
        "raw_count": int(len(raw)),
        "accepted_count": int(len(accepted)),
        "cluster_count": int(len(eligible_groups)),
        "searched_cluster_count": int(len(groups_for_search)),
        "combination_formula": "2^n - 1",
        "theoretical_combination_count": int(theoretical_non_empty),
        "combination_count": int(len(evaluated)),
        "minimum_combo_points": int(minimum_combo_points),
        "capped": bool(capped),
        "groups": groups,
        "baseline": {
            "ellipse_fit": _review_ellipse_payload_from_fit(baseline_ellipse),
            "cluster_shape_comparison": baseline_comparison,
            "ellipse_rms_residual_px": (
                None
                if not np.isfinite(baseline_residual)
                else round(float(baseline_residual), 4)
            ),
        },
        "best": {
            **best,
            "shape_score_improvement": (
                None
                if shape_improvement is None
                else round(float(shape_improvement), 4)
            ),
            "note": (
                "best raw arc-cluster combination by cluster shape, residual, "
                "and point coverage; diagnostic only"
            ),
        },
        "top_candidates": ranked[:8],
        "diagnostic_only": True,
    }


def _review_ellipse_payload_from_fit(
    ellipse: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not ellipse:
        return None
    center = ellipse.get("center_px")
    if center is None and ellipse.get("center_x_px") is not None:
        center = [ellipse.get("center_x_px"), ellipse.get("center_y_px")]
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        return None
    return {
        "status": ellipse.get("status", "candidate"),
        "center_px": [round(float(center[0]), 4), round(float(center[1]), 4)],
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


def _raw_arc_groups(
    points_px: np.ndarray,
    *,
    center_px: np.ndarray,
    rough_radius_px: float,
) -> list[dict[str, Any]]:
    points = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return []

    items: list[dict[str, Any]] = []
    for source_index, point in enumerate(points):
        dx = float(point[0]) - float(center_px[0])
        dy = float(point[1]) - float(center_px[1])
        angle = float(np.arctan2(dy, dx) % (2.0 * np.pi))
        items.append(
            {
                "source_index": int(source_index),
                "point_px": [float(point[0]), float(point[1])],
                "angle": angle,
            }
        )
    items.sort(key=lambda item: float(item["angle"]))

    angles = np.array([float(item["angle"]) for item in items], dtype=np.float64)
    wrapped_angles = np.concatenate([angles, [angles[0] + 2.0 * np.pi]])
    diffs = np.diff(wrapped_angles)
    positive_diffs = diffs[diffs > 1e-6]
    median_gap = (
        float(np.median(positive_diffs))
        if len(positive_diffs)
        else float(np.deg2rad(2.0))
    )
    angle_gap = float(np.clip(median_gap * 5.0, np.deg2rad(8.0), np.deg2rad(24.0)))
    distance_gap = max(8.0, float(rough_radius_px) * 0.38)

    groups: list[list[dict[str, Any]]] = [[items[0]]]
    for item in items[1:]:
        previous = groups[-1][-1]
        angular_gap = float(item["angle"]) - float(previous["angle"])
        spatial_gap = float(
            np.hypot(
                float(item["point_px"][0]) - float(previous["point_px"][0]),
                float(item["point_px"][1]) - float(previous["point_px"][1]),
            )
        )
        if angular_gap <= angle_gap and spatial_gap <= distance_gap:
            groups[-1].append(item)
        else:
            groups.append([item])

    if len(groups) > 1:
        first = groups[0][0]
        last = groups[-1][-1]
        wrap_gap = (float(first["angle"]) + 2.0 * np.pi) - float(last["angle"])
        wrap_distance = float(
            np.hypot(
                float(first["point_px"][0]) - float(last["point_px"][0]),
                float(first["point_px"][1]) - float(last["point_px"][1]),
            )
        )
        if wrap_gap <= angle_gap and wrap_distance <= distance_gap:
            groups[0] = groups[-1] + groups[0]
            groups.pop()

    payloads: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        group_points = _points_array([item["point_px"] for item in group])
        angles_deg = [float(np.rad2deg(float(item["angle"]))) for item in group]
        payloads.append(
            {
                "group_id": int(index),
                "count": int(len(group)),
                "source_indices": [int(item["source_index"]) for item in group],
                "points_px": _round_points(group_points),
                "angle_start_deg": round(float(min(angles_deg)), 3),
                "angle_end_deg": round(float(max(angles_deg)), 3),
            }
        )
    return payloads


def _points_array(points_px: list[Any]) -> np.ndarray:
    points = np.asarray(points_px or [], dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return points.reshape(-1, 2)


def _rejected_point_records(
    filter_stats: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    stats = filter_stats or {}
    records = stats.get("rejected_point_reasons") or []
    if records:
        return [record for record in records if _record_point(record) is not None]
    fallback_points = stats.get("rejected_points_px") or []
    return [
        {
            "point_px": [float(point[0]), float(point[1])],
            "primary_reason": "unknown_rejected",
            "reasons": ["unknown_rejected"],
        }
        for point in fallback_points
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]


def _record_point(record: dict[str, Any]) -> list[float] | None:
    point = record.get("point_px")
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    return [float(point[0]), float(point[1])]


def _points_for_reasons(
    records: list[dict[str, Any]],
    reasons: set[str],
) -> np.ndarray:
    if not reasons:
        return np.empty((0, 2), dtype=np.float64)
    points: list[list[float]] = []
    for record in records:
        record_reasons = {str(reason) for reason in record.get("reasons") or []}
        record_reasons.add(str(record.get("primary_reason") or "unknown_rejected"))
        if record_reasons.intersection(reasons):
            point = _record_point(record)
            if point is not None:
                points.append(point)
    return _points_array(points)


def _records_for_reasons(
    records: list[dict[str, Any]],
    reasons: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        record_reasons = _record_reasons(record)
        if record_reasons.intersection(reasons) and _record_point(record) is not None:
            selected.append(record)
    return selected


def _record_reasons(record: dict[str, Any]) -> set[str]:
    record_reasons = {str(reason) for reason in record.get("reasons") or []}
    record_reasons.add(str(record.get("primary_reason") or "unknown_rejected"))
    return record_reasons


def _stack_points(
    accepted: np.ndarray,
    add_points: np.ndarray,
) -> np.ndarray:
    if len(add_points) == 0:
        return accepted
    if len(accepted) == 0:
        return add_points
    return np.vstack([accepted, add_points])


def _ellipse_center_or_mean(
    ellipse: dict[str, Any] | None,
    points: np.ndarray,
) -> np.ndarray:
    if ellipse and ellipse.get("center_x_px") is not None and ellipse.get("center_y_px") is not None:
        return np.array(
            [float(ellipse["center_x_px"]), float(ellipse["center_y_px"])],
            dtype=np.float64,
        )
    if ellipse and isinstance(ellipse.get("center_px"), (list, tuple)):
        center = ellipse["center_px"]
        return np.array([float(center[0]), float(center[1])], dtype=np.float64)
    return np.mean(points, axis=0) if len(points) else np.array([0.0, 0.0], dtype=np.float64)


def _rough_radius_from_points(points: np.ndarray, center: np.ndarray) -> float:
    if len(points) == 0:
        return 40.0
    distances = np.linalg.norm(points - center.reshape(1, 2), axis=1)
    return float(np.median(distances)) if len(distances) else 40.0


def _angular_rejected_groups(
    records: list[dict[str, Any]],
    *,
    center_px: np.ndarray,
    rough_radius_px: float,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for record in records:
        point = _record_point(record)
        if point is None:
            continue
        dx = float(point[0]) - float(center_px[0])
        dy = float(point[1]) - float(center_px[1])
        angle = float(np.arctan2(dy, dx) % (2.0 * np.pi))
        items.append(
            {
                "point_px": [float(point[0]), float(point[1])],
                "angle": angle,
                "reasons": sorted(_record_reasons(record)),
            }
        )
    if not items:
        return []
    items.sort(key=lambda item: float(item["angle"]))
    angle_gap = np.deg2rad(10.0)
    distance_gap = max(8.0, float(rough_radius_px) * 0.38)
    groups: list[list[dict[str, Any]]] = [[items[0]]]
    for item in items[1:]:
        previous = groups[-1][-1]
        angular_gap = float(item["angle"]) - float(previous["angle"])
        spatial_gap = float(
            np.hypot(
                float(item["point_px"][0]) - float(previous["point_px"][0]),
                float(item["point_px"][1]) - float(previous["point_px"][1]),
            )
        )
        if angular_gap <= angle_gap and spatial_gap <= distance_gap:
            groups[-1].append(item)
        else:
            groups.append([item])

    if len(groups) > 1:
        first = groups[0][0]
        last = groups[-1][-1]
        wrap_gap = (float(first["angle"]) + 2.0 * np.pi) - float(last["angle"])
        wrap_distance = float(
            np.hypot(
                float(first["point_px"][0]) - float(last["point_px"][0]),
                float(first["point_px"][1]) - float(last["point_px"][1]),
            )
        )
        if wrap_gap <= angle_gap and wrap_distance <= distance_gap:
            groups[0] = groups[-1] + groups[0]
            groups.pop()

    payloads: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        reason_counts: dict[str, int] = {}
        for item in group:
            for reason in item.get("reasons") or []:
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
        angles = [float(item["angle"]) for item in group]
        payloads.append(
            {
                "group_id": index,
                "count": len(group),
                "points_px": [item["point_px"] for item in group],
                "reason_counts": reason_counts,
                "angle_start_deg": round(float(np.rad2deg(min(angles))), 3),
                "angle_end_deg": round(float(np.rad2deg(max(angles))), 3),
            }
        )
    return payloads


def _round_points(points: np.ndarray) -> list[list[float]]:
    return [
        [round(float(point[0]), 4), round(float(point[1]), 4)]
        for point in np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ]


def _ellipse_rms_residual_px(
    ellipse: dict[str, Any] | None,
    points: np.ndarray,
) -> float:
    if not ellipse or len(points) == 0:
        return float("inf")
    center = _ellipse_center_or_mean(ellipse, points)
    major = float(ellipse.get("major_axis_px") or 0.0)
    minor = float(ellipse.get("minor_axis_px") or 0.0)
    if major <= 0.0 or minor <= 0.0:
        return float("inf")
    rx = major / 2.0
    ry = minor / 2.0
    angle = np.deg2rad(float(ellipse.get("angle_deg") or 0.0))
    shifted = np.asarray(points, dtype=np.float64).reshape(-1, 2) - center.reshape(1, 2)
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    local_x = shifted[:, 0] * cos_a + shifted[:, 1] * sin_a
    local_y = -shifted[:, 0] * sin_a + shifted[:, 1] * cos_a
    normalized_radius = np.sqrt((local_x / rx) ** 2 + (local_y / ry) ** 2)
    residual_px = np.abs(normalized_radius - 1.0) * ((rx + ry) * 0.5)
    return float(np.sqrt(np.mean(residual_px**2)))


def _compare_ellipse_to_cluster_shape(
    ellipse: dict[str, Any] | None,
    cluster_shape_prior: dict[str, Any],
) -> dict[str, Any]:
    if not ellipse:
        return {"status": "no_ellipse", "score": None, "reasons": ["ellipse_fit_failed"]}
    consensus_major = cluster_shape_prior.get("consensus_major_axis_px")
    consensus_minor = cluster_shape_prior.get("consensus_minor_axis_px")
    consensus_angle = cluster_shape_prior.get("consensus_angle_deg")
    if consensus_major is None or consensus_minor is None or consensus_angle is None:
        return {"status": "no_cluster_shape_prior", "score": None, "reasons": []}

    major_scale_limit = 1.22
    minor_scale_limit = 1.22
    angle_limit = 12.0
    major = float(ellipse.get("major_axis_px") or 0.0)
    minor = float(ellipse.get("minor_axis_px") or 0.0)
    angle = float(ellipse.get("angle_deg") or 0.0)
    major_scale = _symmetric_scale(major, float(consensus_major))
    minor_scale = _symmetric_scale(minor, float(consensus_minor))
    angle_delta = _angle_delta_deg(angle, float(consensus_angle))
    reasons: list[str] = []
    if major_scale > major_scale_limit:
        reasons.append("cluster_ellipse_major_outlier")
    if minor_scale > minor_scale_limit:
        reasons.append("cluster_ellipse_minor_outlier")
    if angle_delta > angle_limit:
        reasons.append("cluster_ellipse_angle_outlier")

    major_penalty = max(0.0, (major_scale - 1.0) / (major_scale_limit - 1.0))
    minor_penalty = max(0.0, (minor_scale - 1.0) / (minor_scale_limit - 1.0))
    angle_penalty = max(0.0, angle_delta / angle_limit)
    score = max(
        0.0,
        100.0 - 28.0 * major_penalty - 28.0 * minor_penalty - 24.0 * angle_penalty,
    )
    return {
        "status": "computed",
        "score": round(score, 2),
        "major_scale": round(major_scale, 4),
        "minor_scale": round(minor_scale, 4),
        "angle_delta_deg": round(angle_delta, 4),
        "is_shape_outlier": bool(reasons),
        "reasons": reasons or ["shape_prior_match"],
    }


def _symmetric_scale(value: float, reference: float) -> float:
    if value <= 0.0 or reference <= 0.0:
        return float("inf")
    scale = float(value) / float(reference)
    return max(scale, 1.0 / scale)


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
    cluster_shape = _cluster_shape_prior(ball)
    if cluster_shape.get("is_shape_outlier"):
        warnings.append("cluster_shape_outlier")
        shape_reasons = {str(reason) for reason in cluster_shape.get("reasons") or []}
        if shape_reasons.intersection(
            {
                "cluster_ellipse_major_outlier",
                "cluster_ellipse_minor_outlier",
            }
        ):
            warnings.append("cluster_ellipse_size_outlier")
        if "cluster_ellipse_angle_outlier" in shape_reasons:
            warnings.append("cluster_ellipse_angle_outlier")
        boundary_filter = (
            ball.get("source_boundary_filter")
            or ball.get("source_radial_boundary_filter")
            or {}
        )
        if int(boundary_filter.get("neighbor_ellipse_rejected_count") or 0) > 0:
            warnings.append("neighbor_ellipse_ownership_conflict")
    return list(dict.fromkeys(warnings))


def _cluster_shape_prior(ball: dict[str, Any]) -> dict[str, Any]:
    joint = ball.get("source_joint_cluster_optimization") or {}
    shape = joint.get("cluster_shape_prior") or {}
    return shape if isinstance(shape, dict) else {}


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
    if "cluster_shape_outlier" in warnings:
        confidence -= 0.16
    if "cluster_ellipse_size_outlier" in warnings:
        confidence -= 0.06
    if "cluster_ellipse_angle_outlier" in warnings:
        confidence -= 0.06
    if "neighbor_ellipse_ownership_conflict" in warnings:
        confidence -= 0.04
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
        ("cluster_shape_outlier", 4.0),
        ("cluster_ellipse_size_outlier", 2.0),
        ("cluster_ellipse_angle_outlier", 2.0),
        ("neighbor_ellipse_ownership_conflict", 1.0),
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
            "cluster_shape_outlier",
            "cluster_ellipse_size_outlier",
            "cluster_ellipse_angle_outlier",
            "neighbor_ellipse_ownership_conflict",
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
                "cluster_shape_outlier",
                "cluster_ellipse_size_outlier",
                "cluster_ellipse_angle_outlier",
                "neighbor_ellipse_ownership_conflict",
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
