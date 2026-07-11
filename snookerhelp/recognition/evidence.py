from __future__ import annotations

from pathlib import Path
from typing import Any

from snookerhelp.core.ball_numbering import (
    CANONICAL_BALL_NUMBERING_SCHEME,
    canonical_ball_id_map,
)
from snookerhelp.core.schema import (
    BallEstimate,
    BallEvidence,
    Confidence,
    ImageModel,
    PhysicalModel,
    TableState,
)


V1_TABLE_STATE_SCHEMA = "snookerhelp.table_state.v1"


def table_state_from_legacy_report(
    report: dict[str, Any],
    *,
    report_stem: str | None = None,
) -> TableState:
    """Convert the current prototype report JSON into a v1 TableState.

    This is an adapter, not a new detector. It deliberately translates old
    candidate/model names into v1 product concepts.
    """
    review_evidence = report.get("review_evidence") or {}
    state = report.get("state") or {}
    image_path = str(report.get("image") or state.get("source_image") or "")
    image_name = report_stem or Path(image_path).stem or "unknown"
    evidence_by_id = {
        int(ball["id"]): ball
        for ball in review_evidence.get("balls", [])
        if "id" in ball
    }
    numbering_by_raw_id = canonical_ball_id_map(list(state.get("balls", [])))
    balls: list[BallEstimate] = []
    for state_ball in state.get("balls", []):
        raw_detector_id = int(state_ball["id"])
        evidence_ball = evidence_by_id.get(raw_detector_id, {})
        numbering = numbering_by_raw_id.get(raw_detector_id, {})
        balls.append(
            _ball_estimate_from_legacy(
                state_ball,
                evidence_ball,
                canonical_ball_id=int(
                    numbering.get("canonical_ball_id", raw_detector_id)
                ),
                numbering=numbering,
            )
        )
    balls.sort(key=lambda ball: ball.ball_id)

    return TableState(
        schema_version=V1_TABLE_STATE_SCHEMA,
        image_name=image_name,
        image_path=image_path or None,
        source_image_uri=review_evidence.get("source_image_path"),
        source_size_px=review_evidence.get("source_size_px") or state.get("source_size_px"),
        table_corners_px=review_evidence.get("table_corner_points_px")
        or (state.get("table") or {}).get("corner_points_px", []),
        camera_model=report.get("camera_model") or state.get("camera_model") or {},
        balls=balls,
        summary=report.get("summary") or state.get("detection") or {},
        diagnostics={
            "legacy_report_schema": report.get("schema_version"),
            "legacy_output_directory": report.get("output_directory"),
            "legacy_panel_count": len(report.get("panels") or []),
            "ball_numbering_scheme": CANONICAL_BALL_NUMBERING_SCHEME,
            "raw_to_canonical_ball_ids": {
                str(raw_id): metadata.get("canonical_ball_id")
                for raw_id, metadata in sorted(numbering_by_raw_id.items())
            },
        },
    )


def _ball_estimate_from_legacy(
    state_ball: dict[str, Any],
    evidence_ball: dict[str, Any],
    *,
    canonical_ball_id: int | None = None,
    numbering: dict[str, Any] | None = None,
) -> BallEstimate:
    raw_detector_id = int(state_ball["id"])
    ball_id = int(canonical_ball_id or raw_detector_id)
    label = str(
        evidence_ball.get("label")
        or state_ball.get("color_label")
        or state_ball.get("class")
        or "unknown"
    )
    evidence = _ball_evidence_from_legacy(ball_id, label, state_ball, evidence_ball, numbering or {})
    confidence = _confidence_from_legacy(evidence_ball)
    source_px = (
        evidence_ball.get("source_center_px")
        or state_ball.get("source_refined_center_px")
        or state_ball.get("source_rough_center_px")
    )
    table_xy_mm = (
        evidence_ball.get("source_refined_table_xy_mm")
        or state_ball.get("source_refined_table_xy_mm")
        or state_ball.get("table_xy_mm")
        or evidence_ball.get("table_xy_mm")
    )
    return BallEstimate(
        ball_id=ball_id,
        label=label,
        source_px=_point(source_px),
        table_xy_mm=_point(table_xy_mm),
        radius_px=_float_or_none(state_ball.get("source_radius_px")),
        radius_mm=_float_or_none(state_ball.get("radius_mm")),
        table_xy_by_height_mm=state_ball.get("source_refined_table_xy_by_z_mm")
        or evidence_ball.get("source_refined_table_xy_by_z_mm")
        or {},
        evidence=evidence,
        confidence=confidence,
        warnings=list(evidence_ball.get("warnings") or []),
    )


def _ball_evidence_from_legacy(
    ball_id: int,
    label: str,
    state_ball: dict[str, Any],
    evidence_ball: dict[str, Any],
    numbering: dict[str, Any] | None = None,
) -> BallEvidence:
    image_model = _image_model_from_legacy(evidence_ball)
    physical_model = _physical_model_from_legacy(evidence_ball)
    boundary_points = evidence_ball.get("boundary_points_px") or state_ball.get(
        "source_boundary_points_px",
        [],
    )
    rejected_boundary_points = evidence_ball.get(
        "boundary_rejected_points_px",
    ) or state_ball.get("source_boundary_rejected_points_px", [])
    return BallEvidence(
        ball_id=ball_id,
        label=label,
        crop_uri=evidence_ball.get("source_crop_path"),
        crop_bounds_px=evidence_ball.get("source_crop_bounds_px"),
        rough_center_px=_point(
            evidence_ball.get("rough_center_px") or state_ball.get("source_rough_center_px")
        ),
        boundary_points_px=[_point(point) for point in boundary_points if _point(point)],
        boundary_rejected_points_px=[
            _point(point) for point in rejected_boundary_points if _point(point)
        ],
        boundary_filter=(
            evidence_ball.get("boundary_filter")
            or state_ball.get("source_boundary_filter")
            or {}
        ),
        boundary_source=_display_term(
            evidence_ball.get("boundary_evidence_source")
            or state_ball.get("source_boundary_evidence_source")
        ),
        image_model=image_model,
        physical_model=physical_model,
        color_confidence=_float_or_none(evidence_ball.get("color_confidence")),
        detection_confidence=_float_or_none(evidence_ball.get("detection_confidence")),
        diagnostics={
            "mask": evidence_ball.get("mask_centroid"),
            "nearest_cushion": evidence_ball.get("nearest_cushion"),
            "rough_to_refined_shift_px": evidence_ball.get("rough_to_refined_shift_px"),
            "neighbor_ellipses": evidence_ball.get("neighbor_ellipses_px")
            or state_ball.get("source_neighbor_ellipses_px")
            or [],
            "evidence_maps": evidence_ball.get("evidence_maps")
            or state_ball.get("source_evidence_maps"),
            "final_image_evidence": evidence_ball.get("final_image_evidence")
            or state_ball.get("source_final_center_policy")
            or {},
            "source_boundary_view_score": evidence_ball.get("boundary_view_score")
            or {},
            "rejection_addback_scenarios": evidence_ball.get(
                "rejection_addback_scenarios",
                [],
            ),
            "consensus_reject_refit": evidence_ball.get("consensus_reject_refit"),
            "arc_combination_refit": evidence_ball.get("arc_combination_refit"),
            "scene_constraints": {
                "global_cluster_solution": evidence_ball.get(
                    "global_cluster_solution"
                )
                or state_ball.get("source_global_cluster_solution")
                or {},
                "joint_cluster": evidence_ball.get("joint_cluster_optimization")
                or state_ball.get("source_joint_cluster_optimization")
                or {},
            },
            "ball_numbering": numbering or {},
        },
    )


def _image_model_from_legacy(evidence_ball: dict[str, Any]) -> ImageModel | None:
    ellipse = evidence_ball.get("ellipse_fit") or evidence_ball.get("radial_ellipse_fit")
    if not ellipse:
        return None
    return ImageModel(
        model_type="edge_ellipse",
        source=_display_term(ellipse.get("source") or "edge_boundary"),
        center_px=_point(ellipse.get("center_px")),
        major_axis_px=_float_or_none(ellipse.get("major_axis_px")),
        minor_axis_px=_float_or_none(ellipse.get("minor_axis_px")),
        angle_deg=_float_or_none(ellipse.get("angle_deg")),
        axis_ratio=_float_or_none(ellipse.get("axis_ratio")),
        point_count=len(
            evidence_ball.get("final_boundary_points_px")
            or evidence_ball.get("boundary_points_px")
            or []
        ),
        quality=_image_quality(evidence_ball),
    )


def _physical_model_from_legacy(evidence_ball: dict[str, Any]) -> PhysicalModel | None:
    projection = evidence_ball.get("sphere_projection") or {}
    if not projection:
        return None
    score = projection.get("observed_fit_score") or {}
    grade = ((evidence_ball.get("physics_c_only_model_decision") or {}).get("sphere_grade") or {})
    ellipse = projection.get("ellipse_fit") or {}
    optimization = projection.get("optimization") or evidence_ball.get(
        "physical_optimization",
        {},
    )
    return PhysicalModel(
        model_type="projected_sphere",
        camera_model=_display_term(projection.get("camera_model")),
        approximate=bool(projection.get("approximate", True)),
        status=str(projection.get("status") or "unknown"),
        projection_mode=str(projection.get("projection_mode") or "forward"),
        projected_center_px=_point(projection.get("projected_center_px") or ellipse.get("center_px")),
        projected_outline_px=projection.get("contour_points_px") or [],
        residual_px=_float_or_none(score.get("rms_error_px")),
        residual_grade=str(grade.get("level") or "unknown"),
        observed_source=_display_term(score.get("source")),
        z_mm=_float_or_none((projection.get("center_xyz_mm") or [None, None, None])[2]),
        optimization=optimization if isinstance(optimization, dict) else {},
        explanation=[
            str(item)
            for item in (projection.get("explanation") or [])
            if item is not None
        ],
    )


def _confidence_from_legacy(evidence_ball: dict[str, Any]) -> Confidence:
    score = _float_or_none(evidence_ball.get("review_confidence"))
    c_only = evidence_ball.get("physics_c_only_model_decision") or {}
    level = str(c_only.get("table_position_trust") or "unknown")
    if level not in {"high", "medium", "low", "needs_review", "unknown"}:
        level = "unknown"
    image_confidence = _level_score(
        ((c_only.get("candidate_c_grade") or {}).get("level"))
        or ((evidence_ball.get("physics_first_model_decision") or {}).get("object_evidence_grade") or {}).get("level")
    )
    physical_confidence = _level_score((c_only.get("sphere_grade") or {}).get("level"))
    scene_confidence = _scene_constraint_confidence(evidence_ball.get("warnings") or [])
    return Confidence(
        score=score,
        level=level,  # type: ignore[arg-type]
        reasons=_product_reasons(c_only.get("reasons") or evidence_ball.get("warnings") or []),
        method="physical_model_plus_image_evidence",
        components={
            "legacy_score": evidence_ball.get("legacy_review_confidence"),
            "physical_image_score": evidence_ball.get("physics_first_review_confidence"),
            "physical_radial_edge_score": evidence_ball.get(
                "physics_c_only_review_confidence",
            ),
            "image_evidence_confidence": image_confidence,
            "physical_model_confidence": physical_confidence,
            "scene_constraint_confidence": scene_confidence,
            "final_confidence": score,
        },
    )


def _level_score(level: Any) -> float | None:
    if level is None:
        return None
    return {
        "high": 0.85,
        "medium": 0.58,
        "low": 0.25,
        "unavailable": 0.0,
        "unknown": None,
    }.get(str(level), None)


def _scene_constraint_confidence(warnings: list[Any]) -> float:
    warning_set = {str(warning) for warning in warnings}
    score = 0.85
    if "duplicate_detection" in warning_set:
        score -= 0.25
    if "near_pocket" in warning_set:
        score -= 0.10
    if "near_cushion" in warning_set:
        score -= 0.07
    if "fallback_suspicious" in warning_set or "no_points" in warning_set:
        score -= 0.25
    if "cluster_shape_outlier" in warning_set:
        score -= 0.18
    if "cluster_ellipse_size_outlier" in warning_set:
        score -= 0.08
    if "cluster_ellipse_angle_outlier" in warning_set:
        score -= 0.08
    if "neighbor_ellipse_ownership_conflict" in warning_set:
        score -= 0.05
    return round(max(0.05, score), 4)


def _image_quality(evidence_ball: dict[str, Any]) -> str:
    c_only = evidence_ball.get("physics_c_only_model_decision") or {}
    grade = c_only.get("candidate_c_grade") or {}
    return str(grade.get("level") or "unknown")


def _product_reasons(reasons: list[Any]) -> list[str]:
    replacements = {
        "candidate_c": "image_evidence",
        "candidate_d": "physical_model",
        "candidate_b": "mask_diagnostic",
        "candidate_a": "circle_diagnostic",
        "sphere": "physical_model",
        "radial": "edge",
        "hough": "rough detector",
        "manual_homography": "table-corner bootstrap",
        "source_refined_center_px": "final source-pixel estimate",
    }
    cleaned: list[str] = []
    for reason in reasons:
        text = str(reason)
        for old, new in replacements.items():
            text = text.replace(old, new)
        cleaned.append(text)
    return cleaned


def _display_term(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    replacements = {
        "radial_boundary_filtered": "filtered edge boundary",
        "radial_edge_filtered": "filtered edge boundary",
        "radial_boundary": "edge boundary",
        "radial_edge": "edge boundary",
        "source_boundary_points_px": "source boundary pixels",
        "source_mask_contour_points_px": "segmentation contour pixels",
        "source_ellipse_fit": "source edge ellipse",
        "mask_contour": "segmentation contour",
        "circle_radial": "circle diagnostic",
        "fallback_radial": "fallback estimate",
        "manual_homography": "table-corner bootstrap",
        "approximate_pinhole_from_corners": "approximate pinhole from table corners",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _point(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        return [round(float(value[0]), 4), round(float(value[1]), 4)]
    except (TypeError, ValueError, IndexError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
