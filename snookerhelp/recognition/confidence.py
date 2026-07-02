from __future__ import annotations

from typing import Any

import numpy as np


def physics_first_score(
    *,
    ball: dict[str, Any],
    evidence_agreement: dict[str, Any] | None,
    consensus_ellipse: dict[str, Any] | None,
    current_decision: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Score a ball using physics-first evidence.

    This deliberately does not delete the legacy circle-baseline path. It gives
    the review UI and benchmark tooling a second, explicit decision model:

    1. Candidate D: projected known-radius sphere from the camera model.
    2. Candidate C/B: observed ellipse and mask/radial evidence agreement.
    3. Candidate A: circle fit only as a measured baseline/fallback signal.

    With the current approximate pinhole model, the maximum trust is capped.
    Once ChArUco intrinsics/extrinsics are available the same function can
    promote Candidate D to high trust.
    """
    sphere_projection = ball.get("source_sphere_projection") or {}
    radius_px = float(ball.get("source_radius_px") or 40.0)
    sphere_grade = _sphere_grade(sphere_projection, radius_px)
    object_grade = _object_evidence_grade(
        ball=ball,
        evidence_agreement=evidence_agreement or {},
        consensus_ellipse=consensus_ellipse,
    )
    approximate = bool(sphere_projection.get("approximate", True))
    current_status = str(current_decision.get("status") or "unknown")
    current_model = str(current_decision.get("selected_model") or "unknown")

    final_center_source = (
        "source_refined_center_px"
        if ball.get("source_refinement_success")
        else "source_rough_center_px"
    )
    final_center = (
        ball.get("source_refined_center_px")
        if ball.get("source_refinement_success")
        else ball.get("source_rough_center_px")
    )

    reasons: list[str] = []
    reasons.extend(object_grade["reasons"])
    reasons.extend(sphere_grade["reasons"])
    if approximate:
        reasons.append("approximate_camera_model")
    for warning in warnings:
        if warning in (
            "near_cushion",
            "near_pocket",
            "fallback_suspicious",
            "no_points",
            "duplicate_detection",
        ):
            reasons.append(warning)

    if sphere_grade["level"] == "unavailable":
        return {
            "status": current_status,
            "selected_model": current_model,
            "final_center_source": final_center_source,
            "final_center_px": final_center,
            "table_position_trust": current_decision.get("table_position_trust", "low"),
            "confidence": None,
            "confidence_delta_vs_current": None,
            "sphere_grade": sphere_grade,
            "object_evidence_grade": object_grade,
            "reasons": sorted(set(reasons + ["physics_unavailable"])),
            "summary": (
                "Physics-first scoring is unavailable because Candidate D could "
                "not be projected for this ball. Legacy decision is shown."
            ),
            "note": "Experimental score. Does not replace detector output yet.",
        }

    status = "review"
    selected_model = "physics_supported_observed_ellipse"
    trust = "medium"
    confidence = 0.45

    if sphere_grade["level"] == "high" and object_grade["level"] == "high":
        selected_model = "physics_sphere_projection"
        status = "accepted"
        trust = "medium" if approximate else "high"
        confidence = 0.78 if approximate else 0.92
        reasons.append("sphere_and_observed_evidence_agree")
    elif sphere_grade["level"] in ("high", "medium") and object_grade["level"] == "high":
        selected_model = "physics_supported_observed_ellipse"
        status = "review" if approximate else "accepted"
        trust = "medium"
        confidence = 0.64 if approximate else 0.82
        reasons.append("observed_evidence_strong_physics_plausible")
    elif sphere_grade["level"] == "high" and object_grade["level"] == "medium":
        selected_model = "physics_sphere_projection_with_partial_observed_support"
        status = "review"
        trust = "medium"
        confidence = 0.58 if approximate else 0.74
        reasons.append("physics_good_observed_evidence_partial")
    elif sphere_grade["level"] == "medium" and object_grade["level"] == "medium":
        selected_model = "observed_ellipse_with_medium_physics_support"
        status = "review"
        trust = "medium"
        confidence = 0.52 if approximate else 0.66
        reasons.append("medium_physics_and_observed_support")
    elif object_grade["level"] == "high" and sphere_grade["level"] == "low":
        selected_model = "observed_ellipse_physics_mismatch"
        status = "review"
        trust = "low"
        confidence = 0.36
        reasons.append("observed_object_good_but_physics_mismatch")
    elif object_grade["level"] == "low" and sphere_grade["level"] in ("high", "medium"):
        selected_model = "physics_projection_weak_observed_support"
        status = "review"
        trust = "low"
        confidence = 0.34
        reasons.append("physics_plausible_but_object_segmentation_weak")
    else:
        selected_model = "low_trust_mixed_evidence"
        status = "fallback" if current_status == "fallback" else "review"
        trust = "low"
        confidence = 0.22
        reasons.append("low_physics_and_observed_support")

    if "near_cushion" in warnings:
        confidence -= 0.04
    if "near_pocket" in warnings:
        confidence -= 0.05
    if "fallback_suspicious" in warnings:
        confidence -= 0.08
    if "no_points" in warnings and object_grade["level"] != "high":
        confidence -= 0.08

    confidence = float(np.clip(confidence, 0.05, 0.95))
    current_confidence = ball.get("review_confidence")
    delta = (
        round(confidence - float(current_confidence), 4)
        if current_confidence is not None
        else None
    )

    return {
        "status": status,
        "selected_model": selected_model,
        "final_center_source": final_center_source,
        "final_center_px": final_center,
        "table_position_trust": trust,
        "confidence": round(confidence, 4),
        "confidence_delta_vs_current": delta,
        "sphere_grade": sphere_grade,
        "object_evidence_grade": object_grade,
        "reasons": sorted(set(reasons)),
        "summary": _summary(
            status=status,
            selected_model=selected_model,
            sphere_grade=sphere_grade,
            object_grade=object_grade,
            approximate=approximate,
        ),
        "note": (
            "Experimental D-first score. Candidate A circle is treated as a "
            "baseline/fallback, while Candidate D plus B/C evidence drives "
            "the trust estimate."
        ),
    }


def combined_confidence(
    legacy_confidence: float,
    *physics_scores: dict[str, Any],
) -> float:
    """Return displayed automatic confidence.

    The review UI should not stay artificially pessimistic when D-first evidence
    is plausible, but it should also keep the legacy low score when physics is
    unavailable or weak.
    """
    displayed = float(legacy_confidence)
    for score in physics_scores:
        physics_confidence = score.get("confidence")
        if physics_confidence is None:
            continue
        if (score.get("sphere_grade") or {}).get("level") == "low":
            continue
        displayed = max(displayed, float(physics_confidence))
    return displayed


def physics_c_only_score(
    *,
    ball: dict[str, Any],
    candidate_c_ellipse: dict[str, Any] | None,
    current_decision: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Experimental score using Candidate D plus Candidate C only.

    This intentionally ignores Candidate B purple-mask agreement. It answers the
    user's current hypothesis: "Trust the physical sphere projection first, and
    use the cream radial/edge ellipse as the only observed support model."
    """
    sphere_projection = ball.get("source_sphere_projection") or {}
    radius_px = float(ball.get("source_radius_px") or 40.0)
    sphere_grade = _sphere_grade(sphere_projection, radius_px)
    c_grade = _candidate_c_grade(ball, candidate_c_ellipse)
    approximate = bool(sphere_projection.get("approximate", True))
    source_success = bool(ball.get("source_refinement_success"))
    final_center_source = "source_refined_center_px" if source_success else "source_rough_center_px"
    final_center = ball.get("source_refined_center_px") if source_success else ball.get("source_rough_center_px")

    reasons: list[str] = []
    reasons.extend(sphere_grade["reasons"])
    reasons.extend(c_grade["reasons"])
    if approximate:
        reasons.append("approximate_camera_model")
    if "near_cushion" in warnings:
        reasons.append("near_cushion")
    if "near_pocket" in warnings:
        reasons.append("near_pocket")

    if sphere_grade["level"] == "unavailable" or c_grade["level"] == "unavailable":
        return {
            "status": current_decision.get("status", "review"),
            "selected_model": "physics_c_only_unavailable",
            "final_center_source": final_center_source,
            "final_center_px": final_center,
            "table_position_trust": current_decision.get("table_position_trust", "low"),
            "confidence": None,
            "confidence_delta_vs_current": None,
            "sphere_grade": sphere_grade,
            "candidate_c_grade": c_grade,
            "reasons": sorted(set(reasons)),
            "summary": "C-only physics score unavailable: Candidate D or Candidate C is missing.",
            "note": "Experimental score using D plus C only; B/purple mask is ignored.",
        }

    status = "review"
    trust = "medium"
    confidence = 0.42
    selected_model = "physics_c_only_observed_ellipse"

    if sphere_grade["level"] == "high" and c_grade["level"] == "high":
        status = "accepted"
        trust = "medium" if approximate else "high"
        confidence = 0.80 if approximate else 0.93
        reasons.append("sphere_and_candidate_c_agree")
    elif sphere_grade["level"] in ("high", "medium") and c_grade["level"] == "high":
        status = "review" if approximate else "accepted"
        trust = "medium"
        confidence = 0.66 if approximate else 0.84
        reasons.append("candidate_c_strong_physics_plausible")
    elif sphere_grade["level"] == "high" and c_grade["level"] == "medium":
        status = "review"
        trust = "medium"
        confidence = 0.60 if approximate else 0.76
        reasons.append("physics_good_candidate_c_partial")
    elif sphere_grade["level"] == "medium" and c_grade["level"] == "medium":
        status = "review"
        trust = "medium"
        confidence = 0.54 if approximate else 0.68
        reasons.append("medium_physics_and_candidate_c_support")
    elif sphere_grade["level"] == "low" and c_grade["level"] in ("high", "medium"):
        status = "review"
        trust = "low"
        confidence = 0.30
        selected_model = "candidate_c_physics_mismatch"
        reasons.append("candidate_c_available_but_sphere_projection_mismatch")
    else:
        status = "fallback" if current_decision.get("status") == "fallback" else "review"
        trust = "low"
        confidence = 0.20
        selected_model = "low_trust_c_only_evidence"
        reasons.append("weak_physics_and_candidate_c_support")

    if "near_pocket" in warnings:
        confidence -= 0.04
    if "fallback_suspicious" in warnings and c_grade["level"] != "high":
        confidence -= 0.05
    confidence = float(np.clip(confidence, 0.05, 0.95))
    legacy_confidence = ball.get("review_confidence")
    return {
        "status": status,
        "selected_model": selected_model,
        "final_center_source": final_center_source,
        "final_center_px": final_center,
        "table_position_trust": trust,
        "confidence": round(confidence, 4),
        "confidence_delta_vs_current": (
            round(confidence - float(legacy_confidence), 4)
            if legacy_confidence is not None
            else None
        ),
        "sphere_grade": sphere_grade,
        "candidate_c_grade": c_grade,
        "reasons": sorted(set(reasons)),
        "summary": (
            f"C-only physics decision is {status}. Candidate D is "
            f"{sphere_grade.get('level')}; Candidate C is {c_grade.get('level')}."
        ),
        "note": "Experimental score using D plus C only; B/purple mask is ignored.",
    }


def _sphere_grade(
    sphere_projection: dict[str, Any],
    radius_px: float,
) -> dict[str, Any]:
    score = sphere_projection.get("observed_fit_score") or {}
    if sphere_projection.get("status") not in {"predicted", "optimized"}:
        return {
            "level": "unavailable",
            "reason": sphere_projection.get("reason", "sphere projection unavailable"),
            "rms_error_px": None,
            "normalized_rms": None,
            "reasons": ["sphere_projection_unavailable"],
        }
    if score.get("status") != "scored":
        return {
            "level": "unavailable",
            "reason": score.get("status", "sphere score unavailable"),
            "rms_error_px": None,
            "normalized_rms": None,
            "reasons": ["sphere_projection_not_scored"],
        }
    rms = float(score.get("rms_error_px") or 999.0)
    normalized = rms / max(1.0, float(radius_px))
    if normalized <= 0.11 or rms <= 4.5:
        level = "high"
        reasons = ["sphere_projection_residual_high_quality"]
    elif normalized <= 0.22 or rms <= 8.5:
        level = "medium"
        reasons = ["sphere_projection_residual_plausible"]
    else:
        level = "low"
        reasons = ["sphere_projection_residual_large"]
    return {
        "level": level,
        "rms_error_px": round(rms, 4),
        "normalized_rms": round(normalized, 4),
        "mean_abs_error_px": score.get("mean_abs_error_px"),
        "p95_abs_error_px": score.get("p95_abs_error_px"),
        "reasons": reasons,
    }


def _object_evidence_grade(
    *,
    ball: dict[str, Any],
    evidence_agreement: dict[str, Any],
    consensus_ellipse: dict[str, Any] | None,
) -> dict[str, Any]:
    status = str(evidence_agreement.get("status") or "unknown")
    radial_count = int(evidence_agreement.get("radial_point_count") or 0)
    mask_count = int(evidence_agreement.get("mask_point_count") or 0)
    ellipse = consensus_ellipse or ball.get("source_ellipse_fit") or ball.get(
        "source_silhouette_ellipse_fit"
    )
    reasons: list[str] = []
    if status == "agreement_high":
        level = "high"
        reasons.append("radial_mask_agreement_high")
    elif status == "agreement_medium":
        level = "medium"
        reasons.append("radial_mask_agreement_medium")
    elif ellipse and mask_count >= 40:
        level = "medium"
        reasons.append("ellipse_or_mask_available")
    else:
        level = "low"
        reasons.append("weak_observed_object_evidence")
    if radial_count <= 0:
        reasons.append("no_radial_points")
    if mask_count <= 0:
        reasons.append("no_mask_points")
    return {
        "level": level,
        "agreement_status": status,
        "radial_point_count": radial_count,
        "mask_point_count": mask_count,
        "has_consensus_ellipse": bool(consensus_ellipse),
        "reasons": sorted(set(reasons)),
    }


def _candidate_c_grade(
    ball: dict[str, Any],
    candidate_c_ellipse: dict[str, Any] | None,
) -> dict[str, Any]:
    point_count = len(ball.get("source_boundary_points_px") or [])
    if not candidate_c_ellipse:
        return {
            "level": "unavailable",
            "source": None,
            "point_count": point_count,
            "axis_ratio": None,
            "reasons": ["candidate_c_missing"],
        }
    axis_ratio = candidate_c_ellipse.get("axis_ratio")
    source = candidate_c_ellipse.get("source", "source_ellipse_fit")
    reasons = ["candidate_c_available", f"candidate_c_source_{source}"]
    if point_count >= 80:
        level = "high"
        reasons.append("candidate_c_many_points")
    elif point_count >= 35:
        level = "medium"
        reasons.append("candidate_c_enough_points")
    else:
        level = "medium"
        reasons.append("candidate_c_low_point_count")
    if axis_ratio is not None and float(axis_ratio) > 1.6:
        level = "medium" if level == "high" else level
        reasons.append("candidate_c_strong_ellipse")
    return {
        "level": level,
        "source": source,
        "point_count": point_count,
        "axis_ratio": round(float(axis_ratio), 4) if axis_ratio is not None else None,
        "reasons": sorted(set(reasons)),
    }


def _summary(
    *,
    status: str,
    selected_model: str,
    sphere_grade: dict[str, Any],
    object_grade: dict[str, Any],
    approximate: bool,
) -> str:
    model_kind = "approximate" if approximate else "calibrated"
    return (
        f"Physics-first decision is {status} using {selected_model}. "
        f"Candidate D is {sphere_grade.get('level')} under the {model_kind} "
        f"camera model; observed B/C evidence is {object_grade.get('level')}."
    )
