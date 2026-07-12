from __future__ import annotations

from typing import Any

import numpy as np

from snookerhelp.recognition.evidence_maps import BallEvidenceMaps, sample_map_at_points
from snookerhelp.recognition.sphere_projection import (
    project_sphere_silhouette,
    score_observed_points_against_silhouette,
)


def optimize_ball_xy_from_sphere_projection(
    *,
    initial_xy_mm: list[float] | tuple[float, float] | np.ndarray,
    camera_model: Any,
    observed_boundary_points_px: list[list[float]] | tuple[Any, ...] | np.ndarray,
    evidence_maps: BallEvidenceMaps | None,
    neighbors: list[dict[str, Any]] | None,
    cushion_context: dict[str, Any] | None,
    ball_radius_mm: float = 26.25,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Optimize table X/Y by fitting a projected sphere to image evidence.

    This is a bounded local optimizer, not a global detector. It starts from
    the current source-center ray/height-plane estimate and asks whether a
    nearby physical sphere projection better explains the local boundary
    evidence.
    """

    cfg = settings or {}
    if not bool(cfg.get("enabled", True)):
        return _unavailable("disabled", "physical optimization is disabled")
    try:
        initial = np.asarray(initial_xy_mm, dtype=np.float64).reshape(2)
    except (TypeError, ValueError):
        return _unavailable("unavailable", "initial XY is unavailable")
    observed = np.asarray(observed_boundary_points_px or [], dtype=np.float64).reshape(-1, 2)
    if len(observed) < int(cfg.get("minimum_observed_points", 18)) and evidence_maps is None:
        return _unavailable("unavailable", "not enough image evidence for optimization")

    z_mm = float(cfg.get("z_mm", ball_radius_mm))
    radius_mm = float(ball_radius_mm)
    search_radius = float(cfg.get("search_radius_mm", 18.0))
    coarse_step = float(cfg.get("coarse_step_mm", 6.0))
    refinement_steps = [coarse_step, coarse_step / 2.0, coarse_step / 4.0]
    if search_radius <= 0.0 or coarse_step <= 0.0:
        return _unavailable("unavailable", "invalid search radius or step")

    initial_projection = _candidate_projection(camera_model, initial, z_mm, radius_mm)
    if initial_projection.get("status") != "predicted":
        return _unavailable("unavailable", initial_projection.get("reason", "sphere projection unavailable"))
    initial_score = _score_candidate(
        xy=initial,
        projection=initial_projection,
        initial_xy=initial,
        observed=observed,
        evidence_maps=evidence_maps,
        neighbors=neighbors or [],
        cushion_context=cushion_context or {},
        ball_radius_mm=radius_mm,
        settings=cfg,
    )

    best_xy = initial.copy()
    best_projection = initial_projection
    best_score = initial_score
    for step in refinement_steps:
        search_center = best_xy.copy()
        offsets = np.arange(-search_radius, search_radius + 1e-6, step, dtype=np.float64)
        for dx in offsets:
            for dy in offsets:
                candidate_xy = search_center + np.array([dx, dy], dtype=np.float64)
                if np.linalg.norm(candidate_xy - initial) > search_radius:
                    continue
                projection = _candidate_projection(camera_model, candidate_xy, z_mm, radius_mm)
                if projection.get("status") != "predicted":
                    continue
                candidate_score = _score_candidate(
                    xy=candidate_xy,
                    projection=projection,
                    initial_xy=initial,
                    observed=observed,
                    evidence_maps=evidence_maps,
                    neighbors=neighbors or [],
                    cushion_context=cushion_context or {},
                    ball_radius_mm=radius_mm,
                    settings=cfg,
                )
                if candidate_score["objective"] < best_score["objective"]:
                    best_xy = candidate_xy
                    best_projection = projection
                    best_score = candidate_score
        search_radius = max(step, search_radius * 0.45)

    movement = float(np.linalg.norm(best_xy - initial))
    improvement = float(initial_score["objective"] - best_score["objective"])
    success = improvement > float(cfg.get("minimum_objective_improvement", 0.015)) and movement <= float(
        cfg.get("maximum_allowed_movement_mm", 28.0)
    )
    residual_px = best_score.get("rms_error_px")
    confidence = _confidence_from_scores(
        success=success,
        improvement=improvement,
        movement_mm=movement,
        residual_px=residual_px,
        evidence_mean=best_score.get("evidence_mean", 0.0),
        approximate=not bool(getattr(camera_model, "is_calibrated", False)),
    )
    reasons = list(best_score.get("reasons", []))
    if success:
        reasons.append("optimized_projection_improved_local_objective")
    else:
        reasons.append("optimization_did_not_improve_enough")
    if not bool(getattr(camera_model, "is_calibrated", False)):
        reasons.append("approximate_camera_model_caps_trust")

    return {
        "status": "optimized" if success else "no_better_solution",
        "enabled": True,
        "success": bool(success),
        "initial_xy_mm": _round_array(initial),
        "optimized_xy_mm": _round_array(best_xy),
        "optimized_source_center_px": _round_array(
            np.asarray(best_projection.get("projected_center_px") or [], dtype=np.float64)
        ),
        "optimized_sphere_curve_px": best_projection.get("contour_points_px", []),
        "optimized_ellipse_fit": best_projection.get("ellipse_fit"),
        "initial_objective": round(float(initial_score["objective"]), 6),
        "optimized_objective": round(float(best_score["objective"]), 6),
        "objective_improvement": round(improvement, 6),
        "initial_residual_px": initial_score.get("rms_error_px"),
        "residual_px": None if residual_px is None else round(float(residual_px), 4),
        "initial_evidence_mean": initial_score.get("evidence_mean"),
        "movement_from_initial_mm": round(movement, 4),
        "confidence": confidence,
        "reasons": sorted(set(reasons)),
        "score_terms": {
            key: value
            for key, value in best_score.items()
            if key not in {"projection", "reasons"}
        },
        "note": (
            "Physical optimization searches table X/Y near the current estimate, "
            "projects a known-radius sphere for each candidate, and scores it "
            "against accepted source-boundary evidence, diagnostic evidence maps, "
            "and a movement prior."
        ),
    }


def _candidate_projection(
    camera_model: Any,
    xy_mm: np.ndarray,
    z_mm: float,
    radius_mm: float,
) -> dict[str, Any]:
    return project_sphere_silhouette(
        camera_model,
        [float(xy_mm[0]), float(xy_mm[1]), float(z_mm)],
        radius_mm,
        sample_count=144,
    )


def _score_candidate(
    *,
    xy: np.ndarray,
    projection: dict[str, Any],
    initial_xy: np.ndarray,
    observed: np.ndarray,
    evidence_maps: BallEvidenceMaps | None,
    neighbors: list[dict[str, Any]],
    cushion_context: dict[str, Any],
    ball_radius_mm: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    contour = projection.get("contour_points_px") or []
    score = score_observed_points_against_silhouette(observed, projection)
    rms = None
    if score and score.get("status") == "scored":
        rms = float(score.get("rms_error_px") or 0.0)
        point_term = min(2.5, rms / float(settings.get("residual_normalizer_px", 8.0)))
    else:
        point_term = float(settings.get("missing_point_term", 0.72))
    visible_contour, occlusion = _visible_contour_points(
        projection=projection,
        contour=contour,
        xy=xy,
        neighbors=neighbors,
        ball_radius_mm=ball_radius_mm,
    )
    evidence_values = sample_map_at_points(
        evidence_maps,
        visible_contour,
        "combined_boundary_score",
    )
    evidence_mean = float(np.mean(evidence_values)) if evidence_values.size else 0.0
    evidence_term = 1.0 - evidence_mean
    movement = float(np.linalg.norm(xy - initial_xy))
    prior_term = movement / max(1e-6, float(settings.get("movement_prior_mm", 18.0)))
    neighbor_term, neighbor_reasons = _neighbor_penalty(xy, neighbors, ball_radius_mm)
    cushion_term, cushion_reasons = _cushion_penalty(xy, cushion_context, ball_radius_mm)
    objective = (
        float(settings.get("point_residual_weight", 0.42)) * point_term
        + float(settings.get("evidence_map_weight", 0.32)) * evidence_term
        + float(settings.get("movement_prior_weight", 0.18)) * prior_term
        + float(settings.get("neighbor_penalty_weight", 0.06)) * neighbor_term
        + float(settings.get("cushion_penalty_weight", 0.02)) * cushion_term
    )
    return {
        "objective": round(float(objective), 6),
        "point_term": round(float(point_term), 6),
        "evidence_term": round(float(evidence_term), 6),
        "movement_prior_term": round(float(prior_term), 6),
        "neighbor_penalty_term": round(float(neighbor_term), 6),
        "cushion_penalty_term": round(float(cushion_term), 6),
        "evidence_mean": round(float(evidence_mean), 6),
        "rms_error_px": None if rms is None else round(float(rms), 6),
        "observed_point_count": int(len(observed)),
        "occlusion": occlusion,
        "reasons": neighbor_reasons + cushion_reasons,
    }


def _visible_contour_points(
    *,
    projection: dict[str, Any],
    contour: list[list[float]] | tuple[Any, ...],
    xy: np.ndarray,
    neighbors: list[dict[str, Any]],
    ball_radius_mm: float,
) -> tuple[list[list[float]], dict[str, Any]]:
    points = np.asarray(contour or [], dtype=np.float64).reshape(-1, 2)
    if len(points) < 3 or not neighbors:
        return points.tolist(), {
            "status": "not_applicable",
            "occluded_arc_fraction": 0.0,
            "occluding_neighbor_count": 0,
        }
    center = np.asarray(projection.get("projected_center_px") or [], dtype=np.float64).reshape(-1)
    if center.size < 2:
        return points.tolist(), {
            "status": "unavailable",
            "occluded_arc_fraction": 0.0,
            "occluding_neighbor_count": 0,
        }
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    visible = np.ones(len(points), dtype=bool)
    occluders: list[dict[str, Any]] = []
    projected_radius_px = float(np.median(np.linalg.norm(points - center[:2][None, :], axis=1)))
    for neighbor in neighbors:
        neighbor_xy = neighbor.get("xy_mm")
        neighbor_px = neighbor.get("source_px")
        if neighbor_xy is None or neighbor_px is None:
            continue
        neighbor_xy_array = np.asarray(neighbor_xy, dtype=np.float64).reshape(2)
        table_distance = float(np.linalg.norm(neighbor_xy_array - xy))
        if table_distance > ball_radius_mm * 2.45:
            continue
        neighbor_source = np.asarray(neighbor_px, dtype=np.float64).reshape(2)
        delta_px = neighbor_source - center[:2]
        source_distance = float(np.linalg.norm(delta_px))
        if source_distance <= 1e-6:
            continue
        direction = float(np.arctan2(delta_px[1], delta_px[0]))
        half_angle = float(
            np.clip(
                np.arcsin(np.clip(projected_radius_px / max(source_distance, 1.0), 0.0, 0.95))
                * 1.25,
                np.deg2rad(10.0),
                np.deg2rad(62.0),
            )
        )
        delta_angle = np.abs((angles - direction + np.pi) % (2.0 * np.pi) - np.pi)
        hidden = delta_angle <= half_angle
        if np.any(hidden):
            visible &= ~hidden
            occluders.append(
                {
                    "id": neighbor.get("id"),
                    "label": neighbor.get("label"),
                    "table_distance_mm": round(table_distance, 4),
                    "source_distance_px": round(source_distance, 4),
                    "arc_half_angle_deg": round(float(np.rad2deg(half_angle)), 4),
                }
            )
    if not occluders:
        return points.tolist(), {
            "status": "no_close_neighbors",
            "occluded_arc_fraction": 0.0,
            "occluding_neighbor_count": 0,
        }
    # Keep a minimum amount of evidence. If neighbors would hide nearly the
    # whole curve, the scene is too ambiguous for this simple local optimizer.
    visible_fraction = float(np.mean(visible))
    if visible_fraction < 0.28:
        visible = np.ones(len(points), dtype=bool)
        status = "disabled_too_much_occlusion"
    else:
        status = "applied"
    return points[visible].tolist(), {
        "status": status,
        "occluded_arc_fraction": round(float(1.0 - np.mean(visible)), 4),
        "occluding_neighbor_count": len(occluders),
        "occluding_neighbors": occluders[:6],
    }


def _neighbor_penalty(
    xy: np.ndarray,
    neighbors: list[dict[str, Any]],
    radius_mm: float,
) -> tuple[float, list[str]]:
    if not neighbors:
        return 0.0, []
    penalty = 0.0
    reasons: list[str] = []
    expected_touch = 2.0 * radius_mm
    for neighbor in neighbors:
        point = neighbor.get("xy_mm")
        if point is None:
            continue
        neighbor_xy = np.asarray(point, dtype=np.float64).reshape(2)
        distance = float(np.linalg.norm(xy - neighbor_xy))
        if distance < expected_touch * 0.72:
            penalty += (expected_touch * 0.72 - distance) / expected_touch
            reasons.append("neighbor_collision_penalty")
        elif distance < expected_touch * 0.95:
            penalty += 0.12 * (expected_touch * 0.95 - distance) / expected_touch
            reasons.append("neighbor_touching_constraint")
    return float(penalty), sorted(set(reasons))


def _cushion_penalty(
    xy: np.ndarray,
    cushion_context: dict[str, Any],
    radius_mm: float,
) -> tuple[float, list[str]]:
    length = cushion_context.get("length_mm")
    width = cushion_context.get("width_mm")
    if length is None or width is None:
        return 0.0, []
    x, y = float(xy[0]), float(xy[1])
    penalty = 0.0
    if x < radius_mm:
        penalty += (radius_mm - x) / radius_mm
    if y < radius_mm:
        penalty += (radius_mm - y) / radius_mm
    if x > float(length) - radius_mm:
        penalty += (x - (float(length) - radius_mm)) / radius_mm
    if y > float(width) - radius_mm:
        penalty += (y - (float(width) - radius_mm)) / radius_mm
    return float(max(0.0, penalty)), ["cushion_radius_constraint"] if penalty > 0.0 else []


def _confidence_from_scores(
    *,
    success: bool,
    improvement: float,
    movement_mm: float,
    residual_px: float | None,
    evidence_mean: float,
    approximate: bool,
) -> dict[str, Any]:
    if not success:
        level = "low"
        score = 0.25
    else:
        residual_score = 0.55 if residual_px is None else float(np.clip(1.0 - residual_px / 12.0, 0.0, 1.0))
        movement_score = float(np.clip(1.0 - movement_mm / 28.0, 0.0, 1.0))
        score = 0.45 * residual_score + 0.35 * float(evidence_mean) + 0.20 * movement_score
        if approximate:
            score = min(score, 0.74)
        level = "high" if score >= 0.78 else "medium" if score >= 0.45 else "low"
    return {
        "score": round(float(score), 4),
        "level": level,
        "approximate_camera_cap_applied": bool(approximate),
        "improvement": round(float(improvement), 6),
    }


def _unavailable(status: str, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "enabled": status != "disabled",
        "success": False,
        "reason": reason,
        "reasons": [reason],
    }


def _round_array(values: np.ndarray) -> list[float]:
    return [round(float(value), 4) for value in np.asarray(values).reshape(-1)]
