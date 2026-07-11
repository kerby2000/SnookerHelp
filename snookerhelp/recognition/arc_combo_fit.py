from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from snookerhelp.recognition.boundary_ownership import ownership_summary_for_points
from snookerhelp.recognition.image_model import fit_ellipse_payload


def arc_combination_refit(
    *,
    points_px: list[Any] | tuple[Any, ...],
    rejected_points_px: list[Any] | tuple[Any, ...] | None,
    filter_stats: dict[str, Any] | None,
    cluster_shape_prior: dict[str, Any] | None,
    neighbor_ellipses: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    include_fixed_shape_candidates: bool = True,
    max_search_groups: int = 10,
    max_fixed_shape_combo_size: int = 4,
) -> dict[str, Any]:
    """Search raw radial boundary arc combinations for a cluster-consistent ellipse.

    The normal boundary filter can reject useful edge arcs in tight clusters
    because neighbouring balls overlap the candidate ellipse. This helper starts
    from pre-filter raw radial samples, splits them into angular/spatial arc
    groups, fits all useful group combinations, and ranks them against the
    same-colour cluster shape prior.
    """

    accepted = _points_array(points_px)
    rejected = _points_array(rejected_points_px)
    stats = filter_stats or {}
    raw = _points_array(stats.get("raw_points_px"))
    if len(raw) < 5:
        raw = _stack_points(accepted, rejected)
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
    groups = _raw_arc_groups(raw, center_px=center, rough_radius_px=rough_radius)
    eligible_groups = [group for group in groups if int(group.get("count") or 0) >= 2]
    if not eligible_groups:
        return {
            "status": "not_computed",
            "reason": "raw points did not form usable arc clusters",
            "raw_count": int(len(raw)),
            "groups": groups,
            "diagnostic_only": True,
        }

    groups_for_search = eligible_groups
    capped = False
    max_search_groups = max(1, int(max_search_groups))
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
    minimum_fixed_shape_points = max(18, minimum_combo_points)
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
            evaluated.append(
                _candidate_payload(
                    group_combo=group_combo,
                    combo_points=combo_points,
                    ellipse=ellipse,
                    prior=prior,
                    raw_count=len(raw),
                    baseline_center=center,
                    neighbor_ellipses=neighbor_ellipses,
                    shape_model="free_ellipse",
                )
            )

            if (
                include_fixed_shape_candidates
                and combo_size >= 2
                and combo_size <= max(1, int(max_fixed_shape_combo_size))
                and len(combo_points) >= minimum_fixed_shape_points
            ):
                fixed_shape_ellipse = _fit_cluster_shape_fixed_ellipse(
                    combo_points,
                    prior=prior,
                    seed_ellipse=ellipse,
                    fallback_center=center,
                )
                if fixed_shape_ellipse is not None:
                    evaluated.append(
                        _candidate_payload(
                            group_combo=group_combo,
                            combo_points=combo_points,
                            ellipse=fixed_shape_ellipse,
                            prior=prior,
                            raw_count=len(raw),
                            baseline_center=center,
                            neighbor_ellipses=neighbor_ellipses,
                            shape_model="cluster_shape_fixed",
                        )
                    )

    baseline_comparison = compare_ellipse_to_cluster_shape(baseline_ellipse, prior)
    baseline_residual = ellipse_rms_residual_px(baseline_ellipse, baseline_points)
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
                "ellipse_fit": review_ellipse_payload(baseline_ellipse),
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
        "fixed_shape_candidates_enabled": bool(include_fixed_shape_candidates),
        "max_fixed_shape_combo_size": int(max_fixed_shape_combo_size),
        "minimum_combo_points": int(minimum_combo_points),
        "minimum_fixed_shape_points": int(minimum_fixed_shape_points),
        "capped": bool(capped),
        "groups": groups,
        "baseline": {
            "ellipse_fit": review_ellipse_payload(baseline_ellipse),
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
                "and point coverage"
            ),
        },
        "top_candidates": ranked[:8],
        "diagnostic_only": True,
    }


def should_promote_arc_combination(refit: dict[str, Any]) -> tuple[bool, list[str]]:
    """Conservative gate for using arc-combo as the final image model."""

    reasons: list[str] = []
    if refit.get("status") != "improved":
        reasons.append(f"status={refit.get('status') or 'missing'}")
    best = refit.get("best") or {}
    comparison = best.get("cluster_shape_comparison") or {}
    shape_score = comparison.get("score")
    residual_px = best.get("ellipse_rms_residual_px")
    point_fraction = best.get("point_fraction")
    group_count = int(best.get("group_count") or 0)
    point_count = int(best.get("point_count") or 0)
    improvement = best.get("shape_score_improvement")
    shape_model = str(best.get("shape_model") or "")

    shape_score_value = None if shape_score is None else float(shape_score)
    residual_value = None if residual_px is None else float(residual_px)
    point_fraction_value = None if point_fraction is None else float(point_fraction)
    improvement_value = None if improvement is None else float(improvement)
    is_shape_outlier = bool(comparison.get("is_shape_outlier"))
    ownership = best.get("boundary_ownership") or {}
    neighbor_owned_fraction = ownership.get("neighbor_owned_fraction")
    target_owned_fraction = ownership.get("target_owned_fraction")

    # Two-level gate:
    # - high shape score passes with the normal residual/coverage checks.
    # - moderate shape score can pass only when the fit is not a cluster-shape
    #   outlier and the raw-arc combination is a clear low-residual improvement.
    # - fixed cluster-shape candidates may pass with less arc coverage because
    #   the shared shape prior supplies the missing degrees of freedom.
    high_shape_score = shape_score_value is not None and shape_score_value >= 82.0
    moderate_shape_score = shape_score_value is not None and shape_score_value >= 72.0
    fixed_shape_candidate = shape_model == "cluster_shape_fixed"
    strong_non_outlier_improvement = (
        moderate_shape_score
        and not is_shape_outlier
        and residual_value is not None
        and residual_value <= 1.6
        and point_count >= 36
        and group_count >= 2
        and improvement_value is not None
        and improvement_value >= 20.0
    )
    fixed_shape_physical_improvement = (
        fixed_shape_candidate
        and moderate_shape_score
        and not is_shape_outlier
        and residual_value is not None
        and residual_value <= 2.8
        and point_count >= 18
        and group_count >= 2
        and improvement_value is not None
        and improvement_value >= 8.0
    )
    if not (
        high_shape_score
        or strong_non_outlier_improvement
        or fixed_shape_physical_improvement
    ):
        reasons.append(f"shape_score={shape_score}")
    max_residual_px = 2.8 if fixed_shape_physical_improvement else 2.5
    min_point_fraction = 0.14 if fixed_shape_physical_improvement else 0.18
    if residual_px is None or float(residual_px) > max_residual_px:
        reasons.append(f"rms={residual_px}")
    if point_fraction is None or float(point_fraction) < min_point_fraction:
        reasons.append(f"point_fraction={point_fraction}")
    if group_count < 2:
        reasons.append(f"group_count={group_count}")
    if point_count < 18:
        reasons.append(f"point_count={point_count}")
    if improvement is None or float(improvement) < 8.0:
        reasons.append(f"shape_improvement={improvement}")
    if is_shape_outlier:
        reasons.append("best_is_shape_outlier")
    if (
        neighbor_owned_fraction is not None
        and target_owned_fraction is not None
        and float(neighbor_owned_fraction) > 0.45
        and float(target_owned_fraction) < 0.45
    ):
        reasons.append(
            f"neighbor_owned_fraction={neighbor_owned_fraction}"
        )
    return not reasons, reasons


def compare_ellipse_to_cluster_shape(
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
        "score": round(float(score), 2),
        "major_scale": round(float(major_scale), 4),
        "minor_scale": round(float(minor_scale), 4),
        "angle_delta_deg": round(float(angle_delta), 4),
        "is_shape_outlier": bool(reasons),
        "reasons": reasons or ["shape_prior_match"],
    }


def review_ellipse_payload(ellipse: dict[str, Any] | None) -> dict[str, Any] | None:
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


def _candidate_payload(
    *,
    group_combo: tuple[dict[str, Any], ...],
    combo_points: np.ndarray,
    ellipse: dict[str, Any],
    prior: dict[str, Any],
    raw_count: int,
    baseline_center: np.ndarray,
    neighbor_ellipses: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    shape_model: str,
) -> dict[str, Any]:
    comparison = compare_ellipse_to_cluster_shape(ellipse, prior)
    residual_px = ellipse_rms_residual_px(ellipse, combo_points)
    ownership = ownership_summary_for_points(
        combo_points,
        target_ellipse=ellipse,
        neighbor_ellipses=neighbor_ellipses,
    )
    shape_score = (
        50.0
        if comparison.get("score") is None
        else float(comparison["score"])
    )
    residual_score = max(0.0, 100.0 - min(100.0, residual_px * 24.0))
    point_fraction = min(1.0, float(len(combo_points)) / max(1.0, float(raw_count)))
    point_score = min(100.0, point_fraction * 155.0)
    multi_arc_score = min(100.0, float(len(group_combo)) * 28.0)
    neighbor_owned_fraction = float(ownership.get("neighbor_owned_fraction") or 0.0)
    target_owned_fraction = float(ownership.get("target_owned_fraction") or 0.0)
    ownership_score = max(
        0.0,
        min(100.0, 100.0 * target_owned_fraction - 70.0 * neighbor_owned_fraction),
    )
    center = _ellipse_center_or_mean(ellipse, combo_points)
    center_shift_px = float(np.linalg.norm(center - baseline_center.reshape(2)))
    center_shift_score = max(0.0, 100.0 - max(0.0, center_shift_px - 8.0) * 4.0)
    # Shared-shape candidates deliberately trade a little raw residual for
    # physical plausibility: in dense clusters, partial arcs can make the free
    # ellipse grow or rotate toward a neighbouring reflection. The center-shift
    # term prevents a low-residual partial arc from jumping to another ball.
    ranking_score = (
        0.40 * shape_score
        + 0.30 * residual_score
        + 0.10 * point_score
        + 0.05 * multi_arc_score
        + 0.08 * ownership_score
        + 0.07 * center_shift_score
    )
    if shape_model == "cluster_shape_fixed":
        # In dense clusters the physically constrained center-only fit is the
        # intended corrective model. Give it a real preference only when it is
        # plausible on image residual, coverage, and cluster shape. This keeps
        # single-arc guesses out while letting multi-arc combinations override
        # free ellipses that became too large or rotated toward a neighbour.
        fixed_shape_bonus = 0.0
        if not bool(comparison.get("is_shape_outlier")):
            fixed_shape_bonus += 4.0
        if residual_px <= 2.5:
            fixed_shape_bonus += 4.0
        if point_fraction >= 0.18:
            fixed_shape_bonus += 2.0
        if len(group_combo) >= 2:
            fixed_shape_bonus += 2.0
        ranking_score += fixed_shape_bonus
    if residual_px > 2.5:
        ranking_score -= min(35.0, (float(residual_px) - 2.5) * 15.0)
    return {
        "shape_model": shape_model,
        "group_ids": [int(group["group_id"]) for group in group_combo],
        "group_count": int(len(group_combo)),
        "point_count": int(len(combo_points)),
        "selected_points_px": round_points(combo_points),
        "ellipse_fit": review_ellipse_payload(ellipse),
        "cluster_shape_comparison": comparison,
        "ellipse_rms_residual_px": round(float(residual_px), 4),
        "point_fraction": round(float(point_fraction), 4),
        "center_shift_px": round(float(center_shift_px), 4),
        "ranking_score": round(float(ranking_score), 4),
        "score_components": {
            "shape_score": round(float(shape_score), 4),
            "residual_score": round(float(residual_score), 4),
            "point_score": round(float(point_score), 4),
            "multi_arc_score": round(float(multi_arc_score), 4),
            "ownership_score": round(float(ownership_score), 4),
            "center_shift_score": round(float(center_shift_score), 4),
        },
        "boundary_ownership": ownership,
    }


def _fit_cluster_shape_fixed_ellipse(
    points: np.ndarray,
    *,
    prior: dict[str, Any],
    seed_ellipse: dict[str, Any] | None,
    fallback_center: np.ndarray,
) -> dict[str, Any] | None:
    """Fit only the center while keeping the shared cluster shape fixed.

    This is the practical version of the graph proposal's shared local
    projection prior. It prevents a partial interior arc from becoming a
    physically impossible large/rotated ellipse, while still letting the image
    evidence choose the center.
    """

    try:
        major = float(prior.get("consensus_major_axis_px"))
        minor = float(prior.get("consensus_minor_axis_px"))
        angle = float(prior.get("consensus_angle_deg")) % 180.0
    except (TypeError, ValueError):
        return None
    if major <= 4.0 or minor <= 4.0 or len(points) < 5:
        return None

    candidates: list[np.ndarray] = []
    linear_center = _linear_fixed_shape_center(points, major, minor, angle)
    if linear_center is not None:
        candidates.append(linear_center)
    if seed_ellipse is not None:
        candidates.append(_ellipse_center_or_mean(seed_ellipse, points))
    candidates.append(np.asarray(fallback_center, dtype=np.float64).reshape(2))
    candidates.append(np.mean(np.asarray(points, dtype=np.float64).reshape(-1, 2), axis=0))

    best_center: np.ndarray | None = None
    best_residual = float("inf")
    for candidate in candidates:
        if not np.all(np.isfinite(candidate)):
            continue
        center, residual = _refine_fixed_shape_center(
            candidate,
            points,
            major=major,
            minor=minor,
            angle=angle,
        )
        if residual < best_residual:
            best_center = center
            best_residual = residual
    if best_center is None:
        return None
    return {
        "status": "candidate",
        "center_px": [float(best_center[0]), float(best_center[1])],
        "major_axis_px": float(major),
        "minor_axis_px": float(minor),
        "angle_deg": float(angle),
        "axis_ratio": float(major / max(1e-6, minor)),
        "source": "raw_arc_cluster_combination_shared_shape",
    }


def _linear_fixed_shape_center(
    points: np.ndarray,
    major: float,
    minor: float,
    angle: float,
) -> np.ndarray | None:
    try:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if len(pts) < 5:
        return None
    theta = np.deg2rad(float(angle))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    u = pts[:, 0] * cos_t + pts[:, 1] * sin_t
    v = -pts[:, 0] * sin_t + pts[:, 1] * cos_t
    rx = max(float(major) / 2.0, 1.0)
    ry = max(float(minor) / 2.0, 1.0)
    matrix = np.column_stack(
        [
            -2.0 * u / (rx * rx),
            -2.0 * v / (ry * ry),
            np.ones(len(pts), dtype=np.float64),
        ]
    )
    rhs = -(u * u / (rx * rx) + v * v / (ry * ry) - 1.0)
    try:
        solution = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    center_u = float(solution[0])
    center_v = float(solution[1])
    return np.asarray(
        [
            center_u * cos_t - center_v * sin_t,
            center_u * sin_t + center_v * cos_t,
        ],
        dtype=np.float64,
    )


def _refine_fixed_shape_center(
    start_center: np.ndarray,
    points: np.ndarray,
    *,
    major: float,
    minor: float,
    angle: float,
) -> tuple[np.ndarray, float]:
    center = np.asarray(start_center, dtype=np.float64).reshape(2).copy()
    best = ellipse_rms_residual_px(
        {
            "center_px": [float(center[0]), float(center[1])],
            "major_axis_px": float(major),
            "minor_axis_px": float(minor),
            "angle_deg": float(angle),
        },
        points,
    )
    step = max(2.5, min(12.0, (float(major) + float(minor)) / 16.0))
    offsets = [
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
    ]
    for _ in range(14):
        improved = False
        for dx, dy in offsets:
            trial = center + np.asarray([dx * step, dy * step], dtype=np.float64)
            residual = ellipse_rms_residual_px(
                {
                    "center_px": [float(trial[0]), float(trial[1])],
                    "major_axis_px": float(major),
                    "minor_axis_px": float(minor),
                    "angle_deg": float(angle),
                },
                points,
            )
            if residual < best:
                center = trial
                best = residual
                improved = True
        if not improved:
            step *= 0.5
    return center, float(best)


def ellipse_rms_residual_px(
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


def round_points(points: np.ndarray) -> list[list[float]]:
    return [
        [round(float(point[0]), 4), round(float(point[1]), 4)]
        for point in np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ]


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
                "points_px": round_points(group_points),
                "angle_start_deg": round(float(min(angles_deg)), 3),
                "angle_end_deg": round(float(max(angles_deg)), 3),
            }
        )
    return payloads


def _points_array(points_px: list[Any] | tuple[Any, ...] | np.ndarray | None) -> np.ndarray:
    if points_px is None:
        return np.empty((0, 2), dtype=np.float64)
    points = np.asarray(points_px, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return points.reshape(-1, 2)


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


def _symmetric_scale(value: float, reference: float) -> float:
    if value <= 0.0 or reference <= 0.0:
        return float("inf")
    scale = float(value) / float(reference)
    return max(scale, 1.0 / scale)


def _angle_delta_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 90.0) % 180.0 - 90.0)
