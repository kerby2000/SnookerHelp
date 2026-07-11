from __future__ import annotations

from typing import Any


def should_promote_cluster_joint_center(
    *,
    ball: dict[str, Any],
    joint: dict[str, Any],
    settings: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Gate scene-level joint-center promotion.

    The adjacent-cluster optimizer is allowed to propose a physically better
    center, but it must not blindly overwrite image evidence. This gate promotes
    only the cases where the image-local estimate is weak and the scene graph
    moved the ball by a plausible amount while improving contact distances.
    """

    cfg = settings or {}
    reasons: list[str] = []

    if not bool(cfg.get("joint_center_promotion_enabled", True)):
        reasons.append("promotion_disabled")

    if joint.get("cluster_status") != "optimized":
        reasons.append(f"cluster_status={joint.get('cluster_status') or 'missing'}")

    component_size = _as_float(joint.get("component_size"), 0.0)
    min_component_size = float(cfg.get("joint_center_promotion_min_component_size", 4))
    if component_size < min_component_size:
        reasons.append(f"component_size={component_size:g}<{min_component_size:g}")

    improvement = _as_float(joint.get("improvement_mm"), 0.0)
    min_improvement = float(cfg.get("joint_center_promotion_min_improvement_mm", 0.75))
    if improvement < min_improvement:
        reasons.append(f"improvement_mm={improvement:.3f}<{min_improvement:.3f}")

    movement = _as_float(joint.get("movement_mm"), 0.0)
    min_movement = float(cfg.get("joint_center_promotion_min_movement_mm", 0.25))
    max_movement = float(cfg.get("joint_center_promotion_max_movement_mm", 10.0))
    if movement < min_movement:
        reasons.append(f"movement_mm={movement:.3f}<{min_movement:.3f}")
    if movement > max_movement:
        reasons.append(f"movement_mm={movement:.3f}>{max_movement:.3f}")

    role = str(joint.get("cluster_role") or "")
    allowed_roles = cfg.get("joint_center_promotion_roles", ["interior"])
    allowed_roles = {str(item) for item in allowed_roles}
    if allowed_roles and role not in allowed_roles:
        reasons.append(f"cluster_role={role or 'missing'}")

    joint_xy = joint.get("joint_xy_mm")
    if not _valid_xy(joint_xy):
        reasons.append("missing_joint_xy_mm")

    if bool(cfg.get("joint_center_promotion_require_weak_image_evidence", True)):
        weak, weak_reasons = _has_weak_image_evidence(ball, joint, cfg)
        if not weak:
            reasons.append("image_evidence_not_weak")
        else:
            reasons.extend([f"weak_evidence:{reason}" for reason in weak_reasons])

    # Shape outliers are useful evidence that the image-local ellipse is bad,
    # but they are not sufficient by themselves. If disabled, they block
    # promotion; by default they are allowed because this path exists mostly for
    # cluster interiors where local arcs are partial or stolen from neighbors.
    if (
        bool(joint.get("cluster_shape_outlier"))
        and not bool(cfg.get("joint_center_promotion_allow_shape_outliers", True))
    ):
        reasons.append("shape_outlier_blocked")

    blocking = [
        reason
        for reason in reasons
        if not reason.startswith("weak_evidence:")
    ]
    return not blocking, reasons


def cluster_joint_promotion_payload(
    *,
    ball: dict[str, Any],
    joint: dict[str, Any],
    source_px: list[float],
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Small serializable payload attached to promoted balls."""

    promote, reasons = should_promote_cluster_joint_center(
        ball=ball,
        joint=joint,
        settings=settings,
    )
    return {
        "status": "promoted" if promote else "not_promoted",
        "promoted": bool(promote),
        "model": "cluster_graph_joint_center",
        "source_px": [round(float(source_px[0]), 4), round(float(source_px[1]), 4)],
        "joint_xy_mm": [
            round(float(joint["joint_xy_mm"][0]), 4),
            round(float(joint["joint_xy_mm"][1]), 4),
        ]
        if _valid_xy(joint.get("joint_xy_mm"))
        else None,
        "initial_xy_mm": [
            round(float(joint["initial_xy_mm"][0]), 4),
            round(float(joint["initial_xy_mm"][1]), 4),
        ]
        if _valid_xy(joint.get("initial_xy_mm"))
        else None,
        "movement_mm": _round_or_none(joint.get("movement_mm")),
        "improvement_mm": _round_or_none(joint.get("improvement_mm")),
        "component_size": int(joint.get("component_size") or 0),
        "cluster_role": joint.get("cluster_role"),
        "cluster_id": joint.get("cluster_id"),
        "cluster_shape_outlier": bool(joint.get("cluster_shape_outlier")),
        "reasons": reasons,
        "note": (
            "Final center comes from arbitrary-cluster contact graph constraints. "
            "Image boundary remains visible for audit, but it is not the final "
            "center source for this ball."
        ),
    }


def _has_weak_image_evidence(
    ball: dict[str, Any],
    joint: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    policy = ball.get("source_final_center_policy") or {}
    if not isinstance(policy, dict):
        return True, ["missing_policy"]

    threshold = float(cfg.get("joint_center_promotion_max_view_score_for_weak", 75.0))
    view_score = _as_float(policy.get("selected_score"), None)
    if view_score is None:
        view_score = _as_float(policy.get("score"), None)
    if view_score is not None and view_score <= threshold:
        reasons.append(f"view_score={view_score:.1f}<={threshold:.1f}")

    residual_threshold = float(
        cfg.get("joint_center_promotion_min_physical_residual_for_weak_px", 2.0)
    )
    sphere = ball.get("source_sphere_projection") or {}
    observed = sphere.get("observed_fit_score") if isinstance(sphere, dict) else {}
    residual = _as_float((observed or {}).get("rms_px"), None)
    if residual is not None and residual >= residual_threshold:
        reasons.append(f"physical_residual_px={residual:.2f}>={residual_threshold:.2f}")

    if bool(joint.get("cluster_shape_outlier")):
        reasons.append("cluster_shape_outlier")

    role = str(joint.get("cluster_role") or "")
    if role == "interior":
        reasons.append("interior_cluster_ball")

    point_count_threshold = int(
        cfg.get("joint_center_promotion_max_point_count_for_weak", 70)
    )
    point_count = int(policy.get("point_count") or 0)
    if 0 < point_count <= point_count_threshold:
        reasons.append(f"point_count={point_count}<={point_count_threshold}")

    return bool(reasons), reasons


def _valid_xy(value: Any) -> bool:
    try:
        return len(value) >= 2 and all(float(value[index]) == float(value[index]) for index in (0, 1))
    except (TypeError, ValueError, IndexError):
        return False


def _as_float(value: Any, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_or_none(value: Any) -> float | None:
    number = _as_float(value, None)
    return None if number is None else round(float(number), 4)
