from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np


def optimize_adjacent_ball_clusters(
    balls: list[dict[str, Any]],
    *,
    ball_radius_mm: float,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit close balls as small local clusters with scene constraints.

    This optimizer is deliberately conservative. It does not replace the final
    ball coordinates by itself. It answers a narrower question:

    "If nearby balls are assumed to be equal-radius physical spheres, is there a
    small joint adjustment that reduces overlaps / implausible contact
    distances without moving any ball too far from its image-derived anchor?"

    The result is diagnostic evidence for review, confidence, and later
    calibrated-camera work. It also gives the per-ball optimizer occlusion
    context a second scene-level consistency check.
    """

    cfg = settings or {}
    if not bool(cfg.get("enabled", True)):
        return {"status": "disabled", "enabled": False, "clusters": [], "by_ball_id": {}}

    anchors: dict[int, np.ndarray] = {}
    labels: dict[int, str] = {}
    for ball in balls:
        ball_id = int(ball.get("id", ball.get("ball_id", 0)))
        xy = _cluster_anchor_xy(ball, ball_radius_mm)
        if ball_id <= 0 or xy is None:
            continue
        anchors[ball_id] = xy
        labels[ball_id] = str(ball.get("color_label") or ball.get("class") or ball.get("label") or "unknown")

    if len(anchors) < 2:
        return {"status": "not_applicable", "enabled": True, "clusters": [], "by_ball_id": {}}

    diameter = float(cfg.get("target_distance_mm", 2.0 * ball_radius_mm))
    graph = _close_neighbor_graph(
        anchors,
        neighbor_distance_mm=float(cfg.get("neighbor_distance_factor", 1.22)) * diameter,
    )
    components = [
        sorted(component)
        for component in _connected_components(graph)
        if len(component) >= 2
    ]
    if not components:
        return {"status": "no_adjacent_clusters", "enabled": True, "clusters": [], "by_ball_id": {}}

    clusters: list[dict[str, Any]] = []
    by_ball_id: dict[str, dict[str, Any]] = {}
    for cluster_index, component in enumerate(components, start=1):
        cluster = _optimize_component(
            cluster_id=cluster_index,
            component=component,
            anchors=anchors,
            labels=labels,
            diameter=diameter,
            settings=cfg,
        )
        clusters.append(cluster)
        for member in cluster["members"]:
            by_ball_id[str(member["id"])] = {
                "cluster_id": cluster["cluster_id"],
                "cluster_status": cluster["status"],
                "component_size": len(cluster["members"]),
                "initial_xy_mm": member["initial_xy_mm"],
                "joint_xy_mm": member["joint_xy_mm"],
                "movement_mm": member["movement_mm"],
                "pair_constraint_count": len(cluster["pair_constraints"]),
                "initial_pair_rms_mm": cluster["initial_pair_rms_mm"],
                "joint_pair_rms_mm": cluster["joint_pair_rms_mm"],
                "improvement_mm": cluster["improvement_mm"],
                "reasons": cluster["reasons"],
                "note": (
                    "Joint cluster fit is a scene-constraint diagnostic. It is "
                    "not applied to final table coordinates unless a later gate "
                    "explicitly enables that."
                ),
            }

    return {
        "status": "computed",
        "enabled": True,
        "cluster_count": len(clusters),
        "clusters": clusters,
        "by_ball_id": by_ball_id,
    }


def _cluster_anchor_xy(ball: dict[str, Any], ball_radius_mm: float) -> np.ndarray | None:
    optimization = ball.get("source_sphere_optimization") or {}
    if optimization.get("success") and optimization.get("optimized_xy_mm") is not None:
        return _xy(optimization.get("optimized_xy_mm"))

    by_z = ball.get("source_refined_table_xy_by_z_mm") or {}
    z_key = f"z_{float(ball_radius_mm):.2f}".replace("-", "m").replace(".", "_")
    projection = by_z.get(z_key) or by_z.get("z_26_25")
    if isinstance(projection, dict) and projection.get("xy_mm") is not None:
        return _xy(projection.get("xy_mm"))

    if ball.get("source_refined_table_xy_mm") is not None:
        return _xy(ball.get("source_refined_table_xy_mm"))
    if ball.get("table_xy_mm") is not None:
        return _xy(ball.get("table_xy_mm"))
    return None


def _xy(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(2)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array


def _close_neighbor_graph(
    anchors: dict[int, np.ndarray],
    *,
    neighbor_distance_mm: float,
) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = {ball_id: set() for ball_id in anchors}
    ids = sorted(anchors)
    for i, left_id in enumerate(ids):
        for right_id in ids[i + 1 :]:
            distance = float(np.linalg.norm(anchors[left_id] - anchors[right_id]))
            if distance <= neighbor_distance_mm:
                graph[left_id].add(right_id)
                graph[right_id].add(left_id)
    return graph


def _connected_components(graph: dict[int, set[int]]) -> list[list[int]]:
    unseen = set(graph)
    components: list[list[int]] = []
    while unseen:
        start = unseen.pop()
        queue: deque[int] = deque([start])
        component = [start]
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.append(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _optimize_component(
    *,
    cluster_id: int,
    component: list[int],
    anchors: dict[int, np.ndarray],
    labels: dict[int, str],
    diameter: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    initial = {ball_id: anchors[ball_id].copy() for ball_id in component}
    xy = {ball_id: anchors[ball_id].copy() for ball_id in component}
    pairs = _cluster_pairs(
        component,
        initial,
        diameter=diameter,
        contact_distance_factor=float(settings.get("contact_distance_factor", 1.16)),
    )
    if not pairs:
        return _component_payload(
            cluster_id=cluster_id,
            status="no_contact_constraints",
            component=component,
            labels=labels,
            initial=initial,
            xy=xy,
            pairs=[],
            diameter=diameter,
            reasons=["no close contact pairs"],
        )

    iterations = int(settings.get("iterations", 36))
    pair_strength = float(settings.get("pair_strength", 0.18))
    collision_strength = float(settings.get("collision_strength", 0.42))
    anchor_blend = float(settings.get("anchor_blend", 0.08))
    max_movement = float(settings.get("max_movement_mm", 14.0))

    for _ in range(max(1, iterations)):
        deltas: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=np.float64))
        counts: dict[int, int] = defaultdict(int)
        for left_id, right_id in pairs:
            vector = xy[right_id] - xy[left_id]
            distance = float(np.linalg.norm(vector))
            if distance < 1e-6:
                direction = np.array([1.0, 0.0], dtype=np.float64)
            else:
                direction = vector / distance
            error = distance - diameter
            strength = collision_strength if distance < diameter else pair_strength
            correction = 0.5 * strength * error * direction
            deltas[left_id] += correction
            deltas[right_id] -= correction
            counts[left_id] += 1
            counts[right_id] += 1

        for ball_id in component:
            if counts[ball_id]:
                xy[ball_id] += deltas[ball_id] / float(counts[ball_id])
            xy[ball_id] = (1.0 - anchor_blend) * xy[ball_id] + anchor_blend * initial[ball_id]
            movement = xy[ball_id] - initial[ball_id]
            norm = float(np.linalg.norm(movement))
            if norm > max_movement:
                xy[ball_id] = initial[ball_id] + movement * (max_movement / norm)

    initial_rms = _pair_rms(initial, pairs, diameter)
    joint_rms = _pair_rms(xy, pairs, diameter)
    improvement = initial_rms - joint_rms
    minimum_improvement = float(settings.get("minimum_improvement_mm", 0.35))
    status = "optimized" if improvement >= minimum_improvement else "no_better_solution"
    reasons = ["equal_radius_contact_constraints", "non_overlap_constraints"]
    if status != "optimized":
        reasons.append("joint_cluster_did_not_improve_enough")
    if any(float(np.linalg.norm(xy[ball_id] - initial[ball_id])) >= max_movement - 1e-6 for ball_id in component):
        reasons.append("cluster_movement_clamped")

    return _component_payload(
        cluster_id=cluster_id,
        status=status,
        component=component,
        labels=labels,
        initial=initial,
        xy=xy,
        pairs=pairs,
        diameter=diameter,
        reasons=reasons,
    )


def _cluster_pairs(
    component: list[int],
    xy: dict[int, np.ndarray],
    *,
    diameter: float,
    contact_distance_factor: float,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    ids = sorted(component)
    for i, left_id in enumerate(ids):
        for right_id in ids[i + 1 :]:
            distance = float(np.linalg.norm(xy[left_id] - xy[right_id]))
            if distance <= diameter * contact_distance_factor:
                pairs.append((left_id, right_id))
    return pairs


def _pair_rms(
    xy: dict[int, np.ndarray],
    pairs: list[tuple[int, int]],
    diameter: float,
) -> float:
    if not pairs:
        return 0.0
    errors = [
        float(np.linalg.norm(xy[left_id] - xy[right_id]) - diameter)
        for left_id, right_id in pairs
    ]
    return float(np.sqrt(np.mean(np.square(errors))))


def _component_payload(
    *,
    cluster_id: int,
    status: str,
    component: list[int],
    labels: dict[int, str],
    initial: dict[int, np.ndarray],
    xy: dict[int, np.ndarray],
    pairs: list[tuple[int, int]],
    diameter: float,
    reasons: list[str],
) -> dict[str, Any]:
    pair_constraints = []
    for left_id, right_id in pairs:
        initial_distance = float(np.linalg.norm(initial[left_id] - initial[right_id]))
        joint_distance = float(np.linalg.norm(xy[left_id] - xy[right_id]))
        pair_constraints.append(
            {
                "left_id": int(left_id),
                "right_id": int(right_id),
                "target_distance_mm": round(float(diameter), 4),
                "initial_distance_mm": round(initial_distance, 4),
                "joint_distance_mm": round(joint_distance, 4),
                "initial_error_mm": round(initial_distance - diameter, 4),
                "joint_error_mm": round(joint_distance - diameter, 4),
            }
        )
    initial_rms = _pair_rms(initial, pairs, diameter)
    joint_rms = _pair_rms(xy, pairs, diameter)
    return {
        "cluster_id": int(cluster_id),
        "status": status,
        "members": [
            {
                "id": int(ball_id),
                "label": labels.get(ball_id, "unknown"),
                "initial_xy_mm": _round_xy(initial[ball_id]),
                "joint_xy_mm": _round_xy(xy[ball_id]),
                "movement_mm": round(float(np.linalg.norm(xy[ball_id] - initial[ball_id])), 4),
            }
            for ball_id in sorted(component)
        ],
        "pair_constraints": pair_constraints,
        "initial_pair_rms_mm": round(initial_rms, 4),
        "joint_pair_rms_mm": round(joint_rms, 4),
        "improvement_mm": round(float(initial_rms - joint_rms), 4),
        "reasons": sorted(set(reasons)),
    }


def _round_xy(value: np.ndarray) -> list[float]:
    return [round(float(value[0]), 4), round(float(value[1]), 4)]
