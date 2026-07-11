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
    balls_by_id: dict[int, dict[str, Any]] = {}
    for ball in balls:
        ball_id = int(ball.get("id", ball.get("ball_id", 0)))
        xy = _cluster_anchor_xy(ball, ball_radius_mm)
        if ball_id <= 0 or xy is None:
            continue
        anchors[ball_id] = xy
        labels[ball_id] = str(ball.get("color_label") or ball.get("class") or ball.get("label") or "unknown")
        balls_by_id[ball_id] = ball

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
            balls_by_id=balls_by_id,
            diameter=diameter,
            settings=cfg,
        )
        clusters.append(cluster)
        for member in cluster["members"]:
            shell = member.get("cluster_shell", {})
            traversal = member.get("cluster_traversal", {})
            shape = member.get("cluster_shape_prior", {})
            by_ball_id[str(member["id"])] = {
                "cluster_id": cluster["cluster_id"],
                "cluster_status": cluster["status"],
                "component_size": len(cluster["members"]),
                "initial_xy_mm": member["initial_xy_mm"],
                "joint_xy_mm": member["joint_xy_mm"],
                "movement_mm": member["movement_mm"],
                "cluster_shell_status": shell.get("status", "not_computed"),
                "cluster_shell": shell.get("shell_index"),
                "cluster_role": shell.get("role"),
                "cluster_perimeter_distance_mm": shell.get("perimeter_distance_mm"),
                "cluster_neighbor_degree": shell.get("neighbor_degree"),
                "cluster_shell_reason": shell.get("reason"),
                "cluster_traversal_status": traversal.get("status", "not_computed"),
                "cluster_traversal_method": traversal.get("method"),
                "cluster_traversal_primary_path": traversal.get("primary_path"),
                "cluster_traversal_primary_rank": traversal.get("primary_rank"),
                "cluster_traversal_rank_cw": traversal.get("outside_in_clockwise_rank"),
                "cluster_traversal_rank_ccw": traversal.get("outside_in_counterclockwise_rank"),
                "cluster_traversal_rank_perimeter_walk": traversal.get("outside_in_perimeter_walk_rank"),
                "cluster_traversal_rank_perimeter_walk_reverse": traversal.get("outside_in_perimeter_walk_reverse_rank"),
                "cluster_traversal_angle_deg_from_top": traversal.get("angle_deg_from_top"),
                "cluster_traversal_note": traversal.get("note"),
                "cluster_traversal": traversal,
                "cluster_shape_prior_status": shape.get("status", "not_computed"),
                "cluster_shape_prior": shape,
                "cluster_shape_outlier": shape.get("is_shape_outlier"),
                "cluster_shape_reasons": shape.get("reasons", []),
                "cluster_shape_consensus_major_axis_px": shape.get("consensus_major_axis_px"),
                "cluster_shape_consensus_minor_axis_px": shape.get("consensus_minor_axis_px"),
                "cluster_shape_consensus_angle_deg": shape.get("consensus_angle_deg"),
                "cluster_shape_major_scale": shape.get("major_scale"),
                "cluster_shape_minor_scale": shape.get("minor_scale"),
                "cluster_shape_angle_delta_deg": shape.get("angle_delta_deg"),
                "pair_constraint_count": len(cluster["pair_constraints"]),
                "initial_pair_rms_mm": cluster["initial_pair_rms_mm"],
                "joint_pair_rms_mm": cluster["joint_pair_rms_mm"],
                "improvement_mm": cluster["improvement_mm"],
                "fit_policy": cluster.get("fit_policy", {}),
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
    balls_by_id: dict[int, dict[str, Any]],
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
    shell_by_id, shell_payload = _classify_component_shells(
        component=component,
        initial=initial,
        pairs=pairs,
        diameter=diameter,
        settings=settings,
    )
    traversal_by_id, traversal_payload = _compute_component_traversal(
        component=component,
        initial=initial,
        shell_by_id=shell_by_id,
        settings=settings,
    )
    shape_by_id, shape_payload = _compute_component_shape_prior(
        component=component,
        labels=labels,
        balls_by_id=balls_by_id,
        settings=settings,
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
            shell_by_id=shell_by_id,
            shell_payload=shell_payload,
            traversal_by_id=traversal_by_id,
            traversal_payload=traversal_payload,
            shape_by_id=shape_by_id,
            shape_payload=shape_payload,
        )

    iterations = int(settings.get("iterations", 36))
    pair_strength = float(settings.get("pair_strength", 0.18))
    collision_strength = float(settings.get("collision_strength", 0.42))
    anchor_blend = float(settings.get("anchor_blend", 0.08))
    max_movement = float(settings.get("max_movement_mm", 14.0))
    perimeter_weighted_fit = bool(settings.get("perimeter_weighted_fit_enabled", True))
    perimeter_mobility = float(settings.get("perimeter_mobility", 0.55))
    interior_mobility = float(settings.get("interior_mobility", 1.0))
    perimeter_anchor_blend = float(settings.get("perimeter_anchor_blend", anchor_blend * 1.25))
    interior_anchor_blend = float(settings.get("interior_anchor_blend", anchor_blend * 0.55))

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
            correction = strength * error * direction
            if perimeter_weighted_fit:
                left_mobility = _cluster_mobility(
                    shell_by_id.get(left_id, {}),
                    perimeter_mobility=perimeter_mobility,
                    interior_mobility=interior_mobility,
                )
                right_mobility = _cluster_mobility(
                    shell_by_id.get(right_id, {}),
                    perimeter_mobility=perimeter_mobility,
                    interior_mobility=interior_mobility,
                )
                total_mobility = max(1e-6, left_mobility + right_mobility)
                deltas[left_id] += correction * (left_mobility / total_mobility)
                deltas[right_id] -= correction * (right_mobility / total_mobility)
            else:
                deltas[left_id] += 0.5 * correction
                deltas[right_id] -= 0.5 * correction
            counts[left_id] += 1
            counts[right_id] += 1

        for ball_id in component:
            if counts[ball_id]:
                xy[ball_id] += deltas[ball_id] / float(counts[ball_id])
            local_anchor_blend = _cluster_anchor_blend(
                shell_by_id.get(ball_id, {}),
                fallback_anchor_blend=anchor_blend,
                perimeter_anchor_blend=perimeter_anchor_blend,
                interior_anchor_blend=interior_anchor_blend,
                perimeter_weighted_fit=perimeter_weighted_fit,
            )
            xy[ball_id] = (1.0 - local_anchor_blend) * xy[ball_id] + local_anchor_blend * initial[ball_id]
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
    if perimeter_weighted_fit:
        reasons.append("perimeter_weighted_cluster_fit")
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
        shell_by_id=shell_by_id,
        shell_payload=shell_payload,
        traversal_by_id=traversal_by_id,
        traversal_payload=traversal_payload,
        shape_by_id=shape_by_id,
        shape_payload=shape_payload,
        fit_policy={
            "perimeter_weighted_fit_enabled": perimeter_weighted_fit,
            "perimeter_mobility": round(perimeter_mobility, 4),
            "interior_mobility": round(interior_mobility, 4),
            "anchor_blend": round(anchor_blend, 4),
            "perimeter_anchor_blend": round(perimeter_anchor_blend, 4),
            "interior_anchor_blend": round(interior_anchor_blend, 4),
        },
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


def _cluster_mobility(
    shell: dict[str, Any],
    *,
    perimeter_mobility: float,
    interior_mobility: float,
) -> float:
    if shell.get("role") == "perimeter":
        return max(0.01, perimeter_mobility)
    if shell.get("role") == "interior":
        return max(0.01, interior_mobility)
    return 1.0


def _cluster_anchor_blend(
    shell: dict[str, Any],
    *,
    fallback_anchor_blend: float,
    perimeter_anchor_blend: float,
    interior_anchor_blend: float,
    perimeter_weighted_fit: bool,
) -> float:
    if not perimeter_weighted_fit:
        return fallback_anchor_blend
    if shell.get("role") == "perimeter":
        return max(0.0, perimeter_anchor_blend)
    if shell.get("role") == "interior":
        return max(0.0, interior_anchor_blend)
    return fallback_anchor_blend


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


def _classify_component_shells(
    *,
    component: list[int],
    initial: dict[int, np.ndarray],
    pairs: list[tuple[int, int]],
    diameter: float,
    settings: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    enabled = bool(settings.get("shell_classification_enabled", True))
    min_size = int(settings.get("shell_classification_min_size", 5))
    perimeter_distance = float(settings.get("shell_perimeter_distance_factor", 0.42)) * diameter
    pair_degree = _pair_degrees(component, pairs)

    if not enabled:
        return (
            {
                ball_id: {
                    "status": "disabled",
                    "shell_index": None,
                    "role": None,
                    "perimeter_distance_mm": None,
                    "neighbor_degree": pair_degree.get(ball_id, 0),
                    "reason": "disabled",
                }
                for ball_id in component
            },
            {
                "status": "disabled",
                "enabled": False,
                "method": "convex_hull_distance_onion",
                "min_cluster_size": min_size,
                "perimeter_distance_mm": round(perimeter_distance, 4),
                "shell_counts": {},
            },
        )

    if len(component) < min_size:
        return (
            {
                ball_id: {
                    "status": "not_large_cluster",
                    "shell_index": None,
                    "role": None,
                    "perimeter_distance_mm": None,
                    "neighbor_degree": pair_degree.get(ball_id, 0),
                    "reason": f"component smaller than {min_size}",
                }
                for ball_id in component
            },
            {
                "status": "not_large_cluster",
                "enabled": True,
                "method": "convex_hull_distance_onion",
                "min_cluster_size": min_size,
                "perimeter_distance_mm": round(perimeter_distance, 4),
                "shell_counts": {},
            },
        )

    remaining = set(component)
    shell_by_id: dict[int, dict[str, Any]] = {}
    shell_counts: dict[str, int] = {}
    shell_index = 1
    while remaining:
        current = sorted(remaining)
        if len(current) <= 2:
            distances = {ball_id: 0.0 for ball_id in current}
            shell_ids = set(current)
            reason = "too_few_points_for_hull"
        else:
            hull_ids = _convex_hull_ids(current, initial)
            hull_points = [initial[ball_id] for ball_id in hull_ids]
            if len(hull_points) < 3:
                distances = {ball_id: 0.0 if ball_id in hull_ids else float("inf") for ball_id in current}
                shell_ids = set(hull_ids) if hull_ids else set(current)
                reason = "degenerate_hull"
            else:
                distances = {
                    ball_id: _distance_to_polygon_edges(initial[ball_id], hull_points)
                    for ball_id in current
                }
                shell_ids = {
                    ball_id
                    for ball_id, distance in distances.items()
                    if distance <= perimeter_distance
                }
                shell_ids.update(hull_ids)
                if not shell_ids:
                    shell_ids = set(hull_ids)
                reason = "convex_hull_distance"

        role = "perimeter" if shell_index == 1 else "interior"
        for ball_id in sorted(shell_ids):
            shell_by_id[ball_id] = {
                "status": "computed",
                "shell_index": shell_index,
                "role": role,
                "perimeter_distance_mm": round(float(distances.get(ball_id, 0.0)), 4),
                "neighbor_degree": pair_degree.get(ball_id, 0),
                "reason": reason,
            }
        shell_counts[str(shell_index)] = len(shell_ids)
        remaining.difference_update(shell_ids)
        shell_index += 1

    return (
        shell_by_id,
        {
            "status": "computed",
            "enabled": True,
            "method": "convex_hull_distance_onion",
            "min_cluster_size": min_size,
            "perimeter_distance_mm": round(perimeter_distance, 4),
            "shell_counts": shell_counts,
            "note": (
                "Shell classification is diagnostic. Shell 1 means the ball "
                "lies on the outer boundary of a large adjacent-ball cluster; "
                "later shells are interior balls with less cloth-side evidence."
            ),
        },
    )


def _compute_component_traversal(
    *,
    component: list[int],
    initial: dict[int, np.ndarray],
    shell_by_id: dict[int, dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    enabled = bool(settings.get("traversal_diagnostics_enabled", True))
    if not enabled:
        return (
            {
                ball_id: {
                    "status": "disabled",
                    "primary_rank": None,
                    "outside_in_clockwise_rank": None,
                    "outside_in_counterclockwise_rank": None,
                    "reason": "disabled",
                }
                for ball_id in component
            },
            {
                "status": "disabled",
                "enabled": False,
                "method": "outside_in_shell_angle",
                "paths": {},
            },
        )

    if not component:
        return ({}, {"status": "not_applicable", "enabled": True, "paths": {}})

    centroid = np.mean([initial[ball_id] for ball_id in component], axis=0)
    groups: dict[int, list[int]] = defaultdict(list)
    for ball_id in component:
        shell = shell_by_id.get(ball_id, {})
        shell_index = shell.get("shell_index")
        try:
            normalized_shell = int(shell_index)
        except (TypeError, ValueError):
            normalized_shell = 999
        groups[normalized_shell].append(ball_id)

    clockwise: list[int] = []
    counterclockwise: list[int] = []
    perimeter_walk: list[int] = []
    perimeter_walk_reverse: list[int] = []
    for shell_index in sorted(groups):
        shell_ids = groups[shell_index]
        ordered = sorted(shell_ids, key=lambda ball_id: (_angle_clockwise_from_top(initial[ball_id], centroid), ball_id))
        clockwise.extend(ordered)
        counterclockwise.extend(reversed(ordered))
        perimeter_walk.extend(
            _perimeter_walk_order(
                shell_ids,
                initial=initial,
                centroid=centroid,
                reverse=False,
            )
        )
        perimeter_walk_reverse.extend(
            _perimeter_walk_order(
                shell_ids,
                initial=initial,
                centroid=centroid,
                reverse=True,
            )
        )

    cw_rank = {ball_id: index for index, ball_id in enumerate(clockwise, start=1)}
    ccw_rank = {ball_id: index for index, ball_id in enumerate(counterclockwise, start=1)}
    perimeter_walk_rank = {
        ball_id: index for index, ball_id in enumerate(perimeter_walk, start=1)
    }
    perimeter_walk_reverse_rank = {
        ball_id: index for index, ball_id in enumerate(perimeter_walk_reverse, start=1)
    }
    traversal_by_id: dict[int, dict[str, Any]] = {}
    for ball_id in component:
        shell = shell_by_id.get(ball_id, {})
        shell_index = shell.get("shell_index")
        shell_members = groups.get(int(shell_index), []) if shell_index is not None else component
        angle = _angle_clockwise_from_top(initial[ball_id], centroid)
        traversal_by_id[ball_id] = {
            "status": "computed",
            "method": "outside_in_shell_angle",
            "primary_path": "outside_in_perimeter_walk",
            "primary_rank": perimeter_walk_rank.get(ball_id),
            "outside_in_clockwise_rank": cw_rank.get(ball_id),
            "outside_in_counterclockwise_rank": ccw_rank.get(ball_id),
            "outside_in_perimeter_walk_rank": perimeter_walk_rank.get(ball_id),
            "outside_in_perimeter_walk_reverse_rank": perimeter_walk_reverse_rank.get(ball_id),
            "shell_size": len(shell_members),
            "angle_deg_from_top": round(float(np.degrees(angle)), 4),
            "note": (
                "Traversal ranks are diagnostic only. They show candidate "
                "outside-in processing paths. The perimeter-walk path starts "
                "from the top-left ball of each shell and walks the outside "
                "edge before visiting interior shells; it does not yet drive "
                "the final ball fit."
            ),
        }

    return (
        traversal_by_id,
        {
            "status": "computed",
            "enabled": True,
            "method": "outside_in_shell_angle",
            "primary_path": "outside_in_perimeter_walk",
            "paths": {
                "outside_in_clockwise": [int(ball_id) for ball_id in clockwise],
                "outside_in_counterclockwise": [int(ball_id) for ball_id in counterclockwise],
                "outside_in_perimeter_walk": [int(ball_id) for ball_id in perimeter_walk],
                "outside_in_perimeter_walk_reverse": [
                    int(ball_id) for ball_id in perimeter_walk_reverse
                ],
            },
            "centroid_xy_mm": _round_xy(centroid),
            "note": (
                "Path diagnostics visit perimeter shell(s) before interior "
                "shell(s). Angular clockwise/counter-clockwise paths and a "
                "top-left-start perimeter walk are emitted so they can be "
                "compared before any future fitting path is allowed to alter "
                "final coordinates."
            ),
        },
    )


def _perimeter_walk_order(
    shell_ids: list[int],
    *,
    initial: dict[int, np.ndarray],
    centroid: np.ndarray,
    reverse: bool,
) -> list[int]:
    if not shell_ids:
        return []
    start_id = min(
        shell_ids,
        key=lambda ball_id: (
            float(initial[ball_id][1]),
            float(initial[ball_id][0]),
            int(ball_id),
        ),
    )
    ordered = sorted(
        shell_ids,
        key=lambda ball_id: (
            _angle_clockwise_from_top(initial[ball_id], centroid),
            ball_id,
        ),
        reverse=not reverse,
    )
    return _rotate_order_to_start(ordered, start_id)


def _rotate_order_to_start(ordered: list[int], start_id: int) -> list[int]:
    if start_id not in ordered:
        return ordered
    index = ordered.index(start_id)
    return ordered[index:] + ordered[:index]


def _angle_clockwise_from_top(point: np.ndarray, centroid: np.ndarray) -> float:
    vector = point - centroid
    angle = float(np.arctan2(vector[0], -vector[1]))
    if angle < 0:
        angle += float(2.0 * np.pi)
    return angle


def _pair_degrees(component: list[int], pairs: list[tuple[int, int]]) -> dict[int, int]:
    degree = {ball_id: 0 for ball_id in component}
    for left_id, right_id in pairs:
        degree[left_id] = degree.get(left_id, 0) + 1
        degree[right_id] = degree.get(right_id, 0) + 1
    return degree


def _convex_hull_ids(ids: list[int], xy: dict[int, np.ndarray]) -> list[int]:
    ordered = sorted(ids, key=lambda ball_id: (float(xy[ball_id][0]), float(xy[ball_id][1]), ball_id))
    if len(ordered) <= 1:
        return ordered

    def cross(origin_id: int, left_id: int, right_id: int) -> float:
        origin = xy[origin_id]
        left = xy[left_id]
        right = xy[right_id]
        return float((left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0]))

    lower: list[int] = []
    for ball_id in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], ball_id) <= 1e-9:
            lower.pop()
        lower.append(ball_id)

    upper: list[int] = []
    for ball_id in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], ball_id) <= 1e-9:
            upper.pop()
        upper.append(ball_id)

    hull = lower[:-1] + upper[:-1]
    return list(dict.fromkeys(hull))


def _compute_component_shape_prior(
    *,
    component: list[int],
    labels: dict[int, str],
    balls_by_id: dict[int, dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Compare same-label cluster ellipses against a robust cluster consensus.

    This is diagnostic only. It does not move centers or replace fitted
    ellipses. Dense clusters can produce plausible-looking but physically
    impossible ellipses when neighboring-ball edges/reflections are sampled as
    if they belonged to the selected ball. Same-color balls in one tight
    cluster should have very similar projected ellipse axes and orientation, so
    this prior flags large shape outliers for review/confidence.
    """

    enabled = bool(settings.get("shape_prior_enabled", True))
    min_component_size = int(settings.get("shape_prior_min_cluster_size", 5))
    min_label_count = int(settings.get("shape_prior_min_label_count", 5))
    min_consensus_members = int(settings.get("shape_prior_min_consensus_members", 4))
    min_points = int(settings.get("shape_prior_min_point_count", 55))
    max_axis_ratio = float(settings.get("shape_prior_max_consensus_axis_ratio", 1.55))
    major_scale_limit = float(settings.get("shape_prior_major_scale_limit", 1.22))
    minor_scale_limit = float(settings.get("shape_prior_minor_scale_limit", 1.22))
    angle_limit = float(settings.get("shape_prior_angle_delta_deg", 12.0))

    if not enabled:
        return (
            {
                ball_id: {
                    "status": "disabled",
                    "enabled": False,
                    "is_shape_outlier": False,
                    "reasons": ["disabled"],
                }
                for ball_id in component
            },
            {
                "status": "disabled",
                "enabled": False,
                "method": "same_label_cluster_ellipse_consensus",
            },
        )

    if len(component) < min_component_size:
        return (
            {
                ball_id: {
                    "status": "not_large_cluster",
                    "enabled": True,
                    "is_shape_outlier": False,
                    "reasons": [f"component smaller than {min_component_size}"],
                }
                for ball_id in component
            },
            {
                "status": "not_large_cluster",
                "enabled": True,
                "method": "same_label_cluster_ellipse_consensus",
                "min_cluster_size": min_component_size,
            },
        )

    groups: dict[str, list[int]] = defaultdict(list)
    for ball_id in component:
        groups[labels.get(ball_id, "unknown")].append(ball_id)

    by_id: dict[int, dict[str, Any]] = {}
    group_payloads: dict[str, dict[str, Any]] = {}
    for label, ids in groups.items():
        ellipses = [
            (ball_id, ellipse)
            for ball_id in ids
            if (ellipse := _cluster_shape_ellipse(balls_by_id.get(ball_id, {}))) is not None
        ]
        if len(ids) < min_label_count or len(ellipses) < min_consensus_members:
            reason = (
                f"same-label group smaller than {min_label_count}"
                if len(ids) < min_label_count
                else f"fewer than {min_consensus_members} valid ellipses"
            )
            for ball_id in ids:
                by_id[ball_id] = {
                    "status": "not_enough_same_label_evidence",
                    "enabled": True,
                    "label": label,
                    "same_label_count": len(ids),
                    "valid_ellipse_count": len(ellipses),
                    "is_shape_outlier": False,
                    "reasons": [reason],
                }
            group_payloads[label] = {
                "status": "not_enough_same_label_evidence",
                "same_label_count": len(ids),
                "valid_ellipse_count": len(ellipses),
                "reason": reason,
            }
            continue

        plausible = [
            (ball_id, ellipse)
            for ball_id, ellipse in ellipses
            if int(ellipse.get("point_count") or 0) >= min_points
            and float(ellipse.get("axis_ratio") or 999.0) <= max_axis_ratio
            and float(ellipse.get("major_axis_px") or 0.0) > 0.0
            and float(ellipse.get("minor_axis_px") or 0.0) > 0.0
        ]
        if len(plausible) < min_consensus_members:
            plausible = [
                (ball_id, ellipse)
                for ball_id, ellipse in ellipses
                if float(ellipse.get("major_axis_px") or 0.0) > 0.0
                and float(ellipse.get("minor_axis_px") or 0.0) > 0.0
            ]
        if len(plausible) < min_consensus_members:
            for ball_id in ids:
                by_id[ball_id] = {
                    "status": "not_enough_consensus_members",
                    "enabled": True,
                    "label": label,
                    "same_label_count": len(ids),
                    "valid_ellipse_count": len(ellipses),
                    "consensus_member_count": len(plausible),
                    "is_shape_outlier": False,
                    "reasons": [f"fewer than {min_consensus_members} consensus ellipses"],
                }
            group_payloads[label] = {
                "status": "not_enough_consensus_members",
                "same_label_count": len(ids),
                "valid_ellipse_count": len(ellipses),
                "consensus_member_count": len(plausible),
            }
            continue

        consensus_major = float(np.median([ellipse["major_axis_px"] for _, ellipse in plausible]))
        consensus_minor = float(np.median([ellipse["minor_axis_px"] for _, ellipse in plausible]))
        consensus_angle = _median_angle_mod_180([ellipse["angle_deg"] for _, ellipse in plausible])
        consensus_ids = [int(ball_id) for ball_id, _ in plausible]
        group_payloads[label] = {
            "status": "computed",
            "same_label_count": len(ids),
            "valid_ellipse_count": len(ellipses),
            "consensus_member_count": len(plausible),
            "consensus_member_ids": consensus_ids,
            "consensus_major_axis_px": round(consensus_major, 4),
            "consensus_minor_axis_px": round(consensus_minor, 4),
            "consensus_angle_deg": round(consensus_angle, 4),
            "limits": {
                "major_scale_limit": major_scale_limit,
                "minor_scale_limit": minor_scale_limit,
                "angle_delta_deg": angle_limit,
            },
        }

        for ball_id in ids:
            ellipse = _cluster_shape_ellipse(balls_by_id.get(ball_id, {}))
            if ellipse is None:
                by_id[ball_id] = {
                    "status": "no_ellipse",
                    "enabled": True,
                    "label": label,
                    "is_shape_outlier": False,
                    "reasons": ["no ellipse available"],
                    **_cluster_shape_consensus_fields(
                        consensus_major,
                        consensus_minor,
                        consensus_angle,
                        consensus_ids,
                    ),
                }
                continue

            major = float(ellipse.get("major_axis_px") or 0.0)
            minor = float(ellipse.get("minor_axis_px") or 0.0)
            angle = float(ellipse.get("angle_deg") or 0.0) % 180.0
            major_scale = _symmetric_scale(major, consensus_major)
            minor_scale = _symmetric_scale(minor, consensus_minor)
            angle_delta = _angle_delta_mod_180(angle, consensus_angle)
            reasons: list[str] = []
            if major_scale > major_scale_limit:
                reasons.append("cluster_ellipse_major_outlier")
            if minor_scale > minor_scale_limit:
                reasons.append("cluster_ellipse_minor_outlier")
            if angle_delta > angle_limit:
                reasons.append("cluster_ellipse_angle_outlier")

            by_id[ball_id] = {
                "status": "computed",
                "enabled": True,
                "method": "same_label_cluster_ellipse_consensus",
                "label": label,
                "same_label_count": len(ids),
                "valid_ellipse_count": len(ellipses),
                "ellipse_source": ellipse.get("source"),
                "ellipse_point_count": ellipse.get("point_count"),
                "ellipse_major_axis_px": round(major, 4),
                "ellipse_minor_axis_px": round(minor, 4),
                "ellipse_angle_deg": round(angle, 4),
                "ellipse_axis_ratio": (
                    round(float(ellipse["axis_ratio"]), 4)
                    if ellipse.get("axis_ratio") is not None
                    else None
                ),
                "major_scale": round(major_scale, 4),
                "minor_scale": round(minor_scale, 4),
                "angle_delta_deg": round(angle_delta, 4),
                "is_shape_outlier": bool(reasons),
                "reasons": reasons,
                **_cluster_shape_consensus_fields(
                    consensus_major,
                    consensus_minor,
                    consensus_angle,
                    consensus_ids,
                ),
            }

    for ball_id in component:
        by_id.setdefault(
            ball_id,
            {
                "status": "not_computed",
                "enabled": True,
                "is_shape_outlier": False,
                "reasons": ["not computed"],
            },
        )

    computed_groups = [
        group for group in group_payloads.values() if group.get("status") == "computed"
    ]
    return (
        by_id,
        {
            "status": "computed" if computed_groups else "not_enough_same_label_evidence",
            "enabled": True,
            "method": "same_label_cluster_ellipse_consensus",
            "groups": group_payloads,
            "note": (
                "Shape prior is diagnostic. Same-color balls in a tight cluster "
                "should have similar projected ellipse size and angle; outliers "
                "usually mean neighboring-ball edges or reflections were sampled."
            ),
        },
    )


def _cluster_shape_ellipse(ball: dict[str, Any]) -> dict[str, Any] | None:
    policy = ball.get("source_final_center_policy") or {}
    candidates = [
        policy.get("ellipse_fit") if isinstance(policy, dict) else None,
        ball.get("source_ellipse_fit"),
        ball.get("source_radial_ellipse_fit"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        major = candidate.get("major_axis_px")
        minor = candidate.get("minor_axis_px")
        angle = candidate.get("angle_deg")
        if major is None or minor is None or angle is None:
            continue
        try:
            payload = dict(candidate)
            payload["major_axis_px"] = float(major)
            payload["minor_axis_px"] = float(minor)
            payload["angle_deg"] = float(angle)
            payload["axis_ratio"] = float(candidate.get("axis_ratio") or (float(major) / max(1e-6, float(minor))))
            payload["point_count"] = int(
                policy.get("point_count")
                or len(policy.get("boundary_points_px") or [])
                or len(ball.get("source_boundary_points_px") or [])
                or len(ball.get("source_radial_boundary_points_px") or [])
                or 0
            )
            payload["source"] = candidate.get("source") or policy.get("observed_source") or "source_ellipse_fit"
            return payload
        except (TypeError, ValueError):
            continue
    return None


def _cluster_shape_consensus_fields(
    consensus_major: float,
    consensus_minor: float,
    consensus_angle: float,
    consensus_ids: list[int],
) -> dict[str, Any]:
    return {
        "consensus_major_axis_px": round(float(consensus_major), 4),
        "consensus_minor_axis_px": round(float(consensus_minor), 4),
        "consensus_angle_deg": round(float(consensus_angle), 4),
        "consensus_member_ids": [int(ball_id) for ball_id in consensus_ids],
    }


def _symmetric_scale(value: float, reference: float) -> float:
    if value <= 0.0 or reference <= 0.0:
        return float("inf")
    scale = float(value) / float(reference)
    return max(scale, 1.0 / scale)


def _median_angle_mod_180(angles_deg: list[float]) -> float:
    if not angles_deg:
        return 0.0
    normalized = [float(angle) % 180.0 for angle in angles_deg]
    # Brute force over observed angles is enough here and more robust than a
    # circular mean when a few bad cluster members are badly rotated.
    return min(
        normalized,
        key=lambda candidate: (
            sum(_angle_delta_mod_180(candidate, angle) for angle in normalized),
            candidate,
        ),
    )


def _angle_delta_mod_180(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 90.0) % 180.0 - 90.0)


def _distance_to_polygon_edges(point: np.ndarray, polygon: list[np.ndarray]) -> float:
    if len(polygon) < 2:
        return 0.0
    distances = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        distances.append(_point_segment_distance(point, start, end))
    return float(min(distances)) if distances else 0.0


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, segment) / length_sq)
    t = max(0.0, min(1.0, t))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


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
    shell_by_id: dict[int, dict[str, Any]] | None = None,
    shell_payload: dict[str, Any] | None = None,
    traversal_by_id: dict[int, dict[str, Any]] | None = None,
    traversal_payload: dict[str, Any] | None = None,
    shape_by_id: dict[int, dict[str, Any]] | None = None,
    shape_payload: dict[str, Any] | None = None,
    fit_policy: dict[str, Any] | None = None,
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
                "cluster_shell": shell_by_id.get(ball_id, {}) if shell_by_id else {},
                "cluster_traversal": traversal_by_id.get(ball_id, {}) if traversal_by_id else {},
                "cluster_shape_prior": shape_by_id.get(ball_id, {}) if shape_by_id else {},
            }
            for ball_id in sorted(component)
        ],
        "pair_constraints": pair_constraints,
        "initial_pair_rms_mm": round(initial_rms, 4),
        "joint_pair_rms_mm": round(joint_rms, 4),
        "improvement_mm": round(float(initial_rms - joint_rms), 4),
        "reasons": sorted(set(reasons)),
        "fit_policy": fit_policy or {},
        "shell_classification": shell_payload or {
            "status": "not_computed",
            "enabled": False,
            "method": "convex_hull_distance_onion",
            "shell_counts": {},
        },
        "traversal": traversal_payload or {
            "status": "not_computed",
            "enabled": False,
            "method": "outside_in_shell_angle",
            "paths": {},
        },
        "shape_prior": shape_payload or {
            "status": "not_computed",
            "enabled": False,
            "method": "same_label_cluster_ellipse_consensus",
        },
    }


def _round_xy(value: np.ndarray) -> list[float]:
    return [round(float(value[0]), 4), round(float(value[1]), 4)]
