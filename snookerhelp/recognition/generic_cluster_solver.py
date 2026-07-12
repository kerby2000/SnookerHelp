from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np

from snookerhelp.recognition.arc_combo_fit import fit_fixed_shape_ellipse_center
from snookerhelp.recognition.cluster_arc_assignment import (
    assign_boundary_points_globally,
)


def solve_generic_cluster_component(
    nodes: list[dict[str, Any]],
    *,
    component: dict[str, Any],
    shape: dict[str, Any],
    camera_model: Any,
    ball_radius_mm: float,
    settings: dict[str, Any],
    source_image: np.ndarray | None = None,
) -> dict[str, Any]:
    """Solve an arbitrary connected component without a rack template.

    All starts consume the same independent centers and the same deduplicated
    union of raw boundary samples.  A start never consumes another start's
    promoted output.  Pixel ownership and center refinement are recomputed for
    the whole component on every iteration.
    """

    if len(nodes) < 2:
        return _abstained(component, nodes, "component_has_fewer_than_two_nodes")
    if shape.get("status") != "computed":
        return _abstained(component, nodes, "shared_shape_is_unavailable")

    independent = np.asarray(
        [node["center_px"] for node in nodes],
        dtype=np.float64,
    ).reshape(-1, 2)
    raw_points = _all_raw_points(nodes)
    if len(raw_points) < int(settings.get("generic_minimum_raw_point_count", 20)):
        return _abstained(component, nodes, "too_few_component_boundary_samples")

    highlight_evidence = _highlight_evidence(
        source_image,
        nodes=nodes,
        shape=shape,
        settings=settings,
    )
    starts = _build_starts(
        nodes,
        independent=independent,
        shape=shape,
        settings=settings,
    )
    baseline = _evaluate_centers(
        independent,
        nodes=nodes,
        independent=independent,
        shape=shape,
        camera_model=camera_model,
        ball_radius_mm=ball_radius_mm,
        settings=settings,
    )
    solutions = [
        _refine_start(
            name,
            centers,
            nodes=nodes,
            independent=independent,
            shape=shape,
            camera_model=camera_model,
            ball_radius_mm=ball_radius_mm,
            settings=settings,
        )
        for name, centers in starts
    ]
    eligible = [solution for solution in solutions if solution["hard_constraints_passed"]]
    # Even when every full-cardinality start overlaps, inspect explicit
    # duplicate hypotheses before abstaining. Removing one genuinely duplicate
    # hypothesis may be the operation that makes the hard constraint feasible.
    comparison_pool = eligible or solutions
    selected = min(comparison_pool, key=lambda item: float(item["objective"]))
    stability = _solution_stability(comparison_pool, selected=selected)
    existence = _existence_hypotheses(
        selected,
        nodes=nodes,
        shape=shape,
        highlight_evidence=highlight_evidence,
        settings=settings,
    )
    duplicates = _duplicate_hypotheses(
        selected,
        nodes=nodes,
        existence=existence,
        camera_model=camera_model,
        ball_radius_mm=ball_radius_mm,
        settings=settings,
    )
    suppressed_ids = [
        int(item["suppress_ball_id"])
        for item in duplicates.get("hypotheses") or []
        if item.get("decision") == "suppress"
    ]

    # A strongly evidenced duplicate is re-solved without the redundant node.
    # This is a separate scenario, not a mutation of the selected full model.
    suppression_solution: dict[str, Any] | None = None
    active_nodes = list(nodes)
    if suppressed_ids:
        active_nodes = [
            node for node in nodes if int(node["ball_id"]) not in suppressed_ids
        ]
        active_indices = [
            index
            for index, node in enumerate(nodes)
            if int(node["ball_id"]) not in suppressed_ids
        ]
        active_independent = independent[active_indices]
        suppression_solution = _refine_start(
            "duplicate_suppression",
            np.asarray(selected["centers_px"], dtype=np.float64)[active_indices],
            nodes=active_nodes,
            independent=active_independent,
            shape=shape,
            camera_model=camera_model,
            ball_radius_mm=ball_radius_mm,
            settings=settings,
        )
        if suppression_solution["hard_constraints_passed"]:
            selected = suppression_solution
            independent = active_independent
        else:
            suppressed_ids = []
            active_nodes = list(nodes)

    missing = _missing_hypotheses(
        selected,
        nodes=active_nodes,
        shape=shape,
        settings=settings,
    )
    gate = _promotion_gate(
        component=component,
        nodes=active_nodes,
        baseline=baseline,
        selected=selected,
        stability=stability,
        duplicates=duplicates,
        suppressed_ids=suppressed_ids,
        missing=missing,
        settings=settings,
    )
    promoted = bool(gate["passed"])
    result = _component_result(
        component=component,
        nodes=active_nodes,
        all_nodes=nodes,
        shape=shape,
        baseline=baseline,
        solutions=solutions,
        selected=selected,
        stability=stability,
        highlight_evidence=highlight_evidence,
        existence=existence,
        duplicates=duplicates,
        missing=missing,
        suppressed_ids=suppressed_ids,
        suppression_solution=suppression_solution,
        promoted=promoted,
        reasons=gate["reasons"],
        settings=settings,
    )
    result["promotion_gate"] = gate
    return result


def _build_starts(
    nodes: list[dict[str, Any]],
    *,
    independent: np.ndarray,
    shape: dict[str, Any],
    settings: dict[str, Any],
) -> list[tuple[str, np.ndarray]]:
    local = independent.copy()
    maximum_shift = float(settings.get("generic_start_max_shift_px", 8.0))
    for index, node in enumerate(nodes):
        points = node.get("raw_points_px") or []
        fit = fit_fixed_shape_ellipse_center(
            points,
            major_axis_px=float(shape["major_axis_px"]),
            minor_axis_px=float(shape["minor_axis_px"]),
            angle_deg=float(shape["angle_deg"]),
            fallback_center_px=independent[index],
            seed_ellipse=_ellipse(independent[index], shape),
            source="generic_cluster_local_fixed_shape_start",
        )
        if fit is None:
            continue
        local[index] = _clamp_from_anchor(
            np.asarray(fit["center_px"], dtype=np.float64),
            independent[index],
            maximum_shift,
        )

    # Deterministic opposing perturbations expose local-minimum sensitivity.
    # They are intentionally subpixel and do not encode a traversal order.
    jitter = float(settings.get("generic_start_jitter_px", 0.75))
    pattern = np.asarray(
        [
            [np.cos(index * 2.399963), np.sin(index * 2.399963)]
            for index in range(len(nodes))
        ],
        dtype=np.float64,
    )
    return [
        ("independent", independent.copy()),
        ("local_fixed_shape", local),
        ("positive_deterministic_jitter", independent + jitter * pattern),
        ("negative_deterministic_jitter", independent - jitter * pattern),
    ]


def _refine_start(
    name: str,
    start: np.ndarray,
    *,
    nodes: list[dict[str, Any]],
    independent: np.ndarray,
    shape: dict[str, Any],
    camera_model: Any,
    ball_radius_mm: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    centers = np.asarray(start, dtype=np.float64).reshape(-1, 2).copy()
    maximum_shift = float(settings.get("generic_max_center_shift_px", 8.0))
    minimum_points = int(settings.get("generic_refine_min_owned_points", 5))
    iterations = max(1, int(settings.get("generic_refine_iterations", 4)))
    blend = float(settings.get("generic_refine_blend", 0.85))

    for _ in range(iterations):
        ownership = _assign(nodes, centers, shape=shape, settings=settings)
        refined = centers.copy()
        for index, node in enumerate(nodes):
            owned = (ownership.get("by_ball_id") or {}).get(
                str(node["ball_id"]),
                {},
            )
            points = owned.get("owned_points_px") or []
            if len(points) < minimum_points:
                continue
            fit = fit_fixed_shape_ellipse_center(
                points,
                major_axis_px=float(shape["major_axis_px"]),
                minor_axis_px=float(shape["minor_axis_px"]),
                angle_deg=float(shape["angle_deg"]),
                fallback_center_px=centers[index],
                seed_ellipse=_ellipse(centers[index], shape),
                source="generic_cluster_global_owned_fixed_shape",
            )
            if fit is None:
                continue
            candidate = np.asarray(fit["center_px"], dtype=np.float64)
            candidate = blend * candidate + (1.0 - blend) * centers[index]
            refined[index] = _clamp_from_anchor(
                candidate,
                independent[index],
                maximum_shift,
            )
        centers = refined

    evaluated = _evaluate_centers(
        centers,
        nodes=nodes,
        independent=independent,
        shape=shape,
        camera_model=camera_model,
        ball_radius_mm=ball_radius_mm,
        settings=settings,
    )
    evaluated["name"] = name
    return evaluated


def _evaluate_centers(
    centers: np.ndarray,
    *,
    nodes: list[dict[str, Any]],
    independent: np.ndarray,
    shape: dict[str, Any],
    camera_model: Any,
    ball_radius_mm: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    ownership = _assign(nodes, centers, shape=shape, settings=settings)
    point_rows = ownership.get("points") or []
    gate = max(1e-6, float(settings.get("global_arc_residual_gate_px", 5.0)))
    clipped = [
        min(float(row.get("best_residual_px") or gate * 2.0), gate * 1.5) / gate
        for row in point_rows
    ]
    residual_term = float(np.mean(np.square(clipped))) if clipped else 2.25
    owned_fraction = float(ownership.get("owned_fraction") or 0.0)
    movement = np.linalg.norm(centers - independent, axis=1)
    movement_limit = max(
        1e-6,
        float(settings.get("generic_max_center_shift_px", 8.0)),
    )
    objective = (
        residual_term
        + float(settings.get("generic_unowned_weight", 0.22))
        * (1.0 - owned_fraction)
        + float(settings.get("generic_movement_weight", 0.04))
        * float(np.mean(movement / movement_limit))
    )
    geometry = _hard_non_overlap(
        centers,
        nodes=nodes,
        camera_model=camera_model,
        ball_radius_mm=ball_radius_mm,
        settings=settings,
    )
    by_ball = ownership.get("by_ball_id") or {}
    coverage_by_id = {
        str(node["ball_id"]): _arc_coverage(
            (by_ball.get(str(node["ball_id"])) or {}).get("owned_points_px") or [],
            center=centers[index],
            shape=shape,
        )
        for index, node in enumerate(nodes)
    }
    return {
        "status": "computed",
        "centers_px": np.asarray(centers).round(4).tolist(),
        "objective": round(float(objective), 7),
        "residual_term": round(float(residual_term), 7),
        "owned_fraction": round(owned_fraction, 4),
        "owned_point_count": int(ownership.get("owned_point_count") or 0),
        "ambiguous_point_count": int(ownership.get("ambiguous_point_count") or 0),
        "unowned_point_count": int(ownership.get("unowned_point_count") or 0),
        "mean_movement_px": round(float(np.mean(movement)), 4),
        "max_movement_px": round(float(np.max(movement)), 4),
        "hard_constraints_passed": bool(geometry["passed"]),
        "hard_non_overlap": geometry,
        "ownership": ownership,
        "arc_coverage_by_ball_id": coverage_by_id,
    }


def _assign(
    nodes: list[dict[str, Any]],
    centers: np.ndarray,
    *,
    shape: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    models = {
        int(node["ball_id"]): _ellipse(centers[index], shape)
        for index, node in enumerate(nodes)
    }
    return assign_boundary_points_globally(
        _all_raw_points(nodes),
        ellipses_by_id=models,
        residual_gate_px=float(settings.get("global_arc_residual_gate_px", 5.0)),
        ambiguity_margin_px=float(
            settings.get("global_arc_ambiguity_margin_px", 0.6)
        ),
    )


def _hard_non_overlap(
    centers: np.ndarray,
    *,
    nodes: list[dict[str, Any]],
    camera_model: Any,
    ball_radius_mm: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    world: list[np.ndarray | None] = []
    for center in centers:
        try:
            point = np.asarray(
                camera_model.image_point_to_world_plane(center, ball_radius_mm),
                dtype=np.float64,
            ).reshape(-1)
            world.append(point[:2] if len(point) >= 2 else None)
        except (AttributeError, TypeError, ValueError):
            world.append(None)

    diameter = 2.0 * float(ball_radius_mm)
    approximate = not bool(getattr(camera_model, "is_calibrated", False))
    tolerance = float(
        settings.get(
            "generic_hard_overlap_tolerance_mm_approximate"
            if approximate
            else "generic_hard_overlap_tolerance_mm_calibrated",
            2.5 if approximate else 0.6,
        )
    )
    minimum_allowed = diameter - tolerance
    violations: list[dict[str, Any]] = []
    distances: list[float] = []
    for left in range(len(centers)):
        for right in range(left + 1, len(centers)):
            if world[left] is None or world[right] is None:
                continue
            distance = float(np.linalg.norm(world[left] - world[right]))
            distances.append(distance)
            if distance + 1e-9 >= minimum_allowed:
                continue
            violations.append(
                {
                    "left_ball_id": int(nodes[left]["ball_id"]),
                    "right_ball_id": int(nodes[right]["ball_id"]),
                    "distance_mm": round(distance, 4),
                    "minimum_allowed_mm": round(minimum_allowed, 4),
                }
            )
    return {
        "passed": not violations,
        "constraint": "world_center_distance_at_least_ball_diameter_minus_camera_tolerance",
        "diameter_mm": round(diameter, 4),
        "camera_tolerance_mm": round(tolerance, 4),
        "camera_model_approximate": approximate,
        "minimum_distance_mm": (
            None if not distances else round(float(min(distances)), 4)
        ),
        "violation_count": len(violations),
        "violations": violations,
    }


def _solution_stability(
    solutions: list[dict[str, Any]],
    *,
    selected: dict[str, Any],
) -> dict[str, Any]:
    reference = np.asarray(selected["centers_px"], dtype=np.float64)
    deltas: list[np.ndarray] = []
    for solution in solutions:
        centers = np.asarray(solution["centers_px"], dtype=np.float64)
        if centers.shape == reference.shape:
            deltas.append(np.linalg.norm(centers - reference, axis=1))
    if not deltas:
        return {"status": "unavailable", "stable": False}
    stacked = np.vstack(deltas)
    return {
        "status": "computed",
        "stable": True,
        "solution_count": len(deltas),
        "mean_center_spread_px": round(float(np.mean(stacked)), 4),
        "max_center_spread_px": round(float(np.max(stacked)), 4),
        "per_ball_max_spread_px": np.max(stacked, axis=0).round(4).tolist(),
    }


def _existence_hypotheses(
    selected: dict[str, Any],
    *,
    nodes: list[dict[str, Any]],
    shape: dict[str, Any],
    highlight_evidence: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    centers = np.asarray(selected["centers_px"], dtype=np.float64)
    full_objective = _ownership_only_objective(selected["ownership"], settings)
    highlights = highlight_evidence.get("by_ball_id") or {}
    rows: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        reduced_nodes = nodes[:index] + nodes[index + 1 :]
        reduced_centers = np.delete(centers, index, axis=0)
        reduced = _assign(
            reduced_nodes,
            reduced_centers,
            shape=shape,
            settings=settings,
        )
        marginal = _ownership_only_objective(reduced, settings) - full_objective
        owned = (selected["ownership"].get("by_ball_id") or {}).get(
            str(node["ball_id"]),
            {},
        )
        highlight = highlights.get(str(node["ball_id"])) or {}
        rows.append(
            {
                "ball_id": int(node["ball_id"]),
                "owned_boundary_point_count": int(
                    owned.get("owned_point_count") or 0
                ),
                "owned_boundary_rms_px": owned.get("rms_residual_px"),
                "arc_coverage_fraction": (
                    selected.get("arc_coverage_by_ball_id", {})
                    .get(str(node["ball_id"]), {})
                    .get("coverage_fraction")
                ),
                "leave_one_out_cost_increase": round(float(marginal), 6),
                "unique_highlight_support": bool(highlight.get("assigned")),
                "highlight_normalized_radius": highlight.get(
                    "normalized_radius"
                ),
                "decision": "supported" if marginal > 0.01 else "uncertain",
            }
        )
    return {
        "status": "computed",
        "model": "leave_one_hypothesis_out_global_pixel_explanation",
        "hypotheses": rows,
        "note": (
            "A detection is not deleted merely because another packing looks "
            "tidier. Its unique pixel support and optional specular-highlight "
            "support are measured explicitly."
        ),
    }


def _duplicate_hypotheses(
    selected: dict[str, Any],
    *,
    nodes: list[dict[str, Any]],
    existence: dict[str, Any],
    camera_model: Any,
    ball_radius_mm: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    centers = np.asarray(selected["centers_px"], dtype=np.float64)
    evidence = {
        int(item["ball_id"]): item
        for item in existence.get("hypotheses") or []
    }
    diameter = 2.0 * float(ball_radius_mm)
    factor = float(settings.get("generic_duplicate_distance_factor", 0.45))
    hypotheses: list[dict[str, Any]] = []
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            distance_mm = _world_distance(
                centers[left],
                centers[right],
                camera_model=camera_model,
                z_mm=ball_radius_mm,
            )
            if distance_mm is None or distance_mm > diameter * factor:
                continue
            left_id = int(nodes[left]["ball_id"])
            right_id = int(nodes[right]["ball_id"])
            left_support = evidence.get(left_id, {})
            right_support = evidence.get(right_id, {})
            left_strength = _existence_strength(left_support, nodes[left])
            right_strength = _existence_strength(right_support, nodes[right])
            if left_strength <= right_strength:
                suppress_id, keep_id = left_id, right_id
                weak, strong = left_strength, right_strength
                weak_payload = left_support
            else:
                suppress_id, keep_id = right_id, left_id
                weak, strong = right_strength, left_strength
                weak_payload = right_support
            same_label = str(nodes[left].get("label")) == str(nodes[right].get("label"))
            strong_ratio = strong / max(weak, 1e-6)
            decision = "unresolved"
            if (
                bool(settings.get("generic_duplicate_suppression_enabled", True))
                and same_label
                and strong_ratio >= float(
                    settings.get("generic_duplicate_min_strength_ratio", 1.8)
                )
                and not bool(weak_payload.get("unique_highlight_support"))
            ):
                decision = "suppress"
            hypotheses.append(
                {
                    "left_ball_id": left_id,
                    "right_ball_id": right_id,
                    "distance_mm": round(float(distance_mm), 4),
                    "same_label": same_label,
                    "keep_ball_id": keep_id,
                    "suppress_ball_id": suppress_id,
                    "support_strength_ratio": round(float(strong_ratio), 4),
                    "decision": decision,
                }
            )
    return {
        "status": "computed",
        "hypothesis_count": len(hypotheses),
        "suppressed_count": sum(
            item["decision"] == "suppress" for item in hypotheses
        ),
        "hypotheses": hypotheses,
    }


def _missing_hypotheses(
    selected: dict[str, Any],
    *,
    nodes: list[dict[str, Any]],
    shape: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    unowned = selected["ownership"].get("unowned_points_px") or []
    clusters = _point_clusters(
        unowned,
        link_px=float(settings.get("generic_missing_point_link_px", 7.0)),
        minimum_count=int(settings.get("generic_missing_min_point_count", 14)),
    )
    active_centers = np.asarray(selected["centers_px"], dtype=np.float64)
    minimum_separation = float(shape["minor_axis_px"]) * float(
        settings.get("generic_missing_min_center_separation_factor", 0.70)
    )
    candidates: list[dict[str, Any]] = []
    for cluster in clusters:
        fit = fit_fixed_shape_ellipse_center(
            cluster,
            major_axis_px=float(shape["major_axis_px"]),
            minor_axis_px=float(shape["minor_axis_px"]),
            angle_deg=float(shape["angle_deg"]),
            fallback_center_px=np.mean(cluster, axis=0),
            source="generic_cluster_unexplained_arc_missing_hypothesis",
        )
        if fit is None:
            continue
        center = np.asarray(fit["center_px"], dtype=np.float64)
        nearest = float(np.min(np.linalg.norm(active_centers - center, axis=1)))
        if nearest < minimum_separation:
            continue
        candidates.append(
            {
                "center_px": _round_point(center),
                "supporting_unowned_point_count": int(len(cluster)),
                "nearest_existing_center_px": round(nearest, 4),
                "decision": "diagnostic_only",
                "reason": "unexplained_arc_cluster_requires_new_detector_confirmation",
            }
        )
    return {
        "status": "diagnostic_only" if candidates else "none_observed",
        "hypothesis_count": len(candidates),
        "hypotheses": candidates,
        "promotion_enabled": False,
        "note": (
            "Raw samples originate near existing candidates, so an unexplained "
            "arc may suggest a missing ball but cannot create one automatically."
        ),
    }


def _promotion_gate(
    *,
    component: dict[str, Any],
    nodes: list[dict[str, Any]],
    baseline: dict[str, Any],
    selected: dict[str, Any],
    stability: dict[str, Any],
    duplicates: dict[str, Any],
    suppressed_ids: list[int],
    missing: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    minimum_size = int(settings.get("generic_promotion_min_component_size", 5))
    if len(nodes) < minimum_size:
        failures.append("component_too_small_for_measured_generic_promotion")
    if not bool(settings.get("generic_joint_promotion_enabled", True)):
        failures.append("generic_joint_promotion_disabled")
    minimum_points = int(settings.get("generic_promotion_min_owned_points", 5))
    required_fraction = float(
        settings.get("generic_promotion_min_supported_node_fraction", 0.80)
    )
    supported = sum(
        int(payload.get("owned_point_count") or 0) >= minimum_points
        for payload in (selected["ownership"].get("by_ball_id") or {}).values()
    )
    if supported < int(np.ceil(required_fraction * len(nodes))):
        failures.append("too_few_nodes_have_unique_global_boundary_support")
    if float(selected.get("owned_fraction") or 0.0) < float(
        settings.get("generic_promotion_min_owned_fraction", 0.65)
    ):
        failures.append("global_boundary_owned_fraction_is_too_low")
    if not bool(selected.get("hard_constraints_passed")):
        failures.append("hard_world_non_overlap_failed")
    if float(stability.get("max_center_spread_px") or np.inf) > float(
        settings.get("generic_promotion_max_start_spread_px", 2.5)
    ):
        failures.append("multiple_start_solutions_disagree")
    improvement = float(baseline["objective"]) - float(selected["objective"])
    if improvement < float(
        settings.get("generic_promotion_min_objective_improvement", 0.0005)
    ):
        failures.append("joint_solution_does_not_improve_pixel_objective")
    unresolved_duplicates = [
        item
        for item in duplicates.get("hypotheses") or []
        if item.get("decision") == "unresolved"
    ]
    if unresolved_duplicates:
        failures.append("duplicate_hypothesis_is_unresolved")
    if missing.get("hypothesis_count") and bool(
        settings.get("generic_abstain_on_missing_hypothesis", True)
    ):
        failures.append("missing_ball_hypothesis_is_unresolved")
    passed = not failures
    reasons = failures or [
        "multiple_independent_starts_converged",
        "global_boundary_ownership_supports_component",
        "hard_world_non_overlap_passed",
        "joint_pixel_objective_improved",
    ]
    if suppressed_ids and passed:
        reasons.append("strong_duplicate_hypothesis_suppressed")
    return {
        "passed": passed,
        "reasons": reasons,
        "component_type": component.get("cluster_type"),
        "supported_node_count": int(supported),
        "required_supported_node_fraction": round(required_fraction, 4),
        "objective_improvement": round(improvement, 7),
        "suppressed_ball_ids": suppressed_ids,
    }


def _component_result(
    *,
    component: dict[str, Any],
    nodes: list[dict[str, Any]],
    shape: dict[str, Any],
    baseline: dict[str, Any],
    solutions: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    highlight_evidence: dict[str, Any],
    promoted: bool,
    reasons: list[str],
    settings: dict[str, Any],
    all_nodes: list[dict[str, Any]] | None = None,
    stability: dict[str, Any] | None = None,
    existence: dict[str, Any] | None = None,
    duplicates: dict[str, Any] | None = None,
    missing: dict[str, Any] | None = None,
    suppressed_ids: list[int] | None = None,
    suppression_solution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    all_nodes = all_nodes or nodes
    suppressed_ids = suppressed_ids or []
    by_ball_id: dict[str, dict[str, Any]] = {}
    active_index = {int(node["ball_id"]): index for index, node in enumerate(nodes)}
    selected_ownership = (selected or {}).get("ownership") or {}
    selected_centers = np.asarray(
        (selected or {}).get("centers_px") or [],
        dtype=np.float64,
    ).reshape(-1, 2)
    for node in all_nodes:
        ball_id = int(node["ball_id"])
        if ball_id in suppressed_ids:
            by_ball_id[str(ball_id)] = {
                "status": "suppressed_duplicate",
                "promoted": False,
                "suppressed": True,
                "solver_mode": "generic_multi_start_global_cluster",
                "component_id": component.get("component_id"),
                "component_size": len(all_nodes),
                "promotion_reasons": ["strong_duplicate_hypothesis_suppressed"],
            }
            continue
        index = active_index.get(ball_id)
        if index is None or selected is None:
            by_ball_id[str(ball_id)] = {
                "status": "abstained",
                "promoted": False,
                "component_id": component.get("component_id"),
                "promotion_reasons": reasons,
            }
            continue
        center = selected_centers[index]
        owned = (selected_ownership.get("by_ball_id") or {}).get(
            str(ball_id),
            {},
        )
        by_ball_id[str(ball_id)] = {
            "status": "promoted" if promoted else "abstained",
            "promoted": promoted,
            "suppressed": False,
            "solver_mode": "generic_multi_start_global_cluster",
            "component_id": component.get("component_id"),
            "component_size": len(all_nodes),
            "initial_source_center_px": _round_point(node["center_px"]),
            "proposed_source_center_px": _round_point(center),
            "movement_px": round(
                float(np.linalg.norm(center - np.asarray(node["center_px"]))),
                4,
            ),
            "ellipse_fit": {
                "status": "candidate",
                **_ellipse(center, shape),
                "axis_ratio": round(
                    float(shape["major_axis_px"])
                    / max(float(shape["minor_axis_px"]), 1e-6),
                    4,
                ),
                "source": "generic_cluster_global_arc_fixed_shape",
            },
            "owned_boundary_points_px": owned.get("owned_points_px") or [],
            "owned_boundary_point_count": int(owned.get("owned_point_count") or 0),
            "owned_boundary_rms_px": owned.get("rms_residual_px"),
            "arc_coverage": (
                selected.get("arc_coverage_by_ball_id", {}).get(str(ball_id))
            ),
            "promotion_reasons": reasons,
            "shape_consensus": shape,
        }
    return {
        "status": "promoted" if promoted else "abstained",
        "promoted": promoted,
        "model": "generic_multi_start_global_cluster",
        "component_id": component.get("component_id"),
        "component_type": component.get("cluster_type"),
        "member_ids": [int(node["ball_id"]) for node in all_nodes],
        "active_member_ids": [int(node["ball_id"]) for node in nodes],
        "suppressed_ball_ids": suppressed_ids,
        "shape_consensus": shape,
        "baseline": _compact_solution(baseline),
        "starting_solutions": [_compact_solution(item) for item in solutions],
        "selected_start": None if selected is None else selected.get("name"),
        "selected_solution": (
            None if selected is None else _compact_solution(selected, include_ownership=True)
        ),
        "hard_non_overlap": (
            {"passed": False, "status": "unavailable"}
            if selected is None
            else selected.get("hard_non_overlap")
        ),
        "solution_stability": stability or {"status": "unavailable"},
        "highlight_existence_evidence": highlight_evidence,
        "existence_hypotheses": existence or {"status": "unavailable"},
        "duplicate_hypotheses": duplicates or {
            "status": "computed",
            "hypothesis_count": 0,
            "hypotheses": [],
        },
        "missing_hypotheses": missing or {
            "status": "none_observed",
            "hypothesis_count": 0,
            "hypotheses": [],
        },
        "suppression_solution": (
            None
            if suppression_solution is None
            else _compact_solution(suppression_solution)
        ),
        "promotion_gate": {"passed": promoted, "reasons": reasons},
        "global_boundary_ownership": selected_ownership,
        "by_ball_id": by_ball_id,
        "note": (
            "No triangle or traversal order is assumed. All active centers are "
            "refined together from one globally owned boundary-sample pool."
        ),
    }


def _highlight_evidence(
    source_image: np.ndarray | None,
    *,
    nodes: list[dict[str, Any]],
    shape: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    if source_image is None or source_image.ndim != 3:
        return {"status": "unavailable", "by_ball_id": {}}
    centers = np.asarray([node["center_px"] for node in nodes], dtype=np.float64)
    margin = int(np.ceil(float(shape["major_axis_px"])))
    x0 = max(0, int(np.floor(np.min(centers[:, 0]) - margin)))
    y0 = max(0, int(np.floor(np.min(centers[:, 1]) - margin)))
    x1 = min(source_image.shape[1], int(np.ceil(np.max(centers[:, 0]) + margin)))
    y1 = min(source_image.shape[0], int(np.ceil(np.max(centers[:, 1]) + margin)))
    if x1 <= x0 or y1 <= y0:
        return {"status": "unavailable", "by_ball_id": {}}
    hsv = cv2.cvtColor(source_image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 2] >= int(settings.get("generic_highlight_min_value", 190)))
        & (hsv[:, :, 1] <= int(settings.get("generic_highlight_max_saturation", 150)))
    ).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    ellipse_area = np.pi * float(shape["major_axis_px"]) * float(shape["minor_axis_px"]) / 4.0
    minimum_area = max(4, int(ellipse_area * 0.002))
    maximum_area = max(minimum_area + 1, int(ellipse_area * 0.22))
    blobs: list[dict[str, Any]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if not minimum_area <= area <= maximum_area:
            continue
        centroid = np.asarray(centroids[index], dtype=np.float64) + [x0, y0]
        blobs.append({"center_px": centroid, "area_px": area})

    candidates: list[tuple[float, int, int]] = []
    maximum_radius = float(settings.get("generic_highlight_max_normalized_radius", 0.45))
    for node_index, node in enumerate(nodes):
        for blob_index, blob in enumerate(blobs):
            radius = _normalized_ellipse_radius(
                blob["center_px"],
                center=np.asarray(node["center_px"], dtype=np.float64),
                shape=shape,
            )
            if radius <= maximum_radius:
                candidates.append((radius, node_index, blob_index))
    assigned_nodes: set[int] = set()
    assigned_blobs: set[int] = set()
    by_ball_id: dict[str, dict[str, Any]] = {
        str(node["ball_id"]): {"assigned": False}
        for node in nodes
    }
    for radius, node_index, blob_index in sorted(candidates):
        if node_index in assigned_nodes or blob_index in assigned_blobs:
            continue
        assigned_nodes.add(node_index)
        assigned_blobs.add(blob_index)
        node = nodes[node_index]
        blob = blobs[blob_index]
        by_ball_id[str(node["ball_id"])] = {
            "assigned": True,
            "center_px": _round_point(blob["center_px"]),
            "area_px": int(blob["area_px"]),
            "normalized_radius": round(float(radius), 4),
        }
    return {
        "status": "computed",
        "method": "one_to_one_bright_low_saturation_core_components",
        "blob_count": len(blobs),
        "assigned_ball_count": len(assigned_nodes),
        "by_ball_id": by_ball_id,
        "note": (
            "Specular highlights are supporting existence evidence only; they "
            "never define the final center or silhouette."
        ),
    }


def _ownership_only_objective(
    ownership: dict[str, Any],
    settings: dict[str, Any],
) -> float:
    gate = max(1e-6, float(settings.get("global_arc_residual_gate_px", 5.0)))
    rows = ownership.get("points") or []
    if not rows:
        return 2.25
    values = [
        min(float(row.get("best_residual_px") or gate * 1.5), gate * 1.5) / gate
        for row in rows
    ]
    return float(np.mean(np.square(values)))


def _arc_coverage(
    points_px: list[Any],
    *,
    center: np.ndarray,
    shape: dict[str, Any],
) -> dict[str, Any]:
    points = _points(points_px)
    if len(points) == 0:
        return {
            "coverage_fraction": 0.0,
            "occupied_bin_count": 0,
            "bin_count": 24,
        }
    angle = np.deg2rad(float(shape["angle_deg"]))
    rotation = np.asarray(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    local = (points - center.reshape(1, 2)) @ rotation.T
    rx = float(shape["major_axis_px"]) / 2.0
    ry = float(shape["minor_axis_px"]) / 2.0
    theta = np.arctan2(local[:, 1] / max(ry, 1e-6), local[:, 0] / max(rx, 1e-6))
    bin_count = 24
    bins = np.floor((theta + np.pi) / (2.0 * np.pi) * bin_count).astype(int)
    occupied = len(set(np.clip(bins, 0, bin_count - 1).tolist()))
    return {
        "coverage_fraction": round(occupied / float(bin_count), 4),
        "occupied_bin_count": int(occupied),
        "bin_count": bin_count,
    }


def _point_clusters(
    points_px: list[Any],
    *,
    link_px: float,
    minimum_count: int,
) -> list[np.ndarray]:
    points = _points(points_px)
    if len(points) < minimum_count:
        return []
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    unseen = set(range(len(points)))
    clusters: list[np.ndarray] = []
    while unseen:
        start = unseen.pop()
        queue: deque[int] = deque([start])
        members = [start]
        while queue:
            current = queue.popleft()
            neighbors = [
                index
                for index in list(unseen)
                if distances[current, index] <= link_px
            ]
            for index in neighbors:
                unseen.remove(index)
                queue.append(index)
                members.append(index)
        if len(members) >= minimum_count:
            clusters.append(points[members])
    return clusters


def _existence_strength(payload: dict[str, Any], node: dict[str, Any]) -> float:
    marginal = max(0.0, float(payload.get("leave_one_out_cost_increase") or 0.0))
    points = float(payload.get("owned_boundary_point_count") or 0.0)
    highlight = 12.0 if payload.get("unique_highlight_support") else 0.0
    raw = min(20.0, len(node.get("raw_points_px") or []) / 6.0)
    return marginal * 50.0 + points + highlight + raw


def _world_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    camera_model: Any,
    z_mm: float,
) -> float | None:
    try:
        left_world = np.asarray(
            camera_model.image_point_to_world_plane(left, z_mm),
            dtype=np.float64,
        ).reshape(-1)
        right_world = np.asarray(
            camera_model.image_point_to_world_plane(right, z_mm),
            dtype=np.float64,
        ).reshape(-1)
    except (AttributeError, TypeError, ValueError):
        return None
    if len(left_world) < 2 or len(right_world) < 2:
        return None
    return float(np.linalg.norm(left_world[:2] - right_world[:2]))


def _normalized_ellipse_radius(
    point: np.ndarray,
    *,
    center: np.ndarray,
    shape: dict[str, Any],
) -> float:
    delta = np.asarray(point, dtype=np.float64) - center
    angle = np.deg2rad(float(shape["angle_deg"]))
    local_x = delta[0] * np.cos(angle) + delta[1] * np.sin(angle)
    local_y = -delta[0] * np.sin(angle) + delta[1] * np.cos(angle)
    return float(
        np.hypot(
            local_x / max(float(shape["major_axis_px"]) / 2.0, 1e-6),
            local_y / max(float(shape["minor_axis_px"]) / 2.0, 1e-6),
        )
    )


def _ellipse(center: np.ndarray, shape: dict[str, Any]) -> dict[str, Any]:
    return {
        "center_px": _round_point(center),
        "major_axis_px": round(float(shape["major_axis_px"]), 4),
        "minor_axis_px": round(float(shape["minor_axis_px"]), 4),
        "angle_deg": round(float(shape["angle_deg"]), 4),
    }


def _all_raw_points(nodes: list[dict[str, Any]]) -> list[list[float]]:
    return [
        [float(point[0]), float(point[1])]
        for node in nodes
        for point in node.get("raw_points_px") or []
    ]


def _points(values: list[Any]) -> np.ndarray:
    try:
        points = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return np.empty((0, 2), dtype=np.float64)
    return points[np.all(np.isfinite(points), axis=1)]


def _clamp_from_anchor(
    point: np.ndarray,
    anchor: np.ndarray,
    maximum_shift: float,
) -> np.ndarray:
    shift = np.asarray(point, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
    length = float(np.linalg.norm(shift))
    if length <= maximum_shift or length <= 1e-9:
        return np.asarray(point, dtype=np.float64)
    return np.asarray(anchor, dtype=np.float64) + shift * (maximum_shift / length)


def _compact_solution(
    solution: dict[str, Any],
    *,
    include_ownership: bool = False,
) -> dict[str, Any]:
    result = {
        key: solution.get(key)
        for key in (
            "name",
            "status",
            "objective",
            "residual_term",
            "owned_fraction",
            "owned_point_count",
            "ambiguous_point_count",
            "unowned_point_count",
            "mean_movement_px",
            "max_movement_px",
            "hard_constraints_passed",
            "hard_non_overlap",
        )
        if key in solution
    }
    if include_ownership:
        result["centers_px"] = solution.get("centers_px")
        result["arc_coverage_by_ball_id"] = solution.get(
            "arc_coverage_by_ball_id"
        )
    return result


def _abstained(
    component: dict[str, Any],
    nodes: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    return {
        "status": "abstained",
        "promoted": False,
        "model": "generic_multi_start_global_cluster",
        "component_id": component.get("component_id"),
        "component_type": component.get("cluster_type"),
        "member_ids": [int(node["ball_id"]) for node in nodes],
        "active_member_ids": [int(node["ball_id"]) for node in nodes],
        "suppressed_ball_ids": [],
        "duplicate_hypotheses": {
            "status": "unavailable",
            "hypothesis_count": 0,
            "hypotheses": [],
        },
        "missing_hypotheses": {
            "status": "unavailable",
            "hypothesis_count": 0,
            "hypotheses": [],
        },
        "promotion_gate": {"passed": False, "reasons": [reason]},
        "by_ball_id": {
            str(node["ball_id"]): {
                "status": "abstained",
                "promoted": False,
                "component_id": component.get("component_id"),
                "promotion_reasons": [reason],
            }
            for node in nodes
        },
    }


def _round_point(point: np.ndarray | list[float]) -> list[float]:
    value = np.asarray(point, dtype=np.float64).reshape(2)
    return [round(float(value[0]), 4), round(float(value[1]), 4)]


__all__ = ["solve_generic_cluster_component"]
