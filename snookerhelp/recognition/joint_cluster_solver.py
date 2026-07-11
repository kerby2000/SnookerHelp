from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from snookerhelp.recognition.arc_combo_fit import fit_fixed_shape_ellipse_center
from snookerhelp.recognition.cluster_arc_assignment import (
    assign_boundary_points_globally,
)
from snookerhelp.recognition.cluster_graph import build_cluster_graph

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - dependency guard for old installations
    linear_sum_assignment = None


def solve_joint_cluster_components(
    balls: list[dict[str, Any]],
    *,
    camera_model: Any,
    ball_radius_mm: float,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve connected ball components from shared pixels and physics.

    The old cluster experiments fitted one ball and then adjusted the next one.
    That ordering lets a bad fit propagate.  This solver instead builds every
    proposal from the same independent input state, assigns the union of image
    boundary samples globally, and evaluates a component as one solution.

    The first promoted mode is deliberately narrow: an intact 15-red rack.  A
    generic component still receives global ownership diagnostics, but remains
    non-promoting until annotated arbitrary-cluster images provide a gate.
    """

    cfg = settings or {}
    if not bool(cfg.get("enabled", True)):
        return _empty_result("disabled", enabled=False)
    if linear_sum_assignment is None:
        return _empty_result(
            "scipy_unavailable",
            enabled=True,
            reasons=["scipy is required for deterministic global assignment"],
        )

    graph = build_cluster_graph(
        balls,
        ball_radius_mm=ball_radius_mm,
        settings=cfg,
    )
    by_raw_id = {_ball_id(ball): ball for ball in balls if _ball_id(ball) > 0}
    component_results: list[dict[str, Any]] = []
    by_ball_id: dict[str, dict[str, Any]] = {}

    for component in graph.get("components") or []:
        member_ids = [
            int(ball_id)
            for ball_id in component.get("member_ids") or []
            if int(ball_id) in by_raw_id
        ]
        members = [by_raw_id[ball_id] for ball_id in member_ids]
        if len(members) < 2:
            continue
        if _is_intact_red_rack_candidate(members, cfg):
            solved = _solve_intact_red_rack(
                members,
                component=component,
                camera_model=camera_model,
                ball_radius_mm=ball_radius_mm,
                settings=cfg,
            )
        else:
            solved = _diagnose_generic_component(
                members,
                component=component,
                settings=cfg,
            )
        component_results.append(solved)
        for ball_id, payload in (solved.get("by_ball_id") or {}).items():
            by_ball_id[str(ball_id)] = payload

    promoted_components = [
        component for component in component_results if component.get("promoted")
    ]
    return {
        "status": (
            "promoted"
            if promoted_components
            else ("diagnostic_only" if component_results else "no_cluster_components")
        ),
        "enabled": True,
        "model": "joint_global_arc_cluster_solver",
        "component_count": len(component_results),
        "promoted_component_count": len(promoted_components),
        "graph": graph,
        "components": component_results,
        "by_ball_id": by_ball_id,
        "note": (
            "All component members are evaluated from the same independent "
            "input state. Sequential traversal and legacy arc add-back fits "
            "are not used as final truth."
        ),
    }


def _solve_intact_red_rack(
    members: list[dict[str, Any]],
    *,
    component: dict[str, Any],
    camera_model: Any,
    ball_radius_mm: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    nodes = [_member_input(ball) for ball in members]
    nodes = [node for node in nodes if node is not None]
    if len(nodes) != 15:
        return _failed_component(
            component,
            "intact_rack_requires_15_valid_source_centers",
        )

    shape = _shared_shape_consensus(nodes, settings)
    if shape.get("status") != "computed":
        return _failed_component(component, "no_reliable_shared_shape", shape=shape)

    lattice = _fit_triangular_lattice(nodes, shape=shape, settings=settings)
    if lattice.get("status") != "computed":
        return _failed_component(
            component,
            str(lattice.get("reason") or "lattice_fit_failed"),
            shape=shape,
            lattice=lattice,
        )

    ownership = _refine_lattice_from_global_arcs(
        nodes,
        lattice=lattice,
        shape=shape,
        settings=settings,
    )
    final_centers = np.asarray(
        ownership.get("refined_centers_px") or lattice["template_centers_px"],
        dtype=np.float64,
    ).reshape(-1, 2)
    initial_centers = np.asarray(
        [node["center_px"] for node in nodes],
        dtype=np.float64,
    )
    assignment = _assign(initial_centers, final_centers)
    if assignment is None:
        return _failed_component(component, "final_assignment_failed")
    _, assigned_node_by_member, movement_px = assignment

    geometry = _component_geometry(
        final_centers,
        cells=np.asarray(lattice["template_cells"], dtype=np.int64),
        camera_model=camera_model,
        ball_radius_mm=ball_radius_mm,
    )
    gates = _rack_promotion_gates(
        lattice=lattice,
        ownership=ownership,
        geometry=geometry,
        settings=settings,
    )
    promoted = bool(gates["passed"])

    by_ball_id: dict[str, dict[str, Any]] = {}
    ownership_by_node = ownership.get("by_node_id") or {}
    for member_index, node in enumerate(nodes):
        node_index = int(assigned_node_by_member[member_index])
        proposed = final_centers[node_index]
        owned = ownership_by_node.get(str(node_index + 1), {})
        ellipse = {
            "status": "candidate",
            "center_px": _round_point(proposed),
            "major_axis_px": round(float(shape["major_axis_px"]), 4),
            "minor_axis_px": round(float(shape["minor_axis_px"]), 4),
            "angle_deg": round(float(shape["angle_deg"]), 4),
            "axis_ratio": round(
                float(shape["major_axis_px"]) / float(shape["minor_axis_px"]),
                4,
            ),
            "source": "joint_cluster_global_arc_fixed_shape",
        }
        by_ball_id[str(node["ball_id"])] = {
            "status": "promoted" if promoted else "diagnostic_only",
            "promoted": promoted,
            "solver_mode": "intact_red_rack_global_arc_joint_fit",
            "component_id": component.get("component_id"),
            "component_size": len(nodes),
            "initial_source_center_px": _round_point(node["center_px"]),
            "proposed_source_center_px": _round_point(proposed),
            "movement_px": round(float(movement_px[member_index]), 4),
            "assigned_lattice_node": node_index + 1,
            "lattice_cell": [
                int(value) for value in lattice["template_cells"][node_index]
            ],
            "ellipse_fit": ellipse,
            "owned_boundary_points_px": owned.get("owned_points_px") or [],
            "owned_boundary_point_count": int(owned.get("owned_point_count") or 0),
            "owned_boundary_rms_px": owned.get("rms_residual_px"),
            "promotion_reasons": gates["reasons"],
            "shape_consensus": shape,
            "lattice_quality": _compact_lattice_quality(lattice),
            "geometry": geometry,
        }

    return {
        "status": "promoted" if promoted else "diagnostic_only",
        "promoted": promoted,
        "model": "intact_red_rack_global_arc_joint_fit",
        "component_id": component.get("component_id"),
        "member_ids": [int(node["ball_id"]) for node in nodes],
        "shape_consensus": shape,
        "lattice": lattice,
        "global_boundary_ownership": {
            key: value
            for key, value in ownership.items()
            if key != "refined_centers_px"
        },
        "geometry": geometry,
        "promotion_gate": gates,
        "by_ball_id": by_ball_id,
        "note": (
            "The 15 red hypotheses are assigned simultaneously to one compact "
            "triangular packing. The final local refinement uses the deduplicated "
            "union of all raw boundary samples, not one ball's radial rays."
        ),
    }


def _diagnose_generic_component(
    members: list[dict[str, Any]],
    *,
    component: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    nodes = [_member_input(ball) for ball in members]
    nodes = [node for node in nodes if node is not None]
    shape = _shared_shape_consensus(nodes, settings)
    if not nodes or shape.get("status") != "computed":
        return _failed_component(
            component,
            "generic_component_has_insufficient_shape_evidence",
            shape=shape,
        )
    ellipses = {
        int(node["ball_id"]): {
            "center_px": _round_point(node["center_px"]),
            "major_axis_px": shape["major_axis_px"],
            "minor_axis_px": shape["minor_axis_px"],
            "angle_deg": shape["angle_deg"],
        }
        for node in nodes
    }
    ownership = assign_boundary_points_globally(
        _all_raw_points(nodes),
        ellipses_by_id=ellipses,
        residual_gate_px=float(settings.get("global_arc_residual_gate_px", 5.0)),
        ambiguity_margin_px=float(
            settings.get("global_arc_ambiguity_margin_px", 0.6)
        ),
    )
    by_ball_id = {
        str(node["ball_id"]): {
            "status": "diagnostic_only",
            "promoted": False,
            "solver_mode": "generic_global_arc_ownership",
            "component_id": component.get("component_id"),
            "component_size": len(nodes),
            "initial_source_center_px": _round_point(node["center_px"]),
            "proposed_source_center_px": _round_point(node["center_px"]),
            "owned_boundary_point_count": int(
                ((ownership.get("by_ball_id") or {}).get(str(node["ball_id"]), {})).get(
                    "owned_point_count"
                )
                or 0
            ),
            "promotion_reasons": [
                "generic_component_promotion_waits_for_annotated_cluster_gate"
            ],
        }
        for node in nodes
    }
    return {
        "status": "diagnostic_only",
        "promoted": False,
        "model": "generic_global_arc_ownership",
        "component_id": component.get("component_id"),
        "member_ids": [int(node["ball_id"]) for node in nodes],
        "shape_consensus": shape,
        "global_boundary_ownership": ownership,
        "promotion_gate": {
            "passed": False,
            "reasons": [
                "generic_component_promotion_waits_for_annotated_cluster_gate"
            ],
        },
        "by_ball_id": by_ball_id,
    }


def _fit_triangular_lattice(
    nodes: list[dict[str, Any]],
    *,
    shape: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    centers = np.asarray([node["center_px"] for node in nodes], dtype=np.float64)
    spacing_seed = float(shape["minor_axis_px"])
    vectors: list[tuple[np.ndarray, float, float]] = []
    for left in range(len(centers)):
        for right in range(left + 1, len(centers)):
            vector = centers[right] - centers[left]
            length = float(np.linalg.norm(vector))
            if not 0.85 * spacing_seed <= length <= 1.15 * spacing_seed:
                continue
            angle = float(np.degrees(np.arctan2(vector[1], vector[0])) % 180.0)
            vectors.append((vector, length, angle))
    if len(vectors) < int(settings.get("rack_min_contact_vectors", 8)):
        return {
            "status": "failed",
            "reason": "not_enough_contact_vectors",
            "contact_vector_count": len(vectors),
        }

    phase_steps = max(60, int(settings.get("rack_phase_search_steps", 240)))
    phases = np.linspace(0.0, 60.0, phase_steps, endpoint=False)
    phase_scores = [
        sum(
            np.exp(-(_nearest_hex_axis_delta(angle, phase) / 4.0) ** 2)
            for _, _, angle in vectors
        )
        for phase in phases
    ]
    phase = float(phases[int(np.argmax(phase_scores))])
    aligned_lengths = [
        length
        for _, length, angle in vectors
        if _nearest_hex_axis_delta(angle, phase) <= 6.0
    ]
    if len(aligned_lengths) < 6:
        return {
            "status": "failed",
            "reason": "contact_directions_are_not_hexagonal",
            "phase_deg": round(phase, 4),
            "aligned_vector_count": len(aligned_lengths),
        }
    spacing = float(np.median(aligned_lengths))
    basis = np.column_stack(
        [
            _direction_vector(phase, spacing),
            _direction_vector(phase + 60.0, spacing),
        ]
    )
    if abs(float(np.linalg.det(basis))) < 1.0:
        return {"status": "failed", "reason": "degenerate_lattice_basis"}

    templates = _triangular_templates(side=5)
    best: dict[str, Any] | None = None
    inverse = np.linalg.inv(basis)
    robust_limit = float(settings.get("rack_assignment_robust_limit_factor", 0.45)) * spacing
    anchor_limit = max(
        8.0,
        float(settings.get("rack_anchor_residual_factor", 0.16)) * spacing,
    )

    for origin in centers:
        coordinates = (inverse @ (centers - origin).T).T
        rounded = np.rint(coordinates).astype(np.int64)
        for orientation, template in enumerate(templates):
            translations = {
                tuple((observed - cell).tolist())
                for observed in rounded
                for cell in template
            }
            for translation in translations:
                cells = template + np.asarray(translation, dtype=np.int64)
                template_centers = origin + cells @ basis.T
                assignment = _assign(centers, template_centers)
                if assignment is None:
                    continue
                _, node_by_member, distances = assignment
                anchor_mask = distances <= anchor_limit
                anchor_count = int(anchor_mask.sum())
                anchor_rms = (
                    float(np.sqrt(np.mean(distances[anchor_mask] ** 2)))
                    if anchor_count
                    else float("inf")
                )
                robust_score = float(
                    np.sum(np.minimum(distances, robust_limit) ** 2)
                )
                # Preserve a consensus of accurate independent detections.
                # A compromise translation that makes every member moderately
                # wrong must never beat a lattice supported by many near-zero
                # anchors plus a few gross outliers.
                selection_key = (-anchor_count, anchor_rms, robust_score)
                if best is None or selection_key < best["selection_key"]:
                    best = {
                        "score": robust_score,
                        "selection_key": selection_key,
                        "orientation": int(orientation),
                        "origin_px": origin.copy(),
                        "cells": cells.copy(),
                        "centers": template_centers.copy(),
                        "node_by_member": node_by_member.copy(),
                        "distances": distances.copy(),
                    }
    if best is None:
        return {"status": "failed", "reason": "no_triangular_assignment"}

    for _ in range(2):
        anchor_mask = best["distances"] <= anchor_limit
        if int(anchor_mask.sum()) < 6:
            break
        assigned_cells = best["cells"][best["node_by_member"]]
        design = np.column_stack(
            [
                np.ones(int(anchor_mask.sum()), dtype=np.float64),
                assigned_cells[anchor_mask, 0],
                assigned_cells[anchor_mask, 1],
            ]
        )
        coefficients = np.linalg.lstsq(
            design,
            centers[anchor_mask],
            rcond=None,
        )[0]
        template_design = np.column_stack(
            [
                np.ones(len(best["cells"]), dtype=np.float64),
                best["cells"][:, 0],
                best["cells"][:, 1],
            ]
        )
        template_centers = template_design @ coefficients
        assignment = _assign(centers, template_centers)
        if assignment is None:
            break
        _, node_by_member, distances = assignment
        best["centers"] = template_centers
        best["node_by_member"] = node_by_member
        best["distances"] = distances
        best["affine_coefficients"] = coefficients

    anchor_mask = best["distances"] <= anchor_limit
    anchor_rms = (
        float(np.sqrt(np.mean(best["distances"][anchor_mask] ** 2)))
        if np.any(anchor_mask)
        else float("inf")
    )
    return {
        "status": "computed",
        "method": "robust_hex_phase_plus_global_triangle_assignment",
        "phase_deg": round(phase, 4),
        "spacing_px": round(spacing, 4),
        "spacing_seed_px": round(spacing_seed, 4),
        "contact_vector_count": len(vectors),
        "aligned_vector_count": len(aligned_lengths),
        "phase_score": round(float(max(phase_scores)), 4),
        "basis_px": np.asarray(basis).round(4).tolist(),
        "template_orientation": int(best["orientation"]),
        "template_cells": np.asarray(best["cells"], dtype=int).tolist(),
        "template_centers_px": np.asarray(best["centers"]).round(4).tolist(),
        "assigned_node_by_member": [
            int(value) for value in best["node_by_member"]
        ],
        "member_assignment_residuals_px": np.asarray(best["distances"]).round(4).tolist(),
        "anchor_residual_limit_px": round(anchor_limit, 4),
        "anchor_count": int(anchor_mask.sum()),
        "anchor_rms_px": round(anchor_rms, 4),
        "outlier_member_count": int((~anchor_mask).sum()),
        "note": (
            "The lattice transform is fitted only from independently aligned "
            "members. Displaced hypotheses do not drag the rack pose."
        ),
    }


def _refine_lattice_from_global_arcs(
    nodes: list[dict[str, Any]],
    *,
    lattice: dict[str, Any],
    shape: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    template = np.asarray(lattice["template_centers_px"], dtype=np.float64)
    centers = template.copy()
    all_points = _all_raw_points(nodes)
    gate = float(settings.get("global_arc_residual_gate_px", 5.0))
    ambiguity = float(settings.get("global_arc_ambiguity_margin_px", 0.6))
    maximum_shift = float(settings.get("rack_arc_refine_max_shift_px", 5.5))
    minimum_points = int(settings.get("rack_arc_refine_min_points", 5))
    iterations = max(1, int(settings.get("rack_arc_refine_iterations", 2)))
    ownership: dict[str, Any] = {}

    for _ in range(iterations):
        models = {
            index + 1: {
                "center_px": _round_point(center),
                "major_axis_px": shape["major_axis_px"],
                "minor_axis_px": shape["minor_axis_px"],
                "angle_deg": shape["angle_deg"],
            }
            for index, center in enumerate(centers)
        }
        ownership = assign_boundary_points_globally(
            all_points,
            ellipses_by_id=models,
            residual_gate_px=gate,
            ambiguity_margin_px=ambiguity,
        )
        refined = centers.copy()
        for index, center in enumerate(centers):
            owned = (ownership.get("by_ball_id") or {}).get(str(index + 1), {})
            points = owned.get("owned_points_px") or []
            if len(points) < minimum_points:
                continue
            fit = fit_fixed_shape_ellipse_center(
                points,
                major_axis_px=float(shape["major_axis_px"]),
                minor_axis_px=float(shape["minor_axis_px"]),
                angle_deg=float(shape["angle_deg"]),
                fallback_center_px=center,
                seed_ellipse=models[index + 1],
                source="joint_cluster_global_arc_fixed_shape",
            )
            if fit is None:
                continue
            candidate = np.asarray(fit["center_px"], dtype=np.float64)
            shift = candidate - template[index]
            length = float(np.linalg.norm(shift))
            if length > maximum_shift:
                candidate = template[index] + shift * (maximum_shift / length)
            refined[index] = candidate
        centers = refined

    final_models = {
        index + 1: {
            "center_px": _round_point(center),
            "major_axis_px": shape["major_axis_px"],
            "minor_axis_px": shape["minor_axis_px"],
            "angle_deg": shape["angle_deg"],
        }
        for index, center in enumerate(centers)
    }
    ownership = assign_boundary_points_globally(
        all_points,
        ellipses_by_id=final_models,
        residual_gate_px=gate,
        ambiguity_margin_px=ambiguity,
    )
    return {
        "status": ownership.get("status"),
        "method": "global_arc_ownership_then_fixed_shape_center_refinement",
        "raw_union_point_count": int(len(all_points)),
        "refined_centers_px": centers.round(4).tolist(),
        "maximum_refinement_shift_px": round(maximum_shift, 4),
        "by_node_id": ownership.get("by_ball_id") or {},
        "owned_point_count": int(ownership.get("owned_point_count") or 0),
        "ambiguous_point_count": int(ownership.get("ambiguous_point_count") or 0),
        "unowned_point_count": int(ownership.get("unowned_point_count") or 0),
        "owned_fraction": ownership.get("owned_fraction"),
        "ambiguous_points_px": ownership.get("ambiguous_points_px") or [],
        "unowned_points_px": ownership.get("unowned_points_px") or [],
        "note": ownership.get("note"),
    }


def _component_geometry(
    centers_px: np.ndarray,
    *,
    cells: np.ndarray,
    camera_model: Any,
    ball_radius_mm: float,
) -> dict[str, Any]:
    world: list[np.ndarray | None] = []
    for center in centers_px:
        try:
            point = np.asarray(
                camera_model.image_point_to_world_plane(center, ball_radius_mm),
                dtype=np.float64,
            ).reshape(-1)
            world.append(point[:2] if len(point) >= 2 else None)
        except (TypeError, ValueError):
            world.append(None)

    neighbor_pairs: list[dict[str, Any]] = []
    hard_overlap_count = 0
    minimum_distance = float("inf")
    neighbor_errors: list[float] = []
    for left in range(len(centers_px)):
        for right in range(left + 1, len(centers_px)):
            if world[left] is None or world[right] is None:
                continue
            distance = float(np.linalg.norm(world[left] - world[right]))
            minimum_distance = min(minimum_distance, distance)
            if distance < 2.0 * ball_radius_mm * 0.78:
                hard_overlap_count += 1
            delta = cells[right] - cells[left]
            is_lattice_neighbor = tuple(delta.tolist()) in {
                (1, 0),
                (0, 1),
                (-1, 1),
                (-1, 0),
                (0, -1),
                (1, -1),
            }
            if not is_lattice_neighbor:
                continue
            error = distance - 2.0 * ball_radius_mm
            neighbor_errors.append(error)
            neighbor_pairs.append(
                {
                    "left_node": left + 1,
                    "right_node": right + 1,
                    "distance_mm": round(distance, 4),
                    "diameter_error_mm": round(error, 4),
                }
            )
    return {
        "status": "computed" if neighbor_pairs else "unavailable",
        "hard_overlap_count": int(hard_overlap_count),
        "minimum_center_distance_mm": (
            None if not np.isfinite(minimum_distance) else round(minimum_distance, 4)
        ),
        "lattice_neighbor_count": len(neighbor_pairs),
        "neighbor_distance_rms_error_mm": (
            None
            if not neighbor_errors
            else round(float(np.sqrt(np.mean(np.asarray(neighbor_errors) ** 2))), 4)
        ),
        "neighbor_pairs": neighbor_pairs,
        "camera_model": getattr(camera_model, "model_name", "unknown"),
        "camera_model_approximate": not bool(
            getattr(camera_model, "is_calibrated", False)
        ),
    }


def _rack_promotion_gates(
    *,
    lattice: dict[str, Any],
    ownership: dict[str, Any],
    geometry: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    minimum_anchors = int(settings.get("rack_promotion_min_anchor_count", 8))
    maximum_anchor_rms = float(settings.get("rack_promotion_max_anchor_rms_px", 6.0))
    minimum_owned_per_node = int(settings.get("rack_promotion_min_owned_points_per_node", 5))
    minimum_owned_nodes = int(settings.get("rack_promotion_min_owned_node_count", 14))
    if int(lattice.get("anchor_count") or 0) < minimum_anchors:
        failures.append("too_few_independent_lattice_anchors")
    anchor_rms = lattice.get("anchor_rms_px")
    if anchor_rms is None or float(anchor_rms) > maximum_anchor_rms:
        failures.append("lattice_anchor_residual_too_large")
    owned_nodes = sum(
        int(payload.get("owned_point_count") or 0) >= minimum_owned_per_node
        for payload in (ownership.get("by_node_id") or {}).values()
    )
    if owned_nodes < minimum_owned_nodes:
        failures.append("too_few_nodes_have_unique_boundary_support")
    if int(geometry.get("hard_overlap_count") or 0) > 0:
        failures.append("hard_world_overlap_remains")
    enabled = bool(settings.get("rack_joint_promotion_enabled", True))
    if not enabled:
        failures.append("rack_joint_promotion_disabled")
    passed = bool(enabled and not failures)
    reasons = list(failures)
    if passed:
        reasons = [
            "robust_rack_lattice_supported_by_independent_members",
            "global_arc_ownership_supports_all_nodes",
            "no_hard_world_overlap",
        ]
    return {
        "passed": passed,
        "reasons": reasons,
        "minimum_anchor_count": minimum_anchors,
        "maximum_anchor_rms_px": maximum_anchor_rms,
        "minimum_owned_points_per_node": minimum_owned_per_node,
        "minimum_owned_node_count": minimum_owned_nodes,
        "owned_node_count": int(owned_nodes),
    }


def _shared_shape_consensus(
    nodes: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    ellipses = [node.get("ellipse") for node in nodes]
    ellipses = [ellipse for ellipse in ellipses if isinstance(ellipse, dict)]
    maximum_ratio = float(settings.get("joint_shape_max_axis_ratio", 1.6))
    plausible = [
        ellipse
        for ellipse in ellipses
        if float(ellipse.get("major_axis_px") or 0.0) > 4.0
        and float(ellipse.get("minor_axis_px") or 0.0) > 4.0
        and float(ellipse.get("axis_ratio") or 999.0) <= maximum_ratio
    ]
    if len(plausible) < int(settings.get("joint_shape_min_member_count", 6)):
        return {
            "status": "insufficient_evidence",
            "plausible_member_count": len(plausible),
            "ellipse_count": len(ellipses),
        }
    major = float(np.median([ellipse["major_axis_px"] for ellipse in plausible]))
    minor = float(np.median([ellipse["minor_axis_px"] for ellipse in plausible]))
    angles = [float(ellipse.get("angle_deg") or 0.0) % 180.0 for ellipse in plausible]
    angle = min(
        angles,
        key=lambda candidate: sum(_angle_delta(candidate, other) for other in angles),
    )
    return {
        "status": "computed",
        "method": "robust_same_component_median_shape",
        "major_axis_px": round(major, 4),
        "minor_axis_px": round(minor, 4),
        "angle_deg": round(float(angle), 4),
        "axis_ratio": round(major / max(minor, 1e-6), 4),
        "supporting_member_count": len(plausible),
        "excluded_member_count": len(ellipses) - len(plausible),
    }


def _member_input(ball: dict[str, Any]) -> dict[str, Any] | None:
    center = _point(
        ball.get("source_final_center_px")
        or ball.get("source_refined_center_px")
        or ball.get("source_initial_refined_center_px")
        or ball.get("source_rough_center_px")
    )
    if center is None:
        return None
    policy = ball.get("source_final_center_policy") or {}
    ellipse = policy.get("ellipse_fit") or ball.get("source_ellipse_fit")
    ellipse = _ellipse_payload(ellipse)
    return {
        "ball_id": _ball_id(ball),
        "label": str(
            ball.get("color_label") or ball.get("class") or "unknown"
        ).lower(),
        "center_px": center,
        "ellipse": ellipse,
        "raw_points_px": _raw_boundary_points(ball),
    }


def _raw_boundary_points(ball: dict[str, Any]) -> list[list[float]]:
    policy = ball.get("source_final_center_policy") or {}
    variant = policy.get("variant") if isinstance(policy.get("variant"), dict) else policy
    filter_stats = (
        variant.get("filter")
        or policy.get("filter")
        or ball.get("source_boundary_filter")
        or {}
    )
    points = filter_stats.get("raw_points_px")
    if not points:
        points = list(variant.get("boundary_points_px") or policy.get("boundary_points_px") or [])
        points.extend(
            variant.get("boundary_rejected_points_px")
            or policy.get("boundary_rejected_points_px")
            or []
        )
    result: list[list[float]] = []
    for value in points or []:
        point = _point(value)
        if point is not None:
            result.append(_round_point(point))
    return result


def _all_raw_points(nodes: list[dict[str, Any]]) -> list[list[float]]:
    return [point for node in nodes for point in node.get("raw_points_px") or []]


def _is_intact_red_rack_candidate(
    members: list[dict[str, Any]],
    settings: dict[str, Any],
) -> bool:
    required_size = int(settings.get("rack_component_size", 15))
    labels = [
        str(ball.get("color_label") or ball.get("class") or "unknown").lower()
        for ball in members
    ]
    red_fraction = labels.count("red") / max(1.0, float(len(labels)))
    return bool(
        len(members) == required_size
        and red_fraction >= float(settings.get("rack_min_red_fraction", 0.9))
    )


def _assign(
    observations: np.ndarray,
    proposals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if linear_sum_assignment is None:
        return None
    distances = np.linalg.norm(
        observations[:, None, :] - proposals[None, :, :],
        axis=2,
    )
    rows, columns = linear_sum_assignment(distances**2)
    node_by_member = np.zeros(len(observations), dtype=np.int64)
    residual_by_member = np.zeros(len(observations), dtype=np.float64)
    node_by_member[rows] = columns
    residual_by_member[rows] = distances[rows, columns]
    return rows, node_by_member, residual_by_member


def _triangular_templates(side: int) -> list[np.ndarray]:
    directions = [
        np.asarray(value, dtype=np.int64)
        for value in [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    ]
    return [
        np.asarray(
            [
                row * directions[index]
                + column * directions[(index + 1) % 6]
                for row in range(side)
                for column in range(side - row)
            ],
            dtype=np.int64,
        )
        for index in range(6)
    ]


def _nearest_hex_axis_delta(angle_deg: float, phase_deg: float) -> float:
    return min(
        _angle_delta(angle_deg, phase_deg + 60.0 * index)
        for index in range(3)
    )


def _direction_vector(angle_deg: float, length: float) -> np.ndarray:
    angle = np.deg2rad(float(angle_deg))
    return float(length) * np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)


def _angle_delta(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 90.0) % 180.0 - 90.0)


def _ellipse_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    center = _point(value.get("center_px"))
    try:
        major = float(value.get("major_axis_px"))
        minor = float(value.get("minor_axis_px"))
        angle = float(value.get("angle_deg") or 0.0) % 180.0
    except (TypeError, ValueError):
        return None
    if center is None or major <= 4.0 or minor <= 4.0:
        return None
    if minor > major:
        major, minor = minor, major
        angle = (angle + 90.0) % 180.0
    return {
        "center_px": _round_point(center),
        "major_axis_px": major,
        "minor_axis_px": minor,
        "angle_deg": angle,
        "axis_ratio": major / max(minor, 1e-6),
        "source": value.get("source"),
    }


def _compact_lattice_quality(lattice: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase_deg": lattice.get("phase_deg"),
        "spacing_px": lattice.get("spacing_px"),
        "contact_vector_count": lattice.get("contact_vector_count"),
        "aligned_vector_count": lattice.get("aligned_vector_count"),
        "anchor_count": lattice.get("anchor_count"),
        "anchor_rms_px": lattice.get("anchor_rms_px"),
        "outlier_member_count": lattice.get("outlier_member_count"),
    }


def _failed_component(
    component: dict[str, Any],
    reason: str,
    **diagnostics: Any,
) -> dict[str, Any]:
    return {
        "status": "diagnostic_only",
        "promoted": False,
        "model": "joint_global_arc_cluster_solver",
        "component_id": component.get("component_id"),
        "member_ids": [int(value) for value in component.get("member_ids") or []],
        "promotion_gate": {"passed": False, "reasons": [reason]},
        "by_ball_id": {
            str(ball_id): {
                "status": "diagnostic_only",
                "promoted": False,
                "component_id": component.get("component_id"),
                "promotion_reasons": [reason],
            }
            for ball_id in component.get("member_ids") or []
        },
        **diagnostics,
    }


def _empty_result(
    status: str,
    *,
    enabled: bool,
    reasons: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "enabled": enabled,
        "model": "joint_global_arc_cluster_solver",
        "component_count": 0,
        "promoted_component_count": 0,
        "components": [],
        "by_ball_id": {},
        "reasons": reasons or [],
    }


def _point(value: Any) -> np.ndarray | None:
    try:
        point = np.asarray(value, dtype=np.float64).reshape(2)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(point)):
        return None
    return point


def _round_point(point: np.ndarray | list[float]) -> list[float]:
    value = np.asarray(point, dtype=np.float64).reshape(2)
    return [round(float(value[0]), 4), round(float(value[1]), 4)]


def _ball_id(ball: dict[str, Any]) -> int:
    try:
        return int(ball.get("id", ball.get("ball_id", 0)))
    except (TypeError, ValueError):
        return 0


__all__ = ["solve_joint_cluster_components"]
