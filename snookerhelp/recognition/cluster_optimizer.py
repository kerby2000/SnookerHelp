from __future__ import annotations

from typing import Any

import numpy as np

from snookerhelp.recognition.boundary_ownership import analyze_ball_boundary_ownership
from snookerhelp.recognition.cluster_graph import build_cluster_graph
from snookerhelp.recognition.cluster_optimize import optimize_adjacent_ball_clusters


def optimize_cluster_scene(
    balls: list[dict[str, Any]],
    *,
    ball_radius_mm: float,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the v1 generic cluster-analysis layer.

    This wraps the existing joint distance diagnostic, adds a graph abstraction,
    and attaches boundary ownership context. It still does not blindly overwrite
    final coordinates; promotion remains a separate explicit gate.
    """

    cfg = settings or {}
    legacy = optimize_adjacent_ball_clusters(
        balls,
        ball_radius_mm=ball_radius_mm,
        settings=cfg,
    )
    graph = build_cluster_graph(
        balls,
        ball_radius_mm=ball_radius_mm,
        settings=cfg,
    )
    neighbor_by_id = _neighbor_ellipses_by_ball(
        balls,
        graph=graph,
        settings=cfg,
    )
    ownership_by_id = {
        str(_ball_id(ball)): analyze_ball_boundary_ownership(
            ball,
            neighbor_ellipses=neighbor_by_id.get(str(_ball_id(ball)), []),
        )
        for ball in balls
        if _ball_id(ball) > 0
    }

    by_ball_id: dict[str, dict[str, Any]] = dict(legacy.get("by_ball_id") or {})
    for ball_id, graph_payload in (graph.get("by_ball_id") or {}).items():
        by_ball_id.setdefault(str(ball_id), {}).update(graph_payload)
    for ball_id, ownership in ownership_by_id.items():
        by_ball_id.setdefault(str(ball_id), {})["boundary_ownership"] = ownership
        by_ball_id[str(ball_id)]["boundary_ownership_summary"] = {
            "accepted": ownership.get("accepted_points", {}),
            "rejected": ownership.get("rejected_points", {}),
        }
        by_ball_id[str(ball_id)]["neighbor_ellipses_px"] = ownership.get(
            "neighbor_ellipses_px",
            [],
        )

    status = legacy.get("status")
    if status in {None, "not_applicable", "no_adjacent_clusters"}:
        status = graph.get("status", "not_applicable")
    return {
        **legacy,
        "status": status,
        "model": "generic_cluster_scene",
        "graph": graph,
        "boundary_ownership_by_ball_id": ownership_by_id,
        "by_ball_id": by_ball_id,
        "v1_note": (
            "Generic cluster graph and boundary ownership diagnostics are "
            "available. Final coordinate promotion remains gated by image and "
            "physical consistency checks."
        ),
    }


def _neighbor_ellipses_by_ball(
    balls: list[dict[str, Any]],
    *,
    graph: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    distance_factor = float(settings.get("ownership_neighbor_distance_factor", 3.2))
    ball_by_id = {_ball_id(ball): ball for ball in balls if _ball_id(ball) > 0}
    graph_neighbors: dict[int, set[int]] = {
        ball_id: set() for ball_id in ball_by_id
    }
    for edge in graph.get("edges") or []:
        if not edge.get("is_cluster_edge"):
            continue
        left = int(edge["left_id"])
        right = int(edge["right_id"])
        graph_neighbors.setdefault(left, set()).add(right)
        graph_neighbors.setdefault(right, set()).add(left)

    result: dict[str, list[dict[str, Any]]] = {}
    for ball_id, ball in ball_by_id.items():
        center = _source_center(ball)
        radius = _source_radius(ball)
        if center is None or radius is None:
            result[str(ball_id)] = []
            continue
        neighbors: list[dict[str, Any]] = []
        candidate_ids = set(graph_neighbors.get(ball_id, set()))
        for other_id, other in ball_by_id.items():
            if other_id == ball_id:
                continue
            other_center = _source_center(other)
            other_radius = _source_radius(other)
            if other_center is None or other_radius is None:
                continue
            source_distance = float(np.linalg.norm(center - other_center))
            maximum_distance = (radius + other_radius) * max(1.0, distance_factor)
            if other_id not in candidate_ids and source_distance > maximum_distance:
                continue
            ellipse = _source_ellipse_payload(
                other,
                fallback_center=other_center,
                fallback_radius=other_radius,
            )
            if ellipse is None:
                continue
            ellipse.update(
                {
                    "id": int(other_id),
                    "label": str(
                        other.get("color_label")
                        or other.get("class")
                        or other.get("label")
                        or "unknown"
                    ),
                    "source_distance_px": round(float(source_distance), 4),
                    "graph_neighbor": bool(other_id in candidate_ids),
                }
            )
            neighbors.append(ellipse)
        result[str(ball_id)] = neighbors
    return result


def _source_ellipse_payload(
    ball: dict[str, Any],
    *,
    fallback_center: np.ndarray,
    fallback_radius: float,
) -> dict[str, Any] | None:
    policy = ball.get("source_final_center_policy") or {}
    ellipse = policy.get("ellipse_fit") or ball.get("source_ellipse_fit")
    if isinstance(ellipse, dict):
        center = ellipse.get("center_px")
        if center is None and ellipse.get("center_x_px") is not None:
            center = [ellipse.get("center_x_px"), ellipse.get("center_y_px")]
        if center is not None:
            major = ellipse.get("major_axis_px")
            minor = ellipse.get("minor_axis_px")
            if major is not None and minor is not None:
                return {
                    "center_px": [float(center[0]), float(center[1])],
                    "major_axis_px": float(major),
                    "minor_axis_px": float(minor),
                    "angle_deg": float(ellipse.get("angle_deg") or 0.0) % 180.0,
                    "source": ellipse.get("source"),
                }
    return {
        "center_px": [float(fallback_center[0]), float(fallback_center[1])],
        "major_axis_px": 2.0 * float(fallback_radius),
        "minor_axis_px": 2.0 * float(fallback_radius),
        "angle_deg": 0.0,
        "source": "fallback_source_circle",
    }


def _source_center(ball: dict[str, Any]) -> np.ndarray | None:
    value = (
        ball.get("source_final_center_px")
        or ball.get("source_refined_center_px")
        or ball.get("source_rough_center_px")
    )
    try:
        point = np.asarray(value, dtype=np.float64).reshape(2)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(point)):
        return None
    return point


def _source_radius(ball: dict[str, Any]) -> float | None:
    try:
        radius = float(ball.get("source_radius_px") or ball.get("radius_px"))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(radius) or radius <= 0.0:
        return None
    return radius


def _ball_id(ball: dict[str, Any]) -> int:
    try:
        return int(ball.get("id", ball.get("ball_id", 0)))
    except (TypeError, ValueError):
        return 0
