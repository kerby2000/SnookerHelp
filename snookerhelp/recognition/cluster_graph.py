from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def build_cluster_graph(
    balls: list[dict[str, Any]],
    *,
    ball_radius_mm: float,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a generic graph of plausible physical ball interactions.

    This is intentionally not a rack/triangle-specific model. It describes any
    dense region as nodes plus pair relationships: separated, near touching,
    touching, impossible overlap, or duplicate-like. Higher-level optimizers can
    then decide whether to use the graph as a constraint.
    """

    cfg = settings or {}
    diameter = float(cfg.get("target_distance_mm", 2.0 * ball_radius_mm))
    near_factor = float(cfg.get("graph_near_distance_factor", 1.35))
    contact_tolerance = float(cfg.get("graph_contact_tolerance_factor", 0.16)) * diameter
    duplicate_factor = float(cfg.get("graph_duplicate_distance_factor", 0.45))
    overlap_factor = float(cfg.get("graph_overlap_distance_factor", 0.82))

    nodes = [_node_payload(ball, ball_radius_mm) for ball in balls]
    nodes = [node for node in nodes if node["id"] is not None]
    node_by_id = {int(node["id"]): node for node in nodes}

    edges: list[dict[str, Any]] = []
    adjacency: dict[int, set[int]] = {int(node["id"]): set() for node in nodes}
    ids = sorted(node_by_id)
    for index, left_id in enumerate(ids):
        for right_id in ids[index + 1 :]:
            edge = _edge_payload(
                node_by_id[left_id],
                node_by_id[right_id],
                diameter_mm=diameter,
                contact_tolerance_mm=contact_tolerance,
                duplicate_distance_mm=duplicate_factor * diameter,
                overlap_distance_mm=overlap_factor * diameter,
                near_distance_mm=near_factor * diameter,
            )
            edges.append(edge)
            if edge["is_cluster_edge"]:
                adjacency[left_id].add(right_id)
                adjacency[right_id].add(left_id)

    components = [
        _component_payload(index, component, node_by_id, edges, diameter)
        for index, component in enumerate(_connected_components(adjacency), start=1)
        if len(component) >= 2
    ]
    by_ball_id: dict[str, dict[str, Any]] = {}
    for component in components:
        for ball_id in component["member_ids"]:
            by_ball_id[str(ball_id)] = {
                "graph_component_id": component["component_id"],
                "graph_component_size": component["size"],
                "graph_cluster_type": component["cluster_type"],
                "graph_cluster_risk": component["risk"],
                "graph_degree": component["degrees"].get(str(ball_id), 0),
                "graph_contact_degree": component["contact_degrees"].get(str(ball_id), 0),
                "graph_component_edge_count": component["edge_count"],
            }

    return {
        "status": "computed" if components else "no_cluster_components",
        "model": "generic_ball_contact_graph",
        "diameter_mm": round(float(diameter), 4),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "cluster_component_count": len(components),
        "nodes": nodes,
        "edges": edges,
        "components": components,
        "by_ball_id": by_ball_id,
        "note": (
            "Generic graph only. It is not a rack template and does not assume "
            "a 15-red triangle."
        ),
    }


def _node_payload(ball: dict[str, Any], ball_radius_mm: float) -> dict[str, Any]:
    ball_id = _int_or_none(ball.get("id", ball.get("ball_id")))
    source_px = _point_or_none(
        ball.get("source_final_center_px")
        or ball.get("source_refined_center_px")
        or ball.get("source_rough_center_px")
    )
    table_xy = _table_xy_or_none(ball, ball_radius_mm)
    ellipse = _ellipse_payload(
        (ball.get("source_final_center_policy") or {}).get("ellipse_fit")
        or ball.get("source_ellipse_fit")
    )
    return {
        "id": ball_id,
        "label": str(ball.get("color_label") or ball.get("class") or ball.get("label") or "unknown"),
        "source_px": source_px,
        "table_xy_mm": table_xy,
        "radius_px": _float_or_none(ball.get("source_radius_px") or ball.get("radius_px")),
        "image_ellipse": ellipse,
    }


def _edge_payload(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    diameter_mm: float,
    contact_tolerance_mm: float,
    duplicate_distance_mm: float,
    overlap_distance_mm: float,
    near_distance_mm: float,
) -> dict[str, Any]:
    left_xy = _array_or_none(left.get("table_xy_mm"))
    right_xy = _array_or_none(right.get("table_xy_mm"))
    table_distance = None
    relation = "unknown"
    residual = None
    if left_xy is not None and right_xy is not None:
        table_distance = float(np.linalg.norm(left_xy - right_xy))
        residual = table_distance - diameter_mm
        if table_distance <= duplicate_distance_mm:
            relation = "duplicate_or_same_ball"
        elif table_distance < overlap_distance_mm:
            relation = "impossible_overlap"
        elif abs(residual) <= contact_tolerance_mm:
            relation = "touching"
        elif table_distance <= near_distance_mm:
            relation = "near_touching"
        else:
            relation = "separated"
    is_cluster_edge = relation in {
        "duplicate_or_same_ball",
        "impossible_overlap",
        "touching",
        "near_touching",
    }
    return {
        "left_id": int(left["id"]),
        "right_id": int(right["id"]),
        "left_label": left.get("label"),
        "right_label": right.get("label"),
        "same_label": bool(left.get("label") == right.get("label")),
        "relation": relation,
        "is_cluster_edge": bool(is_cluster_edge),
        "table_distance_mm": None if table_distance is None else round(float(table_distance), 4),
        "diameter_residual_mm": None if residual is None else round(float(residual), 4),
        "expected_touching_distance_mm": round(float(diameter_mm), 4),
    }


def _component_payload(
    component_id: int,
    member_ids: list[int],
    node_by_id: dict[int, dict[str, Any]],
    edges: list[dict[str, Any]],
    diameter_mm: float,
) -> dict[str, Any]:
    member_set = set(member_ids)
    component_edges = [
        edge
        for edge in edges
        if edge["left_id"] in member_set
        and edge["right_id"] in member_set
        and edge["is_cluster_edge"]
    ]
    degrees = {ball_id: 0 for ball_id in member_ids}
    contact_degrees = {ball_id: 0 for ball_id in member_ids}
    risk_reasons: list[str] = []
    for edge in component_edges:
        left = int(edge["left_id"])
        right = int(edge["right_id"])
        degrees[left] += 1
        degrees[right] += 1
        if edge["relation"] in {"touching", "near_touching"}:
            contact_degrees[left] += 1
            contact_degrees[right] += 1
        if edge["relation"] in {"duplicate_or_same_ball", "impossible_overlap"}:
            risk_reasons.append(edge["relation"])

    max_degree = max(degrees.values()) if degrees else 0
    labels = [node_by_id[ball_id].get("label") for ball_id in member_ids]
    same_label_fraction = (
        max(labels.count(label) for label in set(labels)) / float(len(labels))
        if labels
        else 0.0
    )
    cluster_type = _cluster_type(
        size=len(member_ids),
        edge_count=len(component_edges),
        max_degree=max_degree,
        same_label_fraction=same_label_fraction,
    )
    if cluster_type in {"dense_cluster", "possible_rack_like"}:
        risk_reasons.append("dense_occlusion")
    return {
        "component_id": int(component_id),
        "cluster_type": cluster_type,
        "risk": "high" if risk_reasons else ("medium" if len(member_ids) >= 4 else "low"),
        "risk_reasons": sorted(set(risk_reasons)),
        "size": len(member_ids),
        "member_ids": [int(ball_id) for ball_id in member_ids],
        "edge_count": len(component_edges),
        "degrees": {str(k): int(v) for k, v in degrees.items()},
        "contact_degrees": {str(k): int(v) for k, v in contact_degrees.items()},
        "same_label_fraction": round(float(same_label_fraction), 4),
        "expected_touching_distance_mm": round(float(diameter_mm), 4),
    }


def _cluster_type(
    *,
    size: int,
    edge_count: int,
    max_degree: int,
    same_label_fraction: float,
) -> str:
    if size == 2:
        return "touching_pair"
    density = 0.0
    if size > 1:
        density = edge_count / float(size * (size - 1) / 2.0)
    if size >= 12 and same_label_fraction >= 0.75 and density >= 0.28:
        return "possible_rack_like"
    if size >= 4 and (max_degree >= 3 or density >= 0.35):
        return "dense_cluster"
    return "arbitrary_cluster"


def _connected_components(adjacency: dict[int, set[int]]) -> list[list[int]]:
    unseen = set(adjacency)
    components: list[list[int]] = []
    while unseen:
        start = unseen.pop()
        queue: deque[int] = deque([start])
        component = {start}
        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, set()):
                if neighbor in component:
                    continue
                component.add(neighbor)
                unseen.discard(neighbor)
                queue.append(neighbor)
        components.append(sorted(component))
    return components


def _table_xy_or_none(ball: dict[str, Any], ball_radius_mm: float) -> list[float] | None:
    by_z = ball.get("source_refined_table_xy_by_z_mm") or {}
    z_key = f"z_{float(ball_radius_mm):.2f}".replace("-", "m").replace(".", "_")
    projection = by_z.get(z_key) or by_z.get("z_26_25")
    if isinstance(projection, dict) and projection.get("xy_mm") is not None:
        return _point_or_none(projection.get("xy_mm"))
    return _point_or_none(ball.get("source_refined_table_xy_mm") or ball.get("table_xy_mm"))


def _ellipse_payload(ellipse: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(ellipse, dict):
        return None
    center = _point_or_none(ellipse.get("center_px"))
    if center is None and ellipse.get("center_x_px") is not None:
        center = _point_or_none([ellipse.get("center_x_px"), ellipse.get("center_y_px")])
    if center is None:
        return None
    major = _float_or_none(ellipse.get("major_axis_px"))
    minor = _float_or_none(ellipse.get("minor_axis_px"))
    if major is None or minor is None:
        radius = _float_or_none(ellipse.get("radius_px"))
        if radius is None:
            return None
        major = minor = radius * 2.0
    angle = _float_or_none(ellipse.get("angle_deg")) or 0.0
    return {
        "center_px": center,
        "major_axis_px": round(float(major), 4),
        "minor_axis_px": round(float(minor), 4),
        "angle_deg": round(float(angle) % 180.0, 4),
        "source": ellipse.get("source"),
    }


def _point_or_none(value: Any) -> list[float] | None:
    try:
        if value is None or len(value) < 2:
            return None
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not np.isfinite([x, y]).all():
        return None
    return [round(x, 4), round(y, 4)]


def _array_or_none(value: Any) -> np.ndarray | None:
    point = _point_or_none(value)
    if point is None:
        return None
    return np.asarray(point, dtype=np.float64)


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _int_or_none(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None
