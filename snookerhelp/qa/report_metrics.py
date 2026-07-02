from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from snookerhelp.core.config import resolve_path
from snookerhelp.qa.validation import (
    add_region_to_row,
    ball_by_id,
    ball_points_from_state,
    distance_mm,
    load_yaml_or_json,
    summarize_values,
    table_dimensions_from_state,
)


TOUCHING_DISTANCE_MM = 52.5
CUSHION_RADIUS_MM = 26.25


def load_scenario(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = load_yaml_or_json(resolve_path(path))
    if not isinstance(payload, dict):
        raise ValueError("Scenario file must be a YAML/JSON object")
    return payload


def select_report_ball(
    state: dict[str, Any],
    ball_id: int | None = None,
) -> dict[str, Any]:
    balls = state.get("balls", [])
    if not balls:
        raise ValueError("Cannot select a report ball because no balls were detected")
    if ball_id is not None:
        for ball in balls:
            if int(ball["id"]) == int(ball_id):
                return ball
        raise KeyError(f"No detected ball with id {ball_id}")

    length_mm, width_mm = table_dimensions_from_state(state)

    def score(ball: dict[str, Any]) -> float:
        x_mm, y_mm = _ball_table_xy(ball)
        edge_distance = min(x_mm, length_mm - x_mm, y_mm, width_mm - y_mm)
        residual = ball.get("source_fit_residual_px")
        source_ok = bool(ball.get("source_refinement_success", False))
        return (
            max(0.0, 220.0 - edge_distance)
            + (float(residual) * 12.0 if residual is not None else 35.0)
            + (45.0 if not source_ok else 0.0)
            + (25.0 if ball.get("color_label") in {"white", "black"} else 0.0)
        )

    return max(balls, key=score)


def choose_zoom_balls(
    state: dict[str, Any],
    selected_ball: dict[str, Any] | None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    balls = list(state.get("balls", []))
    if not balls:
        return []
    if limit is None or limit >= len(balls):
        return sorted(balls, key=lambda ball: int(ball["id"]))
    length_mm, width_mm = table_dimensions_from_state(state)
    by_id = {int(ball["id"]): ball for ball in balls}
    chosen_ids: list[int] = []

    def add(ball: dict[str, Any] | None) -> None:
        if ball is None:
            return
        ball_id = int(ball["id"])
        if ball_id not in chosen_ids:
            chosen_ids.append(ball_id)

    add(selected_ball)
    add(min(balls, key=lambda ball: _edge_distance(ball, length_mm, width_mm)))
    add(min(balls, key=lambda ball: _corner_score(ball, length_mm, width_mm)))
    add(_first_label(balls, "white"))
    add(_first_label(balls, "black"))
    add(_cluster_member(balls))
    failed = [ball for ball in balls if not ball.get("source_refinement_success")]
    add(failed[0] if failed else None)
    residuals = [ball for ball in balls if ball.get("source_fit_residual_px") is not None]
    if residuals:
        add(max(residuals, key=lambda ball: float(ball["source_fit_residual_px"])))

    ranked = sorted(
        balls,
        key=lambda ball: (
            _edge_distance(ball, length_mm, width_mm),
            -float(ball.get("source_fit_residual_px") or 0.0),
        ),
    )
    for ball in ranked:
        add(ball)
        if len(chosen_ids) >= limit:
            break
    return [by_id[ball_id] for ball_id in chosen_ids[:limit]]


def build_ball_review_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Compact per-ball payload used by the interactive HTML review controls."""
    rows: list[dict[str, Any]] = []
    for ball in sorted(state.get("balls", []), key=lambda item: int(item["id"])):
        cushion = nearest_cushion_info(state, ball)
        rows.append(
            {
                "id": int(ball["id"]),
                "label": ball.get("color_label", ball.get("class")),
                "source_refinement_success": bool(
                    ball.get("source_refinement_success", False)
                ),
                "source_fit_residual_px": ball.get("source_fit_residual_px"),
                "source_fit_point_count": (ball.get("debug") or {}).get(
                    "source_circle_fit_point_count"
                ),
                "source_radius_px": ball.get("source_radius_px"),
                "source_rough_center_px": ball.get("source_rough_center_px"),
                "source_refined_center_px": ball.get("source_refined_center_px"),
                "rough_to_refined_shift_px": rough_to_refined_shift_px(ball),
                "nearest_cushion": cushion,
                "fit_explanation": fit_explanation(state, ball),
                "zoom_tile": f"zoom_tiles/ball_{int(ball['id']):02d}.png",
            }
        )
    return rows


def build_coordinate_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ball in state.get("balls", []):
        old_xy = ball.get("table_xy_mm")
        source_xy = ball.get("source_refined_table_xy_mm")
        row: dict[str, Any] = {
            "id": ball["id"],
            "label": ball.get("color_label", ball.get("class")),
            "warped_x_mm": old_xy[0] if old_xy else None,
            "warped_y_mm": old_xy[1] if old_xy else None,
            "source_x_mm": source_xy[0] if source_xy else None,
            "source_y_mm": source_xy[1] if source_xy else None,
            "source_refinement_success": ball.get("source_refinement_success", False),
            "source_fit_residual_px": ball.get("source_fit_residual_px"),
        }
        if old_xy and source_xy:
            row["warped_to_source_delta_mm"] = float(
                np.hypot(source_xy[0] - old_xy[0], source_xy[1] - old_xy[1])
            )
        else:
            row["warped_to_source_delta_mm"] = None
        for key, projection in (ball.get("source_refined_table_xy_by_z_mm") or {}).items():
            xy = projection["xy_mm"]
            row[f"{key}_x_mm"] = xy[0]
            row[f"{key}_y_mm"] = xy[1]
        rows.append(row)
    return rows


def summarize_detection(state: dict[str, Any]) -> dict[str, Any]:
    balls = state.get("balls", [])
    residuals = [
        ball.get("source_fit_residual_px")
        for ball in balls
        if ball.get("source_fit_residual_px") is not None
    ]
    source_successes = [
        ball for ball in balls if bool(ball.get("source_refinement_success", False))
    ]
    return {
        "ball_count": len(balls),
        "source_refinement_success_count": len(source_successes),
        "source_refinement_success_fraction": (
            len(source_successes) / len(balls) if balls else None
        ),
        "source_fit_residual_px": summarize_values(residuals),
        "camera_model": state.get("camera_model", {}),
        "detection": state.get("detection", {}),
    }


def build_physical_validation(
    state: dict[str, Any],
    scenario: dict[str, Any] | None,
    center_mode: str = "source_refined",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if scenario is None:
        rows.extend(_auto_touching_rows(state, center_mode=center_mode))
        mode = "auto_candidate_touching_pairs"
    else:
        rows.extend(_scenario_touching_rows(state, scenario, center_mode=center_mode))
        rows.extend(_scenario_cushion_rows(state, scenario, center_mode=center_mode))
        rows.extend(_scenario_spot_rows(state, scenario, center_mode=center_mode))
        mode = "scenario"
    error_values = [row["abs_error_mm"] for row in rows if row.get("abs_error_mm") is not None]
    return {
        "mode": mode,
        "center_mode": center_mode,
        "row_count": len(rows),
        "summary": summarize_values(error_values),
        "rows": rows,
    }


def z_projection_comparison(ball: dict[str, Any]) -> list[dict[str, Any]]:
    old_xy = ball.get("table_xy_mm")
    rows: list[dict[str, Any]] = []
    for key, projection in (ball.get("source_refined_table_xy_by_z_mm") or {}).items():
        xy = projection["xy_mm"]
        delta = (
            float(np.hypot(xy[0] - old_xy[0], xy[1] - old_xy[1]))
            if old_xy is not None
            else None
        )
        rows.append(
            {
                "key": key,
                "z_mm": float(projection["z_mm"]),
                "x_mm": float(xy[0]),
                "y_mm": float(xy[1]),
                "delta_from_warped_mm": delta,
                "approximate": bool(projection.get("approximate", True)),
            }
        )
    return sorted(rows, key=lambda row: row["z_mm"])


def rough_to_refined_shift_px(ball: dict[str, Any]) -> float | None:
    rough = ball.get("source_rough_center_px")
    refined = ball.get("source_refined_center_px")
    if rough is None or refined is None:
        return None
    return float(np.hypot(refined[0] - rough[0], refined[1] - rough[1]))


def nearest_cushion_info(
    state: dict[str, Any],
    ball: dict[str, Any],
) -> dict[str, Any]:
    length_mm, width_mm = table_dimensions_from_state(state)
    x_mm, y_mm = _ball_table_xy(ball)
    distances = {
        "left": x_mm,
        "right": length_mm - x_mm,
        "bottom": y_mm,
        "top": width_mm - y_mm,
    }
    cushion, distance = min(distances.items(), key=lambda item: item[1])
    return {
        "name": cushion,
        "distance_mm": float(distance),
        "is_near": bool(distance <= 105.0),
    }


def fit_explanation(state: dict[str, Any], ball: dict[str, Any]) -> str:
    """Human-readable explanation of why the source circle fit was accepted/fallback."""
    point_count = (ball.get("debug") or {}).get("source_circle_fit_point_count")
    residual = ball.get("source_fit_residual_px")
    shift = rough_to_refined_shift_px(ball)
    cushion = nearest_cushion_info(state, ball)
    notes: list[str] = []
    if ball.get("source_refinement_success"):
        detail = "accepted source radial-boundary circle fit"
        if point_count:
            detail += f" from {point_count} boundary samples"
        if residual is not None:
            detail += f"; residual {float(residual):.2f} px"
        if shift is not None:
            detail += f"; rough-to-refined shift {shift:.2f} px"
        notes.append(detail)
    else:
        notes.append(
            "fallback to inverse-warped rough center because the source boundary "
            "fit did not pass support/residual/shift checks"
        )
    if residual is not None and float(residual) > 2.5:
        notes.append("high residual: inspect for touching balls, shadow, cushion contact, or partial boundary")
    if cushion["is_near"]:
        notes.append(
            f"near {cushion['name']} cushion ({cushion['distance_mm']:.1f} mm from edge); "
            "warped view may look oval, so judge the source-image fit"
        )
    if not state.get("camera_model", {}).get("is_calibrated", False):
        notes.append("XY-by-Z rows are approximate until calibrated_pinhole mode is configured")
    return ". ".join(notes) + "."


def _scenario_touching_rows(
    state: dict[str, Any],
    scenario: dict[str, Any],
    center_mode: str,
) -> list[dict[str, Any]]:
    pairs = scenario.get("touching_pairs", scenario.get("pairs", [])) or []
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        a = ball_by_id(state, int(pair.get("ball_a", pair.get("a"))), center_mode)
        b = ball_by_id(state, int(pair.get("ball_b", pair.get("b"))), center_mode)
        expected = float(pair.get("expected_distance_mm", TOUCHING_DISTANCE_MM))
        measured = distance_mm(a, b)
        rows.append(
            _validation_row(
                kind="touching_pair",
                item=f"{a['id']}-{b['id']}",
                measured_mm=measured,
                expected_mm=expected,
                x_mm=0.5 * (a["x_mm"] + b["x_mm"]),
                y_mm=0.5 * (a["y_mm"] + b["y_mm"]),
                state=state,
                notes=pair.get("notes"),
                index=index,
            )
        )
    return rows


def _auto_touching_rows(
    state: dict[str, Any],
    center_mode: str,
    limit: int = 12,
) -> list[dict[str, Any]]:
    points = ball_points_from_state(state, center_mode=center_mode)
    candidates = []
    for a, b in combinations(points, 2):
        measured = distance_mm(a, b)
        if 40.0 <= measured <= 70.0:
            candidates.append((abs(measured - TOUCHING_DISTANCE_MM), a, b, measured))
    candidates.sort(key=lambda item: item[0])
    rows: list[dict[str, Any]] = []
    for index, (_, a, b, measured) in enumerate(candidates[:limit], start=1):
        rows.append(
            _validation_row(
                kind="auto_touching_candidate",
                item=f"{a['id']}-{b['id']}",
                measured_mm=measured,
                expected_mm=TOUCHING_DISTANCE_MM,
                x_mm=0.5 * (a["x_mm"] + b["x_mm"]),
                y_mm=0.5 * (a["y_mm"] + b["y_mm"]),
                state=state,
                notes="auto nearest candidate, not ground truth",
                index=index,
            )
        )
    return rows


def _scenario_cushion_rows(
    state: dict[str, Any],
    scenario: dict[str, Any],
    center_mode: str,
) -> list[dict[str, Any]]:
    touches = scenario.get("cushion_touches", scenario.get("touches", [])) or []
    length_mm, width_mm = table_dimensions_from_state(state)
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(touches, start=1):
        point = ball_by_id(state, int(spec.get("ball_id", spec.get("ball"))), center_mode)
        cushion = str(spec["cushion"]).lower()
        if cushion == "left":
            measured = point["x_mm"]
        elif cushion == "right":
            measured = length_mm - point["x_mm"]
        elif cushion == "bottom":
            measured = point["y_mm"]
        elif cushion == "top":
            measured = width_mm - point["y_mm"]
        else:
            raise ValueError(f"Unsupported cushion: {cushion}")
        expected = float(spec.get("expected_radius_mm", CUSHION_RADIUS_MM))
        rows.append(
            _validation_row(
                kind="cushion_touch",
                item=f"{point['id']}:{cushion}",
                measured_mm=measured,
                expected_mm=expected,
                x_mm=point["x_mm"],
                y_mm=point["y_mm"],
                state=state,
                notes=spec.get("notes"),
                index=index,
            )
        )
    return rows


def _scenario_spot_rows(
    state: dict[str, Any],
    scenario: dict[str, Any],
    center_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    spots = scenario.get("spots", {}) or {}
    spot_tests = scenario.get("spot_tests", scenario.get("spot_mappings", [])) or []
    if isinstance(spots, dict):
        for name, spec in spots.items():
            if isinstance(spec, dict) and any(key in spec for key in ("ball_id", "ball")):
                spot_tests.append({"spot": name, **spec})
    for index, spec in enumerate(spot_tests, start=1):
        spot_name = str(spec["spot"])
        spot = spots[spot_name]
        if isinstance(spot, (list, tuple)):
            expected_x, expected_y = float(spot[0]), float(spot[1])
        else:
            expected_x = float(spot.get("x_mm", spot.get("x")))
            expected_y = float(spot.get("y_mm", spot.get("y")))
        if spec.get("ball_id", spec.get("ball")) is not None:
            point = ball_by_id(state, int(spec.get("ball_id", spec.get("ball"))), center_mode)
        else:
            label = str(spec.get("label", spec.get("class", spot_name)))
            candidates = [
                point for point in ball_points_from_state(state, center_mode)
                if point["label"] == label
            ]
            if not candidates:
                continue
            point = min(
                candidates,
                key=lambda item: float(
                    np.hypot(item["x_mm"] - expected_x, item["y_mm"] - expected_y)
                ),
            )
        measured = float(np.hypot(point["x_mm"] - expected_x, point["y_mm"] - expected_y))
        rows.append(
            _validation_row(
                kind="spot",
                item=f"{spot_name}:{point['id']}",
                measured_mm=measured,
                expected_mm=0.0,
                x_mm=expected_x,
                y_mm=expected_y,
                state=state,
                notes=spec.get("notes"),
                index=index,
            )
        )
    return rows


def _validation_row(
    kind: str,
    item: str,
    measured_mm: float,
    expected_mm: float,
    x_mm: float,
    y_mm: float,
    state: dict[str, Any],
    notes: str | None,
    index: int,
) -> dict[str, Any]:
    length_mm, width_mm = table_dimensions_from_state(state)
    signed_error = measured_mm - expected_mm
    row = {
        "index": index,
        "kind": kind,
        "item": item,
        "measured_mm": float(measured_mm),
        "expected_mm": float(expected_mm),
        "signed_error_mm": float(signed_error),
        "abs_error_mm": abs(float(signed_error)),
        "status": _status(abs(float(signed_error))),
        "x_mm": float(x_mm),
        "y_mm": float(y_mm),
        "notes": notes,
    }
    add_region_to_row(row, x_mm, y_mm, length_mm, width_mm, 105.0)
    return row


def _status(abs_error_mm: float) -> str:
    if abs_error_mm <= 3.0:
        return "pass"
    if abs_error_mm <= 8.0:
        return "warn"
    return "fail"


def _ball_table_xy(ball: dict[str, Any]) -> tuple[float, float]:
    xy = ball.get("source_refined_table_xy_mm") or ball.get("table_xy_mm")
    return float(xy[0]), float(xy[1])


def _edge_distance(ball: dict[str, Any], length_mm: float, width_mm: float) -> float:
    x_mm, y_mm = _ball_table_xy(ball)
    return min(x_mm, length_mm - x_mm, y_mm, width_mm - y_mm)


def _corner_score(ball: dict[str, Any], length_mm: float, width_mm: float) -> float:
    x_mm, y_mm = _ball_table_xy(ball)
    return min(x_mm, length_mm - x_mm) + min(y_mm, width_mm - y_mm)


def _first_label(balls: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    for ball in balls:
        if ball.get("color_label", ball.get("class")) == label:
            return ball
    return None


def _cluster_member(balls: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(balls) < 2:
        return None
    best_ball = None
    best_distance = float("inf")
    positions = [(ball, _ball_table_xy(ball)) for ball in balls]
    for ball, (x_mm, y_mm) in positions:
        nearest = min(
            float(np.hypot(x_mm - other_x, y_mm - other_y))
            for other, (other_x, other_y) in positions
            if other is not ball
        )
        if nearest < best_distance:
            best_distance = nearest
            best_ball = ball
    return best_ball
