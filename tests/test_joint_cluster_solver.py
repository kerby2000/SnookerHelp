from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from snookerhelp.recognition.cluster_arc_assignment import (
    assign_boundary_points_globally,
)
from snookerhelp.recognition.generic_cluster_solver import _missing_hypotheses
from snookerhelp.recognition.joint_cluster_solver import (
    solve_joint_cluster_components,
)


class _LinearCamera:
    model_name = "synthetic_linear_camera"
    is_calibrated = True

    def __init__(self, mm_per_px: float) -> None:
        self.mm_per_px = float(mm_per_px)

    def image_point_to_world_plane(self, point_px, z_mm: float) -> np.ndarray:
        point = np.asarray(point_px, dtype=np.float64).reshape(2)
        return np.asarray(
            [point[0] * self.mm_per_px, point[1] * self.mm_per_px, z_mm],
            dtype=np.float64,
        )


def test_global_arc_assignment_uses_one_owner_per_sample() -> None:
    ellipses = {
        1: _ellipse([100.0, 100.0], major=82.0, minor=76.0),
        2: _ellipse([178.0, 100.0], major=82.0, minor=76.0),
    }
    points = _ellipse_points([100.0, 100.0], 82.0, 76.0, 5.0, count=80)
    points += _ellipse_points([178.0, 100.0], 82.0, 76.0, 5.0, count=80)
    points += [[139.0, 100.0], [139.0, 100.0], [139.0, 100.0]]

    result = assign_boundary_points_globally(
        points,
        ellipses_by_id=ellipses,
        residual_gate_px=3.0,
        ambiguity_margin_px=0.8,
    )

    owned_1 = result["by_ball_id"]["1"]["owned_points_px"]
    owned_2 = result["by_ball_id"]["2"]["owned_points_px"]
    assert result["status"] == "computed"
    assert len(owned_1) > 60
    assert len(owned_2) > 60
    assert result["point_count"] < len(points)
    assert not {tuple(point) for point in owned_1} & {tuple(point) for point in owned_2}


def test_joint_rack_solver_recovers_corrupted_centers_without_traversal() -> None:
    spacing_px = 78.0
    major_px = 82.0
    minor_px = 76.0
    angle_deg = 5.0
    origin = np.asarray([500.0, 400.0], dtype=np.float64)
    basis = np.asarray(
        [
            [spacing_px, 0.5 * spacing_px],
            [0.0, np.sqrt(3.0) * 0.5 * spacing_px],
        ],
        dtype=np.float64,
    )
    cells = np.asarray(
        [(row, column) for row in range(5) for column in range(5 - row)],
        dtype=np.float64,
    )
    truth = origin + cells @ basis.T
    observed = truth.copy()
    observed[[5, 8, 11, 13, 14]] += np.asarray(
        [
            [28.0, -26.0],
            [-31.0, 27.0],
            [30.0, 24.0],
            [-27.0, -29.0],
            [33.0, -23.0],
        ],
        dtype=np.float64,
    )

    balls = []
    mm_per_px = 52.5 / spacing_px
    for index, (initial, center) in enumerate(zip(observed, truth), start=1):
        points = _ellipse_points(
            center,
            major_px,
            minor_px,
            angle_deg,
            count=96,
        )
        # Lamp-like interior samples must not become globally owned boundary arcs.
        points.extend(
            [
                [float(center[0] + offset), float(center[1] - 8.0)]
                for offset in np.linspace(-12.0, 12.0, 7)
            ]
        )
        balls.append(
            {
                "id": index,
                "class": "red",
                "source_final_center_px": initial.tolist(),
                "source_refined_table_xy_by_z_mm": {
                    "z_26_25": {
                        "xy_mm": (center * mm_per_px).tolist(),
                    }
                },
                "source_final_center_policy": {
                    "ellipse_fit": _ellipse(
                        initial,
                        major=major_px + (index % 3 - 1) * 1.2,
                        minor=minor_px + (index % 2) * 0.8,
                        angle=angle_deg + (index % 4 - 1.5) * 0.6,
                    ),
                    "filter": {"raw_points_px": points},
                },
            }
        )

    result = solve_joint_cluster_components(
        balls,
        camera_model=_LinearCamera(mm_per_px),
        ball_radius_mm=26.25,
        settings={
            "rack_joint_promotion_enabled": True,
            "rack_component_size": 15,
            "rack_min_red_fraction": 0.9,
            "joint_shape_min_member_count": 6,
            "rack_promotion_min_anchor_count": 8,
            "rack_promotion_max_anchor_rms_px": 6.0,
            "rack_promotion_min_owned_points_per_node": 5,
            "rack_promotion_min_owned_node_count": 14,
        },
    )

    assert result["status"] == "promoted"
    assert result["promoted_component_count"] == 1
    proposed = np.asarray(
        [
            result["by_ball_id"][str(index)]["proposed_source_center_px"]
            for index in range(1, 16)
        ],
        dtype=np.float64,
    )
    errors = _set_errors(proposed, truth)
    assert float(np.mean(errors)) < 1.0
    assert float(np.max(errors)) < 2.5
    assert all(
        result["by_ball_id"][str(index)]["owned_boundary_point_count"] >= 5
        for index in range(1, 16)
    )
    assert "traversal" not in result["components"][0]["model"]


def test_joint_solver_leaves_loose_balls_unpromoted() -> None:
    balls = [
        {
            "id": 1,
            "class": "red",
            "source_final_center_px": [100.0, 100.0],
            "source_refined_table_xy_by_z_mm": {
                "z_26_25": {"xy_mm": [100.0, 100.0]}
            },
        },
        {
            "id": 2,
            "class": "blue",
            "source_final_center_px": [500.0, 500.0],
            "source_refined_table_xy_by_z_mm": {
                "z_26_25": {"xy_mm": [500.0, 500.0]}
            },
        },
    ]

    result = solve_joint_cluster_components(
        balls,
        camera_model=_LinearCamera(1.0),
        ball_radius_mm=26.25,
    )

    assert result["status"] == "no_cluster_components"
    assert result["promoted_component_count"] == 0
    assert result["by_ball_id"] == {}


def test_generic_joint_solver_refines_arbitrary_cluster_from_multiple_starts() -> None:
    spacing_px = 78.0
    major_px = 82.0
    minor_px = 76.0
    angle_deg = 6.0
    truth = np.asarray(
        [
            [300.0, 300.0],
            [378.0, 300.0],
            [339.0, 367.55],
            [417.0, 367.55],
            [378.0, 435.10],
        ],
        dtype=np.float64,
    )
    observed = truth.copy()
    observed[2] += [5.5, -4.0]
    mm_per_px = 52.5 / spacing_px
    balls = _cluster_balls(
        observed,
        truth,
        major_px=major_px,
        minor_px=minor_px,
        angle_deg=angle_deg,
        mm_per_px=mm_per_px,
    )

    result = solve_joint_cluster_components(
        balls,
        camera_model=_LinearCamera(mm_per_px),
        ball_radius_mm=26.25,
        settings={
            "generic_joint_promotion_enabled": True,
            "generic_promotion_min_component_size": 5,
            "joint_shape_min_member_count": 5,
            "generic_promotion_min_objective_improvement": 0.0001,
            "generic_hard_overlap_tolerance_mm_calibrated": 1.0,
            "generic_abstain_on_missing_hypothesis": False,
        },
    )

    assert result["status"] == "promoted"
    component = result["components"][0]
    assert component["model"] == "generic_multi_start_global_cluster"
    assert len(component["starting_solutions"]) == 4
    assert component["solution_stability"]["max_center_spread_px"] < 1.0
    proposed = np.asarray(
        [
            result["by_ball_id"][str(index)]["proposed_source_center_px"]
            for index in range(1, 6)
        ]
    )
    assert float(np.max(np.linalg.norm(proposed - truth, axis=1))) < 1.2
    assert component["hard_non_overlap"]["passed"]
    assert component["missing_hypotheses"]["promotion_enabled"] is False


def test_generic_joint_solver_suppresses_only_strong_duplicate_hypothesis() -> None:
    spacing_px = 78.0
    truth = np.asarray(
        [
            [300.0, 300.0],
            [378.0, 300.0],
            [339.0, 367.55],
            [417.0, 367.55],
            [378.0, 435.10],
        ],
        dtype=np.float64,
    )
    mm_per_px = 52.5 / spacing_px
    balls = _cluster_balls(
        truth,
        truth,
        major_px=82.0,
        minor_px=76.0,
        angle_deg=5.0,
        mm_per_px=mm_per_px,
    )
    duplicate = dict(balls[0])
    duplicate["id"] = 6
    duplicate["source_final_center_policy"] = {
        **balls[0]["source_final_center_policy"],
        "filter": {
            "raw_points_px": balls[0]["source_final_center_policy"]["filter"][
                "raw_points_px"
            ][:8]
        },
    }
    balls.append(duplicate)

    result = solve_joint_cluster_components(
        balls,
        camera_model=_LinearCamera(mm_per_px),
        ball_radius_mm=26.25,
        settings={
            "generic_joint_promotion_enabled": True,
            "generic_promotion_min_component_size": 5,
            "joint_shape_min_member_count": 5,
            "generic_duplicate_suppression_enabled": True,
            "generic_duplicate_min_strength_ratio": 1.5,
            "generic_hard_overlap_tolerance_mm_calibrated": 1.0,
            "generic_promotion_min_objective_improvement": -1.0,
            "generic_abstain_on_missing_hypothesis": False,
        },
    )

    component = result["components"][0]
    assert component["duplicate_hypotheses"]["suppressed_count"] == 1
    assert result["suppressed_ball_ids"] == [6]
    assert result["by_ball_id"]["6"]["status"] == "suppressed_duplicate"
    assert component["selected_solution"]["hard_non_overlap"]["passed"]


def test_generic_joint_solver_abstains_when_hard_overlap_is_unresolved() -> None:
    centers = np.asarray(
        [
            [300.0, 300.0],
            [310.0, 300.0],
            [378.0, 300.0],
            [339.0, 367.55],
            [417.0, 367.55],
        ],
        dtype=np.float64,
    )
    mm_per_px = 52.5 / 78.0
    balls = _cluster_balls(
        centers,
        centers,
        major_px=82.0,
        minor_px=76.0,
        angle_deg=5.0,
        mm_per_px=mm_per_px,
    )

    result = solve_joint_cluster_components(
        balls,
        camera_model=_LinearCamera(mm_per_px),
        ball_radius_mm=26.25,
        settings={
            "generic_joint_promotion_enabled": True,
            "generic_promotion_min_component_size": 5,
            "joint_shape_min_member_count": 5,
            "generic_duplicate_suppression_enabled": False,
            "generic_hard_overlap_tolerance_mm_calibrated": 1.0,
            "generic_abstain_on_missing_hypothesis": False,
        },
    )

    component = result["components"][0]
    assert component["status"] == "abstained"
    assert "hard_world_non_overlap_failed" in component["promotion_gate"]["reasons"]
    assert component["duplicate_hypotheses"]["hypothesis_count"] >= 1


def test_generic_joint_solver_reports_unexplained_arc_as_missing_hypothesis() -> None:
    unexplained = _ellipse_points(
        [230.0, 100.0],
        82.0,
        76.0,
        5.0,
        count=64,
    )
    result = _missing_hypotheses(
        {
            "centers_px": [[100.0, 100.0]],
            "ownership": {"unowned_points_px": unexplained},
        },
        nodes=[{"ball_id": 1, "center_px": [100.0, 100.0]}],
        shape={
            "major_axis_px": 82.0,
            "minor_axis_px": 76.0,
            "angle_deg": 5.0,
        },
        settings={
            "generic_missing_min_point_count": 14,
            "generic_missing_point_link_px": 7.0,
            "generic_missing_min_center_separation_factor": 0.70,
        },
    )

    assert result["status"] == "diagnostic_only"
    assert result["hypothesis_count"] == 1
    assert result["hypotheses"][0]["decision"] == "diagnostic_only"
    assert result["promotion_enabled"] is False


def _cluster_balls(
    observed: np.ndarray,
    truth: np.ndarray,
    *,
    major_px: float,
    minor_px: float,
    angle_deg: float,
    mm_per_px: float,
) -> list[dict]:
    balls = []
    for index, (initial, center) in enumerate(zip(observed, truth), start=1):
        balls.append(
            {
                "id": index,
                "class": "red",
                "source_final_center_px": initial.tolist(),
                "source_refined_table_xy_by_z_mm": {
                    "z_26_25": {"xy_mm": (initial * mm_per_px).tolist()}
                },
                "source_final_center_policy": {
                    "ellipse_fit": _ellipse(
                        initial,
                        major=major_px,
                        minor=minor_px,
                        angle=angle_deg,
                    ),
                    "filter": {
                        "raw_points_px": _ellipse_points(
                            center,
                            major_px,
                            minor_px,
                            angle_deg,
                            count=96,
                        )
                    },
                },
            }
        )
    return balls


def _ellipse(center, *, major: float, minor: float, angle: float = 5.0) -> dict:
    point = np.asarray(center, dtype=np.float64).reshape(2)
    return {
        "status": "candidate",
        "center_px": point.tolist(),
        "major_axis_px": float(major),
        "minor_axis_px": float(minor),
        "angle_deg": float(angle),
        "axis_ratio": float(major / minor),
    }


def _ellipse_points(
    center,
    major: float,
    minor: float,
    angle_deg: float,
    *,
    count: int,
) -> list[list[float]]:
    center = np.asarray(center, dtype=np.float64).reshape(2)
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    local = np.column_stack(
        [0.5 * major * np.cos(theta), 0.5 * minor * np.sin(theta)]
    )
    angle = np.deg2rad(angle_deg)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    return (center + local @ rotation.T).tolist()


def _set_errors(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    rows, columns = linear_sum_assignment(distances)
    return distances[rows, columns]
