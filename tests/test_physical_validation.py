from pathlib import Path

import pytest

from tools.evaluate_cushion_touch import _touch_row
from tools.evaluate_spot_positions import _spot_row
from tools.evaluate_touching_balls import _distance_row, _evaluate_rack_reds
from snookerhelp.qa.validation import ball_points_from_state, classify_table_region


def state() -> dict:
    return {
        "source_image": "Media/example.JPG",
        "table": {
            "length_mm": 1000.0,
            "width_mm": 500.0,
            "coordinate_origin": "bottom_left_inner_playing_surface",
            "warp_px_per_mm": 1.0,
            "processing_margin_mm": 10.0,
        },
        "balls": [
            _ball(1, "red", 200.0, 200.0),
            _ball(2, "red", 252.5, 200.0),
            _ball(3, "red", 305.0, 200.0),
            _ball(4, "blue", 500.0, 250.0),
        ],
    }


def _ball(identifier: int, label: str, x_mm: float, y_mm: float) -> dict:
    return {
        "id": identifier,
        "class": label,
        "color_label": label,
        "x_mm": x_mm,
        "y_mm": y_mm,
        "table_xy_mm": [x_mm, y_mm],
        "refined_center_px": [x_mm + 10.0, 500.0 - y_mm + 10.0],
        "source_refined_warped_center_px": [x_mm + 11.0, 500.0 - y_mm + 8.0],
        "source_refined_table_xy_mm": [x_mm + 1.0, y_mm + 2.0],
        "source_refined_table_xy_by_z_mm": {
            "z_0_00": {
                "z_mm": 0.0,
                "xy_mm": [x_mm, y_mm],
                "xyz_mm": [x_mm, y_mm, 0.0],
                "approximate": True,
            },
            "z_26_25": {
                "z_mm": 26.25,
                "xy_mm": [x_mm + 1.0, y_mm + 2.0],
                "xyz_mm": [x_mm + 1.0, y_mm + 2.0, 26.25],
                "approximate": True,
            },
        },
        "radius_px": 26.25,
        "radius_mm": 26.25,
        "source_radius_px": 30.0,
        "source_refinement_success": True,
        "detection_confidence": 0.9,
    }


def test_region_classification_prioritizes_corners() -> None:
    assert classify_table_region(500, 250, 1000, 500, 100) == "center"
    assert classify_table_region(30, 250, 1000, 500, 100) == "left_edge"
    assert classify_table_region(30, 30, 1000, 500, 100) == "pockets/corners"
    assert classify_table_region(500, 480, 1000, 500, 100) == "top_edge"


def test_touching_distance_row_reports_ball_diameter_error() -> None:
    test_state = state()
    row = _distance_row(
        state_path=Path("sample_state.json"),
        state=test_state,
        mode="touching_pairs",
        center_method="warped",
        pair_id=1,
        a={"id": 1, "label": "red", "x_px": 210, "y_px": 310, "x_mm": 200, "y_mm": 200},
        b={"id": 2, "label": "red", "x_px": 262.5, "y_px": 310, "x_mm": 252.5, "y_mm": 200},
        expected_distance_mm=52.5,
        region_margin_mm=105.0,
        notes=None,
    )

    assert row["distance_mm"] == pytest.approx(52.5)
    assert row["signed_error_mm"] == pytest.approx(0.0)
    assert row["region"] == "center"


def test_rack_red_nearest_neighbors_use_red_balls_only() -> None:
    rows = _evaluate_rack_reds(
        states=[(Path("rack_state.json"), state())],
        expected_diameter_mm=52.5,
        region_margin_mm=105.0,
        center_modes=["warped"],
    )

    assert len(rows) == 3
    assert [row["distance_mm"] for row in rows] == pytest.approx(
        [52.5, 52.5, 52.5]
    )


def test_cushion_touch_row_compares_center_distance_to_radius() -> None:
    row = _touch_row(
        state_path=Path("cushion_state.json"),
        state=state(),
        point={"id": 5, "label": "red", "x_px": 36.25, "y_px": 250, "x_mm": 26.25, "y_mm": 250},
        cushion="left",
        expected_radius_mm=26.25,
        region_margin_mm=105.0,
        notes=None,
    )

    assert row["distance_to_cushion_mm"] == pytest.approx(26.25)
    assert row["abs_error_mm"] == pytest.approx(0.0)
    assert row["region"] == "left_edge"


def test_spot_row_reports_radial_error() -> None:
    row = _spot_row(
        state_path=Path("spot_state.json"),
        state=state(),
        spot_name="blue",
        expected={"x_mm": 500.0, "y_mm": 250.0},
        point={"id": 4, "label": "blue", "x_px": 512.0, "y_px": 258.0, "x_mm": 502.0, "y_mm": 252.0},
        region_margin_mm=105.0,
        notes=None,
    )

    assert row["dx_mm"] == pytest.approx(2.0)
    assert row["dy_mm"] == pytest.approx(2.0)
    assert row["error_mm"] == pytest.approx(2.828427, rel=1e-6)


def test_ball_points_can_select_source_z_plane_coordinates() -> None:
    points = ball_points_from_state(state(), center_mode="source_z_26_25")

    assert points[0]["x_mm"] == pytest.approx(201.0)
    assert points[0]["y_mm"] == pytest.approx(202.0)
    assert points[0]["z_mm"] == pytest.approx(26.25)
    assert points[0]["center_mode"] == "source_z_26_25"
