import numpy as np

from snookerhelp.recognition.boundary_ownership import classify_boundary_points
from snookerhelp.recognition.cluster_graph import build_cluster_graph


def test_cluster_graph_classifies_touching_pair() -> None:
    result = build_cluster_graph(
        [
            {
                "id": 1,
                "class": "red",
                "source_refined_table_xy_by_z_mm": {"z_26_25": {"xy_mm": [100.0, 100.0]}},
            },
            {
                "id": 2,
                "class": "red",
                "source_refined_table_xy_by_z_mm": {"z_26_25": {"xy_mm": [152.5, 100.0]}},
            },
        ],
        ball_radius_mm=26.25,
    )

    assert result["status"] == "computed"
    assert result["components"][0]["cluster_type"] == "touching_pair"
    assert result["edges"][0]["relation"] == "touching"
    assert result["by_ball_id"]["1"]["graph_degree"] == 1


def test_cluster_graph_flags_duplicate_like_overlap() -> None:
    result = build_cluster_graph(
        [
            {
                "id": 1,
                "class": "red",
                "source_refined_table_xy_by_z_mm": {"z_26_25": {"xy_mm": [100.0, 100.0]}},
            },
            {
                "id": 2,
                "class": "red",
                "source_refined_table_xy_by_z_mm": {"z_26_25": {"xy_mm": [110.0, 100.0]}},
            },
        ],
        ball_radius_mm=26.25,
    )

    edge = result["edges"][0]
    assert edge["relation"] == "duplicate_or_same_ball"
    assert result["components"][0]["risk"] == "high"


def test_boundary_ownership_assigns_target_and_neighbor_points() -> None:
    target = {
        "center_px": [100.0, 100.0],
        "major_axis_px": 80.0,
        "minor_axis_px": 60.0,
        "angle_deg": 0.0,
    }
    neighbor = {
        "id": 2,
        "center_px": [180.0, 100.0],
        "major_axis_px": 80.0,
        "minor_axis_px": 60.0,
        "angle_deg": 0.0,
    }

    ownership = classify_boundary_points(
        [[140.0, 100.0], [180.0, 70.0], [220.0, 100.0]],
        target_ellipse=target,
        neighbor_ellipses=[neighbor],
    )

    categories = [point["category"] for point in ownership["points"]]
    assert categories[0] == "target_boundary"
    assert "neighbor_owned" in categories
    assert ownership["summary"]["target_boundary"] == 1


def test_boundary_ownership_reports_no_target_ellipse() -> None:
    ownership = classify_boundary_points(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        target_ellipse=None,
        neighbor_ellipses=[],
    )

    assert ownership["status"] == "no_target_ellipse"
    assert ownership["summary"] == {"unknown": 2}
