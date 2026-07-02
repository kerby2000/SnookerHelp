from snookerhelp.recognition.cluster_optimize import optimize_adjacent_ball_clusters


def test_joint_cluster_fit_reduces_overlap_without_overwriting_final_position() -> None:
    balls = [
        {
            "id": 1,
            "class": "red",
            "source_refined_table_xy_by_z_mm": {"z_26_25": {"xy_mm": [100.0, 100.0]}},
        },
        {
            "id": 2,
            "class": "red",
            "source_refined_table_xy_by_z_mm": {"z_26_25": {"xy_mm": [144.0, 100.0]}},
        },
    ]

    result = optimize_adjacent_ball_clusters(
        balls,
        ball_radius_mm=26.25,
        settings={
            "enabled": True,
            "target_distance_mm": 52.5,
            "neighbor_distance_factor": 1.25,
            "contact_distance_factor": 1.25,
            "minimum_improvement_mm": 0.01,
            "iterations": 32,
        },
    )

    cluster = result["clusters"][0]
    ball_one = result["by_ball_id"]["1"]

    assert result["status"] == "computed"
    assert cluster["status"] == "optimized"
    assert cluster["joint_pair_rms_mm"] < cluster["initial_pair_rms_mm"]
    assert ball_one["joint_xy_mm"] != ball_one["initial_xy_mm"]
    assert "not applied to final table coordinates" in ball_one["note"]


def test_joint_cluster_fit_is_not_applicable_without_close_neighbors() -> None:
    balls = [
        {
            "id": 1,
            "class": "red",
            "source_refined_table_xy_by_z_mm": {"z_26_25": {"xy_mm": [100.0, 100.0]}},
        },
        {
            "id": 2,
            "class": "blue",
            "source_refined_table_xy_by_z_mm": {"z_26_25": {"xy_mm": [260.0, 100.0]}},
        },
    ]

    result = optimize_adjacent_ball_clusters(
        balls,
        ball_radius_mm=26.25,
        settings={"enabled": True, "neighbor_distance_factor": 1.2},
    )

    assert result["status"] == "no_adjacent_clusters"
    assert result["clusters"] == []
    assert result["by_ball_id"] == {}
