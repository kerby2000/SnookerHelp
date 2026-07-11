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


def test_large_triangle_cluster_gets_perimeter_and_interior_shells() -> None:
    diameter = 52.5
    spacing_y = diameter * 0.8660254
    balls = []
    ball_id = 1
    for row, count in enumerate([5, 4, 3, 2, 1]):
        for column in range(count):
            balls.append(
                {
                    "id": ball_id,
                    "class": "red",
                    "source_refined_table_xy_by_z_mm": {
                        "z_26_25": {
                            "xy_mm": [
                                100.0 + column * diameter + row * diameter * 0.5,
                                100.0 + row * spacing_y,
                            ]
                        }
                    },
                }
            )
            ball_id += 1

    result = optimize_adjacent_ball_clusters(
        balls,
        ball_radius_mm=26.25,
        settings={
            "enabled": True,
            "target_distance_mm": diameter,
            "neighbor_distance_factor": 1.25,
            "contact_distance_factor": 1.25,
            "shell_classification_enabled": True,
            "shell_classification_min_size": 5,
            "shell_perimeter_distance_factor": 0.58,
            "minimum_improvement_mm": 999.0,
            "iterations": 1,
        },
    )

    cluster = result["clusters"][0]
    shell = cluster["shell_classification"]
    roles = [member["cluster_shell"]["role"] for member in cluster["members"]]
    shell_indices = [member["cluster_shell"]["shell_index"] for member in cluster["members"]]
    traversal = cluster["traversal"]
    traversal_ranks = [
        member["cluster_traversal"]["outside_in_clockwise_rank"]
        for member in cluster["members"]
    ]

    assert result["status"] == "computed"
    assert cluster["members"] and len(cluster["members"]) == 15
    assert shell["status"] == "computed"
    assert shell["shell_counts"] == {"1": 12, "2": 3}
    assert roles.count("perimeter") == 12
    assert roles.count("interior") == 3
    assert set(shell_indices) == {1, 2}
    assert result["by_ball_id"]["7"]["cluster_role"] == "interior"
    assert traversal["status"] == "computed"
    assert set(traversal["paths"]) == {
        "outside_in_clockwise",
        "outside_in_counterclockwise",
        "outside_in_perimeter_walk",
        "outside_in_perimeter_walk_reverse",
    }
    assert len(traversal["paths"]["outside_in_clockwise"]) == 15
    assert set(traversal["paths"]["outside_in_clockwise"]) == set(range(1, 16))
    assert len(traversal["paths"]["outside_in_perimeter_walk"]) == 15
    assert set(traversal["paths"]["outside_in_perimeter_walk"]) == set(range(1, 16))
    assert traversal["primary_path"] == "outside_in_perimeter_walk"
    assert traversal["paths"]["outside_in_perimeter_walk"][0] == 1
    assert sorted(traversal_ranks) == list(range(1, 16))
    assert result["by_ball_id"]["7"]["cluster_traversal_primary_rank"] > 12
    assert result["by_ball_id"]["1"]["cluster_traversal_rank_perimeter_walk"] == 1


def test_small_adjacent_cluster_is_not_classified_as_large_shell() -> None:
    balls = [
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
        {
            "id": 3,
            "class": "red",
            "source_refined_table_xy_by_z_mm": {"z_26_25": {"xy_mm": [126.25, 145.47]}},
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
            "shell_classification_enabled": True,
            "shell_classification_min_size": 5,
        },
    )

    cluster = result["clusters"][0]

    assert cluster["shell_classification"]["status"] == "not_large_cluster"
    assert result["by_ball_id"]["1"]["cluster_shell_status"] == "not_large_cluster"
    assert result["by_ball_id"]["1"]["cluster_role"] is None


def test_same_color_cluster_shape_prior_flags_implausible_ellipse() -> None:
    diameter = 52.5
    balls = []
    for ball_id, (x_mm, y_mm) in enumerate(
        [
            (100.0, 100.0),
            (152.5, 100.0),
            (205.0, 100.0),
            (126.25, 145.47),
            (178.75, 145.47),
            (152.5, 190.94),
        ],
        start=1,
    ):
        major = 94.0
        minor = 78.0
        angle = 3.0
        if ball_id == 6:
            major = 128.0
            minor = 99.0
            angle = 28.0
        balls.append(
            {
                "id": ball_id,
                "class": "red",
                "source_refined_table_xy_by_z_mm": {
                    "z_26_25": {"xy_mm": [x_mm, y_mm]},
                },
                "source_final_center_policy": {
                    "point_count": 90,
                    "ellipse_fit": {
                        "status": "candidate",
                        "center_px": [1000.0 + ball_id * 10.0, 1000.0],
                        "major_axis_px": major,
                        "minor_axis_px": minor,
                        "angle_deg": angle,
                        "axis_ratio": major / minor,
                        "source": "test",
                    },
                },
            }
        )

    result = optimize_adjacent_ball_clusters(
        balls,
        ball_radius_mm=26.25,
        settings={
            "enabled": True,
            "target_distance_mm": diameter,
            "neighbor_distance_factor": 1.25,
            "contact_distance_factor": 1.25,
            "shape_prior_enabled": True,
            "shape_prior_min_cluster_size": 5,
            "shape_prior_min_label_count": 5,
            "shape_prior_min_consensus_members": 4,
            "shape_prior_major_scale_limit": 1.22,
            "shape_prior_minor_scale_limit": 1.22,
            "shape_prior_angle_delta_deg": 12.0,
            "iterations": 1,
        },
    )

    outlier = result["by_ball_id"]["6"]["cluster_shape_prior"]
    stable = result["by_ball_id"]["1"]["cluster_shape_prior"]

    assert outlier["status"] == "computed"
    assert outlier["is_shape_outlier"] is True
    assert "cluster_ellipse_major_outlier" in outlier["reasons"]
    assert "cluster_ellipse_angle_outlier" in outlier["reasons"]
    assert stable["is_shape_outlier"] is False
