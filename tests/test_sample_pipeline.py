from snookerhelp.recognition.estimator import StateEstimator


def test_empty_reference_has_no_detections() -> None:
    estimator = StateEstimator.from_config()
    frame = estimator.process("Media/01_empty_table/DSC00544.JPG")
    assert frame.state["detection"]["ball_count"] == 0


def test_random_balls_detect_full_inventory() -> None:
    estimator = StateEstimator.from_config()
    frame = estimator.process("Media/02_random_balls/DSC00525.JPG")
    labels = [ball["class"] for ball in frame.state["balls"]]
    assert len(labels) == 22
    assert labels.count("red") == 15
    for label in ("white", "yellow", "green", "brown", "blue", "pink", "black"):
        assert labels.count(label) == 1
    for ball in frame.state["balls"]:
        policy = ball.get("source_final_center_policy") or {}
        if ball["class"] in {"green", "blue", "brown"}:
            assert policy.get("selected_map") == "chroma_difference"
        else:
            assert policy.get("selected_map") == "ball_vs_cloth_probability"


def test_processed_state_can_be_saved_as_json(tmp_path) -> None:
    estimator = StateEstimator.from_config()
    frame, output_directory = estimator.process_and_save(
        "Media/02_random_balls/DSC00525.JPG", tmp_path
    )

    assert frame.state["detection"]["ball_count"] == 22
    assert (output_directory / "DSC00525_state.json").stat().st_size > 0
    first_ball = frame.state["balls"][0]
    assert len(first_ball["raw_hough_center_px"]) == 2
    assert len(first_ball["warped_center_px"]) == 2
    assert len(first_ball["refined_center_px"]) == 2
    assert len(first_ball["source_rough_center_px"]) == 2
    assert len(first_ball["source_refined_center_px"]) == 2
    assert "source_refined_table_xy_by_z_mm" in first_ball
    assert "z_26_25" in first_ball["source_refined_table_xy_by_z_mm"]
    assert len(first_ball["table_xy_mm"]) == 2
    assert first_ball["table_xy_mm_approximate"] is True
    assert "source_refinement_success" in first_ball
    assert "source_radius_px" in first_ball
    assert "fit_residual_px" in first_ball
    assert "color_confidence" in first_ball
    assert "detection_confidence" in first_ball


def test_inventory_selection_does_not_replace_a_distant_red_with_duplicate() -> None:
    estimator = StateEstimator.from_config()
    frame = estimator.process("Media/02_random_balls/DSC00524.JPG")
    reds = [
        ball for ball in frame.state["balls"] if ball["class"] == "red"
    ]

    assert len(reds) == 15
    assert min(ball["x_mm"] for ball in reds) < 650


def test_isolated_overlapping_red_duplicate_does_not_hide_center_ball() -> None:
    estimator = StateEstimator.from_config()
    frame = estimator.process("Media/02_random_balls/DSC00526.JPG")
    reds = [
        ball
        for ball in frame.state["balls"]
        if ball["class"] == "red"
    ]
    centers = [ball["source_refined_center_px"] for ball in reds]

    assert len(reds) == 15
    assert any(4140.0 <= x <= 4280.0 and 2960.0 <= y <= 3070.0 for x, y in centers)
    assert sum(4600.0 <= x <= 4760.0 and 1210.0 <= y <= 1310.0 for x, y in centers) == 1


def test_color_plausibility_keeps_true_green_over_false_green() -> None:
    estimator = StateEstimator.from_config()
    frame = estimator.process("Media/02_random_balls/DSC00528.JPG")
    greens = [
        ball
        for ball in frame.state["balls"]
        if ball["class"] == "green"
    ]
    reds = [
        ball
        for ball in frame.state["balls"]
        if ball["class"] == "red"
    ]
    green_centers = [ball["source_refined_center_px"] for ball in greens]
    red_centers = [ball["source_refined_center_px"] for ball in reds]

    assert len(greens) == 1
    assert any(940.0 <= x <= 1065.0 and 2230.0 <= y <= 2340.0 for x, y in green_centers)
    assert sum(5360.0 <= x <= 5510.0 and 2940.0 <= y <= 3065.0 for x, y in red_centers) == 1


def test_near_cushion_selection_suppresses_edge_duplicates() -> None:
    estimator = StateEstimator.from_config()
    frame = estimator.process("Media/03_near_cushions/DSC00529.JPG")
    reds = [
        ball
        for ball in frame.state["balls"]
        if ball["class"] == "red"
    ]
    centers = [ball["source_refined_center_px"] for ball in reds]

    assert len(reds) == 15
    assert any(760.0 <= x <= 920.0 and 680.0 <= y <= 790.0 for x, y in centers)
    assert sum(2350.0 <= x <= 2530.0 and 3260.0 <= y <= 3360.0 for x, y in centers) == 2
    assert sum(5520.0 <= x <= 5650.0 and 1560.0 <= y <= 1640.0 for x, y in centers) == 1
    assert sum(5520.0 <= x <= 5650.0 and 2700.0 <= y <= 2785.0 for x, y in centers) == 1


def test_dense_cluster_detects_full_legal_inventory() -> None:
    estimator = StateEstimator.from_config()
    frame = estimator.process("Media/05_clusters/DSC00540.JPG")
    labels = [ball["class"] for ball in frame.state["balls"]]

    assert len(labels) == 22
    assert labels.count("red") == 15
    for label in ("white", "yellow", "green", "brown", "blue", "pink", "black"):
        assert labels.count(label) == 1


def test_dsc00542_green_blue_and_red_cluster_evidence_regression() -> None:
    estimator = StateEstimator.from_config()
    frame = estimator.process("Media/05_clusters/DSC00542.JPG")
    balls = frame.state["balls"]
    labels = [ball["class"] for ball in balls]

    assert len(labels) == 22
    assert labels.count("red") == 15

    green = next(ball for ball in balls if ball["class"] == "green")
    blue = next(ball for ball in balls if ball["class"] == "blue")
    for ball in (green, blue):
        maps = ball["source_evidence_maps"] or {}
        assert maps.get("status") == "computed"
        assert maps["maps"]["physical_projection_band"]["p95"] > 0.0
        assert maps["maps"]["ball_vs_cloth_probability"]["p95"] > 0.0
        assert (ball["source_sphere_projection"] or {}).get("projection_mode") in {
            "forward",
            "optimized",
        }

    cluster = (frame.state.get("scene_constraints") or {}).get(
        "adjacent_ball_clusters",
        {},
    )
    clustered_reds = [
        ball
        for ball in balls
        if ball["class"] == "red"
        and (ball.get("source_joint_cluster_optimization") or {}).get("cluster_status")
        == "optimized"
    ]

    assert cluster.get("status") == "computed"
    assert cluster.get("cluster_count", 0) >= 1
    assert len(clustered_reds) >= 10
    assert max(
        (ball["source_joint_cluster_optimization"]["component_size"] for ball in clustered_reds),
        default=0,
    ) >= 10


def test_weak_green_blue_boundary_recovery_regression() -> None:
    estimator = StateEstimator.from_config()
    frame = estimator.process("Media/02_random_balls/DSC00525.JPG")
    balls = frame.state["balls"]

    for label in ("green", "blue"):
        ball = next(item for item in balls if item["class"] == label)
        maps = ball["source_evidence_maps"]
        color_model = maps["local_color_model"]

        assert maps["status"] == "computed"
        assert color_model["separation_lab"] >= 20.0
        assert maps["weights"]["probability"] > maps["weights"]["edge"]
        assert maps["maps"]["ball_vs_cloth_probability"]["p95"] > 0.0
