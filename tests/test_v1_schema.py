from snookerhelp.recognition import table_state_from_legacy_report
from snookerhelp.review.schema import (
    V1_REVIEW_SCHEMA,
    default_review_feedback,
    review_feedback_from_legacy,
)


def test_v1_table_state_adapter_uses_product_language() -> None:
    report = {
        "image": "Media/05_clusters/DSC00542.JPG",
        "summary": {"ball_count": 1},
        "camera_model": {"mode": "approximate_pinhole_from_corners"},
        "review_evidence": {
            "source_image_path": "source.jpg",
            "source_size_px": {"width": 6000, "height": 4000},
            "table_corner_points_px": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "balls": [
                {
                    "id": 22,
                    "label": "green",
                    "source_crop_path": "crops/ball_022.jpg",
                    "source_center_px": [5490.0, 2086.0],
                    "final_image_evidence": {
                        "status": "computed",
                        "used_for_final_position": True,
                        "selected_map": "chroma_difference",
                        "selected_label": "Chroma difference",
                        "reason": "label override",
                    },
                    "rough_center_px": [5488.0, 2084.0],
                    "boundary_evidence_source": "radial_boundary",
                    "boundary_points_px": [[1.0, 2.0], [3.0, 4.0]],
                    "boundary_rejected_points_px": [[9.0, 10.0]],
                    "boundary_filter": {
                        "status": "filtered",
                        "raw_count": 3,
                        "accepted_count": 2,
                        "rejected_count": 1,
                    },
                    "ellipse_fit": {
                        "source": "radial_boundary",
                        "center_px": [5491.0, 2090.0],
                        "major_axis_px": 114.0,
                        "minor_axis_px": 74.0,
                        "angle_deg": 0.5,
                        "axis_ratio": 1.54,
                    },
                    "sphere_projection": {
                        "status": "predicted",
                        "camera_model": "approximate_pinhole_from_corners",
                        "approximate": True,
                        "projected_center_px": [5492.0, 2088.0],
                        "contour_points_px": [[2.0, 3.0], [4.0, 5.0]],
                        "center_xyz_mm": [100.0, 200.0, 26.25],
                        "observed_fit_score": {
                            "source": "source_boundary_points_px",
                            "rms_error_px": 4.2,
                        },
                        "projection_mode": "forward",
                        "optimization": {
                            "status": "no_better_solution",
                            "success": False,
                            "joint_cluster": {
                                "cluster_id": 1,
                                "cluster_status": "optimized",
                                "component_size": 2,
                            },
                        },
                        "explanation": [
                            "Blue curve = forward projection from current estimated 3D ball center.",
                            "Approximate camera model limits trust.",
                        ],
                    },
                    "evidence_maps": {
                        "status": "computed",
                        "local_color_model": {"separation_lab": 18.0},
                        "assets": [
                            {
                                "key": "gray_edge",
                                "label": "Grayscale edge",
                                "uri": "evidence_maps/ball_022_gray_edge.png",
                            }
                        ],
                        "boundary_variants": {
                            "ball_vs_cloth_probability": {
                                "status": "computed",
                                "key": "ball_vs_cloth_probability",
                                "label": "Ball-vs-cloth probability",
                                "sampling": "outward_drop",
                                "source": "evidence_map_ball_vs_cloth_probability",
                                "points_px": [[11.0, 12.0], [13.0, 14.0]],
                                "rejected_points_px": [[15.0, 16.0]],
                                "ellipse_fit": {
                                    "status": "candidate",
                                    "source": "evidence_map_ball_vs_cloth_probability",
                                    "center_px": [5490.5, 2087.5],
                                    "major_axis_px": 112.0,
                                    "minor_axis_px": 78.0,
                                    "angle_deg": 2.0,
                                    "axis_ratio": 1.4359,
                                },
                                "filter": {
                                    "status": "filtered",
                                    "accepted_count": 2,
                                    "rejected_count": 1,
                                },
                            }
                        },
                    },
                    "joint_cluster_optimization": {
                        "cluster_id": 1,
                        "cluster_status": "optimized",
                        "component_size": 2,
                    },
                    "physics_c_only_model_decision": {
                        "table_position_trust": "medium",
                        "reasons": ["sphere_and_candidate_c_agree"],
                        "sphere_grade": {"level": "high"},
                        "candidate_c_grade": {"level": "high"},
                    },
                    "legacy_review_confidence": 0.2,
                    "physics_first_review_confidence": 0.6,
                    "physics_c_only_review_confidence": 0.8,
                    "review_confidence": 0.8,
                    "warnings": [],
                }
            ],
        },
        "state": {
            "balls": [
                {
                    "id": 22,
                    "class": "green",
                    "source_refined_center_px": [5490.0, 2086.0],
                    "source_rough_center_px": [5488.0, 2084.0],
                    "source_radius_px": 39.9,
                    "radius_mm": 26.25,
                    "source_refined_table_xy_mm": [3000.0, 500.0],
                    "source_refined_table_xy_by_z_mm": {
                        "z_26_25": {"xy_mm": [3000.0, 500.0]}
                    },
                }
            ],
        },
    }

    table_state = table_state_from_legacy_report(report)
    payload = table_state.to_dict()
    ball = payload["balls"][0]

    assert payload["schema_version"] == "snookerhelp.table_state.v1"
    assert payload["image_name"] == "DSC00542"
    assert ball["evidence"]["image_model"]["model_type"] == "edge_ellipse"
    assert ball["evidence"]["physical_model"]["model_type"] == "projected_sphere"
    assert ball["confidence"]["method"] == "physical_model_plus_image_evidence"
    assert ball["evidence"]["boundary_source"] == "edge boundary"
    assert ball["evidence"]["boundary_rejected_points_px"] == [[9.0, 10.0]]
    assert "projection_recovered_boundary_points_px" not in ball["evidence"]
    assert ball["evidence"]["boundary_filter"]["rejected_count"] == 1
    assert ball["evidence"]["image_model"]["source"] == "edge boundary"
    assert ball["evidence"]["physical_model"]["projection_mode"] == "forward"
    assert ball["evidence"]["physical_model"]["optimization"]["status"] == "no_better_solution"
    assert (
        ball["evidence"]["diagnostics"]["scene_constraints"]["joint_cluster"]["cluster_status"]
        == "optimized"
    )
    assert ball["evidence"]["diagnostics"]["evidence_maps"]["local_color_model"]["separation_lab"] == 18.0
    assert ball["evidence"]["diagnostics"]["evidence_maps"]["assets"][0]["key"] == "gray_edge"
    assert (
        ball["evidence"]["diagnostics"]["evidence_maps"]["boundary_variants"][
            "ball_vs_cloth_probability"
        ]["sampling"]
        == "outward_drop"
    )
    assert (
        ball["evidence"]["diagnostics"]["final_image_evidence"]["selected_map"]
        == "chroma_difference"
    )
    assert (
        ball["evidence"]["diagnostics"]["final_image_evidence"][
            "used_for_final_position"
        ]
        is True
    )
    assert "final_confidence" in ball["confidence"]["components"]
    assert "candidate_a" not in str(payload).lower()
    assert "candidate_b" not in str(payload).lower()
    assert "candidate_c" not in str(payload).lower()
    assert "candidate_d" not in str(payload).lower()


def test_review_feedback_adapter_reads_legacy_review_json() -> None:
    review = review_feedback_from_legacy(
        {
            "balls": [
                {
                    "id": 3,
                    "decision": "ok",
                    "issue_tags": ["near_cushion"],
                    "confidence_source": "human",
                    "confidence": 0.7,
                    "comment": "manual check",
                    "manual_correction": {"center_px": [10.0, 20.0], "model": "ellipse"},
                }
            ],
            "missing_ball_hints": [{"label_guess": "red", "source_px": [1.0, 2.0]}],
            "audit_trail": [{"action": "ok"}],
        },
        image_name="DSC00529",
    )

    payload = review.to_dict()
    assert payload["schema_version"] == V1_REVIEW_SCHEMA
    assert payload["balls"][0]["ball_id"] == 3
    assert payload["balls"][0]["manual_correction"]["source_px"] == [10.0, 20.0]
    assert payload["missing_balls"][0]["label_guess"] == "red"


def test_default_review_feedback_initializes_all_balls_unreviewed() -> None:
    feedback = default_review_feedback(image_name="DSC00524", ball_ids=[1, 2, 3])

    assert feedback.schema_version == V1_REVIEW_SCHEMA
    assert [ball.ball_id for ball in feedback.balls] == [1, 2, 3]
    assert {ball.decision for ball in feedback.balls} == {"unreviewed"}
