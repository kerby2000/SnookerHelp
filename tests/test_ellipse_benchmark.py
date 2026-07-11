from __future__ import annotations

from snookerhelp.qa.ellipse_benchmark import evaluate_ellipse_benchmark


def test_ellipse_benchmark_reports_measured_errors_and_worst_ball() -> None:
    table_state = {
        "image_name": "example",
        "balls": [
            {
                "ball_id": 8,
                "label": "red",
                "source_px": [100.0, 100.0],
                "evidence": {
                    "image_model": {
                        "center_px": [100.0, 100.0],
                        "major_axis_px": 90.0,
                        "minor_axis_px": 72.0,
                        "angle_deg": 0.0,
                    },
                    "diagnostics": {
                        "final_image_evidence": {
                            "selected_map": "ball_vs_cloth_probability",
                        },
                        "evidence_maps": {
                            "boundary_variants": {
                                "ball_vs_cloth_probability": {
                                    "ellipse_fit": {
                                        "center_px": [100.0, 100.0],
                                        "major_axis_px": 90.0,
                                        "minor_axis_px": 72.0,
                                        "angle_deg": 0.0,
                                    }
                                }
                            }
                        },
                    },
                },
            },
            {
                "ball_id": 9,
                "label": "red",
                "source_px": [130.0, 100.0],
                "evidence": {
                    "image_model": {
                        "center_px": [130.0, 100.0],
                        "major_axis_px": 100.0,
                        "minor_axis_px": 80.0,
                        "angle_deg": 15.0,
                    },
                    "diagnostics": {},
                },
            },
        ],
    }
    ground_truth = {
        "schema_version": "snookerhelp.ground_truth.v1",
        "image_name": "example",
        "balls": [
            {
                "ball_id": 8,
                "label": "red",
                "ellipse_px": {
                    "center_px": [100.0, 100.0],
                    "major_axis_px": 90.0,
                    "minor_axis_px": 72.0,
                    "angle_deg": 0.0,
                },
            },
            {
                "ball_id": 9,
                "label": "red",
                "ellipse_px": {
                    "center_px": [120.0, 100.0],
                    "major_axis_px": 90.0,
                    "minor_axis_px": 72.0,
                    "angle_deg": 0.0,
                },
            },
        ],
    }

    result = evaluate_ellipse_benchmark(table_state, ground_truth)

    assert result["summary"]["annotated_ball_count"] == 2
    assert result["summary"]["computed_ball_count"] == 2
    assert result["summary"]["median_source_center_error"] == 5.0
    assert result["worst_balls"][0]["ball_id"] == 9
    assert result["by_evidence_map"]["ball_vs_cloth_probability"]["summary"][
        "mean_annotation_score"
    ] == 100.0


def test_ellipse_benchmark_matches_interchangeable_reds_spatially() -> None:
    ellipse = {
        "major_axis_px": 80.0,
        "minor_axis_px": 72.0,
        "angle_deg": 3.0,
    }
    table_state = {
        "balls": [
            {
                "ball_id": 8,
                "label": "red",
                "source_px": [200.0, 100.0],
                "evidence": {
                    "image_model": {**ellipse, "center_px": [200.0, 100.0]},
                    "diagnostics": {},
                },
            },
            {
                "ball_id": 9,
                "label": "red",
                "source_px": [100.0, 100.0],
                "evidence": {
                    "image_model": {**ellipse, "center_px": [100.0, 100.0]},
                    "diagnostics": {},
                },
            },
        ]
    }
    ground_truth = {
        "schema_version": "snookerhelp.ground_truth.v1",
        "balls": [
            {
                "ball_id": 8,
                "label": "red",
                "ellipse_px": {**ellipse, "center_px": [100.0, 100.0]},
            },
            {
                "ball_id": 9,
                "label": "red",
                "ellipse_px": {**ellipse, "center_px": [200.0, 100.0]},
            },
        ],
    }

    result = evaluate_ellipse_benchmark(table_state, ground_truth)

    assert result["summary"]["mean_source_center_error"] == 0.0
    assert result["summary"]["mean_contour_rms_error"] == 0.0
    assert result["balls"][0]["matched_detector_ball_id"] == 9
    assert result["balls"][1]["matched_detector_ball_id"] == 8
