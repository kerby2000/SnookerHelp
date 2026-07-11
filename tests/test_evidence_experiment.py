from __future__ import annotations

import cv2
import numpy as np

from snookerhelp.recognition.evidence_experiment import run_evidence_experiment


def _scene() -> tuple[np.ndarray, dict]:
    image = np.full((240, 320, 3), (35, 150, 45), dtype=np.uint8)
    cv2.circle(image, (160, 120), 32, (20, 20, 230), -1, cv2.LINE_AA)
    outline = []
    for angle in np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False):
        outline.append([160.0 + 32.0 * np.cos(angle), 120.0 + 32.0 * np.sin(angle)])
    table_state = {
        "image_name": "synthetic",
        "table_corners_px": [[0, 0], [319, 0], [319, 239], [0, 239]],
        "balls": [
            {
                "ball_id": 8,
                "label": "red",
                "source_px": [160.0, 120.0],
                "radius_px": 32.0,
                "evidence": {
                    "physical_model": {
                        "status": "predicted",
                        "projected_center_px": [160.0, 120.0],
                        "projected_outline_px": outline,
                    },
                    "diagnostics": {"neighbor_ellipses": []},
                },
            }
        ],
    }
    return image, table_state


def test_evidence_experiment_returns_recomputed_map_fit_and_decomposed_score() -> None:
    image, table_state = _scene()
    result = run_evidence_experiment(
        source_image=image,
        table_state=table_state,
        ball_id=8,
        evidence_settings={
            "map_boundary_angle_count": 180,
            "map_boundary_minimum_points": 24,
            "map_boundary_minimum_strength": 0.01,
            "global_cloth_erode_px": 2,
        },
        parameters={
            "map_key": "ball_vs_cloth_probability",
            "ball_reference_mode": "selected_ball",
            "probability_offset_factor": 0.2,
            "probability_scale_factor": 0.2,
        },
    )

    assert result["status"] == "computed"
    assert result["map_png_data_uri"].startswith("data:image/png;base64,")
    assert result["experiment"]["ellipse_fit"]["center_px"]
    assert result["experiment"]["view_score"]["is_ground_truth_accuracy"] is False
    assert result["effective_parameters"]["ball_reference_mode"] == "selected_ball"
    assert result["note"].startswith("Experiment output is transient")


def test_experiment_scores_against_manual_ellipse_when_available() -> None:
    image, table_state = _scene()
    result = run_evidence_experiment(
        source_image=image,
        table_state=table_state,
        ball_id=8,
        evidence_settings={
            "map_boundary_angle_count": 180,
            "map_boundary_minimum_points": 24,
            "map_boundary_minimum_strength": 0.01,
            "global_cloth_erode_px": 2,
        },
        parameters={"map_key": "chroma_difference"},
        ground_truth_ball={
            "ellipse_px": {
                "center_px": [160.0, 120.0],
                "major_axis_px": 64.0,
                "minor_axis_px": 64.0,
                "angle_deg": 0.0,
            }
        },
    )

    comparison = result["experiment"]["annotation_comparison"]
    assert result["ground_truth_available"] is True
    assert comparison["status"] == "computed"
    assert comparison["score_is_ground_truth_based"] is True

