import numpy as np

from snookerhelp.calibration.camera_core import PinholeCameraModel
from snookerhelp.recognition.physical_optimize import (
    _visible_contour_points,
    optimize_ball_xy_from_sphere_projection,
)
from snookerhelp.recognition.sphere_projection import project_sphere_silhouette


def test_physical_optimizer_moves_toward_observed_sphere_projection() -> None:
    camera = PinholeCameraModel(
        camera_matrix=np.array(
            [
                [1000.0, 0.0, 320.0],
                [0.0, 1000.0, 240.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        distortion_coefficients=np.zeros(5),
        rotation_world_to_camera=np.diag([1.0, -1.0, -1.0]),
        translation_world_to_camera=np.array([0.0, 0.0, 1000.0]),
        is_calibrated=True,
    )
    true_xy = np.array([120.0, 80.0])
    initial_xy = np.array([108.0, 80.0])
    observed = project_sphere_silhouette(
        camera,
        [true_xy[0], true_xy[1], 26.25],
        26.25,
        sample_count=96,
    )["contour_points_px"]

    result = optimize_ball_xy_from_sphere_projection(
        initial_xy_mm=initial_xy,
        camera_model=camera,
        observed_boundary_points_px=observed,
        evidence_maps=None,
        neighbors=[],
        cushion_context={"length_mm": 3569.0, "width_mm": 1778.0},
        ball_radius_mm=26.25,
        settings={
            "enabled": True,
            "search_radius_mm": 18.0,
            "coarse_step_mm": 6.0,
            "minimum_observed_points": 20,
            "minimum_objective_improvement": 0.001,
            "movement_prior_weight": 0.02,
            "evidence_map_weight": 0.0,
            "point_residual_weight": 0.9,
        },
    )

    optimized_xy = np.asarray(result["optimized_xy_mm"], dtype=np.float64)
    assert result["success"] is True
    assert np.linalg.norm(optimized_xy - true_xy) < np.linalg.norm(initial_xy - true_xy)
    assert result["residual_px"] < 1.0


def test_occlusion_mask_marks_arc_facing_close_neighbor() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    contour = [[100.0 + 40.0 * np.cos(a), 100.0 + 35.0 * np.sin(a)] for a in angles]
    visible, occlusion = _visible_contour_points(
        projection={"projected_center_px": [100.0, 100.0]},
        contour=contour,
        xy=np.array([500.0, 500.0]),
        neighbors=[
            {
                "id": 2,
                "label": "red",
                "xy_mm": [550.0, 500.0],
                "source_px": [150.0, 100.0],
            }
        ],
        ball_radius_mm=26.25,
    )

    assert occlusion["status"] == "applied"
    assert occlusion["occluding_neighbor_count"] == 1
    assert 0.0 < occlusion["occluded_arc_fraction"] < 0.75
    assert len(visible) < len(contour)
