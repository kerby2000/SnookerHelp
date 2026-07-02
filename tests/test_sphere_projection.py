import cv2
import numpy as np
import pytest

from snookerhelp.calibration.camera_core import HomographyCameraModel, PinholeCameraModel
from snookerhelp.calibration.charuco_core import CharucoBoardSpec, create_charuco_board
from snookerhelp.recognition.sphere_projection import (
    project_sphere_silhouette,
    score_observed_points_against_silhouette,
)
from snookerhelp.core.table import TableModel
from snookerhelp.calibration.homography_bootstrap import TableWarp


def test_project_sphere_silhouette_returns_physical_candidate() -> None:
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
    )

    projection = project_sphere_silhouette(
        camera,
        center_xyz_mm=[100.0, 0.0, 26.25],
        radius_mm=26.25,
        sample_count=72,
    )

    assert projection["status"] == "predicted"
    assert projection["sample_count"] == 72
    assert len(projection["contour_points_px"]) == 72
    assert projection["ellipse_fit"]["axis_ratio"] >= 1.0
    assert projection["projected_center_px"] == pytest.approx(
        camera.world_point_to_image([100.0, 0.0, 26.25]),
        abs=1e-3,
    )

    score = score_observed_points_against_silhouette(
        projection["contour_points_px"],
        projection,
    )
    assert score["status"] == "scored"
    assert score["rms_error_px"] == pytest.approx(0.0, abs=1e-6)


def test_project_sphere_silhouette_requires_calibrated_camera() -> None:
    table = TableModel(
        name="test",
        length_mm=1000,
        width_mm=500,
        ball_radius_mm=25,
        px_per_mm=1,
        origin="bottom_left_inner_playing_surface",
    )
    warp = TableWarp.from_corners(
        table,
        [[100, 100], [1100, 100], [1100, 600], [100, 600]],
    )
    projection = project_sphere_silhouette(
        HomographyCameraModel(warp),
        center_xyz_mm=[100.0, 100.0, 25.0],
        radius_mm=25.0,
    )

    assert projection["status"] == "unavailable"
    assert "homography" in projection["reason"]


def test_calitar_charuco_board_profile_matches_expected_corner_count() -> None:
    if not hasattr(cv2, "aruco"):
        pytest.skip("OpenCV ArUco module is unavailable")
    spec = CharucoBoardSpec(
        name="CALITAR CALI100020TAR.5",
        squares_x=20,
        squares_y=15,
        square_length_mm=32.0,
        marker_length_mm=24.0,
        dictionary="DICT_5X5_1000",
    )

    board = create_charuco_board(spec)
    corners = board.getChessboardCorners()

    assert board.getChessboardSize() == (20, 15)
    assert corners.shape == (19 * 14, 3)
    assert corners[0].tolist() == pytest.approx([32.0, 32.0, 0.0])
