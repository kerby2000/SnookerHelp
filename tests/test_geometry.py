import numpy as np
import pytest

from snookerhelp.core.table import TableModel
from snookerhelp.calibration.homography_bootstrap import TableWarp
from snookerhelp.calibration.camera_core import (
    build_camera_model,
    HomographyCameraModel,
    PinholeCameraModel,
    z_plane_key,
)
from snookerhelp.recognition.source_refinement import (
    estimate_source_radius_px,
    source_roi_bounds,
)


def test_warped_coordinates_use_bottom_left_origin() -> None:
    table = TableModel(
        name="test",
        length_mm=1000,
        width_mm=500,
        ball_radius_mm=25,
        px_per_mm=1,
        origin="bottom_left_inner_playing_surface",
    )
    warp = TableWarp.from_corners(
        table, [[10, 20], [1010, 20], [1010, 520], [10, 520]]
    )

    assert warp.warped_px_to_table_mm(100, 50) == (100, 450)


def test_homography_round_trip() -> None:
    table = TableModel(
        name="test",
        length_mm=1000,
        width_mm=500,
        ball_radius_mm=25,
        px_per_mm=1,
        origin="bottom_left_inner_playing_surface",
    )
    warp = TableWarp.from_corners(
        table, [[20, 30], [1050, 10], [1020, 550], [5, 510]]
    )
    points = np.float32([[100, 50], [900, 450]])
    source = warp.warped_to_source(points)
    restored = __import__("cv2").perspectiveTransform(
        source.reshape(-1, 1, 2), warp.homography
    ).reshape(-1, 2)

    assert np.allclose(restored, points, atol=1e-3)


def test_source_to_warped_inverse_homography_round_trip() -> None:
    table = TableModel(
        name="test",
        length_mm=1000,
        width_mm=500,
        ball_radius_mm=25,
        px_per_mm=1,
        origin="bottom_left_inner_playing_surface",
        processing_margin_mm=50,
    )
    warp = TableWarp.from_corners(
        table, [[20, 30], [1050, 10], [1020, 550], [5, 510]]
    )
    source_points = np.float32([[40, 42], [500, 250], [1000, 530]])

    warped = warp.source_to_warped(source_points)
    restored = warp.warped_to_source(warped)

    assert np.allclose(restored, source_points, atol=1e-3)


def test_source_radius_and_roi_conversion_are_in_source_pixels() -> None:
    table = TableModel(
        name="test",
        length_mm=1000,
        width_mm=500,
        ball_radius_mm=25,
        px_per_mm=1,
        origin="bottom_left_inner_playing_surface",
    )
    warp = TableWarp.from_corners(
        table, [[100, 100], [1100, 100], [1100, 600], [100, 600]]
    )

    radius = estimate_source_radius_px(
        warp,
        warped_center=(500, 250),
        warped_radius_px=25,
    )
    roi = source_roi_bounds(
        image_shape=(800, 1200, 3),
        center=(600, 350),
        radius_px=radius,
        margin_factor=2.0,
    )

    assert radius == pytest.approx(25.0, abs=0.1)
    assert roi == (550, 300, 651, 401)


def test_homography_camera_model_preserves_requested_z_height() -> None:
    table = TableModel(
        name="test",
        length_mm=1000,
        width_mm=500,
        ball_radius_mm=25,
        px_per_mm=1,
        origin="bottom_left_inner_playing_surface",
    )
    warp = TableWarp.from_corners(
        table, [[100, 100], [1100, 100], [1100, 600], [100, 600]]
    )
    camera = HomographyCameraModel(warp)

    source_point = np.array([600.0, 350.0])
    world = camera.image_point_to_world_plane(source_point, z_mm=25.0)
    restored_source = camera.world_point_to_image(world)

    assert world[2] == pytest.approx(25.0)
    assert np.allclose(restored_source, source_point, atol=1e-3)


def test_homography_camera_model_projects_all_z_planes_to_same_xy() -> None:
    table = TableModel(
        name="test",
        length_mm=1000,
        width_mm=500,
        ball_radius_mm=25,
        px_per_mm=1,
        origin="bottom_left_inner_playing_surface",
    )
    warp = TableWarp.from_corners(
        table, [[100, 100], [1100, 100], [1100, 600], [100, 600]]
    )
    camera = HomographyCameraModel(warp)

    projections = camera.project_image_point_to_z_planes(
        [600.0, 350.0],
        [0.0, 26.25, 52.5],
    )

    xy_values = [projection["xy_mm"] for projection in projections.values()]
    assert xy_values[0] == pytest.approx(xy_values[1])
    assert xy_values[1] == pytest.approx(xy_values[2])
    assert projections[z_plane_key(26.25)]["approximate"] is True


def test_pinhole_camera_model_intersects_requested_z_plane() -> None:
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

    world_z0 = camera.image_point_to_world_plane([420.0, 240.0], z_mm=0.0)
    world_z500 = camera.image_point_to_world_plane([420.0, 240.0], z_mm=500.0)
    image_point = camera.world_point_to_image([100.0, 0.0, 0.0])

    assert world_z0 == pytest.approx([100.0, 0.0, 0.0], abs=1e-6)
    assert world_z500 == pytest.approx([50.0, 0.0, 500.0], abs=1e-6)
    assert image_point == pytest.approx([420.0, 240.0], abs=1e-6)


def test_approximate_pinhole_from_corners_projects_different_z_planes() -> None:
    table = TableModel(
        name="test",
        length_mm=1000,
        width_mm=500,
        ball_radius_mm=25,
        px_per_mm=1,
        origin="bottom_left_inner_playing_surface",
    )
    reference_camera = PinholeCameraModel(
        camera_matrix=np.array(
            [
                [1000.0, 0.0, 640.0],
                [0.0, 1000.0, 360.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        distortion_coefficients=np.zeros(5),
        rotation_world_to_camera=np.diag([1.0, -1.0, -1.0]),
        translation_world_to_camera=np.array([-500.0, 250.0, 1200.0]),
    )
    source_corners = [
        reference_camera.world_point_to_image([0.0, 500.0, 0.0]),
        reference_camera.world_point_to_image([1000.0, 500.0, 0.0]),
        reference_camera.world_point_to_image([1000.0, 0.0, 0.0]),
        reference_camera.world_point_to_image([0.0, 0.0, 0.0]),
    ]
    warp = TableWarp.from_corners(table, source_corners)

    camera = build_camera_model(
        warp,
        {
            "mode": "approximate_pinhole_from_corners",
            "image_width_px": 1280,
            "image_height_px": 720,
            "focal_length_mm": 8.0,
            "sensor_width_mm": 10.24,
            "sensor_height_mm": 5.76,
            "principal_point_px": [640.0, 360.0],
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )

    source_point = reference_camera.world_point_to_image([700.0, 250.0, 25.0])
    projections = camera.project_image_point_to_z_planes(
        source_point,
        [0.0, 25.0, 50.0],
    )

    assert camera.model_name == "approximate_pinhole_from_corners"
    assert camera.is_calibrated is False
    assert camera.camera_center_world_mm == pytest.approx([500.0, 250.0, 1200.0], abs=1e-2)
    assert projections[z_plane_key(25.0)]["xy_mm"] == pytest.approx([700.0, 250.0], abs=1e-2)
    assert projections[z_plane_key(0.0)]["xy_mm"][0] > projections[z_plane_key(25.0)]["xy_mm"][0]
    assert projections[z_plane_key(25.0)]["approximate"] is True


def test_padded_warp_preserves_table_coordinates_outside_boundary() -> None:
    table = TableModel(
        name="test",
        length_mm=1000,
        width_mm=500,
        ball_radius_mm=25,
        px_per_mm=1,
        origin="bottom_left_inner_playing_surface",
        processing_margin_mm=100,
    )
    warp = TableWarp.from_corners(
        table, [[100, 100], [1100, 100], [1100, 600], [100, 600]]
    )

    assert table.warp_width_px == 1200
    assert table.warp_height_px == 700
    assert warp.warped_px_to_table_mm(100, 100) == (0, 500)
    assert warp.warped_px_to_table_mm(50, 650) == (-50, -50)
    assert warp.table_mm_to_warped_px(0, 500) == (100, 100)
    assert warp.table_mm_to_warped_px(-50, -50) == (50, 650)
