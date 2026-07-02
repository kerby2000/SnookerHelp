from pathlib import Path


def test_active_charuco_intrinsics_command_uses_v1_package_entrypoint() -> None:
    text = Path("tools/calibrate_camera_charuco.py").read_text(encoding="utf-8")

    assert "snookerhelp.calibration.charuco" in text
    assert "vision.charuco_calibration" not in text


def test_active_table_pose_command_uses_v1_package_entrypoint() -> None:
    text = Path("tools/estimate_table_pose_charuco.py").read_text(encoding="utf-8")

    assert "snookerhelp.calibration.charuco" in text
    assert "vision.charuco_calibration" not in text


def test_v1_calibration_dispatch_is_package_native_for_charuco_paths() -> None:
    text = Path("snookerhelp/tools/calibrate.py").read_text(encoding="utf-8")

    assert "calibrate_intrinsics_command" in text
    assert "estimate_table_pose_command" in text
    assert "click_table_corners_command" in text
    assert "_KIND_TO_COMMAND" in text
    assert "runpy" not in text


def test_active_table_corner_clicker_uses_v1_package_entrypoint() -> None:
    text = Path("tools/click_table_corners.py").read_text(encoding="utf-8")

    assert "snookerhelp.calibration.homography_bootstrap" in text
    assert "vision.config" not in text


def test_legacy_calibration_commands_are_deleted() -> None:
    assert not Path("legacy/tools_v0/calibrate_camera_charuco.py").exists()
    assert not Path("legacy/tools_v0/estimate_table_pose_charuco.py").exists()
    assert not Path("legacy/tools_v0/click_table_corners.py").exists()


def test_charuco_core_is_calibration_package_owned() -> None:
    core_text = Path("snookerhelp/calibration/charuco_core.py").read_text(encoding="utf-8")

    assert "class CharucoBoardSpec" in core_text
    assert not Path("vision/charuco_calibration.py").exists()
