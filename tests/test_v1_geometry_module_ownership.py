from pathlib import Path


def test_camera_model_is_calibration_package_owned() -> None:
    core_text = Path("snookerhelp/calibration/camera_core.py").read_text(encoding="utf-8")

    assert "class HomographyCameraModel" in core_text
    assert "class PinholeCameraModel" in core_text
    assert not Path("vision/camera_model.py").exists()


def test_sphere_projection_is_recognition_package_owned() -> None:
    core_text = Path("snookerhelp/recognition/sphere_projection.py").read_text(
        encoding="utf-8"
    )

    assert "def project_sphere_silhouette" in core_text
    assert not Path("vision/sphere_projection.py").exists()
