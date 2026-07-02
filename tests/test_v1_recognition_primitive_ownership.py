from pathlib import Path


def test_ellipse_fit_is_recognition_package_owned() -> None:
    core_text = Path("snookerhelp/recognition/image_model.py").read_text(encoding="utf-8")

    assert "def fit_ellipse_payload" in core_text
    assert not Path("vision/ellipse_fit.py").exists()


def test_color_classifier_is_recognition_package_owned() -> None:
    core_text = Path("snookerhelp/recognition/color.py").read_text(encoding="utf-8")

    assert "class BallColorClassifier" in core_text
    assert "class ColorMeasurement" in core_text
    assert not Path("vision/ball_color_classifier.py").exists()
