from pathlib import Path


def test_accuracy_and_validation_helpers_are_package_owned() -> None:
    assert "def build_accuracy_report" in Path("snookerhelp/qa/accuracy.py").read_text(
        encoding="utf-8"
    )
    assert "def classify_table_region" in Path("snookerhelp/qa/validation.py").read_text(
        encoding="utf-8"
    )


def test_vision_accuracy_and_validation_sources_are_deleted() -> None:
    assert not Path("vision/accuracy.py").exists()
    assert not Path("vision/validation.py").exists()


def test_v1_validation_clis_use_package_qa_helpers() -> None:
    for path in [
        Path("snookerhelp/qa/accuracy_cli.py"),
        Path("snookerhelp/qa/cushion_touch_cli.py"),
        Path("snookerhelp/qa/repeatability_cli.py"),
        Path("snookerhelp/qa/spot_positions_cli.py"),
        Path("snookerhelp/qa/touching_balls_cli.py"),
    ]:
        text = path.read_text(encoding="utf-8")
        assert "snookerhelp.qa.validation" in text or "snookerhelp.qa.accuracy" in text
