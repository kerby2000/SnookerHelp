from pathlib import Path


def test_active_annotation_command_uses_v1_review_package_entrypoint() -> None:
    text = Path("tools/annotate_ball_centers.py").read_text(encoding="utf-8")

    assert "snookerhelp.review.annotation" in text
    assert "vision.state_estimator" not in text


def test_v1_review_annotation_module_owns_editor_logic() -> None:
    text = Path("snookerhelp/review/annotation.py").read_text(encoding="utf-8")

    assert "Detector-seeded manual annotation editor" in text
    assert "def main(argv:" in text


def test_legacy_annotation_command_is_deleted() -> None:
    assert not Path("legacy/tools_v0/annotate_ball_centers.py").exists()
