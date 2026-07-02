from pathlib import Path


def test_active_export_review_feedback_command_uses_v1_exporter() -> None:
    text = Path("tools/export_review_feedback.py").read_text(encoding="utf-8")

    assert "snookerhelp.tools.export_feedback" in text
    assert "legacy.tools_v0" not in text
    assert "_rows_for_report" not in text


def test_legacy_export_review_feedback_command_is_deleted() -> None:
    assert not Path("legacy/tools_v0/export_review_feedback.py").exists()
