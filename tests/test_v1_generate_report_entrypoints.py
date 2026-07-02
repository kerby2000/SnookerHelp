from pathlib import Path


def test_active_generate_image_report_command_uses_v1_package_entrypoint() -> None:
    text = Path("tools/generate_image_report.py").read_text(encoding="utf-8")

    assert "snookerhelp.tools.generate_reports" in text
    assert "vision.reporting" not in text


def test_active_generate_dataset_reports_command_uses_v1_package_entrypoint() -> None:
    text = Path("tools/generate_dataset_reports.py").read_text(encoding="utf-8")

    assert "snookerhelp.tools.generate_reports" in text
    assert "vision.reporting" not in text


def test_v1_dataset_index_points_to_v1_review_ui() -> None:
    text = Path("snookerhelp/tools/generate_reports.py").read_text(encoding="utf-8")

    assert "python tools/review_reports.py --reports" in text
    assert "OK/NOK decisions" in text
    assert "Export feedback JSON" not in text


def test_legacy_report_generation_commands_are_deleted() -> None:
    assert not Path("legacy/tools_v0/generate_image_report.py").exists()
    assert not Path("legacy/tools_v0/generate_dataset_reports.py").exists()
