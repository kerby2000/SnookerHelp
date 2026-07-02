from pathlib import Path


def test_v1_report_generation_uses_package_owned_builder() -> None:
    text = Path("snookerhelp/tools/generate_reports.py").read_text(encoding="utf-8")

    assert "snookerhelp.qa.reporting import generate_image_report" in text
    assert "vision.reporting" not in text


def test_vision_reporting_source_is_deleted() -> None:
    assert not Path("vision/reporting.py").exists()


def test_v1_reporting_module_owns_generate_image_report() -> None:
    text = Path("snookerhelp/qa/reporting.py").read_text(encoding="utf-8")

    assert "def generate_image_report" in text
    assert "report.json" in text
