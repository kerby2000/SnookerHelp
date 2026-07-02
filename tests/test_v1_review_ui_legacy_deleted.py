from pathlib import Path


def test_old_review_app_and_review_server_sources_are_deleted() -> None:
    assert not Path("vision/review_app.py").exists()
    assert not Path("legacy/ui_v0/review_app.py").exists()
    assert not Path("legacy/tools_v0/review_reports.py").exists()
    assert not Path("legacy/tools_v0/review_reports_v1.py").exists()


def test_normal_review_reports_command_uses_v1_server() -> None:
    text = Path("tools/review_reports.py").read_text(encoding="utf-8")

    assert "snookerhelp.review.server" in text
    assert "vision.review_app" not in text
    assert "render_review_app_html" not in text
