from pathlib import Path

from snookerhelp.qa.report_html import PANEL_FILES, render_report_html


def test_static_report_html_is_v1_owned_qa_artifact() -> None:
    html = render_report_html(
        {
            "image": "Media/example/DSC00001.JPG",
            "summary": {"ball_count": 0},
            "physical_validation": {"rows": [], "summary": {}},
            "selected_ball": {"id": None, "label": None},
            "coordinate_rows": [],
            "camera_model": {"mode": "test"},
            "state": {"balls": []},
            "ball_review_rows": [],
        }
    )

    assert PANEL_FILES
    assert "SnookerHelp visual debug report" in html
    assert "static QA artifact" in html
    assert "snookerhelp_visual_feedback_v1" not in html


def test_old_static_feedback_renderer_has_been_deleted() -> None:
    active_text = Path("snookerhelp/qa/report_html.py").read_text(encoding="utf-8")

    assert "legacy.static_reports_v0" not in active_text
    assert not Path("legacy/static_reports_v0/report_html.py").exists()
    assert not Path("vision/report_html.py").exists()
