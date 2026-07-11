from __future__ import annotations

from snookerhelp.qa.ellipse_accuracy import compare_ellipses


def _ellipse(*, center=(100.0, 200.0), major=90.0, minor=72.0, angle=7.0):
    return {
        "center_px": list(center),
        "major_axis_px": major,
        "minor_axis_px": minor,
        "angle_deg": angle,
    }


def test_identical_ellipses_have_zero_error_and_full_annotation_score() -> None:
    result = compare_ellipses(_ellipse(), _ellipse())

    assert result["status"] == "computed"
    assert result["center_error_px"] == 0.0
    assert result["contour_rms_error_px"] == 0.0
    assert result["annotation_score"] == 100.0
    assert result["score_is_ground_truth_based"] is True


def test_shifted_ellipse_reports_measured_center_and_contour_error() -> None:
    result = compare_ellipses(
        _ellipse(center=(103.0, 204.0)),
        _ellipse(),
    )

    assert result["status"] == "computed"
    assert result["center_error_px"] == 5.0
    assert result["contour_rms_error_px"] > 0.0
    assert result["annotation_score"] < 100.0

