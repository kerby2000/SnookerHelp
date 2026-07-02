import pytest

from snookerhelp.qa.accuracy import build_accuracy_report, match_points_by_class


def point(identifier: int, label: str, x: float, y: float) -> dict:
    return {
        "id": identifier,
        "label": label,
        "x_px": x,
        "y_px": y,
        "x_mm": x / 2,
        "y_mm": y / 2,
    }


def test_class_aware_nearest_neighbor_matching() -> None:
    detections = [
        point(1, "red", 10.5, 10.0),
        point(2, "red", 50.0, 50.5),
        point(3, "blue", 30.0, 30.0),
    ]
    annotations = [
        point(10, "red", 50.0, 50.0),
        point(11, "red", 10.0, 10.0),
        point(12, "blue", 30.0, 30.0),
    ]

    matches, missed, extras = match_points_by_class(
        detections, annotations
    )

    assert not missed
    assert not extras
    assert {
        (match["detection"]["id"], match["annotation"]["id"])
        for match in matches
    } == {(1, 11), (2, 10), (3, 12)}


def test_accuracy_report_converts_pixel_error_to_millimeters() -> None:
    detections = [point(1, "red", 13.0, 14.0)]
    annotations = [point(10, "red", 10.0, 10.0)]

    report = build_accuracy_report(
        detections=detections,
        annotations=annotations,
        px_per_mm=2.0,
    )

    assert report["summary"]["matched_balls"] == 1
    assert report["matches"][0]["error_px"] == pytest.approx(5.0)
    assert report["matches"][0]["error_mm"] == pytest.approx(2.5)
    assert report["summary"]["mean_error_mm"] == pytest.approx(2.5)
