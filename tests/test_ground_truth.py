from __future__ import annotations

import json

from snookerhelp.core.ground_truth import (
    GROUND_TRUTH_SCHEMA,
    ground_truth_from_dict,
    load_ground_truth,
    save_ground_truth,
    upsert_ball_ground_truth,
)
from snookerhelp.core.schema import GroundTruthBall, GroundTruthEllipse


def test_ground_truth_normalizes_ellipse_axes_and_round_trips(tmp_path) -> None:
    value = ground_truth_from_dict(
        {
            "schema_version": GROUND_TRUTH_SCHEMA,
            "image_name": "DSC00540",
            "coordinate_system": "source_px",
            "balls": [
                {
                    "ball_id": 9,
                    "label": "red",
                    "point": [1200, 2100],
                    "ellipse_px": {
                        "center_px": [1200, 2100],
                        "major_axis_px": 74,
                        "minor_axis_px": 96,
                        "angle_deg": 15,
                        "visible_arcs_deg": [[350, 20]],
                    },
                }
            ],
        },
        image_name="DSC00540",
    )

    ellipse = value.balls[0].ellipse_px
    assert ellipse is not None
    assert ellipse.major_axis_px == 96
    assert ellipse.minor_axis_px == 74
    assert ellipse.angle_deg == 105

    output = save_ground_truth(value, tmp_path / "DSC00540.json")
    loaded = load_ground_truth(output)
    assert loaded.to_dict() == value.to_dict()
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == GROUND_TRUTH_SCHEMA


def test_upsert_ball_ground_truth_replaces_matching_ball_id() -> None:
    base = ground_truth_from_dict(
        {
            "balls": [
                {
                    "ball_id": 9,
                    "label": "red",
                    "point": [10, 20],
                }
            ]
        },
        image_name="DSC00540",
    )
    replacement = GroundTruthBall(
        ball_id=9,
        label="red",
        coordinate_system="source_px",
        point=[11.0, 21.0],
        ellipse_px=GroundTruthEllipse(
            center_px=[11.0, 21.0],
            major_axis_px=90.0,
            minor_axis_px=75.0,
            angle_deg=4.0,
        ),
    )

    updated = upsert_ball_ground_truth(base, replacement)

    assert len(updated.balls) == 1
    assert updated.balls[0].point == [11.0, 21.0]
    assert updated.balls[0].ellipse_px is not None

