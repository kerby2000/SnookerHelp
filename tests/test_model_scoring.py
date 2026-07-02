from __future__ import annotations

from snookerhelp.recognition.confidence import (
    combined_confidence,
    physics_c_only_score,
    physics_first_score,
)


def _ball(rms_error_px: float) -> dict:
    return {
        "source_refinement_success": True,
        "source_refined_center_px": [100.0, 100.0],
        "source_radius_px": 40.0,
        "source_sphere_projection": {
            "status": "predicted",
            "approximate": True,
            "observed_fit_score": {
                "status": "scored",
                "rms_error_px": rms_error_px,
                "mean_abs_error_px": rms_error_px * 0.8,
                "p95_abs_error_px": rms_error_px * 1.5,
            },
        },
        "review_confidence": 0.12,
    }


def test_physics_first_raises_confidence_when_sphere_and_observed_evidence_agree() -> None:
    score = physics_first_score(
        ball=_ball(4.0),
        evidence_agreement={
            "status": "agreement_high",
            "radial_point_count": 120,
            "mask_point_count": 180,
        },
        consensus_ellipse={"center_px": [100.0, 100.0]},
        current_decision={
            "status": "review",
            "selected_model": "circle_radial",
            "table_position_trust": "low",
        },
        warnings=["elongated"],
    )

    assert score["status"] == "accepted"
    assert score["selected_model"] == "physics_sphere_projection"
    assert score["table_position_trust"] == "medium"
    assert score["confidence"] > 0.7
    assert combined_confidence(0.12, score) == score["confidence"]


def test_physics_first_keeps_low_confidence_for_bad_sphere_residual() -> None:
    score = physics_first_score(
        ball=_ball(18.0),
        evidence_agreement={
            "status": "agreement_high",
            "radial_point_count": 120,
            "mask_point_count": 180,
        },
        consensus_ellipse={"center_px": [100.0, 100.0]},
        current_decision={
            "status": "review",
            "selected_model": "circle_radial",
            "table_position_trust": "low",
        },
        warnings=["sphere_projection_mismatch"],
    )

    assert score["status"] == "review"
    assert score["table_position_trust"] == "low"
    assert score["sphere_grade"]["level"] == "low"
    assert combined_confidence(0.42, score) == 0.42


def test_physics_first_falls_back_when_projection_unavailable() -> None:
    ball = _ball(3.0)
    ball["source_sphere_projection"] = {"status": "unavailable", "reason": "no camera"}

    score = physics_first_score(
        ball=ball,
        evidence_agreement={"status": "agreement_low"},
        consensus_ellipse=None,
        current_decision={
            "status": "fallback",
            "selected_model": "fallback_radial",
            "table_position_trust": "low",
        },
        warnings=["fallback_suspicious"],
    )

    assert score["status"] == "fallback"
    assert score["confidence"] is None


def test_physics_c_only_ignores_mask_agreement_and_uses_candidate_c() -> None:
    score = physics_c_only_score(
        ball={
            **_ball(6.5),
            "source_boundary_points_px": [[0.0, 0.0]] * 100,
        },
        candidate_c_ellipse={
            "source": "radial_edge",
            "axis_ratio": 1.25,
            "center_px": [100.0, 100.0],
        },
        current_decision={
            "status": "review",
            "selected_model": "circle_radial",
            "table_position_trust": "low",
        },
        warnings=["mask_centroid_disagreement"],
    )

    assert score["status"] == "review"
    assert score["selected_model"] == "physics_c_only_observed_ellipse"
    assert score["candidate_c_grade"]["level"] == "high"
    assert score["confidence"] > 0.6
