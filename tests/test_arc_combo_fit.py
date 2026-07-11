import math

import numpy as np

from snookerhelp.recognition.arc_combo_fit import (
    arc_combination_refit,
    should_promote_arc_combination,
)


def _ellipse_arc(
    *,
    center: tuple[float, float],
    major: float,
    minor: float,
    angle_deg: float,
    start_deg: float,
    end_deg: float,
    count: int,
) -> list[list[float]]:
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    points: list[list[float]] = []
    for local_angle in np.linspace(math.radians(start_deg), math.radians(end_deg), count):
        x_local = (major / 2.0) * math.cos(local_angle)
        y_local = (minor / 2.0) * math.sin(local_angle)
        x = center[0] + x_local * cos_t - y_local * sin_t
        y = center[1] + x_local * sin_t + y_local * cos_t
        points.append([x, y])
    return points


def test_arc_combo_refit_promotes_shared_shape_from_rejected_arc_clusters() -> None:
    """Dense-cluster bridge: useful rejected arcs can recover a shared-shape fit."""

    prior = {
        "consensus_major_axis_px": 100.0,
        "consensus_minor_axis_px": 80.0,
        "consensus_angle_deg": 5.0,
    }
    true_arc_a = _ellipse_arc(
        center=(1000.0, 1000.0),
        major=100.0,
        minor=80.0,
        angle_deg=5.0,
        start_deg=18.0,
        end_deg=72.0,
        count=22,
    )
    true_arc_b = _ellipse_arc(
        center=(1000.0, 1000.0),
        major=100.0,
        minor=80.0,
        angle_deg=5.0,
        start_deg=198.0,
        end_deg=252.0,
        count=22,
    )
    false_reflection = _ellipse_arc(
        center=(1038.0, 986.0),
        major=145.0,
        minor=72.0,
        angle_deg=31.0,
        start_deg=112.0,
        end_deg=160.0,
        count=20,
    )
    raw_points = false_reflection + true_arc_a + true_arc_b

    refit = arc_combination_refit(
        points_px=false_reflection,
        rejected_points_px=true_arc_a + true_arc_b,
        filter_stats={"raw_points_px": raw_points},
        cluster_shape_prior=prior,
        neighbor_ellipses=[],
    )
    promote, reasons = should_promote_arc_combination(refit)

    assert refit["status"] == "improved"
    assert promote is True, reasons
    best = refit["best"]
    assert best["shape_model"] == "cluster_shape_fixed"
    assert best["group_count"] >= 2
    assert best["point_count"] >= 18
    assert best["cluster_shape_comparison"]["is_shape_outlier"] is False
    center = best["ellipse_fit"]["center_px"]
    assert abs(center[0] - 1000.0) < 3.0
    assert abs(center[1] - 1000.0) < 3.0


def test_arc_combo_refit_does_not_promote_single_arc_shared_shape() -> None:
    """A single short arc is underconstrained and must remain diagnostic."""

    prior = {
        "consensus_major_axis_px": 100.0,
        "consensus_minor_axis_px": 80.0,
        "consensus_angle_deg": 5.0,
    }
    false_reflection = _ellipse_arc(
        center=(1038.0, 986.0),
        major=145.0,
        minor=72.0,
        angle_deg=31.0,
        start_deg=112.0,
        end_deg=160.0,
        count=24,
    )
    single_true_arc = _ellipse_arc(
        center=(1000.0, 1000.0),
        major=100.0,
        minor=80.0,
        angle_deg=5.0,
        start_deg=18.0,
        end_deg=72.0,
        count=24,
    )

    refit = arc_combination_refit(
        points_px=false_reflection,
        rejected_points_px=single_true_arc,
        filter_stats={"raw_points_px": false_reflection + single_true_arc},
        cluster_shape_prior=prior,
        neighbor_ellipses=[],
    )
    promote, reasons = should_promote_arc_combination(refit)

    assert promote is False
    assert reasons


def test_arc_combo_promotes_strong_non_outlier_improvement() -> None:
    promote, reasons = should_promote_arc_combination(
        {
            "status": "improved",
            "best": {
                "group_count": 3,
                "point_count": 53,
                "point_fraction": 0.78,
                "ellipse_rms_residual_px": 0.97,
                "shape_score_improvement": 77.5,
                "cluster_shape_comparison": {
                    "score": 77.5,
                    "is_shape_outlier": False,
                    "reasons": ["shape_prior_match"],
                },
            },
        }
    )

    assert promote is True
    assert reasons == []


def test_arc_combo_rejects_low_rms_shape_outlier() -> None:
    promote, reasons = should_promote_arc_combination(
        {
            "status": "improved",
            "best": {
                "group_count": 3,
                "point_count": 23,
                "point_fraction": 0.24,
                "ellipse_rms_residual_px": 0.81,
                "shape_score_improvement": 47.95,
                "cluster_shape_comparison": {
                    "score": 47.95,
                    "is_shape_outlier": True,
                    "reasons": ["cluster_ellipse_angle_outlier"],
                },
            },
        }
    )

    assert promote is False
    assert "best_is_shape_outlier" in reasons


def test_arc_combo_rejects_candidate_owned_by_neighbor() -> None:
    promote, reasons = should_promote_arc_combination(
        {
            "status": "improved",
            "best": {
                "group_count": 3,
                "point_count": 60,
                "point_fraction": 0.70,
                "ellipse_rms_residual_px": 0.90,
                "shape_score_improvement": 40.0,
                "cluster_shape_comparison": {
                    "score": 88.0,
                    "is_shape_outlier": False,
                    "reasons": ["shape_prior_match"],
                },
                "boundary_ownership": {
                    "neighbor_owned_fraction": 0.55,
                    "target_owned_fraction": 0.25,
                },
            },
        }
    )

    assert promote is False
    assert "neighbor_owned_fraction=0.55" in reasons
