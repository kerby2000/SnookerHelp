from snookerhelp.recognition.cluster_promotion import (
    should_promote_cluster_joint_center,
)


BASE_SETTINGS = {
    "joint_center_promotion_enabled": True,
    "joint_center_promotion_min_component_size": 4,
    "joint_center_promotion_roles": ["interior"],
    "joint_center_promotion_min_improvement_mm": 0.75,
    "joint_center_promotion_min_movement_mm": 0.25,
    "joint_center_promotion_max_movement_mm": 10.0,
    "joint_center_promotion_require_weak_image_evidence": True,
    "joint_center_promotion_max_view_score_for_weak": 75.0,
    "joint_center_promotion_min_physical_residual_for_weak_px": 2.0,
}


def test_promotes_weak_interior_cluster_joint_center() -> None:
    promote, reasons = should_promote_cluster_joint_center(
        ball={
            "source_final_center_policy": {
                "selected_score": 49.0,
                "point_count": 58,
            },
            "source_sphere_projection": {
                "observed_fit_score": {"rms_px": 2.5},
            },
        },
        joint={
            "cluster_status": "optimized",
            "component_size": 15,
            "cluster_role": "interior",
            "joint_xy_mm": [100.0, 120.0],
            "initial_xy_mm": [104.0, 122.0],
            "movement_mm": 4.5,
            "improvement_mm": 2.1,
            "cluster_shape_outlier": True,
        },
        settings=BASE_SETTINGS,
    )

    assert promote is True
    assert any(reason.startswith("weak_evidence:") for reason in reasons)


def test_does_not_promote_high_quality_perimeter_ball() -> None:
    promote, reasons = should_promote_cluster_joint_center(
        ball={
            "source_final_center_policy": {
                "selected_score": 94.0,
                "point_count": 112,
            },
            "source_sphere_projection": {
                "observed_fit_score": {"rms_px": 0.8},
            },
        },
        joint={
            "cluster_status": "optimized",
            "component_size": 15,
            "cluster_role": "perimeter",
            "joint_xy_mm": [100.0, 120.0],
            "initial_xy_mm": [101.0, 120.0],
            "movement_mm": 1.0,
            "improvement_mm": 2.1,
            "cluster_shape_outlier": False,
        },
        settings=BASE_SETTINGS,
    )

    assert promote is False
    assert "cluster_role=perimeter" in reasons
    assert "image_evidence_not_weak" in reasons


def test_does_not_promote_implausibly_large_joint_move() -> None:
    promote, reasons = should_promote_cluster_joint_center(
        ball={
            "source_final_center_policy": {"selected_score": 40.0},
            "source_sphere_projection": {
                "observed_fit_score": {"rms_px": 3.4},
            },
        },
        joint={
            "cluster_status": "optimized",
            "component_size": 8,
            "cluster_role": "interior",
            "joint_xy_mm": [100.0, 120.0],
            "initial_xy_mm": [120.0, 120.0],
            "movement_mm": 20.0,
            "improvement_mm": 3.0,
        },
        settings=BASE_SETTINGS,
    )

    assert promote is False
    assert "movement_mm=20.000>10.000" in reasons


def test_does_not_promote_when_cluster_fit_did_not_improve() -> None:
    promote, reasons = should_promote_cluster_joint_center(
        ball={
            "source_final_center_policy": {"selected_score": 50.0},
            "source_sphere_projection": {
                "observed_fit_score": {"rms_px": 2.4},
            },
        },
        joint={
            "cluster_status": "no_better_solution",
            "component_size": 8,
            "cluster_role": "interior",
            "joint_xy_mm": [100.0, 120.0],
            "initial_xy_mm": [103.0, 120.0],
            "movement_mm": 3.0,
            "improvement_mm": 0.1,
        },
        settings=BASE_SETTINGS,
    )

    assert promote is False
    assert "cluster_status=no_better_solution" in reasons
    assert "improvement_mm=0.100<0.750" in reasons
