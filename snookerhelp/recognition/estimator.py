from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from snookerhelp.core.schema import TableState
from snookerhelp.recognition.evidence import table_state_from_legacy_report

import cv2
import numpy as np

from snookerhelp.recognition.classical_rough_detector import ClassicalBallDetector, DetectionResult
from snookerhelp.calibration.camera_core import build_camera_model, configured_z_planes, z_plane_key
from snookerhelp.core.config import PROJECT_ROOT, load_yaml, resolve_path
from snookerhelp.review.overlay import draw_source_overlay, draw_warped_overlay
from snookerhelp.recognition.source_refinement import (
    fit_radial_boundary_variant_from_feature,
    refine_source_ball,
)
from snookerhelp.recognition.evidence_maps import (
    compute_ball_evidence_maps,
    compute_full_table_evidence_maps,
    estimate_global_cloth_reference,
)
from snookerhelp.recognition.physical_optimize import (
    optimize_ball_xy_from_sphere_projection,
)
from snookerhelp.recognition.cluster_optimizer import optimize_cluster_scene
from snookerhelp.recognition.joint_cluster_solver import (
    solve_joint_cluster_components,
)
from snookerhelp.recognition.arc_combo_fit import (
    arc_combination_refit,
    should_promote_arc_combination,
)
from snookerhelp.recognition.cluster_promotion import (
    cluster_joint_promotion_payload,
    should_promote_cluster_joint_center,
)
from snookerhelp.recognition.sphere_projection import (
    project_sphere_silhouette,
    score_observed_points_against_silhouette,
)
from snookerhelp.core.table import TableModel
from snookerhelp.calibration.homography_bootstrap import TableWarp


@dataclass(frozen=True)
class ProcessedFrame:
    state: dict[str, Any]
    source_overlay: np.ndarray
    warped_overlay: np.ndarray
    difference_visualization: np.ndarray


class StateEstimator:
    def __init__(
        self,
        pipeline_config: dict[str, Any],
        table: TableModel,
        table_warp: TableWarp,
        detector: ClassicalBallDetector,
        background_path: Path,
        camera_model: Any,
        projection_z_planes_mm: list[float],
    ):
        self.pipeline_config = pipeline_config
        self.table = table
        self.table_warp = table_warp
        self.detector = detector
        self.background_path = background_path
        self.camera_model = camera_model
        self.projection_z_planes_mm = projection_z_planes_mm
        background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        if background is None:
            raise FileNotFoundError(f"Could not read background: {background_path}")
        self.source_background = background
        self.warped_background = table_warp.warp_image(background)

    @classmethod
    def from_config(
        cls, config_path: str | Path = "configs/sony_dev.yaml"
    ) -> "StateEstimator":
        pipeline_config = load_yaml(config_path)
        pipeline = pipeline_config["pipeline"]
        table_config = load_yaml(pipeline["table_model"])
        calibration_config = load_yaml(pipeline["calibration"])
        detector_config = load_yaml(pipeline["detector"])
        camera_config = _camera_model_config(pipeline.get("camera_model", {}))

        table = TableModel.from_config(table_config)
        table_warp = TableWarp.from_corners(
            table, calibration_config["corner_points_px"]
        )
        margin = float(table.processing_margin_px)
        detector = ClassicalBallDetector(
            detector_config,
            table.ball_radius_px,
            surface_bounds_px=(
                margin,
                margin,
                float(table.warp_width_px - margin - 1),
                float(table.warp_height_px - margin - 1),
            ),
        )
        background_path = resolve_path(detector_config["background_image"])
        camera_model = build_camera_model(table_warp, camera_config)
        projection_z_planes_mm = configured_z_planes(camera_config)
        return cls(
            pipeline_config=pipeline_config,
            table=table,
            table_warp=table_warp,
            detector=detector,
            background_path=background_path,
            camera_model=camera_model,
            projection_z_planes_mm=projection_z_planes_mm,
        )

    def process(self, image_path: str | Path) -> ProcessedFrame:
        source_path = resolve_path(image_path)
        source_image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source_image is None:
            raise FileNotFoundError(f"Could not read image: {source_path}")

        warped_image = self.table_warp.warp_image(source_image)
        result = self.detector.detect(warped_image, self.warped_background)
        result = self._add_source_refinement(source_image, result)
        state = self._build_state(source_path, source_image, result)
        return ProcessedFrame(
            state=state,
            source_overlay=draw_source_overlay(
                source_image, result.balls, self.table_warp
            ),
            warped_overlay=draw_warped_overlay(
                warped_image, result.balls, self.table_warp
            ),
            difference_visualization=result.difference_visualization,
        )

    def process_and_save(
        self,
        image_path: str | Path,
        output_root: str | Path | None = None,
    ) -> tuple[ProcessedFrame, Path]:
        frame = self.process(image_path)
        source_path = resolve_path(image_path)
        if output_root is None:
            output_root = self.pipeline_config["output"]["directory"]
        output_directory = resolve_path(output_root) / source_path.stem
        output_directory.mkdir(parents=True, exist_ok=True)

        with (output_directory / f"{source_path.stem}_state.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(frame.state, handle, indent=2)
            handle.write("\n")

        output_config = self.pipeline_config.get("output", {})
        if output_config.get("save_source_overlay", True):
            cv2.imwrite(
                str(output_directory / f"{source_path.stem}_overlay.jpg"),
                frame.source_overlay,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
        if output_config.get("save_warped_overlay", True):
            cv2.imwrite(
                str(output_directory / f"{source_path.stem}_warped.jpg"),
                frame.warped_overlay,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
        if output_config.get("save_difference", True):
            cv2.imwrite(
                str(output_directory / f"{source_path.stem}_difference.png"),
                frame.difference_visualization,
            )
        return frame, output_directory

    def _build_state(
        self,
        source_path: Path,
        source_image: np.ndarray,
        result: DetectionResult,
    ) -> dict[str, Any]:
        balls: list[dict[str, Any]] = []
        evidence_map_settings = dict(self.detector.config.get("evidence_maps", {}))
        global_cloth_model = estimate_global_cloth_reference(
            source_image=source_image,
            table_corners_px=self.table_warp.corner_points_px,
            balls=result.balls,
            settings=evidence_map_settings,
        )
        full_table_evidence_maps = compute_full_table_evidence_maps(
            source_image=source_image,
            table_corners_px=self.table_warp.corner_points_px,
            settings={
                **evidence_map_settings,
                "global_cloth_model": global_cloth_model,
            },
        )
        for index, ball in enumerate(result.balls, start=1):
            x_mm, y_mm = self.table_warp.warped_px_to_table_mm(
                ball.x_px, ball.y_px
            )
            initial_by_z = self._source_point_to_table_by_z(ball)
            initial_sphere_projection = self._source_sphere_projection(
                ball,
                initial_by_z,
            )
            evidence_maps = self._source_evidence_maps(
                source_image=source_image,
                ball=ball,
                sphere_projection=initial_sphere_projection,
                global_cloth_model=global_cloth_model,
                full_table_evidence_maps=full_table_evidence_maps,
            )
            neighbor_ellipses = self._neighbor_source_ellipses(result, index)
            final_evidence = self._source_final_image_evidence(
                ball=ball,
                evidence_maps=evidence_maps,
                neighbor_ellipses=neighbor_ellipses,
            )
            final_source_px = final_evidence.get("center_px") or _point_or_none(
                ball.source_x_px,
                ball.source_y_px,
            )
            final_boundary_points = final_evidence.get("boundary_points_px") or list(
                ball.source_boundary_points_px
            )
            final_rejected_points = final_evidence.get(
                "boundary_rejected_points_px",
            ) or list(ball.source_boundary_rejected_points_px)
            final_ellipse_fit = final_evidence.get("ellipse_fit") or ball.source_ellipse_fit
            by_z = self._source_point_to_table_by_z(ball, source_px=final_source_px)
            ball_radius_key = z_plane_key(self.table.ball_radius_mm)
            source_radius_projection = by_z.get(ball_radius_key)
            source_radius_xy = (
                source_radius_projection["xy_mm"]
                if source_radius_projection is not None
                else None
            )
            sphere_projection = self._source_sphere_projection(
                ball,
                by_z,
                observed_points=final_boundary_points,
                observed_source=final_evidence.get("observed_source")
                or "source_final_boundary_points_px",
            )
            physical_optimization = self._source_physical_optimization(
                ball=ball,
                by_z=by_z,
                sphere_projection=sphere_projection,
                evidence_maps=evidence_maps,
                result=result,
                current_index=index,
                observed_points=final_boundary_points,
            )
            sphere_projection = self._sphere_projection_with_optimization(
                sphere_projection=sphere_projection,
                physical_optimization=physical_optimization,
                observed_points=final_boundary_points,
            )
            balls.append(
                {
                    "id": index,
                    "class": ball.label,
                    "color_label": ball.label,
                    "x_mm": round(x_mm, 2),
                    "y_mm": round(y_mm, 2),
                    "table_xy_mm": [round(x_mm, 4), round(y_mm, 4)],
                    "table_xy_mm_approximate": True,
                    "table_xy_mm_method": (
                        "cloth_plane_homography_from_warped_center"
                    ),
                    "raw_hough_center_px": [
                        round(ball.raw_x_px, 4),
                        round(ball.raw_y_px, 4),
                    ],
                    "warped_center_px": [
                        round(ball.x_px, 4),
                        round(ball.y_px, 4),
                    ],
                    "refined_center_px": [
                        round(ball.x_px, 4),
                        round(ball.y_px, 4),
                    ],
                    "source_rough_center_px": _point_or_none(
                        ball.source_rough_x_px,
                        ball.source_rough_y_px,
                    ),
                    "source_initial_refined_center_px": _point_or_none(
                        ball.source_x_px,
                        ball.source_y_px,
                    ),
                    "source_refined_center_px": _point_or_none_from_sequence(
                        final_source_px,
                    ),
                    "source_final_center_px": _point_or_none_from_sequence(
                        final_source_px,
                    ),
                    "source_final_center_policy": final_evidence,
                    "source_refined_warped_center_px": _point_or_none(
                        *self._source_point_to_warped(ball, source_px=final_source_px)
                    ),
                    "source_refined_table_xy_mm": _point_or_none_from_sequence(
                        source_radius_xy
                    ),
                    "source_refined_table_xy_mm_approximate": True,
                    "source_refined_table_xy_by_z_mm": _round_projection_by_z(by_z),
                    "raw_hough_radius_px": round(ball.raw_radius_px, 4),
                    "radius_px": round(ball.radius_px, 4),
                    "radius_mm": round(
                        ball.radius_px / self.table.px_per_mm, 4
                    ),
                    "source_radius_px": (
                        round(ball.source_radius_px, 4)
                        if ball.source_radius_px is not None
                        else None
                    ),
                    "source_roi_px": (
                        [int(value) for value in ball.source_roi_px]
                        if ball.source_roi_px is not None
                        else None
                    ),
                    "source_boundary_points_px": _round_points(
                        final_boundary_points
                    ),
                    "source_boundary_rejected_points_px": _round_points(
                        final_rejected_points
                    ),
                    "source_boundary_filter": final_evidence.get("filter")
                    or ball.source_boundary_filter,
                    "source_boundary_evidence_source": (
                        final_evidence.get("observed_source")
                        or ball.source_boundary_evidence_source
                    ),
                    "source_ellipse_fit": _round_ellipse_fit(
                        final_ellipse_fit
                    ),
                    "source_radial_boundary_points_px": _round_points(
                        ball.source_boundary_points_px
                    ),
                    "source_radial_boundary_rejected_points_px": _round_points(
                        ball.source_boundary_rejected_points_px
                    ),
                    "source_radial_boundary_filter": ball.source_boundary_filter,
                    "source_radial_boundary_evidence_source": (
                        ball.source_boundary_evidence_source
                    ),
                    "source_radial_ellipse_fit": _round_ellipse_fit(
                        ball.source_ellipse_fit
                    ),
                    "source_mask_centroid_px": _point_or_none_from_sequence(
                        ball.source_mask_centroid_px
                    ),
                    "source_mask_area_px": (
                        round(float(ball.source_mask_area_px), 4)
                        if ball.source_mask_area_px is not None
                        else None
                    ),
                    "source_mask_contour_points_px": _round_points(
                        ball.source_mask_contour_points_px
                    ),
                    "source_silhouette_ellipse_fit": _round_ellipse_fit(
                        ball.source_silhouette_ellipse_fit
                    ),
                    "source_sphere_projection": sphere_projection,
                    "source_evidence_maps": (
                        evidence_maps.summary if evidence_maps is not None else None
                    ),
                    "source_sphere_optimization": physical_optimization,
                    "source_fit_residual_px": (
                        round(ball.source_fit_residual_px, 4)
                        if ball.source_fit_residual_px is not None
                        else None
                    ),
                    "source_refinement_success": ball.source_refinement_success,
                    "fit_residual_px": (
                        round(ball.fit_residual_px, 4)
                        if ball.fit_residual_px is not None
                        else None
                    ),
                    "color_confidence": round(ball.color_confidence, 4),
                    "detection_confidence": round(ball.confidence, 4),
                    "confidence": round(ball.confidence, 4),
                    "debug": {
                        "warped_center_px": [
                            round(ball.x_px, 4),
                            round(ball.y_px, 4),
                        ],
                        "raw_hough_center_px": [
                            round(ball.raw_x_px, 4),
                            round(ball.raw_y_px, 4),
                        ],
                        "raw_hough_radius_px": round(
                            ball.raw_radius_px, 4
                        ),
                        "circle_refinement_success": ball.refinement_success,
                        "circle_fit_point_count": ball.fit_point_count,
                        "source_refinement_success": (
                            ball.source_refinement_success
                        ),
                        "source_circle_fit_point_count": (
                            ball.source_fit_point_count
                        ),
                        "source_rough_center_px": _point_or_none(
                            ball.source_rough_x_px,
                            ball.source_rough_y_px,
                        ),
                        "source_initial_refined_center_px": _point_or_none(
                            ball.source_x_px,
                            ball.source_y_px,
                        ),
                        "source_refined_center_px": _point_or_none_from_sequence(
                            final_source_px,
                        ),
                        "source_final_center_policy": final_evidence,
                        "source_boundary_point_count": len(
                            final_boundary_points
                        ),
                        "source_boundary_evidence_source": (
                            final_evidence.get("observed_source")
                            or ball.source_boundary_evidence_source
                        ),
                        "source_ellipse_fit": _round_ellipse_fit(
                            final_ellipse_fit
                        ),
                        "source_radial_boundary_point_count": len(
                            ball.source_boundary_points_px
                        ),
                        "source_radial_boundary_evidence_source": (
                            ball.source_boundary_evidence_source
                        ),
                        "source_radial_ellipse_fit": _round_ellipse_fit(
                            ball.source_ellipse_fit
                        ),
                        "source_mask_centroid_px": _point_or_none_from_sequence(
                            ball.source_mask_centroid_px
                        ),
                        "source_mask_area_px": (
                            round(float(ball.source_mask_area_px), 4)
                            if ball.source_mask_area_px is not None
                            else None
                        ),
                        "source_mask_contour_point_count": len(
                            ball.source_mask_contour_points_px
                        ),
                        "source_silhouette_ellipse_fit": _round_ellipse_fit(
                            ball.source_silhouette_ellipse_fit
                        ),
                        "source_sphere_projection": sphere_projection,
                        "source_evidence_maps": (
                            evidence_maps.summary if evidence_maps is not None else None
                        ),
                        "source_sphere_optimization": physical_optimization,
                        "detection_score": round(ball.detection_score, 4),
                        "color_confidence": round(
                            ball.color_confidence, 4
                        ),
                        "median_lab_difference": round(
                            ball.median_lab_difference, 3
                        ),
                        "foreground_fraction": round(
                            ball.foreground_fraction, 4
                        ),
                        "median_hsv": list(ball.hsv),
                        "median_lab": list(ball.lab),
                    },
                }
            )

        joint_cluster_solution = self._source_joint_cluster_solution(balls)
        self._apply_joint_cluster_solution(balls, joint_cluster_solution)

        cluster_optimization = self._source_cluster_optimization(balls)
        cluster_by_ball = cluster_optimization.get("by_ball_id", {})
        for ball in balls:
            ball_cluster = cluster_by_ball.get(str(ball["id"]), {})
            ball["source_joint_cluster_optimization"] = ball_cluster
            ball["debug"]["source_joint_cluster_optimization"] = ball_cluster
            if ball_cluster:
                source_optimization = ball.get("source_sphere_optimization")
                if isinstance(source_optimization, dict):
                    source_optimization["joint_cluster"] = ball_cluster
                sphere_projection_payload = ball.get("source_sphere_projection")
                if isinstance(sphere_projection_payload, dict):
                    sphere_projection_payload["joint_cluster"] = ball_cluster
        self._apply_arc_combo_promotions(balls)
        self._apply_cluster_joint_promotions(balls)

        try:
            source_name = str(source_path.relative_to(PROJECT_ROOT))
        except ValueError:
            source_name = str(source_path)
        try:
            background_name = str(self.background_path.relative_to(PROJECT_ROOT))
        except ValueError:
            background_name = str(self.background_path)

        return {
            "source_image": source_name,
            "source_size_px": {
                "width": int(source_image.shape[1]),
                "height": int(source_image.shape[0]),
            },
            "background_image": background_name,
            "table": {
                "name": self.table.name,
                "length_mm": self.table.length_mm,
                "width_mm": self.table.width_mm,
                "coordinate_origin": self.table.origin,
                "corner_points_px": self.table_warp.corner_points_px.tolist(),
                "warp_px_per_mm": self.table.px_per_mm,
                "processing_margin_mm": self.table.processing_margin_mm,
            },
            "detection": {
                "raw_candidate_count": result.raw_candidate_count,
                "gated_candidate_count": result.gated_candidate_count,
                "duplicate_suppressed_candidate_count": (
                    result.duplicate_suppressed_candidate_count
                ),
                "ball_count": len(result.balls),
                "geometry_note": (
                    "Warped image is cloth-plane rectification; ball shapes "
                    "near edges are not expected to remain circular."
                ),
                "global_cloth_model": global_cloth_model,
                "full_table_evidence_maps": (
                    full_table_evidence_maps.summary
                    if full_table_evidence_maps is not None
                    else {"status": "unavailable"}
                ),
            },
            "joint_cluster_solver": joint_cluster_solution,
            "scene_constraints": {
                "adjacent_ball_clusters": cluster_optimization,
            },
            "camera_model": {
                "mode": self.camera_model.model_name,
                "is_calibrated": bool(self.camera_model.is_calibrated),
                "is_approximate": not bool(self.camera_model.is_calibrated),
                "projection_z_planes_mm": self.projection_z_planes_mm,
                "camera_center_world_mm": _point3_or_none_from_sequence(
                    getattr(self.camera_model, "camera_center_world_mm", None)
                ),
                "geometry_note": (
                    "manual_homography mode cannot model height parallax; "
                    "pinhole modes use ray-plane intersections. "
                    "approximate_pinhole_from_corners is not a substitute for "
                    "ChArUco calibration."
                ),
            },
            "balls": balls,
        }

    def _add_source_refinement(
        self,
        source_image: np.ndarray,
        result: DetectionResult,
    ) -> DetectionResult:
        source_config = dict(self.detector.config.get("source_refinement", {}))
        difference_config = self.detector.config.get("difference", {})
        source_config.setdefault("blur_kernel", difference_config.get("blur_kernel", 5))
        balls = []
        for ball in result.balls:
            fit = refine_source_ball(
                source_image=source_image,
                source_background=self.source_background,
                table_warp=self.table_warp,
                warped_center=(ball.x_px, ball.y_px),
                warped_radius_px=ball.radius_px,
                config=source_config,
            )
            balls.append(
                replace(
                    ball,
                    source_rough_x_px=fit.rough_x,
                    source_rough_y_px=fit.rough_y,
                    source_x_px=fit.x,
                    source_y_px=fit.y,
                    source_radius_px=fit.radius,
                    source_fit_residual_px=fit.residual_px,
                    source_fit_point_count=fit.point_count,
                    source_refinement_success=fit.success,
                    source_roi_px=fit.roi,
                    source_boundary_points_px=fit.boundary_points,
                    source_boundary_rejected_points_px=fit.boundary_rejected_points,
                    source_boundary_filter=fit.boundary_filter_stats,
                    source_boundary_evidence_source=fit.boundary_evidence_source,
                    source_ellipse_fit=fit.ellipse_fit,
                    source_mask_centroid_px=fit.mask_centroid,
                    source_mask_area_px=fit.mask_area_px,
                    source_mask_contour_points_px=fit.mask_contour_points,
                    source_silhouette_ellipse_fit=fit.silhouette_ellipse_fit,
                )
            )
        return DetectionResult(
            balls=balls,
            difference=result.difference,
            difference_visualization=result.difference_visualization,
            raw_candidate_count=result.raw_candidate_count,
            gated_candidate_count=result.gated_candidate_count,
            duplicate_suppressed_candidate_count=(
                result.duplicate_suppressed_candidate_count
            ),
        )

    def _source_point_to_table_by_z(
        self,
        ball: Any,
        source_px: list[float] | tuple[float, float] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if source_px is None:
            if ball.source_x_px is None or ball.source_y_px is None:
                return {}
            source_px = [float(ball.source_x_px), float(ball.source_y_px)]
        if len(source_px) < 2:
            return {}
        return self.camera_model.project_image_point_to_z_planes(
            (float(source_px[0]), float(source_px[1])),
            self.projection_z_planes_mm,
        )

    def _source_sphere_projection(
        self,
        ball: Any,
        by_z: dict[str, dict[str, Any]],
        observed_points: list[Any] | tuple[Any, ...] | None = None,
        observed_source: str | None = None,
    ) -> dict[str, Any]:
        ball_radius_key = z_plane_key(self.table.ball_radius_mm)
        projection = by_z.get(ball_radius_key)
        if projection is None:
            return {
                "status": "unavailable",
                "reason": "no ball-center Z-plane projection is available",
            }
        sphere = project_sphere_silhouette(
            self.camera_model,
            projection["xyz_mm"],
            self.table.ball_radius_mm,
        )
        observed_source = observed_source or "source_boundary_points_px"
        observed_points = observed_points if observed_points is not None else ball.source_boundary_points_px
        if not observed_points:
            observed_source = "source_mask_contour_points_px"
            observed_points = ball.source_mask_contour_points_px
        score = score_observed_points_against_silhouette(observed_points, sphere)
        if score is not None:
            sphere = {
                **sphere,
                "observed_fit_score": {
                    **score,
                    "source": observed_source,
                },
            }
        return sphere

    def _source_evidence_maps(
        self,
        *,
        source_image: np.ndarray,
        ball: Any,
        sphere_projection: dict[str, Any],
        global_cloth_model: dict[str, Any] | None = None,
        full_table_evidence_maps: Any | None = None,
    ) -> Any:
        settings = dict(self.detector.config.get("evidence_maps", {}))
        if not bool(settings.get("enabled", True)):
            return None
        if global_cloth_model is not None:
            settings["global_cloth_model"] = global_cloth_model
        if full_table_evidence_maps is not None:
            settings["_full_table_evidence_maps"] = full_table_evidence_maps
        center = _point_or_none(ball.source_x_px, ball.source_y_px) or _point_or_none(
            ball.source_rough_x_px,
            ball.source_rough_y_px,
        )
        return compute_ball_evidence_maps(
            source_image=source_image,
            center_px=center,
            radius_px=ball.source_radius_px,
            label=ball.label,
            sphere_projection=sphere_projection,
            settings=settings,
        )

    def _source_final_image_evidence(
        self,
        *,
        ball: Any,
        evidence_maps: Any,
        neighbor_ellipses: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        policy = dict(self.detector.config.get("evidence_maps", {})).get(
            "final_position_policy",
            {},
        )
        if not bool(policy.get("enabled", False)):
            return {
                "status": "disabled",
                "used_for_final_position": False,
                "reason": "final evidence-map policy is disabled",
            }
        if evidence_maps is None:
            return {
                "status": "unavailable",
                "used_for_final_position": False,
                "reason": "evidence maps are unavailable",
            }
        key = _final_evidence_map_key(ball.label, policy)
        feature = getattr(evidence_maps, _feature_attribute_for_map(key), None)
        if feature is None:
            return {
                "status": "unavailable",
                "used_for_final_position": False,
                "selected_map": key,
                "reason": "selected evidence map is unavailable",
            }
        variant = fit_radial_boundary_variant_from_feature(
            feature=feature,
            roi=evidence_maps.roi,
            center_px=_point_or_none(ball.source_x_px, ball.source_y_px)
            or _point_or_none(ball.source_rough_x_px, ball.source_rough_y_px),
            radius_px=ball.source_radius_px,
            evidence_source=f"final_evidence_map_{key}",
            settings=dict(self.detector.config.get("evidence_maps", {})),
            use_outward_drop=_map_uses_outward_drop(key),
            neighbor_ellipses=neighbor_ellipses,
        )
        if not variant or variant.get("status") != "computed":
            return {
                "status": "fallback",
                "used_for_final_position": False,
                "selected_map": key,
                "reason": "selected evidence map did not produce a usable boundary",
                "variant": variant,
            }
        ellipse = variant.get("ellipse_fit") or {}
        center = ellipse.get("center_px")
        if not center:
            return {
                "status": "fallback",
                "used_for_final_position": False,
                "selected_map": key,
                "reason": "selected evidence map did not produce an ellipse center",
                "variant": variant,
            }
        quality_issue = _final_evidence_quality_issue(
            ellipse=ellipse,
            reference_center=(
                _point_or_none(ball.source_x_px, ball.source_y_px)
                or _point_or_none(ball.source_rough_x_px, ball.source_rough_y_px)
            ),
            radius_px=ball.source_radius_px,
            policy=policy,
        )
        if quality_issue is not None:
            return {
                "status": "fallback",
                "used_for_final_position": False,
                "selected_map": key,
                "reason": quality_issue,
                "variant": variant,
            }
        label = _final_evidence_map_label(key)
        return {
            "status": "computed",
            "used_for_final_position": True,
            "selected_map": key,
            "selected_label": label,
            "observed_source": f"final_evidence_map_{key}",
            "center_px": [float(center[0]), float(center[1])],
            "boundary_points_px": variant.get("points_px") or [],
            "boundary_rejected_points_px": variant.get("rejected_points_px") or [],
            "ellipse_fit": ellipse,
            "filter": variant.get("filter") or {},
            "sampling": variant.get("sampling"),
            "point_count": len(variant.get("points_px") or []),
            "rejected_point_count": len(variant.get("rejected_points_px") or []),
            "reason": (
                "label override"
                if str(ball.label).lower() in (policy.get("label_overrides") or {})
                else "default map"
            ),
            "note": (
                "Only the configured final-position evidence map is promoted "
                "to final source/table coordinates. Other evidence maps remain "
                "diagnostics."
            ),
        }

    def _neighbor_source_ellipses(
        self,
        result: DetectionResult,
        current_index: int,
    ) -> list[dict[str, Any]]:
        """Return nearby source-space ball ellipses used for ownership filtering."""

        if current_index < 1 or current_index > len(result.balls):
            return []
        settings = dict(self.detector.config.get("evidence_maps", {}))
        if not bool(settings.get("neighbor_ellipse_rejection_enabled", True)):
            return []
        current = result.balls[current_index - 1]
        current_center = _source_detection_center(current)
        current_radius = _source_detection_radius(current)
        if current_center is None or current_radius is None:
            return []
        distance_factor = float(settings.get("neighbor_ellipse_rejection_distance_factor", 3.2))
        neighbors: list[dict[str, Any]] = []
        for index, other in enumerate(result.balls, start=1):
            if index == current_index:
                continue
            other_center = _source_detection_center(other)
            other_radius = _source_detection_radius(other)
            if other_center is None or other_radius is None:
                continue
            distance = float(
                np.hypot(
                    float(other_center[0]) - float(current_center[0]),
                    float(other_center[1]) - float(current_center[1]),
                ),
            )
            maximum_distance = max(
                current_radius + other_radius,
                (current_radius + other_radius) * max(1.0, distance_factor),
            )
            if distance > maximum_distance:
                continue
            ellipse = _source_detection_ellipse_payload(
                other,
                fallback_center=other_center,
                fallback_radius=other_radius,
            )
            if ellipse:
                ellipse["id"] = index
                ellipse["label"] = other.label
                ellipse["distance_px"] = round(distance, 4)
                neighbors.append(ellipse)
        return neighbors

    def _source_physical_optimization(
        self,
        *,
        ball: Any,
        by_z: dict[str, dict[str, Any]],
        sphere_projection: dict[str, Any],
        evidence_maps: Any,
        result: DetectionResult,
        current_index: int,
        observed_points: list[Any] | tuple[Any, ...] | None = None,
    ) -> dict[str, Any]:
        settings = dict(self.detector.config.get("physical_optimization", {}))
        if not bool(settings.get("enabled", True)):
            return {
                "status": "disabled",
                "enabled": False,
                "success": False,
                "reason": "physical optimization is disabled",
            }
        ball_radius_key = z_plane_key(self.table.ball_radius_mm)
        projection = by_z.get(ball_radius_key)
        if projection is None:
            return {
                "status": "unavailable",
                "enabled": True,
                "success": False,
                "reason": "no ball-center Z-plane projection is available",
            }
        observed_points = list(
            observed_points if observed_points is not None else ball.source_boundary_points_px
        )
        neighbors = self._neighbor_table_points(result, current_index)
        return optimize_ball_xy_from_sphere_projection(
            initial_xy_mm=projection["xy_mm"],
            camera_model=self.camera_model,
            observed_boundary_points_px=observed_points,
            evidence_maps=evidence_maps,
            neighbors=neighbors,
            cushion_context={
                "length_mm": self.table.length_mm,
                "width_mm": self.table.width_mm,
            },
            ball_radius_mm=self.table.ball_radius_mm,
            settings=settings,
        )

    def _source_cluster_optimization(
        self,
        balls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        settings = dict(self.detector.config.get("cluster_optimization", {}))
        return optimize_cluster_scene(
            balls,
            ball_radius_mm=self.table.ball_radius_mm,
            settings=settings,
        )

    def _source_joint_cluster_solution(
        self,
        balls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        settings = dict(self.detector.config.get("joint_cluster_solver", {}))
        return solve_joint_cluster_components(
            balls,
            camera_model=self.camera_model,
            ball_radius_mm=self.table.ball_radius_mm,
            settings=settings,
        )

    def _apply_joint_cluster_solution(
        self,
        balls: list[dict[str, Any]],
        solution: dict[str, Any],
    ) -> None:
        """Apply only component solutions that passed the joint solver gates."""

        by_ball_id = solution.get("by_ball_id") or {}
        for ball in balls:
            payload = by_ball_id.get(str(ball.get("id"))) or {}
            ball["source_global_cluster_solution"] = payload
            debug = ball.get("debug")
            if isinstance(debug, dict):
                debug["source_global_cluster_solution"] = payload
            if not payload.get("promoted"):
                continue

            center = payload.get("proposed_source_center_px")
            ellipse = payload.get("ellipse_fit")
            selected_points = payload.get("owned_boundary_points_px") or []
            if not center or not isinstance(ellipse, dict) or not selected_points:
                payload["promoted"] = False
                payload["status"] = "invalid_promotion_payload"
                payload.setdefault("promotion_reasons", []).append(
                    "missing_center_ellipse_or_owned_points"
                )
                continue

            previous_policy = ball.get("source_final_center_policy") or {}
            previous_filter = (
                previous_policy.get("filter")
                or ball.get("source_boundary_filter")
                or {}
            )
            raw_points = previous_filter.get("raw_points_px") or []
            rejected_points = _points_not_selected(raw_points, selected_points)
            promoted_filter = {
                **previous_filter,
                "status": "joint_cluster_global_ownership",
                "method": "global_arc_ownership_fixed_shape_joint_fit",
                "accepted_count": len(selected_points),
                "rejected_count": len(rejected_points),
                "joint_cluster_solver": {
                    "solver_mode": payload.get("solver_mode"),
                    "component_id": payload.get("component_id"),
                    "assigned_lattice_node": payload.get(
                        "assigned_lattice_node"
                    ),
                    "owned_boundary_rms_px": payload.get(
                        "owned_boundary_rms_px"
                    ),
                    "promotion_reasons": payload.get("promotion_reasons") or [],
                },
            }
            promoted_policy = {
                **previous_policy,
                "status": "computed",
                "used_for_final_position": True,
                "selected_map": previous_policy.get("selected_map")
                or "ball_vs_cloth_probability",
                "selected_label": previous_policy.get("selected_label")
                or "Ball-vs-cloth probability",
                "observed_source": "joint_cluster_global_arc_ownership",
                "final_center_source": "joint_cluster_global_solution",
                "center_px": [round(float(center[0]), 4), round(float(center[1]), 4)],
                "boundary_points_px": _round_points(selected_points),
                "boundary_rejected_points_px": _round_points(rejected_points),
                "ellipse_fit": _round_ellipse_fit(ellipse),
                "filter": promoted_filter,
                "point_count": len(selected_points),
                "rejected_point_count": len(rejected_points),
                "reason": "joint cluster solution passed component gates",
                "note": (
                    "Final center and outline come from one simultaneous cluster "
                    "solution. Boundary samples are globally owned and cannot "
                    "steer multiple neighboring balls."
                ),
                "joint_cluster_solver": payload,
                "independent_image_evidence": _independent_policy_summary(
                    previous_policy
                ),
            }

            ball["source_final_center_policy"] = promoted_policy
            ball["source_refined_center_px"] = promoted_policy["center_px"]
            ball["source_final_center_px"] = promoted_policy["center_px"]
            ball["source_position_source"] = "joint_cluster_global_solution"
            ball["source_boundary_points_px"] = _round_points(selected_points)
            ball["source_boundary_rejected_points_px"] = _round_points(
                rejected_points
            )
            ball["source_boundary_filter"] = promoted_filter
            ball["source_boundary_evidence_source"] = promoted_policy[
                "observed_source"
            ]
            ball["source_ellipse_fit"] = _round_ellipse_fit(ellipse)
            ball["source_refined_warped_center_px"] = _point_or_none(
                *self._source_point_to_warped(ball, source_px=center)
            )

            by_z = self.camera_model.project_image_point_to_z_planes(
                (float(center[0]), float(center[1])),
                self.projection_z_planes_mm,
            )
            ball_radius_key = z_plane_key(self.table.ball_radius_mm)
            source_radius_projection = by_z.get(ball_radius_key)
            ball["source_refined_table_xy_by_z_mm"] = _round_projection_by_z(by_z)
            ball["source_refined_table_xy_mm"] = (
                _point_or_none_from_sequence(source_radius_projection["xy_mm"])
                if source_radius_projection is not None
                else None
            )
            sphere_projection = _state_ball_sphere_projection(
                camera_model=self.camera_model,
                table_radius_mm=self.table.ball_radius_mm,
                by_z=by_z,
                observed_points=selected_points,
                observed_source="joint_cluster_global_arc_ownership",
            )
            sphere_projection["global_cluster_solution"] = payload
            ball["source_sphere_projection"] = sphere_projection
            if isinstance(debug, dict):
                debug["source_refined_center_px"] = promoted_policy["center_px"]
                debug["source_sphere_projection"] = sphere_projection

    def _apply_arc_combo_promotions(
        self,
        balls: list[dict[str, Any]],
    ) -> None:
        """Promote cluster-consistent raw-arc fits into the final image model."""

        settings = dict(self.detector.config.get("cluster_optimization", {}))
        if not bool(settings.get("legacy_arc_combo_promotion_enabled", False)):
            return
        include_fixed_shape_candidates = bool(
            settings.get("arc_combo_fixed_shape_candidates_enabled", True)
        )
        max_search_groups = int(settings.get("arc_combo_max_search_groups", 10))
        max_fixed_shape_combo_size = int(
            settings.get("arc_combo_max_fixed_shape_combo_size", 5)
        )
        for ball in balls:
            joint = ball.get("source_joint_cluster_optimization") or {}
            shape_prior = joint.get("cluster_shape_prior") or {}
            if not isinstance(shape_prior, dict):
                continue
            policy = ball.get("source_final_center_policy") or {}
            if not isinstance(policy, dict):
                continue
            variant = policy.get("variant")
            if not isinstance(variant, dict):
                variant = policy

            points_px = (
                variant.get("points_px")
                or variant.get("boundary_points_px")
                or policy.get("boundary_points_px")
                or ball.get("source_boundary_points_px")
                or []
            )
            rejected_points_px = (
                variant.get("rejected_points_px")
                or variant.get("boundary_rejected_points_px")
                or policy.get("boundary_rejected_points_px")
                or ball.get("source_boundary_rejected_points_px")
                or []
            )
            filter_stats = (
                variant.get("filter")
                or policy.get("filter")
                or ball.get("source_boundary_filter")
                or {}
            )

            refit = arc_combination_refit(
                points_px=points_px,
                rejected_points_px=rejected_points_px,
                filter_stats=filter_stats,
                cluster_shape_prior=shape_prior,
                neighbor_ellipses=joint.get("neighbor_ellipses_px") or [],
                include_fixed_shape_candidates=include_fixed_shape_candidates,
                max_search_groups=max_search_groups,
                max_fixed_shape_combo_size=max_fixed_shape_combo_size,
            )
            promote, reject_reasons = should_promote_arc_combination(refit)
            policy["arc_combination_refit"] = {
                **refit,
                "promotion_checked": True,
                "promoted": bool(promote),
                "promotion_reject_reasons": reject_reasons,
            }
            if not promote:
                ball["source_final_center_policy"] = policy
                continue

            best = refit.get("best") or {}
            ellipse = best.get("ellipse_fit") or {}
            center = ellipse.get("center_px")
            selected_points = best.get("selected_points_px") or []
            if not center or not selected_points:
                policy["arc_combination_refit"]["promoted"] = False
                policy["arc_combination_refit"]["promotion_reject_reasons"] = [
                    "missing promoted center or points"
                ]
                ball["source_final_center_policy"] = policy
                continue

            raw_points = (filter_stats.get("raw_points_px") or [])
            rejected_points = _points_not_selected(raw_points, selected_points)
            promoted_filter = {
                **filter_stats,
                "arc_combo_promotion": {
                    "status": "promoted",
                    "group_ids": best.get("group_ids") or [],
                    "ranking_score": best.get("ranking_score"),
                    "ellipse_rms_residual_px": best.get("ellipse_rms_residual_px"),
                    "cluster_shape_comparison": best.get("cluster_shape_comparison"),
                    "note": (
                        "Final boundary points are selected raw arc clusters; "
                        "remaining raw samples are shown as rejected red dots."
                    ),
                },
                "accepted_count": len(selected_points),
                "rejected_count": len(rejected_points),
            }
            selected_map = policy.get("selected_map") or "ball_vs_cloth_probability"
            selected_label = policy.get("selected_label") or _final_evidence_map_label(
                selected_map
            )
            promoted_policy = {
                **policy,
                "status": "computed",
                "used_for_final_position": True,
                "selected_map": selected_map,
                "selected_label": selected_label,
                "observed_source": f"arc_combo_{selected_map}",
                "center_px": [round(float(center[0]), 4), round(float(center[1]), 4)],
                "boundary_points_px": selected_points,
                "boundary_rejected_points_px": rejected_points,
                "ellipse_fit": {
                    **ellipse,
                    "source": f"arc_combo_{selected_map}",
                },
                "filter": promoted_filter,
                "point_count": len(selected_points),
                "rejected_point_count": len(rejected_points),
                "reason": "cluster-consistent raw arc combination promoted",
                "note": (
                    "Promoted because raw arc clusters produced an ellipse that "
                    "fits the same-colour cluster shape better than the filtered "
                    "baseline."
                ),
            }
            promoted_policy["arc_combination_refit"] = {
                **refit,
                "promotion_checked": True,
                "promoted": True,
                "promotion_reject_reasons": [],
            }
            ball["source_final_center_policy"] = promoted_policy
            ball["source_refined_center_px"] = promoted_policy["center_px"]
            ball["source_final_center_px"] = promoted_policy["center_px"]
            ball["source_boundary_points_px"] = _round_points(selected_points)
            ball["source_boundary_rejected_points_px"] = _round_points(rejected_points)
            ball["source_boundary_filter"] = promoted_filter
            ball["source_boundary_evidence_source"] = promoted_policy["observed_source"]
            ball["source_ellipse_fit"] = _round_ellipse_fit(promoted_policy["ellipse_fit"])

            by_z = self.camera_model.project_image_point_to_z_planes(
                (float(center[0]), float(center[1])),
                self.projection_z_planes_mm,
            )
            ball_radius_key = z_plane_key(self.table.ball_radius_mm)
            source_radius_projection = by_z.get(ball_radius_key)
            ball["source_refined_table_xy_by_z_mm"] = _round_projection_by_z(by_z)
            ball["source_refined_table_xy_mm"] = (
                _point_or_none_from_sequence(source_radius_projection["xy_mm"])
                if source_radius_projection is not None
                else None
            )

            sphere_projection = _state_ball_sphere_projection(
                camera_model=self.camera_model,
                table_radius_mm=self.table.ball_radius_mm,
                by_z=by_z,
                observed_points=selected_points,
                observed_source=promoted_policy["observed_source"],
            )
            if isinstance(ball.get("source_sphere_projection"), dict):
                existing = ball["source_sphere_projection"]
                if isinstance(existing.get("joint_cluster"), dict):
                    sphere_projection["joint_cluster"] = existing["joint_cluster"]
            ball["source_sphere_projection"] = sphere_projection
            debug = ball.get("debug")
            if isinstance(debug, dict):
                debug["arc_combo_promotion"] = promoted_policy["arc_combination_refit"]
                debug["source_sphere_projection"] = sphere_projection

    def _apply_cluster_joint_promotions(
        self,
        balls: list[dict[str, Any]],
    ) -> None:
        """Promote safe arbitrary-cluster graph centers into final coordinates.

        The lower-level optimizer remains diagnostic. This method is the only
        place where a scene-level contact graph can replace the final source
        center, and only after the explicit gate in cluster_promotion.py passes.
        """

        settings = dict(self.detector.config.get("cluster_optimization", {}))
        if not bool(settings.get("joint_center_promotion_enabled", False)):
            return
        for ball in balls:
            joint = ball.get("source_joint_cluster_optimization") or {}
            if not isinstance(joint, dict):
                continue

            promote, reject_reasons = should_promote_cluster_joint_center(
                ball=ball,
                joint=joint,
                settings=settings,
            )
            if not promote:
                _attach_cluster_promotion_check(
                    ball,
                    {
                        "status": "not_promoted",
                        "promoted": False,
                        "model": "cluster_graph_joint_center",
                        "promotion_reject_reasons": reject_reasons,
                    },
                )
                continue

            joint_xy = joint.get("joint_xy_mm")
            try:
                source_px_array = self.camera_model.world_point_to_image(
                    [
                        float(joint_xy[0]),
                        float(joint_xy[1]),
                        float(self.table.ball_radius_mm),
                    ]
                )
                source_px = [
                    round(float(source_px_array[0]), 4),
                    round(float(source_px_array[1]), 4),
                ]
            except (TypeError, ValueError, IndexError):
                _attach_cluster_promotion_check(
                    ball,
                    {
                        "status": "not_promoted",
                        "promoted": False,
                        "model": "cluster_graph_joint_center",
                        "promotion_reject_reasons": [
                            "could_not_project_joint_xy_to_source_px"
                        ],
                    },
                )
                continue

            payload = cluster_joint_promotion_payload(
                ball=ball,
                joint=joint,
                source_px=source_px,
                settings=settings,
            )
            if not payload.get("promoted"):
                _attach_cluster_promotion_check(ball, payload)
                continue

            previous_center = ball.get("source_final_center_px") or ball.get(
                "source_refined_center_px"
            )
            policy = ball.get("source_final_center_policy") or {}
            if not isinstance(policy, dict):
                policy = {}
            promoted_policy = {
                **policy,
                "status": "computed",
                "used_for_final_position": True,
                "final_center_source": "cluster_graph_joint_center",
                "observed_source": "cluster_graph_joint_center",
                "center_px": source_px,
                "reason": "cluster graph contact constraints promoted",
                "note": payload["note"],
                "cluster_joint_promotion": payload,
            }

            ball["source_final_center_policy"] = promoted_policy
            ball["source_refined_center_px"] = source_px
            ball["source_final_center_px"] = source_px
            ball["source_position_source"] = "cluster_graph_joint_center"
            ball["source_refined_warped_center_px"] = _point_or_none(
                *self._source_point_to_warped(ball, source_px=source_px)
            )

            by_z = self.camera_model.project_image_point_to_z_planes(
                (float(source_px[0]), float(source_px[1])),
                self.projection_z_planes_mm,
            )
            ball_radius_key = z_plane_key(self.table.ball_radius_mm)
            source_radius_projection = by_z.get(ball_radius_key)
            ball["source_refined_table_xy_by_z_mm"] = _round_projection_by_z(by_z)
            ball["source_refined_table_xy_mm"] = (
                _point_or_none_from_sequence(source_radius_projection["xy_mm"])
                if source_radius_projection is not None
                else None
            )

            observed_points = ball.get("source_boundary_points_px") or []
            sphere_projection = _state_ball_sphere_projection(
                camera_model=self.camera_model,
                table_radius_mm=self.table.ball_radius_mm,
                by_z=by_z,
                observed_points=observed_points,
                observed_source="cluster_graph_joint_center_with_image_boundary_audit",
            )
            sphere_projection["joint_cluster"] = joint
            sphere_projection["cluster_joint_promotion"] = payload
            ball["source_sphere_projection"] = sphere_projection

            source_optimization = ball.get("source_sphere_optimization")
            if isinstance(source_optimization, dict):
                source_optimization["joint_cluster_promotion"] = payload

            debug = ball.get("debug")
            if isinstance(debug, dict):
                debug["cluster_joint_promotion"] = {
                    **payload,
                    "previous_source_center_px": previous_center,
                }
                debug["source_refined_center_px"] = source_px
                debug["source_sphere_projection"] = sphere_projection

    def _sphere_projection_with_optimization(
        self,
        *,
        sphere_projection: dict[str, Any],
        physical_optimization: dict[str, Any],
        observed_points: list[Any],
    ) -> dict[str, Any]:
        if not physical_optimization.get("success"):
            return {
                **sphere_projection,
                "projection_mode": "forward",
                "optimization": physical_optimization,
                "explanation": [
                    "Blue curve = forward projection from current estimated 3D ball center.",
                    "It is not fitted to the blob unless physical optimization succeeds.",
                    "Approximate camera model limits trust.",
                ],
            }
        optimized = {
            **sphere_projection,
            "status": "optimized",
            "projection_mode": "optimized",
            "projected_center_px": physical_optimization.get(
                "optimized_source_center_px",
                sphere_projection.get("projected_center_px"),
            ),
            "contour_points_px": physical_optimization.get(
                "optimized_sphere_curve_px",
                sphere_projection.get("contour_points_px", []),
            ),
            "optimization": physical_optimization,
            "forward_projection": {
                "status": sphere_projection.get("status"),
                "projected_center_px": sphere_projection.get("projected_center_px"),
                "contour_points_px": sphere_projection.get("contour_points_px", []),
                "observed_fit_score": sphere_projection.get("observed_fit_score"),
            },
            "explanation": [
                "Blue curve = optimized physical sphere projection near the current 3D estimate.",
                "Forward projection from the original estimate is kept in forward_projection.",
                "Approximate camera model limits trust.",
            ],
        }
        score = score_observed_points_against_silhouette(observed_points, optimized)
        if score is not None:
            optimized["observed_fit_score"] = {
                **score,
                "source": "source_boundary_points_px",
            }
        return optimized

    def _neighbor_table_points(
        self,
        result: DetectionResult,
        current_index: int,
    ) -> list[dict[str, Any]]:
        neighbors: list[dict[str, Any]] = []
        for index, other in enumerate(result.balls, start=1):
            if index == current_index:
                continue
            x_mm, y_mm = self.table_warp.warped_px_to_table_mm(
                other.x_px,
                other.y_px,
            )
            neighbors.append(
                {
                    "id": int(index),
                    "label": other.label,
                    "xy_mm": [float(x_mm), float(y_mm)],
                    "source_px": _point_or_none(
                        other.source_x_px if other.source_x_px is not None else other.source_rough_x_px,
                        other.source_y_px if other.source_y_px is not None else other.source_rough_y_px,
                    ),
                }
            )
        return neighbors

    def _source_point_to_warped(
        self,
        ball: Any,
        source_px: list[float] | tuple[float, float] | None = None,
    ) -> tuple[float | None, float | None]:
        if source_px is None:
            if ball.source_x_px is None or ball.source_y_px is None:
                return None, None
            source_px = [float(ball.source_x_px), float(ball.source_y_px)]
        if len(source_px) < 2:
            return None, None
        warped = self.table_warp.source_to_warped(
            np.float32([[float(source_px[0]), float(source_px[1])]])
        )[0]
        return float(warped[0]), float(warped[1])


def _point_or_none(
    x_value: float | None,
    y_value: float | None,
) -> list[float] | None:
    if x_value is None or y_value is None:
        return None
    return [round(float(x_value), 4), round(float(y_value), 4)]


def _point_or_none_from_sequence(values: Any) -> list[float] | None:
    if values is None:
        return None
    return _point_or_none(float(values[0]), float(values[1]))


def _point3_or_none_from_sequence(values: Any) -> list[float] | None:
    if values is None:
        return None
    return [
        round(float(values[0]), 4),
        round(float(values[1]), 4),
        round(float(values[2]), 4),
    ]


def _round_projection_by_z(
    projections: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rounded: dict[str, dict[str, Any]] = {}
    for key, projection in projections.items():
        xy = projection["xy_mm"]
        xyz = projection["xyz_mm"]
        rounded[key] = {
            "z_mm": round(float(projection["z_mm"]), 4),
            "xy_mm": [round(float(xy[0]), 4), round(float(xy[1]), 4)],
            "xyz_mm": [
                round(float(xyz[0]), 4),
                round(float(xyz[1]), 4),
                round(float(xyz[2]), 4),
            ],
            "approximate": bool(projection.get("approximate", True)),
            "camera_model": projection.get("camera_model"),
        }
    return rounded


def _round_points(points: Any) -> list[list[float]]:
    if points is None:
        return []
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return []
    return [
        [round(float(point[0]), 4), round(float(point[1]), 4)]
        for point in arr.reshape(-1, 2)
    ]


def _round_ellipse_fit(ellipse: dict[str, Any] | None) -> dict[str, Any] | None:
    if not ellipse:
        return None
    rounded: dict[str, Any] = {}
    for key, value in ellipse.items():
        if isinstance(value, (int, float)):
            rounded[key] = round(float(value), 4)
        else:
            rounded[key] = value
    return rounded


def _points_not_selected(
    raw_points: list[Any] | tuple[Any, ...] | np.ndarray,
    selected_points: list[Any] | tuple[Any, ...] | np.ndarray,
    *,
    tolerance_px: float = 0.75,
) -> list[list[float]]:
    """Return raw boundary samples not used by the promoted final fit."""

    raw = _points_array_or_empty(raw_points)
    selected = _points_array_or_empty(selected_points)
    if len(raw) == 0:
        return []
    if len(selected) == 0:
        return _round_points(raw)

    rejected: list[list[float]] = []
    for point in raw:
        distances = np.linalg.norm(selected - point.reshape(1, 2), axis=1)
        if float(np.min(distances)) > tolerance_px:
            rejected.append([round(float(point[0]), 4), round(float(point[1]), 4)])
    return rejected


def _attach_cluster_promotion_check(
    ball: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    policy = ball.get("source_final_center_policy")
    if isinstance(policy, dict):
        policy["cluster_joint_promotion"] = payload
    debug = ball.get("debug")
    if isinstance(debug, dict):
        debug["cluster_joint_promotion"] = payload


def _points_array_or_empty(points: Any) -> np.ndarray:
    if points is None:
        return np.empty((0, 2), dtype=np.float64)
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return arr.reshape(-1, 2)


def _state_ball_sphere_projection(
    *,
    camera_model: Any,
    table_radius_mm: float,
    by_z: dict[str, dict[str, Any]],
    observed_points: list[Any] | tuple[Any, ...] | np.ndarray,
    observed_source: str,
) -> dict[str, Any]:
    ball_radius_key = z_plane_key(table_radius_mm)
    projection = by_z.get(ball_radius_key)
    if projection is None:
        return {
            "status": "unavailable",
            "reason": "no ball-center Z-plane projection is available",
        }
    sphere = project_sphere_silhouette(
        camera_model,
        projection["xyz_mm"],
        table_radius_mm,
    )
    score = score_observed_points_against_silhouette(observed_points, sphere)
    if score is not None:
        sphere = {
            **sphere,
            "observed_fit_score": {
                **score,
                "source": observed_source,
            },
        }
    return sphere


def _final_evidence_map_key(label: str, policy: dict[str, Any]) -> str:
    label_key = str(label or "").lower()
    overrides = policy.get("label_overrides") or {}
    if label_key in overrides:
        return str(overrides[label_key])
    return str(policy.get("default_map") or "ball_vs_cloth_probability")


def _independent_policy_summary(policy: dict[str, Any]) -> dict[str, Any]:
    ellipse = policy.get("ellipse_fit") or {}
    return {
        "selected_map": policy.get("selected_map"),
        "observed_source": policy.get("observed_source"),
        "center_px": _point_or_none_from_sequence(policy.get("center_px")),
        "ellipse_fit": _round_ellipse_fit(ellipse),
        "point_count": int(policy.get("point_count") or 0),
        "rejected_point_count": int(policy.get("rejected_point_count") or 0),
        "note": "Independent pre-cluster estimate retained for benchmark audit.",
    }


def _feature_attribute_for_map(key: str) -> str:
    mapping = {
        "gray_edge": "gray_edge",
        "lab_delta_e": "lab_delta_e",
        "chroma_difference": "chroma_difference",
        "ball_vs_cloth_probability": "ball_probability",
        "physical_projection_band": "physical_band_score",
        "combined_boundary_score": "combined_boundary_score",
    }
    return mapping.get(str(key), str(key))


def _map_uses_outward_drop(key: str) -> bool:
    return str(key) in {
        "lab_delta_e",
        "chroma_difference",
        "ball_vs_cloth_probability",
        "combined_boundary_score",
    }


def _final_evidence_map_label(key: str) -> str:
    return {
        "gray_edge": "Grayscale edge",
        "lab_delta_e": "Lab Delta-E",
        "chroma_difference": "Chroma difference",
        "ball_vs_cloth_probability": "Ball-vs-cloth probability",
        "physical_projection_band": "Physical projection band",
        "combined_boundary_score": "Combined boundary score",
    }.get(str(key), str(key).replace("_", " "))


def _final_evidence_quality_issue(
    *,
    ellipse: dict[str, Any],
    reference_center: list[float] | tuple[float, float] | None,
    radius_px: float | None,
    policy: dict[str, Any],
) -> str | None:
    try:
        radius = float(radius_px)
    except (TypeError, ValueError):
        radius = 0.0

    axis_ratio = ellipse.get("axis_ratio")
    if axis_ratio is not None:
        try:
            maximum_axis_ratio = float(policy.get("maximum_axis_ratio", 2.0))
            if float(axis_ratio) > maximum_axis_ratio:
                return (
                    "selected evidence map ellipse is too elongated "
                    f"(axis ratio {float(axis_ratio):.2f} > {maximum_axis_ratio:.2f})"
                )
        except (TypeError, ValueError):
            pass

    if radius <= 0:
        return None

    major_axis = ellipse.get("major_axis_px")
    if major_axis is not None:
        try:
            maximum_major_factor = float(policy.get("maximum_major_axis_radius_factor", 3.2))
            if float(major_axis) > radius * maximum_major_factor:
                return (
                    "selected evidence map ellipse is too large "
                    f"({float(major_axis):.1f}px > {maximum_major_factor:.1f}× radius)"
                )
        except (TypeError, ValueError):
            pass

    center = ellipse.get("center_px")
    if reference_center is not None and center is not None:
        try:
            shift = float(
                np.hypot(
                    float(center[0]) - float(reference_center[0]),
                    float(center[1]) - float(reference_center[1]),
                ),
            )
            maximum_shift = radius * float(policy.get("maximum_center_shift_radius_factor", 0.9))
            if shift > maximum_shift:
                return (
                    "selected evidence map center moved too far "
                    f"({shift:.1f}px > {maximum_shift:.1f}px)"
                )
        except (TypeError, ValueError, IndexError):
            pass

    return None


def _source_detection_center(ball: Any) -> list[float] | None:
    return _point_or_none(
        ball.source_x_px if ball.source_x_px is not None else ball.source_rough_x_px,
        ball.source_y_px if ball.source_y_px is not None else ball.source_rough_y_px,
    )


def _source_detection_radius(ball: Any) -> float | None:
    value = ball.source_radius_px if ball.source_radius_px is not None else ball.radius_px
    if value is None:
        return None
    try:
        radius = float(value)
    except (TypeError, ValueError):
        return None
    return radius if radius > 2.0 else None


def _source_detection_ellipse_payload(
    ball: Any,
    *,
    fallback_center: list[float],
    fallback_radius: float,
) -> dict[str, Any] | None:
    ellipse = ball.source_ellipse_fit or ball.source_silhouette_ellipse_fit
    if ellipse:
        center = ellipse.get("center_px")
        if center is None and "center_x_px" in ellipse and "center_y_px" in ellipse:
            center = [ellipse["center_x_px"], ellipse["center_y_px"]]
        major = ellipse.get("major_axis_px")
        minor = ellipse.get("minor_axis_px")
        if center is not None and major is not None and minor is not None:
            return {
                "center_px": [float(center[0]), float(center[1])],
                "major_axis_px": float(major),
                "minor_axis_px": float(minor),
                "angle_deg": float(ellipse.get("angle_deg") or 0.0),
                "axis_ratio": (
                    None
                    if ellipse.get("axis_ratio") is None
                    else float(ellipse.get("axis_ratio"))
                ),
                "source": ellipse.get("source") or "neighbor_source_ellipse",
            }
    return {
        "center_px": [float(fallback_center[0]), float(fallback_center[1])],
        "major_axis_px": float(fallback_radius) * 2.0,
        "minor_axis_px": float(fallback_radius) * 2.0,
        "angle_deg": 0.0,
        "axis_ratio": 1.0,
        "source": "neighbor_source_radius_fallback",
    }


def _camera_model_config(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, str):
        loaded = load_yaml(config)
        return dict(loaded.get("camera_model", loaded))
    if not isinstance(config, dict):
        raise ValueError("pipeline.camera_model must be a mapping or YAML path")
    if "config_file" not in config:
        return dict(config)
    loaded = load_yaml(config["config_file"])
    loaded_model = dict(loaded.get("camera_model", loaded))
    overrides = {
        key: value
        for key, value in config.items()
        if key != "config_file" and value is not None
    }
    return {**loaded_model, **overrides}



@dataclass(slots=True)
class V1EstimateResult:
    """v1 processing result adapted from the validated estimator output."""

    table_state: TableState
    legacy_frame: ProcessedFrame


class V1Estimator:
    """v1 facade over the canonical estimator implementation.

    New code should depend on this class or on `TableState`; the old vision
    import path is kept only as a compatibility shim.
    """

    def __init__(self, estimator: StateEstimator):
        self._estimator = estimator

    @classmethod
    def from_config(cls, config_path: str | Path = "configs/sony_dev.yaml") -> "V1Estimator":
        return cls(StateEstimator.from_config(config_path))

    def process_image(self, image_path: str | Path) -> V1EstimateResult:
        frame = self._estimator.process(image_path)
        report = {
            "image": str(image_path),
            "camera_model": frame.state.get("camera_model", {}),
            "summary": frame.state.get("detection", {}),
            "state": frame.state,
            "review_evidence": {},
        }
        return V1EstimateResult(
            table_state=table_state_from_legacy_report(
                report,
                report_stem=Path(image_path).stem,
            ),
            legacy_frame=frame,
        )


__all__ = ["ProcessedFrame", "StateEstimator", "V1EstimateResult", "V1Estimator"]
