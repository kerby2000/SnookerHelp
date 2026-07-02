from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from snookerhelp.recognition.color import BallColorClassifier, convert_color_spaces
from snookerhelp.recognition.circle_fit import refine_circle


@dataclass(frozen=True)
class BallDetection:
    raw_x_px: float
    raw_y_px: float
    raw_radius_px: float
    x_px: float
    y_px: float
    radius_px: float
    fit_residual_px: float | None
    fit_point_count: int
    refinement_success: bool
    label: str
    confidence: float
    detection_score: float
    color_confidence: float
    median_lab_difference: float
    foreground_fraction: float
    hsv: tuple[int, int, int]
    lab: tuple[int, int, int]
    source_rough_x_px: float | None = None
    source_rough_y_px: float | None = None
    source_x_px: float | None = None
    source_y_px: float | None = None
    source_radius_px: float | None = None
    source_fit_residual_px: float | None = None
    source_fit_point_count: int = 0
    source_refinement_success: bool = False
    source_roi_px: tuple[int, int, int, int] | None = None
    source_boundary_points_px: tuple[tuple[float, float], ...] = ()
    source_boundary_rejected_points_px: tuple[tuple[float, float], ...] = ()
    source_boundary_filter: dict[str, Any] | None = None
    source_boundary_evidence_source: str | None = None
    source_ellipse_fit: dict[str, Any] | None = None
    source_mask_centroid_px: tuple[float, float] | None = None
    source_mask_area_px: float | None = None
    source_mask_contour_points_px: tuple[tuple[float, float], ...] = ()
    source_silhouette_ellipse_fit: dict[str, Any] | None = None


@dataclass(frozen=True)
class DetectionResult:
    balls: list[BallDetection]
    difference: np.ndarray
    difference_visualization: np.ndarray
    raw_candidate_count: int
    gated_candidate_count: int
    duplicate_suppressed_candidate_count: int = 0


class ClassicalBallDetector:
    def __init__(
        self,
        config: dict[str, Any],
        expected_radius_px: float,
        surface_bounds_px: tuple[float, float, float, float] | None = None,
    ):
        self.config = config
        self.expected_radius_px = expected_radius_px
        self.surface_bounds_px = surface_bounds_px
        self.classifier = BallColorClassifier()

    def detect(
        self,
        warped_image: np.ndarray,
        warped_background: np.ndarray,
    ) -> DetectionResult:
        if warped_image.shape != warped_background.shape:
            raise ValueError("Image and background must have identical warped shapes")

        difference_config = self.config["difference"]
        kernel = int(difference_config["blur_kernel"])
        if kernel % 2 == 0:
            kernel += 1

        image_blur = cv2.GaussianBlur(warped_image, (kernel, kernel), 0)
        background_blur = cv2.GaussianBlur(warped_background, (kernel, kernel), 0)
        image_lab = cv2.cvtColor(image_blur, cv2.COLOR_BGR2LAB).astype(np.float32)
        background_lab = cv2.cvtColor(
            background_blur, cv2.COLOR_BGR2LAB
        ).astype(np.float32)
        difference = np.linalg.norm(image_lab - background_lab, axis=2)

        visualization_scale = float(difference_config["visualization_scale"])
        feature = np.uint8(np.clip(difference * visualization_scale, 0, 255))
        feature = cv2.GaussianBlur(feature, (5, 5), 1)

        hough = self.config["hough"]
        circles = cv2.HoughCircles(
            feature,
            cv2.HOUGH_GRADIENT,
            dp=float(hough["dp"]),
            minDist=max(
                1.0,
                self.expected_radius_px
                * float(hough["minimum_center_distance_radius_factor"]),
            ),
            param1=float(hough["edge_threshold"]),
            param2=float(hough["accumulator_threshold"]),
            minRadius=max(
                1,
                int(
                    round(
                        self.expected_radius_px
                        * float(hough["minimum_radius_factor"])
                    )
                ),
            ),
            maxRadius=max(
                2,
                int(
                    round(
                        self.expected_radius_px
                        * float(hough["maximum_radius_factor"])
                    )
                ),
            ),
        )
        raw_circles = (
            []
            if circles is None
            else np.round(circles[0]).astype(int).tolist()
        )

        hsv_image, lab_image = convert_color_spaces(warped_image)
        candidates: list[BallDetection] = []
        for rank, (x, y, radius) in enumerate(raw_circles):
            candidate = self._measure_candidate(
                x=x,
                y=y,
                radius=radius,
                rank=rank,
                difference=difference,
                hsv_image=hsv_image,
                lab_image=lab_image,
            )
            if candidate is not None:
                candidates.append(candidate)

        gated_candidate_count = len(candidates)
        candidates = self._suppress_duplicate_candidates(candidates)
        duplicate_suppressed_count = gated_candidate_count - len(candidates)
        selected = self._select_candidates(candidates)
        selected = [
            self._refine_candidate(ball, warped_image, difference)
            for ball in selected
        ]
        visualization = cv2.applyColorMap(feature, cv2.COLORMAP_TURBO)
        return DetectionResult(
            balls=sorted(selected, key=lambda ball: (ball.x_px, ball.y_px)),
            difference=difference,
            difference_visualization=visualization,
            raw_candidate_count=len(raw_circles),
            gated_candidate_count=gated_candidate_count,
            duplicate_suppressed_candidate_count=duplicate_suppressed_count,
        )

    def _measure_candidate(
        self,
        x: int,
        y: int,
        radius: int,
        rank: int,
        difference: np.ndarray,
        hsv_image: np.ndarray,
        lab_image: np.ndarray,
    ) -> BallDetection | None:
        height, width = difference.shape
        sampling = self.config["sampling"]
        sample_radius = max(
            3,
            int(
                round(
                    self.expected_radius_px
                    * float(sampling["inner_radius_factor"])
                )
            ),
        )
        x0, x1 = max(0, x - sample_radius), min(width, x + sample_radius + 1)
        y0, y1 = max(0, y - sample_radius), min(height, y + sample_radius + 1)
        if x0 >= x1 or y0 >= y1:
            return None

        yy, xx = np.ogrid[y0 - y : y1 - y, x0 - x : x1 - x]
        disk = xx * xx + yy * yy <= sample_radius * sample_radius
        difference_values = difference[y0:y1, x0:x1][disk]
        if difference_values.size == 0:
            return None

        median_difference = float(np.median(difference_values))
        difference_config = self.config["difference"]
        if median_difference < float(
            difference_config["minimum_median_lab_distance"]
        ):
            return None

        foreground_fraction = float(
            np.mean(
                difference_values
                > float(difference_config["foreground_lab_distance"])
            )
        )
        hsv_pixels = hsv_image[y0:y1, x0:x1][disk]
        lab_pixels = lab_image[y0:y1, x0:x1][disk]
        color = self.classifier.classify(
            hsv_pixels,
            lab_pixels,
            int(sampling["highlight_value_limit"]),
            int(sampling["minimum_non_highlight_pixels"]),
        )

        border_distance = min(x, y, width - 1 - x, height - 1 - y)
        border_support = max(
            0.0, min(float(border_distance), self.expected_radius_px)
        ) / self.expected_radius_px
        radius_fit = max(
            0.0,
            1.0 - abs(radius - self.expected_radius_px) / 10.0,
        )
        score = (
            median_difference / 100.0
            + foreground_fraction
            + radius_fit
            + border_support * 0.15
            - rank * 0.002
        )
        detection_confidence = float(
            np.clip((score - 1.25) / 1.75, 0.05, 0.99)
        )
        confidence = float(
            np.clip(
                0.72 * detection_confidence + 0.28 * color.confidence,
                0.05,
                0.99,
            )
        )
        return BallDetection(
            raw_x_px=float(x),
            raw_y_px=float(y),
            raw_radius_px=float(radius),
            x_px=float(x),
            y_px=float(y),
            radius_px=float(radius),
            fit_residual_px=None,
            fit_point_count=0,
            refinement_success=False,
            label=color.label,
            confidence=confidence,
            detection_score=float(score),
            color_confidence=color.confidence,
            median_lab_difference=median_difference,
            foreground_fraction=foreground_fraction,
            hsv=color.hsv,
            lab=color.lab,
        )

    def _refine_candidate(
        self,
        ball: BallDetection,
        warped_image: np.ndarray,
        difference: np.ndarray,
    ) -> BallDetection:
        refinement_config = self.config.get("circle_refinement", {})
        if not bool(refinement_config.get("enabled", True)):
            return ball
        fit = refine_circle(
            warped_image=warped_image,
            difference=difference,
            approximate_center=(ball.raw_x_px, ball.raw_y_px),
            approximate_radius=ball.raw_radius_px,
            config=refinement_config,
        )
        if not fit.success:
            return ball
        return BallDetection(
            raw_x_px=ball.raw_x_px,
            raw_y_px=ball.raw_y_px,
            raw_radius_px=ball.raw_radius_px,
            x_px=fit.x,
            y_px=fit.y,
            radius_px=fit.radius,
            fit_residual_px=fit.residual_px,
            fit_point_count=fit.point_count,
            refinement_success=True,
            label=ball.label,
            confidence=ball.confidence,
            detection_score=ball.detection_score,
            color_confidence=ball.color_confidence,
            median_lab_difference=ball.median_lab_difference,
            foreground_fraction=ball.foreground_fraction,
            hsv=ball.hsv,
            lab=ball.lab,
        )

    def _select_candidates(
        self, candidates: list[BallDetection]
    ) -> list[BallDetection]:
        selection = self.config["selection"]
        max_balls = int(selection["max_balls"])
        ranked = sorted(
            candidates,
            key=self._candidate_selection_score,
            reverse=True,
        )
        if not bool(selection.get("use_inventory_limits", False)):
            return ranked[:max_balls]

        limits = {
            str(label): int(limit)
            for label, limit in selection["inventory_limits"].items()
        }
        diversity_weight = float(selection.get("spatial_diversity_weight", 0.0))
        selected: list[BallDetection] = []
        for label, limit in limits.items():
            selected.extend(
                self._select_spatially_diverse(
                    [ball for ball in ranked if ball.label == label],
                    limit,
                    diversity_weight,
                )
            )
        return sorted(
            selected,
            key=self._candidate_selection_score,
            reverse=True,
        )[:max_balls]

    def _suppress_duplicate_candidates(
        self,
        candidates: list[BallDetection],
    ) -> list[BallDetection]:
        selection = self.config.get("selection", {})
        if not bool(selection.get("duplicate_suppression_enabled", False)):
            return candidates

        threshold = self.expected_radius_px * float(
            selection.get("duplicate_center_distance_radius_factor", 1.45)
        )
        same_label_only = bool(selection.get("duplicate_same_label_only", True))
        near_edge_only = bool(
            selection.get("duplicate_suppression_near_edge_only", True)
        )
        kept: list[BallDetection] = []
        for candidate in sorted(
            candidates,
            key=self._candidate_selection_score,
            reverse=True,
        ):
            is_duplicate = False
            for accepted in kept:
                if same_label_only and candidate.label != accepted.label:
                    continue
                distance = float(
                    np.hypot(
                        candidate.x_px - accepted.x_px,
                        candidate.y_px - accepted.y_px,
                    )
                )
                if distance >= threshold:
                    continue
                edge_duplicate = (
                    self._is_near_surface_edge(candidate)
                    or self._is_near_surface_edge(accepted)
                )
                isolated_duplicate = self._is_isolated_overlap_duplicate(
                    candidate,
                    accepted,
                    candidates,
                )
                if near_edge_only and not edge_duplicate and not isolated_duplicate:
                    continue
                is_duplicate = True
                break
            if not is_duplicate:
                kept.append(candidate)
        return kept

    def _candidate_selection_score(self, ball: BallDetection) -> float:
        """Score used only to select among already-gated candidates.

        The detector still uses Hough/difference evidence to create candidates.
        This score adds a small geometric prior for near-cushion balls: if a
        candidate is close to a table edge, prefer the center whose distance to
        that edge is near one ball radius. This resolves duplicate hypotheses on
        elongated cushion balls without requiring manual annotations.
        """
        selection = self.config.get("selection", {})
        score = float(ball.detection_score)
        edge_bonus = float(selection.get("edge_touch_bonus", 0.0))
        if edge_bonus <= 0.0 or self.surface_bounds_px is None:
            return score

        left, top, right, bottom = self.surface_bounds_px
        edge_distance = min(
            abs(ball.x_px - left),
            abs(right - ball.x_px),
            abs(ball.y_px - top),
            abs(bottom - ball.y_px),
        )
        max_distance = self.expected_radius_px * float(
            selection.get("edge_touch_max_distance_radius_factor", 3.0)
        )
        if edge_distance > max_distance:
            return score

        radius = max(float(self.expected_radius_px), 1.0)
        plausibility = max(0.0, 1.0 - abs(edge_distance - radius) / radius)
        return score + edge_bonus * plausibility

    def _is_near_surface_edge(self, ball: BallDetection) -> bool:
        if self.surface_bounds_px is None:
            return False
        selection = self.config.get("selection", {})
        left, top, right, bottom = self.surface_bounds_px
        edge_distance = min(
            abs(ball.x_px - left),
            abs(right - ball.x_px),
            abs(ball.y_px - top),
            abs(bottom - ball.y_px),
        )
        return bool(
            edge_distance
            <= self.expected_radius_px
            * float(selection.get("edge_touch_max_distance_radius_factor", 3.0))
        )

    def _is_isolated_overlap_duplicate(
        self,
        candidate: BallDetection,
        accepted: BallDetection,
        candidates: list[BallDetection],
    ) -> bool:
        selection = self.config.get("selection", {})
        if not bool(selection.get("isolated_overlap_suppression_enabled", True)):
            return False
        if candidate.label != accepted.label:
            return False

        radius = self.expected_radius_px * float(
            selection.get("isolated_overlap_neighbor_radius_factor", 3.0)
        )
        maximum_cluster_size = int(
            selection.get("isolated_overlap_max_same_label_cluster_size", 2)
        )

        def local_count(ball: BallDetection) -> int:
            return sum(
                1
                for other in candidates
                if other.label == ball.label
                and np.hypot(other.x_px - ball.x_px, other.y_px - ball.y_px)
                <= radius
            )

        return bool(
            max(local_count(candidate), local_count(accepted))
            <= maximum_cluster_size
        )

    def _select_spatially_diverse(
        self,
        candidates: list[BallDetection],
        limit: int,
        diversity_weight: float,
    ) -> list[BallDetection]:
        if len(candidates) <= limit or diversity_weight <= 0:
            return candidates[:limit]

        selected = [candidates[0]]
        remaining = candidates[1:]
        expected_diameter = self.expected_radius_px * 2.0
        while remaining and len(selected) < limit:
            best = max(
                remaining,
                key=lambda candidate: (
                    self._candidate_selection_score(candidate)
                    + diversity_weight
                    * min(
                        1.0,
                        min(
                            np.hypot(
                                candidate.x_px - accepted.x_px,
                                candidate.y_px - accepted.y_px,
                            )
                            for accepted in selected
                        )
                        / expected_diameter,
                    )
                ),
            )
            selected.append(best)
            remaining.remove(best)
        return selected


__all__ = ["BallDetection", "ClassicalBallDetector", "DetectionResult"]
