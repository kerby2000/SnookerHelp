from __future__ import annotations

import cv2
import numpy as np

from snookerhelp.recognition.color import BallColorClassifier
from snookerhelp.recognition.classical_rough_detector import BallDetection
from snookerhelp.calibration.homography_bootstrap import TableWarp


def draw_warped_overlay(
    image: np.ndarray,
    balls: list[BallDetection],
    table_warp: TableWarp,
) -> np.ndarray:
    overlay = image.copy()
    height, width = overlay.shape[:2]
    margin = table_warp.table.processing_margin_px
    right = width - margin - 1
    bottom = height - margin - 1
    grid_step = int(round(500 * table_warp.table.px_per_mm))
    cv2.rectangle(
        overlay,
        (margin, margin),
        (right, bottom),
        (255, 220, 0),
        2,
        cv2.LINE_AA,
    )
    for x in range(margin, right + 1, grid_step):
        cv2.line(overlay, (x, margin), (x, bottom), (80, 80, 80), 1)
    for y in range(margin, bottom + 1, grid_step):
        cv2.line(overlay, (margin, y), (right, y), (80, 80, 80), 1)

    warning = (
        "Cloth-plane rectification: ball shapes near edges are not expected "
        "to stay circular"
    )
    cv2.rectangle(overlay, (0, 0), (width, 42), (20, 20, 20), -1)
    _draw_label(overlay, warning, (12, 28), scale=0.62, thickness=2)

    for index, ball in enumerate(balls, start=1):
        center = (int(round(ball.x_px)), int(round(ball.y_px)))
        radius = int(round(table_warp.table.ball_radius_px))
        color = BallColorClassifier.display_bgr(ball.label)
        cv2.circle(overlay, center, radius, color, 1, cv2.LINE_AA)
        cv2.drawMarker(
            overlay,
            center,
            (255, 255, 255),
            cv2.MARKER_CROSS,
            11,
            2,
            cv2.LINE_AA,
        )
        x_mm, y_mm = table_warp.warped_px_to_table_mm(
            ball.x_px, ball.y_px
        )
        text = f"{index}:{ball.label} ({x_mm:.0f},{y_mm:.0f})"
        _draw_label(overlay, text, (center[0] + radius + 4, center[1] - 4))
    return overlay


def draw_source_overlay(
    image: np.ndarray,
    balls: list[BallDetection],
    table_warp: TableWarp,
) -> np.ndarray:
    overlay = image.copy()
    corners = np.round(table_warp.corner_points_px).astype(np.int32)
    cv2.polylines(overlay, [corners], True, (255, 220, 0), 8, cv2.LINE_AA)

    for index, ball in enumerate(balls, start=1):
        center, radius, success = _source_circle_for_overlay(ball, table_warp)
        color = BallColorClassifier.display_bgr(ball.label)
        thickness = 8 if success else 4
        cv2.circle(overlay, center, radius, color, thickness, cv2.LINE_AA)
        cv2.circle(overlay, center, 8, (255, 255, 255), -1, cv2.LINE_AA)
        suffix = "" if success else "*"
        _draw_label(
            overlay,
            f"{index}:{ball.label}{suffix}",
            (center[0] + radius + 10, center[1] - 10),
            scale=1.2,
            thickness=3,
        )
    return overlay


def _source_circle_for_overlay(
    ball: BallDetection,
    table_warp: TableWarp,
) -> tuple[tuple[int, int], int, bool]:
    if (
        ball.source_x_px is not None
        and ball.source_y_px is not None
        and ball.source_radius_px is not None
        and ball.source_radius_px > 0
    ):
        return (
            (int(round(ball.source_x_px)), int(round(ball.source_y_px))),
            max(4, int(round(ball.source_radius_px))),
            bool(ball.source_refinement_success),
        )

    warped_points = np.float32(
        [
            [ball.x_px, ball.y_px],
            [ball.x_px + ball.radius_px, ball.y_px],
        ]
    )
    source_points = table_warp.warped_to_source(warped_points)
    center = tuple(np.round(source_points[0]).astype(int))
    radius = max(
        4,
        int(round(np.linalg.norm(source_points[1] - source_points[0]))),
    )
    return center, radius, False


def _draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    x = int(np.clip(origin[0], 0, max(0, image.shape[1] - width - 5)))
    y = int(
        np.clip(
            origin[1],
            height + baseline + 3,
            image.shape[0] - baseline - 3,
        )
    )
    cv2.rectangle(
        image,
        (x - 3, y - height - baseline - 3),
        (x + width + 3, y + baseline + 3),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


__all__ = ["draw_source_overlay", "draw_warped_overlay"]
