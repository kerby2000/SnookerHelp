from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from snookerhelp.recognition.color import BallColorClassifier
from snookerhelp.qa.report_metrics import (
    build_coordinate_rows,
    nearest_cushion_info,
    rough_to_refined_shift_px,
    z_projection_comparison,
)


WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GRAY = (150, 150, 150)
LIGHT_GRAY = (235, 235, 235)
GREEN = (40, 180, 70)
AMBER = (0, 165, 255)
RED = (40, 40, 230)
BLUE = (220, 120, 40)
YELLOW = (0, 220, 255)
MAGENTA = (210, 80, 210)
CYAN = (230, 210, 60)
CLOTH = (55, 115, 55)
CLOTH_DARK = (35, 85, 35)


def source_detection_panel(source_image: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    scale = min(2200 / source_image.shape[1], 1400 / source_image.shape[0], 1.0)
    canvas = cv2.resize(source_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _draw_source_table_boundary(canvas, state, scale_x=scale, scale_y=scale)
    for ball in state.get("balls", []):
        rough = ball.get("source_rough_center_px")
        refined = ball.get("source_refined_center_px")
        radius = ball.get("source_radius_px")
        color = BallColorClassifier.display_bgr(ball.get("color_label", "unknown"))
        if rough is not None:
            _draw_cross(canvas, _scale_point(rough, scale), YELLOW, size=16, thickness=2)
        if refined is not None:
            center = _scale_point(refined, scale)
            _draw_cross(canvas, center, GREEN, size=18, thickness=3)
            if radius is not None:
                cv2.circle(canvas, center, max(4, int(round(radius * scale))), color, 2, cv2.LINE_AA)
            residual = ball.get("source_fit_residual_px")
            ok = "ok" if ball.get("source_refinement_success") else "fallback"
            text = (
                f"{ball['id']} {ball.get('color_label')} "
                f"r={_fmt(radius)} res={_fmt(residual)} {ok}"
            )
            _draw_label(canvas, text, (center[0] + 12, center[1] - 12), scale=0.52)
    _draw_banner(
        canvas,
        "01 Source detection: cyan=table/cushion boundary, yellow=rough center, green=refined center, circle=source fit",
    )
    return canvas


def warped_detection_panel(warped_image: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    canvas = warped_image.copy()
    for ball in state.get("balls", []):
        center = _int_point(ball.get("warped_center_px", ball.get("refined_center_px")))
        radius = int(round(float(ball.get("radius_px", 26))))
        color = BallColorClassifier.display_bgr(ball.get("color_label", "unknown"))
        cv2.circle(canvas, center, radius, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, center, 4, WHITE, -1, cv2.LINE_AA)
        _draw_label(
            canvas,
            f"{ball['id']} {ball.get('color_label')}",
            (center[0] + radius + 5, center[1] - 5),
            scale=0.5,
        )
    _draw_banner(
        canvas,
        "02 Warped detection: cloth-plane visualization only; balls near edges are not expected to stay circular",
    )
    return _fit_image(canvas, 2200, 1250)


def source_zoom_grid_panel(
    source_image: np.ndarray,
    state: dict[str, Any],
    zoom_balls: list[dict[str, Any]],
) -> np.ndarray:
    tile_w, tile_h = 360, 500
    columns = 4
    rows = max(1, int(np.ceil(len(zoom_balls) / columns)))
    canvas = np.full((rows * tile_h + 70, columns * tile_w, 3), 245, dtype=np.uint8)
    _draw_text(
        canvas,
        "03 Source zoom grid: all detected balls; cyan table edge, yellow rough center, green refined center",
        (18, 42),
        0.72,
        BLACK,
        2,
    )
    for index, ball in enumerate(zoom_balls):
        row, col = divmod(index, columns)
        x0, y0 = col * tile_w, 70 + row * tile_h
        tile = source_zoom_tile_panel(source_image, state, ball, tile_w, tile_h)
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile
    return canvas


def source_zoom_tile_panel(
    source_image: np.ndarray,
    state: dict[str, Any],
    ball: dict[str, Any],
    width: int = 420,
    height: int = 560,
) -> np.ndarray:
    return _zoom_tile(source_image, state, ball, width, height, include_caption=True)


def geometry_panel(
    source_image: np.ndarray,
    state: dict[str, Any],
    selected_ball: dict[str, Any] | None,
) -> np.ndarray:
    canvas = np.full((1100, 1800, 3), 250, dtype=np.uint8)
    _draw_text(canvas, "04 Geometry for selected ball", (24, 44), 1.0, BLACK, 2)
    if selected_ball is None:
        _draw_note_box(
            canvas,
            "No balls were detected in this image.\nGeometry panel is intentionally empty.",
            (80, 150),
            (760, 150),
            AMBER,
        )
        return canvas
    _draw_text(
        canvas,
        f"Ball {selected_ball['id']} {selected_ball.get('color_label')} | camera model: "
        f"{state.get('camera_model', {}).get('mode')} | calibrated="
        f"{state.get('camera_model', {}).get('is_calibrated')}",
        (24, 82),
        0.58,
        BLACK,
        1,
    )

    roi = _selected_roi(source_image, state, selected_ball, 470, 470)
    canvas[115:585, 35:505] = roi
    _draw_text(canvas, "Source ROI", (50, 615), 0.62, BLACK, 1)

    _draw_side_view(canvas, (560, 115, 650, 470), state)
    _draw_geometry_top_view(canvas, (55, 680, 1130, 360), state, selected_ball)
    _draw_z_delta_table(canvas, (1230, 125), selected_ball)

    if not state.get("camera_model", {}).get("is_calibrated", False):
        _draw_note_box(
            canvas,
            "Approximate geometry: manual_homography mode cannot model true height parallax.\n"
            "Z-plane rows are still shown so the report shape is ready for calibrated_pinhole mode.",
            (1220, 840),
            (540, 165),
            AMBER,
        )
    return canvas


def error_comparison_panel(state: dict[str, Any]) -> np.ndarray:
    canvas = np.full((1180, 1900, 3), 248, dtype=np.uint8)
    _draw_text(canvas, "05 Coordinate comparison: warped-derived -> source-refined projection", (24, 44), 0.9, BLACK, 2)
    table_rect = (45, 90, 1220, 650)
    _draw_table(canvas, table_rect, state)
    max_delta = 0.0
    for ball in state.get("balls", []):
        old_xy = ball.get("table_xy_mm")
        source_xy = ball.get("source_refined_table_xy_mm")
        if old_xy is None or source_xy is None:
            continue
        p_old = _table_point(old_xy[0], old_xy[1], table_rect, state)
        p_new = _table_point(source_xy[0], source_xy[1], table_rect, state)
        delta = float(np.hypot(source_xy[0] - old_xy[0], source_xy[1] - old_xy[1]))
        max_delta = max(max_delta, delta)
        color = GREEN if delta <= 3 else AMBER if delta <= 10 else RED
        cv2.arrowedLine(canvas, p_old, p_new, color, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(canvas, p_old, 4, RED, -1, cv2.LINE_AA)
        cv2.circle(canvas, p_new, 5, GREEN, -1, cv2.LINE_AA)
        _draw_text(canvas, str(ball["id"]), (p_new[0] + 5, p_new[1] - 5), 0.42, BLACK, 1)
    _draw_text(canvas, f"Max warped->source delta: {max_delta:.2f} mm", (60, 780), 0.65, BLACK, 2)
    _draw_coordinate_table(canvas, (1310, 95), build_coordinate_rows(state), max_rows=22)
    return canvas


def physical_validation_panel(
    state: dict[str, Any],
    validation: dict[str, Any],
) -> np.ndarray:
    canvas = np.full((1100, 1800, 3), 248, dtype=np.uint8)
    _draw_text(canvas, "06 Physical validation", (24, 44), 0.95, BLACK, 2)
    _draw_text(
        canvas,
        f"Mode: {validation['mode']} | center mode: {validation['center_mode']} | rows: {validation['row_count']}",
        (24, 82),
        0.6,
        BLACK,
        1,
    )
    table_rect = (45, 125, 1120, 610)
    _draw_table(canvas, table_rect, state)
    for row in validation.get("rows", []):
        p = _table_point(row["x_mm"], row["y_mm"], table_rect, state)
        color = _status_color(row["status"])
        cv2.circle(canvas, p, 10, color, -1, cv2.LINE_AA)
        _draw_text(canvas, str(row["index"]), (p[0] + 10, p[1] - 8), 0.42, BLACK, 1)
    _draw_validation_table(canvas, (1210, 125), validation.get("rows", []))
    summary = validation.get("summary", {})
    _draw_note_box(
        canvas,
        "Green <=3 mm, amber <=8 mm, red >8 mm.\n"
        f"Mean abs error: {_fmt(summary.get('mean'))} mm\n"
        f"Median abs error: {_fmt(summary.get('median'))} mm\n"
        f"Max abs error: {_fmt(summary.get('max'))} mm",
        (55, 790),
        (520, 185),
        GREEN if (summary.get("max") or 999) <= 3 else AMBER if (summary.get("max") or 999) <= 8 else RED,
    )
    if validation["mode"].startswith("auto"):
        _draw_note_box(
            canvas,
            "No scenario YAML was provided.\nRows are auto candidate touching pairs, not ground-truth labels.",
            (620, 790),
            (610, 140),
            AMBER,
        )
    return canvas


def pipeline_summary_panel(
    state: dict[str, Any],
    validation: dict[str, Any],
) -> np.ndarray:
    canvas = np.full((900, 1800, 3), 250, dtype=np.uint8)
    _draw_text(canvas, "07 Pipeline summary", (24, 46), 1.0, BLACK, 2)
    boxes = [
        ("source image", "real photo"),
        ("rough detection", f"{state.get('detection', {}).get('ball_count')} balls"),
        ("source refinement", _refinement_summary(state)),
        ("camera model", state.get("camera_model", {}).get("mode", "unknown")),
        ("Z-plane projection", ", ".join(str(z) for z in state.get("camera_model", {}).get("projection_z_planes_mm", []))),
        ("table_state.json", "coordinates + debug"),
        ("validation", f"{validation['row_count']} rows"),
    ]
    x = 50
    y = 165
    box_w = 215
    for index, (heading, detail) in enumerate(boxes):
        px = x + index * 245
        _draw_pipeline_box(canvas, (px, y), (box_w, 120), heading, detail)
        if index < len(boxes) - 1:
            cv2.arrowedLine(canvas, (px + box_w + 5, y + 60), (px + 240, y + 60), BLACK, 2, cv2.LINE_AA)
    notes = [
        "Final position should be judged from source-image center + camera model + ray/plane projection.",
        "Warped view remains useful for rough detection and debug, but it is a cloth-plane visualization.",
        "Physical checks are the evidence: touching balls, cushion touches, spots, and repeatability.",
    ]
    yy = 390
    for note in notes:
        _draw_text(canvas, f"- {note}", (70, yy), 0.72, BLACK, 2)
        yy += 58
    _draw_text(canvas, "Detection summary", (70, 610), 0.74, BLACK, 2)
    _draw_text(canvas, f"Raw candidates: {state.get('detection', {}).get('raw_candidate_count')}", (90, 655), 0.58, BLACK, 1)
    _draw_text(canvas, f"Gated candidates: {state.get('detection', {}).get('gated_candidate_count')}", (90, 690), 0.58, BLACK, 1)
    _draw_text(canvas, f"Camera calibrated: {state.get('camera_model', {}).get('is_calibrated')}", (90, 725), 0.58, BLACK, 1)
    _draw_text(canvas, f"Validation mean abs error: {_fmt(validation.get('summary', {}).get('mean'))} mm", (90, 760), 0.58, BLACK, 1)
    return canvas


def write_panel(path: str | Path, image: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def _zoom_tile(
    source_image: np.ndarray,
    state: dict[str, Any],
    ball: dict[str, Any],
    width: int,
    height: int,
    include_caption: bool,
) -> np.ndarray:
    image_area_h = min(width, height if not include_caption else width)
    caption_h = height - image_area_h
    center = ball.get("source_refined_center_px") or ball.get("source_rough_center_px")
    if center is None:
        center = [source_image.shape[1] / 2, source_image.shape[0] / 2]
    radius = float(ball.get("source_radius_px") or 40.0)
    crop_half = int(max(80, radius * 3.2))
    x0 = max(0, int(center[0] - crop_half))
    y0 = max(0, int(center[1] - crop_half))
    x1 = min(source_image.shape[1], int(center[0] + crop_half))
    y1 = min(source_image.shape[0], int(center[1] + crop_half))
    crop = source_image[y0:y1, x0:x1].copy()
    if crop.size == 0:
        crop = np.full((image_area_h, width, 3), 230, dtype=np.uint8)
    scale_x = width / crop.shape[1]
    scale_y = image_area_h / crop.shape[0]
    tile_img = cv2.resize(crop, (width, image_area_h), interpolation=cv2.INTER_CUBIC)
    _draw_source_table_boundary(
        tile_img,
        state,
        offset_x=float(x0),
        offset_y=float(y0),
        scale_x=scale_x,
        scale_y=scale_y,
        thickness=2,
    )

    def local(point: list[float] | None) -> tuple[int, int] | None:
        if point is None:
            return None
        return (
            int(round((point[0] - x0) * scale_x)),
            int(round((point[1] - y0) * scale_y)),
        )

    rough = local(ball.get("source_rough_center_px"))
    refined = local(ball.get("source_refined_center_px"))
    if rough is not None:
        _draw_cross(tile_img, rough, YELLOW, size=14, thickness=2)
    if refined is not None:
        _draw_cross(tile_img, refined, GREEN, size=16, thickness=2)
        cv2.circle(tile_img, refined, max(5, int(round(radius * (scale_x + scale_y) * 0.5))), BallColorClassifier.display_bgr(ball.get("color_label", "unknown")), 2, cv2.LINE_AA)
    tile = np.full((height, width, 3), 245, dtype=np.uint8)
    tile[:image_area_h] = tile_img
    cv2.rectangle(tile, (0, 0), (width - 1, height - 1), (180, 180, 180), 1)
    if not include_caption or caption_h <= 0:
        return tile
    residual = ball.get("source_fit_residual_px")
    point_count = (ball.get("debug") or {}).get("source_circle_fit_point_count")
    shift = rough_to_refined_shift_px(ball)
    cushion = nearest_cushion_info(state, ball)
    text1 = (
        f"#{ball['id']} {ball.get('color_label')} "
        f"r={_fmt(ball.get('source_radius_px'))} res={_fmt(residual)} px pts={point_count}"
    )
    text2 = "fit accepted: source radial boundary" if ball.get("source_refinement_success") else "fallback: fit rejected/failed"
    text3 = f"rough->refined shift={_fmt(shift)} px"
    text4 = f"nearest cushion: {cushion['name']} {_fmt(cushion['distance_mm'])} mm"
    _draw_text(tile, text1, (8, image_area_h + 24), 0.46, BLACK, 1)
    _draw_text(tile, text2, (8, image_area_h + 50), 0.42, GREEN if ball.get("source_refinement_success") else RED, 1)
    _draw_text(tile, text3, (8, image_area_h + 76), 0.42, BLACK, 1)
    _draw_text(tile, text4, (8, image_area_h + 102), 0.42, CYAN if cushion["is_near"] else GRAY, 1)
    return tile


def _selected_roi(
    source_image: np.ndarray,
    state: dict[str, Any],
    ball: dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray:
    return _zoom_tile(source_image, state, ball, width, height, include_caption=False)[:height, :width]


def _draw_side_view(canvas: np.ndarray, rect: tuple[int, int, int, int], state: dict[str, Any]) -> None:
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), WHITE, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), GRAY, 1)
    _draw_text(canvas, "Side-view geometry", (x + 18, y + 34), 0.62, BLACK, 1)
    camera = (x + 95, y + 80)
    cv2.rectangle(canvas, (camera[0] - 28, camera[1] - 16), (camera[0] + 28, camera[1] + 16), BLUE, -1)
    cv2.circle(canvas, (camera[0] + 36, camera[1]), 13, BLUE, 2)
    z0 = y + h - 70
    z26 = z0 - 95
    z52 = z0 - 190
    cv2.line(canvas, (x + 40, z0), (x + w - 40, z0), CLOTH, 4)
    cv2.line(canvas, (x + 40, z26), (x + w - 40, z26), GREEN, 2)
    cv2.line(canvas, (x + 40, z52), (x + w - 40, z52), AMBER, 2)
    p0 = (x + w - 100, z0)
    p26 = (x + w - 185, z26)
    p52 = (x + w - 270, z52)
    cv2.line(canvas, camera, p0, YELLOW, 2, cv2.LINE_AA)
    cv2.line(canvas, camera, p52, YELLOW, 1, cv2.LINE_AA)
    cv2.circle(canvas, p0, 7, CLOTH, -1)
    cv2.circle(canvas, p26, 7, GREEN, -1)
    cv2.circle(canvas, p52, 7, AMBER, -1)
    _draw_text(canvas, "Z=0 cloth", (x + w - 170, z0 - 8), 0.48, CLOTH_DARK, 1)
    _draw_text(canvas, "Z=26.25 center", (x + w - 210, z26 - 8), 0.48, GREEN, 1)
    _draw_text(canvas, "Z=52.5 top", (x + w - 190, z52 - 8), 0.48, AMBER, 1)
    if not state.get("camera_model", {}).get("is_calibrated", False):
        _draw_text(canvas, "Approximate: no calibrated ray yet", (x + 18, y + h - 22), 0.48, RED, 1)


def _draw_geometry_top_view(canvas: np.ndarray, rect: tuple[int, int, int, int], state: dict[str, Any], ball: dict[str, Any]) -> None:
    _draw_text(canvas, "Top-view position under height assumptions", (rect[0], rect[1] - 18), 0.62, BLACK, 1)
    _draw_table(canvas, rect, state)
    points: list[tuple[str, list[float], tuple[int, int, int]]] = []
    if ball.get("table_xy_mm"):
        points.append(("warped/cloth shortcut", ball["table_xy_mm"], RED))
    for key in ("z_0_00", "z_26_25", "z_52_50"):
        projection = (ball.get("source_refined_table_xy_by_z_mm") or {}).get(key)
        if projection:
            points.append((f"source {projection['z_mm']} mm", projection["xy_mm"], GREEN if projection["z_mm"] == 26.25 else BLUE if projection["z_mm"] == 0 else AMBER))
    for idx, (name, xy, color) in enumerate(points):
        p = _table_point(xy[0], xy[1], rect, state)
        cv2.circle(canvas, p, 10, color, -1, cv2.LINE_AA)
        _draw_text(canvas, str(idx + 1), (p[0] + 9, p[1] - 8), 0.45, BLACK, 1)
        _draw_text(canvas, f"{idx + 1}: {name} ({xy[0]:.1f},{xy[1]:.1f})", (rect[0] + 18, rect[1] + 26 + idx * 28), 0.48, color, 1)


def _draw_z_delta_table(canvas: np.ndarray, origin: tuple[int, int], ball: dict[str, Any]) -> None:
    x, y = origin
    _draw_text(canvas, "Z-plane table coordinates", (x, y), 0.64, BLACK, 1)
    y += 36
    _draw_text(canvas, "Z mm       X mm       Y mm       delta from warped", (x, y), 0.48, BLACK, 1)
    y += 28
    for row in z_projection_comparison(ball):
        _draw_text(
            canvas,
            f"{row['z_mm']:>5.2f}   {row['x_mm']:>8.1f}   {row['y_mm']:>8.1f}   {_fmt(row['delta_from_warped_mm'])}",
            (x, y),
            0.5,
            BLACK,
            1,
        )
        y += 28


def _draw_coordinate_table(canvas: np.ndarray, origin: tuple[int, int], rows: list[dict[str, Any]], max_rows: int) -> None:
    x, y = origin
    _draw_text(canvas, "Per-ball coordinates", (x, y), 0.64, BLACK, 1)
    y += 34
    _draw_text(canvas, "id label  warped XY        source XY        d_mm", (x, y), 0.45, BLACK, 1)
    y += 26
    for row in rows[:max_rows]:
        text = (
            f"{row['id']:>2} {str(row['label'])[:5]:<5} "
            f"{_fmt(row['warped_x_mm'])},{_fmt(row['warped_y_mm'])}  "
            f"{_fmt(row['source_x_mm'])},{_fmt(row['source_y_mm'])}  "
            f"{_fmt(row['warped_to_source_delta_mm'])}"
        )
        _draw_text(canvas, text, (x, y), 0.42, BLACK, 1)
        y += 23


def _draw_validation_table(canvas: np.ndarray, origin: tuple[int, int], rows: list[dict[str, Any]]) -> None:
    x, y = origin
    _draw_text(canvas, "Validation rows", (x, y), 0.66, BLACK, 1)
    y += 34
    _draw_text(canvas, "kind/item       measured  expected  error   status", (x, y), 0.45, BLACK, 1)
    y += 28
    for row in rows[:24]:
        color = _status_color(row["status"])
        text = (
            f"{row['kind'][:12]:<12} {row['item']:<7} "
            f"{row['measured_mm']:>7.2f} {row['expected_mm']:>7.2f} "
            f"{row['signed_error_mm']:>+7.2f} {row['status']}"
        )
        _draw_text(canvas, text, (x, y), 0.43, color, 1)
        y += 25


def _draw_pipeline_box(canvas: np.ndarray, origin: tuple[int, int], size: tuple[int, int], heading: str, detail: str) -> None:
    x, y = origin
    w, h = size
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (232, 242, 255), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BLUE, 2)
    _draw_text(canvas, heading, (x + 12, y + 36), 0.55, BLACK, 1)
    for index, line in enumerate(_wrap(detail, 22)[:3]):
        _draw_text(canvas, line, (x + 12, y + 66 + index * 23), 0.42, BLACK, 1)


def _draw_table(canvas: np.ndarray, rect: tuple[int, int, int, int], state: dict[str, Any]) -> None:
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CLOTH_DARK, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CLOTH, 5)
    for i in range(1, 6):
        xx = x + int(w * i / 6)
        cv2.line(canvas, (xx, y), (xx, y + h), (80, 130, 80), 1)
    for i in range(1, 4):
        yy = y + int(h * i / 4)
        cv2.line(canvas, (x, yy), (x + w, yy), (80, 130, 80), 1)


def _draw_source_table_boundary(
    image: np.ndarray,
    state: dict[str, Any],
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    thickness: int = 3,
) -> None:
    corners = (state.get("table") or {}).get("corner_points_px")
    if not corners or len(corners) < 4:
        return
    points = np.array(
        [
            [
                int(round((float(point[0]) - offset_x) * scale_x)),
                int(round((float(point[1]) - offset_y) * scale_y)),
            ]
            for point in corners
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [points], isClosed=True, color=CYAN, thickness=thickness, lineType=cv2.LINE_AA)


def _table_point(x_mm: float, y_mm: float, rect: tuple[int, int, int, int], state: dict[str, Any]) -> tuple[int, int]:
    x, y, w, h = rect
    table = state["table"]
    px = x + int(round(float(x_mm) / float(table["length_mm"]) * w))
    py = y + h - int(round(float(y_mm) / float(table["width_mm"]) * h))
    return px, py


def _fit_image(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    scale = min(max_width / image.shape[1], max_height / image.shape[0], 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _scale_point(point: list[float], scale: float) -> tuple[int, int]:
    return int(round(point[0] * scale)), int(round(point[1] * scale))


def _int_point(point: list[float]) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def _draw_cross(image: np.ndarray, center: tuple[int, int], color: tuple[int, int, int], size: int, thickness: int) -> None:
    x, y = center
    cv2.line(image, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def _draw_label(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.5) -> None:
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(np.clip(origin[0], 0, max(0, image.shape[1] - w - 8)))
    y = int(np.clip(origin[1], h + baseline + 5, image.shape[0] - baseline - 5))
    cv2.rectangle(image, (x - 3, y - h - baseline - 3), (x + w + 3, y + baseline + 3), BLACK, -1)
    _draw_text(image, text, (x, y), scale, WHITE, 1)


def _draw_banner(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 46), BLACK, -1)
    _draw_text(image, text, (14, 31), 0.65, WHITE, 2)


def _draw_note_box(canvas: np.ndarray, text: str, origin: tuple[int, int], size: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    w, h = size
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 3)
    for index, line in enumerate(text.splitlines()):
        _draw_text(canvas, line, (x + 16, y + 34 + index * 30), 0.55, BLACK, 1)


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(image, str(text), origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def _status_color(status: str) -> tuple[int, int, int]:
    return {"pass": GREEN, "warn": AMBER, "fail": RED}.get(status, GRAY)


def _refinement_summary(state: dict[str, Any]) -> str:
    balls = state.get("balls", [])
    if not balls:
        return "0 balls"
    ok = sum(1 for ball in balls if ball.get("source_refinement_success"))
    return f"{ok}/{len(balls)} source fits"


def _wrap(text: str, width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines
